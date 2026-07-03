# Cycle-accurate emulation — measuring the throughput mechanism on real RTL cycles

Moves the perf story off the pure `[EST]` roofline in one specific, defensible way: the
**memory-stall mechanism the roofline assumes is now measured on real RTL cycles** by running the
*assembled* system top `glm_fp8_system` (compute die + `expert_cache_pf` + `kv_cache_pager` +
Flash arbiter + `ddr5_xbar` + `weight_loader`) under a cycle-accurate testbench with the Flash
latency model engaged, sweeping the knobs, and counting cycles.

This is the software-emulation core of "FPGA emulation" — **measured cycles**, not a board run.
It does **not** produce an absolute real-753B tok/s (that needs full-scale weights / a real FPGA);
it validates the *mechanism and scaling* the roofline rests on.

## Harness

- `test/glm_fp8_system_perf_tb.v` — the functional system TB (`token == standalone glm_model_fp8`,
  X-aware) extended with three overridable params (`FLASH_LAT_CFG`, `DDR_NCH_CFG`,
  `CACHE_SLOTS_CFG`, wired into both the DUT and the TB Flash-PHY responder), a free-running cycle
  counter (`cyc_per_tok` = start → `tok_valid`), and a machine-readable `PERF …` line reading the
  DUT's exposed counters `ec_demand_stall_cycles` / `ec_hit_count` / `ec_miss_count`.
- `tools/perf_sweep.sh` — compiles once per config via `iverilog -P glm_fp8_system_perf_tb.<P>=<v>`,
  runs, and tabulates.

**The functional binding held at every valid config** (`ALL 3 TESTS PASSED`, token == standalone
reference) — so every cycle number below is from a *correct-token* run. The FLASH sweep was
**independently re-run** (two separate runs produced byte-identical numbers).

## Measured sweep

`cyc_per_tok` = cold-token compute latency (start→tok_valid). `stall` = `ec_demand_stall_cycles`
(cycles the cache/Flash subsystem was in a demand-miss refill) over 3 tokens. **`EFF_CYC` =
`cyc_per_tok + stall`** — the *integrated* latency (see §Integration). `MEM%` = `stall / EFF_CYC`.

| sweep | FLASH_LAT | DDR_NCH | CACHE_SLOTS | cyc/tok | stall | **EFF_CYC** | **MEM%** | hit | miss |
|---|---|---|---|---|---|---|---|---|---|
| FLASH | 8 | 4 | 4 | 7947 | 33 | **7980** | 0.4% | 93 | 3 |
| FLASH | 64 | 4 | 4 | 7947 | 201 | **8148** | 2.4% | 93 | 3 |
| FLASH | 256 | 4 | 4 | 7947 | 777 | **8724** | 8.9% | 93 | 3 |
| FLASH | 1024 | 4 | 4 | 7947 | 3081 | **11028** | 27.9% | 93 | 3 |
| DDRNCH | 256 | 1 | 4 | — | — | — | — | — | — (degenerate: NCH must be ≥2; TB fabric-liveness fails) |
| DDRNCH | 256 | 2 | 4 | 7947 | 777 | 8724 | 8.9% | 93 | 3 |
| SLOTS | 256 | 4 | 2 | 7947 | 1554 | **9501** | 16.3% | 90 | 6 |
| SLOTS | 256 | 4 | 4 | 7947 | 777 | 8724 | 8.9% | 93 | 3 |

## Three measured findings

1. **The memory-stall mechanism is exactly the roofline's, measured on real cycles.** `stall`
   rises strictly linearly: `stall = 3·FLASH_LAT + 9`, where the slope **3 = the number of
   demand-miss Flash fetches** across the 3 tokens. This *is* the roofline term *exposed stall =
   (#misses) × (Flash latency)* — no longer assumed, but counted on the RTL.
2. **Stall ∝ miss count (cache-capacity is a real, measured signal).** Halving `CACHE_SLOTS`
   (4→2) raised misses 3→6 and **doubled** the stall (777→1554) → `EFF_CYC` 8724→9501, MEM%
   8.9→16.4. The cache size moves the exposed stall exactly as the miss count moves.
3. **Effective latency is memory-bound as latency grows.** With the exposed stall integrated,
   the memory fraction climbs 0.4% → 2.4% → 8.9% → **27.9%** as FLASH_LAT goes 8 → 1024 — at the
   *slice's* tiny miss count. At real scale (§Extrapolation) it dominates.

## Integration — why EFF_CYC, and the honest topology note

Raw `cyc_per_tok` is **FLASH_LAT-invariant (flat 7947)** — by construction, not because the knob
is dead. In the TB, the compute die `u_model` receives expert weights *combinationally* from the
weight-source stub, and the host FSM waits only on `mdl_done`; the `expert_cache_pf` that
accumulates `ec_demand_stall_cycles` sits on a **parallel observer path** that models the
Flash-refill stall but does not gate `mdl_done`. So `cyc_per_tok` measures pure compute latency and
`stall` measures the refill cost *concurrently*.

**Real hardware cannot compute a missed expert's GEMM until its weights arrive from Flash.** The
faithful cost is therefore **compute + the *exposed* (un-prefetch-hidden) demand-stall** — exactly
`ec_demand_stall_cycles`, which counts only demand misses the prefetcher did *not* hide. Hence
**`EFF_CYC = cyc_per_tok + stall`** is the honest integrated latency. This is a modeling statement
made explicit (the die stalls on missed weights), not an RTL change — we did **not** alter the DUT
to force the number, which would fabricate it.

## Extrapolation to real scale (measured mechanism → projection)

The slice has **miss = 3 per 3 tokens** (tiny weights → almost everything fits the cache). At the
real config the per-token miss count is large: ~75 MoE layers × ~8 routed experts × `(1 − hit)`.
With the GLM-trace hit rate `h ≈ 27%` (measured, `make cache-study` / `IMPROVEMENT_PLAN.md`), that
is ~**hundreds of demand misses per token**. Using the **measured** per-miss cost (each miss
exposes ≈ `FLASH_LAT` cycles, from finding 1) and the **measured** hit rate:

```
real exposed stall / token  ≈  (layers · experts · (1−h)) · FLASH_LAT_exposed   ≫  compute
```

so `EFF_CYC` becomes **memory-dominated** — the slice's 0.4–28% memory fraction is a *floor*, and
real scale pushes it toward the roofline's ~75–80% Flash-bound regime. This is a
**measured-mechanism projection** (per-miss cost and hit rate are measured; only the miss *count*
is scaled by the known config), which is stronger than the pure first-order roofline.

## What this does / does not establish

- **Establishes (measured):** the memory-stall term is linear in Flash latency and proportional to
  miss count on real RTL cycles; the assembled system produces a *correct* token while these hold;
  the integrated `EFF_CYC` decomposition (compute vs exposed Flash stall).
- **Does NOT establish:** an absolute real-753B tok/s (needs full-scale weights / a real FPGA run),
  achievable-vs-peak Flash bandwidth on silicon, or overlap efficiency at production widths. Those
  remain the FPGA-prototype step (`PRODUCT_ROADMAP.md`).

All of this is at the committed **slice** dims. The value is the *mechanism and scaling*, not the
absolute magnitude — see the extrapolation for how it maps to the real regime.

## Reproduce

```sh
bash tools/perf_sweep.sh      # FLASH_LAT / DDR_NCH / CACHE_SLOTS sweep -> the table above
# single point:
iverilog -g2012 -I src -P glm_fp8_system_perf_tb.FLASH_LAT_CFG=1024 \
    -o build/perf test/glm_fp8_system_perf_tb.v <glm_fp8_system_sim source list from the Makefile>
vvp build/perf     # -> ALL 3 TESTS PASSED + PERF flash_lat=1024 ... stall=3081 ...
```
