# `fpga/` — FPGA fit / vendor-flow scaffold for the GLM-5.2-FP8 accelerator

**This is Phase D0.2 "FPGA fit"** from
[`../docs/USBC_PRODUCT_PLAN.md`](../docs/USBC_PRODUCT_PLAN.md) — *the single
biggest unknown in the whole product plan.* Getting the real FPGA utilization is
what sets the **FPGA class**, and the FPGA class sets the device's **size,
thermal budget, BOM, and price** (§8 of that doc: *"cost is FPGA-dominated"*).
Everything downstream is bounded by the number this flow returns.

> **Status update: the yosys wall is broken; the vendor P&R fit is still open.**
> Using `yosys 0.66`'s **`synth_gowin`** (not the `abc -lut4` path that timed out),
> the FP8 compute datapath **maps** — `glm_matmul_fp8` at `KMAX=256` mapped in ~77 s
> to **~17.8 K LUT4-equiv + 20 DSP mults (`MULT18X18`/`MULT9X9`) + ~5.4 K DFF**, by
> **inferring hardware DSPs** for the multiplies. So "the compute die is un-mappable"
> was an `abc -lut4` limitation, not a design one.
> **The whole-system elaboration hang is now FIXED (RTL).** It was a real area bug:
> `glm_matmul_fp8`'s dequant was an **O(NB²)** unrolled fold (NB=ceil(KMAX/BLK); at the
> product KMAX=16384 → NB=128 → 256 FP pipes + ~40k delay FFs *per PE* — unbuildable).
> Rewritten to an **O(1)** sequential fold, **bit-exact** (matmul 224/224, bitacc
> 14/14+argmax, model argmax 4/31/20); yosys now elaborates `glm_matmul_fp8` at product
> KMAX. **What still needs the vendor flow:** the *routed* LUT/DSP/BSRAM/**Fmax** (yosys
> 0.66 can't route, flattens the O(NB) `accx` block-accumulator to registers instead of
> BRAM, and its SAT-based `SHARE` pass doesn't scale at full NB). Gowin EDA is not
> installed here, so those cells stay a **template to fill in**.
> Full plan + measured probes: [`../docs/FPGA_DEMO_PLAN.md`](../docs/FPGA_DEMO_PLAN.md).

---

## Why this exists (what yosys 0.66 can and can't give)

`make synth-glm` only *elaborates* the product top + `check -assert`; it emits no LUT
count. Two former yosys walls are now **retired**:
1. **`abc -lut4` timeout** — broken: `yosys synth_gowin` maps the FP8 datapath by
   inferring hardware DSPs (glm_matmul_fp8 leaf @ KMAX=256: ~17.8K LUT-eq + 20 DSP).
2. **whole-system elaboration hang** — was a real area bug (glm_matmul_fp8's O(NB²)
   dequant); **fixed** to O(1), bit-exact. The design now elaborates at product KMAX.

What yosys 0.66 **still can't** give, and why this vendor flow exists:
- **Routed** LUT/DSP/BSRAM utilization + **per-clock Fmax** — yosys `synth_gowin` is a
  mapper, not a P&R; only Gowin EDA / `nextpnr` route + time.
- **BRAM inference** for the O(NB) `accx` block-accumulator memory (one `[128,128]`
  block sum each — 128 words at product KMAX). yosys 0.66 *flattens it to registers*
  (huge + slow); **GowinSynthesis infers BSRAM** — confirm a non-trivial BSRAM count in
  the report (if accx lands in LUTs/FFs, the fit will be wrong-huge — a red flag to fix).
- **Scale**: yosys 0.66's SAT-based `SHARE` pass doesn't finish on the full-NB design; a
  production mapper doesn't hit that.

Reference anchor: [`../docs/PHYSICAL_SKY130.md`](../docs/PHYSICAL_SKY130.md) mapped only
the **memory-system controllers** on ECP5 (**~71,475 LUT4 ≈ 85% of an ECP5-85**) — so the
system needs a mid-size FPGA; this flow measures the *whole* GW5AT-138 fit.
*(If GowinSynthesis itself struggles on the FP8 math, that outcome is itself a finding —
record it.)*

## Target device

**Gowin GW5AT-138** — the **Sipeed Tang Mega 138K Pro** board (~138K LUT, on-board
DDR3). This is the mid-FPGA candidate referenced throughout
[`../docs/MINIATURIZATION.md`](../docs/MINIATURIZATION.md) (Phase E1 "gated on
Gowin EDA for the real LUT/DSP/BSRAM numbers on GW5AT-138"). The controllers'
~71,475 LUT4 anchor above is *why* size matters: the default config may or may not
fit 138K — which is exactly what D0.2 measures. If it overflows, the **compact**
config (below) and/or a larger FPGA are the levers.

---

## Directory contents

| File | What it is |
|------|-----------|
| [`gowin/build_gowin.tcl`](gowin/build_gowin.tcl) | **Primary flow.** Gowin `gw_sh` Tcl script: sets device, adds the 24 `src/` files + include dir, sets top, adds the SDC, runs synthesis (and optionally P&R), points at the resource/timing reports. `COMPACT=` and `FLOW=` env flags. |
| [`gowin/aipu.sdc`](gowin/aipu.sdc) | Timing constraints: `create_clock` for `host_clk` + `core_clk`, and `set_clock_groups -asynchronous` (the two domains cross only via `cdc_async_fifo`). Pin-assignment notes (pins go in a `.cst`, not the SDC). |
| [`gowin/glm_fp8_system_cdc_compact.v`](gowin/glm_fp8_system_cdc_compact.v) | Thin passthrough wrapper that fixes the 5 compact parameters (Gowin `gw_sh` can't override top params from Tcl the way yosys `-chparam` can). Used when `COMPACT=1`. Touches nothing in `src/`. |
| [`nextpnr/README.md`](nextpnr/README.md) | **Fallback flow.** Fully open-source path: yosys `synth_gowin` + `nextpnr-himbaechel` for GW5A, with the honest caveat that it can hit the same FP8/`abc` wall and that GW5A support is newer/approximate. |
| `gemm_harness.v`, `sm_harness.v` | Pre-existing synthesizable P&R harnesses for two *leaf* units (`gemm_systolic`, `softmax_unit`). They bury wide tile-memory ports so those units can be placed & routed standalone — the same pattern you'd need to fully P&R the whole top (see below). Not part of the GLM-5.2-FP8 top flow. |

---

## How to run

You need **Gowin EDA** installed and licensed, with `gw_sh` on your `PATH`. Run
from the **repo root** (paths in the script resolve relative to itself, so any CWD
works, but the root is cleanest):

```sh
# Default (committed) config, synthesis-only -> the RESOURCE FIT (LUT/DSP/BSRAM/FF)
gw_sh fpga/gowin/build_gowin.tcl

# Compact miniaturization config (byte-identical token, smaller area)
COMPACT=1 gw_sh fpga/gowin/build_gowin.tcl

# Attempt the FULL flow (synthesis + Place & Route + routed Fmax)
FLOW=all gw_sh fpga/gowin/build_gowin.tcl

# Both together
COMPACT=1 FLOW=all gw_sh fpga/gowin/build_gowin.tcl
```

**Before the first run, confirm the device part string** in
`gowin/build_gowin.tcl` (`set PART …`). It is a clearly-marked **placeholder**;
the comment there explains how to find the exact GW5AT-138 part number for the
Tang Mega 138K Pro (Gowin IDE device picker, `get_device_info`, or an existing
`.gprj`).

The open-source fallback is in [`nextpnr/README.md`](nextpnr/README.md).

### Full P&R vs synthesis-only (important)

- **`FLOW=syn` (default) — synthesis-only — is the D0.2 answer.** It reports
  LUT/DSP/BSRAM/registers with **no pin placement required**, so it always runs.
- **`FLOW=all` (full Place & Route)** additionally gives routed **Fmax**, but the
  raw top `glm_fp8_system_cdc` exposes **thousands of memory-side port bits**
  (e.g. `mem_resp_data` = `DDR_NCH*DDR_DATA_W` = up to 1024 bits, `logits` =
  `VOCAB*16` = 4096 bits, `h_state` = 2048 bits, …). No package has that many
  user I/O, so P&R will **fail on I/O** unless those wide ports are first **buried
  inside a bring-up harness** — exactly what `gemm_harness.v` / `sm_harness.v` do
  for their leaf units. Writing that top-level harness (drive the host interface;
  model DDR5/Flash/loader responses internally as BRAM/regs; expose only a
  handful of pins) is a follow-on task; until then, use `FLOW=syn` for the fit.

---

## What to measure and record

Per config (default and compact), from the reports under `./impl/`:

| Metric | Where (Gowin) | Notes |
|--------|---------------|-------|
| **LUT** (LUT4 / ALU) | `impl/gwsynthesis/*_syn.rpt*` | primary area number |
| **Registers (FF)** | same | flops used |
| **BSRAM** (block RAM) | same | KV ring, FIFOs, caches, weight staging |
| **DSP / MULT** | same | FP8 matmul multipliers, if inferred |
| **Utilization %** | same | vs GW5AT-138 totals — does it fit? |
| **Fmax per clock** | `impl/pnr/*timing*` (only `FLOW=all`) | `host_clk` and `core_clk` separately |

Cross-check against the yosys structural cell count from `make synth-glm` /
`make synth-glm-compact` (flattened cell histogram) for a sanity comparison.

---

## Results table — **TEMPLATE, fill in after running**

> All cells are `TBD` — **nothing here is measured.** Fill each in from your
> Gowin run and note the tool version + exact part string used.

**Environment**

| Field | Value |
|-------|-------|
| Gowin EDA version | `TBD` |
| Device part string | `TBD` (confirm from IDE / `get_device_info`) |
| Board | Sipeed Tang Mega 138K Pro (GW5AT-138) |
| Flow | `TBD` (syn / all) |
| Date | `TBD` |

**Fit — DEFAULT config** (PE_N=4, DDR_NCH=4, KV_RESIDENT=16, EFIFO_DEPTH=16, CACHE_SLOTS=4)

| Resource | Used | Available (GW5AT-138) | Util % |
|----------|------|-----------------------|--------|
| LUT | `TBD` | ~138K | `TBD` |
| Registers (FF) | `TBD` | `TBD` | `TBD` |
| BSRAM | `TBD` | `TBD` | `TBD` |
| DSP / MULT | `TBD` | `TBD` | `TBD` |
| Fmax `host_clk` | `TBD` MHz | target 100 MHz | — |
| Fmax `core_clk` | `TBD` MHz | target ~67 MHz | — |
| **Fits GW5AT-138?** | `TBD (yes/no)` | | |

**Fit — COMPACT config** (PE_N=2, DDR_NCH=2, KV_RESIDENT=8, EFIFO_DEPTH=8, CACHE_SLOTS=2 — byte-identical token)

| Resource | Used | Available (GW5AT-138) | Util % |
|----------|------|-----------------------|--------|
| LUT | `TBD` | ~138K | `TBD` |
| Registers (FF) | `TBD` | `TBD` | `TBD` |
| BSRAM | `TBD` | `TBD` | `TBD` |
| DSP / MULT | `TBD` | `TBD` | `TBD` |
| Fmax `host_clk` | `TBD` MHz | target 100 MHz | — |
| Fmax `core_clk` | `TBD` MHz | target ~67 MHz | — |
| **Fits GW5AT-138?** | `TBD (yes/no)` | | |

**Decision (D0.2 gate):** from these numbers, pick the FPGA class and record
whether the default fits, whether compact is needed, or whether a larger FPGA is
required — then propagate to size / thermal / BOM / price in
[`../docs/USBC_PRODUCT_PLAN.md`](../docs/USBC_PRODUCT_PLAN.md) §6–§8.

---

## Caveats (honest)

- **Un-run scaffold.** Nothing here has been executed; Gowin isn't installed in
  the authoring environment. The numbers come from *you* running Gowin.
- **⚠ Check BSRAM inference first (the #1 red flag).** The `glm_matmul_fp8` `accx`
  block-accumulator is an O(NB) memory (128 words × 48b per matmul at product KMAX).
  It **must** map to **BSRAM** — if the synthesis report shows it in LUTs/registers
  instead, the fit will read wrong-huge. GowinSynthesis normally infers BSRAM for it;
  if not, add a RAM style attribute / `set_option` and re-run. (This is the memory
  yosys 0.66 flattens to registers — the vendor tool must not.)
- **The FP8 math now maps** (yosys `synth_gowin` via DSP inference; the O(NB²) dequant
  bug is fixed). If GowinSynthesis still struggles on it, that's a finding to record,
  not a script bug.
- **Pre-flight:** the O(NB²) dequant fix is committed (`glm_matmul_fp8`), so the source
  list here elaborates at product KMAX. Confirm you're on a commit that includes it
  (matmul TB ALL 224 PASS at NB=2 and NB=16) before running the vendor flow.
- **Part string is a placeholder.** Confirm the exact GW5AT-138 package/speed/
  grade for your board before running (comment in `build_gowin.tcl`).
- **Board pinout is user-provided.** Pin locations live in a board-specific
  `.cst`; only commented placeholders are given (`aipu.sdc`). Not needed for the
  synthesis-only resource fit.
- **Full P&R needs a bring-up harness** to bury the wide memory-side ports; the
  synthesis-only fit does not. See "Full P&R vs synthesis-only" above.
- **`gw_sh` command spellings vary by Gowin version.** The script targets the
  common `set_device`/`add_file`/`set_option`/`run` API and flags every spot you
  may need to adjust.

## References

- [`../docs/USBC_PRODUCT_PLAN.md`](../docs/USBC_PRODUCT_PLAN.md) — Phase **D0.2**
  FPGA fit (the gating unknown), §6 FPGA class, §8 BOM/pricing.
- [`../docs/MINIATURIZATION.md`](../docs/MINIATURIZATION.md) — the **compact**
  config (L0), and Phase E1 "gated on Gowin EDA … on GW5AT-138".
- [`../docs/PHYSICAL_SKY130.md`](../docs/PHYSICAL_SKY130.md) — the yosys ECP5
  partial fit and the ~71,475 LUT4 controllers anchor / `abc` wall.
- Repo `Makefile` targets: `synth-glm`, `synth-glm-compact`, `sim-glm-compact`
  (the structural + byte-identical-token gates this fit builds on).
