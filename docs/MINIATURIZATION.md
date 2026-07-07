# Chip miniaturization — plan

> **Scope note (FPGA-fit now; rung-③ ASIC groundwork later).** See the hardware ladder
> ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)) for the staged framing. The near-term product path is
> an **FPGA card** ([`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) P3.2), so "shrink the die" here means
> **fit a smaller / cheaper FPGA** (fewer LUT/DSP/BSRAM) on the **prove-it/custom-board rungs ①–②** —
> *not* a tapeout today. But this die-minimization is **not a dead-end**: the same by-construction area
> cuts are the **die-area groundwork for the rung-③ ASIC**, where a smaller die is exactly what lowers
> **$/unit + power at manufacturing volume** (the ASIC is the *endgame*, not out of scope — it is
> sequenced after the FPGA proves PMF). The analysis stays valid — every lever is correctness-invariant
> (byte-identical token) — and the *study itself* is **deprioritized behind a green P1** (real-model
> fidelity + full-scale correctness): compute is NVMe/PCIe-starved and already cheap, so shrinking it
> buys cost/power headroom, not throughput. Revisit once P1 is green and the vendor flow (E1) can
> measure the real LUT delta.

How to make the FP8 compute die dramatically smaller (for a smaller / cheaper FPGA on rungs ①–②, and
lower $/unit + power for the rung-③ ASIC at volume — i.e. a cheaper, cooler **local single-user box**),
ranked and phased. Grounded in the architecture's defining property. (Every lever cuts BOM/power; **none
change the product's single-user interactive tok/s** — that speed is set by memory bandwidth and is
therefore *rung-dependent* [~5–8 rung ① / ~15–40 rung ② / ~40+ rung ③, all **[EST]**; see
[`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)] — the die is NVMe/PCIe-bandwidth-bound, so compute is
nearly free.)

## Thesis — an NVMe/PCIe-bandwidth-bound die should be *minimal*

The workload is **NVMe/PCIe-bandwidth-bound**: the die sits ~75–80 % idle behind the NVMe→DDR
expert stream ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md), [`ULTRA_PERF.md`](ULTRA_PERF.md); the DDR
tier is rung-dependent — DDR4 on rung ①, DDR5/HBM on rung ②, per
[`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). So
**compute speed is nearly free** — you can make the die much *slower* (more serial, more shared)
with **zero throughput loss**, up to the point where compute time exceeds the exposed NVMe/PCIe stall.
Yet the die today carries **parallel / duplicated compute hardware sized for a throughput the
NVMe/PCIe-starved workload cannot use** — separate `glm_matmul_fp8` instances per operator, duplicated
fp32 tail units. That unused parallelism is the miniaturization target.

### The "compute is nearly free" budget (the enabling constraint)
Die utilization ≈ 20–25 % → a **~4–5× compute-slowdown budget** before compute becomes the
bottleneck. Every serialization/sharing lever spends from this budget; stay under it and
throughput is unchanged. **Measurable now** (no vendor tools): the cycle-accurate emulation
(`EXPERT_STALL`, [`CYCLE_EMULATION.md`](CYCLE_EMULATION.md)) reports `cyc_per_tok` and the exposed
stall — run a serialized variant and confirm the token window is still stall-dominated
(`compute_cyc` still < `exposed_stall`). This is **enabler E2** below.

## Lever catalog (ranked by die-area impact)

Savings are **estimates** — the exact LUT delta is not measurable here (yosys 0.66 can't map the
FP8 datapaths through ABC; see [`PHYSICAL_SKY130.md`](PHYSICAL_SKY130.md)); the reduction is
by-construction and every lever keeps the **decoded token byte-identical** (verified functionally).

| # | lever | mechanism | est saving | time cost | budget-safe? | effort | risk |
|---|---|---|---|---|---|---|---|
| **L0** ✅ | compact config | right-size PE_N/DDR_NCH/KV_RESIDENT/EFIFO/CACHE_SLOTS (result-invariant) | PE array halved + smaller fabric | more cycles | ✔ | done | low (**DONE**, byte-identical) |
| **L1** ✅ | **cross-op matmul sharing** | *(bounded, DONE)* swiglu gate/up GEMMs run at different times → one shared `u_mm` via a 1-bit `up_pass` arbiter + 2:1 weight mux (`u_mm_u` removed) | **~12K LUT4** (2 swiglu × 6186 measured matmul core, 6→4 engines/block; −1519 generic cells/expert measured) | **≈ 0** (already sequential) | ✔ (free) | large refactor | **DONE byte-identical** (e8659bd) |
| **L2** ❌ | tail vector-ALU sharing | *(assessed — NOT bounded-viable)* only `glm_softmax` instances the pipelined primitives, and its 4 pipes are **distinct ops** (exp/add/mul/rsqrt — nothing to merge); RMSNorm/RoPE/act use **inline `glm_fp.vh` fp32 macros**, not shareable module instances | small (fp32 tail ≪ FP8 GEMM) | small | ✔ | high (cross-module scheduler) | **skip — reward≪risk** |
| **L3** ◐ | intra-op serialization | swiglu gate/up → 1 **captured by L1**; the remaining piece is the **cross-module 3-way hoist** (mla+router+swiglu → one engine) | further PE-array cut | 2×+ that op | ✔ *within budget* | high | **deferred** (needs PE_N=8 + top-level ports + arbiter) |
| **L4** ✅ | shared dequant/fold | after L1 each `glm_matmul_fp8` already carries a single BFP-accumulate + block-scale fold; **nothing further** | small | — | ✔ | falls out of L1 | **subsumed by L1** |
| **L5** ✅ | memory-fabric trim | *(assessed — already spent)* the 4 controllers (ddr5_xbar, kv_cache_pager, expert_cache_ctrl/pf) were **already trimmed** (6b2c82f, 899ea64): minimal-width regs, verilator-clean; QDEPTH off-limits (NVMe/PCIe latency-hide) | small | none | ✔ (except QDEPTH) | med | **no change (already minimal)** |
| **L6** ⚠ | bit-serial FP8 MAC | 1-bit/cycle multiply → tiny multiplier | large per-PE | **16–32×** | ✖ **OVERSHOOTS budget** (compute becomes the bottleneck) | high | **high — skip** unless a deeper-idle regime is proven |
| **L7** ⚠ | tail precision trade | bf16 tail → fp16/bf12 | moderate | none | n/a | med | **NOT byte-identical** (fidelity trade) — separate decision |
| **L8** ◐ | repo dead-code quarantine | *(legacy TPU DONE)* the legacy scalar **TPU v2.0** core (16 src + 19 TBs: `tpu_top`/`soc`/`axi`, decoder, regfile, `gemm_systolic`, `conv2d_unit`, `vector_alu`, …) is **isolated to `legacy/`**, off the GLM datapath — **`make all` is now GLM-only** (the prove-it gate; `make legacy` still builds the old core). Remainder (bf16 golden, redundant `batched_moe`) still in-tree | **0 on the chip** | — | ✔ | low (TPU done) | none (hygiene only) |

## Phased roadmap

- **Phase A — resource right-sizing (DONE).** L0 compact config: `synth-glm-compact` / `sim-glm-compact`,
  byte-identical token (`{0,11,11}`), defaults untouched. [`OPERATION_FLOW.md`](OPERATION_FLOW.md) §10.
- **Phase B — shared compute (DONE, bounded).** L1 cross-op matmul sharing landed on swiglu
  (gate/up → one `u_mm`, 6→4 engines/block, byte-identical). L2 tail-ALU sharing **assessed and
  skipped** (fp32 tail is inline-macro'd + distinct ops → not a bounded win; reward ≪ FP8-GEMM area).
- **Phase C — bounded serialization (mostly subsumed / deferred).** L3's swiglu part was captured by
  L1; L4 (shared fold) is subsumed (one fold per engine); L5 (fabric trim) was already spent. The
  **only remaining lever is the cross-module 3-way hoist** (mla+router+swiglu → one engine) — invasive
  (PE_N=8 + top-level ports + arbiter), **deferred to after E1** so the LUT payoff can justify the risk.
- **Phase D — validate & FPGA-fit.** E1 measure the real LUT/DSP/BSRAM on the vendor flow
  (Gowin EDA / nextpnr) to confirm the fit on the target FPGA (e.g. GW5AT-138), and E2 pin the
  exact serialization budget. **This is where the estimates become numbers.**
- **Out of scope / caution.** L6 (bit-serial — overshoots the budget), L7 (precision trade — not
  byte-identical, a fidelity decision), and cutting QDEPTH (hurts the NVMe/PCIe latency-hide).

## Enablers (unblock the above)

- **E1 — measurement.** yosys 0.66 cannot map the FP8 datapaths (ABC wall). A concrete cell/LUT
  number needs **Gowin EDA** (free, login-gated + license; a user step) or a newer yosys/nextpnr.
  Without it, all savings are by-construction; correctness is the verified invariant.
- **E2 — the serialization budget.** Use the `EXPERT_STALL` cycle-emulation to measure
  `compute_cyc` vs `exposed_stall` per token; every Phase-B/C lever must keep
  `compute_cyc < exposed_stall` (else throughput drops). This is buildable now.

## Verification methodology (how each lever stays honest)

1. **Byte-identical token** — the lever is a *result-invariant restructuring* (same math, different
   scheduling/sharing), so the system TB (`token == standalone glm_model_fp8`) must still print the
   SAME token stream. Mirror the `sim-glm-compact` gate: run baseline + reduced, `diff` the tokens.
2. **Budget check (E2)** — the cycle-emulation must still show the token window stall-dominated.
3. **Area (deferred to E1)** — measured on the vendor flow; until then, by-construction.

## Honest constraints
- **Big refactors.** L1–L4 restructure the die datapath (arbitrated shared engines), unlike the
  parametric L0. Higher engineering + verification cost.
- **Unmeasurable area here.** The LUT deltas are estimates until E1.
- **The budget is a ceiling.** Serialization is free only while `compute_cyc < exposed_stall`;
  past that (e.g. L6 bit-serial) throughput drops — the levers are ordered to respect it.

## Status
- **Phase A: DONE** (L0 compact config committed, byte-identical).
- **Phase B: DONE (bounded).** **L1 landed** (e8659bd): swiglu gate/up merged onto one `u_mm`
  engine → **6→4 FP8 GEMM engines/block, ≈12K LUT4** (2 × 6186 measured matmul core; −1519 generic
  cells/expert measured via yosys `stat`, [`PPA_FP8.md`](PPA_FP8.md) §1.3), **byte-identical** (token
  `{4,31,20}` gworst_rel 0.00689655; swiglu 1024 err/tol 0.1004; swiglu_pem 513; decoder 9 — all
  == baseline). **L2 assessed and skipped** (only softmax uses shareable primitive modules and its
  4 pipes are distinct ops; RMSNorm/RoPE/act use inline `glm_fp.vh` macros — a cross-module fp32
  scheduler is high-risk for a fp32-tail reward that is ≪ the FP8-GEMM area).
- **All L1-style bounded merges are now EXHAUSTED:** after L1, every chip module holds exactly
  **one** `glm_matmul_fp8` (mla `u_mm_fp8` already time-shares its 7 projections; router `u_gemv`;
  swiglu `u_mm`; mtp `u_proj`). L4 is subsumed (one fold per engine); L5 was already trimmed.
- **Phase C remaining = the cross-module 3-way hoist (L3).** Hoist mla+router+swiglu onto a
  *single* shared engine at the decoder-block level. This is the last real area lever but is
  **invasive**: PE_N reconcile (mla=4, router tile=8, swiglu TN=4 → PE_N=8), new top-level operand
  ports, and a 3-way arbiter — high byte-identical risk. **Deferred:** its payoff (4→3 engines/block)
  is **unmeasurable here** (yosys ABC wall), so it should be gated on **E1 (Gowin EDA)** proving the
  LUT delta justifies the risk, rather than done blind.
- **Phase D: gated on E1** (Gowin EDA) for the real LUT/DSP/BSRAM numbers on GW5AT-138.
