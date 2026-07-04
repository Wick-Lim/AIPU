# Chip miniaturization — plan

How to make the FP8 compute die dramatically smaller (for a smaller FPGA / lower cost / lower
power), ranked and phased. Grounded in the architecture's defining property.

## Thesis — a Flash-bandwidth-bound die should be *minimal*

The workload is **Flash-bandwidth-bound**: the die sits ~75–80 % idle behind the Flash→DDR5
expert stream ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md), [`ULTRA_PERF.md`](ULTRA_PERF.md)). So
**compute speed is nearly free** — you can make the die much *slower* (more serial, more shared)
with **zero throughput loss**, up to the point where compute time exceeds the exposed Flash stall.
Yet the die today carries **parallel / duplicated compute hardware sized for a throughput the
Flash-starved workload cannot use** — separate `glm_matmul_fp8` instances per operator, duplicated
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
| **L1** ⭐ | **cross-op matmul sharing** | mla / router / swiglu GEMMs run at *different times* in a layer → one shared `glm_matmul_fp8` via an arbiter (K/PE_N reconfigured per use) | **~20–32K LUT** (≈3–4 instances × ~8K) | **≈ 0** (already sequential) | ✔ (free) | large refactor | med (arbiter + per-use reconfig) |
| **L2** | tail vector-ALU sharing | one `glm_fp_pipe` (add/mul/exp/rsqrt) time-mux'd across softmax / RoPE / RMSNorm / act instead of duplicated fp32 logic | moderate (dup fp32) | small extra cycles | ✔ | med | med (scheduler) |
| **L3** | intra-op serialization | swiglu's parallel gate/up matmuls → 1 (2× that op); PE_M/PE_N → 1 extreme | further PE-array cut | 2×+ that op | ✔ *within budget* | med | med (verify budget) |
| **L4** | shared dequant/fold | after L1 the BFP accumulator + block-scale dequant fold is already one instance; further fold the drain | small | — | ✔ | low (falls out of L1) | low |
| **L5** | memory-fabric trim | simplify crossbar arbitration; **do NOT cut QDEPTH** (Flash-bound needs the latency-hide) | small | none | ✔ (except QDEPTH) | med | med |
| **L6** ⚠ | bit-serial FP8 MAC | 1-bit/cycle multiply → tiny multiplier | large per-PE | **16–32×** | ✖ **OVERSHOOTS budget** (compute becomes the bottleneck) | high | **high — skip** unless a deeper-idle regime is proven |
| **L7** ⚠ | tail precision trade | bf16 tail → fp16/bf12 | moderate | none | n/a | med | **NOT byte-identical** (fidelity trade) — separate decision |
| **L8** | repo dead-code quarantine | move the 44 non-chip modules (legacy TPU, bf16 golden, redundant `batched_moe`) out of the build | **0 on the chip** | — | ✔ | low | none (hygiene only) |

## Phased roadmap

- **Phase A — resource right-sizing (DONE).** L0 compact config: `synth-glm-compact` / `sim-glm-compact`,
  byte-identical token (`{0,11,11}`), defaults untouched. [`OPERATION_FLOW.md`](OPERATION_FLOW.md) §10.
- **Phase B — shared compute (the big structural win).** L1 cross-op matmul sharing (elegant,
  ~free in time) → then L2 tail-ALU sharing. Target ~2–3× die reduction. **Gate:** the system TB
  token stays byte-identical + the cycle-emulation confirms compute is still stall-dominated (E2).
- **Phase C — bounded serialization.** L3 intra-op + L4 shared fold + L5 fabric trim, each spending
  from the compute budget and re-checked against E2. Stop before compute overtakes the Flash stall.
- **Phase D — validate & tape-target.** E1 measure the real LUT/DSP/BSRAM on the vendor flow
  (Gowin EDA / nextpnr) to confirm the fit on the target FPGA (e.g. GW5AT-138), and E2 pin the
  exact serialization budget. **This is where the estimates become numbers.**
- **Out of scope / caution.** L6 (bit-serial — overshoots the budget), L7 (precision trade — not
  byte-identical, a fidelity decision), and cutting QDEPTH (hurts the Flash latency-hide).

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
- **Phase A: DONE** (compact config committed, byte-identical).
- **Phase B: planned** — L1 cross-op matmul sharing is the recommended next step (biggest area
  win, ~free in time, byte-identical). L2 follows.
- **Phases C/D: planned**, D gated on E1 (Gowin EDA) for real numbers.
