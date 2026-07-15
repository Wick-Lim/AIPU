# AIPU — a GLM-5.2 Q4_K local-inference accelerator in Verilog

[![Slides — AIPU Accelerator (prior FP8 architecture, branch fp8)](https://img.shields.io/badge/%F0%9F%93%8A%20Slides-prior%20FP8%20architecture%20%28branch%20fp8%29-999999?logo=googleslides&logoColor=white&labelColor=555)](https://docs.google.com/presentation/d/1EMqJOJCNTBaCVEf5EfYI_CymVg--K8ezVyCJcr7ovys/present)

> **📊 Presentation:** the slide deck ([Google Slides](https://docs.google.com/presentation/d/1EMqJOJCNTBaCVEf5EfYI_CymVg--K8ezVyCJcr7ovys/present)) documents the **prior FP8 architecture** (now preserved on branch **`fp8`** + tag `fp8-verified-baseline`). `main` has since retargeted to the **Q4_K local-inference** track described below; the deck has **not** been re-cut for Q4_K yet.

A synthesizable Verilog accelerator with one goal: **run one real model well on a local, offline
box** — the published GGUF k-quant of GLM-5.2,
[`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.2-GGUF), a 753B-param MoE
(~40B active/token, `GlmMoeDsaForCausalLM`) in ~4-bit **Q4_K**. The Q4_K **GEMM core** is
**bit-exact to an independent ggml-Q4_K reference** (`tools/q4k_ref.py`), the full GLM-5.2 operator
datapath is assembled in Q4_K and **elaborates clean at the true 753B shape**, and the whole thing is
wrapped by the single-module memory system (multi-channel DDR5 + NVMe expert cache + weight/boot
loaders + multi-clock CDC) that streams the real model — with the memory controllers **bounded-model-
checked and unbounded-k-induction-proven**. The assembled model has an **end-to-end numeric golden**
(`make model-q4k`: full forward vs our numpy reference, 1155 tests bit-exact), and the datapath
consumes the full **Q6_K/Q8_0/F16 mixed-type** checkpoint mix (`make mixedtype`). The goldens are our
**own** ggml reimplementation — now **cross-checked against real GGUF bytes at the dequant layer**
(376,586,240 weights across two real published GGUFs — Q4_K, Q6_K **and Q8_0** — bitwise-equal to
llama.cpp's own `dequantize_row_*` — [`docs/GGUF_CROSSCHECK.md`](docs/GGUF_CROSSCHECK.md)). What is
**not** yet done is honest and stated up front: llama.cpp **whole-runtime** numeric equality is
out-of-contract (attention/accumulation orders differ by design) and the 467 GB GLM file itself has
not been consumed end-to-end.
See [*What's proven*](#whats-proven--against-an-independent-ggml-q4k-reference-scoped-honestly)
below for the exact status of every claim.

> **The product is a LOCAL, single-user box that runs with the ethernet unplugged.** One box, one
> user, running the full 753B model **fully offline / air-gapped** — a frontier model finally usable
> *on the work, and in the disconnected places (SCIFs, OT/critical-infra, field/edge), you're currently
> locked out of* — and you own it outright. Nothing leaves because there is **no path out**: the audit
> is literally *"does it still work with the ethernet cable unplugged?" — yes.* That non-egress is the
> **proof, not the pitch**, and it ends the "secured cloud" debate — in-VPC, zero-retention, and TEE
> deployments all still need a connection and fail the unplugged test. Offline *alone* is table-stakes
> for any local box; the moat is the **combination — offline + full frontier (753B) + appliance/seat
> price**. (Provisioned once with the ~467 GB Q4_K weights — itself doable offline — then fully
> disconnected; model updates are physical re-provisioning.) No per-token API fees, and no vendor that
> can rate-limit, deprecate, or cut you off. The number that matters is **single-user
> interactive throughput**, and it's set by the hardware rung you build on: **~5–8 tok/s on the
> prove-it FPGA today → ~15–40 on the funded custom board [EST]** (rung ③ silicon reaches ~40+ at volume)
> after stacking the faithful levers — the old flat ~25–40 was the funded rung's number, not a
> near-term-cheap one; see the 3-rung [`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md). (These
> ranges now have a **measured design-point menu** [EST] behind them — the rung-③ **primary** design
> point is the **512GB full-residency box, ≈80 tok/s [measured-inputs EST]** (U(K) and the MTP
> accept rate r both GLM-family measured; ~95 if GLM-5.2's deeper MTP hits its published accept depth
> — [`docs/R3_APPLIANCE_SPEC.md`](docs/R3_APPLIANCE_SPEC.md)). **Execution order (2026-07-11): prototype-first** —
> the *build sequence* leads with a retail-parts prototype (24GB×20, 1280-bit, 480GB, **~110 tok/s [EST]**;
> §5c — presumes a MAC array sized to consume 1.54 TB/s, i.e. ~6.6K–7.3K lanes @490MHz per §3 (the
> spread is an unresolved bit/wt inconsistency: §3 uses 4.5, §2's own 467GB/753B implies 4.96); a smaller
> array bottlenecks the box below its bandwidth) that defers the 32GB NDA procurement and on-substrate packaging to the volume SKU. The streaming 54–127 tok/s point
> survives as the hybrid-SKU-if-h≥0.75 upside — see
> [`docs/H_MEASUREMENT.md`](docs/H_MEASUREMENT.md) and the update note below.) The design
> is deliberately NVMe/PCIe-bandwidth-bound to keep it cheap. Where these docs mention *aggregate /
> datacenter batching* (per-user ~0.14 tok/s at B≈256), that is a **secondary analysis of a
> different deployment**, not this appliance — see [`docs/USBC_PRODUCT_PLAN.md`](docs/USBC_PRODUCT_PLAN.md).

> **Naming.** **AIPU** (AI Processing Unit, repo [`Wick-Lim/AIPU`](https://github.com/Wick-Lim/AIPU))
> is the whole accelerator. The project was formerly *TPU*; the classic *5-stage scalar TPU
> core* (*"TPU v2.0"*, `tpu_*` modules) was a separate legacy design and has been **removed**
> from the repo — it was never on the GLM product path (see git history if you need it).
>
> **Branches:** `main` develops exactly one thing — the **GLM-5.2 Q4_K local-inference accelerator**
> at rung ① (FPGA prove-it), the offline single-user box, with the near-term goal being the working
> FPGA demo. The **prior FP8 datacenter-native track** is preserved on branch **`fp8`** + tag
> **`fp8-verified-baseline`**, and the compression research study lives on branch
> **`research/compression-study`** + tag **`compression-study-baseline`** (the older research
> prototype on `prototype`, frozen at `47fb7f8`) — all referenced here as prior/preserved, never
> current. The full product (rungs ②③) is the roadmap, not
> main's current code ([`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md),
> [`NEXT_STEPS_PLAN.md`](NEXT_STEPS_PLAN.md)).

> **Why Q4_K.** For a cost-constrained local appliance, Q4_K is the coherent target: the published
> [`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.2-GGUF) is **~467 GB, ~38%
> smaller than the 753 GB FP8 checkpoint** (the hot-set / routed-expert bytes scale down
> ~proportionally, and the BOM is memory-dominated), and Q4_K is the format local inference (llama.cpp)
> actually runs. The moat is stated **scoped**: the Q4_K GEMM core is **bit-exact to `tools/q4k_ref.py`,
> our own faithful reimplementation of ggml's `dequantize_row_q4_K`** — a reimplementation now
> **proven bitwise-equal to real GGUF bytes at the dequant layer** (376,586,240 weights — Q4_K/Q6_K/Q8_0, two real published GGUFs — vs llama.cpp's
> own dequant — [`docs/GGUF_CROSSCHECK.md`](docs/GGUF_CROSSCHECK.md)); llama.cpp's **whole-runtime**
> arithmetic (attention/accumulation order) stays out-of-contract, and the 467 GB GLM file itself has
> not been run end-to-end. The **full UD-Q4_K_XL mix is now consumable**: the
> dynamic mix's Q6_K/Q8_0/F16 tensors have RTL consumers, bit-exact to the same reimpl golden (see the
> mixed-type row below). tok/s stays **[EST]**. See [`docs/Q4K_RETARGET.md`](docs/Q4K_RETARGET.md) and
> [`docs/Q4K_SYSTEM_PLAN.md`](docs/Q4K_SYSTEM_PLAN.md).

---

## What's proven — against an independent ggml-Q4_K reference, scoped honestly

The project's defining property is verification discipline. This table is the honest status of the
**Q4_K** track on `main`. Each row is tagged **PROVEN** (a gated functional/bit-exact simulation),
**FORMAL** (a solver proof over the memory/control plane only), **ELABORATED** (a structural
elaboration, no functional golden), or **NOT-YET** (a real, currently-open gap). Every "bit-exact"
here means **bit-exact to our ggml-Q4_K reference `tools/q4k_ref.py`** — which is itself proven
bitwise-equal to a real published GGUF's bytes at the dequant layer
([`docs/GGUF_CROSSCHECK.md`](docs/GGUF_CROSSCHECK.md)); llama.cpp whole-runtime equality stays a
separate, out-of-contract question (see the ledger rows below).

| What | Status | Evidence (make target + counts) |
|---|---|---|
| **Q4_K primitives** — fp16→fp32 decode + `get_scale_min_k4` (`q4k.vh`) | **PROVEN — bit-exact vs ggml** | `make q4k` · `q4k_prim` **18/18** |
| **Q4_K GEMM core** (`glm_matmul_q4k`) — block dequant → fp32 MAC → bf16 | **PROVEN — bit-exact vs ggml Q4_K** (the one true bit-exact datapath result) | `make q4k` · `glm_matmul_q4k` **160/160** |
| **Q4_K MoE expert** (`swiglu_expert_q4k`) — gate/up/down + silu | **PROVEN — functional** (self-labeled; not bit-exact) | `make q4k` · `swiglu_expert_q4k` **240/240** |
| **Q4_K MoE router** (`moe_router_q4k`) — gate GEMV → sigmoid → top-K → renorm | **PROVEN — structural/functional invariants** (not a numeric golden) | `make q4k` · `moe_router_q4k` **40/40** |
| **Assembled Q4_K spec-decode loop** (`spec_decode_top`) | **PROVEN — spec==greedy** *(DUT-vs-DUT self-consistency; the "greedy golden" is itself a `glm_model_q4k` sharing the same weight ROMs — a real lossless-speculation safety property, **not** a numeric golden)* | `make unittests` · **18/18** |
| **Larger spec loops** (`spec_batched_top` / `spec_chain_top`, K>1) | **PROVEN — spec==greedy** vs an **independent** `glm_model_q4k` reference (same DUT-vs-DUT caveat; kept out of `unittests` — minutes-long) | `make spec-slow` |
| **Adaptive spec-chain draft depth** (`spec_depth_adapt` policy + `spec_decode_seq` `ADAPT`/`k_cur`) — runtime per-pass depth `k_cur`∈[1..K] | **PROVEN — spec==greedy under ANY depth schedule** (output-invariant by construction: `k_cur` only bounds how many drafts are scanned; `ADAPT=0` default is **yosys sequential-equivalence-proven** identical for existing consumers) | `make spec-adapt` · `spec_depth_adapt` **31,522** + `spec_decode_seq` K-sweep **3,702** (K = 1/2/3/4/6/8) + K=1-exact **621** |
| **Generic bf16/fp32 datapath twins** (`glm_matmul`, `mla_attn`, `glm_model`, `mtp_head`, …) — the structural siblings of the Q4_K units | **PROVEN — fp32/fp64-golden** (~35 per-unit TBs) — *but these are the generic twins with **zero** Q4_K; the assembled Q4_K path is not what they verify* | `make unittests` |
| **Memory-system controllers** — routing/one-hot, FIFO no-overflow/underflow, token-accounting, ECC identity, done-gates | **FORMAL — BMC**, 7 controllers + 1 ECC-ring datapath, no counterexample (bounded from reset) | `make formal` |
| **Selected controllers** (`boot_loader`, `kv_cache_pager`, `spec_decode_seq`, `ddr5_xbar`, `flash_xbar`) | **FORMAL — unbounded k-induction** (all reachable states; documented residual BOUNDED gaps in `docs/FORMAL.md`) | `make formal-ind` |
| **Whole 2-clock Q4_K product top** (`glm_q4k_system_cdc` + every Q4_K compute/memory/CDC leaf) | **ELABORATED** — yosys `hierarchy -check` + `check -assert` exit 0 (no unresolved hierarchy / comb loop / multiple driver / inferred latch); structural sign-off, **not a sim** | `make synth-glm` |
| **Full 753B UD-Q4_K_XL-shape elaboration** (`glm_model_q4k` at DIM 6144 / L=78 / 256-expert / VOCAB 154880) | **ELABORATED** — type/width check only, *"no stimulus, no golden, no run"* | `test/full_config_elab_wrap.v` ([`FULL_CONFIG_ELAB.md`](docs/FULL_CONFIG_ELAB.md)) |
| **Residency-SKU configuration** (`glm_q4k_system` `RESIDENT=1` — the rung-③ 512GB-LPDDR5X full-residency box: expert refills served by the DDR-tier crossbar, flash carries ONLY KV-spill + boot) | **PROVEN — two-DUT contrast sim**: RESIDENT=1 refills complete via tagged xbar reads (`TAG_EFILL`) while a sim invariant `$fatal`s if the expert class ever reaches the flash path; RESIDENT=0 DUT on the same stimulus drives the flash channel exactly as before, and the default-parameter module is **formally equivalent** (yosys `equiv_simple`+`equiv_induct`, sequential) to the pre-RESIDENT commit. RTL↔SKU block mapping: [`docs/R3_APPLIANCE_SPEC.md`](docs/R3_APPLIANCE_SPEC.md) §5a | `make resident` **19/19** + `make resident-equiv` **PROVEN** |
| **Batched MLA attention (`mla_attn_q4k` PE_M>1, sparse/per-row)** — the Q4_K sibling of the fp8 per-row oracle | **PROVEN — DUT-vs-DUT bit-exact**: a PE_M=3 batched forward (per-row pos / s_len / q-dependent DSA) === per-row PE_M=1 re-runs, plus dense↔sparse full-window cross-check and per-row-KV windows; selection divergence probed live (non-vacuous) | `make mla-sparse` **53/53** |
| **Batched assembled-model golden (`glm_model_q4k` PE_M>1)** — closes the last SCALE-FUNCTIONAL item-3 gap (the batch axis was spec==greedy-only) | **PROVEN — bit-exact**: PE_M=2 forward's row _r_ === standalone PE_M=1 on that row's token (logits+argmax+h_state), row 0 anchored to the assembled numpy golden | `make batched-q4k` |
| **Real-dims Q4_K operator sweep** — SCALE-FUNCTIONAL item 2 (the fp8 track's real-magnitude sweep, re-run on Q4_K) | **PROVEN — same goldens at real dims**: GEMM K=6144 (NSB=24) bit-exact, router 256/top-8, SwiGLU INTER=2048, softmax LEN=2048, rmsnorm LEN=6144, rope ROT_DIM=64 (`define overrides only — slice defaults byte-identical) | `make scale-ops` |
| **Cycle-accurate throughput / stall harness (`glm_q4k_system`)** — the fp8 perf harness ported to Q4_K (audit's most-repeated PENDING) | **MEASURED (not a golden, but tokens held bit-exact)**: slice ~10,896 cyc/token; residency pivot confirmed on real cycles — RESIDENT=1 exposes 35 stall cyc/token vs 2,567 at RESIDENT=0/FLASH_LAT=1024 (~73×). [`docs/CYCLE_EMULATION.md`](docs/CYCLE_EMULATION.md) | `make perf-q4k` |
| **End-to-end numeric golden for the assembled Q4_K model** (`glm_model_q4k` full forward: embed → L×(MLA+DSA+MoE) → final-norm → LM head → argmax) | **PROVEN — bit-exact vs our assembled numpy reference** (`tools/glm_model_q4k_ref.py`, which imports the same `q4k_ref.py` dequant) — logits + argmax + h_state all byte-identical. *Still our own reimpl, not llama.cpp — see the next row.* | `make model-q4k` **1155/1155** (+ `model-q4k-acthw` **1155/1155** proving the ACT_HW resource knob result-invariant) |
| **Bit-exactness to the real GGUF bytes (dequant layer)** | **PROVEN** — `q4k_ref.py` vs llama.cpp's own `dequantize_row_q4_K/_q6_K/_q8_0` on two real published GGUFs' raw blocks: **376,586,240 weights (Q4_K + Q6_K + Q8_0, the Q8_0 cross-confirmed on two distributions) all bitwise-equal** → by transitivity with the existing RTL≡q4k_ref gates, the RTL dequant ≡ the real files' ggml dequant. ([`docs/GGUF_CROSSCHECK.md`](docs/GGUF_CROSSCHECK.md)) | `tools/gguf_crosscheck.py` |
| **llama.cpp full-runtime numeric equality** | **NOT-YET / out-of-contract** — attention/accumulation orders differ by design; our contract is ggml-exact dequant (now proven on real bytes for Q4_K/Q6_K/Q8_0) + our own documented fp32 order. The 467 GB GLM file itself has not been consumed end-to-end | — |
| **Mixed-type path** (Q6_K / Q8_0 / F16 tensors the dynamic UD-Q4_K_XL mix keeps at higher precision) | **PROVEN — bit-exact vs ggml-reimpl goldens** — `q4k_mixed.vh` dequant primitives + per-column `w_type` routing in `glm_matmul_q4k` + `desc_wtype` in the weight loader; the loader→GEMM path is bit-exact for **all four types** incl. a 24-tile mixed sequence | `make mixedtype` · `q6k_prim` + `q8_0_prim` + `glm_matmul_mixed` **32/32** + `weight_loader_q4k_mixed` **192/192** |
| **FPGA fit — real Vivado synth + place & route on XCKU3P** | **MEASURED** — compact config + `ACT_HW=1`: **142.3K LUT (87.5%)**, ~100K FF, 421 DSP, 0 BRAM, fits and routes, hold met; routed Fmax **10.2 → 17.2 → 46.5 MHz** across the (bit-exact) fmax-repipeline rounds (rope / glm_act+rmsnorm / matmul), **campaign closed at 4.6×** — the worst path is now route-dominated (physical work, not arithmetic; 200 MHz-class is rung-②/③ work) — see [`fpga/README.md`](fpga/README.md) | `bash fpga/run_fit.sh` · `fpga/results/` |
| **Throughput / energy / BOM / TCO / LOI** | **NOT-YET [EST]** — every tok/s is roofline-modeled (now with **measured model-side inputs** — h proxy-traced, U(K) **GLM-family measured** on GLM-4.5-Air, and the MTP accept rate r **measured** via a vLLM MTP sweep: r₁=0.87, per-position 0.87/0.60/0.32/0.13/0.04, A_eff plateau ~2.9 → optimum K=1–2, design point ≈80 tok/s [measured-inputs EST] — [`docs/H_MEASUREMENT.md`](docs/H_MEASUREMENT.md)); BOM/TCO and the target LOI are planning docs, not evidence | — |

**Honest moat statement.** A **UD-Q4_K_XL-native GLM-5.2 RTL datapath** whose GEMM core is **bit-exact
to an independent ggml reimplementation** (`tools/q4k_ref.py`) for **all four checkpoint types**
(Q4_K/Q6_K/Q8_0/F16), whose **assembled full forward pass is bit-exact to a numpy reference of the
whole model** (`make model-q4k`, 1155 tests: logits+argmax+h_state), verified at a small-but-faithful
slice, **elaboration-clean at the real 753B UD-Q4_K_XL shape**, wrapped by memory-system controllers
with **BMC + unbounded k-induction** safety proofs, and **placed & routed on a real XCKU3P**
(87.5% LUT, routed Fmax 46.5 MHz — the bit-exact fmax-repipeline campaign is closed at 4.6×, the
remaining worst path being route-dominated) — with the **dequant layer proven on real GGUF bytes**
(376,586,240 weights — Q4_K/Q6_K/Q8_0 across two real published GGUFs — bitwise-equal to
llama.cpp's own dequant, [`docs/GGUF_CROSSCHECK.md`](docs/GGUF_CROSSCHECK.md)). Still honest and
open: llama.cpp **whole-runtime** numeric equality is out-of-contract by design, the 467 GB GLM
checkpoint has not been run end-to-end, and every tok/s, cost,
and LOI claim is **[EST]**, not measured on hardware.

**Modeled, not silicon — flagged [EST].** All throughput/energy figures come from a
bandwidth-roofline model (`tokens/s ≈ NVMe_BW / [(1−h)·footprint] · K`) — where the spec-decode
multiplier `K` must be read as **A/U(K) ≈ 1.1–1.3× at K=4** with the **measured** union factor
U(K) ([`docs/H_MEASUREMENT.md`](docs/H_MEASUREMENT.md)), not a full ×K — **not** from silicon —
though the roofline's two model-side multipliers now have **measured inputs**: expert-reuse
h-curves (proxy-traced from OLMoE — since the full-residency pivot, h decides only the hybrid SKU)
and the MTP union factor U(K), now **GLM-family measured** on a GLM-4.5-Air MoE-gate trace,
superseding the OLMoE first pass ([`docs/H_MEASUREMENT.md`](docs/H_MEASUREMENT.md)), and the compute die now has a **real routed
netlist** ([`fpga/README.md`](fpga/README.md)). The tok/s is **rung-dependent** — bandwidth is set by the chip's IO pins + PHYs,
which is set by budget ([`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)): **~5–8 tok/s on the
prove-it FPGA (rung ①), ~15–40 on the funded custom board (rung ②), ~40+ at volume (rung ③)** [EST],
with **~9 → ~3 J/token** [EST] after stacking the NVMe-bandwidth levers. Read them as an optimistic
ceiling ([`ULTRA_PERF.md`](docs/ULTRA_PERF.md), [`IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md)).
*Update — measured design-point menu [EST; U(K) is now **GLM-family MEASURED** — a GLM-4.5-Air
MoE-gate trace (U(4)=2.60–2.71, U(8)=4.19–4.41) supersedes the first-pass OLMoE proxy —
[`docs/H_MEASUREMENT.md`](docs/H_MEASUREMENT.md),
[`docs/MOE_LOCALITY_RESEARCH.md`](docs/MOE_LOCALITY_RESEARCH.md)]: NVMe ×1–2, no multipliers
~0.5–1 tok/s; 90 GB DRAM cache + 100 GB/s → 13–24; 90 GB + 200 GB/s (ONFI 64ch) → 25–47;
225 GB + 200 GB/s → 54–127. (Updated 2026-07: the rung-③ **primary** design point is now the
**512GB LPDDR5X full-residency box, design point ≈80 tok/s [measured-inputs EST]** (pre-measurement band ~76–95) —
[`docs/R3_APPLIANCE_SPEC.md`](docs/R3_APPLIANCE_SPEC.md); this streaming menu stays active for
rung-①, the hybrid upside SKU (h≥0.75), and >512GB checkpoints, so 54–127 is now the hybrid-SKU
note, not the primary design point.)* What
*is* validated on real RTL cycles is the roofline's underlying **memory-stall mechanism** (Q4_K measured,
`make perf-q4k`: exposed stall linear in `FLASH_LAT` — 11 cyc/token at 8 → 2,567 at 1024, RESIDENT=0 —
and 35 independent of `FLASH_LAT` at RESIDENT=1); the absolute tok/s
stays [EST] ([`CYCLE_EMULATION.md`](docs/CYCLE_EMULATION.md)).

**Out of scope** (vendor IP / hardware / resource-gated): DDR5/NVMe (PCIe)/USB-C **PHYs** (TB-stubbed),
**board bring-up** (the FPGA P&R itself is now done in-repo — [`fpga/README.md`](fpga/README.md) —
but running on a physical dev board needs the board + its pin XDC + a MIG bridge; the near-term
product is an FPGA card, with ASIC/tapeout **sequenced later as the rung ③ volume endgame** — see
[`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)), and a **full-model multi-GPU numeric validation** of the
assembled Q4_K forward pass at the real 753B shape. *(The tokenizer + host software scaffold exist —
see [`host/`](host/README.md).)*

---

## The target: `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`

UD-Q4_K_XL is a **dynamic k-quant mix**: most tensors are Q4_K; sensitive ones are kept at higher
precision (Q6_K / Q8_0 / F16). Each type dequantizes exactly per ggml, then the **same** GEMM contract
runs (dequant → fp32 MAC → bf16). **All four types now have RTL consumers**: `glm_matmul_q4k` selects
the dequant front-end per column via `w_type` (latched per tile, off the per-beat path) and
`weight_loader_q4k` carries the type in its descriptor (`desc_wtype`) — the loader→GEMM path is
bit-exact for every type, including a 24-tile mixed sequence (`make mixedtype`).

| Type | Block layout | Dequant | Golden (`q4k_ref.py`) | RTL consumer |
|---|---|---|---|---|
| **Q4_K** | 256 wt / 144 B: fp16 `d`,`dmin` + 12 B of 6-bit scales/mins + 128 B of 4-bit quants | `w = (d·sc)·q − (dmin·m)` | ✅ bit-exact | ✅ `q4k.vh` + `glm_matmul_q4k.v` (**160/160**) |
| **Q6_K** | 256 wt / 210 B: fp16 `d` + 16 int8 scales + 6-bit quants | `w = (d·sc)·(q−32)` | ✅ bit-exact | ✅ `q4k_mixed.vh` + `w_type` arm (**`q6k_prim` + mixed 32/32 + loader-mixed 192/192**) |
| **Q8_0** | 32 wt: fp16 `d` + 32 int8 | `w = d·q` | ✅ bit-exact | ✅ `q4k_mixed.vh` + `w_type` arm (same gates) |
| **F16** | passthrough | `w = fp16→fp32` | ✅ (exact) | ✅ `w_type` passthrough arm (same gates) |

**GEMM contract** (`glm_matmul_q4k`, bit-exact to `tools/q4k_ref.py:matmul_q4k_col`):
`out = bf16( Σ_k fp32(a_k) · w_deq_k )` — bf16 activations, per-weight ggml Q4_K dequant, the proven
fp32 sequential accumulate (the same accumulate as `glm_matmul_pipe`, weight source swapped), bf16 RNE
output. All fp32 ops are `glm_fp.vh`'s IEEE `fp32_mul`/`fp32_add`. The `modules_to_not_convert` set
(norms / router / embed / lm_head) stays **bf16** — a *matching* bf16 tail, not an approximation.

Architecture (the slice preserves every ratio; these are **model** dims, independent of quant): hidden
6144, 78 layers (`first_k_dense_replace=3`), 64 heads (`head_dim=192`), **MLA** latent attention
(`qk_nope 192 + qk_rope 64`, `v 256`, `kv_lora 512`, **`q_lora 2048`**), **MoE** 256 experts top-8 + 1
shared (`moe_intermediate 2048`), dense `intermediate 12288`, **DSA** sparse attention (`index_topk
2048`), vocab 154880, 1M context, `rope_theta 8e6` interleaved, RMSNorm `eps 1e-5`, MTP
(`num_nextn_predict_layers 1`).

> `q_lora 2048` was **confirmed against the real GLM-5.2 safetensors** (`q_a_proj.weight [2048,6144]`)
> during the prior FP8 track — an earlier DeepSeek-standard guess of `q_lora 1536` was corrected.
> `kv_lora 512` is **[PENDING safetensors]** (the DeepSeek-standard value, not yet directly confirmed
> against `kv_a_proj`). These are model-architecture facts, unchanged by the quant format. See
> [`ACCEL_GLM52.md`](docs/ACCEL_GLM52.md) (records `q_lora_rank = 2048`, confirmed vs real safetensors).

---

## How it works

```
  1 TB NVMe    ──►  flash_xbar    ──►  DDR working ──►  ddr5_xbar  ──►   Q4_K compute die   ──►  token
 (~467 GB Q4_K   N-channel banked    set / cache       N-channel     (MLA + DSA + MoE, ggml
  GGUF weights)  + deep queue        (LRU+freq+pf)     banked read    Q4_K dequant → fp32 MAC → bf16 tail)
```

The workload is **NVMe/PCIe-bandwidth-bound**, so the system is built around streaming MoE experts
from the NVMe SSD through a DDR working cache into a mostly-idle Q4_K die (tier: **NVMe** bulk/slow →
**DDR** hot set/fast → die). *(The DDR tier is rung-dependent — DDR4 on the prove-it FPGA, DDR5/HBM on
the custom board — see [`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md).)* `flash_xbar` is the committed name of
that storage-read fabric — a medium-agnostic address→weight-bytes crossbar with deep-queue
latency-hiding; in the product it fronts an **NVMe/PCIe host-controller backend** (the NAND-specific
backend is the swapped part, not the abstraction). Its N-channel banking maps to **PCIe lanes /
multiple NVMe drives** — order-of-magnitude ~3.5 GB/s per PCIe Gen3 ×4 drive, ~7 GB/s Gen4, scaling
with lanes/drives [EST]. Every Q4_K **weight** matmul dequantizes per ggml (256-weight super-block:
4-bit quants scaled by a 6-bit per-sub-block scale/min via `get_scale_min_k4`, with fp16 `d`/`dmin`)
→ fp32 MAC → bf16; norms, softmax, rope, residual and the activation×activation attention matmuls stay
bf16. The whole memory system (`expert_cache_pf`, `kv_cache_pager`, `ddr5_xbar`, the FIFO/arbiter/CDC
logic) is **byte-agnostic** — it moves addresses/slots/IDs, never weight bytes — so it carried over from
the FP8 track by parameter/doc, not logic ([`docs/Q4K_SYSTEM_PLAN.md`](docs/Q4K_SYSTEM_PLAN.md)).

**Verification methodology.** Every GLM-5.2 unit is checked against an **independent golden** — the
generic bf16/fp32 twins against fp64/fp32 X-aware goldens, the Q4_K GEMM core against the ggml-Q4_K
reference (`tools/q4k_ref.py`); on success a TB prints `ALL N TESTS PASSED`, on any mismatch it prints
the failing case and `$fatal`s. Regression is **byte-identical**; the memory controllers add **bounded
model checking** (yosys-smtbmc + z3), some lifted to **unbounded k-induction**. **No formal proof
touches the Q4_K numeric datapath** — formal scope is routing/FIFO/token-accounting/ECC/done-gate
safety only.

---

## Detailed status

### Q4_K datapath

The Q4_K numeric proof lives at three levels: the **GEMM core** (bit-exact vs ggml, all four
checkpoint types), the **assembled full forward** (`make model-q4k`: `glm_model_q4k` vs the numpy
reference `tools/glm_model_q4k_ref.py`, 1155 tests bit-exact on logits+argmax+h_state), and the
**assembled spec-loop** level (spec==greedy, DUT-vs-DUT). The standalone per-unit numeric TBs
(`glm_model_tb` / `mla_attn_tb` / …) still build against the **generic bf16/fp32 twins**; the
assembled Q4_K path's numeric golden is the `model-q4k` gate.

| Unit | What is Q4_K | Verification status |
|---|---|---|
| `q4k.vh` | fp16→fp32 decode, `get_scale_min_k4`, int→fp32 primitives | **bit-exact vs ggml** (`q4k_prim` 18/18) |
| `glm_matmul_q4k.v` | Q4_K-native block-scaled GEMM (dequant → fp32 MAC → bf16) | **bit-exact vs ggml Q4_K** (160/160) — the one true bit-exact datapath result |
| `swiglu_expert_q4k.v` | gate/up/down GEMMs on the Q4_K core + bf16 silu tail | **functional** vs Q4_K golden (240/240) |
| `moe_router_q4k.v` | gate GEMV Q4_K, bf16 sigmoid/topk/renorm | **structural/functional invariants** (40/40) |
| `mla_attn_q4k.v` | weight projections Q4_K, bf16 attn/rope/norm/softmax/dsa | exercised in the assembled `model-q4k` numeric golden + the spec loops (no *standalone* unit golden) |
| `glm_decoder_block_q4k.v` | one full Q4_K decoder layer | exercised in the assembled `model-q4k` numeric golden + the spec loops |
| `glm_model_q4k.v` | full Q4_K forward pass | **bit-exact vs the assembled numpy golden** (`make model-q4k`, 1155/1155) + **spec==greedy** in the spec loops (DUT-vs-DUT) |
| `mtp_head_q4k.v` | Q4_K multi-token-prediction (t+2) head | exercised in the spec loops |

The Q4_K wrappers carry the same `PE_M` decode-batching machinery as the prior FP8 track, and the
multi-sequence SoC top (`glm_q4k_soc_ms`) exists; what the ledger **proves** on `main` is the
**spec==greedy** safety property (`spec_decode_top` 18/18, plus the larger `spec_batched_top` /
`spec_chain_top` under `make spec-slow`, and spec==greedy under **any runtime depth schedule**
under `make spec-adapt`). Batching throughput and multi-seq serving remain a
**capability of the silicon**, not the B=1 personal box, and their numeric claims are scoped to
spec==greedy self-consistency — not an assembled-model numeric golden.

### Single-module system (real-753B memory/streaming hardware) — BUILT

These blocks are **byte-agnostic** (they move addresses/slots/IDs, never weight bytes), so they carried
over from the FP8 track unchanged in logic.

| Unit | Role | Verification |
|---|---|---|
| `expert_cache_pf.v` | DDR5 routed-expert cache: LRU + freq + prefetch | `make expert-cache` (LRU/freq/prefetch TBs, 623 + policy); **BMC-proven in BOTH modes** (PF_ENABLE=0 and =1, `make formal`) |
| `kv_cache_pager.v` | MLA latent-KV ring + DSA-gather + NVMe overflow; **`NSEQ` per-seq ring windows** | 73 tests (+ NSEQ>1); **BMC + k-induction** (+ ECC=1 datapath BMC) |
| `ddr5_xbar.v` | N-channel banked DDR5 read fabric (~N× BW) | 3073 tests (7.93× @8ch); **BMC + k-induction** |
| `flash_xbar.v` | N-channel banked **storage-read** fabric (deep queue hides read latency); fronts the **NVMe/PCIe** backend | 2049 tests (7.99× latency-hide); **BMC + k-induction** |
| `weight_loader_q4k.v` | checkpoint Q4_K (quants + `d`/`dmin`/scales) → matmul pull DMA | loader-fed == direct-fed |
| `boot_loader.v` | power-up NVMe→DDR5 model-load sequencer | 9240 tests; **BMC + k-induction** |
| `spec_decode_seq.v` | MTP speculative-decode sequencer (K>1 draft) | **BMC + k-induction** |
| **`glm_q4k_system_cdc.v`** | production top: Q4_K compute + xbar + loader + 2-clock host/USB CDC | structural sign-off (`make synth-glm`, exit 0) |

### Product-hardening (on `main`) — DONE

Whole-chip structural gate (`make synth-glm` on `glm_q4k_system_cdc`, caught a real multi-driver bug);
per-row DSA sparse decode + union key-fetch + SWIN scratch decouple; `spec_chain_top` multi-token
accept; SECDED scrub + lane-partitioned pager ECC + MBIST/ICG clock-gating; 2-domain `reset_sync` + CDC
sign-off; full-config elaboration study (found + guarded a latent dense≥MoE FFN-width constraint). See
[`P2_MEMORY_MAP.md`](docs/P2_MEMORY_MAP.md), [`P12_SCALEUP.md`](docs/P12_SCALEUP.md).

### Performance / power levers — format-agnostic (memory/power side)

The die is only ~20–25% utilized (NVMe-bandwidth-starved), so these memory/power levers — not
compute-side wins — are what move tok/s. All are measured on the RTL/trace harnesses; the absolute
tok/s they feed stays **[EST]**.

| Lever | What | Measured |
|---|---|---|
| `flash_xbar.v` | parallel storage-read channels (PCIe lanes / multi-NVMe) + deep outstanding queue | **7.99× latency-hide + N× banking** |
| `tools/flash_layout.py` | offline expert→channel (PCIe-lane/drive) placement (kill hotspots) | **39% → 55% of 8× peak (~+40%)** |
| `spec_decode_seq.v` K>1 | multi-token speculative draft | **K=2 ≈ +23%** (spec == greedy) |
| `clk_en_ctrl.v` | gate the ~75%-idle die | **73.75% of idle dynamic power gated** (formally safe, 13 064 checks) |
| `clk_throttle.v` | DVFS/eco frequency prescaler — run the die **f/div** in the ~4–5× slack | **peak-power/thermal cap** (USB-C "eco mode"), byte-identical, **BMC-proven** (`make formal`) |
| `expert_prefetch_top.v` | predictor-driven prefetch | **measured NO-OP** (popular experts already resident — honest) |

The compute-side die-shrink / accumulator / fold-pipeline wins were established on the **prior FP8
track** (branch `fp8`) — see the appendix. On an NVMe-bound die they improve area/power/timing but
**do not move tok/s**; the striping strategy for the bandwidth that *does* is in
[`FLASH_STRIPING.md`](docs/FLASH_STRIPING.md).

---

## Documents

- **[`docs/Q4K_RETARGET.md`](docs/Q4K_RETARGET.md)** — the Q4_K local-device numerics (FP8 → GGUF
  k-quants): the dequant math, the GEMM contract, the ggml-Q4_K golden, and the per-type status.
  **Start here for "what is Q4_K-exact and what isn't."**
- **[`docs/Q4K_SYSTEM_PLAN.md`](docs/Q4K_SYSTEM_PLAN.md)** — the non-trivial retarget work plan: the
  weight-bus swap at the die boundary, the four system tops, the weight path, and the Makefile Q4 gate.
- **[`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)** — **the honest hardware plan**: a 3-rung ladder (① prove-it FPGA ~5–8 tok/s now → ② funded custom board ~15–40 → ③ SoC/ASIC ~40+ at volume [EST]) — performance is set by memory bandwidth, which is set by the chip's IO/PHY budget. **Start here for "how fast, on what hardware."**
- **[`docs/R3_APPLIANCE_SPEC.md`](docs/R3_APPLIANCE_SPEC.md)** — the **rung-③ appliance concept spec (v3)** and the rung-③ **primary design point**: 512GB LPDDR5X full-residency box (the whole ~467 GB checkpoint resident; cold storage = one M.2 NVMe, boot-load ~70 s; the ONFI streaming tier is off the primary SKU, retained for the hybrid upside SKU) — design point **≈80 tok/s [measured-inputs EST]** with the adaptive spec-chain (GLM-Air-measured U(K) **and** measured accept rate r — the vLLM MTP sweep put the memory-bound optimum at K=1–2; ~95 if GLM-5.2's deeper MTP hits its published accept depth), **≥50–78W [EST, 재도출]** (DRAM rail 35–53W from §4's own 4–6 pJ/bit; SoC term UNVERIFIED — the old ~40–60W rested on a ~25–40W rail that reproduces at no design point), 120×80mm board with on-substrate 1024-bit memory, clock/node/lane derivation (**~5.2K lanes @490MHz** for this 1.1 TB/s point; **~7.3K** for the 1.54 TB/s v3-proto — re-derived, §3; the older "~3K" was sized against the retired 650 GB/s streaming rate), 12–16nm), BOM ~$1.8–2.4k, and the honest competitive bracket (Opus/Gemini-Pro-class per-user speed; Groq/Cerebras is a different bracket).
- **[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)** — product direction (RTL/silicon track): the fidelity gate, robustness/vendor-IP/physical/software/manufacturing phases, the **FPGA-card** product path (ASIC = the rung ③ volume endgame, sequenced after FPGA proves PMF).
- **[`docs/USBC_PRODUCT_PLAN.md`](docs/USBC_PRODUCT_PLAN.md)** — productization plan for the **USB-C external device** (the appliance track): form factor, power, thermal, host software, BOM/pricing (**[EST]**, planning-doc — not validated), phased D0–D5 gates. The heavy traffic stays internal → USB-C carries only tokens.
- **[`host/`](host/README.md)** — the **host software scaffold**: a local **OpenAI-compatible server** (`python3 host/aipu_server.py`, stdlib only) mirroring the RTL host interface, the **real GLM-5.2 BPE tokenizer** (+ byte fallback), the **GLM chat template**, OpenAI **sampling params**, and 3 backends — `MockDevice`, a **simulator backend** (drives the RTL model slice via `vvp`), and USB (later). `make host-test` (18 tests). *(Honest note: the simulator backend is **fp8-era** — it still hardcodes the removed `glm_model_fp8` build (now only on branch `fp8`), so it does not run on `main`; a `glm_model_q4k` port is pending.)*
- **[`fpga/`](fpga/README.md)** — the **FPGA fit, MEASURED**: Vivado batch synth + full place&route of the product top on **XCKU3P** (compact config + `ACT_HW=1`): **142.3K LUT / 87.5%, fits and routes**, routed Fmax **46.5 MHz** after three bit-exact repipeline rounds (**campaign closed at 4.6×** — the worst path is now route-dominated); includes the routable bring-up harness, the Docker/license bring-up notes, and the measured reports in `fpga/results/`.
- **[`docs/OPERATION_FLOW.md`](docs/OPERATION_FLOW.md)** — the end-to-end operational flow: boot (NVMe→DDR5), per-token decode through every block, weight streaming, batching + union-skip MoE + speculative decode, CDC, and the per-token bottleneck. **Start here for "how it all runs."**
- **[`docs/ACCEL_GLM52.md`](docs/ACCEL_GLM52.md)** — accelerator architecture: exact config, MLA + DSA + MoE detail, the fp64-golden methodology, memory/streaming, RTL build order.
- **[`docs/SYSTEM_SINGLE_PACKAGE.md`](docs/SYSTEM_SINGLE_PACKAGE.md)** — single-module system (Q4_K die + DDR working cache + 1 TB NVMe SSD): tiering, expert caching, the bottleneck/perf/cost model.
- **Perf / power / physical:** [`IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md) · [`LOW_POWER.md`](docs/LOW_POWER.md) (energy is ~80% NVMe-read bytes → amortize the fetch; DVFS **frequency** is RTL-realized via `clk_throttle`, the J/token half is voltage/vendor; projected ~9 → ~1.5–3 J/token [EST]) · [`ULTRA_PERF.md`](docs/ULTRA_PERF.md) · [`FLASH_STRIPING.md`](docs/FLASH_STRIPING.md) · [`CYCLE_EMULATION.md`](docs/CYCLE_EMULATION.md) (cycle-accurate: the memory-stall mechanism measured on real RTL cycles) · [`MINIATURIZATION.md`](docs/MINIATURIZATION.md) · [`FORMAL.md`](docs/FORMAL.md).
- **Verification / scale:** [`FULL_CONFIG_ELAB.md`](docs/FULL_CONFIG_ELAB.md) (the RTL elaborates clean at the **true 753B config** — verilator, 0 errors; **elaboration study, not a sim**) · [`COVERAGE.md`](docs/COVERAGE.md) (verilator line/toggle/branch structural coverage over the verilatable unit TBs, `make coverage` — structural, **not** a substitute for the fidelity suite) · [`P12_SCALEUP.md`](docs/P12_SCALEUP.md) · [`P2_MEMORY_MAP.md`](docs/P2_MEMORY_MAP.md).
- **Prior-track (FP8) evidence** — see the appendix below. Still on `main`: [`SCALE_FUNCTIONAL.md`](docs/SCALE_FUNCTIONAL.md), [`PHYSICAL_SKY130.md`](docs/PHYSICAL_SKY130.md). The pure-FP8 validation write-ups (`REAL_CKPT_VALIDATION.md`, `BIT_ACCURACY.md`, `PPA_FP8.md`, `MODAL_VALIDATE.md`) live on branch `fp8` (`git checkout fp8`) alongside the FP8 tooling they document.

---

## Build / test

```sh
brew install icarus-verilog verilator yosys     # iverilog 13.0, verilator 5.048, yosys 0.66

make unittests   # build+run every per-unit TB (GLM-5.2 bf16/fp32 twins + Q4_K units + spec_decode_top + system units)
make q4k         # the Q4_K sub-gate: q4k_prim 18 / glm_matmul_q4k 160 / swiglu_expert_q4k 240 / moe_router_q4k 40
make spec-slow   # the long spec-decode tops (spec_batched_top + spec_chain_top): spec==greedy at larger K
make spec-adapt  # adaptive draft depth (runtime k_cur in [1..K] + accept-rate policy): spec==greedy under ANY depth schedule
make formal      # bounded model checking (yosys-smtbmc + z3) of the memory controllers + clk_throttle
make formal-ind  # unbounded k-induction of boot_loader / kv_cache_pager / spec_decode_seq / ddr5_xbar / flash_xbar
make coverage    # verilator line/toggle/branch structural coverage over the verilatable unit TBs
make host-test   # host OpenAI-server + device-protocol + tokenizer scaffold tests (18)
make synth-glm   # yosys whole-chip structural gate on the Q4_K product top glm_q4k_system_cdc
make resident    # residency-SKU config gate (RESIDENT=1 vs =0 two-DUT contrast, 19 tests)
make resident-equiv  # formal proof: default-param glm_q4k_system == pre-RESIDENT commit (yosys equiv)
make cdc         # targeted structural CDC check of glm_q4k_system_cdc (not a commercial CDC tool)
make all         # the GLM rung-① FPGA prove-it gate: unittests + synth-glm + formal
```

The one true bit-exact datapath result, compiled standalone (**zsh does not word-split** — list sources
explicitly):

```sh
mkdir -p build
python3 tools/q4k_matmul_gen.py >/dev/null            # -> build/q4k_vec.txt (random tiles + ggml-Q4_K goldens)
iverilog -g2012 -Wall -I src -o build/glm_matmul_q4k_sim test/glm_matmul_q4k_tb.v src/glm_matmul_q4k.v
vvp build/glm_matmul_q4k_sim      # -> ALL 160 TESTS PASSED (bit-exact vs ggml Q4_K)
```

---

## Slice configuration & substrate

The RTL is built at a small-but-faithful **slice** keeping every operator and ratio:
MODEL_DIM=128, 6 layers (3 dense + 3 MoE), 4 heads, MLA nope16/rope16/v32, q_lora64/kv_lora32,
8-expert top-2 + 1 shared, INTER_MOE=64, INTER_DENSE=256, VOCAB=256, S_MAX=8. Running the real
753B model adds the memory/streaming system + array scaling ([`ACCEL_GLM52.md`](docs/ACCEL_GLM52.md)).

**FPGA fit is MEASURED** ([`fpga/README.md`](fpga/README.md)): the whole 2-clock product top
synthesizes AND places-and-routes on a **Kintex UltraScale+ XCKU3P** at the compact config with the
`ACT_HW=1` result-invariant knob — **142,320 LUT (87.5%), ~100K FF, 421 DSP, 0 BRAM**, hold
met, routed Fmax **46.5 MHz** after three bit-exact fmax-repipeline rounds (rope, glm_act+rmsnorm,
matmul — **campaign closed at 4.6×**: the remaining worst path is route-dominated, i.e. physical
work, not arithmetic, and 46.5 MHz sits in the bring-up demo's target band; 200 MHz-class is
rung-②/③ work). Historical note: a partitioned `synth_ecp5` of just the six memory-system
controllers already summed to ~85% of an ECP5-85, which is why the target moved up to the KU3P class. Because the workload is
NVMe/PCIe-bandwidth-bound (the die sits ~75–80% idle behind the NVMe storage), an FPGA card is the
committed **near-term** product — at this rung an ASIC's faster *compute* would be largely wasted. The
real ceiling is **memory bandwidth (IO pins + PHY)**, which is what an ASIC breaks (HBM stacks +
many-channel controllers + near-memory compute) — so ASIC is **not out of scope: it is the rung ③
volume endgame**, sequenced after the FPGA proves product-market fit
([`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)).

---

## Appendix — Prior track: FP8 (branch `fp8` + tag `fp8-verified-baseline`)

Before the Q4_K retarget, `main` developed a **datacenter-native FP8 E4M3** accelerator targeting the
published [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) safetensors. That track is
**preserved, not deleted** — every FP8 source file (`*_fp8.v`, `fp8_e4m3.vh`), TB, and evidence doc
lives on branch **`fp8`** (local + `origin/fp8`) and tag **`fp8-verified-baseline`**. It is referenced
here as **prior/preserved**; none of it is on `main`. The FP8 evidence rows below held **on that
branch** — they are **not** claims about the current Q4_K `main`:

| Evidence (FP8 track — branch `fp8`) | Against | Result |
|---|---|---|
| Operator bit-accuracy vs the real checkpoint | published `GLM-5.2-FP8` safetensors (`kv_a_proj` F8_E4M3) | **9216/9216 = 100%** bf16-exact, **argmax 16/16** (`REAL_CKPT_VALIDATION.md`, branch `fp8`) |
| FP8 E4M3 arithmetic | fp64, **exhaustive** | **ALL 66069** (256 decodes + all 256×256 multiplies) |
| Operators at REAL GLM-5.2 dims | fp64 goldens | GEMM K=6144, router 256/top-8, SwiGLU 2048, MLA real geo — bit-exact ([`SCALE_FUNCTIONAL.md`](docs/SCALE_FUNCTIONAL.md)) |
| Real sky130 place-and-route (`glm_matmul_fp8`) | SkyWater sky130 PDK, OpenROAD | synth→floorplan→legalized placement, **357,320 µm²**, post-placement timing MET ([`PHYSICAL_SKY130.md`](docs/PHYSICAL_SKY130.md)) |
| Modal partial-F1 (assembled real-weight FFN, first 6 layers) | HF reference | argmax 6/6, worst `max_abs` 0.0015 (`REAL_CKPT_VALIDATION.md`, branch `fp8`) |
| Truncated full-model token chain (real weights, DSA threaded) | fp32-accumulate ref | argmax match (20259 == 20259), top-8 preserved (`REAL_CKPT_VALIDATION.md`, branch `fp8`) |
| Compute-side PPA wins | — | BFP fixed-point accumulator −87.6% cells vs fp32-accumulate; fold-pipeline +25% fmax; `weight_decomp` 1.34× lossless — all **FP8-specific**, bit-identical (`PPA_FP8.md`, branch `fp8`) |

To inspect or run the FP8 track: `git checkout fp8` (or `git checkout fp8-verified-baseline`). The
memory-system controllers, CDC, ECC/MBIST, and clock-gating blocks are shared byte-agnostic logic and
exist on both branches.
