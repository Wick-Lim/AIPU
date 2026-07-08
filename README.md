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
checked and unbounded-k-induction-proven**. What is **not** yet done is honest and stated up front:
there is **no end-to-end numeric golden** for the assembled model (it is exercised only as
speculative-decode == greedy self-consistency), the goldens are our **own** ggml reimplementation
(never the real GGUF bytes or llama.cpp), and the datapath is **Q4_K-only** (no Q6_K/Q8_0/F16 mixed-type
path). See [*What's proven*](#whats-proven--against-an-independent-ggml-q4k-reference-scoped-honestly)
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
> near-term-cheap one; see the 3-rung [`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md). The design
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
> **`fp8-verified-baseline`** (and the older research prototype on `prototype`, frozen at `47fb7f8`) —
> referenced here as prior/preserved, never current. The full product (rungs ②③) is the roadmap, not
> main's current code ([`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md),
> [`NEXT_STEPS_PLAN.md`](NEXT_STEPS_PLAN.md)).

> **Why Q4_K.** For a cost-constrained local appliance, Q4_K is the coherent target: the published
> [`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.2-GGUF) is **~467 GB, ~38%
> smaller than the 753 GB FP8 checkpoint** (the hot-set / routed-expert bytes scale down
> ~proportionally, and the BOM is memory-dominated), and Q4_K is the format local inference (llama.cpp)
> actually runs. The moat is stated **scoped**: the Q4_K GEMM core is **bit-exact to `tools/q4k_ref.py`,
> our own faithful reimplementation of ggml's `dequantize_row_q4_K`** — **not** to the real downloaded
> GGUF bytes or to llama.cpp's runtime arithmetic, and **not** to the *full* UD-Q4_K_XL mix (the dynamic
> mix keeps sensitive tensors at Q6_K/Q8_0/F16, for which the RTL has **no consumer** — see the gaps
> below). tok/s stays **[EST]**. See [`docs/Q4K_RETARGET.md`](docs/Q4K_RETARGET.md) and
> [`docs/Q4K_SYSTEM_PLAN.md`](docs/Q4K_SYSTEM_PLAN.md).

---

## What's proven — against an independent ggml-Q4_K reference, scoped honestly

The project's defining property is verification discipline. This table is the honest status of the
**Q4_K** track on `main`. Each row is tagged **PROVEN** (a gated functional/bit-exact simulation),
**FORMAL** (a solver proof over the memory/control plane only), **ELABORATED** (a structural
elaboration, no functional golden), or **NOT-YET** (a real, currently-open gap). Every "bit-exact"
here means **bit-exact to our ggml-Q4_K reference `tools/q4k_ref.py`**, not to the real GGUF file.

| What | Status | Evidence (make target + counts) |
|---|---|---|
| **Q4_K primitives** — fp16→fp32 decode + `get_scale_min_k4` (`q4k.vh`) | **PROVEN — bit-exact vs ggml** | `make q4k` · `q4k_prim` **18/18** |
| **Q4_K GEMM core** (`glm_matmul_q4k`) — block dequant → fp32 MAC → bf16 | **PROVEN — bit-exact vs ggml Q4_K** (the one true bit-exact datapath result) | `make q4k` · `glm_matmul_q4k` **160/160** |
| **Q4_K MoE expert** (`swiglu_expert_q4k`) — gate/up/down + silu | **PROVEN — functional** (self-labeled; not bit-exact) | `make q4k` · `swiglu_expert_q4k` **240/240** |
| **Q4_K MoE router** (`moe_router_q4k`) — gate GEMV → sigmoid → top-K → renorm | **PROVEN — structural/functional invariants** (not a numeric golden) | `make q4k` · `moe_router_q4k` **40/40** |
| **Assembled Q4_K spec-decode loop** (`spec_decode_top`) | **PROVEN — spec==greedy** *(DUT-vs-DUT self-consistency; the "greedy golden" is itself a `glm_model_q4k` sharing the same weight ROMs — a real lossless-speculation safety property, **not** a numeric golden)* | `make unittests` · **19/19** |
| **Larger spec loops** (`spec_batched_top` / `spec_chain_top`, K>1) | **PROVEN — spec==greedy** vs an **independent** `glm_model_q4k` reference (same DUT-vs-DUT caveat; kept out of `unittests` — minutes-long) | `make spec-slow` |
| **Generic bf16/fp32 datapath twins** (`glm_matmul`, `mla_attn`, `glm_model`, `mtp_head`, …) — the structural siblings of the Q4_K units | **PROVEN — fp32/fp64-golden** (~35 per-unit TBs) — *but these are the generic twins with **zero** Q4_K; the assembled Q4_K path is not what they verify* | `make unittests` |
| **Memory-system controllers** — routing/one-hot, FIFO no-overflow/underflow, token-accounting, ECC identity, done-gates | **FORMAL — BMC**, 7 controllers + 1 ECC-ring datapath, no counterexample (bounded from reset) | `make formal` |
| **Selected controllers** (`boot_loader`, `kv_cache_pager`, `spec_decode_seq`, `ddr5_xbar`, `flash_xbar`) | **FORMAL — unbounded k-induction** (all reachable states; documented residual BOUNDED gaps in `docs/FORMAL.md`) | `make formal-ind` |
| **Whole 2-clock Q4_K product top** (`glm_q4k_system_cdc` + every Q4_K compute/memory/CDC leaf) | **ELABORATED** — yosys `hierarchy -check` + `check -assert` exit 0 (no unresolved hierarchy / comb loop / multiple driver / inferred latch); structural sign-off, **not a sim** | `make synth-glm` |
| **Full 753B UD-Q4_K_XL-shape elaboration** (`glm_model_q4k` at DIM 6144 / L=78 / 256-expert / VOCAB 154880) | **ELABORATED** — type/width check only, *"no stimulus, no golden, no run"* | `test/full_config_elab_wrap.v` ([`FULL_CONFIG_ELAB.md`](docs/FULL_CONFIG_ELAB.md)) |
| **End-to-end numeric golden for the assembled Q4_K model** | **NOT-YET** — the assembled Q4_K path is exercised **only** inside the spec loops (DUT-vs-DUT); nothing asserts the assembled forward pass matches ggml/llama.cpp numerically | — |
| **Bit-exactness to the real UD-Q4_K_XL GGUF / llama.cpp** | **NOT-YET** — all goldens are our own ggml reimpl (`tools/q4k_ref.py`), never the real GGUF bytes or llama.cpp runtime | — |
| **Mixed-type path** (Q6_K / Q8_0 / F16 tensors the dynamic UD-Q4_K_XL mix keeps at higher precision) | **NOT-YET** — RTL is **Q4_K-only** (`grep -ril q6_k\|q8_0 src/` = 0); those types have Python-only goldens in `q4k_ref.py` with **no RTL consumer**, so the chip cannot consume a real UD-Q4_K_XL checkpoint as-is | — |
| **Throughput / energy / FPGA-fit / BOM / TCO / LOI** | **NOT-YET [EST]/[PENDING]** — every tok/s is roofline-modeled; no PnR/Fmax/LUT result in-repo (Vivado/Gowin-blocked); BOM/TCO and the target LOI are planning docs, not evidence | — |

**Honest moat statement.** A **Q4_K-native GLM-5.2 RTL datapath** whose Q4_K GEMM core is **bit-exact to
an independent ggml-Q4_K reference** (`tools/q4k_ref.py`), verified at a small-but-faithful slice,
**elaboration-clean at the real 753B UD-Q4_K_XL shape**, and wrapped by memory-system controllers with
**BMC + unbounded k-induction** safety proofs — **not yet** bit-verified against the real GGUF /
llama.cpp, exercised end-to-end only as **spec==greedy self-consistency** (no assembled-model numeric
golden), and **Q4_K-only** (no Q6_K/Q8_0/F16 mixed-type path). Every tok/s, FPGA-fit, cost, and LOI
claim is **[EST]/[PENDING]**, not measured.

**Modeled, not silicon — flagged [EST].** All throughput/energy figures come from a
bandwidth-roofline model (`tokens/s ≈ NVMe_BW / [(1−h)·footprint] · K`), **not** from a routed
netlist or silicon. The tok/s is **rung-dependent** — bandwidth is set by the chip's IO pins + PHYs,
which is set by budget ([`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)): **~5–8 tok/s on the
prove-it FPGA (rung ①), ~15–40 on the funded custom board (rung ②), ~40+ at volume (rung ③)** [EST],
with **~9 → ~3 J/token** [EST] after stacking the NVMe-bandwidth levers. Read them as an optimistic
ceiling ([`ULTRA_PERF.md`](docs/ULTRA_PERF.md), [`IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md)). What
*is* validated on real RTL cycles is the roofline's underlying **memory-stall mechanism** (exposed stall
exactly `3·FLASH_LAT + 9`, faithful `cyc_per_tok` grows with storage-read latency); the absolute tok/s
stays [EST] ([`CYCLE_EMULATION.md`](docs/CYCLE_EMULATION.md)).

**Out of scope** (vendor IP / hardware / resource-gated): DDR5/NVMe (PCIe)/USB-C **PHYs** (TB-stubbed),
**full-chip FPGA P&R + board bring-up** (needs Gowin/Vivado + a board; the near-term product is an
FPGA card, with ASIC/tapeout **sequenced later as the rung ③ volume endgame** — see
[`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)), and a **full-model multi-GPU numeric validation** of the
assembled Q4_K forward pass. *(The tokenizer + host software scaffold exist — see
[`host/`](host/README.md).)*

---

## The target: `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`

UD-Q4_K_XL is a **dynamic k-quant mix**: most tensors are Q4_K; sensitive ones are kept at higher
precision (Q6_K / Q8_0 / F16). Each type dequantizes exactly per ggml, then the **same** GEMM contract
runs (dequant → fp32 MAC → bf16). The **RTL consumes Q4_K only** — the higher-precision types have
bit-exact goldens in `tools/q4k_ref.py` but **no RTL consumer** yet (see the gap in the table above).

| Type | Block layout | Dequant | Golden (`q4k_ref.py`) | RTL consumer |
|---|---|---|---|---|
| **Q4_K** | 256 wt / 144 B: fp16 `d`,`dmin` + 12 B of 6-bit scales/mins + 128 B of 4-bit quants | `w = (d·sc)·q − (dmin·m)` | ✅ bit-exact | ✅ `q4k.vh` + `glm_matmul_q4k.v` |
| **Q6_K** | 256 wt / 210 B | `w = d·sc·(q−32)` | ✅ (Python-only) | ❌ none — **NOT-YET** |
| **Q8_0** | 32 wt: fp16 `d` + 32 int8 | `w = d·q` | ✅ (Python-only) | ❌ none — **NOT-YET** |
| **F16** | passthrough | `w = fp16→fp32` | ✅ (exact) | primitive only (`q4k.vh`) |

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

The Q4_K numeric proof lives at the **GEMM-core** level (bit-exact vs ggml) and the **assembled
spec-loop** level (spec==greedy, DUT-vs-DUT). The standalone per-unit numeric TBs
(`glm_model_tb` / `mla_attn_tb` / `glm_decoder_block_tb` / `mtp_head_tb`) build against the **generic
bf16/fp32 twins** (`src/glm_model.v` / `mla_attn.v` / …, **zero Q4_K**), so the assembled Q4_K path
(`glm_model_q4k` …) has **no standalone numeric golden** — see the NOT-YET rows above.

| Unit | What is Q4_K | Verification status |
|---|---|---|
| `q4k.vh` | fp16→fp32 decode, `get_scale_min_k4`, int→fp32 primitives | **bit-exact vs ggml** (`q4k_prim` 18/18) |
| `glm_matmul_q4k.v` | Q4_K-native block-scaled GEMM (dequant → fp32 MAC → bf16) | **bit-exact vs ggml Q4_K** (160/160) — the one true bit-exact datapath result |
| `swiglu_expert_q4k.v` | gate/up/down GEMMs on the Q4_K core + bf16 silu tail | **functional** vs Q4_K golden (240/240) |
| `moe_router_q4k.v` | gate GEMV Q4_K, bf16 sigmoid/topk/renorm | **structural/functional invariants** (40/40) |
| `mla_attn_q4k.v` | weight projections Q4_K, bf16 attn/rope/norm/softmax/dsa | exercised **only** in the assembled spec loops (no standalone Q4_K numeric golden) |
| `glm_decoder_block_q4k.v` | one full Q4_K decoder layer | exercised **only** in the assembled spec loops |
| `glm_model_q4k.v` | full Q4_K forward pass | **spec==greedy** in the spec loops (DUT-vs-DUT); **no** numeric golden |
| `mtp_head_q4k.v` | Q4_K multi-token-prediction (t+2) head | exercised in the spec loops |

The Q4_K wrappers carry the same `PE_M` decode-batching machinery as the prior FP8 track, and the
multi-sequence SoC top (`glm_q4k_soc_ms`) exists; what the ledger **proves** on `main` is the
**spec==greedy** safety property (`spec_decode_top` 19/19, plus the larger `spec_batched_top` /
`spec_chain_top` under `make spec-slow`). Batching throughput and multi-seq serving remain a
**capability of the silicon**, not the B=1 personal box, and their numeric claims are scoped to
spec==greedy self-consistency — not an assembled-model numeric golden.

### Single-module system (real-753B memory/streaming hardware) — BUILT

These blocks are **byte-agnostic** (they move addresses/slots/IDs, never weight bytes), so they carried
over from the FP8 track unchanged in logic.

| Unit | Role | Verification |
|---|---|---|
| `expert_cache_pf.v` | DDR5 routed-expert cache: LRU + freq + prefetch | 623 tests; **BMC-proven** (PF_ENABLE=0) |
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
- **[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)** — product direction (RTL/silicon track): the fidelity gate, robustness/vendor-IP/physical/software/manufacturing phases, the **FPGA-card** product path (ASIC = the rung ③ volume endgame, sequenced after FPGA proves PMF).
- **[`docs/USBC_PRODUCT_PLAN.md`](docs/USBC_PRODUCT_PLAN.md)** — productization plan for the **USB-C external device** (the appliance track): form factor, power, thermal, host software, BOM/pricing (**[EST]**, planning-doc — not validated), phased D0–D5 gates. The heavy traffic stays internal → USB-C carries only tokens.
- **[`host/`](host/README.md)** — the **host software scaffold**: a local **OpenAI-compatible server** (`python3 host/aipu_server.py`, stdlib only) mirroring the RTL host interface, the **real GLM-5.2 BPE tokenizer** (+ byte fallback), the **GLM chat template**, OpenAI **sampling params**, and 3 backends — `MockDevice`, a **simulator backend** (drives the RTL model slice via `vvp`), and USB (later). `make host-test` (18 tests). *(Note: the simulator backend still points at the prior `glm_model_fp8` build path and is being retargeted to `glm_model_q4k`.)*
- **[`fpga/`](fpga/README.md)** — the **FPGA-fit vendor-flow scaffold** (Gowin GW5AT-138 / Tang Mega 138K): `gw_sh` synth+P&R script + SDC (host/core async clocks) + compact-config wrapper + nextpnr fallback. Run it (needs Gowin, a user step) for the real LUT/DSP/BSRAM/Fmax — **NOT-YET** in-repo, the unknown that gates device size/thermal/BOM/price.
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
make formal      # bounded model checking (yosys-smtbmc + z3) of the memory controllers + clk_throttle
make formal-ind  # unbounded k-induction of boot_loader / kv_cache_pager / spec_decode_seq / ddr5_xbar / flash_xbar
make coverage    # verilator line/toggle/branch structural coverage over the verilatable unit TBs
make host-test   # host OpenAI-server + device-protocol + tokenizer scaffold tests (18)
make synth-glm   # yosys whole-chip structural gate on the Q4_K product top glm_q4k_system_cdc
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

**FPGA fit is NOT-YET measured.** No PnR/Fmax/LUT result for the Q4_K compute die is in-repo — that is
the unknown the [`fpga/`](fpga/README.md) vendor-flow scaffold exists to close (needs Gowin/Vivado, a
user step). What *is* known structurally: a partitioned `synth_ecp5` of the six memory-system
controllers already sums to **~71,475 LUT4 — ~85% of an ECP5-85 on the controllers alone** — so the
full system does **not** fit an ECP5-85 and needs a larger FPGA. Because the workload is
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
