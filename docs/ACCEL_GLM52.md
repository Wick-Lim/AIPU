# ACCEL_GLM52 — the Q4_K compute-die architecture spec for GLM-5.2 (`GlmMoeDsaForCausalLM`)

> **Track & scope (2026-07-08).** This is **THE architecture spec** for the compute die: the
> MLA + DeepSeek-DSA sparse attention, the 256-expert MoE, the MTP speculative head, the
> operator hierarchy, and the exact GLM-5.2 config. The **current / `main` track is Q4_K-native**
> — the datapath streams **GGML Q4_K** weights (per-ggml dequant → fp32 MAC → bf16), targeting
> the published [`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.2-GGUF).
> Unlike the prior FP8 track there is **no activation quant**: activations are **bf16**
> throughout; only *weights* are quantized. Modules are named `glm_*_q4k` (`glm_model_q4k`,
> `glm_decoder_block_q4k`, `mla_attn_q4k`, `moe_router_q4k`, `swiglu_expert_q4k`,
> `mtp_head_q4k`, `glm_matmul_q4k`, `glm_q4k_soc`/`glm_q4k_soc_ms`, `weight_loader_q4k`), plus
> the shared bf16 leaves `glm_softmax`, `rmsnorm_unit`, `rope_interleave_unit`, `dsa_indexer`,
> `topk_select`, `sampler`. The **prior FP8 datacenter-native track** — a *different arithmetic
> contract* (FP8 activations + FP8 weights) — is preserved on branch **`fp8`** + tag
> **`fp8-verified-baseline`**, referenced here as prior/preserved, never current.

> **What is actually proven (read this before any "bit-exact" below).**
> - The bit-exact-vs-ggml results are the **Q4_K GEMM core** (`make q4k` →
>   `glm_matmul_q4k`, `q4k_prim`) and — no longer Q4_K-only — the **Q6_K/Q8_0/F16 mixed-type
>   path** (`make mixedtype`: `src/q4k_mixed.vh` dequant primitives, per-column `w_type` routing
>   in `glm_matmul_q4k`, `desc_wtype` in `weight_loader_q4k`), both bit-exact to the team's
>   **own** ggml reimplementation `tools/q4k_ref.py` — **NOT** the real downloaded GGUF bytes and
>   **NOT** llama.cpp's runtime (llama.cpp quantizes activations to Q8_K and dots in integer;
>   this RTL uses bf16 activations + fp32 accumulate — a **different** arithmetic contract).
> - The **assembled `glm_model_q4k` now HAS an end-to-end numeric golden** (`make model-q4k`):
>   the full forward (embed → L×(MLA+DSA+MoE) → final norm → LM head → argmax) is bit-exact —
>   **ALL 1155 TESTS** (logits+argmax+h_state) — vs the numpy reference
>   `tools/glm_model_q4k_ref.py`, plus `make model-q4k-acthw` (the same golden through the
>   ACT_HW=1 serialized-activation datapath, also 1155). Caveat that stays: the golden is the
>   team's **own** numpy reimpl, **NOT** llama.cpp/GGUF. **Spec-decode == greedy
>   self-consistency** remains as an additional DUT-vs-DUT safety property.
> - The generic **bf16/fp32 twins** (`glm_model`, `mla_attn`, `mtp_head`, …) that carry the
>   per-unit fp32/fp64-golden TBs are the *structural siblings* of the Q4_K units — they contain
>   **zero Q4_K**, so they do **not** verify the assembled Q4_K numeric path.
>
> See the honest status table in [`README.md`](../README.md) for the exact per-claim evidence.

> Chief-architect synthesis. ONE coherent architecture combining **compute-completeness**
> (every real GLM-5.2 operator → a concrete hardware unit) with **memory-for-scale**
> (tiered memory + expert/weight streaming + 1M-context latent-KV paging).
>
> **Honesty contract.** Two things are kept rigorously separate throughout:
> - **DERIVED / BUILDABLE** — the small-but-faithful RTL decoder block we actually
>   build and verify on iverilog/verilator/yosys: the Q4_K GEMM core bit-exact to the
>   independent ggml-Q4_K reference (`tools/q4k_ref.py`), the surrounding operators against
>   fp32/fp64 goldens, and the assembled model bit-exact vs its own end-to-end numpy golden
>   (`tools/glm_model_q4k_ref.py`, `make model-q4k`) plus spec==greedy self-consistency.
> - **SYSTEM-LEVEL ESTIMATE** — the full 753B-param multi-chip / streaming machine, designed on
>   paper, sized from the config, never claimed as "built".
>
> Where a number is a system estimate it is tagged **[SYS-EST]**; where it is proven by
> the buildable slice it is tagged **[BUILT]**; **[DERIVED]** = mapped but pending build.
>
> **Local-device target (Q4_K).** For the local appliance, the target weight store is the
> published `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL` (**~467 GB**, ~38% smaller than the 753 GB the
> same model would take in one byte/param; the on-box footprint / hot-set / routed-expert bytes
> below scale ~proportionally). UD-Q4_K_XL is a **dynamic mix** — most tensors Q4_K, sensitive
> ones kept at Q6_K/Q8_0/F16 (~0.6 B/param average). The **moat, stated scoped:** the Q4_K GEMM
> core is **bit-exact to `tools/q4k_ref.py`, the team's own faithful reimplementation of ggml's
> `dequantize_row_q4_K`** — **NOT** bit-exact to the real downloaded GGUF file or to llama.cpp.
> The RTL now **consumes the full UD-Q4_K_XL type mix**: Q6_K/Q8_0/F16 have RTL consumers
> (`src/q4k_mixed.vh` dequant primitives, per-column `w_type` routing in `glm_matmul_q4k`,
> `desc_wtype` in `weight_loader_q4k`; `make mixedtype`, bit-exact to the same `tools/q4k_ref.py`
> reimpl) — so a real checkpoint's type mix can be consumed natively, though still not
> bit-verified against the real GGUF bytes. See [`Q4K_RETARGET.md`](Q4K_RETARGET.md) and
> [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md).

---

## 1. Target: exact GLM-5.2 config + honest scale reality

### 1.1 Exact config (from HF `zai-org/GLM-5.2` config.json)

| Field | Value |
|---|---|
| architectures | `GlmMoeDsaForCausalLM` |
| hidden_size H | 6144 |
| num_hidden_layers L | 78 |
| vocab V | 154880 |
| max_position_embeddings | 1,048,576 (1M) |
| dtype | bfloat16 |
| **MLA attention** | 64 heads; qk = rope 64 + nope 192 = **256**; v_head = **256**; num_kv_heads 64; attention_bias **false** |
| rope_theta | 8e6; **rope_interleave true** (decoupled RoPE on the 64-dim rope part only; NoPE on the 192 part) |
| **DSA sparse** | index_topk **2048**; index_topk_freq **4**; index_skip_topk_offset **3** (IndexShare: 1 fresh indexer pass per 4 layers, reused by the next 3); indexer_rope_interleave true |
| **MoE** | n_routed_experts **256**; num_experts_per_tok **8** (top-8); n_shared_experts **1**; moe_intermediate_size **2048**; routed_scaling_factor **2.5**; moe_layer_freq 1; **first_k_dense_replace 3** (layers 0–2 dense FFN, intermediate 12288) |
| FFN | SwiGLU, hidden_act silu (gate/up + silu(gate)⊙up + down) |
| norm | RMSNorm eps **1e-5**, pre-attn + pre-FFN + QK-norm in MLA + final |
| MTP | num_nextn_predict_layers **1** (speculative t+2 head) |
| scale | **~753B total**, **~40B active/token** |

**Underspecified fields (the ONLY two assumptions).** config.json does not expose
`q_lora_rank` / `kv_lora_rank`. GLM-5.2 is DeepSeek-MLA-derived → we size with the
DeepSeek-standard **kv_lora_rank = 512** (**[PENDING safetensors]** — the standard value, not
yet directly confirmed against `kv_a_proj`) and **q_lora_rank = 2048** (**confirmed vs the real
GLM-5.2 safetensors** `q_a_proj.weight [2048,6144]` during the prior FP8 track — an earlier
DeepSeek-standard guess of 1536 was corrected). These are model-architecture facts, independent
of the quant format; both are RTL parameters, overridable from the real weights. Everything else
above is exact.

### 1.2 Honest scale reality — what one chip can and cannot do

**Product scope (who this is for).** The target is a **local, single-user personal box** that
runs the full GLM-5.2 model **fully offline / air-gapped — nothing leaves because there is
no path out** (the audit is literally "does it work with the ethernet unplugged?" — yes). The
on-box residency described in this doc is exactly what makes that possible: the entire **~467 GB**
UD-Q4_K_XL GGUF lives **on the box**, streamed from a ~1 TB NVMe SSD (M.2 / PCIe) with a fast DDR
hot-weight cache (rung-dependent — DDR4 on the prove-it FPGA, DDR5/HBM on the funded custom board;
see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)) (see [`USBC_PRODUCT_PLAN.md`](USBC_PRODUCT_PLAN.md)),
so after a **one-time provisioning** load (itself doable in a secure facility; new-model/weight
updates are a physical re-provision) the card serves one user (**B=1**) with no internet and no
cloud, ever. That unlocks frontier-model use in the disconnected / locked-out environments cloud
can't reach (SCIFs, isolated OT/critical-infra, field/edge, air-gapped compliance) and removes
vendor dependency (can't be rate-limited, deprecated, or cut off) — and it categorically excludes
every cloud option, **including "secured cloud"** (in-VPC / zero-retention / TEE enclaves), which
all need connectivity and fail the unplugged test. The moat is the **combination** — offline
**and** full-frontier (753B) **and** appliance/seat price — not offline alone (a 70B laptop model
is offline too). Any multi-chip / aggregate-batch framing in this doc is a **secondary, non-target
datacenter analysis of the same silicon**, kept for sizing — not the product's deployment.

- **Weights:** ~753B params = **725B cold routed experts** (75 MoE layers × 256 experts ×
  37.75M) + **~28B hot** (MLA projections, dense-front FFN, norms, router, embed/LM head).
  bf16 ≈ **1.5 TB**; the shipped **UD-Q4_K_XL** dynamic mix (~0.6 B/param avg) ≈ **~467 GB**
  (a pure 4-bit Q4_K image would be ~376 GB; the sensitive Q6_K/Q8_0/F16 tensors add the rest).
  **[SYS-EST]**
- **Latent-KV cache:** 576 elts/token/layer (c_kv 512 + shared k_rope 64) =
  **1.125 KB/token/layer bf16**, 87.8 KB/token across 78 layers. 128K ctx → **11.8 GB**;
  1M ctx → **94.2 GB**. (A 64-head MHA cache would be 670 GB / 5.36 TB — MLA is ~57×
  smaller.) **[SYS-EST]**
- **Verdict:** No single buildable chip holds 1.5 TB of weights + ~94 GB of cache. GLM-5.2
  is **inherently a large-memory / multi-chip streaming system.** We are honest about this.
  - **One chip CAN:** hold the HOT working set (MLA proj, shared expert, router, norms,
    rope angle table), the active 8/256 routed experts for the current layer/batch, the DSA
    top-2048 gather window, and the GEMM/attention tile memory; and stream cold experts +
    append/gather the latent cache over AXI DMA.
  - **One chip CANNOT:** resident-hold 725B cold experts or the full 1M-context cache.
    Those live in tiered DRAM/HBM (and across chips at full scale).
- **What we BUILD:** a small-but-faithful decoder block (§8) that keeps **every operator and
  every structural ratio intact** and runs on one FPGA/sim, using the **same DMA
  gather/append datapath** so the streaming control logic is exercised at small scale.

---

## 2. Architecture overview (text block diagram)

```
                       ┌──────────────────────────────────────────────────────────┐
                       │       LAYER CONTROL (glm_model_q4k / glm_decoder_block_q4k)│
                       │  walks 78 layers · per-layer mode:                         │
                       │   • FFN mode: DENSE(L0-2, inter 12288) | MoE(L3-77)        │
                       │   • indexer mode: FRESH (L mod 4 == 3) | REUSE (next 3)    │
                       │   • MLA→DSA→ATTN→FFN→residual; bf16 residual stream,        │
                       │     Q4_K weight dequant→fp32 MAC→bf16 out (glm_matmul_q4k)  │
                       └───────────────┬──────────────────────────────┬───────────┘
                                       │ control / activation tiles    │ weight-pull / DMA
   ┌───────────────────────────────────┴──────────────────────────────┴───────────────────┐
   │                          DECODER-BLOCK DATAPATH (one layer)                            │
   │                                                                                        │
   │  x (bf16 residual stream)                                                              │
   │     │                                                                                  │
   │  [rmsnorm_unit] pre-attn ──────────────► MLA ATTENTION (mla_attn_q4k orchestrator)     │
   │     │                                     │                                            │
   │     │   Q path:  W_dq→q_lora→[rmsnorm]→W_uq→ split nope192 | rope64                    │
   │     │   KV path: W_dkv→c_kv(512)*  W_kr→k_rope(64)*  [rmsnorm(c_kv)]→W_uk,W_uv          │
   │     │            (* = appended to LATENT CACHE)                                        │
   │     │   [rope_interleave_unit] θ=8e6 on q_rope & k_rope only (fp32 angles)             │
   │     │             │                                                                    │
   │     │             ▼                                                                    │
   │     │      DSA INDEXER (dsa_indexer.v)  ── small-dim score over ALL keys →             │
   │     │        topk_select.v → top-2048 index list  (IndexShare cache / reuse)           │
   │     │             │ index list                                                         │
   │     │             ▼                                                                    │
   │     │      [gather 2048 K/V rows] → inner attention (in mla_attn_q4k)                  │
   │     │        QK^T(qk=256) · causal mask · [glm_softmax bf16] · A·V(v=256) · W_o         │
   │     │             │   (1/√qk_head_dim softmax scale APPLIED — see §4.1)                 │
   │  x += ◄───────────┘  (bf16 residual add)                                               │
   │     │                                                                                  │
   │  [rmsnorm_unit] pre-FFN ──────────────►  FFN                                           │
   │     │      DENSE mode (L0-2):  swiglu_expert_q4k (inter 12288)                         │
   │     │      MoE mode (L3-77):   moe_router_q4k (W_g→sigmoid→top-8→renorm→×2.5)          │
   │     │                     → 8 routed swiglu_expert_q4k + 1 shared → combine (fp32 acc) │
   │  x += ◄───────────┘  (bf16 residual add)                                               │
   └───────────────────────────────────────────────────────────────────────────────────────┘
        │ (after 78 layers)                              ▲ weights / cache
        ▼                                                │
   [rmsnorm_unit final] → LM head (W_lm 6144×154880, bf16 glm_matmul_pipe GEMV) → [sampler.v]
   MTP head (mtp_head_q4k): own norm + small attn/FFN, hidden+pred-embed → t+2, shares LM head

   ┌──────────────────────── MEMORY / STREAMING SYSTEM (glm_q4k_soc / glm_q4k_system_cdc) ──┐
   │ T0 on-chip SRAM: activations, GEMM tiles, index-set scratch, softmax scratch,          │
   │    HOT weights, active 8/256 experts, 2048-row gather window                            │
   │ T1 DDR/HBM: hot-weight overflow + recent expert working set + active cache window       │
   │ T2 NVMe/host: full 725B cold experts + long-tail latent cache + ~467 GB Q4_K image      │
   │                                                                                        │
   │ weight_loader_q4k / flash_xbar / ddr5_xbar / expert_cache_pf / kv_cache_pager /         │
   │ cdc_async_fifo:                                                                         │
   │   • EXPERT STREAM: router top-8 ids → DMA gather experts (dominant traffic)             │
   │   • CACHE APPEND : write [c_kv|k_rope] per token/layer (ring buffer)                    │
   │   • CACHE GATHER : kv_cache_pager reads the 2048 DSA-selected rows                       │
   │   • two-clock CDC: host/bus domain ↔ core (compute) domain                              │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Operator coverage table — EVERY GLM-5.2 op → a hardware unit

Precision policy (as built, Q4_K track): **weights Q4_K, activations bf16, reductions fp32,
outputs bf16 — no activation quant.** Every *weight* matmul dequantizes its Q4_K super-block per
ggml (`w = d·sc·q − dmin·m`, `get_scale_min_k4`) to fp32; the activation arrives **bf16**; the
fp32 products are **sequentially fp32-accumulated in K order** and rounded to **bf16 (RNE)** on
output (`glm_matmul_q4k`, the same accumulate structure as the proven bf16 `glm_matmul_pipe` —
only the weight source changed). The MoE combine uses an **fp32 accumulator**; the **residual
stream stays bf16 across all 78 layers** (a real GLM-5.2 `modules_to_not_convert` boundary — not
an fp32 residual). Norms, softmax, RoPE, router, embed and the LM head stay **bf16** (the
`modules_to_not_convert` set). All fp32 ops are `glm_fp.vh`'s IEEE `fp32_mul` / `fp32_add` — the
datapath's canonical numerics (no fixed-point Q-format; the legacy scalar-TPU `tpu_defs.vh`
Q15.16/Q7.8 path was removed, see the correction below).

Weight-type key: **Q4_K** = weight matmul, ggml Q4_K dequant → fp32 MAC → bf16 (`glm_matmul_q4k`;
the same engine also consumes UD-Q4_K_XL's Q6_K/Q8_0/F16 columns via per-column `w_type` —
`src/q4k_mixed.vh`, `make mixedtype`, bit-exact to the same reimpl golden).
**bf16** = a `modules_to_not_convert` op or an activation×activation matmul (`glm_matmul_pipe` /
`glm_softmax` / the elementwise leaves). No activation quant anywhere.

| # | GLM-5.2 operator | Hardware unit (as-built Q4_K) | Precision |
|---|---|---|---|
| 1 | Embedding (id → row 6144) | gather DMA (`weight_loader_q4k` / row fetch) | bf16 gather/store |
| 2 | RMSNorm (pre-attn, pre-FFN, q_lora, c_kv, final) | `rmsnorm_unit` | **fp32** Σx² reduce + rsqrt, bf16 γ / I·O |
| 3 | MLA Q down/up (W_dq, W_uq) | `glm_matmul_q4k` (orch by `mla_attn_q4k`) | **Q4_K** wt, bf16 act, fp32 acc → bf16 |
| 4 | MLA KV down (W_dkv, W_kr) → c_kv, k_rope | `glm_matmul_q4k` + cache-append DMA | **Q4_K** wt, bf16 act, fp32 acc → bf16 |
| 5 | MLA KV up (W_uk, W_uv) → k_nope, v | `glm_matmul_q4k` | **Q4_K** wt, bf16 act, fp32 acc → bf16 |
| 6 | Decoupled RoPE (interleave, θ=8e6, 64-dim only) | `rope_interleave_unit` | **fp32** angle table, bf16 apply |
| 7 | DSA indexer scoring (small-dim dot over all keys) | `dsa_indexer` | **fp32** score accum, bf16 in |
| 8 | DSA top-2048 select (+causal recent window) | `topk_select` (in `dsa_indexer`) | index compare (argmax-style) |
| 9 | IndexShare reuse (freq 4, offset 3) | `mla_attn_q4k` / block control + index cache | control |
| 10 | QK^T over selected keys (qk 256) | inner attention in `mla_attn_q4k` (act×act) | **bf16** ×, fp32 acc → bf16 *(1/√d applied, §4.1)* |
| 11 | Causal mask + softmax over N≤2048 | `glm_softmax` | **bf16** max + exp-sum + probs |
| 12 | A·V (v_head 256) + output proj W_o | AV in `mla_attn_q4k` (bf16) + `glm_matmul_q4k` (W_o) | bf16 AV / **Q4_K** W_o |
| 13 | MoE router (W_g → sigmoid → top-8) | `moe_router_q4k` (W_g via `glm_matmul_q4k`) | **Q4_K** W_g GEMV, bf16 tail, fp32 score |
| 14 | Gate renorm-to-1 **then** ×2.5 | `moe_router_q4k` (two explicit stages) | **fp32** → bf16 (silent-bug guard) |
| 15 | SwiGLU expert (W_gate, W_up, silu⊙, W_down) | `swiglu_expert_q4k` | **Q4_K** wt, bf16 act, silu, fp32 acc → bf16 |
| 16 | Dense-front FFN (L0-2, inter 12288) | `swiglu_expert_q4k` dense mode | **Q4_K** wt, bf16 (mode flag) |
| 17 | MoE combine (Σ gₑ·yₑ + y_shared, residual) | combine stage in `glm_decoder_block_q4k` | **fp32** accum → bf16 residual add |
| 18 | MTP head (t+2 speculative) | `mtp_head_q4k` | **Q4_K** wt, bf16 act, fp32 reduce |
| 19 | Final RMSNorm | `rmsnorm_unit` | fp32 reduce (bf16 I·O) |
| 20 | LM head (W_lm 6144×154880 GEMV) | `glm_matmul_pipe` (streamed) | **bf16** GEMV (`modules_to_not_convert`) |
| 21 | Sampling (temp/top-k/top-p/multinomial) | `sampler` (+`glm_softmax`) | fp32/bf16 logits |
| — | Expert streaming + cache append/gather | `glm_q4k_soc` / `flash_xbar` / `ddr5_xbar` / `expert_cache_pf` / `kv_cache_pager` / `cdc_async_fifo` | DMA burst |

> **⚠️ History note (as-built vs early plan).** An earlier version of this doc mapped these
> operators onto the classic **scalar-TPU** tensor units (`gemm_ml`, `gemm_systolic`,
> `softmax_unit`, `attention_unit`, `scatter_gather`, `fused_ops_unit`, `tpu_soc`, `tpu_axi`,
> `tile_memory`, `tpu_defs.vh` with its Q15.16/Q7.8 fixed-point). That was **early planning
> only** — those modules were **LEGACY**, **never on the GLM product path**, and have been
> **removed** from the repo (see git history). The as-built GLM datapath instantiates its own
> purpose-built units: the Q4_K weight engine `glm_matmul_q4k` (and its bf16 sibling
> `glm_matmul_pipe` for the `modules_to_not_convert` tail), `glm_softmax`, `mla_attn_q4k`,
> `swiglu_expert_q4k`, `moe_router_q4k`, `mtp_head_q4k`, `glm_act`, plus the shared leaves
> `rmsnorm_unit` / `rope_interleave_unit` / `dsa_indexer` / `topk_select` / `sampler`. The table
> above already lists the **as-built** units.

**As-built Q4_K compute units:** `glm_matmul_q4k` (+ bf16 `glm_matmul_pipe`), `glm_softmax`,
`rmsnorm_unit`, `rope_interleave_unit`, `dsa_indexer` (+`topk_select`), `mla_attn_q4k`,
`moe_router_q4k`, `swiglu_expert_q4k`, `mtp_head_q4k`, `glm_act`, `sampler`, orchestrated by
`glm_decoder_block_q4k` / `glm_model_q4k`.
**Memory / streaming:** `weight_loader_q4k`, `flash_xbar`, `ddr5_xbar`, `expert_cache_pf`,
`kv_cache_pager`, `boot_loader`, `cdc_async_fifo`, `reset_sync` — wrapped by `glm_q4k_soc` /
`glm_q4k_soc_ms` / `glm_q4k_system_cdc`.

---

## 4. MLA + DSA detail

### 4.1 MLA latent attention (`mla_attn_q4k.v` orchestrator)

`mla_attn_q4k` is an **FSM orchestrator** (not a monolithic datapath): it sequences the Q4_K
weight engine `glm_matmul_q4k`, `rmsnorm_unit`, `rope_interleave_unit`, `dsa_indexer`,
`glm_softmax`, and its own inner QK^T/AV datapath over on-chip scratch, and owns the latent cache.
All projection weights (W_dq, W_uq, W_dkv, W_kr, W_uk, W_uv, W_o) are **Q4_K** (ggml dequant →
fp32 MAC → bf16); the activation×activation QK^T and A·V are **bf16** (`glm_softmax` in between).

- **Q path:** `x(6144) → W_dq(6144×2048) → q_lora(2048) → rmsnorm → W_uq(2048×16384) →
  q(64 heads × 256)`. Each head's 256 split **nope[192] | rope[64]** by lane slicing.
  Parameterized so a no-q-LoRA collapse to one 6144×16384 proj is also supported.
- **KV path (the compression):** `x → W_dkv(6144×512) → c_kv(512)` and
  `x → W_kr(6144×64) → k_rope(64, shared across all 64 heads)`. **Only c_kv + k_rope are
  cached** (576 elts/token/layer). At attention time `c_kv → rmsnorm → W_uk → k_nope(64×192)`
  and `→ W_uv → v(64×256)` reconstruct K/V on the fly (the K/V up-projections depend only on the
  key, so across a batch of query rows they are the **shared** fetch — see §5). Per head
  K=[k_nope192|k_rope64]=256, V=256.
- **QK-norm:** RMSNorm applied on **q_lora** and on **c_kv** (the MLA-internal norms,
  eps 1e-5, fp32 reduce) — not just pre-attention.
- **Decoupled RoPE:** `rope_interleave_unit` rotates **only** q_rope[64] (per head) and the
  single shared k_rope[64]; the 192 nope dims pass through. **Adjacent-pair interleave**
  (rotate (x[2i], x[2i+1]) — NOT rotate_half), **θ=8e6**, cos/sin from an **fp32 angle
  table** (position up to 2²⁰ makes θ^(−2i/64) span a range bf16 angles cannot resolve),
  applied in bf16.
- **Softmax scale — gap FIXED.** `mla_attn_q4k` now applies the `1/√qk_head_dim` softmax scale
  once at score capture (elaboration-time fp32 constant `SM_SCALE_F32`, bit-identical to the
  numpy golden's `np.float32(1)/np.sqrt(np.float32(QK_DIM))`). The earlier omission was a real
  correctness gap; it was exposed and closed by the assembled end-to-end golden
  (`make model-q4k` vs `tools/glm_model_q4k_ref.py` — §6).
- **Absorb mode (designed, [SYS-EST] — not in the as-built unit).** In decode one can fold W_uk
  into W_uq and W_uv into W_o so K/V are never materialized and attention runs directly on c_kv.
  The as-built `mla_attn_q4k` **materializes** K/V per key (sharing that fetch across the batch's
  query rows); absorb-mode is a **paper** decode-time optimization, **not** currently implemented
  or verified.
- **Cache footprint:** 1.125 KB/token/layer bf16; append + gather via `kv_cache_pager`.

### 4.2 DSA indexer + top-2048 + IndexShare (`dsa_indexer.v` + `topk_select.v`)

- **Indexer scoring:** project q and cached latent (c_kv/k_rope) to a **small indexer head
  dim** (decoupled-rope'd, indexer_rope_interleave=true, same `rope_interleave_unit`); cheap
  **dot-product score s(q,k_j) over ALL past keys j** — the one O(S) streaming pass over the
  ring buffer, fp32 accumulate. Far cheaper than full attention (small dim).
- **Top-k select:** keep **index_topk=2048** highest scores per query (argmax/top-k via a
  streaming threshold + partial-bitonic `topk_select`, **NOT** a dense softmax), **union with
  the causal recent window**; future keys structurally excluded; explicit causal mask still
  applied on the recent window. Emits a 2048-entry index list.
- **Dense fallback:** when S ≤ index_topk the selector is a **no-op** (all keys kept) ⇒ exact
  dense attention. Exercised as a slice mode (S=4, topk=8) against the bf16 twin.
- **IndexShare FSM:** index_topk_freq=4 + index_skip_topk_offset=3 ⇒ a **fresh** index set is
  computed on layers {3, 7, 11, …} and **reused** by the next 3 layers. ~20 indexer passes
  cover all 78 layers, not 78. The block control (`glm_decoder_block_q4k` + `mla_attn_q4k`) holds
  the per-window valid index buffer + a layer-mod-4 counter.
- **FLOP cap (the whole point):** QK^T + A·V is constant `64·2048·512·2 = 134.2 MFLOP/query`
  once S>2048. vs dense: 2.0× cheaper at 4K, 64× at 128K, **512× at 1M**. **[SYS-EST]**

---

## 5. MoE detail (`moe_router_q4k.v` + `swiglu_expert_q4k.v` + combine)

- **Router:** `logits = x · W_g(6144×256)` via `glm_matmul_q4k` (W_g is a **Q4_K** weight,
  ggml dequant → fp32 MAC → bf16; the routing "tail" is bf16). GLM/DeepSeek-v3 style:
  **sigmoid** (or softmax — parameterized) scores, group/**top-8 of 256** via `topk_select`.
- **Gate math — ORDER IS CORRECTNESS-CRITICAL** (two explicit pipeline stages, never folded):
  (1) **renormalize** the 8 selected gate weights to sum 1; **then** (2) multiply by
  **routed_scaling_factor = 2.5**. `moe_router_q4k`'s TB checks the renorm/scale invariants
  (`make q4k` → `moe_router_q4k` **40/40** — structural/functional invariants, *not* a numeric
  golden); wrong order is a silent bf16-tolerance-passing-but-wrong bug.
- **SwiGLU expert** (8 routed + 1 always-on shared, moe_inter=2048): `g = x·W_gate(6144×2048)`,
  `u = x·W_up(6144×2048)` on `glm_matmul_q4k` (**Q4_K** weights, bf16 activations); `h = silu(g)
  ⊙ u` (silu + elementwise multiply via `glm_act`, fp32 then bf16); `y_e = h · W_down(2048×6144)`
  (**Q4_K**). Shared expert: identical shape, always runs, gate=1. `make q4k` → `swiglu_expert_q4k`
  **240/240** (functional, self-labeled — *not* bit-exact).
- **Combine:** `out = Σ_{e∈top8} gate_e·y_e + y_shared`, **fp32 accumulate**, then **bf16
  residual-add** (the residual stream is bf16 — §3). Combine lives in `glm_decoder_block_q4k`.
- **Dense-front (L0,1,2):** `swiglu_expert_q4k` in **dense mode**, intermediate_size=12288, no
  router, always-active. One unit covers both FFN modes (mode flag selects inter size).
- **Streaming (dominant DMA traffic) [SYS-EST]:** per token only 8/256 experts active
  (8·37.75M = 302M params ≈ **~604 MB bf16 / ~170 MB Q4_K per MoE layer**; consistent with the
  roofline's ~14 GB of routed experts across all 75 MoE layers per token). Router top-8 ids key
  an `expert_cache_pf` / `weight_loader_q4k` DMA gather of exactly those experts from the NVMe/DDR
  tiers into the resident expert buffer; shared expert + MLA proj + norms are HOT/resident, routed
  experts COLD/streamed. **Batching tokens amortizes loads** (route a whole batch, load each
  needed expert once).
  - **Batching present in the as-built Q4_K RTL.** The Q4_K units
    (`swiglu_expert_q4k`, `moe_router_q4k`, `mla_attn_q4k`, `mtp_head_q4k`, `glm_decoder_block_q4k`,
    `glm_model_q4k`, `glm_q4k_soc_ms`) all carry the `PE_M` parameter with `[0:PE_M-1]` per-row
    buffers, so **B token rows share ONE weight-fetch stream** (at `PE_M=1` every PE_M-indexed
    construct constant-folds to the committed single-token forward — byte-identical). In
    `glm_decoder_block_q4k` the `PE_M>1` grouped MoE fetches **only the UNION of the selected
    experts** across the batch (an expert-axis scan + combinational membership test, not all
    `N_EXPERT`). On the real 256-expert config this is up to **~32× fewer NVMe expert fetches** at
    small batch (union of ≤8 vs 256), tending to 1× as `B→256` where the union approaches the full
    expert set. **[SYS-EST]**
  - **Multi-sequence batching (`PER_ROW_SEQ`).** *(An aggregate / multi-user serving capability of
    the same silicon — a **secondary, non-target datacenter regime**, NOT the single-user product,
    which runs one sequence at `B=1`. The same `PE_M` weight-sharing also serves the product by
    batching a sequence's speculative-decode draft tokens.)* With `PER_ROW_SEQ=1` each `PE_M` row
    is a **DIFFERENT sequence**: `mla_attn_q4k` builds a per-row-slot union and emits `kc_seq` to
    route every KV fetch to that sequence's own cache window, so **each row attends its OWN
    sequence's KV** while the **query-side weight/projection fetch stays SHARED** (the batching
    bandwidth win). `PER_ROW_SEQ`/`seq_vec`/`kc_seq`/`SWIN` thread model→decoder→mla; a batched
    multi-seq SoC top (`glm_q4k_soc_ms`) drives it with a real `NSEQ`-window `kv_cache_pager` +
    `expert_cache_pf` + host FSM (prefill B seqs → 1 forward → commit B tokens), and at `N_STEPS>1`
    becomes a continuous-batching decode loop.
  - **⚠️ Verification status (honest).** The *bit-exact weight-share*, *union-byte-identical*, and
    *multi-seq per-row-bit-exact* results — with the specific regression counts (swiglu 513 /
    router 192 / mla 6 / mtp 44; `glm_*_multiseq_tb`: 2 seqs ~41% / B=4 ~52% fewer attn-weight
    beats) — were established on the **PRIOR FP8 track** (branch `fp8`; the multiseq TBs are FP8).
    Those checks are also **DUT-vs-DUT self-consistency** (a batched run vs independent `PE_M=1`
    runs of the *same* model), **not** a numeric golden vs ggml/llama.cpp. On `main` the Q4_K
    units carry the identical `PE_M`/`PER_ROW_SEQ` parameters, but a **Q4_K re-run of these
    batching regressions is [PENDING]**; the FP8 counts above are kept **as prior-FP8
    measurements**, not relabeled as Q4_K.

---

## 6. Correctness & quantization + golden methodology

This is the honest verification picture. Read "golden" carefully — the goldens are **the team's
own** references, never the real GGUF bytes or llama.cpp.

**(a) The bit-exact-vs-ggml results — the Q4_K weight math + the Q6_K/Q8_0/F16 mix.**
`tools/q4k_ref.py` is an independent reimplementation of ggml's `dequantize_row_q4_K` + the Q4_K
matmul contract. The RTL Q4_K primitives (`q4k.vh`: fp16→fp32 decode, `get_scale_min_k4`) and the
Q4_K GEMM core (`glm_matmul_q4k`) are **bit-exact** to it: `make q4k` → `q4k_prim` **18/18**,
`glm_matmul_q4k` **160/160**. The **mixed-type path** (Q6_K/Q8_0/F16: `src/q4k_mixed.vh` dequant
primitives, per-column `w_type` routing in `glm_matmul_q4k`, `desc_wtype` in `weight_loader_q4k`)
is bit-exact to the same reimpl golden: `make mixedtype` → `q6k_prim`, `q8_0_prim`,
`glm_matmul_mixed` **32/32**, `weight_loader_q4k_mixed` **192/192** (incl. a 24-tile mixed
sequence). This proves the *weight dequant → fp32 MAC → bf16* contract, **not** the real
GGUF/llama.cpp runtime (which uses Q8_K-quantized activations + integer dot — a different
arithmetic contract); the assembled model has its own golden, (c).

**(b) Per-unit goldens are on the generic bf16/fp32 TWINS, not the `_q4k` product.** The
`rmsnorm_unit`, `rope_interleave_unit`, `dsa_indexer`/`topk_select`, `mla_attn`, `mtp_head`,
`glm_softmax`, `sampler` TBs check each unit against an fp32/fp64 golden of the same equation —
but `mla_attn` / `glm_model` / `mtp_head` here are the **generic bf16 twins** (`src/glm_model.v`
etc., **zero** Q4_K). They verify the *structure/math*, not the assembled Q4_K numeric path.
The Q4_K operator wrappers are checked only **functionally / by invariant**: `swiglu_expert_q4k`
**240/240** (functional, self-labeled — not bit-exact), `moe_router_q4k` **40/40**
(renorm/top-K invariants — not a numeric golden).

**(c) The assembled `glm_model_q4k` end-to-end numeric golden — DONE.** `make model-q4k` runs the
full forward (embed → Lx(MLA+DSA+MoE) → final norm → LM head → argmax) against the numpy
reference `tools/glm_model_q4k_ref.py`: **ALL 1155 TESTS bit-exact** (logits+argmax+h_state).
`make model-q4k-acthw` repeats the same golden through the ACT_HW=1 serialized-activation
datapath (also 1155) — ACT_HW is a result-invariant resource knob. The caveat that **stays**: the
golden is the team's **own** numpy reimpl, **NOT** llama.cpp/GGUF — nothing yet asserts the
forward pass matches the real GGUF bytes or llama.cpp's runtime. Separately,
**spec-decode == greedy self-consistency**: `spec_decode_top` **19/19** (`make unittests`) proves
the speculative loop emits exactly what greedy would — the "greedy golden" is *itself* a
`glm_model_q4k` sharing the same weight ROMs (a lossless-speculation safety property, a
DUT-vs-DUT check). Larger loops (`spec_batched_top` / `spec_chain_top`) via `make spec-slow`.

**Numerics policy (as built — matches §3):**
1. RMSNorm Σx² in **fp32** (6144 bf16 terms overflow bf16 precision), bf16 I/O.
2. RoPE angle table in **fp32** (position 2²⁰ spans the frequency range beyond bf16).
3. GEMM: Q4_K weight dequant → fp32 products → **fp32 sequential accumulate** → bf16 RNE out.
4. **residual stream is bf16 across all layers** (a `modules_to_not_convert` boundary — *not* an
   fp32 residual; the MoE combine uses an fp32 accumulator, but the running residual is bf16).
5. router **renorm-then-scale** order (checked by `moe_router_q4k`'s invariant TB).
   All fp32 ops are `glm_fp.vh`'s IEEE `fp32_mul`/`fp32_add` (no fixed-point Q-format).

**Numeric gap closed:** MLA now applies the `1/√qk_head_dim` softmax scale (§4.1) — the earlier
omission was exposed and fixed via the assembled-model golden (`make model-q4k`).

**Quantization:** **weights Q4_K** (256-weight super-block: fp16 `d`/`dmin` + 6-bit scales +
4-bit codes, dequant per ggml); **activations and the latent KV cache are bf16** — there is no
activation quant and no INT8 cache (unlike the prior FP8 track). The RTL is **no longer
Q4_K-only**: the dynamic UD-Q4_K_XL mix keeps sensitive tensors at Q6_K/Q8_0/F16, and the RTL now
has consumers for all of them (`src/q4k_mixed.vh` dequant primitives, per-column `w_type` routing
in `glm_matmul_q4k`, `desc_wtype` in `weight_loader_q4k` — `make mixedtype`, bit-exact to the
same `tools/q4k_ref.py` reimpl golden, §6a). The chip CAN consume a real UD-Q4_K_XL checkpoint's
type mix; bit-verification against the real GGUF bytes remains **[NOT-YET]**.

**Structural / elaboration sign-off (not a sim):** the whole 2-clock Q4_K top
(`glm_q4k_system_cdc`) passes yosys `hierarchy -check` + `check -assert` (`make synth-glm`, 0
unresolved); the full 753B UD-Q4_K_XL shape (`glm_model_q4k` at DIM 6144 / L=78 / 256-expert /
VOCAB 154880) elaborates clean (`test/full_config_elab_wrap.v`, `iverilog -tnull`, type/width only —
*no stimulus, no golden, no run*).

**Verification gates:** (1) iverilog functional vs golden (`make q4k` / `make unittests` /
`make mixedtype` / `make model-q4k`); (2) yosys structural synth (`make synth-glm`); (3) routed
PnR / Fmax / LUT-DSP fit — **MEASURED**: Vivado ML 2026.1 real synth + full place&route of
`glm_q4k_system_cdc` on XCKU3P (compact config + ACT_HW=1): **141,710 LUT (87.1%)**, 99.6K FF,
**421 DSP**, 0 BRAM, hold met; routed Fmax **10.2 → 17.2 → 46.5 MHz** through three bit-exact
repipeline rounds, campaign ongoing — see `fpga/README.md` + `fpga/results/`. (The old
Gowin/nextpnr scaffold was removed, superseded by the Vivado flow.)

---

## 7. Memory & scale system (tiered + streaming + 1M paging)

**Tier 0 — on-chip SRAM:** active token activations, current-layer GEMM tiles, DSA index-set
scratch, softmax scratch, HOT weights (MLA proj, shared expert, router W_g, RMSNorm γ, the fp32
RoPE angle table), the active 8/256 routed experts, and the **2048-row DSA gather window**
(bounded regardless of S).

**Tier 1 — DDR/HBM fast tier [SYS-EST]:** hot-weight overflow + recently-used routed-expert
working set + the active latent-cache window (rung-dependent — DDR4 on the prove-it FPGA,
DDR5/HBM on the funded custom board; see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)).

**Tier 2 — NVMe / host [SYS-EST]:** full 725B cold routed-expert pool (1.45 TB bf16 /
**~435 GB Q4_K**) + long-tail latent cache (bf16) + the staged **~467 GB UD-Q4_K_XL** model image.

**Datapath (`glm_q4k_soc` / `glm_q4k_system_cdc` — DMA + async-CDC, two-clock):**
- **Expert stream** (dominant traffic): router top-8 ids → `expert_cache_pf` / `weight_loader_q4k`
  DMA gather of those experts from T1/T2 → resident expert buffer → run SwiGLU → evict.
- **Cache append:** write [c_kv(512) | k_rope(64)] per token/layer to the per-layer ring
  buffer (append-only), via `kv_cache_pager`.
- **Cache gather:** `kv_cache_pager` reads the 2048 DSA-selected rows
  (`eff_addr = cache_base + index·stride`).
- **Storage fabric:** `flash_xbar` (N-channel banked NVMe/PCIe read) → `ddr5_xbar`
  (N-channel banked DDR read) → die; `cdc_async_fifo` crosses the host/bus ↔ core clocks.

**Latent-KV 1M paging [SYS-EST]:** the indexer streams the whole ring cheaply (small dim) to
produce the index list; **attention gathers only the 2048 selected pages** (+ recent window).
So at 1M context only a 2048-row working set is resident even though the full cache is 94.2 GB.
(A designed absorb-mode could further cut cache reads by attending directly on c_kv — not
currently implemented; §4.1.)

**RTL slice memory [DERIVED]:** 8 tiny experts, S=32 latent ring, 256-entry vocab — fits one
chip, but uses the **same DMA append/gather datapath** so streaming control is exercised at
small scale.

---

## 8. RTL BUILD PLAN

### 8.1 Small-but-faithful verifiable config (keeps every operator + every ratio)

| Param | Slice | Real | Keeps intact |
|---|---|---|---|
| hidden H | 128 | 6144 | — |
| layers L | 6 (3 dense + 3 MoE) | 78 | first_k_dense_replace=3 |
| heads | 4 | 64 | — |
| qk split | rope 16 + nope 16 = 32; v 32 | 64+192=256; v256 | **nope/rope split** |
| q_lora / kv_lora | 64 / 32 | 2048 / 512 | **MLA low-rank** |
| rope_theta / interleave | 8e6 / true | 8e6 / true | **exact RoPE math** |
| DSA index_topk | 8 (S=32) + S=4 dense test | 2048 | **sparsity + dense fallback** |
| index_topk_freq / offset | 4 / 3 | 4 / 3 | **IndexShare (L3 computes, L4-5 reuse)** |
| experts / top-k / shared | 8 / top-2 / 1 | 256 / top-8 / 1 | **router top-k + shared** |
| moe_inter / scaling | 64 / 2.5 | 2048 / 2.5 | **renorm-then-scale** |
| dense inter | 256 | 12288 | **dense/MoE mode switch** |
| MTP nextn | 1 | 1 | **t+2 head** |
| vocab V | 256 | 154880 | — |
| eps / dtype | 1e-5 / Q4_K wt · bf16 act · fp32 acc · bf16 residual | same | **numerics policy** |

### 8.2 Build ORDER (dependency-driven — leaf math units first, orchestrators last)

The GEMM engine is the **Q4_K weight core `glm_matmul_q4k`** (with its bf16 sibling
`glm_matmul_pipe` for the `modules_to_not_convert` tail); `glm_softmax` replaces the legacy
`softmax_unit`. The order below is how the datapath was built up.

1. **`rmsnorm_unit.v`** ← FIRST UNIT (see §8.4). Leaf, no deps, used everywhere (5 sites).
2. **`rope_interleave_unit.v`** — fp32 angle table, adjacent-pair rotation; needed by MLA +
   DSA indexer.
3. **`topk_select.v`** — shared top-k selector (used by both DSA and router); build before
   either consumer.
4. **`swiglu_expert_q4k.v`** — wraps `glm_matmul_q4k` (Q4_K gate/up/down) + silu/mul via
   `glm_act`; dense/MoE modes. (Independent of attention; can proceed in parallel after #1.)
5. **`moe_router_q4k.v`** — Q4_K W_g GEMV (`glm_matmul_q4k`) + sigmoid + `topk_select` +
   renorm-then-×2.5.
6. **`dsa_indexer.v`** — small-dim scoring over ring + `topk_select` + dense fallback +
   IndexShare index-list cache.
7. **`mla_attn_q4k.v`** — orchestrator: Q/KV low-rank paths (Q4_K), latent RMSNorm, RoPE, cache
   append, batched shared-key K/V pass; drives `glm_matmul_q4k` + `rmsnorm_unit` +
   `rope_interleave_unit` + `dsa_indexer` + its inner QK^T/AV + `glm_softmax`.
8. **`mtp_head_q4k.v`** — own norm + small attn/FFN (Q4_K), shares LM head.
9. **`sampler.v`** — temp/top-k/top-p/softmax/multinomial (LFSR).
10. **`glm_decoder_block_q4k.v` / `glm_model_q4k.v`** — walk the layers: dense/MoE pattern,
    fresh/reuse-indexer pattern, **bf16 residual stream** (not fp32), MoE combine (fp32
    accumulator), PE_M batching + union-of-experts fetch.
11. **Model top wiring** — embed (gather) → L blocks → final `rmsnorm_unit` → LM head
    (**bf16** `glm_matmul_pipe`) → `sampler`; MTP head wired separately. Memory/streaming:
    `weight_loader_q4k`, `flash_xbar`, `ddr5_xbar`, `expert_cache_pf`, `kv_cache_pager`,
    `boot_loader`, `cdc_async_fifo`, `reset_sync` — wrapped by `glm_q4k_soc` /
    `glm_q4k_soc_ms` / `glm_q4k_system_cdc`.

### 8.3 Verification — per-unit then per-block

- The Q4_K weight core (`q4k.vh`, `glm_matmul_q4k`) is **bit-exact vs the ggml-Q4_K reference**
  `tools/q4k_ref.py` (`make q4k`); the surrounding leaves get their own iverilog TB vs an
  fp32/fp64 golden of the same equation — but the `mla_attn`/`glm_model`/`mtp_head` goldens run
  against the **generic bf16 twins** (`src/glm_model.v` …), *not* the `_q4k` product (§6b).
- yosys structural synth (`make synth-glm`); routed PnR/Fmax — **measured** on Vivado/XCKU3P (§6).
- Block bring-up: (a) attention sub-block vs the twin's golden attention; (b) FFN sub-block
  (router+experts+combine) — Q4_K units checked functionally / by invariant; (c) the **assembled
  Q4_K model is bit-exact vs its end-to-end numpy golden** (`make model-q4k`, all 1155 tests;
  §6c), plus **spec-decode == greedy self-consistency** (`spec_decode_top` 19/19).

### 8.4 The CONCRETE FIRST unit to build

**`rmsnorm_unit.v`.** Rationale:
- **Leaf, zero new dependencies** — pure datapath (fp32 Σx² reduce → rsqrt → per-channel γ
  multiply), no orchestration, builds and verifies in isolation immediately.
- **Highest reuse** — used at 5 sites (pre-attn, pre-FFN, q_lora, c_kv QK-norm, final), so it
  unblocks both the MLA path and the FFN path.
- **Locks the numerics contract** — it is the canonical place to prove the mandatory
  **fp32 reduce** (6144 bf16 terms overflow bf16) against the golden near-ULP, establishing
  the bf16-compute/fp32-reduce discipline every later unit inherits.
- **TB:** random vectors of LEN∈{128, 6144} → compare `y = x/√(mean(x²)+1e-5)·γ` against the
  fp32 golden; assert near-ULP.

### 8.5 Proven-by-build vs designed-on-paper

| Proven by the buildable slice **[BUILT]** | Designed on paper **[SYS-EST]** |
|---|---|
| Q4_K weight core **bit-exact vs ggml ref** (`glm_matmul_q4k`, `q4k_prim`) + Q6_K/Q8_0/F16 mixed path (`make mixedtype`) | 725B cold-expert residency (1.45 TB bf16 / ~435 GB Q4_K) |
| Per-unit twins vs fp32/fp64 golden (rmsnorm, rope, dsa, mla, mtp) | 94.2 GB 1M-context latent cache (bf16) |
| Q4_K operators functional/invariant (swiglu 240, router 40) | Q4_K residency of the ~467 GB UD-Q4_K_XL image |
| Assembled model **bit-exact vs own numpy golden** (`make model-q4k`, 1155) + spec==greedy (19/19) | Multi-chip HBM farm + cross-chip expert sharding |
| Structural sign-off: 753B-shape elaboration + whole-top synth; routed Vivado PnR/Fmax on XCKU3P (`fpga/`) | 512× FLOP-cap payoff at 1M context |
| PE_M batching + union-fetch (params present; bit-exact only on prior FP8 track) | Full-rate expert-stream bandwidth at 753B |

---

## 9. Perf / power note (SECONDARY)

Throughput and power are explicitly secondary to correctness, and every number here is
**[EST]/[SYS-EST]**, never measured on silicon. Qualitatively: the design is
**NVMe/PCIe-bandwidth-bound**, not compute-bound — per token only ~40B params are active, but
they weigh **~25 GB in Q4_K** (~0.6 B/param), of which the **~14 GB of routed experts** (top-8,
changing every token) must stream from NVMe/Flash each token (~170 MB Q4_K per MoE layer; the ~9
GB hot-set of attention/dense/shared can cache in DDR). So expert streaming bandwidth and batching
(amortizing each expert load across a token batch / the union-of-experts fetch) dominate
throughput; the DSA 2048-key FLOP cap keeps attention compute constant at long context, shifting
the 1M bottleneck to latent-cache bandwidth (the indexer's O(S) pass) rather than attention FLOPs.
Power follows the same story — NVMe/DDR traffic for expert + cache movement dwarfs the PE-array
energy. The tok/s is **rung-dependent** (bandwidth set by IO pins/PHYs → budget): roughly
**~5–8 tok/s on the prove-it FPGA (rung ①) → ~15–40 on the funded custom board (rung ②) → ~40+ at
volume (rung ③)** with **~9 → ~3 J/token**, all **[EST]** until a running board —
see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md), [`ULTRA_PERF.md`](ULTRA_PERF.md). A routed PPA
result **is now in-repo**: Vivado ML 2026.1 real synth + full place&route of `glm_q4k_system_cdc`
on XCKU3P (compact config + ACT_HW=1) — **141,710 LUT (87.1%)**, 99.6K FF, **421 DSP**, 0 BRAM,
hold met; routed Fmax **10.2 → 17.2 → 46.5 MHz** through three bit-exact fmax repipeline rounds
(`rope_interleave_unit` 10-stage; `glm_act` 20-stage + rmsnorm reduce/rsqrt; `glm_matmul_q4k`
dequant+MAC 5-stage), campaign ongoing — see `fpga/README.md` + `fpga/results/`. ACT_HW is a
result-invariant resource knob (`glm_act` HW_LANES serialization).

> **Prior-FP8 note.** Earlier revisions of this doc/track carried specific **FP8** measurements
> (sky130 PPA slack, LUT/DSP counts, cycle counts, byte/BOM figures). Those are **prior-FP8**
> results (branch `fp8`) for a **deleted** datapath; **no Q4_K re-run exists**. They are *not*
> reproduced here rather than relabeled as Q4_K numbers.

---

*Buildable RTL under `src/` (`glm_*_q4k` compute + memory-system modules). Q4_K GEMM golden
`tools/q4k_ref.py`; per-unit/block TBs under `test/`. The Q4_K weight core and the Q6_K/Q8_0/F16
mixed path are bit-exact vs the ggml reimpl; the assembled model is bit-exact vs its own
end-to-end numpy golden (`make model-q4k`) and also exercised as spec==greedy self-consistency.*
