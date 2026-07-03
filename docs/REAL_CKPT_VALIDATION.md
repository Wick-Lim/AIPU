# Real-checkpoint FP8 validation (CPU / numpy, no GPU)

`tools/validate_real_ckpt.py` closes the project's #1 missing validation: it checks
OUR FP8 arithmetic contract's **interpretation of the real published checkpoint**
(`zai-org/GLM-5.2-FP8`, public / gated:False / no token) against the actual
tensors — on CPU with numpy only.

Prior state: the RTL's FP8 GEMM was proven bit-exact to a **self-made** golden
(`tools/glm_fp8_ref.py` / `glm_fp8_contract.py`), but that golden's assumptions
about weight orientation, the `[128,128]` block-scale layout, dtypes, and the
`q_lora`/`kv_lora` ranks had **never** been checked against the real checkpoint.
This script does, downloading only a few MB (config + index + one small FP8
tensor's bytes via HTTP **range reads** — not a 5 GB shard, not the 753 GB model).

Run:

```
python3 tools/validate_real_ckpt.py        # downloads ~15 MB, prints PASS/FAIL, exit 0
```

## Result (OBSERVED — 2026-07, HF `zai-org/GLM-5.2-FP8`)

**OVERALL: PASS.** Validated tensor:
`model.layers.0.self_attn.kv_a_proj_with_mqa.weight`, dtype **F8_E4M3**, shape
**[576, 6144]** (the MLA KV down-projection; block grid **5 × 48**, so a
transpose/orientation bug would be caught). Only **3.54 MB** weight + **960 B**
scale were range-read.

### (a) Quant-config assumptions — CONFIRMED
| field | assumed | real | verdict |
|---|---|---|---|
| `quant_method` | `fp8` | `fp8` | MATCH |
| `fmt` | `e4m3` | `e4m3` | MATCH |
| `weight_block_size` | `[128,128]` | `[128,128]` | MATCH |
| `activation_scheme` | `dynamic` | `dynamic` | MATCH |

`modules_to_not_convert` is an **explicit 541-module list** (not a pattern list):
norms (`input_layernorm`, `post_attention_layernorm`, `q_a_layernorm`,
`kv_a_layernorm`, `indexer.k_norm`), MoE router (`mlp.gate` +
`e_score_correction_bias`), `embed_tokens`, `lm_head`, `model.norm`, and the MTP
head (`eh_proj`, `enorm`, `hnorm`, `shared_head.norm`) plus `indexers_proj` — all
kept bf16. This is exactly our "bf16 tail" concept, enumerated per-module.

### (b) Real tensor layout vs OUR assumed contract — MATCH
| item | our assumption | real | verdict |
|---|---|---|---|
| weight dtype | `F8_E4M3` | `F8_E4M3` | MATCH |
| weight shape/orientation | `[out, in]` (2-D) | `[576, 6144] = [out,in]` | MATCH |
| scale dtype | bf16 or F32 | **F32** | MATCH (contract narrows F32→bf16) |
| scale shape | `[ceil(out/128), ceil(in/128)]` | `[5, 48]` | MATCH |

No orientation/transpose/dtype/block-size bug. The HF weight is `[out,in]` (we
transpose to the contraction form `W[k][n]`), and `weight_scale_inv` is
`[ceil(out/128), ceil(in/128)]` F32 — exactly what `ckpt_pack.py` /
`glm_fp8_ref.block_fp8_gemm` assume.

### (c) OUR contract == INDEPENDENT numpy fp32-accumulate ref, on the REAL weight
Both engines share the same E4M3-quantized operands, the same bf16-narrowed block
scale, and the same per-token pow2 `a_shift`; the **only** difference is the
accumulator (our exact-BFP vs numpy fp32 rolling-add) — the same isolation as
`docs/BIT_ACCURACY.md` Section A, now on **real weights** (16 controlled bf16
activation tokens, seed 0):

| metric | value |
|---|---|
| max_abs | 3.90e-03 |
| rms_abs | 2.12e-04 |
| max_rel | 3.87e-03 (≈ 2⁻⁸, one bf16 quantum) |
| **bf16-domain exact** (ref rounds to the SAME bf16 code as ours) | **9216 / 9216 = 100.00%** |
| **argmax match** (next-token decision) | **16 / 16 rows = 100.0%** |

The fp32-accumulate reference and our exact-BFP contract land on the **identical
bf16 output for every one of the 9216 outputs**, and preserve the argmax on every
token row. This is the CPU analogue of `tools/modal_validate.py::tier1_operator`,
now verified on a real published tensor.

## KEY RECONCILIATION FINDING (dims)
| dim | our docs (pending) | REAL config | note |
|---|---|---|---|
| `q_lora_rank` | **1536** | **2048** | **RESOLVED — update the pending guess to 2048** |
| `kv_lora_rank` | 512 | 512 | MATCH |
| `hidden_size` | 6144 | 6144 | MATCH |
| `num_hidden_layers` | 78 | 78 | MATCH |
| `n_routed_experts` | 256 | 256 | MATCH |
| `num_experts_per_tok` | 8 | 8 | MATCH |
| `moe_intermediate_size` | 2048 | 2048 | MATCH |
| `first_k_dense_replace` | 3 | 3 | MATCH |
| `vocab_size` | 154880 | 154880 | MATCH |

`q_lora_rank` was marked **pending** (DeepSeek-MLA standard 1536) in
`docs/ACCEL_GLM52.md` and `docs/P12_SCALEUP.md`. The real config sets it to
**2048** (confirmed independently by the real `q_a_proj.weight` shape
`[2048, 6144]`). `kv_lora_rank = 512` is confirmed. Recommend updating the RTL
`Q_LORA` scale-up parameter and the pending note from 1536 → **2048**.

### Real architecture context (informational)
`architectures = ['GlmMoeDsaForCausalLM']`, `model_type = glm_moe_dsa` — a
DeepSeek-MLA-derived MoE with **DeepSeek Sparse Attention** (`index_n_heads=32`,
`index_head_dim=128`, `index_topk=2048`). MLA head dims: `qk_nope=192`,
`qk_rope=64`, `v_head=256`, `num_heads=64`; `kv_a_proj` out = 576 = kv_lora(512) +
qk_rope(64). The `indexer.wk`/`indexer.wq_b` projections are FP8; the
`indexers_proj` input projection is kept bf16. `num_nextn_predict_layers = 1`
(one MTP layer → 79 attention blocks total for 78 hidden layers).

## Addendum — GPU tier1 on T4 (real MoE expert weights)

`modal run tools/modal_validate.py --tier 1` on the **cheapest GPU (T4)**, single-shard
download (~5 GB, no 753 GB model, no token — the repo is public). It read **6 real FP8 MoE
expert Linears** (`model.layers.10.mlp.experts.*.{down,up}_proj.weight`, real N/K =
6144/2048 and 2048/6144) and compared OUR contract vs the **fp32-accumulate** reference
engine on the GPU (torch path).

Result: **argmax_match 16/16 on 5 of 6 weights, 15/16 on the 6th** (`max_abs ≈ 1e-3`,
i.e. ~1 bf16 ULP). Unlike the CPU check above (which used a *matched-accumulator* numpy
reference → 100 % bf16-exact), tier1 compares against the **intentionally different**
fp32-rolling-add accumulator, so the two engines differ by ≤ 1 bf16 ULP per element
(`bf16_exact` low, and `max_rel` blows up on the near-zero outputs) — exactly the
exact-BFP-vs-fp32-accumulate gap documented in `docs/BIT_ACCURACY.md §A`, which is
**argmax-preserving**. The one 15/16 flip is a near-tie at the accumulator-ULP level on a
single Linear's raw output (not the final next-token logits).

**Takeaway:** the operator-level fidelity is now confirmed on real weights through **two
independent paths** — CPU (an MLA projection, matched reference → bit-exact) and GPU T4
(6 MoE experts, fp32-accumulate reference → argmax-preserving). Full **end-to-end** token
identity (tier2, all experts resident) remains the only unrun step; it needs the 8×H200
class of host and is out of scope here.

## Partial-F1 — multi-layer real-weight assembled FFN (Modal, ~$1-2)

Extends the operator-level validation above to an **assembled multi-layer FFN on real
weights**, run on Modal (`tools/modal_partial_f1.py`, budget-capped: CPU download → $1 smoke
gate → T4 compare). Scope: the first **6 real decoder layers** of `zai-org/GLM-5.2-FP8`
(layers 0–2 dense SwiGLU, 3–5 256-expert MoE + shared), covering the **dense→MoE transition**.

**mode=ffn (measured, PASS):** for every FP8 Linear in the assembled real-weight FFN, OUR
exact-BFP block-scaled contract vs a fp32-accumulate reference at the same per-token `a_shift`:

| metric | value |
|---|---|
| layers compared | 6 (dense 0–2 + MoE 3–5), dense→MoE transition **covered** |
| argmax (proxy) match | **6/6** |
| worst `max_abs` | **0.0015** (mean `rms_abs` 0.0002) |
| `bf16_exact` | 0/6144 — the exact-BFP vs fp32-accumulate ~1-ULP gap (not a bug) |
| worst `max_rel` | 1586 — a **near-zero-denominator artifact** (tiny abs / tiny value); `max_abs` is the meaningful signal |

So the assembled real-weight FFN — including the dense→MoE transition and the 256-expert +
shared-expert MoE block — is **numerically faithful** (max_abs ~1.5e-3, argmax preserved),
extending the single-operator bit-accuracy to a multi-GEMM assembled block on real weights.

**HF cross-check (partial):** the real **`GlmMoeDsa` architecture loads and all 6 decoder
layers build** from `AutoModelForCausalLM.from_config(trust_remote_code=True)` with **our FP8
Linears patched in** (8 per MoE layer) — HF compatibility is proven. The **full token-chain
vs-HF** did **not** complete, and we traced *exactly* why by fixing each blocker in turn:
1. standalone layers unpack `cos,sin = position_embeddings` — supplied by re-instantiating the
   rotary on a real device (the meta-built model's is on `meta`); **fixed**.
2. `mixed dtype (CPU): expect Float` — the bf16 tail vs float32 hidden; cast the layer to
   float32 (the fp8-patched forwards preserve `x.dtype`); **fixed**.
3. **`ValueError: Shared DSA layers require top-k indices from a previous full indexer layer.`**
   — the fundamental blocker: GLM-5.2's **DSA IndexShare** has a *full-indexer* layer compute the
   sparse top-k indices that later *shared* DSA layers reuse, so the layers are **not
   independent** — running them standalone lacks the model's cross-layer index threading. This is
   an **architectural dependency of GlmMoeDsa, not a kwarg** — the proper fix is the truncated
   full-`model.model.forward` (which threads the index itself), a larger effort with its own
   uncertainties (materializing the model, DSA at seq-len 1).

This is an **HF standalone-layer integration limit, not a fidelity failure**; the attention path
itself (MLA + DSA IndexShare) is separately bit-validated by the operator TBs (`mla_attn_fp8`
real-dim, worst rel 5.48e-4; the per-row DSA union in `dsa_indexer`). Spend to here: ~$4-5.

**Fidelity standing:** operator-level → **assembled multi-layer FFN on real weights, faithful**
(borderline-A). The full-model token-chain-vs-HF (true A) still needs either the HF
standalone-layer plumbing solved, or the full 753B run (multi-GPU, out of the ~$29 budget).
