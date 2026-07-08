# Q4_K_XL retarget — local-device numerics (FP8 → GGUF k-quants)

*The accelerator's local-device target: the published **`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`**
(467 GB, ~38% smaller than the 753 GB FP8 checkpoint).
The FP8 datacenter-native baseline is preserved on branch **`fp8`** + tag
`fp8-verified-baseline`; `main` develops the Q4_K local-device track.*

## Why
FP8 E4M3 is the **datacenter-native** format (runs natively on H100-class silicon). GGUF
k-quants (Q4_K etc.) are the **local-inference-native** format — what llama.cpp and every
local device actually run. For a cost-constrained **local appliance**, Q4_K is the coherent
target: ~half the memory footprint (the BOM is memory-dominated). The verifiable claim is
scoped to our own ggml reimpl: the compute core is **bit-exact to `tools/q4k_ref.py`**
(self-referential — **NOT** the real GGUF bytes / llama.cpp; that whole-file check is OPEN,
because the RTL uses bf16 activations + fp32 accumulate whereas llama.cpp uses Q8_K-quantized
activations + an integer dot — a different arithmetic contract). The moat is **offline +
full-frontier + turnkey per-seat**, not GGUF bit-exactness.

## The numerics (leaf GEMM core + dequant — bit-exact to the ggml reference `q4k_ref.py`)
*Scope: the **leaf** Q4_K dequant + GEMM core is bit-exact to our own ggml reimpl
`tools/q4k_ref.py`. An end-to-end numeric golden for the **assembled** `glm_model_q4k`
DOES NOT EXIST (NOT-YET; the assembled path is exercised only inside the spec loops,
DUT-vs-DUT — see README's NOT-YET rows).*

UD-Q4_K_XL is a **dynamic mix**: most tensors Q4_K, sensitive ones kept at higher precision
(Q6_K / Q8_0 / F16). In the *reference* every type dequantizes exactly per ggml; **in RTL the
datapath is Q4_K-only today** (Q6_K/Q8_0/F16 have Python-only goldens in `q4k_ref.py` with
**no RTL consumer** — the mixed-type path is NOT-YET; see `Q4K_SYSTEM_PLAN.md §2.5`).

| Type | Block | Dequant | Golden | RTL consumer |
|---|---|---|---|---|
| **Q4_K** | 256 wt / 144 B: fp16 d,dmin + 12B 6-bit scales/mins + 128B 4-bit | `w=(d·sc)·q−(dmin·m)` | `q4k_ref.py` (ggml) | `q4k.vh` **18/18** + `glm_matmul_q4k.v` **160/160** (bit-exact vs ggml Q4_K) |
| **Q6_K** | 256 wt / 210 B: fp16 d + ql/qh (6-bit signed) + int8 scales[16] | `w=d·sc·(q−32)` | `q4k_ref.py` (Python only) | *none — Q4_K-only datapath (NOT-YET)* |
| **Q8_0** | 32 wt: fp16 d + 32 int8 | `w=d·q` | `q4k_ref.py` (Python only) | *none — Q4_K-only datapath (NOT-YET)* |
| **F16** | passthrough | `w=fp16→fp32` | (exact) | `fp16_to_fp32` primitive only (no F16-tensor matmul path) |

**GEMM contract** (`glm_matmul_q4k`, bit-exact to `tools/q4k_ref.py:matmul_q4k_col`):
`out = bf16( Σ_k fp32(a_k) · w_deq_k )` — bf16 activations, per-weight ggml dequant, the
proven fp32 sequential accumulate (same as `glm_matmul_pipe`, weight source swapped), bf16
RNE output. All fp32 ops are `glm_fp.vh`'s IEEE `fp32_mul`/`fp32_add` (confirmed == numpy
fp32 through the full MAC).

## Files (on `main`)
- `tools/q4k_ref.py` — bit-exact ggml dequant golden (Q4_K/Q6_K/Q8_0) + fp32-MAC contract.
- `tools/q4k_matmul_gen.py` — random-tile + golden-output vector generator for the RTL TB.
- `src/q4k.vh` — Q4_K primitives (exact IEEE fp16→fp32, `get_scale_min_k4`, int→fp32).
- `src/glm_matmul_q4k.v` — the Q4_K-native GEMM core (drop-in sibling of the now-removed FP8 `glm_matmul_fp8`, preserved on branch `fp8`).
- `test/q4k_prim_tb.v`, `test/glm_matmul_q4k_tb.v` — the verification gates.

## Progress
- ✅ **Numerics** (Q4_K dequant + `q4k.vh` primitives) — **bit-exact to the ggml reference**
  (`q4k_prim` **18/18**). Q6_K/Q8_0 have Python-only goldens in `q4k_ref.py` (no RTL consumer).
- ✅ **`glm_matmul_q4k`** — arbitrary-K, multi-super-block GEMM core, **160/160 bit-exact vs ggml Q4_K**.
- ✅ **`swiglu_expert_q4k`** — first datapath operator (MoE expert: gate/up/down + silu + merge)
  on Q4_K, **240/240** functional vs golden. **Proves the retarget pattern end-to-end.**

**The pattern** (each `*_fp8` operator → `*_q4k`): keep the FSM; drop the FP8 activation-shift
machinery (`glm_matmul_q4k` takes bf16 acts direct); swap the weight interface from FP8
(8-bit codes + bf16 block scales) to Q4_K (4-bit codes + per-pass super-block d/dmin/scales);
drive `glm_matmul_q4k`; verify against a Q4_K golden (bit-exact GEMMs + shared silu/sigmoid/
topk tail). swiglu_expert_q4k is the worked reference for the rest.

### Leaf-operator layer — COMPLETE (the compute datapath on Q4_K)
All three GLM compute operators now run on the Q4_K core:
- ✅ `swiglu_expert_q4k` — SwiGLU FFN expert — **240/240** functional.
- ✅ `moe_router_q4k` — MoE gating (GEMV→sigmoid→top-K→renorm) — **40/40** (renorm invariant).
- ✅ `mla_attn_q4k` — MLA attention (7 proj + RoPE + softmax + DSA) — elaborates clean;
  the single FP8 engine → `glm_matmul_q4k`, a_shift removed, bf16 score engine unchanged.
  (Full RoPE/softmax/DSA functional golden is a separate large effort; the Q4_K numerics
  are bit-exact-proven at `glm_matmul_q4k`.)

## Remaining phases (system + cleanup — rewiring, not novel)

*(Orchestration is done: `glm_decoder_block_q4k`, `glm_model_q4k`, `mtp_head_q4k`
**already exist and pass** — they elaborate clean and run inside the spec loops as
DUT-vs-DUT self-consistency, per `Q4K_SYSTEM_PLAN.md §0`. The remaining OPEN item at this
layer is an **end-to-end numeric golden for the assembled `glm_model_q4k` vs `q4k_ref.py`,
which DOES NOT EXIST** — see README's NOT-YET rows. The phases below are the non-orchestration
work.)*

1. **Weight path** — `weight_loader` / memory image / `expert_cache` / `ddr5_xbar` sizing move
   from the FP8 [128,128] block layout to the GGUF super-block layout (~half the bytes); the
   provisioning packer reads the real GGUF (per-tensor type map = the dynamic mix).
2. **Per-tensor type routing** — select Q4_K/Q6_K/Q8_0/F16 per tensor from the GGUF header
   (NOT-YET — the RTL datapath is Q4_K-only today).
3. **Remove FP8** from `main` (preserved on branch `fp8`); update the Makefile gate.
4. **Docs/site** — footprint/BOM/tok/s and the §03 moat row to the GGUF basis.

*The 467 GB GGUF can't be downloaded on the dev host (disk), so per-tensor type verification
against the real file happens when a box with disk is available; the dequant math is proven
against the ggml spec + goldens now (same methodology as the FP8 track).*
