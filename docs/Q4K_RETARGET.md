# Q4_K_XL retarget — local-device numerics (FP8 → GGUF k-quants)

*The accelerator's local-device target: the published **`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`**
(467 GB, ~38% smaller than the 753 GB FP8 checkpoint, "generally lossless" per Unsloth).
The FP8 datacenter-native baseline is preserved on branch **`fp8`** + tag
`fp8-verified-baseline`; `main` develops the Q4_K local-device track.*

## Why
FP8 E4M3 is the **datacenter-native** format (runs natively on H100-class silicon). GGUF
k-quants (Q4_K etc.) are the **local-inference-native** format — what llama.cpp and every
local device actually run. For a cost-constrained **local appliance**, Q4_K is the coherent
target: ~half the memory footprint (the BOM is memory-dominated), and the moat moves —
still verifiable — from *"bit-exact to the published FP8 safetensors"* to **"bit-exact to
the published UD-Q4_K_XL GGUF"** (a file anyone can download and check).

## The numerics (COMPLETE + verified, bit-exact to ggml)
UD-Q4_K_XL is a **dynamic mix**: most tensors Q4_K, sensitive ones kept at higher precision
(Q6_K / Q8_0 / F16). Every type dequantizes exactly per ggml, then the SAME GEMM contract
runs (dequant → fp32 MAC), so one datapath carries the whole mix.

| Type | Block | Dequant | Golden | RTL |
|---|---|---|---|---|
| **Q4_K** | 256 wt / 144 B: fp16 d,dmin + 12B 6-bit scales/mins + 128B 4-bit | `w=(d·sc)·q−(dmin·m)` | `q4k_ref` 1600/1600 | `q4k.vh` + `glm_matmul_q4k.v` **800/800** |
| **Q6_K** | 256 wt / 210 B: fp16 d + ql/qh (6-bit signed) + int8 scales[16] | `w=d·sc·(q−32)` | 2100/2100 | primitives (q4k.vh fp16/int→fp32) |
| **Q8_0** | 32 wt: fp16 d + 32 int8 | `w=d·q` | 2100/2100 | primitives |
| **F16** | passthrough | `w=fp16→fp32` | (exact) | `fp16_to_fp32` (q4k.vh) |

**GEMM contract** (`glm_matmul_q4k`, bit-exact to `tools/q4k_ref.py:matmul_q4k_col`):
`out = bf16( Σ_k fp32(a_k) · w_deq_k )` — bf16 activations, per-weight ggml dequant, the
proven fp32 sequential accumulate (same as `glm_matmul_pipe`, weight source swapped), bf16
RNE output. All fp32 ops are `glm_fp.vh`'s IEEE `fp32_mul`/`fp32_add` (confirmed == numpy
fp32 through the full MAC).

## Files (on `main`)
- `tools/q4k_ref.py` — bit-exact ggml dequant golden (Q4_K/Q6_K/Q8_0) + fp32-MAC contract.
- `tools/q4k_matmul_gen.py` — random-tile + golden-output vector generator for the RTL TB.
- `src/q4k.vh` — Q4_K primitives (exact IEEE fp16→fp32, `get_scale_min_k4`, int→fp32).
- `src/glm_matmul_q4k.v` — the Q4_K-native GEMM core (drop-in sibling of `glm_matmul_fp8`).
- `test/q4k_prim_tb.v`, `test/glm_matmul_q4k_tb.v` — the verification gates.

## Remaining phases (integration — the large lift)
The numerics core is done + verified; making the whole accelerator *run* on Q4_K is the
integration, comparable in size to the original FP8 datapath build:

1. **Datapath rewire** — the `glm_*_fp8` modules (decoder_block, mla_attn, moe_router,
   swiglu_expert, glm_model, mtp_head, soc/system tops) instantiate `glm_matmul_fp8`; retarget
   them to `glm_matmul_q4k` and its Q4_K weight interface (4-bit codes + fp16 d/dmin + 6-bit
   scales super-block), each verified against a Q4_K golden.
2. **Weight path** — `weight_loader` / memory image / `expert_cache` / `ddr5_xbar` sizing move
   from the FP8 [128,128] block layout to the GGUF super-block layout (~half the bytes); the
   provisioning packer reads the real GGUF (per-tensor type map = the dynamic mix).
3. **Per-tensor type routing** — select Q4_K/Q6_K/Q8_0/F16 per tensor from the GGUF header.
4. **Remove FP8** from `main` (preserved on branch `fp8`); update the Makefile gate.
5. **Docs/site** — footprint/BOM/tok/s and the §03 moat row to the GGUF basis.

*The 467 GB GGUF can't be downloaded on the dev host (disk), so per-tensor type verification
against the real file happens when a box with disk is available; the dequant math is proven
against the ggml spec + goldens now (same methodology as the FP8 track).*
