# AIPU — full operational flow (end-to-end)

How the whole accelerator runs one real GLM-5.2-FP8 token, from power-up through a decoded
token, across every committed RTL block. Grounded in main @ current state (PE_M 4/4, grouped-MoE
union-skip, cycle-accurate emulation). This is the *operational* view; per-block detail lives in
[`ACCEL_GLM52.md`](ACCEL_GLM52.md) / [`SYSTEM_SINGLE_PACKAGE.md`](SYSTEM_SINGLE_PACKAGE.md).

The one fact that shapes everything: **the workload is Flash-bandwidth-bound.** The 753 GB model
lives in Flash; each token streams its active experts through a DDR5 cache into a mostly-idle FP8
die. Throughput ≈ `Flash_BW / [(1−h)·footprint] · K`.

## 0. Physical / logical stack

```
  ┌───────────────────────────── AIPU module (one board / one die) ─────────────────────────────┐
  │                                                                                              │
  │   HOST (USB-C)  ──req──►┌─CDC─┐──►┌──────────────── glm_fp8_system (compute clock) ─────────┐ │
  │                 ◄─tok───┤fifo │   │                                                         │ │
  │                         └─────┘   │   glm_model_fp8  (the FP8 compute die)                  │ │
  │   glm_fp8_system_cdc              │      embed → [decoder_block ×78 time-mux] → norm → LMhead│ │
  │   (2-clock top, reset_sync)       │            │  ▲ pull weights / KV / experts             │ │
  │                                   │            ▼  │                                         │ │
  │                                   │   weight_loader · expert_cache_pf · kv_cache_pager       │ │
  │                                   │            │  ▲                                          │ │
  │                                   │      ddr5_xbar (N-ch)     flash_xbar (N-ch, QDEPTH)      │ │
  │                                   └──────│──────────────────────────│─────────────────────┘ │
  │                                   ┌──────▼──────┐            ┌───────▼────────┐               │
  │                                   │ 64 GB DDR5  │            │  1 TB Flash    │               │
  │                                   │ (working)   │◄─boot_loader│ (753 GB model) │               │
  │                                   └─────────────┘   load     └────────────────┘               │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **`glm_fp8_system_cdc`** — the 2-clock chip top: host/USB clock ↔ compute clock via
  `cdc_async_fifo` (request in, token out), `reset_sync` per domain.
- **`glm_fp8_system`** — the compute-domain core: the die + the memory subsystem.
- Memories are TB-modeled here; real DDR5 / ONFI-Flash / USB-C PHYs are vendor IP (out of scope).

## 1. Boot — resident-set load (Flash → DDR5)

At power-up **`boot_loader`** DMAs the **resident set** — everything touched *every* token
(attention weights, the dense-FFN weights, the MoE router `W_g`, embeddings, LM-head, norm gammas)
— from Flash into the 64 GB DDR5 fast tier. Its registered **`done` gate is the single release**
for inference. The **256 routed experts stay in Flash** (753 GB ≫ 64 GB) and are demand-streamed
per token (§4). Pure DMA — no arithmetic, byte-exact.

## 2. Per-token decode — the pipeline

One decode step (`glm_model_fp8`), for a batch of **`PE_M` = B** token rows:

```
 token_id[0..B-1]
   │ ① embed lookup (bf16 table, em_* pull; SERIAL per row)          -> x0[r]  (B × MODEL_DIM)
   ▼
 ┌── for layer l = 0 .. 77  (ONE time-multiplexed glm_decoder_block_fp8, mode = dense|MoE) ──┐
 │  ② RMSNorm(x)                                                                             │
 │  ③ MLA attention  ──────────────► h = x + attn                                            │
 │       (7 FP8 projections, latent KV, DSA sparse index, interleaved RoPE, bf16 softmax)    │
 │  ④ RMSNorm(h)                                                                             │
 │  ⑤ FFN:  l<3 → dense SwiGLU  |  l≥3 → MoE (router top-8/256 → union-skip experts+shared)  │
 │       ──────────────► x_{l+1} = h + ffn                                                   │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
   │ ⑥ final RMSNorm  (PE_M rmsnorm_units, lockstep off ONE shared gamma pull)
   ▼
 ⑦ LM-head GEMV (bf16, glm_matmul_pipe over VOCAB=154880)  -> logits[r]
   │ ⑧ argmax / sample (per row)                            -> next_token[r]
   ▼
 ⑨ mtp_head_fp8: t+2 speculative draft (per row)           -> draft token (for §6)
```

Key structural facts:
- **One decoder block, time-multiplexed over 78 layers** (not 78 copies) — the per-layer weights
  are *pulled* fresh each layer; the block is a fixed amount of logic. The residual `x` streams
  layer→layer.
- **PE_M batching (4/4 wrappers):** all B rows share ONE weight-fetch stream per GEMM
  (`aw_req`/`fw_req`/… pulse identically to a single-row run) — "B rows == 1 fetch". Each row
  carries its own residual + bf16 tail (norm/RoPE/softmax/argmax) and its own dynamic-quant
  `a_shift`.
- **All weight / KV / embedding delivery is via pull ports** (`em_*`, `aw_*`, `kc_*`, `rw_*`,
  `fw_*`, `pw_*`, `lw_*`) answered by the memory subsystem — no combinational loop into the die.

### 2a. Attention (step ③) — `mla_attn_fp8`
Latent MLA: down-project to a small KV latent, cache it in the **`kv_cache_pager`** ring, gather
the sparse key set the **DSA indexer** selects (IndexShare: the index is computed by a full
indexer layer and *reused* by shared layers), apply interleaved RoPE, bf16 softmax over the top-K
window (SWIN, decoupled from the 1M position field), up-project through `W_o`. All 7 weight
projections are FP8 E4M3 block-scaled GEMMs (`glm_matmul_fp8`); scores/probs/softmax stay bf16.

### 2b. MoE FFN (step ⑤, layers ≥ 3) — `moe_router_fp8` + grouped experts
1. **Router** (`moe_router_fp8`): FP8 GEMV `x·W_g` → top-8 of 256 experts + renormalized gates
   (×2.5 shared-scale), per row.
2. **Union-skip grouped dispatch** (in `glm_decoder_block_fp8`, PE_M>1): scan the expert axis and
   evaluate **only the UNION** of experts any of the B rows selected (a combinational `any_has`
   skip — non-selected experts are **never fetched**). For each union expert, fetch its SwiGLU
   weights **once** and run `swiglu_expert_fp8` at PE_M over all B rows; each row accumulates
   `gate·expert(x)` only if it selected that expert. Byte-identical to per-row; up to ~32× fewer
   Flash expert-fetches at small B (the aggregate-throughput lever). Then the always-on **shared
   expert** (weight 1). (`batched_moe.v` is the standalone reference of this dispatch.)

## 3. Weight paths — resident vs demand-streamed

| what | where it lives | path to the die | per-token cost |
|---|---|---|---|
| attention / dense-FFN / router `W_g` / norms / embed / LM-head | **DDR5-resident** (boot-loaded) | `ddr5_xbar` → `weight_loader` → die pull | small, fixed |
| **routed experts** (the 753 GB bulk) | **Flash** | `flash_xbar` → `expert_cache_pf` (DDR5 LRU) → die | **the bottleneck** |

- **`flash_xbar`** banks reads across N Flash dies and hides NAND's ~10–100 µs latency with a
  deep per-channel outstanding queue (QDEPTH ~ FLASH_LAT, Little's law) → ~N× aggregate BW.
  Placement matters: expert→channel layout (`flash_layout.py`) + the proposed sub-expert striping
  keep all channels busy ([`FLASH_STRIPING.md`](FLASH_STRIPING.md)).
- **`expert_cache_pf`** — DDR5 routed-expert cache: LRU + frequency + confidence-thresholded
  prefetch; a demand miss stalls the die for the exposed Flash-refill (see §7). `weight_decomp`
  (optional) losslessly decompresses on the Flash→loader refill (fewer Flash bytes).
- **`weight_loader`** — turns cache/DDR5 responses (FP8 codes + [128,128] block scales) into the
  die's matmul pull stream, bit-exactly.

## 4. Batching & speculative decode (throughput layers)

- **Batching (PE_M = B):** B independent token rows decode in lockstep, sharing one weight fetch
  per GEMM. With **union-skip**, the per-token routed-expert footprint shrinks with B toward the
  union (`E[distinct]=256·(1−0.96875^B)`), realizing the aggregate-throughput regime.
- **Speculative decode (`spec_batched_top`):** the MTP head drafts K tokens; the main model
  **verifies all K+1 positions in ONE PE_M=K+1 weight-load** (Flash traffic ÷ up to K+1), and the
  committed stream is proven **== greedy** (spec==greedy safety). `spec_chain_top` chains the
  MTP steps.

## 5. Clocking / CDC
Host/USB requests cross into the compute domain through `cdc_async_fifo` (+ `reset_sync` per
domain); the decoded token crosses back out. The committed token is byte-identical across the
async clock boundary (`glm_fp8_system_cdc`, 31-test binding).

## 6. Timing & bandwidth — how fast, and why

- **The die is ~75 % idle**, gated behind Flash bandwidth; `clk_en_ctrl`/ICG clock-gates the idle
  cycles. So die-side fmax is **not** the throughput knob — Flash BW is.
- **Measured (cycle-accurate emulation, `EXPERT_STALL`):** a demand-miss delays the token by the
  exposed Flash-refill; `cyc_per_tok` grows with `FLASH_LAT` (`stall = 3·FLASH_LAT+9` at the
  slice; `cyc_per_tok` 7947→8607 @FLASH_LAT=256), the roofline *mechanism* measured on real RTL
  cycles ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md)).
- **Projected (roofline, `[EST]`):** single-user ~6–16 tok/s, ~3 J/token; batched aggregate
  ~40–85 tok/s. All `[EST]` — see [`ULTRA_PERF.md`](ULTRA_PERF.md); real silicon lands below the
  roofline (achievable-vs-peak BW, second-order walls).

## 7. Per-token bottleneck (the honest critical path)

```
  token latency ≈ compute_cycles(die, ~fixed)  +  exposed_Flash_stall(demand-miss experts)
                                                    └── the dominant term at real scale ──┘
  where exposed_Flash_stall ≈ (routed-expert misses this token) × FLASH_LAT_exposed
        routed-expert misses ≈ 75 MoE layers × ~8 experts × (1 − cache_hit_rate)   [after union-skip: only the union]
```
The whole architecture (flash_xbar QDEPTH, expert cache + prefetch, weight decomp, union-skip,
batching, striping) exists to shrink that second term.

## 8. Module → function map

| stage | module(s) | verified |
|---|---|---|
| chip top / CDC | `glm_fp8_system_cdc`, `cdc_async_fifo`, `reset_sync` | token == standalone across async clks (31) |
| system core | `glm_fp8_system` | token == standalone (3) |
| compute die | `glm_model_fp8` → `glm_decoder_block_fp8` | full FP8 fwd, next-token argmax == golden |
| attention | `mla_attn_fp8`, `dsa_indexer`, `kv_cache_pager` | ops bit-exact; real-dim rel 5.48e-4 |
| MoE | `moe_router_fp8`, `swiglu_expert_fp8`, grouped union-skip, `batched_moe` | union==per-row bit-exact (union_tb 4) |
| FP8 GEMM | `glm_matmul_fp8` (+ Ph1 fold pipeline) | exhaustive E4M3; 224; byte-identical |
| head | `mtp_head_fp8`, `sampler`, LM-head `glm_matmul_pipe` | mtp PE_M 44; spec==greedy |
| memory | `flash_xbar`, `ddr5_xbar`, `expert_cache_pf`, `weight_loader`, `boot_loader`, `weight_decomp` | BMC + k-induction; loader bit-exact |
| spec decode | `spec_batched_top`, `spec_chain_top`, `spec_decode_seq` | spec==greedy (run via `make spec-slow`) |

## 9. Honest scope
This is the flow of the **committed slice** (every operator + ratio faithful) plus the
memory/streaming system that runs the real 753B. What is *modeled* (not silicon): the PHYs, and
all `[EST]` tok/s/J. Fidelity is operator-bit-exact vs the real checkpoint + assembled-FFN faithful
(borderline-A); the full-model token-chain-vs-HF and a real FPGA run remain (see
[`REAL_CKPT_VALIDATION.md`](REAL_CKPT_VALIDATION.md), [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md)).
