# AIPU — full operational flow (end-to-end)

> **Prior FP8 track.** This doc describes the FP8 datapath, now the *prior* track (preserved on
> branch `fp8`). The current product track is **Q4_K** — see [`Q4K_RETARGET.md`](Q4K_RETARGET.md) /
> [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md). RTL/test names below of the form `*_fp8` map to their
> `*_q4k` equivalents on main.

> **⚠️ TRACK NOTE (2026-07-08). The current / `main` track is Q4_K-native** (GGML Q4_K,
> targeting `unsloth/GLM-5.2-GGUF:UD-Q4_K_XL`). **FP8 is the PRIOR / PRESERVED track** on
> branch **`fp8`** (tag `fp8-verified-baseline`), removed from `main` in commit `cbef69d`.
> This walkthrough still names the **FP8** blocks (`glm_fp8_system_cdc`, `glm_fp8_system`,
> `glm_model_fp8`, `glm_decoder_block_fp8`, `moe_router_fp8`, `swiglu_expert_fp8`,
> `glm_fp8_soc_ms`) and a **753 GB FP8** model on NVMe. The **Q4_K equivalents** that exist on
> `main` are `glm_q4k_system_cdc`, `glm_q4k_system`, `glm_model_q4k`, `glm_decoder_block_q4k`,
> `moe_router_q4k`, `swiglu_expert_q4k`, `glm_q4k_soc_ms`, `weight_loader_q4k.v`; the resident
> model is the ~Q4_K GGUF, not FP8. Any "== golden / bit-exact" the body attributes to the
> assembled model is **DUT-vs-DUT spec==greedy self-consistency**, not a numeric golden vs
> ggml/llama.cpp (only `glm_matmul_q4k` is bit-exact, and only vs the team's own
> `tools/q4k_ref.py`). A deeper Q4_K rewrite of this doc is deferred.

How the whole accelerator runs one real GLM-5.2-FP8 token, from power-up through a decoded
token, across every committed RTL block. Grounded in main @ current state (PE_M 4/4, grouped-MoE
union-skip, cycle-accurate emulation) — **main develops exactly the GLM-5.2-FP8 accelerator at
rung-① (the offline FPGA prove-it demo); the full product at rungs ②③ is roadmap, not code in main
now** (see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). This is the *operational* view; per-block detail lives in
[`ACCEL_GLM52.md`](ACCEL_GLM52.md) / [`SYSTEM_SINGLE_PACKAGE.md`](SYSTEM_SINGLE_PACKAGE.md).

The one fact that shapes everything: **the workload is NVMe/PCIe-bandwidth-bound.** The 753 GB model
lives on the NVMe SSD; each token streams its active experts through a DDR5 cache into a mostly-idle FP8
die. Throughput ≈ `NVMe_BW / [(1−h)·footprint] · K`.

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
  │                                   │ 64 GB DDR5  │            │  NVMe (1-4TB)  │               │
  │                                   │ (working)   │◄─boot_loader│ (753 GB model) │               │
  │                                   └─────────────┘   load     └────────────────┘               │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **`glm_fp8_system_cdc`** — the 2-clock chip top: host/USB clock ↔ compute clock via
  `cdc_async_fifo` (request in, token out), `reset_sync` per domain.
- **`glm_fp8_system`** — the compute-domain core: the die + the memory subsystem.
- Memories are TB-modeled here; real DDR5 / NVMe (PCIe) / USB-C PHYs are vendor IP (out of scope).
- **DDR is rung-dependent** — the diagram's `64 GB DDR5` is the *funded* rung-② point, **not THE spec**:
  the near-term prove-it FPGA runs DDR4 (~4 ch, ~100 GB/s), the funded custom board runs DDR5 multi-ch /
  HBM (~300–600 GB/s); see the hardware ladder ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). NVMe (the
  753 GB model store) is the same on every rung — performance is set by memory bandwidth, i.e. by which
  silicon the budget buys.

## 1. Boot — execution conditions & resident-set load (NVMe → DDR5)

**Inference is NOT released by power-on** — it is released by `boot_loader.done`. The full power-on
→ ready sequence (its *execution conditions*, in order):

| # | condition | what | who |
|---|---|---|---|
| 1 | **power** | all rails up | board |
| 2 | **clocks stable** | `host_clk` (USB), `core_clk` (compute), memory clk — PLLs locked | board / vendor IP |
| 3 | **reset sequenced** | `host_rst` / `core_rst` (per-domain, sync active-high) cleanly de-asserted (`reset_sync`) | RTL |
| 4 | **memory PHY init** | DDR5 training + NVMe/PCIe controller init | vendor IP |
| 5 | **model present on NVMe** | the 753 GB FP8 model **pre-written** (one-time provisioning, `ckpt_pack.py` / `flash_layout.py`) | manufacturing / setup |
| 6 | 🔑 **`boot_loader.done`** | DMA the **~28 GB resident set** (all-layer attention, dense-FFN, MoE router `W_g`, shared expert, embeddings, LM-head, norm gammas) **NVMe → DDR5** — its registered `done` is the **single gate that releases inference** | RTL (`boot_loader`, 9240 tests, BMC-proven) |
| 7 | **USB enumerated** | host driver loaded, endpoint open | host + vendor USB IP |

The **256 routed experts stay on the NVMe SSD** (753 GB ≫ the DDR working set) and are demand-streamed per token
(§4). Boot 6 is pure DMA — no arithmetic, byte-exact.

**Timing (one boot, [EST]):** PLL lock (~ms) + DDR5 training (~10–100 ms) + resident load (~28 GB /
NVMe read BW — ~4–8 s on one Gen3/4 ×4 drive at ~3.5–7 GB/s, dropping toward ~1 s only with several
NVMe striped across more PCIe lanes) + USB enum (~ms) ≈ **~1–2 s (multi-NVMe array) to a few seconds
(single drive) power-on → ready**. Short boot, not instant-on.

**Three timescales:** ① *one-time provisioning* — write the 753 GB model to the NVMe SSD. ② *every
power-on* — conditions 1–7 (~1–2 s). ③ *per token* — §2 (demand-stream experts from NVMe; KV
lives in DDR5, per session).

**Host interface (what USB-C carries — all on `host_clk`):** in = `start` (pulse), `prompt_tok`
(token ID), `start_pos`, `s_len`; out = `next_tok` (token ID), `tok_valid`, `busy`, `done`. Just
token IDs + position/length — the heavy weight/KV traffic never crosses USB-C (it is all inside
`glm_fp8_system` on `core_clk`).

> **Real-hardware note.** The RTL takes clocks/resets as ports and treats DDR5 / NVMe (PCIe) / USB PHYs as
> vendor IP (conditions 2, 4, 7 are the board/vendor bring-up — device-plan Phase D1,
> [`USBC_PRODUCT_PLAN.md`](USBC_PRODUCT_PLAN.md)). The RTL's own execution condition is the
> `boot_loader.done` gate (6) once those are up.

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
   `gate·expert(x)` only if it selected that expert. (Inside the expert, gate/up/down share **one**
   FP8 GEMM engine — §10a.) Byte-identical to per-row; up to ~32× fewer
   NVMe expert-fetches at small B (the aggregate-throughput lever). Then the always-on **shared
   expert** (weight 1). (`batched_moe.v` is the standalone reference of this dispatch.)

## 3. Weight paths — resident vs demand-streamed

| what | where it lives | path to the die | per-token cost |
|---|---|---|---|
| attention / dense-FFN / router `W_g` / norms / embed / LM-head | **DDR5-resident** (boot-loaded) | `ddr5_xbar` → `weight_loader` → die pull | small, fixed |
| **routed experts** (the 753 GB bulk) | **NVMe SSD** | `flash_xbar` → `expert_cache_pf` (DDR5 LRU) → die | **the bottleneck** |

- **`flash_xbar`** is the storage-read fabric (a committed RTL identifier, kept as-is); in the product it
  **fronts the NVMe/PCIe backend** — a labeled placeholder, since the crossbar's read-request /
  latency-hiding abstraction (address → weight bytes) is medium-agnostic, so the NAND-specific backend is
  swapped for an NVMe/PCIe host controller. It banks reads across N channels (**PCIe lanes / multiple NVMe
  drives**) and hides storage-read latency (~10–100 µs) with a deep per-channel outstanding queue
  (QDEPTH ~ FLASH_LAT, Little's law) → ~N× aggregate BW. Placement matters: expert→channel layout
  (`flash_layout.py`) + the proposed sub-expert striping keep all channels busy
  ([`FLASH_STRIPING.md`](FLASH_STRIPING.md)).
- **`expert_cache_pf`** — DDR5 routed-expert cache: LRU + frequency + confidence-thresholded
  prefetch; a demand miss stalls the die for the exposed NVMe-refill (see §7). `weight_decomp`
  (optional) losslessly decompresses on the NVMe→loader refill (fewer NVMe bytes = more effective NVMe
  bandwidth + less read energy).
- **`weight_loader`** — turns cache/DDR5 responses (FP8 codes + [128,128] block scales) into the
  die's matmul pull stream, bit-exactly.

## 4. Batching & speculative decode (throughput layers)

> **Scope.** The product — a local, single-user box — runs at **B=1**; **speculative decode** (last
> bullet) is its single-user tok/s lever. The **batching / multi-sequence / continuous-batching**
> layers below, and the **aggregate-throughput regime** they realize, are the **non-target
> datacenter deployment** of the same silicon (§6) — kept as analysis, not the product.

- **Batching (PE_M = B):** B independent token rows decode in lockstep, sharing one weight fetch
  per GEMM. With **union-skip**, the per-token routed-expert footprint shrinks with B toward the
  union (`E[distinct]=256·(1−0.96875^B)`), realizing the aggregate-throughput regime.
- **Multi-sequence batching (`PER_ROW_SEQ`):** the B rows need not be one prompt — each PE_M row
  can be a **different sequence**, attending its OWN sequence's KV window (`kc_seq` routes each KV
  fetch to that sequence's `kv_cache_pager` window / `kv_mem` slot) while the query-side weight
  fetch stays SHARED across rows (the batching-bandwidth win, ~41–52 % fewer attn-weight beats than
  B separate runs). Per-row argmax/logits proven bit-exact vs per-seq PE_M=1, dense + sparse;
  byte-identical at `PER_ROW_SEQ=0`. `glm_fp8_soc_ms` is the batched multi-seq SoC top
  (`glm_model_fp8` at PE_M=B + `NSEQ`-window pager + a REAL per-layer KV store `kv_mem` owned by
  the top + host FSM: prefill B seqs → 1 forward → commit B tokens).
- **Continuous-batching decode loop (`glm_fp8_soc_ms`, `N_STEPS>1`):** one host `start` decodes
  **N tokens per sequence** — a `RUN→DECAP→RUN` loop that runs one PE_M=B forward, streams the B
  argmax out (`tok_valid`), writes each decode token's latent into `kv_mem` at the growing position
  (`s_len + dec_step`) for every layer, feeds the argmax back as the next step's input, and advances
  position/extent. Each row's step-k token is bit-exact vs a standalone PE_M=1 model decoding that
  sequence alone N steps; `N_STEPS=1` is byte-identical to the single-step top.
- **Speculative decode (`spec_batched_top`):** the MTP head drafts K tokens; the main model
  **verifies all K+1 positions in ONE PE_M=K+1 weight-load** (NVMe traffic ÷ up to K+1), and the
  committed stream is proven **== greedy** (spec==greedy safety). `spec_chain_top` chains the
  MTP steps.

## 5. Clocking / CDC
Host/USB requests cross into the compute domain through `cdc_async_fifo` (+ `reset_sync` per
domain); the decoded token crosses back out. The committed token is byte-identical across the
async clock boundary (`glm_fp8_system_cdc`, 31-test binding).

## 6. Timing & bandwidth — how fast, and why

- **The die is ~75 % idle**, gated behind NVMe/PCIe bandwidth; `clk_en_ctrl`/ICG clock-gates the idle
  cycles. So die-side fmax is **not** the throughput knob — NVMe/PCIe BW is.
- **Measured (cycle-accurate emulation, `EXPERT_STALL`):** a demand-miss delays the token by the
  exposed NVMe-refill; `cyc_per_tok` grows with `FLASH_LAT` (`stall = 3·FLASH_LAT+9` at the
  slice; `cyc_per_tok` 7947→8607 @FLASH_LAT=256), the roofline *mechanism* measured on real RTL
  cycles ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md)).
- **Projected (roofline, `[EST]`) — staged to the hardware ladder ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)):**
  single-user tok/s is set by memory bandwidth, which is set by the silicon the budget buys, so it is
  **rung-dependent** — **~5–8 tok/s on the near-term prove-it FPGA** (DDR4 ~4 ch, the buildable demo
  *now*), **~15–40 tok/s on the funded custom board** (DDR5 multi-ch / HBM, rung ②), **~40+ tok/s at
  ASIC volume** (rung ③, custom silicon w/ HBM stacks + near-memory compute — lower $/seat + power once
  the NRE amortizes over volume). All `[EST]`, ~3 J/token; the old flat "~25–40" is the **funded rung-②**
  number, not the cheap near-term box. **This is the product: a fully offline / air-gapped local box** that
  runs the whole 753B frontier model with the ethernet unplugged. The capability it unlocks is frontier AI where the cloud can't
  reach — SCIFs, isolated OT / critical-infra, field/edge, or anywhere a vendor connection is itself the
  liability; the proof is binary — **nothing leaves because there's no path out** (the host link carries
  only token IDs + position, §1), so it passes the unplugged-ethernet test that every cloud option fails,
  "secured cloud" included (in-VPC / zero-retention / TEE all need connectivity). Offline alone is
  table-stakes (a 70B laptop is offline too); the moat is the **combination** — offline + full-frontier
  (753B) + appliance price. Honest caveat: the 753 GB model is written to the NVMe SSD once (one-time
  provisioning, §1) and model updates are physical re-provisioning. The *batched aggregate* ~40–85 tok/s
  figure is a **non-target datacenter regime** (per-user floors at ~0.14 tok/s there), kept only as
  analysis. All `[EST]` — see [`ULTRA_PERF.md`](ULTRA_PERF.md); real silicon lands below the roofline
  (achievable-vs-peak BW, second-order walls).

## 7. Per-token bottleneck (the honest critical path)

```
  token latency ≈ compute_cycles(die, ~fixed)  +  exposed_NVMe_stall(demand-miss experts)
                                                    └── the dominant term at real scale ─┘
  where exposed_NVMe_stall ≈ (routed-expert misses this token) × FLASH_LAT_exposed
        routed-expert misses ≈ 75 MoE layers × ~8 experts × (1 − cache_hit_rate)   [after union-skip: only the union]
```
The whole architecture (flash_xbar QDEPTH, expert cache + prefetch, weight decomp, union-skip,
batching, striping) exists to shrink that second term.

## 8. Module → function map

| stage | module(s) | verified |
|---|---|---|
| chip top / CDC | `glm_fp8_system_cdc`, `cdc_async_fifo`, `reset_sync` | token == standalone across async clks (31) |
| system core | `glm_fp8_system` | token == standalone (3) |
| batched multi-seq SoC | `glm_fp8_soc_ms` (PE_M=B model + `NSEQ` pager + `kv_mem` + host FSM; `N_STEPS` decode loop) | per-row token == per-seq PE_M=1 (dense/sparse); decode-loop step-k bit-exact |
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
all `[EST]` tok/s/J. Fidelity is operator-bit-exact vs the real checkpoint + a **truncated full-model
token chain on real weights (dense→MoE seam, real 256-expert route) argmax-identical, DSA threaded
(A-ish)** — the DSA-IndexShare + fused-expert blockers retired; deeper depth / a real FPGA run remain
(see [`REAL_CKPT_VALIDATION.md`](REAL_CKPT_VALIDATION.md), [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md)).

## 10. Compact config (FPGA miniaturization)

Target: fit the chip on a **smaller FPGA** (Tang Mega 138K / GW5AT-138). Five parameters of
`glm_fp8_system` (and its 2-clock top `glm_fp8_system_cdc`) set **capacity / parallelism /
bandwidth — never the math**. Shrinking them makes the elaborated logic smaller (fewer LUT/FF in
the matmul PE array, the DDR5 crossbar, the KV ring, the expert-request FIFO, and the expert
cache) and slower / lower-BW, but the **decoded token is byte-identical**. They are *result-invariant*.

**Compact synth config** (overrides on the `glm_fp8_system_cdc` full-model defaults):

| param | default | compact | true safe-min¹ | constraint | what it sizes | shrink / cost when reduced |
|---|---|---|---|---|---|---|
| `PE_N`        | 4  | **2** | 1              | —                     | matmul PE-array columns          | halves the PE array (**biggest die saving**); tiles more → same output, more cycles |
| `DDR_NCH`     | 4  | **2** | 2              | power-of-two          | DDR5 fabric channels (`ddr5_xbar`)| smaller crossbar; ~½ aggregate read BW (NVMe/PCIe-bound anyway) |
| `KV_RESIDENT` | 16 | **8** | `S_MAX` (=8)²  | POW2, `>= S_MAX`      | latent-KV ring capacity          | smaller ring RAM; more cold-row NVMe gathers |
| `EFIFO_DEPTH` | 16 | **8** | 2 (1 passed³)  | power-of-two          | routed-expert request FIFO depth | smaller FIFO; risk of drop only under bursty routing |
| `CACHE_SLOTS` | 4  | **2** | 1              | —                     | GDDR6 expert-cache slots         | smaller tag/data array; more misses → more NVMe stalls |

¹ Verified in the system TB slice (`test/glm_fp8_system_tb.v`, `S_MAX=4`): every value listed
still prints `ALL 3 TESTS PASSED` with the **same token stream** as the committed config. The
compact column is the recommended FPGA set (keeps head-room over the true minimum).
² `KV_RESIDENT >= S_MAX` and POW2. Slice `S_MAX=4` → min 4; the **full model `S_MAX=8`** → min 8,
so the compact synth uses 8. ³ `EFIFO_DEPTH=1` also passed the slice (no FIFO overflow observed)
but leaves zero slack against a burst; 2 is the recommended floor.

**Why the token cannot change.** `kv_cache_pager`, `ddr5_xbar`, `expert_cache_pf` and the expert
FIFO are *transparent* to the compute die: the die pulls its weight/KV bytes same-cycle from the
weight/KV source (§2–§4); the pager/cache/xbar are the bandwidth/observability plumbing around it.
Reducing their capacity changes only counters (hit/miss, cold-row NVMe fetches, xbar req/resp) —
never the FP8 arithmetic. `PE_N` tiles the *same* matmul into more/fewer columns and reduces to
the identical accumulated sum. Proven: committed and compact runs emit the byte-identical token
stream `tok = {0, 11, 11}`; the all-minimums run (`PE_N=1 DDR_NCH=2 KV_RESIDENT=4 EFIFO_DEPTH=2
CACHE_SLOTS=1`) also matches.

**Build / verify.**
```
make sim-glm-compact    # runs the system TB at BOTH the committed and compact configs and
                        # asserts the compact token stream == committed (byte-identical)
make synth-glm-compact  # structural elaboration + check -assert + stat of the compact hierarchy
```
The TB knobs are overridable header parameters, e.g.
`iverilog -Pglm_fp8_system_tb.PE_N=2 ...`; the default (no `-P`) is the committed slice.

**Measurement caveat (honest).** yosys 0.66 cannot map the FP8 datapaths through ABC, so no LUT
count is emitted here — `synth-glm-compact` elaborates + `check -assert`s + `stat`s the compact
hierarchy. The **area reduction is by construction** (fewer PE columns / channels / ring+FIFO+cache
entries); the **byte-identical token is the verified invariant** (`make sim-glm-compact`). A real
LUT/FF delta needs the vendor flow (Gowin / nextpnr) on the elaborated compact netlist.

### 10a. Structural engine sharing (L1 — landed, byte-identical)
Beyond the *parametric* compact config (L0 above), the die also shrinks *structurally*: because
gate / up / down run at **different times** inside one expert, `swiglu_expert_fp8` now runs all
three on **one shared `glm_matmul_fp8`** (a 1-bit `up_pass` register + 2:1 weight/scale mux selects
the up-projection port; the old parallel `u_mm_u` engine is gone). Effect: swiglu 2→1 GEMM engines,
so each decoder block drops from **6→4** FP8 GEMM engines (dense + MoE swiglu each shed one) — the
freed matmul core is **6186 LUT4** each (measured, `PPA_FP8.md` §1.3), so **≈12K LUT4/block**;
the per-expert generic-cell delta is a measured **−1519**. After L1 **every chip module holds exactly one `glm_matmul_fp8`**
(mla already time-shares its 7 projections on one engine; router one; mtp one), so the bounded
byte-identical merges are exhausted; the last area lever is the invasive cross-module 3-way hoist,
deferred to after vendor measurement. This is free in time (the NVMe-bound die has the slack) and
keeps the decoded token byte-identical (`{4,31,20}`, gworst_rel 0.00689655). Full lever catalog and
status in [`MINIATURIZATION.md`](MINIATURIZATION.md).
