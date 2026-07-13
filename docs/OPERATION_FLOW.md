# AIPU — full operational flow (end-to-end)

> **Track.** This is the **Q4_K** operational flow — the current / `main` product track (GGML
> Q4_K, targeting [`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.2-GGUF),
> ~467 GB). The **prior FP8 datacenter-native track** is preserved on branch **`fp8`** (tag
> `fp8-verified-baseline`), referenced here as prior/preserved, **never current**. The Q4_K module
> names used below are `glm_q4k_system_cdc` / `glm_q4k_system` / `glm_model_q4k` /
> `glm_decoder_block_q4k` / `mla_attn_q4k` / `moe_router_q4k` / `swiglu_expert_q4k` /
> `glm_matmul_q4k` / `mtp_head_q4k` / `weight_loader_q4k` / `glm_q4k_soc_ms`.

> **⚠️ Verification honesty (read before any "verified/bit-exact" below).** All bit-exact results are
> vs the team's **own** ggml references (`tools/q4k_ref.py` / `tools/glm_model_q4k_ref.py`) — **not**
> the real downloaded GGUF bytes and **not** llama.cpp (a *different* arithmetic contract: llama.cpp
> quantizes activations to Q8_K + integer dot; this RTL uses **bf16 activations + fp32 accumulate**).
> Within that scope: `glm_matmul_q4k` is bit-exact (`make q4k`), the **assembled `glm_model_q4k` now
> has an end-to-end numeric golden** (`make model-q4k` / `model-q4k-acthw`: full forward vs the numpy
> reference, ALL 1155 tests bit-exact on logits+argmax+h_state), and the RTL **consumes the mixed
> Q6_K/Q8_0/F16 types** of a real UD-Q4_K_XL checkpoint (`make mixedtype`, bit-exact to the same
> reference). Real-checkpoint validation vs GGUF/llama.cpp remains **OPEN**. Every tok/s / J figure
> below is **[EST]**, roofline-modeled; the **FPGA fit is MEASURED** (Vivado ML 2026.1 full P&R on
> XCKU3P, routed Fmax 46.5 MHz — see [`fpga/`](../fpga/README.md)). See the [README](../README.md)
> for the full honest ledger.

How the whole accelerator runs one real GLM-5.2 (Q4_K) token, from power-up through a decoded
token, across every committed RTL block. Grounded in main @ current state (PE_M batching, grouped-MoE
union-skip, cycle-accurate emulation) — **main develops exactly the GLM-5.2 Q4_K accelerator at
rung-① (the offline FPGA prove-it demo); the full product at rungs ②③ is roadmap, not code in main
now** (see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). This is the *operational* view; per-block detail lives in
[`ACCEL_GLM52.md`](ACCEL_GLM52.md) / [`SYSTEM_SINGLE_PACKAGE.md`](SYSTEM_SINGLE_PACKAGE.md).

The one fact that shapes everything: **the workload is NVMe/PCIe-bandwidth-bound.** The ~467 GB Q4_K
model lives on the NVMe SSD; each token streams its active experts through a DDR5 cache into a mostly-idle
Q4_K die. Throughput ≈ `NVMe_BW / [(1−h)·footprint] · K` — with the measured correction that **K (the
spec multiplier) must be read as A/U(K)**: the K+1 verify rows fetch the *union* of their experts
(measured U(4)=2.25–2.64 on the OLMoE proxy; superseded 2026-07 by the GLM-family measurement,
GLM-4.5-Air: U(4)=2.60–2.71), so the amortization is ~1.1–1.3× at K=4 (A≈3), not ~2×; measured h/U
in [`H_MEASUREMENT.md`](H_MEASUREMENT.md) (see also
[`MOE_LOCALITY_RESEARCH.md`](MOE_LOCALITY_RESEARCH.md)). *(Updated 2026-07: this NVMe-streaming
operating point is the **rung-①** (this demo) / hybrid-upside-SKU / >512 GB regime; the **rung-③
primary design point is now FULL RESIDENCY** — the whole ~467 GB checkpoint resident in 512 GB
LPDDR5X (~1.1 TB/s), h=1 by construction, no per-token NVMe streaming, design point ≈80 tok/s
[EST] — see [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md).)*

## 0. Physical / logical stack

```
  ┌───────────────────────────── AIPU module (one board / one die) ─────────────────────────────┐
  │                                                                                              │
  │   HOST (USB-C)  ──req──►┌─CDC─┐──►┌──────────────── glm_q4k_system (compute clock) ─────────┐ │
  │                 ◄─tok───┤fifo │   │                                                         │ │
  │                         └─────┘   │   glm_model_q4k  (the Q4_K compute die)                 │ │
  │   glm_q4k_system_cdc              │      embed → [decoder_block ×78 time-mux] → norm → LMhead│ │
  │   (2-clock top, reset_sync)       │            │  ▲ pull weights / KV / experts             │ │
  │                                   │            ▼  │                                         │ │
  │                                   │   weight_loader_q4k · expert_cache_pf · kv_cache_pager   │ │
  │                                   │            │  ▲                                          │ │
  │                                   │      ddr5_xbar (N-ch)     flash_xbar (N-ch, QDEPTH)      │ │
  │                                   └──────│──────────────────────────│─────────────────────┘ │
  │                                   ┌──────▼──────┐            ┌───────▼────────┐               │
  │                                   │ DDR5 (work) │            │  NVMe (1-4 TB)  │              │
  │                                   │ hot-set+LRU │◄─boot_loader│ (~467 GB model)│              │
  │                                   └─────────────┘   load     └────────────────┘               │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **`glm_q4k_system_cdc`** — the 2-clock chip top: host/USB clock ↔ compute clock via
  `cdc_async_fifo` (request in, token out), `reset_sync` per domain.
- **`glm_q4k_system`** — the compute-domain core: the die + the memory subsystem.
- Memories are TB-modeled here; real DDR5 / NVMe (PCIe) / USB-C PHYs are vendor IP (out of scope).
- **DDR is rung-dependent** — the diagram's DDR5 working store is the *funded* rung-② point, **not THE spec**:
  the near-term prove-it FPGA runs DDR4 (~4 ch, ~100 GB/s), the funded custom board runs DDR5 multi-ch /
  HBM (~300–600 GB/s); see the hardware ladder ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). NVMe (the
  ~467 GB model store) is the same on rungs ①② — performance is set by memory bandwidth, i.e. by which
  silicon the budget buys. *(Updated 2026-07: on the **rung-③ primary full-residency SKU** the whole
  checkpoint is LPDDR5X-resident (512 GB) and cold storage is one commodity M.2 NVMe boot drive
  (boot-load ~70 s), not a per-token stream — see [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md).)*

## 1. Boot — execution conditions & resident-set load (NVMe → DDR5)

**Inference is NOT released by power-on** — it is released by `boot_loader.done`. The full power-on
→ ready sequence (its *execution conditions*, in order):

| # | condition | what | who |
|---|---|---|---|
| 1 | **power** | all rails up | board |
| 2 | **clocks stable** | `host_clk` (USB), `core_clk` (compute), memory clk — PLLs locked | board / vendor IP |
| 3 | **reset sequenced** | `host_rst` / `core_rst` (per-domain, sync active-high) cleanly de-asserted (`reset_sync`) | RTL |
| 4 | **memory PHY init** | DDR5 training + NVMe/PCIe controller init | vendor IP |
| 5 | **model present on NVMe** | the ~467 GB Q4_K model **pre-written** (one-time provisioning, `ckpt_pack_q4k.py` / `flash_layout.py`) | manufacturing / setup |
| 6 | 🔑 **`boot_loader.done`** | DMA the **~17 GB resident hot partition** (all-layer attention, dense-FFN, MoE router `W_g`, shared expert, embeddings, LM-head, norm gammas — ~28B non-routed params × ~0.62 B/param; canonical byte constants: [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2) **NVMe → DDR5** — its registered `done` is the **single gate that releases inference** | RTL (`boot_loader`, **BMC + unbounded k-induction proven**) |
| 7 | **USB enumerated** | host driver loaded, endpoint open | host + vendor USB IP |

The **256 routed experts stay on the NVMe SSD** (~467 GB ≫ the ~17 GB DDR-resident hot partition) and are demand-streamed per token
(§4). Boot 6 is pure DMA — no arithmetic, byte-exact.

**Timing (one boot, [EST]) — this §1 describes the streaming-SKU boot (only the ~17 GB hot partition
is loaded; the 256 routed experts stay on NVMe and demand-stream per token):** PLL lock (~ms) + DDR5
training (~10–100 ms) + hot-partition load (~17 GB / NVMe read BW — ~2.5–5 s on one Gen3/4 ×4 drive at
~3.5–7 GB/s, dropping toward ~1–2 s with several NVMe striped across more PCIe lanes) + USB enum (~ms)
≈ **~2 s (multi-NVMe array) to a few seconds (single drive) power-on → ready** on the streaming SKU.
Short boot, not instant-on.

> **Primary full-residency SKU (R3) boots differently and much slower.** The rung-③ residency box
> ([`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md)) holds the **whole ~467 GB** checkpoint in 512 GB
> LPDDR5X, so `boot_loader` copies the **full ~467 GB NVMe → LPDDR5X**, not a 17 GB slice. That is a
> **~70 s cold boot on EVERY power-on** (measured-inputs [EST]: ~467 GB ÷ NVMe read BW). LPDDR5X is
> volatile, so this reload happens on every cold start — the ~1–2 s number above is the streaming SKU
> only and does **not** apply to the residency box.

**Three timescales:** ① *one-time provisioning* — write the ~467 GB Q4_K model to the NVMe SSD.
② *every power-on* — conditions 1–7 (streaming SKU ~1–2 s; **primary residency SKU ~70 s** for the full
~467 GB NVMe→LPDDR5X load). ③ *per token* — §2 (streaming SKU demand-streams experts from NVMe; the
residency SKU reads them from resident LPDDR5X; KV lives in DDR5/LPDDR5X, per session).

**Host interface (what USB-C carries — all on `host_clk`):** in = `start` (pulse), `prompt_tok`
(token ID), `start_pos`, `s_len`; out = `next_tok` (token ID), `tok_valid`, `busy`, `done`. Just
token IDs + position/length — the heavy weight/KV traffic never crosses USB-C (it is all inside
`glm_q4k_system` on `core_clk`).

> **Real-hardware note.** The RTL takes clocks/resets as ports and treats DDR5 / NVMe (PCIe) / USB PHYs as
> vendor IP (conditions 2, 4, 7 are the board/vendor bring-up — device-plan Phase D1,
> [`USBC_PRODUCT_PLAN.md`](USBC_PRODUCT_PLAN.md)). The RTL's own execution condition is the
> `boot_loader.done` gate (6) once those are up.

## 2. Per-token decode — the pipeline

One decode step (`glm_model_q4k`), for a batch of **`PE_M` = B** token rows:

```
 token_id[0..B-1]
   │ ① embed lookup (bf16 table, em_* pull; SERIAL per row)          -> x0[r]  (B × MODEL_DIM)
   ▼
 ┌── for layer l = 0 .. 77  (ONE time-multiplexed glm_decoder_block_q4k, mode = dense|MoE) ──┐
 │  ② RMSNorm(x)                                                                             │
 │  ③ MLA attention  ──────────────► h = x + attn                                            │
 │       (7 Q4_K weight projections, latent KV, DSA sparse index, interleaved RoPE, bf16 softmax) │
 │  ④ RMSNorm(h)                                                                             │
 │  ⑤ FFN:  l<3 → dense SwiGLU  |  l≥3 → MoE (router top-8/256 → union-skip experts+shared)  │
 │       ──────────────► x_{l+1} = h + ffn                                                   │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
   │ ⑥ final RMSNorm  (PE_M rmsnorm_units, lockstep off ONE shared gamma pull)
   ▼
 ⑦ LM-head GEMV (bf16, glm_matmul_pipe over VOCAB=154880)  -> logits[r]
   │ ⑧ argmax / sample (per row)                            -> next_token[r]
   ▼
 ⑨ mtp_head_q4k: t+2 speculative draft (per row)           -> draft token (for §6)
```

Key structural facts:
- **One decoder block, time-multiplexed over 78 layers** (not 78 copies) — the per-layer weights
  are *pulled* fresh each layer; the block is a fixed amount of logic. The residual `x` streams
  layer→layer.
- **PE_M batching:** all B rows share ONE weight-fetch stream per GEMM
  (`aw_req`/`fw_req`/… pulse identically to a single-row run) — "B rows == 1 fetch". Each row
  carries its own residual and its own **bf16 activation tail** (norm/RoPE/softmax/argmax). Q4_K is a
  **weight-only** quant: activations stay **bf16**, weights dequantize to fp32 for the MAC (no per-row
  activation-scale bookkeeping — that `a_shift` machinery was FP8-only, prior track).
- **All weight / KV / embedding delivery is via pull ports** (`em_*`, `aw_*`, `kc_*`, `rw_*`,
  `fw_*`, `pw_*`, `lw_*`) answered by the memory subsystem — no combinational loop into the die.

### 2a. Attention (step ③) — `mla_attn_q4k`
Latent MLA: down-project to a small KV latent, cache it in the **`kv_cache_pager`** ring, gather
the sparse key set the **DSA indexer** selects (IndexShare: the index is computed by a full
indexer layer and *reused* by shared layers), apply interleaved RoPE, bf16 softmax over the top-K
window (SWIN, decoupled from the 1M position field), up-project through `W_o`. All 7 weight
projections are **Q4_K block-dequant GEMMs** (`glm_matmul_q4k`) — per-column super-block (fp16
`d`,`dmin` + packed 6-bit block-scales/mins + 4-bit codes) dequantized to fp32, `bf16 activation ×
fp32 weight` MAC, fp32-accumulated in K, rounded to bf16; scores/probs/softmax stay bf16.

> **(RESOLVED)** The MLA softmax `1/sqrt(qk_head_dim)` scale — previously omitted by both the DUT
> and its TB golden (a silent divergence) — is now **applied**, and the assembled forward is checked
> against the independent numpy golden (`make model-q4k`, 1155 bit-exact).

### 2b. MoE FFN (step ⑤, layers ≥ 3) — `moe_router_q4k` + grouped experts
1. **Router** (`moe_router_q4k`): Q4_K GEMV `x·W_g` → top-8 of 256 experts + renormalized gates
   (×2.5 routed-scaling), per row. (`moe_router_q4k` gates on structural/renorm invariants,
   40/40 — **not** a numeric golden.)
2. **Union-skip grouped dispatch** (folded **inline into `glm_decoder_block_q4k`**, PE_M>1): scan the
   expert axis and evaluate **only the UNION** of experts any of the B rows selected (a combinational
   `any_has` skip — non-selected experts are **never fetched**). For each union expert, fetch its SwiGLU
   weights **once** and run `swiglu_expert_q4k` at PE_M over all B rows; each row accumulates
   `gate·expert(x)` only if it selected that expert. (Inside the expert, gate/up/down share **one**
   Q4_K GEMM engine — §10a.) **Byte-identical to per-row by construction**; up to ~32× fewer
   NVMe expert-fetches at small B (the aggregate-throughput lever). Then the always-on **shared
   expert** (weight 1). *(The standalone `batched_moe.v` reference of this dispatch was retired in
   the Q4_K retarget — the union-skip logic now lives inline in `glm_decoder_block_q4k`.)*

## 3. Weight paths — resident vs demand-streamed

| what | where it lives | path to the die | per-token cost |
|---|---|---|---|
| attention / dense-FFN / router `W_g` / norms / embed / LM-head | **DDR5-resident** (the ~17 GB hot partition, boot-loaded; per-token touch ~11 GB — [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md) §2) | `ddr5_xbar` → `weight_loader_q4k` → die pull | small, fixed |
| **routed experts** (the ~467 GB bulk) | **NVMe SSD** | `flash_xbar` → `expert_cache_pf` (DDR5 LRU) → die | **the bottleneck** |

- **`flash_xbar`** is the storage-read fabric (a committed RTL identifier, kept as-is); in the product it
  **fronts the NVMe/PCIe backend** — a labeled placeholder, since the crossbar's read-request /
  latency-hiding abstraction (address → weight bytes) is medium-agnostic, so the NAND-specific backend is
  swapped for an NVMe/PCIe host controller. It banks reads across N channels (**PCIe lanes / multiple NVMe
  drives**) and hides storage-read latency (~10–100 µs) with a deep per-channel outstanding queue
  (QDEPTH ~ FLASH_LAT, Little's law) → ~N× aggregate BW. Placement matters: expert→channel layout
  (`flash_layout.py`) + the proposed sub-expert striping keep all channels busy
  ([`FLASH_STRIPING.md`](FLASH_STRIPING.md)).
- **`expert_cache_pf`** — DDR5 routed-expert cache: LRU + frequency + confidence-thresholded
  prefetch; a demand miss stalls the die for the exposed NVMe-refill (see §7). **Honest caveat:** the
  predictor-driven prefetch is, so far, a **measured no-op** — top-8-of-256 routing entropy caps the
  achievable hit-rate, so the win comes from the LRU working-set fit, not prediction (see the perf
  docs). `weight_decomp` (optional) losslessly decompresses on the NVMe→loader refill (fewer NVMe bytes
  = more effective NVMe bandwidth + less read energy).
- **`weight_loader_q4k`** — turns cache/DDR5 responses (Q4_K super-blocks: fp16 `d`,`dmin` + packed
  6-bit block-scales/mins + 4-bit codes) into the die's matmul pull stream, **bit-exactly vs
  `tools/q4k_ref.py`**.

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
  B separate runs [EST]). `glm_q4k_soc_ms` is the batched multi-seq SoC top
  (`glm_model_q4k` at PE_M=B + `NSEQ`-window pager + a REAL per-layer KV store `kv_mem` owned by
  the top + host FSM: prefill B seqs → 1 forward → commit B tokens). *Byte-identical at
  `PER_ROW_SEQ=0` by construction; the per-row-argmax == per-seq-PE_M=1 bit-exactness was a
  **prior-FP8 result** (branch `fp8`) — a Q4_K re-run is **PENDING**.*
- **Continuous-batching decode loop (`glm_q4k_soc_ms`, `N_STEPS>1`):** one host `start` decodes
  **N tokens per sequence** — a `RUN→DECAP→RUN` loop that runs one PE_M=B forward, streams the B
  argmax out (`tok_valid`), writes each decode token's latent into `kv_mem` at the growing position
  (`s_len + dec_step`) for every layer, feeds the argmax back as the next step's input, and advances
  position/extent. *`N_STEPS=1` is byte-identical to the single-step top by construction; the step-k
  == standalone-PE_M=1 equivalence was a **prior-FP8** result, Q4_K re-run **PENDING**.*
- **Speculative decode (`spec_batched_top`):** the MTP head drafts K tokens; the main model
  **verifies all K+1 positions in ONE PE_M=K+1 weight-load** (NVMe traffic ÷ up to K+1 on the
  shared-weight streams; measured caveat — the routed-expert amortization is only A/U(K) ≈ 1.1–1.3×
  at K=4, since the verify rows union their experts, measured U(4)=2.25–2.64 on the OLMoE proxy —
  superseded 2026-07 by the GLM-family measurement, GLM-4.5-Air: U(4)=2.60–2.71:
  [`H_MEASUREMENT.md`](H_MEASUREMENT.md)), and the
  committed stream is proven **== greedy** (spec==greedy safety — a real lossless-speculation
  property, but **DUT-vs-DUT self-consistency**: the greedy reference is itself a `glm_model_q4k`,
  **not** a numeric golden vs ggml/llama.cpp). `spec_chain_top` chains the MTP steps.
  `spec_decode_top` runs in `make unittests` (**18/18**); the larger K>1 loops
  (`spec_batched_top` / `spec_chain_top`) run via `make spec-slow` (minutes-long). *(Updated
  2026-07: **adaptive draft depth landed** — `spec_decode_seq` `ADAPT` param (default 0,
  yosys sequential-equivalence proven unchanged) + the new `spec_depth_adapt` policy module,
  K adaptive in [1..5], output-invariant by construction (spec==greedy for ANY depth schedule);
  gate `make spec-adapt`.)*

## 5. Clocking / CDC
Host/USB requests cross into the compute domain through `cdc_async_fifo` (+ `reset_sync` per
domain); the decoded token crosses back out. CDC soundness is **structurally signed off** — `make
cdc` (`tools/cdc_check.py`) asserts every `host_clk`↔`core_clk` crossing flows through a recognized
synchronizer (async FIFO / 2-FF / `reset_sync`, no raw multi-bit capture), and the whole 2-clock top
(`glm_q4k_system_cdc`) elaborates clean under `make synth-glm` (`hierarchy -check` + `check -assert`,
exit 0). *(A running byte-identical-across-async-clocks token-binding TB existed on the FP8 track;
the Q4_K analogue is a structural sign-off, not a sim — see §8.)*

## 6. Timing & bandwidth — how fast, and why

- **The die is ~75 % idle**, gated behind NVMe/PCIe bandwidth; `clk_en_ctrl`/ICG clock-gates the idle
  cycles. So die-side fmax is **not** the throughput knob — NVMe/PCIe BW is.
- **Stall mechanism (cycle-accurate emulation, `EXPERT_STALL`) — validated on real RTL cycles:** a
  demand-miss delays the token by the exposed NVMe-refill; the exposed stall rises **exactly
  `stall = 3·FLASH_LAT + 9`** (slope 3 = DDR beats per expert-refill slice), so `cyc_per_tok` grows
  with storage-read latency — the roofline *mechanism* measured on real RTL cycles
  ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md)). **Q4_K measured (2026-07-11, `make perf-q4k`):**
  the harness is ported (`test/glm_q4k_system_perf_tb.v`) and re-run on `glm_q4k_system` — slice
  `cyc_per_tok` ≈ **10,896**; the residency pivot is confirmed on real cycles (RESIDENT=1 exposes
  35 stall cyc/token vs 2,567 at RESIDENT=0/`FLASH_LAT=1024`, ~73×). The FP8 counts in that doc are
  retained only as the historical mechanism reference. *(Update: with the
  routed Fmax measured at 46.5 MHz, the demo wall-clock is now computable — slice `cyc_per_tok`
  ~8.0–11.0K (FP8-era absolute; Q4_K similar ballpark plus a few hundred repipeline-latency cycles)
  → ~170–240 µs/token ≈ **~4,200–5,800 slice tok/s** — the correctness-demo speed of the tiny slice,
  NOT a GLM-5.2 product number.)*
- **Projected (roofline, `[EST]`) — staged to the hardware ladder ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)):**
  single-user tok/s is set by memory bandwidth, which is set by the silicon the budget buys, so it is
  **rung-dependent** — **~5–8 tok/s on the near-term prove-it FPGA** (DDR4 ~4 ch, the buildable demo
  *now*), **~15–40 tok/s on the funded custom board** (DDR5 multi-ch / HBM, rung ②), **~40+ tok/s at
  ASIC volume** (rung ③, custom silicon w/ HBM stacks + near-memory compute — lower $/seat + power once
  the NRE amortizes over volume). All `[EST]`, ~3 J/token; the old flat "~25–40" is the **funded rung-②**
  number, not the cheap near-term box. *(Update — measured-roofline design-point menu [EST, with
  measured-proxy h/U inputs from [`H_MEASUREMENT.md`](H_MEASUREMENT.md), OLMoE trace]: 1–2 NVMe, no
  multipliers ~0.5–1 tok/s; 90 GB DRAM + 100 GB/s → 13–24; 90 GB + 200 GB/s (ONFI 64-ch) → 25–47;
  225 GB + 200 GB/s → 54–127 — the "100 tok/s" design point. bandwidth-h: 20% pool cached (~90 GB
  GLM-scale) → h=0.36–0.60; 50% (~225 GB) → 0.72–0.88.)* *(Updated 2026-07: the **rung-③ primary
  design point is now FULL RESIDENCY** — 512 GB LPDDR5X (~1.1 TB/s) holds the whole ~467 GB
  checkpoint, design point **≈80 tok/s [measured-inputs EST]** (U(K) **and** the MTP accept rate r
  both GLM-family measured — job B's vLLM MTP sweep put the memory-bound optimum at K=1–2; ~95 if
  GLM-5.2's deeper MTP hits its published accept depth) — see
  [`R3_APPLIANCE_SPEC.md`](R3_APPLIANCE_SPEC.md); "54–127" is no longer the rung-③ design point —
  the streaming menu above stays true for rung ①, the hybrid upside SKU, and >512 GB checkpoints.
  U(K) is now GLM-family measured (GLM-4.5-Air, [`H_MEASUREMENT.md`](H_MEASUREMENT.md) 2nd
  measurement: U(4)=2.60–2.71, U(8)=4.19–4.41), superseding the OLMoE first-pass; h stays
  proxy-measured but is no longer product-deciding — residency ⇒ h=1 by construction, h-curves
  matter only for the hybrid-SKU decision.)* **This is the product: a
  fully offline / air-gapped local box** that
  runs the whole 753B-param frontier model with the ethernet unplugged. The capability it unlocks is frontier AI where the cloud can't
  reach — SCIFs, isolated OT / critical-infra, field/edge, or anywhere a vendor connection is itself the
  liability; the proof is binary — **nothing leaves because there's no path out** (the host link carries
  only token IDs + position, §1), so it passes the unplugged-ethernet test that every cloud option fails,
  "secured cloud" included (in-VPC / zero-retention / TEE all need connectivity). Offline alone is
  table-stakes (a 70B laptop is offline too); the moat is the **combination** — offline + full-frontier
  (753B) + appliance price. Honest caveat: the ~467 GB model is written to the NVMe SSD once (one-time
  provisioning, §1) and model updates are physical re-provisioning. The *batched aggregate* tok/s
  figure is a **non-target datacenter regime** (per-user floors at ~0.14 tok/s at B≈256), kept only as
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

| stage | module(s) | status (Q4_K, per the honest ledger) |
|---|---|---|
| chip top / CDC | `glm_q4k_system_cdc`, `cdc_async_fifo`, `reset_sync` | CDC structurally signed off (`make cdc`); 2-clock top elaborates clean (`make synth-glm`) |
| system core | `glm_q4k_system` | ELABORATED — `hierarchy -check` + `check -assert` clean (`make synth-glm`); no Q4_K system-level numeric TB |
| batched multi-seq SoC | `glm_q4k_soc_ms` (PE_M=B model + `NSEQ` pager + `kv_mem` + host FSM; `N_STEPS` decode loop) | mechanism present; per-row / decode-loop bit-exactness = prior-FP8 result, Q4_K re-run **PENDING** |
| compute die | `glm_model_q4k` → `glm_decoder_block_q4k` | assembled Q4_K forward **bit-exact vs the numpy golden** `tools/glm_model_q4k_ref.py` (`make model-q4k` + `model-q4k-acthw`, ALL 1155: logits+argmax+h_state) — still our own numpy reimpl, not llama.cpp/GGUF; plus spec==greedy self-consistency |
| attention | `mla_attn_q4k`, `dsa_indexer`, `kv_cache_pager` | covered end-to-end within the assembled `make model-q4k` golden; `1/sqrt(d)` softmax scale now **applied**; `kv_cache_pager` BMC + k-induction (+ ECC ring) |
| MoE | `moe_router_q4k`, `swiglu_expert_q4k`, inline union-skip | router 40/40 (renorm invariants, **not** numeric); swiglu 240/240 (functional, self-labeled, **not** bit-exact); union byte-identical **by construction** |
| Q4_K GEMM | `glm_matmul_q4k` + `q4k.vh` primitives | **PROVEN — bit-exact vs ggml `tools/q4k_ref.py`**: `glm_matmul_q4k` 160/160, `q4k_prim` 18/18 (`make q4k`); mixed-type Q6_K/Q8_0/F16 consumers bit-exact to the same reference (`make mixedtype`) |
| head | `mtp_head_q4k`, `sampler`, LM-head `glm_matmul_pipe` | LM head + argmax covered within the assembled `make model-q4k` golden (1155); also exercised via spec==greedy |
| memory | `flash_xbar`, `ddr5_xbar`, `expert_cache_pf`, `weight_loader_q4k`, `boot_loader`, `weight_decomp` | FORMAL — BMC + unbounded k-induction (7 controllers + ECC ring); `weight_loader_q4k` bit-exact vs `q4k_ref.py` |
| spec decode | `spec_decode_top`, `spec_batched_top`, `spec_chain_top`, `spec_decode_seq`, `spec_depth_adapt` | spec==greedy (DUT-vs-DUT): `spec_decode_top` 18/18 (`make unittests`); K>1 loops via `make spec-slow`; `spec_decode_seq` BMC + k-induction; adaptive depth (`ADAPT` + `spec_depth_adapt`, K∈[1..5], output-invariant) via `make spec-adapt` |

## 9. Honest scope
This is the flow of the **committed slice** (every operator + ratio faithful) plus the
memory/streaming system that runs the real 753B-param model. What is *modeled* (not silicon): the PHYs,
and all `[EST]` tok/s/J. **Fidelity, stated honestly:** all bit-exact results are vs the team's
**own** ggml references — `glm_matmul_q4k` (+ the `q4k.vh` primitives) vs `tools/q4k_ref.py`
(`make q4k`; mixed-type Q6_K/Q8_0/F16 via `make mixedtype`), and the assembled `glm_model_q4k` full
forward vs the numpy golden `tools/glm_model_q4k_ref.py` (`make model-q4k` / `model-q4k-acthw`, ALL
1155 bit-exact; the MLA `1/sqrt(qk_head_dim)` softmax scale is applied) — **not** the real GGUF bytes
and **not** llama.cpp. The FPGA P&R fit is **MEASURED** (Vivado ML 2026.1 on XCKU3P, routed Fmax
46.5 MHz, campaign closed — see [`fpga/`](../fpga/README.md)). Still **OPEN**: bit-exactness vs the
real downloaded GGUF / llama.cpp, and the board bring-up/run (see
[`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md), [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md)). *(The prior-FP8
real-checkpoint validation — operator-bit-exact vs the published `GLM-5.2-FP8` safetensors + a truncated
real-weight token chain — lives on branch `fp8`; it is **not** a Q4_K result.)*

## 10. Compact config (FPGA miniaturization)

Target: fit the chip on a **small FPGA** — measured target: **XCKU3P via the Vivado flow** (the old
Gowin Tang Mega 138K / GW5AT-138 / nextpnr scaffold was removed, superseded). Five parameters of
`glm_q4k_system` (and its 2-clock top `glm_q4k_system_cdc`) set **capacity / parallelism /
bandwidth — never the math**. Shrinking them makes the elaborated logic smaller (fewer LUT/FF in
the matmul PE array, the DDR5 crossbar, the KV ring, the expert-request FIFO, and the expert
cache) and slower / lower-BW, but the **decoded token is byte-identical by construction**. They are
*result-invariant*.

**Compact synth config** (overrides on the `glm_q4k_system_cdc` full-model defaults):

| param | default | compact | true safe-min | constraint | what it sizes | shrink / cost when reduced |
|---|---|---|---|---|---|---|
| `PE_N`        | 4  | **2** | 1              | —                     | matmul PE-array columns          | halves the PE array (**biggest die saving**); tiles more → same output, more cycles |
| `DDR_NCH`     | 4  | **2** | 2              | power-of-two          | DDR5 fabric channels (`ddr5_xbar`)| smaller crossbar; ~½ aggregate read BW (NVMe/PCIe-bound anyway) |
| `KV_RESIDENT` | 16 | **8** | `S_MAX` (=8)   | POW2, `>= S_MAX`      | latent-KV ring capacity          | smaller ring RAM; more cold-row NVMe gathers |
| `EFIFO_DEPTH` | 16 | **8** | 2              | power-of-two          | routed-expert request FIFO depth | smaller FIFO; risk of drop only under bursty routing |
| `CACHE_SLOTS` | 4  | **2** | 1              | —                     | expert-cache slots               | smaller tag/data array; more misses → more NVMe stalls |

The compact column is the recommended FPGA set (keeps head-room over the true minimum).
`KV_RESIDENT >= S_MAX` and POW2; the full model runs `S_MAX=8`, so the compact synth uses 8.
`EFIFO_DEPTH=2` is the recommended floor (leaves a little slack against a routing burst).

**Why the token cannot change.** `kv_cache_pager`, `ddr5_xbar`, `expert_cache_pf` and the expert
FIFO are *transparent* to the compute die: the die pulls its weight/KV bytes same-cycle from the
weight/KV source (§2–§4); the pager/cache/xbar are the bandwidth/observability plumbing around it.
Reducing their capacity changes only counters (hit/miss, cold-row NVMe fetches, xbar req/resp) —
never the Q4_K arithmetic. `PE_N` tiles the *same* matmul into more/fewer columns and reduces to
the identical accumulated sum. The byte-identical token is therefore a **structural invariant** (by
construction), not a measured one on this track.

**Build / verify.**
```
make synth-glm          # structural elaboration of glm_q4k_system_cdc (the full compute + memory + CDC
                        # hierarchy): hierarchy -check + proc + opt + check -assert + stat, exit 0.
```
The TB knobs are overridable header parameters (e.g. `iverilog -Pglm_q4k_system_cdc.PE_N=2 ...`).

> **Prior-FP8 note.** On the FP8 track a running **system TB** (`glm_fp8_system_tb.v`) plus `make
> sim-glm-compact` / `make synth-glm-compact` targets asserted the compact token stream was
> **byte-identical** to the committed config across the parameter sweep (all-minimums included). The
> **Q4_K equivalents do not exist on `main`** — a Q4_K system / compact-sweep TB is **PENDING**; the
> current Q4_K guarantee is the *by-construction* structural argument above plus `make synth-glm`.

**Measurement caveat (honest).** yosys cannot map the fp32/Q4_K dequant datapaths through ABC, so no
LUT count is emitted here — `make synth-glm` elaborates + `check -assert`s + `stat`s the hierarchy.
The **area reduction is by construction** (fewer PE columns / channels / ring+FIFO+cache entries).
The vendor flow is now **Vivado** (the Gowin/nextpnr scaffold was removed), and the full-system
compact-config fit is **measured** there — 142,320 LUT (87.5%) / 421 DSP on XCKU3P (see
[`fpga/`](../fpga/README.md)); per-lever LUT/FF deltas still need per-config Vivado runs.

### 10a. Structural engine sharing (landed, byte-identical by construction)
Beyond the *parametric* compact config above, the die also shrinks *structurally*: because
gate / up / down run at **different times** inside one expert, `swiglu_expert_q4k` runs all
three on **one shared `glm_matmul_q4k`** (a 1-bit `up_pass` register + a 2:1 weight/scale mux selects
the up-projection port — there is no second parallel engine). Effect: swiglu 2→1 GEMM engines, so
each decoder block sheds GEMM engines (dense + MoE swiglu each drop one). After this landed, **every
chip module holds exactly one `glm_matmul_q4k`** (`mla_attn_q4k` already time-shares its 7 projections
on one engine; `moe_router_q4k` one; `mtp_head_q4k` one), so the bounded byte-identical merges are
exhausted; the last area lever is the invasive cross-module 3-way hoist, deferred to after vendor
measurement. This is free in time (the NVMe-bound die has the slack) and keeps the decoded token
**byte-identical by construction**.

> **Prior-FP8 measurement (branch `fp8`, not re-run for Q4_K).** The area of the merge was quantified
> on the FP8 track: the freed matmul core was **~6186 LUT4** each (≈12K LUT4/block), per-expert
> generic-cell delta **−1519**, decoded token byte-identical (FP8 slice). Those numbers are **FP8**;
> a Q4_K re-measure is **PENDING** (yosys can't map the datapath; the Vivado flow now exists, but the
> per-lever per-config run has not been done). Full lever
> catalog and status in [`MINIATURIZATION.md`](MINIATURIZATION.md).
