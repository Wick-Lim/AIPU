# Product Roadmap — GLM-5.2-FP8 accelerator (product, not research)

The `prototype` branch (frozen at `47fb7f8`) holds the **research prototype**: the full FP8
datapath + memory system + ultra-perf batching stack, **bit-exact and mechanism-proven at a
small-but-faithful slice**, with honest gaps documented. It answers *"does the architecture work,
and how fast can it go?"* — **yes**, and the levers are measured.

`main` now develops the **GLM-5.2-FP8 accelerator at rung ① — the FPGA prove-it track**: a
manufacturable design whose near-term goal is a **working FPGA demo** proving the published
`zai-org/GLM-5.2-FP8` runs **reliably on real low-end FPGA silicon, offline and bit-exact**. The
**full product** — the funded custom board (rung ②) and the volume ASIC/SoC (rung ③) — is the
**roadmap documented below**, not the code `main` develops right now. The mindset shifts from
*demonstrate + measure a mechanism* to *run the real model correctly, at full scale, robustly, and
ship it.*

> **What the product IS: a LOCAL, single-user personal box that works with the ethernet unplugged.**
> One box, one user, running the full 753 B model **locally** — it **works fully offline / air-gapped:
> nothing leaves because there's no path out**. The audit is literally *"does it still answer with the
> ethernet cable unplugged?"* — **yes**, the strongest, binary form of 'non-egress'; it categorically
> excludes *every* cloud option (in-VPC / zero-retention / confidential-computing TEE all still need a
> connection and fail the unplugged test). Lead with the **capability** that unlocks, not the defense:
> finally run a **full frontier** model in the disconnected places you're locked out of today (SCIF /
> classified, isolated OT & critical-infra, field/edge, denied-connectivity), and **own it outright** —
> no per-token API fees, no vendor that can rate-limit, deprecate, or cut you off. *Offline alone is
> table-stakes (a 70 B laptop model is offline too); the moat is the **combination** — offline + full
> frontier (753 B) + appliance/seat price.* (Honest: the 753 GB checkpoint is loaded **once** — itself
> doable offline — and model updates are physical re-provisioning; after that, fully disconnected.) The
> performance metric that
> matters is **single-user interactive throughput**, and it is **rung-dependent** (set by the
> silicon's memory bandwidth — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)): **~5–8 tok/s [EST]
> on the near-term prove-it FPGA (rung ①), ~15–40 tok/s [EST] on the funded custom board (rung ②)**,
> and ~40+ at manufacturing volume (rung ③) — the **same bit-exact FP8 RTL** on every rung, only the
> memory interface changes. The design is deliberately **NVMe/PCIe-bandwidth-bound to be cheap** (an
> NVMe SSD holds the whole model; **fast DDR** — DDR4 on rung ①, DDR5 or HBM on rung ② — caches the
> hot working set). Any **aggregate / datacenter-batch** numbers in these docs
> (B≈256, ~50 tok/s *aggregate*, per-user ~0.14 tok/s) are a **secondary analysis of a different,
> non-target deployment** of the same silicon — the RTL supports it, but it is **not this product**,
> and its per-user latency never describes the box you plug in.

> **Two tracks.** This doc is the **RTL / silicon track** (make the chip correct, full-scale,
> robust, synthesizable). The **device / appliance track** — the USB-C external box (form factor,
> power, thermal, host software, enclosure, manufacturing, pricing) — is in
> [`USBC_PRODUCT_PLAN.md`](USBC_PRODUCT_PLAN.md). Its first gates (real-model fidelity + FPGA fit)
> are P1.1 and the vendor-flow measurement here.

---

## The research → product gap (what must change)

| Dimension | Prototype (have) | Product (need) |
|---|---|---|
| Correctness scope | operator bit-exact + a **truncated full-model token chain on real weights** (dense→MoE seam, real 256-expert route, DSA threaded, argmax-identical on a real prompt — DSA-IndexShare + fused-expert blockers retired) | the **full 753 GB checkpoint** produces the real model's tokens **at full depth** end-to-end |
| Scale | small faithful slice (128/6/8); the **full 753B config elaborates clean** (verilator, 0 errors) | full config *simulated/run* (6144, 78 layers, 256 experts, vocab 154880, 1M ctx) |
| Batching/KV | PE_M batch on all 4 wrappers; per-row position/extent threaded model→decoder→mla; KV pager has `NSEQ` INDEPENDENT ring windows; **multi-sequence batched attention (`PER_ROW_SEQ`) is real end-to-end through the full model** — per-row-slot union + `kc_seq` routing, each row attends its own sequence's KV while sharing the query-side weight fetch (full-model TB: 2 seqs, per-row argmax/logits bit-exact, ~41% fewer attn-weight beats than two runs); all byte-identical at PER_ROW_SEQ=0; **a batched multi-seq top `glm_fp8_soc_ms` (PE_M=B model + real NSEQ-window pager + host FSM: prefill B seqs → 1 forward → commit B tokens; per-row bit-exact)**; **`DSA_REAL_IDX=1` (query-dependent IndexShare) works under multi-seq via a per-sequence `kidx_buf` pre-fetch**; **a REAL per-layer KV store (`kv_mem`) owned by the top** (host writes per (seq,layer,pos), model reads combinationally); **real draft chaining (`spec_chain_top`) + batched_moe full B-coverage (`make bcov`, B∈{1,2,3,5,8})**; **multi-step continuous-batching DECODE LOOP (`glm_fp8_soc_ms` `N_STEPS>1`: one start decodes N tokens/seq, argmax fed back, extent/pos grow, decode-token KV written to `kv_mem` and attended — each row's step-k token bit-exact vs a standalone PE_M=1 model decoding that seq alone N steps)** | a resident dense DRAFT model (K_eff↑, needs weights); real-checkpoint (P1.1, GPU) |
| Memory | DDR5/NVMe/USB-C **stubbed** (TB) | licensed **PHY IP** integrated + signed off |
| Verification | bounded BMC (+ clk_throttle) + directed TBs at slice; **verilator line/toggle/branch coverage** (`make coverage`, 87.8% line merged) | coverage *closure*, constrained-random regression, gate-level sim, k-induction, production-width formal |
| Reliability | none | ECC, error recovery, CDC sign-off, reset/init hardening, DFT/scan |
| Physical | slice-scale yosys estimates + real sky130 realizability (placement, timing met) | full synth + **FPGA** P&R + timing closure via the vendor flow → **bitstream** (rungs ①②; ASIC/tapeout is the rung-③ volume endgame, not out of scope — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)) |
| Software | weight-pack tools (ckpt_pack/flash_layout); **host scaffold built** — OpenAI-compatible server + device protocol + **real GLM BPE tokenizer** + chat template + sampling ([`host/`](../host/README.md)) | production host **driver** (real USB backend), runtime/scheduler, quant-layout pipeline |
| Manufacturing | — | PCB, BOM, assembly, qualification |

---

## The #1 product gate (do this FIRST)

**Real-checkpoint full-model fidelity.** Until the actual GLM-5.2-FP8 weights produce the actual
model's next tokens through our datapath, there is no product. The bridge is built
(`tools/ckpt_pack.py` + `docs/BIT_ACCURACY.md` §C). Execute it on a GPU host:
1. Download `zai-org/GLM-5.2-FP8`; run a transformers/vLLM reference forward on a prompt → golden
   next-token logits/argmax.
2. Run our **bit-accurate software model** (`tools/glm_fp8_ref.py`, scaled to full config) on the
   same weights + prompt → argmax.
3. Assert token match over a corpus. Any divergence = a per-Linear scale-orientation / bf16-tail /
   RoPE/KV/MoE plumbing bug to fix in the RTL contract.

This is gating: it validates the *assembly* of hundreds of operators on *trained* weights — the
one thing the slice cannot.

---

## Product phases (main)

### P1 — Full-scale + real-model correctness  *(gates everything)*
- P1.1 Real-checkpoint validation (above). **Blocking.**
- P1.2 Scale the RTL/params to the full config; verify via the bit-accurate software model + a
  full-config RTL elaboration (synth-only where sim is intractable). **Buildability fix (done):** the
  core GEMM `glm_matmul_fp8` dequant was a fully-unrolled block-scale fold — O(NB²) area
  (NB=ceil(KMAX/BLK); at the product KMAX=16384 → NB=128, i.e. 256 FP pipes + ~40k delay FFs *per PE*),
  so lint-"elaborates-clean" ≠ buildable. Rewrote it as a **sequential** fold (one fp32_mul + one
  fp32_add reused over all blocks), **O(NB²)→O(1)** FP pipes, **bit-exact** (matmul TB 224/224 at NB=2
  and NB=16; `make bitacc` 14/14 + argmax 28/28; slice-model argmax unchanged). yosys now elaborates
  `glm_matmul_fp8` at KMAX=16384 (was an elaboration hang) — the product config is now buildable.
- P1.3 Close the prototype correctness gaps for product: **per-position causal KV** (replace the
  PE_M shared-pos decode-batch regime with a real per-row position/KV). **Done so far:** per-row
  position/extent threaded model→decoder→mla; the **KV pager storage side** carries `NSEQ`
  independent ring windows (per-seq counter/window/eviction + `flash_seq` cold keying; NSEQ=1
  byte-identical, verified by directed multi-seq TB + BMC/k-induction); and **multi-sequence
  batched attention (`PER_ROW_SEQ`) is now real end-to-end through the FULL MODEL** — mla replaces
  the shared-seq union-skip with a per-row-slot union tagged by `union_seq`, emits `kc_seq` to
  route each fetch to its sequence's window, and `PER_ROW_SEQ`/`seq_vec`/`kc_seq`/`SWIN` thread
  model→decoder→mla. Proven: mla multi-seq TB (32, incl. weight-share), and the **full
  glm_model_fp8 batches 2 DIFFERENT sequences** (per-row argmax/logits bit-exact vs per-seq PE_M=1
  goldens; query-side weight stream shared, ~41% fewer attn-weight beats than two separate runs) —
  now BOTH the DENSE and the **SPARSE** regime (S>TOPK_ATTN — the real long-context/DSA regime;
  unblocked by the `dsa_indexer` `LANES[IDXW:0]`-truncation fix that had frozen the sparse group
  loop for S_MAX≤4); PER_ROW_SEQ=0 byte-identical throughout. And a **batched multi-sequence TOP
  exists and is productionizing** — `glm_fp8_soc_ms` runs `glm_model_fp8` at PE_M=B with a REAL
  `NSEQ`-window `kv_cache_pager` **and the `expert_cache_pf` routed-expert cache** (batched MoE
  union-skip dedups experts across sequences → the cache sees fewer distinct NVMe fetches than B
  separate decodes): a host FSM prefills B sequences into their own windows (`append_seq`), runs
  ONE batched forward (row r → sequence r via `seq_vec`; `kc_seq` → `gather_seq`), and commits B
  next tokens — each row bit-exact vs a per-seq PE_M=1 model, query-side weights shared, dense +
  sparse (`glm_fp8_soc_ms_tb`, 3 cases). And **`DSA_REAL_IDX=1` (real query-dependent IndexShare)
  now works under multi-seq** — the DSA index pre-fetch carries a PER-SEQUENCE `kidx_buf[seq]` so
  each row scores its top-K against ITS OWN sequence's key vectors (`mla_attn_fp8_multiseq_dsareal_tb`,
  4 sparse cases, per-row bit-exact; byte-identical for `DSA_REAL_IDX=0` and single-seq).  And a
  **REAL per-layer KV store now lives in the top** — `glm_fp8_soc_ms` OWNS the KV data in `kv_mem`
  (L*NSEQ windows x KV_RESIDENT positions), written by the host prefill/decap per (seq, layer, pos)
  and read combinationally by the model (window = db_layer*NSEQ+kc_seq), replacing the TB stub
  (verified: all 3 SoC cases pass with kv_mem serving, dense + sparse).  **batched_moe full
  B-coverage: DONE** (`make bcov` — B∈{1,2,3,5,8} × routing patterns {same,distinct,random,overlap},
  each re-proving batched_moe(PE_M=B) == B independent PE_M=1 expert runs BIT-EXACT with every union
  expert fetched once).  **Real draft chaining: DONE** (`spec_chain_top` — the MTP head runs
  recurrently on its chain hidden-state `h_mtp` to mint K drafts, then a PE_M=K+1 batched-verify in
  one weight-load commits the accepted prefix; spec==greedy).  **scale-up VERIFIED at B=4**
  (`glm_model_fp8_multiseq4_tb`:
  4 different sequences batched in one forward, all 4 rows per-row bit-exact vs per-seq PE_M=1, dense
  AND sparse, ~52% fewer attn-weight beats than 4 separate decodes).  **Multi-step continuous-batching
  DECODE LOOP: DONE** — `glm_fp8_soc_ms` at `N_STEPS>1` turns one `start` into an N-token decode: the
  host FSM runs the batched forward, streams the B argmax via `tok_valid`, writes each decode token's
  latent into `kv_mem` at its growing position (`s_len_r + dec_step`) for every layer, feeds the
  argmax back as the next step's input, and advances `pos`/extent — looping RUN→DECAP→RUN until N
  tokens are committed per sequence.  `glm_fp8_soc_ms_loop_tb` (`make` target `glm_fp8_soc_ms(decode-loop)`)
  pins it: for dense, mixed, and sparse regimes, each row's step-k token is BIT-EXACT to a standalone
  PE_M=1 model decoding that sequence alone for N steps (same feedback + growing extent), and the
  step-1 forward attends the step-0 decode key it wrote to `kv_mem` (a wrong feedback token,
  position, or decode-KV read would diverge from the reference).  N_STEPS=1 is byte-identical to the
  single-step top.  **Remains (beyond RTL):** a RESIDENT DENSE DRAFT model for higher accept rate
  (K_eff 3–5 vs self-draft's ~1.7–2.2 — needs a trained small draft's weights); real-checkpoint
  validation (the standing P1.1 gate — needs a GPU host).

### P2 — Productize the RTL (robustness)
- P2.1 ECC on DDR5 + NVMe read path; error detection / correction / retry / recovery paths.
- P2.2 Full CDC sign-off across USB / memory / compute clock domains; reset/init/boot-load hardening.
- P2.3 Reliability (FPGA path): the built ECC (SECDED scrub) + BRAM-ECC + CDC/reset hardening
  carry over; MBIST maps to a BRAM self-test. ASIC-specific DFT (scan-chain insertion, boundary
  scan) is **not needed on the FPGA rungs (①②)** — the vendor JTAG/config handles device test — and
  is **deferred to the rung-③ ASIC endgame**, where it becomes required (see
  [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)).
- P2.4 Power: real ICG clock-gating cells, power domains, DVFS hooks, thermal budget.
- P2.5 Verification closure: functional + code coverage targets, constrained-random regression,
  gate-level (post-synth) sim, production-width controller formal + k-induction for unboundedness.

### P3 — Vendor IP + physical implementation
- P3.1 License + integrate the PHYs: the DDR controller+PHY (**DDR4 on rung ①, DDR5 multi-channel or
  HBM on rung ②** — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)), PCIe/NVMe host controller, USB-C device.
- P3.2 Target — **FPGA-card product for the near/mid term** (rungs ①②, the committed path — see
  [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)); **ASIC is the rung-③ volume endgame, not out of scope.**
  A data-center-class FPGA + on-board DDR + NVMe (M.2/PCIe) runs the real model streamed; bitstream via
  the vendor flow. This is a **staged ladder**: rung ① (low-end FPGA, ~5–8 tok/s [EST]) proves it
  cheap, rung ② (custom mid-FPGA board, DDR5/HBM, ~15–40 tok/s [EST]) is the funded interactive product.
  - **Why FPGA first, ASIC at volume:** the workload is **memory-bandwidth-bound** — performance is set
    by memory bandwidth, which is set by the chip's IO pins + hard PHYs, which is set by budget. The FPGA
    rungs prove the RTL on real silicon and reach product-market fit *without* a multi-million NRE bet.
    An **ASIC is exactly what breaks the FPGA's IO/PHY ceiling**: it integrates **HBM stacks +
    many-channel controllers + near-memory FP8 compute** that no FPGA package offers, at **~TB/s** with
    **lower $/seat + lower power** once the NRE amortizes over manufacturing volume (rung ③). So the
    earlier "ASIC out of scope" — argued from *"compute-bound → ASIC's compute edge is wasted"* — is
    **superseded**: the real bottleneck is bandwidth, and ASIC is precisely how a bandwidth-bound
    product scales past the FPGA package. Sequence: **FPGA (①②) to prove + fund → ASIC (③) when volume
    justifies the NRE and demands lower $/seat + higher tok/s + lower power.** Not now (no volume, no
    capital); the endgame later, for cost-down + performance + power at volume.
- P3.3 Full-scale STA (SDC), power sign-off, signal/power integrity.

### P4 — System, software, manufacturing
- P4.1 PCB: multi-layer controlled-impedance board (FPGA + DDR5/HBM + NVMe via M.2/PCIe + USB-C), BOM,
  assembly. (This is the **rung ② custom product board**; the Tang Mega 138K Pro dev board — 32 MB onboard
  Flash + 1 GB soldered DDR3 — is bring-up/reduced-demo only, not the shipping hardware.)
- P4.2 Software stack: USB-C host driver, tokenizer, the checkpoint→NVMe quant-layout pipeline
  (productionize `ckpt_pack.py`/`flash_layout.py`), inference runtime + continuous-batch scheduler.
- P4.3 Qualification: reliability (temp/voltage/aging), compliance (USB-C, EMI), yield/binning.

---

## What stays / what changes from the prototype

- **Keep (the core IP):** the FP8 datapath, MLA/DSA/MoE, the memory-system controllers
  (expert_cache_pf, kv_cache_pager, ddr5_xbar, flash_xbar, weight_loader, boot_loader), the
  batching stack (PE_M batch-widening **complete on all four FP8 wrappers** — swiglu/router/mla/mtp
  — + model + batched_moe + spec_batched_top, and the PE_M>1 grouped MoE now **union-skips** to
  fetch only the experts any row selected, byte-identical), the optimizations
  (BFP accumulator, parallel indexer, decompressor), the formal harnesses, the bit-accuracy kit.
  These are the hard, validated parts — they carry forward.
  *(One-time naming note: `flash_xbar` — like `FLASH_LAT`, `flash_req`, `flash_seq` — is a
  **committed RTL identifier**, kept as-is. It is the medium-agnostic storage-read fabric
  (address → weight bytes, with read-request issue + latency-hiding). In the product its
  NAND-specific backend is a labeled placeholder swapped for an **NVMe/PCIe host controller**; the
  crossbar abstraction and everything above it — compute die, weight_loader, expert_cache_pf,
  kv_cache_pager — are unchanged.)*
- **Change (the mindset):** every prototype "demonstrates the mechanism / honest gap / TB-driven"
  becomes a closed, full-scale, covered, signed-off product feature. Correctness is now *the real
  model on real weights*, not a slice.
- **Out of pure RTL (but on the product critical path):** PHY IP, physical implementation,
  software, PCB, manufacturing. These dominate product cost/time and are mostly vendor/EDA/board
  work, not algorithm design.

## Immediate next step (product)

**P1.1**: run the real-checkpoint validation procedure (needs a GPU host) — the standing gate. The
per-position causal-KV gap (the prototype's PE_M shared-pos limitation) is now **largely closed** on
`main`: per-row position/extent, multi-sequence batched attention (`PER_ROW_SEQ`), and the
multi-step continuous-batching decode loop (`glm_fp8_soc_ms` `N_STEPS>1`) make batched decode
position-accurate and per-row bit-exact; the remaining P1.3 item is a resident dense DRAFT model
(needs weights). Everything else (P2–P4) sequences behind a green P1.
