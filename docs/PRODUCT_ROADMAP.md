# Product Roadmap — GLM-5.2-FP8 accelerator (product, not research)

The `prototype` branch (frozen at `47fb7f8`) holds the **research prototype**: the full FP8
datapath + memory system + ultra-perf batching stack, **bit-exact and mechanism-proven at a
small-but-faithful slice**, with honest gaps documented. It answers *"does the architecture work,
and how fast can it go?"* — **yes**, and the levers are measured.

`main` now develops the **product**: a manufacturable accelerator that **runs the published
`zai-org/GLM-5.2-FP8` reliably**. The mindset shifts from *demonstrate + measure a mechanism* to
*run the real model correctly, at full scale, robustly, and ship it.*

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
| Memory | DDR5/Flash/USB-C **stubbed** (TB) | licensed **PHY IP** integrated + signed off |
| Verification | bounded BMC (+ clk_throttle) + directed TBs at slice; **verilator line/toggle/branch coverage** (`make coverage`, 87.8% line merged) | coverage *closure*, constrained-random regression, gate-level sim, k-induction, production-width formal |
| Reliability | none | ECC, error recovery, CDC sign-off, reset/init hardening, DFT/scan |
| Physical | slice-scale yosys estimates + real sky130 realizability (placement, timing met) | full synth + **FPGA** P&R + timing closure via the vendor flow → **bitstream** (ASIC/tapeout out of scope) |
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
  full-config RTL elaboration (synth-only where sim is intractable).
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
  union-skip dedups experts across sequences → the cache sees fewer distinct Flash fetches than B
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
- P2.1 ECC on DDR5 + Flash; error detection / correction / retry / recovery paths.
- P2.2 Full CDC sign-off across USB / memory / compute clock domains; reset/init/boot-load hardening.
- P2.3 Reliability (FPGA path): the built ECC (SECDED scrub) + BRAM-ECC + CDC/reset hardening
  carry over; MBIST maps to a BRAM self-test. ASIC-specific DFT (scan-chain insertion, boundary
  scan) is **out of scope** — on FPGA the vendor JTAG/config handles device test.
- P2.4 Power: real ICG clock-gating cells, power domains, DVFS hooks, thermal budget.
- P2.5 Verification closure: functional + code coverage targets, constrained-random regression,
  gate-level (post-synth) sim, production-width controller formal + k-induction for unboundedness.

### P3 — Vendor IP + physical implementation
- P3.1 License + integrate the PHYs: DDR5 multi-channel controller+PHY, NVMe/Flash, USB-C device.
- P3.2 Target — **FPGA-card product** (the committed path; **ASIC is out of scope**). A
  data-center-class FPGA + on-board DDR + NVMe/Flash runs the real model streamed; bitstream via
  the vendor flow.
  - **Why not ASIC:** the workload is **Flash-bandwidth-bound**, so the die already sits
    ~75–80% idle behind Flash. An ASIC's headline advantage (faster / denser compute) is therefore
    largely *wasted* here — it would only buy power/efficiency and unit-cost-at-volume, at
    multi-million NRE + months–years and no do-overs. For a bandwidth-bound design the FPGA-card
    (compute is cheap; the memory interface is what matters) is not a stepping stone but a
    legitimate end product. ASIC is only revisited if volume/power economics ever justify the NRE.
- P3.3 Full-scale STA (SDC), power sign-off, signal/power integrity.

### P4 — System, software, manufacturing
- P4.1 PCB: multi-layer controlled-impedance board (FPGA + DDR5 + Flash + USB-C), BOM, assembly.
- P4.2 Software stack: USB-C host driver, tokenizer, the checkpoint→Flash quant-layout pipeline
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
- **Change (the mindset):** every prototype "demonstrates the mechanism / honest gap / TB-driven"
  becomes a closed, full-scale, covered, signed-off product feature. Correctness is now *the real
  model on real weights*, not a slice.
- **Out of pure RTL (but on the product critical path):** PHY IP, physical implementation,
  software, PCB, manufacturing. These dominate product cost/time and are mostly vendor/EDA/board
  work, not algorithm design.

## Immediate next step (product)

**P1.1 + P1.3a**: run the real-checkpoint validation procedure (needs a GPU host) and, in parallel
on `main`, start closing the per-position causal-KV gap (the prototype's PE_M shared-pos
limitation) so batched decode is position-accurate for product. Everything else (P2–P4) sequences
behind a green P1.
