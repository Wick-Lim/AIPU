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
| Batching/KV | PE_M batch on all 4 wrappers; **per-row position/extent now threaded model→decoder→mla** (byte-identical); grouped MoE **union-skips**; ÷K TB-driven; batched_moe B=4 | per-row **KV cache** (multi-seq) at production widths; real draft chaining; full B coverage |
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
  PE_M shared-pos decode-batch regime with a real per-row position/KV), real draft chaining for
  batched-verify, full B-coverage for batched_moe.

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
