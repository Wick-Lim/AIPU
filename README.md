# AIPU — a GLM-5.2-FP8 inference accelerator in Verilog

[![Slides — AIPU Accelerator: Formally Verified FP8 Architecture](https://img.shields.io/badge/%F0%9F%93%8A%20Slides-AIPU%20Accelerator%3A%20Formally%20Verified%20FP8%20Architecture-F4B400?logo=googleslides&logoColor=white&labelColor=555)](https://docs.google.com/presentation/d/1EMqJOJCNTBaCVEf5EfYI_CymVg--K8ezVyCJcr7ovys/present)

> **📊 Presentation (paper slides): [*AIPU Accelerator: Formally Verified FP8 Architecture*](https://docs.google.com/presentation/d/1EMqJOJCNTBaCVEf5EfYI_CymVg--K8ezVyCJcr7ovys/present)** — the design, verification, and results as a deck (Google Slides). *(Click the badge above, or the link.)*

A synthesizable Verilog accelerator with one goal: **run one real model well —
[`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8)**, the published FP8
checkpoint of GLM-5.2 (`GlmMoeDsaForCausalLM`), a 753B-param MoE (~40B active/token) in
native **FP8 E4M3**. The complete GLM-5.2 operator datapath is built in FP8, verified
against independent fp64/fp8 goldens at a small-but-faithful slice, and wrapped by the
single-module memory system (multi-channel DDR5 + NVMe expert cache + weight/boot loaders +
multi-clock CDC) that streams the real 753B model — with the memory controllers
bounded-model-checked.

> **The product is a LOCAL, single-user box that runs with the ethernet unplugged.** One box, one
> user, running the full 753B model **fully offline / air-gapped** — a frontier model finally usable
> *on the work, and in the disconnected places (SCIFs, OT/critical-infra, field/edge), you're currently
> locked out of* — and you own it outright. Nothing leaves because there is **no path out**: the audit
> is literally *"does it still work with the ethernet cable unplugged?" — yes.* That non-egress is the
> **proof, not the pitch**, and it ends the "secured cloud" debate — in-VPC, zero-retention, and TEE
> deployments all still need a connection and fail the unplugged test. Offline *alone* is table-stakes
> for any local box; the moat is the **combination — offline + full frontier (753B) + appliance/seat
> price**. (Provisioned once with the 753 GB weights — itself doable offline — then fully disconnected;
> model updates are physical re-provisioning.) No per-token API fees, and no vendor that can rate-limit,
> deprecate, or cut you off. The number that matters is **single-user
> interactive throughput**, and it's set by the hardware rung you build on: **~5–8 tok/s on the
> prove-it FPGA today → ~15–40 on the funded custom board [EST]** (rung ③ silicon reaches ~40+ at volume)
> after stacking the faithful levers — the old flat ~25–40 was the funded rung's number, not a
> near-term-cheap one; see the 3-rung [`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md). The design
> is deliberately NVMe/PCIe-bandwidth-bound to keep it cheap. Where these docs mention *aggregate /
> datacenter batching* (per-user ~0.14 tok/s at B≈256), that is a **secondary analysis of a
> different deployment**, not this appliance — see [`docs/USBC_PRODUCT_PLAN.md`](docs/USBC_PRODUCT_PLAN.md).

> **Naming.** **AIPU** (AI Processing Unit, repo [`Wick-Lim/AIPU`](https://github.com/Wick-Lim/AIPU))
> is the whole accelerator. The project was formerly *TPU*; the classic *5-stage scalar TPU
> core* (*"TPU v2.0"*, `tpu_*` modules) was a separate legacy design and has been **removed**
> from the repo — it was never on the GLM product path (see git history if you need it).
>
> **Branches:** `prototype` (frozen at `47fb7f8`) = the research prototype (full FP8 datapath +
> memory system + batching stack, bit-exact at the slice); **`main` develops exactly one thing —
> the GLM-5.2-FP8 accelerator at rung ① (FPGA prove-it)**, the offline single-user box, with the
> near-term goal being the working FPGA demo. The full product (rungs ②③) is the roadmap, not
> main's current code ([`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md),
> [`NEXT_STEPS_PLAN.md`](NEXT_STEPS_PLAN.md)).

> **Local-device retarget (Q4_K).** For a cost-constrained local appliance, `main` now develops the
> **Q4_K local-inference track**: the target weight store is the published
> [`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.2-GGUF) — **467 GB, ~38%
> smaller than the 753 GB FP8 checkpoint** (the hot-set / routed-expert bytes scale down
> ~proportionally, and the BOM is memory-dominated). The moat moves — still verifiable — from *"bit-exact
> to the published FP8 safetensors"* to **"bit-exact to the published UD-Q4_K_XL GGUF (no re-quantization;
> generally lossless per Unsloth)"**, a file anyone can download and check. The **FP8** datacenter-native
> baseline is preserved on branch **`fp8`** + tag **`fp8-verified-baseline`** — every FP8 evidence row
> below still holds there. Numerics are bit-exact to ggml; tok/s stays **[EST]**. See
> [`docs/Q4K_RETARGET.md`](docs/Q4K_RETARGET.md).

---

## What's proven — measured on real artifacts, not claimed

The project's defining property is verification discipline: nothing below is a model
estimate — each row is a real measurement against a real artifact (the published checkpoint,
exhaustive enumeration, a real PDK, or a formal solver).

| Evidence | Against | Result |
|---|---|---|
| **Operator bit-accuracy vs the real checkpoint** | published `GLM-5.2-FP8` safetensors (`kv_a_proj` F8_E4M3) | **9216/9216 = 100%** bf16-exact, **argmax 16/16** ([`REAL_CKPT_VALIDATION.md`](docs/REAL_CKPT_VALIDATION.md)) |
| **Real MoE experts on GPU** | real expert weights, Modal T4 | **argmax-preserving** (tier1) |
| **Modal partial-F1** — assembled real-weight FFN | first **6 real** `GLM-5.2-FP8` layers (dense 0–2 + MoE 3–5), our FP8 Linears, vs HF | **argmax 6/6**, worst `max_abs` **0.0015** — numerically faithful ([`REAL_CKPT_VALIDATION.md`](docs/REAL_CKPT_VALIDATION.md)) |
| **FP8 E4M3 arithmetic** | fp64, **exhaustive** | **ALL 66069** (256 decodes + all 256×256 multiplies) |
| **Operators at REAL GLM-5.2 dims** | fp64 goldens | GEMM **K=6144**, router **256/top-8**, SwiGLU **2048**, MLA real geo — all bit-exact ([`SCALE_FUNCTIONAL.md`](docs/SCALE_FUNCTIONAL.md)) |
| **Full FP8 forward pass** (slice) | fp8 golden | **next-token argmax == golden** |
| **F0 assembled-model cross-check** — independent numpy ref | RTL **fp64 golden**, every per-layer stage | **bit-exact** (256/256 logits + 128/128 @ 2×; argmax numpy == golden == DUT) — validates the assembly, torch/HF-free (`tools/glm_full_ref_np.py`) |
| **Real sky130 place-and-route** (`glm_matmul_fp8`) | SkyWater sky130 PDK, OpenROAD | synth→floorplan→**legalized placement**, **357,320 µm²**, post-placement timing **MET** (+15.89 ns) ([`PHYSICAL_SKY130.md`](docs/PHYSICAL_SKY130.md)) |
| **Memory-system controllers** | z3 | **BMC** (6 controllers) + **unbounded k-induction** ([`FORMAL.md`](docs/FORMAL.md)) |
| **Cycle-accurate memory-stall mechanism** | assembled system, **real RTL cycles** | exposed stall is exactly **`3·FLASH_LAT + 9`**; faithful `cyc_per_tok` **grows** with storage-read latency (flat 7947 → **8607** @ FLASH_LAT=256) ([`CYCLE_EMULATION.md`](docs/CYCLE_EMULATION.md)) |
| **PE_M batch path** (0 extra weight BW) — **4/4 wrappers** | per-row single-token refs | swiglu **513** / router **192** / mla **6** / mtp **44** — bit-exact + weight-share, *"B rows == 1 fetch stream"* |
| **Multi-sequence batched attention** — each PE_M row a *different* sequence (a batched-serving capability of the silicon; the **personal box runs B=1**) | per-seq PE_M=1 goldens | per-row argmax/logits **bit-exact** at B=2 (~41% fewer attn-weight beats than 2 runs) and **B=4** (~52%), dense + sparse; `PER_ROW_SEQ=0` byte-identical ([`PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)) |
| **Truncated full-model token chain** (real weights, DSA threaded, incl. the dense→MoE seam) | fp32-accumulate ref, real GLM prompt | **argmax match** (real 256-expert route; "The capital of France is" → **20259 == 20259**), top-8 preserved — the DSA-IndexShare + fused-expert blockers **retired** ([`REAL_CKPT_VALIDATION.md`](docs/REAL_CKPT_VALIDATION.md)) |
| **Full 753B config elaboration** | verilator, true dims (6144/78/154880/256-expert) | **0 errors** — parameterization threads clean at real scale; full-config lints cleared (SELRANGE 4122→0, byte-identical) ([`FULL_CONFIG_ELAB.md`](docs/FULL_CONFIG_ELAB.md)) |

**Modeled, not silicon — flagged [EST].** All throughput/energy figures come from a
bandwidth-roofline model (`tokens/s ≈ NVMe_BW / [(1−h)·footprint] · K`), **not** from a routed
netlist or silicon. The tok/s is **rung-dependent** — bandwidth is set by the chip's IO pins + PHYs,
which is set by budget ([`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)): **~5–8 tok/s on the
prove-it FPGA (rung ①), ~15–40 on the funded custom board (rung ②), ~40+ at volume (rung ③)** [EST],
with **~9 → ~3 J/token** [EST] after stacking the NVMe-bandwidth levers (the old flat ~25–40 was the
funded-rung ceiling, not a near-term-cheap number). Read them as
an optimistic ceiling ([`ULTRA_PERF.md`](docs/ULTRA_PERF.md),
[`IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md)). What *is* measured — the proven row above — is
the roofline's underlying **memory-stall mechanism**, now validated on real RTL cycles (stall exactly
`3·FLASH_LAT + 9`, faithful `cyc_per_tok` grows with storage-read latency); the absolute tok/s stays [EST]
([`CYCLE_EMULATION.md`](docs/CYCLE_EMULATION.md)).

**Fidelity — A-ish (firmer).** The model *assembly* is bit-exact against an independent numpy reference
(F0); the real-weight FFN of the first six layers is faithful on GPU (Modal partial-F1, argmax 6/6);
and — new — a **truncated full-model token chain on real weights** now passes: running the truncated
`GlmMoeDsa` model's own `forward` threads the DSA index itself (retiring the **DSA-IndexShare** blocker),
and patching the fused `GlmMoeDsaNaiveMoe` routes the real 256 experts through our contract (retiring the
**fused-expert** blocker) — argmax-identical through the **dense→MoE seam** on a real tokenized prompt,
top-8 preserved. Full A now needs deeper depth / the full 753B run (multi-GPU), not a plumbing fix
([`REAL_CKPT_VALIDATION.md`](docs/REAL_CKPT_VALIDATION.md)).

**Out of scope** (vendor IP / hardware / resource-gated): DDR5/NVMe (PCIe)/USB-C **PHYs** (TB-stubbed),
**full-chip FPGA P&R + board bring-up** (D0.2/D1 — needs Gowin + a board; the near-term product is an
FPGA card, with ASIC/tapeout **sequenced later as the rung ③ volume endgame** — not abandoned, see
[`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)), and a **full-model 8×H200 GPU validation** (substituted by the
CPU-bit-exact + T4 + truncated-full-model evidence above). *(The tokenizer + host software are now
built — see [`host/`](host/README.md).)*

---

## The target: `zai-org/GLM-5.2-FP8`

The checkpoint's `config.json` *quantization_config* is what drives the hardware:

| Field | Value | Hardware consequence |
|---|---|---|
| `quant_method` / `fmt` | **fp8 / e4m3** | a 4-bit-exponent / 3-bit-mantissa float multiply (4×4 mantissa multiply) |
| `weight_block_size` | **[128, 128]** | one bf16 dequant scale per 128×128 weight block (block-scaled accumulation) |
| `activation_scheme` | **dynamic** | activations quantized to E4M3 at runtime (per-token pow2 scale, on-chip) |
| `modules_to_not_convert` | norms / router / embed / lm_head | those stay **bf16** — our "bf16 tail" *matches* the checkpoint, not an approximation |

Architecture (the slice preserves every ratio): hidden 6144, 78 layers
(`first_k_dense_replace=3`), 64 heads (`head_dim=192`), **MLA** latent attention
(`qk_nope 192 + qk_rope 64`, `v 256`, `kv_lora 512`, **`q_lora 2048`**), **MoE** 256 experts
top-8 + 1 shared (`moe_intermediate 2048`), dense `intermediate 12288`, **DSA** sparse
attention (`index_topk 2048`), vocab 154880, 1M context, `rope_theta 8e6` interleaved,
RMSNorm `eps 1e-5`, MTP (`num_nextn_predict_layers 1`).

> `q_lora 2048` / `kv_lora 512` are **confirmed against the real safetensors**
> (`q_a_proj.weight [2048,6144]`) — an earlier DeepSeek-standard guess of `q_lora 1536` was
> corrected. See [`REAL_CKPT_VALIDATION.md`](docs/REAL_CKPT_VALIDATION.md).

---

## How it works

```
  1 TB NVMe   ──►  flash_xbar   ──►  64 GB DDR5   ──►  ddr5_xbar  ──►   FP8 compute die   ──►  token
 (753B FP8     N-channel banked   expert cache      N-channel      (MLA + DSA + MoE, native
  weights)     + deep queue       (LRU+freq+pf)     banked read     FP8 E4M3, bf16 tail)
```

The workload is **NVMe/PCIe-bandwidth-bound**, so the system is built around streaming MoE experts
from the NVMe SSD through a DDR working cache into a mostly-idle FP8 die (tier: **NVMe** bulk/slow →
**DDR** hot set/fast → die). *(The **64 GB DDR5** in the diagram is the funded rung ② spec; the DDR tier
is rung-dependent — DDR4 on the prove-it FPGA, DDR5/HBM on the custom board — see
[`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md).)* `flash_xbar` is the committed name of that storage-read fabric — a
medium-agnostic address→weight-bytes crossbar with deep-queue latency-hiding; in the product it fronts
an **NVMe/PCIe host-controller backend** (the NAND-specific backend is the swapped part, not the
abstraction). Its N-channel banking maps to **PCIe lanes / multiple NVMe drives** — order-of-magnitude
~3.5 GB/s per PCIe Gen3 ×4 drive, ~7 GB/s Gen4, scaling with lanes/drives [EST] — so the "more devices
→ more bandwidth" story survives the medium swap. Every FP8 **weight** matmul
is E4M3 (4×4 mantissa multiply → block accumulate → per-[128,128]-block scale → bf16); norms,
softmax, rope, residual and the activation×activation attention matmuls stay bf16.

**Verification methodology.** Every GLM-5.2 unit is checked against an **independent fp64 (or
faithful-fp8) X-aware golden**; on success a TB prints `ALL N TESTS PASSED`, on any mismatch it
prints the failing case and `$fatal`s. Regression is **byte-identical**; the memory controllers
add **bounded model checking** (yosys-smtbmc + z3), some lifted to **unbounded k-induction**.

---

## Detailed status

### FP8 E4M3 datapath — COMPLETE

Full FP8 forward pass runs and predicts the correct next token.

| Unit | What is FP8 | Verification |
|---|---|---|
| `fp8_e4m3.vh` | E4M3 decode / encode-RNE+saturate / 4×4 mantissa multiply | **exhaustive** — ALL 66069 vs fp64 |
| `glm_matmul_fp8.v` | block-scaled FP8 GEMM ([128,128], dynamic act); **BFP fixed-point accumulator** (bit-exact at ACC_FRAC=18, **−87.6% cells** vs fp32-accumulate) | 224 tests; real **K=6144** bit-exact |
| `swiglu_expert_fp8.v` | gate/up/down GEMMs FP8 on **one shared engine** (L1: `up_pass` arbiter, `u_mm_u` removed → 6→4 GEMM engines/block), bf16 silu tail; **PE_M-batched** | 1024 + PE_M 513 (0 extra weight BW), byte-identical |
| `mla_attn_fp8.v` | 7 weight projections FP8, bf16 attn/rope/norm/softmax/dsa; **PE_M-batched, per-row pos/s_len** | slice 7; **real-dim (Q2048/KV512/…) worst rel 5.48e-4**; PE_M 6 |
| `moe_router_fp8.v` | gate GEMV FP8, bf16 sigmoid/topk/renorm; **PE_M-batched** | 185 + real 256/top-8 + PE_M 192 |
| `glm_decoder_block_fp8.v` | one full FP8 decoder layer; **grouped MoE + union-skip** (PE_M>1 fetches only the *union* of selected experts) | 9 tests; union-skip byte-identical (*"evaluated 3 == distinct-selected, skipped 5 of 8, bit-exact"*) |
| **`glm_model_fp8.v`** | **full FP8 forward pass** | **next-token argmax == fp8 golden** |
| `mtp_head_fp8.v` | FP8 multi-token-prediction (t+2) head; **PE_M-batched** | 6 + PE_M 44 (all weight ports: B rows == 1 fetch) |

**Batching is complete — 4/4 FP8 wrappers** (swiglu / router / mla / mtp) carry a `PE_M` param + per-row
buffers, verified bit-exact and weight-sharing (*"B rows == 1 fetch stream"*). On top of that the PE_M>1
**grouped MoE** in `glm_decoder_block_fp8` fetches only the **union** of selected experts (a `T_ESCAN`
scan + combinational `any_has`), byte-identical to the all-expert path — up to **~32× fewer NVMe expert
fetches** at small batch on the real 256-expert config (≈ no benefit at B≈256, where the union ≈ all).

**Multi-sequence batching is real end-to-end** (beyond same-sequence decode-batching) — a proof of what the
same silicon *could* serve in the **non-target multi-user/aggregate regime** (the personal box itself runs
B=1; see the identity note above), not the appliance's own path. Each PE_M row can now
be a **different sequence**: `mla_attn_fp8` carries a `PER_ROW_SEQ` mode (per-row-slot union + `kc_seq`
routing — each row attends its *own* sequence's KV while the query-side weight/projection fetch stays shared,
the batching bandwidth win), threaded model→decoder→mla via `seq_vec`/`kc_seq`. Proven full-model:
`glm_model_fp8` batches 2 different sequences (per-row argmax/logits bit-exact vs per-seq PE_M=1, ~41% fewer
attn-weight beats), scaled to **B=4** (~52% fewer beats), both dense **and** sparse; byte-identical at
`PER_ROW_SEQ=0`. A batched multi-seq **SoC top `glm_fp8_soc_ms`** wraps the PE_M=B model with a real
`NSEQ`-window `kv_cache_pager` + `expert_cache_pf` + a host FSM (prefill B seqs → 1 forward → commit B
tokens), holding the per-layer KV in a **real store `kv_mem`** owned by the top. It runs a **multi-step
continuous-batching decode loop** (`N_STEPS>1`: one `start` decodes N tokens/seq, argmax fed back, each
decode token's KV written to `kv_mem` at the growing position and attended — each row's step-k token
bit-exact vs a standalone PE_M=1 model; `N_STEPS=1` byte-identical). `DSA_REAL_IDX=1` (query-dependent
IndexShare) works under multi-seq via a per-sequence `kidx_buf` pre-fetch, and `batched_moe` has full
B-coverage (`make bcov`, B∈{1,2,3,5,8} × routing patterns, each re-proving batched == B independent PE_M=1
runs bit-exact with every union expert fetched once).

### Single-module system (real-753B memory/streaming hardware) — BUILT

| Unit | Role | Verification |
|---|---|---|
| `expert_cache_pf.v` | DDR5 routed-expert cache: LRU + freq + prefetch | 623 tests; **BMC-proven** |
| `kv_cache_pager.v` | MLA latent-KV ring + DSA-gather + NVMe overflow; **`NSEQ` independent per-seq ring windows** | 73 tests (+ NSEQ>1 multi-seq); **BMC-proven** |
| `ddr5_xbar.v` | N-channel banked DDR5 read fabric (~N× BW) | 3073 tests (7.93× @8ch); **BMC + k-induction** |
| `flash_xbar.v` | N-channel banked **storage-read** fabric (deep queue hides read latency); fronts the **NVMe/PCIe** backend (banking = PCIe lanes / drives) | 2049 tests (7.99× latency-hide); **BMC-proven** |
| `weight_loader.v` | checkpoint FP8 + block-scale → matmul pull DMA | 240 tests (loader-fed == direct-fed) |
| `boot_loader.v` | power-up NVMe→DDR5 model-load sequencer | 9240 tests; **BMC-proven** |
| `spec_decode_seq/_top.v` | MTP speculative-decode loop (K>1 draft) | 621/1379/19 tests; **BMC-proven** |
| **`glm_fp8_system_cdc.v`** | production top: compute + xbar + loader + 2-clock host/USB CDC | 31 tests (token == standalone across async clocks) |

### Product-hardening (on `main`) — DONE

Whole-chip structural gate (`make synth-glm`, caught a real multi-driver bug); per-row DSA
sparse decode + union key-fetch + SWIN scratch decouple; `spec_chain_top` multi-token accept;
SECDED scrub + lane-partitioned pager ECC + MBIST/ICG clock-gating; `weight_decomp` refill +
C8 loopback closed; 2-domain `reset_sync` + CDC sign-off; full-config elaboration study (found
+ guarded a latent dense≥MoE FFN-width constraint). See
[`P2_MEMORY_MAP.md`](docs/P2_MEMORY_MAP.md), [`P12_SCALEUP.md`](docs/P12_SCALEUP.md).

### Performance / power levers — measured

| Lever | What | Measured |
|---|---|---|
| `flash_xbar.v` | parallel storage-read channels (PCIe lanes / multi-NVMe) + deep outstanding queue | **7.99× latency-hide + N× banking** |
| `tools/flash_layout.py` | offline expert→channel (PCIe-lane/drive) placement (kill hotspots) | **39% → 55% of 8× peak (~+40%)** |
| `weight_decomp.v` | on-chip lossless FP8 decompress | **1.34×** bit-exact (order-0; optional order-1 `weight_decomp2` ~1.42× on a distinct intra-stream axis, not the ~1.34× cross-expert cap) |
| `spec_decode_seq.v` K>1 | multi-token speculative draft | **K=2 ≈ +23%** (spec == greedy) |
| `clk_en_ctrl.v` | gate the ~75%-idle die | **73.75% of idle dynamic power gated** (formally safe, 13 064 checks) |
| `clk_throttle.v` | DVFS/eco frequency prescaler — run the die **f/div** in the ~4–5× slack | **peak-power/thermal cap** (USB-C "eco mode"), byte-identical, **BMC-proven** (`make formal`) |
| `expert_prefetch_top.v` | predictor-driven prefetch | **measured NO-OP** (popular experts already resident — honest) |
| `glm_matmul_fp8.v` fold-pipeline (Ph1) | register the block-dequant / accumulate-fold drain (`DEQ_LAT +1`, latency-transparent) | **+25% fmax** on the isolated fold segment (64.2 → 80.7 MHz; full 2×2/K256 45 → 70 MHz), **bit-identical** (ALL 224) |

The die is only ~20–25% utilized (NVMe-bandwidth-starved), so compute-side wins (the −87.6%-cell
accumulator, the **L0/L1 die-shrink** — compact config + swiglu engine-share, both byte-identical,
[`MINIATURIZATION.md`](docs/MINIATURIZATION.md), fmax fixes, BMC) improve area/power/timing/
correctness but **do not move tok/s** — NVMe bandwidth does. That same idleness is *why* the
die-shrink is free: there is a ~4–5× compute-slowdown budget to spend on sharing/serialization. The striping strategy for that bandwidth is in
[`FLASH_STRIPING.md`](docs/FLASH_STRIPING.md).

---

## Documents

- **📊 [Slides — *AIPU Accelerator: Formally Verified FP8 Architecture*](https://docs.google.com/presentation/d/1EMqJOJCNTBaCVEf5EfYI_CymVg--K8ezVyCJcr7ovys/present)** — the project as a presentation deck (design, verification, results).
- **[`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)** — **the honest hardware plan**: a 3-rung ladder (① prove-it FPGA ~5–8 tok/s now → ② funded custom board ~15–40 → ③ SoC/ASIC ~40+ at volume [EST]) — performance is set by memory bandwidth, which is set by the chip's IO/PHY budget. **Start here for "how fast, on what hardware."**
- **[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)** — product direction (RTL/silicon track): the fidelity gate, robustness/vendor-IP/physical/software/manufacturing phases, the **FPGA-card** product path (ASIC = the rung ③ volume endgame, sequenced after FPGA proves PMF).
- **[`docs/USBC_PRODUCT_PLAN.md`](docs/USBC_PRODUCT_PLAN.md)** — productization plan for the **USB-C external device** (the appliance track): form factor, power (~80–110 W self-powered), thermal, host software, BOM/pricing (~$2.5–8 k [EST]), phased D0–D5 gates. The heavy traffic stays internal → USB-C carries only tokens.
- **[`host/`](host/README.md)** — the **host software scaffold** (D2): a local **OpenAI-compatible server** (`python3 host/aipu_server.py`, stdlib only) mirroring the RTL host interface, the **real GLM-5.2 BPE tokenizer** (+ byte fallback), the **GLM chat template**, OpenAI **sampling params**, and 3 backends — `MockDevice`, **`SimulatorBackend`** (drives the real `glm_model_fp8` slice via `vvp`), and USB (at D1). `make host-test` (18 tests).
- **[`fpga/`](fpga/README.md)** — the **D0.2 FPGA-fit vendor-flow scaffold** (Gowin GW5AT-138 / Tang Mega 138K): `gw_sh` synth+P&R script + SDC (host/core async clocks) + compact-config wrapper + nextpnr fallback. Run it (needs Gowin, a user step) for the real LUT/DSP/BSRAM/Fmax — the unknown that gates device size/thermal/BOM/price.
- **[`docs/OPERATION_FLOW.md`](docs/OPERATION_FLOW.md)** — the end-to-end operational flow: boot (NVMe→DDR5), per-token decode through every block (embed → 78-layer time-mux decoder → LM head → token), weight streaming, batching + union-skip MoE + speculative decode, CDC, and the per-token bottleneck. **Start here for "how it all runs."**
- **[`docs/ACCEL_GLM52.md`](docs/ACCEL_GLM52.md)** — accelerator architecture: exact config, MLA + DSA + MoE detail, the fp64-golden methodology, memory/streaming, RTL build order.
- **[`docs/SYSTEM_SINGLE_PACKAGE.md`](docs/SYSTEM_SINGLE_PACKAGE.md)** — single-module system (FP8 die + DDR working cache + 1 TB NVMe SSD, e.g. a USB-C box): tiering, expert caching, the bottleneck/perf/cost model. (The DDR tier is rung-dependent — DDR4/DDR5/HBM per [`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md).)
- **Evidence:** [`REAL_CKPT_VALIDATION.md`](docs/REAL_CKPT_VALIDATION.md) (real-checkpoint bit-exact + T4) · [`SCALE_FUNCTIONAL.md`](docs/SCALE_FUNCTIONAL.md) (operators at real dims) · [`PHYSICAL_SKY130.md`](docs/PHYSICAL_SKY130.md) (real sky130 area/P&R) · [`MODAL_VALIDATE.md`](docs/MODAL_VALIDATE.md) (GPU validation harness).
- **Perf / power / physical:** [`IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md) · [`LOW_POWER.md`](docs/LOW_POWER.md) (the bit-exact low-power path: energy is ~80% NVMe-read bytes → amortize the fetch; the DVFS **frequency** knob is now RTL-realized (`clk_throttle`, peak-power/eco), the J/token half is voltage/vendor; projected ~9 → ~1.5–3 J/token [EST]) · [`ULTRA_PERF.md`](docs/ULTRA_PERF.md) · [`FLASH_STRIPING.md`](docs/FLASH_STRIPING.md) · [`CYCLE_EMULATION.md`](docs/CYCLE_EMULATION.md) (cycle-accurate: the memory-stall mechanism measured on real RTL cycles) · [`MINIATURIZATION.md`](docs/MINIATURIZATION.md) (die shrink: compute is nearly free on an NVMe-bound die → shared engines. **L0 compact config + L1 landed** — swiglu gate/up merged → 6→4 FP8 GEMM engines/block, byte-identical) · [`PPA_FP8.md`](docs/PPA_FP8.md) · [`FORMAL.md`](docs/FORMAL.md).
- **Verification / scale:** [`FULL_CONFIG_ELAB.md`](docs/FULL_CONFIG_ELAB.md) (the RTL elaborates clean at the **true 753B config** — verilator, 0 errors; full-config SELRANGE lints since cleared 4122→0 byte-identical) · [`COVERAGE.md`](docs/COVERAGE.md) (verilator line/toggle/branch coverage, `make coverage` — merged **87.8% line / 80.1% toggle / 88.9% branch**) · [`P12_SCALEUP.md`](docs/P12_SCALEUP.md) · [`P2_MEMORY_MAP.md`](docs/P2_MEMORY_MAP.md).

---

## Build / test

```sh
brew install icarus-verilog verilator yosys     # iverilog 13.0, verilator 5.048, yosys 0.66

make unittests   # build+run every per-unit TB (GLM-5.2 + FP8 + system units)
make formal      # bounded model checking (yosys-smtbmc + z3) of the memory controllers + clk_throttle
make coverage    # verilator line/toggle/branch coverage over the verilatable unit TBs
make host-test   # host OpenAI-server + device-protocol + tokenizer scaffold tests (18)
make cache-study # GLM-trace hit-rate / batching / prefetch / decompress / layout measurements
make lint        # verilator lint of the GLM top (diagnostic; not part of `all`)
make synth-glm   # yosys whole-chip structural gate on the GLM top glm_fp8_system_cdc
make synth-glm-compact  # same gate on the FPGA-miniaturization config (PE_N=2/DDR_NCH=2/…, byte-identical)
make sim-glm-compact    # system TB proving the compact config decodes the SAME token stream
make all         # the GLM rung-① FPGA prove-it gate: unittests + synth-glm + formal
```

Per-GLM-unit compile (list sources explicitly — **zsh does not word-split**):

```sh
mkdir -p build
# FP8 E4M3 primitives (exhaustive):
iverilog -g2012 -Wall -I src -o build/fp8 test/fp8_e4m3_tb.v && vvp build/fp8   # ALL 66069 TESTS PASSED
# full FP8 forward-pass capstone:
iverilog -g2012 -Wall -I src -o build/glm_model_fp8_sim test/glm_model_fp8_tb.v \
    src/glm_model_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v \
    src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v \
    src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v \
    src/sampler.v src/glm_fp_pipe.v
vvp build/glm_model_fp8_sim      # -> next-token argmax matches the fp8 golden
```

---

## Slice configuration & substrate

The RTL is built at a small-but-faithful **slice** keeping every operator and ratio:
MODEL_DIM=128, 6 layers (3 dense + 3 MoE), 4 heads, MLA nope16/rope16/v32, q_lora64/kv_lora32,
8-expert top-2 + 1 shared, INTER_MOE=64, INTER_DENSE=256, VOCAB=256, S_MAX=8. Running the real
753B model adds the memory/streaming system + array scaling ([`ACCEL_GLM52.md`](docs/ACCEL_GLM52.md)).

**FPGA sanity note.** An FP8 GEMM (`glm_matmul_fp8`, PE 1×1 time-multiplexed) fits a **Tang Nano
20K** (GW2A-18, ~49% LUT, ~0 DSP) — fp32 does not (`mla_attn` alone ≈ 396 DSP-equiv, 8× the
device), which is the point of FP8: its 4×4 mantissa multiply frees the scarce DSP and spends
LUTs on the accumulator. This is a **silicon-fabric sanity test, not the deliverable** (the board
cannot run GLM-5.2); the target is a **large (data-center-class) FPGA** with the DDR5+NVMe system.
A partitioned `synth_ecp5` of the six memory-system controllers already sums to **~71,475 LUT4 — ~85%
of an ECP5-85 on the controllers alone** — so the full system does **not** fit an ECP5-85 and needs a
larger FPGA; the compute die's ECP5 size is **not yet measured** (a yosys-0.66 `synth_ecp5` scalability
limit — an earlier "die 32–64× over" figure was a `KMAX` synth artifact, since disproven)
([`PHYSICAL_SKY130.md`](docs/PHYSICAL_SKY130.md)). Because the workload is NVMe/PCIe-bandwidth-bound (the die
sits ~75–80% idle behind the NVMe storage), an FPGA card is the committed **near-term** product — at this
rung an ASIC's faster *compute* would be largely wasted. But the real ceiling is **memory bandwidth (IO
pins + PHY)**, which is exactly what an ASIC breaks (HBM stacks + many-channel controllers + near-memory
compute at ~TB/s, lower $/seat + power at volume) — so ASIC is **not out of scope: it is the rung ③ volume
endgame**, sequenced after the FPGA proves product-market fit ([`docs/HARDWARE_LADDER.md`](docs/HARDWARE_LADDER.md)).
