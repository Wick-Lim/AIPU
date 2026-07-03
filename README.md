# AIPU — a GLM-5.2-FP8 inference accelerator in Verilog

A synthesizable Verilog accelerator with one goal: **run one real model well —
[`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8)**, the published FP8
checkpoint of GLM-5.2 (`GlmMoeDsaForCausalLM`), a 753B-param MoE (~40B active/token) in
native **FP8 E4M3**. The complete GLM-5.2 operator datapath is built in FP8, verified
against independent fp64/fp8 goldens at a small-but-faithful slice, and wrapped by the
single-module memory system (multi-channel DDR5 + Flash expert cache + weight/boot loaders +
multi-clock CDC) that streams the real 753B model — with the memory controllers
bounded-model-checked.

> **Naming.** **AIPU** (AI Processing Unit, repo [`Wick-Lim/AIPU`](https://github.com/Wick-Lim/AIPU))
> is the whole accelerator. The project was formerly *TPU*; the classic *5-stage scalar TPU
> core* underneath keeps its own name (*"TPU v2.0"*, `tpu_*` modules) as the control substrate.
>
> **Branches:** `prototype` (frozen at `fee8501`) = the research prototype (full FP8 datapath +
> memory system + batching stack, bit-exact at the slice); **`main`** = the product track, taking
> it toward a shippable accelerator ([`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md),
> [`NEXT_STEPS_PLAN.md`](NEXT_STEPS_PLAN.md)).

---

## What's proven — measured on real artifacts, not claimed

The project's defining property is verification discipline: nothing below is a model
estimate — each row is a real measurement against a real artifact (the published checkpoint,
exhaustive enumeration, a real PDK, or a formal solver).

| Evidence | Against | Result |
|---|---|---|
| **Operator bit-accuracy vs the real checkpoint** | published `GLM-5.2-FP8` safetensors (`kv_a_proj` F8_E4M3) | **9216/9216 = 100%** bf16-exact, **argmax 16/16** ([`REAL_CKPT_VALIDATION.md`](docs/REAL_CKPT_VALIDATION.md)) |
| **Real MoE experts on GPU** | real expert weights, Modal T4 | **argmax-preserving** (tier1) |
| **FP8 E4M3 arithmetic** | fp64, **exhaustive** | **ALL 66069** (256 decodes + all 256×256 multiplies) |
| **Operators at REAL GLM-5.2 dims** | fp64 goldens | GEMM **K=6144**, router **256/top-8**, SwiGLU **2048**, MLA real geo — all bit-exact ([`SCALE_FUNCTIONAL.md`](docs/SCALE_FUNCTIONAL.md)) |
| **Full FP8 forward pass** (slice) | fp8 golden | **next-token argmax == golden** |
| **Real sky130 place-and-route** (`glm_matmul_fp8`) | SkyWater sky130 PDK, OpenROAD | synth→floorplan→**legalized placement**, **357,320 µm²**, post-placement timing **MET** (+15.89 ns) ([`PHYSICAL_SKY130.md`](docs/PHYSICAL_SKY130.md)) |
| **Memory-system controllers** | z3 | **BMC** (6 controllers) + **unbounded k-induction** ([`FORMAL.md`](docs/FORMAL.md)) |
| **PE_M batch path** (0 extra weight BW) | per-row single-token refs | swiglu **513** / router **192** / mla **6** — *"4 rows == 1 fetch stream"* |

**Modeled, not silicon — flagged [EST].** All throughput/energy figures come from a
bandwidth-roofline model (`tokens/s ≈ Flash_BW / [(1−h)·footprint] · K`), **not** from a routed
netlist or silicon: single-user **~3 → ~30+ tok/s** and **~9 → ~3 J/token** [EST] after stacking
the Flash levers. Read them as an optimistic ceiling ([`ULTRA_PERF.md`](docs/ULTRA_PERF.md),
[`IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md)). The roofline's *memory-stall mechanism* is,
however, now **measured on real RTL cycles** (assembled system, cycle-accurate): the exposed stall
is exactly linear in Flash latency (`stall = 3·FLASH_LAT + 9`) and proportional to miss count — so
the mechanism is validated even though the absolute tok/s is not ([`CYCLE_EMULATION.md`](docs/CYCLE_EMULATION.md)).

**Out of scope** (vendor IP / not attempted): DDR5/Flash/USB-C **PHYs** (TB-stubbed), the
**tokenizer** (software), **full-chip FPGA P&R + board bring-up** (ASIC/tapeout is out of scope — the product is an FPGA card), and a **full-model 8×H200 GPU
validation** (resource-gated; substituted by the CPU-bit-exact + T4 evidence above).

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
  1 TB Flash  ──►  flash_xbar   ──►  64 GB DDR5   ──►  ddr5_xbar  ──►   FP8 compute die   ──►  token
 (753B FP8     N-channel banked   expert cache      N-channel      (MLA + DSA + MoE, native
  weights)     + deep queue       (LRU+freq+pf)     banked read     FP8 E4M3, bf16 tail)
```

The workload is **Flash-bandwidth-bound**, so the system is built around streaming MoE experts
from Flash through a DDR5 working cache into a mostly-idle FP8 die. Every FP8 **weight** matmul
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
| `swiglu_expert_fp8.v` | gate/up/down GEMMs FP8, bf16 silu tail; **PE_M-batched** | 1024 + PE_M 513 (0 extra weight BW) |
| `mla_attn_fp8.v` | 7 weight projections FP8, bf16 attn/rope/norm/softmax/dsa; **PE_M-batched, per-row pos/s_len** | slice 7; **real-dim (Q2048/KV512/…) worst rel 5.48e-4**; PE_M 6 |
| `moe_router_fp8.v` | gate GEMV FP8, bf16 sigmoid/topk/renorm; **PE_M-batched** | 185 + real 256/top-8 + PE_M 192 |
| `glm_decoder_block_fp8.v` | one full FP8 decoder layer | 9 tests, dense + MoE |
| **`glm_model_fp8.v`** | **full FP8 forward pass** | **next-token argmax == fp8 golden** |
| `mtp_head_fp8.v` | FP8 multi-token-prediction (t+2) head; **PE_M-batched** | 6 + PE_M 44 (all weight ports: B rows == 1 fetch) |

### Single-module system (real-753B memory/streaming hardware) — BUILT

| Unit | Role | Verification |
|---|---|---|
| `expert_cache_pf.v` | DDR5 routed-expert cache: LRU + freq + prefetch | 623 tests; **BMC-proven** |
| `kv_cache_pager.v` | MLA latent-KV ring + DSA-gather + Flash overflow | 73 tests; **BMC-proven** |
| `ddr5_xbar.v` | N-channel banked DDR5 read fabric (~N× BW) | 3073 tests (7.93× @8ch); **BMC + k-induction** |
| `flash_xbar.v` | N-channel banked **Flash** read fabric (deep queue hides NAND latency) | 2049 tests (7.99× latency-hide); **BMC-proven** |
| `weight_loader.v` | checkpoint FP8 + block-scale → matmul pull DMA | 240 tests (loader-fed == direct-fed) |
| `boot_loader.v` | power-up Flash→DDR5 model-load sequencer | 9240 tests; **BMC-proven** |
| `spec_decode_seq/_top.v` | MTP speculative-decode loop (K>1 draft) | 621/1379/19 tests; **BMC-proven** |
| **`glm_fp8_system_cdc.v`** | production top: compute + xbar + loader + 2-clock host/USB CDC | 31 tests (token == standalone across async clocks) |

### Product-hardening (on `main`) — DONE

Whole-chip structural gate (`make synth-glm`, caught a real multi-driver bug); per-row DSA
sparse decode + union key-fetch + SWIN scratch decouple; `spec_chain_top` multi-token accept;
SECDED scrub + lane-partitioned pager ECC + MBIST/ICG clock-gating; `weight_decomp` refill +
C8 loopback closed; 2-domain `reset_sync` + CDC sign-off; full-config elaboration study (found
+ guarded a latent dense≥MoE FFN-width constraint). See [`ROADMAP.md`](docs/ROADMAP.md),
[`P2_MEMORY_MAP.md`](docs/P2_MEMORY_MAP.md), [`P12_SCALEUP.md`](docs/P12_SCALEUP.md).

### Performance / power levers — measured

| Lever | What | Measured |
|---|---|---|
| `flash_xbar.v` | parallel Flash channels + deep outstanding queue | **7.99× latency-hide + N× banking** |
| `tools/flash_layout.py` | offline expert→channel placement (kill hotspots) | **39% → 55% of 8× peak (~+40%)** |
| `weight_decomp.v` | on-chip lossless FP8 decompress | **1.34×** bit-exact |
| `spec_decode_seq.v` K>1 | multi-token speculative draft | **K=2 ≈ +23%** (spec == greedy) |
| `clk_en_ctrl.v` | gate the ~75%-idle die | **74% of idle dynamic power gated** |
| `expert_prefetch_top.v` | predictor-driven prefetch | **measured NO-OP** (popular experts already resident — honest) |

The die is only ~20–25% utilized (Flash-starved), so compute-side wins (the −87.6%-cell
accumulator, fmax fixes, BMC) improve area/power/timing/correctness but **do not move tok/s** —
Flash bandwidth does. The striping strategy for that bandwidth is in
[`FLASH_STRIPING.md`](docs/FLASH_STRIPING.md).

---

## Documents

- **[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)** — product direction: the fidelity gate, robustness/vendor-IP/physical/software/manufacturing phases, the **FPGA-card** product path (ASIC out of scope).
- **[`docs/ACCEL_GLM52.md`](docs/ACCEL_GLM52.md)** — accelerator architecture: exact config, MLA + DSA + MoE detail, the fp64-golden methodology, memory/streaming, RTL build order.
- **[`docs/SYSTEM_SINGLE_PACKAGE.md`](docs/SYSTEM_SINGLE_PACKAGE.md)** — single-module system (FP8 die + 64 GB DDR5 + 1 TB Flash, e.g. a USB-C box): tiering, expert caching, the bottleneck/perf/cost model.
- **Evidence:** [`REAL_CKPT_VALIDATION.md`](docs/REAL_CKPT_VALIDATION.md) (real-checkpoint bit-exact + T4) · [`SCALE_FUNCTIONAL.md`](docs/SCALE_FUNCTIONAL.md) (operators at real dims) · [`PHYSICAL_SKY130.md`](docs/PHYSICAL_SKY130.md) (real sky130 area/P&R) · [`MODAL_VALIDATE.md`](docs/MODAL_VALIDATE.md) (GPU validation harness).
- **Perf / physical:** [`IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md) · [`ULTRA_PERF.md`](docs/ULTRA_PERF.md) · [`FLASH_STRIPING.md`](docs/FLASH_STRIPING.md) · [`CYCLE_EMULATION.md`](docs/CYCLE_EMULATION.md) (cycle-accurate: the memory-stall mechanism measured on real RTL cycles) · [`PPA_FP8.md`](docs/PPA_FP8.md) · [`FORMAL.md`](docs/FORMAL.md).
- **Scale / memory:** [`P12_SCALEUP.md`](docs/P12_SCALEUP.md) · [`P2_MEMORY_MAP.md`](docs/P2_MEMORY_MAP.md) · [`ROADMAP.md`](docs/ROADMAP.md).
- **Scalar core substrate:** [`SPEC.md`](SPEC.md) · [`docs/ISA.md`](docs/ISA.md) · [`docs/PPA.md`](docs/PPA.md).

---

## Build / test

```sh
brew install icarus-verilog verilator yosys     # iverilog 13.0, verilator 5.048, yosys 0.66

make unittests   # build+run every per-unit TB (GLM-5.2 + FP8 + system units)
make formal      # bounded model checking (yosys-smtbmc + z3) of the memory controllers
make cache-study # GLM-trace hit-rate / batching / prefetch / decompress / layout measurements
make lint        # verilator --lint-only -Wall
make synth-glm   # yosys whole-chip structural gate on the product top glm_fp8_system_cdc
make all         # test + hazard + unittests + lint + synth + synth-glm + formal (full CI)
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

Underneath sits the **scalar TPU v2.0 core** — a classic 5-stage pipeline (IF→ID→EX→MEM→WB) with
hazard forwarding, load-use stalls, and a busy-stall that hands tensor ops to dedicated units
(`gemm_systolic`, `conv2d_unit`, `softmax_unit`, `attention_unit`) + an AXI4-Lite wrapper — used
as the control/integration substrate. Detail in [`SPEC.md`](SPEC.md) / [`docs/ISA.md`](docs/ISA.md).

**FPGA sanity note.** An FP8 GEMM (`glm_matmul_fp8`, PE 1×1 time-multiplexed) fits a **Tang Nano
20K** (GW2A-18, ~49% LUT, ~0 DSP) — fp32 does not (`mla_attn` alone ≈ 396 DSP-equiv, 8× the
device), which is the point of FP8: its 4×4 mantissa multiply frees the scarce DSP and spends
LUTs on the accumulator. This is a **silicon-fabric sanity test, not the deliverable** (the board
cannot run GLM-5.2); the target is a **large (data-center-class) FPGA** with the DDR5+Flash system. Because the workload is Flash-bandwidth-bound (the die sits ~75–80% idle behind Flash), an FPGA card is the committed product — an ASIC's faster compute would be largely wasted, so ASIC is out of scope.
