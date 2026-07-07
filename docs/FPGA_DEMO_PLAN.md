# FPGA measured-demo plan (D0.2 — the single biggest unknown)

*Turns the [`fpga/`](../fpga/README.md) "FPGA fit" scaffold into a measured result and a hardware
demo. **Why it matters:** the FPGA class sets the box's size / thermal / **BOM / per-seat price**,
and the per-seat price is what makes the [`ICP.md`](ICP.md) real. Everything downstream is bounded
by the number this track returns.*

**Product frame:** local, single-user box (B=1). This **is** what `main` develops right now — the
**rung-① "prove-it"** plan for the **GLM-5.2-FP8 accelerator** on the staged [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) — the cheap, near-term proof that the *same* verified
RTL runs the real model's tokens on real FPGA silicon (**~5–8 tok/s [EST]**, slow but bit-exact). The
**Sipeed Tang Mega 138K Pro** (**Gowin GW5AT-138**, ~138 K LUT, on-board DDR3) is used here as the
**toolchain / bring-up board** for the synth + P&R flow; the real rung-① prove-it target is a **low-end
Kintex UltraScale+ (KU3P-class)** board with **DDR4 (~4 ch, ~100 GB/s) + 1 NVMe** (~$1–2 k box) — the
DDR4 memory system is what sets the ~5–8 tok/s [EST]. See [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) for
the staged context and [`MINIATURIZATION.md`](MINIATURIZATION.md).

---

## The wall we had to break first

The repo's honest status was: `make synth-glm` only *elaborates* the top; it emits **no LUT count**
because **yosys 0.66's `abc -lut4` times out on `glm_matmul_fp8`'s FP8 accumulator banks**. Only the
memory controllers had ever been mapped (~71 K LUT4 on ECP5); the **compute die was unmeasured**
([`PHYSICAL_SKY130.md`](PHYSICAL_SKY130.md), [`../fpga/README.md`](../fpga/README.md)).

**Broken (this session).** `yosys synth_gowin` maps the same FP8 datapath **by inferring hardware
DSP multipliers** (`MULT18X18` / `MULT9X9`) instead of LUT-mapping every multiply — which sidesteps
the `abc -lut4` accumulator blow-up. Measured leaf, `glm_matmul_fp8` at **KMAX = 256** (the module
default), **~77 s**:

| Primitive | Count |
|---|---|
| LUT1–4 | ~8.4 K |
| ALU (arith LUT) | ~5.0 K |
| wide MUX (LUT5–8) | ~1.4 K cells |
| **≈ LUT4-equivalent** | **~17.8 K** |
| **DSP mult** (`MULT18X18` ×2 + `MULT9X9` ×18) | **20** |
| DFF | ~5.4 K |

This is a methodology result: **the FP8 compute datapath maps to Gowin primitives; the "yosys wall"
was an `abc -lut4` limitation, not an un-mappable design.** DSP inference is the key.

## Whole-system fit — the elaboration wall was a real area bug, now FIXED in RTL

The first diagnosis (that KMAX=16384 "wouldn't elaborate") turned out to have a **root cause in the RTL,
not just the tool**: `glm_matmul_fp8`'s block-scale **dequant fold was fully unrolled** — one
`fp32_mul_pipe` + one `fp32_add_pipe` **per K-block**, plus `O(NB²)` alignment-delay registers, where
`NB = ceil(KMAX/BLK)`. At the product LM-head cap (KMAX=16384 → **NB=128**) that is **128 mul + 128 add
pipes and ~40k delay registers *per PE*** — millions of FFs / hundreds of FP pipes across the array.
So the core GEMM, as written, was **genuinely unbuildable at the product config** (a P1.2 scale bug),
and yosys just surfaced it as an elaboration hang.

**Fixed this session (bit-exact).** The dequant was rewritten as a **sequential** block-scaled fold:
one `fp32_mul_pipe` + one `fp32_add_pipe`, **reused** across all NB blocks and all NOUT outputs, folding
in block order `((0+p0)+p1)+…` with the **same two roundings per block** (mul-then-add, deliberately not
a fused MAC). Area drops from **O(NB²) → O(1)** in FP pipes (256 → 2 dequant pipes at NB=128); the only
remaining NB-scaling is the integer block-accumulator memory `accx` (**O(NB)** words, BRAM-able — one
`[128,128]`-block sum each, which the block-scaled contract genuinely needs).

**Verified BIT-EXACT** — same golden, no output change anywhere:

| Check | Result |
|---|---|
| `glm_matmul_fp8_tb` @ KMAX=256 (NB=2) | **ALL 224 PASS** vs exact golden |
| `glm_matmul_fp8_tb` @ KMAX=2048 (NB=16) | **ALL 224 PASS** (many-block fold) |
| `make bitacc` (matmul == real GLM-5.2-FP8 contract) | **14/14 + argmax 28/28**, contract bit-exact |
| full-slice `glm_model_fp8` next-token argmax | unchanged (integrated regression) |

**Effect on synth.** yosys 0.66 now **elaborates + `proc`s `glm_matmul_fp8` at KMAX=16384 to a finite
netlist** (was impossible before — it hung deriving the O(NB²) fold); the dequant is down to **2 FP
pipes**. The remaining slowness at full NB is yosys 0.66's **SAT-based `SHARE` pass** (a tool
scalability limit, not the design) — which the vendor flow (Gowin `GowinSynthesis`) / `nextpnr` don't
hit. So the **RTL-side blocker is retired**; the routed LUT/DSP/BSRAM/**Fmax** still come from the
vendor flow, but the design is now buildable at product scale.

### Measured: `glm_matmul_fp8` synth across KMAX (dequant O(NB²)→O(1))

| Config | dequant FP pipes | elaborate + `proc` (yosys 0.66) | `synth_gowin` map |
|---|---|---|---|
| KMAX = 256 (NB=2) | 2 (was 4) | ✅ | ✅ **~77 s**, ~17.8 K LUT-equiv + 20 DSP |
| KMAX = 2048 (NB=16) | 2 (was 32) | ✅ | ✅ elaborates; SAT-`SHARE` pass slow (tool) |
| KMAX = 16384 (NB=128) | **2 (was 256)** | ✅ **finite netlist** (was: elaboration hang) | SAT-`SHARE` pass = vendor-flow / newer-yosys |

**Caveats (honest):** the LUT/DSP figures are generic-`synth_gowin` estimates, **not** a routed Gowin
EDA / nextpnr result — final LUT/DSP/BSRAM/**Fmax** need the vendor flow. What's now established: the
core GEMM is **bit-exact and buildable at the product config** (the O(NB²) dequant is gone); the routed
whole-system number is a vendor-flow measurement, no longer blocked by an RTL area explosion.

## The demo ladder (each rung is cheap and de-risks the next)

| Rung | What | Tooling | Proves | Status |
|---|---|---|---|---|
| **L0** | **Gowin-synth fit** — LUT/DSP/BSRAM of the compact top vs GW5AT-138 | `yosys synth_gowin` (have it) → **vendor flow for the routed top** | the design *fits a class of FPGA* → BOM band | leaf ✅ (abc wall broken via DSP inference); **RTL area blocker retired** (matmul dequant O(NB²)→O(1), bit-exact — the design now elaborates at product KMAX); routed top number = vendor flow |
| **L1** | **P&R one leaf** (`glm_matmul_fp8`) on the board → real **Fmax** | Gowin EDA / nextpnr | real clock → real tok/s extrapolation (tok/s = Fmax ÷ `cyc_per_tok`) | scaffold ready (`fpga/gemm_harness.v`) |
| **L2** | **P&R the compact top** → full routed fit + Fmax | Gowin EDA / nextpnr | the whole product top places & routes | scaffold ready (`fpga/gowin/`) |
| **L3** | **Reduced-config forward on the board** — a few real GLM-5.2 layers, weights streamed from on-board Flash/SD → **measured tok/s** | board + `ckpt_pack.py` image | *real silicon token at a measured tok/s* — **THE demo** | needs board |

**tok/s is already grounded in RTL, not hand-waved:** the memory-stall mechanism and `cyc_per_tok`
are measured on real RTL cycles ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md); stall = `3·FLASH_LAT+9`,
`cyc_per_tok` grows with storage-read latency). (`FLASH_LAT` and the `flash_xbar` read path are
committed RTL names for a **medium-agnostic storage-read abstraction** — address → weight bytes,
latency-hidden — that in the product fronts an **NVMe/PCIe** backend, not a NAND die.) So once L1/L2
give a routed **Fmax**, `tok/s = Fmax ÷
cyc_per_tok` is a *measured*, not modeled, single-user number — the thing that converts the
[`ULTRA_PERF.md`](ULTRA_PERF.md) [EST] ladder into fact. On this rung the DDR4-bound target is
**~5–8 tok/s [EST]** (the [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) rung-① proof); the funded custom
board (rung-②, DDR5/HBM) is where the **~15–40 tok/s [EST]** interactive product lands, and rung-③
(SoC/ASIC at volume) reaches **~40+ [EST]** — all the *same* bit-exact RTL, only the memory bandwidth
the silicon can feed it changes.

## Cost / what a real demo needs

- **L0–L2 are near-free:** the tooling is `yosys` (have) + **Gowin EDA (free, license-gated)** or
  open `nextpnr-himbaechel`; the board (**Tang Mega 138K Pro, ~$200–300**) is only needed to *program*.
- **L3 (the money shot)** needs the board + a Flash/SD-resident quantized weight image (`ckpt_pack.py`
  produces it — the demo board's on-board Flash/SD is all a reduced config needs) and a reduced config
  (a few layers) — **not** the product's full **1–4 TB NVMe model store** or the **DDR5/HBM + NVMe/PCIe
  (M.2) host controller** (those are **rung-② custom-board / vendor-IP items** — see
  [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md); DDR5/HBM is a funded-board spec, not the near-term proof). The
  demo is a **reduced-config proof that real weights produce real tokens on real silicon at a measured
  rate**, not the shippable box.

## Why this is the investable lever

An investor discounts every `[EST]`. The demo ladder converts three of them to fact, cheaply:
1. **Fit/BOM** (L0–L2): "it fits a $X FPGA" → the per-seat price the [`ICP.md`](ICP.md) economics need.
2. **Single-user tok/s** (L1/L2 Fmax × measured `cyc_per_tok`): the rung-① proof speed (**~5–8 tok/s
   [EST]** on DDR4), measured not modeled — the funded rung-② board is where **~15–40 tok/s [EST]** lands
   ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)).
3. **Real tokens on silicon** (L3): the whole thesis, demonstrable on a desk.

Paired with **one signed design-partner LOI** from the primary ICP (a law-firm innovation team), that
is the pre-seed package: *a measured box + a customer who wants it.*
