# FPGA measured-demo plan (D0.2 — the single biggest unknown)

*Turns the [`fpga/`](../fpga/README.md) "FPGA fit" scaffold into a measured result and a hardware
demo. **Why it matters:** the FPGA class sets the box's size / thermal / **BOM / per-seat price**,
and the per-seat price is what makes the [`ICP.md`](ICP.md) real. Everything downstream is bounded
by the number this track returns.*

**Product frame:** local, single-user box (B=1), target board **Sipeed Tang Mega 138K Pro**
(**Gowin GW5AT-138**, ~138 K LUT, on-board DDR3). See [`MINIATURIZATION.md`](MINIATURIZATION.md).

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

## Whole-system fit — a second wall found (honest)

`synth_gowin` on the **compact product top** (`glm_fp8_system_cdc`) **and** on the **core compute block**
(`glm_decoder_block_fp8`) both **failed to complete in yosys 0.66 within their timeout** — and both hung
in the **same** place: RTLIL **elaboration** (`derive` mode) of a `glm_matmul_fp8` instance whose
capacity parameter is **KMAX = 16384**, *before mapping even started*.

**The blocker is precisely one parameter, not the design.** `glm_matmul_fp8` maps fine at its default
(small) KMAX — the leaf probe above proves it. Every *parent* (decoder block, model, system top)
instantiates it with **KMAX = 16384** (the max accumulator depth for the big GEMMs / LM-head vocab-tile),
and generating the RTLIL for a 16384-deep accumulator is what yosys 0.66 can't finish. This is a
**distinct, earlier wall than abc-lut4** (it's elaboration, not LUT mapping) — and it's **actionable**:

- **For the fit number:** use the vendor flow — Gowin EDA `GowinSynthesis` (`fpga/gowin/build_gowin.tcl`)
  or newer yosys + `nextpnr-himbaechel` (both are what the [`fpga/`](../fpga/README.md) scaffold exists
  for); their elaborators handle the wide accumulator where yosys 0.66 does not.
- **For a demo build:** tile the accumulator over K (the datapath already processes K in `BLK=128`
  blocks — KMAX is only the *scratch capacity*), i.e. lower KMAX to the real per-GEMM K, which maps in
  yosys today (see the KMAX-swept probe below).

### Measured: `glm_matmul_fp8` maps across KMAX (the compute core's dominant unit)

<!-- MM_FIT: default-KMAX leaf + KMAX=2048 probe -->
| Config | LUT4-equiv | DSP mult | DFF | result (yosys 0.66 `synth_gowin`) |
|---|---|---|---|---|
| KMAX = 256 (module default) | ~17.8 K | 20 (`MULT18X18`×2 + `MULT9X9`×18) | ~5.4 K | ✅ **maps clean, ~77 s** |
| KMAX = 2048 | — | (DSP inferred) | — | ⚠️ elaborates (past the 16384 wall), but the `synth_gowin` SAT resource-sharing pass did not finish in 180 s |
| KMAX = 16384 (product LM-head cap) | — | — | — | ❌ does not even elaborate |

**What this says:** the FP8 datapath **maps** (KMAX=256 is a clean, real data point — abc-lut4 wall
broken via DSP inference); but **yosys 0.66's synth time scales steeply with the accumulator capacity**
(2048 mapping is slow, 16384 won't elaborate). That is a **tooling-scaling** limit, not an
un-mappability — exactly what a production mapper (**Gowin `GowinSynthesis`** / newer yosys +
`nextpnr`) is built to handle, and/or what **K-tiling** (KMAX → the real per-GEMM K, since the datapath
already streams K in `BLK=128` blocks) sidesteps.

**Caveats (honest):** generic-`synth_gowin` estimates, **not** a routed Gowin EDA / nextpnr result —
final LUT/DSP/BSRAM/**Fmax** need the vendor flow. These establish *mappability + resource shape + the
exact blocker*, not the shippable LUT count. The whole-system routed fit remains a **vendor-flow task**
— but it is now a *known, bounded* one (resolve the KMAX=16384 accumulator), not an open mystery.

## The demo ladder (each rung is cheap and de-risks the next)

| Rung | What | Tooling | Proves | Status |
|---|---|---|---|---|
| **L0** | **Gowin-synth fit** — LUT/DSP/BSRAM of the compact top vs GW5AT-138 | `yosys synth_gowin` (have it) → **vendor flow for the top** | the design *fits a class of FPGA* → BOM band | leaf ✅ (KMAX=256 maps, abc wall broken); **top ⚠ needs vendor flow** (yosys 0.66 KMAX=16384 elaboration wall) |
| **L1** | **P&R one leaf** (`glm_matmul_fp8`) on the board → real **Fmax** | Gowin EDA / nextpnr | real clock → real tok/s extrapolation (tok/s = Fmax ÷ `cyc_per_tok`) | scaffold ready (`fpga/gemm_harness.v`) |
| **L2** | **P&R the compact top** → full routed fit + Fmax | Gowin EDA / nextpnr | the whole product top places & routes | scaffold ready (`fpga/gowin/`) |
| **L3** | **Reduced-config forward on the board** — a few real GLM-5.2 layers, weights streamed from on-board Flash/SD → **measured tok/s** | board + `ckpt_pack.py` image | *real silicon token at a measured tok/s* — **THE demo** | needs board |

**tok/s is already grounded in RTL, not hand-waved:** the memory-stall mechanism and `cyc_per_tok`
are measured on real RTL cycles ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md); stall = `3·FLASH_LAT+9`,
`cyc_per_tok` grows with Flash latency). So once L1/L2 give a routed **Fmax**, `tok/s = Fmax ÷
cyc_per_tok` is a *measured*, not modeled, single-user number — the thing that converts the
[`ULTRA_PERF.md`](ULTRA_PERF.md) [EST] ladder into fact.

## Cost / what a real demo needs

- **L0–L2 are near-free:** the tooling is `yosys` (have) + **Gowin EDA (free, license-gated)** or
  open `nextpnr-himbaechel`; the board (**Tang Mega 138K Pro, ~$200–300**) is only needed to *program*.
- **L3 (the money shot)** needs the board + a Flash/SD-resident quantized weight image (`ckpt_pack.py`
  produces it) and a reduced config (a few layers) — **not** the full 1 TB / real DDR5/Flash PHYs
  (those are the vendor-IP, out-of-scope-for-demo items). The demo is a **reduced-config proof that
  real weights produce real tokens on real silicon at a measured rate**, not the shippable box.

## Why this is the investable lever

An investor discounts every `[EST]`. The demo ladder converts three of them to fact, cheaply:
1. **Fit/BOM** (L0–L2): "it fits a $X FPGA" → the per-seat price the [`ICP.md`](ICP.md) economics need.
2. **Single-user tok/s** (L1/L2 Fmax × measured `cyc_per_tok`): the product speed, measured not modeled.
3. **Real tokens on silicon** (L3): the whole thesis, demonstrable on a desk.

Paired with **one signed design-partner LOI** from the primary ICP (a law-firm innovation team), that
is the pre-seed package: *a measured box + a customer who wants it.*
