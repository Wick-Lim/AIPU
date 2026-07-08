# FPGA measured-demo plan (D0.2 — the single biggest unknown)

*Turns the [`fpga/`](../fpga/README.md) "FPGA fit" scaffold into a measured result and a hardware
demo. **Why it matters:** the FPGA class sets the box's size / thermal / **BOM / per-seat price**,
and the per-seat price is what makes the [`ICP.md`](ICP.md) real. Everything downstream is bounded
by the number this track returns.*

**Product frame:** local, single-user box (B=1). This **is** what `main` develops right now — the
**rung-① "prove-it"** plan for the **GLM-5.2 Q4_K accelerator** (GGML UD-Q4_K_XL, ~467 GB) on the
staged [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) — the cheap, near-term proof that the *same* verified
RTL runs the real model's tokens on real FPGA silicon (**~5–8 tok/s [EST]**, slow but memory-bound).
Two boards, two jobs (see [`PART_SELECTION.md`](PART_SELECTION.md) §the ①a-vs-①b split):

- **①a fit-measurement / bring-up board** — the **Sipeed Tang Mega 138K Pro** (**Gowin GW5AT-138**,
  ~138 K LUT, on-board DDR3). Cheap way to exercise the synth + P&R flow and read a fit. It is
  **DDR3-only, no NVMe**, so it can *measure the fit* but **cannot** run the streaming token demo.
- **①b demo target** — a **low-end Kintex UltraScale+ (KU3P-class, XCKU3P)** board with **DDR4 + 1
  NVMe** (~$1–2 k box). The DDR4/NVMe memory system is what sets the ~5–8 tok/s [EST]; this is also
  the **repo-designated part** whose **routed fit + Fmax on Vivado is the PENDING gate** below.

See [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) for the staged context, [`PART_SELECTION.md`](PART_SELECTION.md)
for the board BOM, and [`MINIATURIZATION.md`](MINIATURIZATION.md) for the compact config.

---

## What is DONE (structure only) vs the one PENDING gate

The honest status splits cleanly: the Q4_K RTL is **structurally signed-off at product scale**, but the
**routed fit + Fmax is not yet measured** — and that routed number is the single unknown that sizes the box.

**DONE — structural / elaboration sign-off (no LUT/Fmax, no functional golden):**

| Check | What it proves | Target |
|---|---|---|
| **Whole 2-clock Q4_K product top** `glm_q4k_system_cdc` (+ every Q4_K compute/memory/CDC leaf) elaborates and passes `hierarchy -check` + `check -assert` (exit 0 — no unresolved hierarchy, comb loop, multiple driver, or inferred latch) | the assembled Q4_K chip is a **structurally sound netlist** — *structural sign-off, not a sim* | `make synth-glm` |
| **Full 753B UD-Q4_K_XL-shape elaboration** of `glm_model_q4k` (DIM 6144 / L=78 / 256-expert / VOCAB 154880) | the RTL **elaborates cleanly at the true product shape** — type/width / `$clog2` / part-select only; *no stimulus, no golden, no run* | `test/full_config_elab_wrap.v` ([`FULL_CONFIG_ELAB.md`](FULL_CONFIG_ELAB.md)) |
| **Core GEMM is buildable at product scale by construction** — `glm_matmul_q4k` is a **sequential streaming-K fp32-accumulate** fold (one Q4_K super-block = 256 weights; `NSB = ceil(KMAX/256)` super-blocks along K), so it uses **O(1) FP pipes** regardless of K, with the only K-scaling being the small block-accumulator memory (BRAM-able) | the matmul does **not** blow up in FP pipes / FFs at the LM-head K depth — no unrolled per-block explosion | bit-exact gate `make q4k` (`glm_matmul_q4k` **160/160** vs the ggml Q4_K ref `tools/q4k_ref.py` — **not** the real GGUF) |

**PENDING — the gate that actually sizes the box (`[PENDING — needs Vivado]`):**

> The **Q4_K routed fit + Fmax on Vivado (XCKU3P)** — authoritative routed **LUT / FF / DSP / BRAM /
> URAM** utilization **+ per-clock Fmax** on the real KU3P-class part. This is the number that locks the
> FPGA device, and through it the size / thermal / **BOM / per-seat price** ([`PART_SELECTION.md`](PART_SELECTION.md),
> §the `[PENDING — needs Vivado]` rows). It is **not measured** — Vivado is Linux/Windows-only and is not
> installed in this environment. The flow is committed at [`../fpga/vivado/synth_ku3p.tcl`](../fpga/vivado/synth_ku3p.tcl)
> (default part `xcku3p-ffvb676-2-e`, compact config); its outputs drop straight into the PENDING rows.

**No Q4_K LUT/DSP number is quoted in this doc**, because none has been measured. Structural cell
histograms from `make synth-glm` are a *sanity* cross-check, **not** a routed fit — only Vivado P&R
gives a real LUT/DSP/BRAM/Fmax on the target part.

> **Prior FP8 track (branch `fp8`) — methodology carried forward; numbers are NOT Q4_K.** The prior
> FP8 datacenter track ran an open **Gowin `synth_gowin`** synth exploration that established two
> transferable methodology points: (1) **DSP inference** (`MULT18X18`/`MULT9X9`) maps a block-scaled
> quant datapath and **sidesteps the `abc -lut4` accumulator timeout**, and (2) a **sequential
> block-scaled dequant fold is O(1) in FP pipes** (the pattern `glm_matmul_q4k` uses today). Those
> runs produced **FP8-specific** figures — e.g. the `glm_matmul_fp8` leaf @ KMAX=256 mapped in ~77 s
> to **~17.8 K LUT4-equivalent + 20 DSP mults + ~5.4 K DFF** — which live on branch `fp8` and are
> **prior-track measurements, not Q4_K, and not routed** (no P&R Fmax). **A Q4_K re-run is PENDING**
> and supersedes them; do not read the FP8 numbers as the current product's fit.

## The demo ladder (each rung is cheap and de-risks the next)

| Rung | What | Tooling | Proves | Status |
|---|---|---|---|---|
| **L0** | **Synth fit** — structural cell histogram of the compact top | `make synth-glm` (yosys `hierarchy -check` + `check -assert`, have it) | the design is a **sound netlist** that resolves at product scale → sanity area band | **DONE** (structural sign-off, exit 0) — *not* a routed fit |
| **L0′** | **Routed fit + Fmax on the real part** — LUT/FF/DSP/BRAM/URAM + Fmax on **XCKU3P** | **Vivado** (`fpga/vivado/synth_ku3p.tcl`) | the design *fits a class of FPGA* → **BOM band + per-seat price** | **PENDING gate** — Vivado not installed here; the one number that sizes the box |
| **L1** | **P&R one leaf** on the board → real **Fmax** | Gowin EDA / nextpnr / Vivado | real clock → real tok/s extrapolation (tok/s = Fmax ÷ `cyc_per_tok`) | leaf P&R-harness pattern exists (`fpga/gemm_harness.v`, `sm_harness.v` — for `gemm_systolic` / `softmax_unit`) |
| **L2** | **P&R the compact top** → full routed fit + Fmax | Vivado (`fpga/vivado/`) / Gowin (`fpga/gowin/`) | the whole product top places & routes | scaffold ready (needs a top-level bring-up harness to bury the wide memory-side ports) |
| **L3** | **Reduced-config forward on the board** — a few real GLM-5.2 layers, Q4_K weights streamed from on-board Flash/SD → **measured tok/s** | board + a Q4_K weight image (`tools/ckpt_pack_q4k.py`) | *real silicon token at a measured tok/s* — **THE demo** | needs board |

**tok/s is already grounded in RTL, not hand-waved:** the workload is **memory-bandwidth-bound**
(tok/s ≈ sustained weight bandwidth ÷ **~25 GB per token** = ~40 B active params × ~0.6 B/param), and
the memory-stall mechanism + `cyc_per_tok` are measured on real RTL cycles
([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md); stall = `3·FLASH_LAT+9`, `cyc_per_tok` grows with
storage-read latency). (`FLASH_LAT` and the `flash_xbar` read path are committed RTL names for a
**medium-agnostic storage-read abstraction** — address → weight bytes, latency-hidden — that in the
product fronts an **NVMe/PCIe** backend, not a NAND die.) So once L0′/L1/L2 give a routed **Fmax**,
`tok/s = Fmax ÷ cyc_per_tok` is a *measured*, not modeled, single-user number — the thing that
converts the [`ULTRA_PERF.md`](ULTRA_PERF.md) [EST] ladder into fact. On this rung the target is
**~5–8 tok/s [EST]**, set by the sustained weight bandwidth the board can feed (~100 GB/s): the ~14 GB
of per-token **routed experts** stream from NVMe (the wall — they change every token) while the ~9 GB
hot set (attention / dense / shared) caches in **DDR4**. The funded custom board (rung-②, DDR5/HBM) is
where the **~15–40 tok/s [EST]** interactive product lands, and rung-③ (SoC/ASIC at volume) reaches
**~40+ [EST]** — all the *same* verified RTL, only the memory bandwidth the silicon can feed it changes.

## Cost / what a real demo needs

- **L0 is free and DONE** (`make synth-glm`, yosys). **L0′/L1/L2** need a vendor P&R tool: **Vivado ML
  Standard/WebPACK (free, covers XCKU3P)** for the authoritative KU3P fit, or **Gowin EDA (free,
  license-gated)** / open `nextpnr-himbaechel` for the Tang Mega bring-up board. The Tang Mega 138K Pro
  (**~$200–300**) is only needed to *program* a bring-up run.
- **L3 (the money shot)** needs a **KU3P-class board (DDR4 + NVMe)** + a Flash/SD-resident Q4_K weight
  image (`tools/ckpt_pack_q4k.py` produces the RTL weight-memory layout `weight_loader_q4k.v` reads)
  and a reduced config (a few layers) — **not** the product's full **1–4 TB NVMe model store** or the
  **DDR5/HBM + NVMe/PCIe (M.2) host controller** (those are **rung-② custom-board / vendor-IP items** —
  see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md); DDR5/HBM is a funded-board spec, not the near-term
  proof). The demo is a **reduced-config proof that real weights produce real tokens on real silicon at
  a measured rate**, not the shippable box.

> **Weight-image honesty (per [`Q4K_SYSTEM_PLAN.md`](Q4K_SYSTEM_PLAN.md)):** `tools/ckpt_pack_q4k.py`
> round-trips its gen → pack → unpack against a **synthetic tiny GGUF it fabricates in-memory**, proven
> bit-exact vs the ggml dequant mirrors in `tools/q4k_ref.py` — **not** against the real 467 GB
> published GGUF (never downloaded) and **not** through llama.cpp. The RTL weight path is **Q4_K-only**
> (`weight_loader_q4k.v` has no Q6_K/Q8_0/F16 consumer), so a real **UD-Q4_K_XL** checkpoint's sensitive
> higher-precision tensors are **[OPEN]** — bit-exactness vs the file people download is **not**
> validated.

## Why this is the investable lever

An investor discounts every `[EST]`. The demo ladder converts three of them to fact, cheaply:
1. **Fit/BOM** (L0′ routed Vivado fit on XCKU3P): "it fits a $X FPGA" → the per-seat price the
   [`ICP.md`](ICP.md) economics need. *(The structural netlist already resolves at product scale —
   `make synth-glm`; only the routed LUT/DSP/BRAM/Fmax is left.)*
2. **Single-user tok/s** (L1/L2 Fmax × measured `cyc_per_tok`): the rung-① proof speed (**~5–8 tok/s
   [EST]**, DDR4 + NVMe), measured not modeled — the funded rung-② board is where **~15–40 tok/s [EST]**
   lands ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)).
3. **Real tokens on silicon** (L3): the whole thesis, demonstrable on a desk.

Paired with **one signed design-partner LOI** from the primary ICP (a law-firm innovation team) —
**[PENDING]** — that is the pre-seed package: *a measured box + a customer who wants it.*
