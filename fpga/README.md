# `fpga/` — FPGA fit / vendor-flow scaffold for the GLM-5.2-FP8 accelerator

**This is Phase D0.2 "FPGA fit"** from
[`../docs/USBC_PRODUCT_PLAN.md`](../docs/USBC_PRODUCT_PLAN.md) — *the single
biggest unknown in the whole product plan.* Getting the real FPGA utilization is
what sets the **FPGA class**, and the FPGA class sets the device's **size,
thermal budget, BOM, and price** (§8 of that doc: *"cost is FPGA-dominated"*).
Everything downstream is bounded by the number this flow returns.

> **Honest status: nothing here has been run.** Gowin EDA is not installed in the
> environment where this scaffold was written, so there are **no measured numbers
> yet**. This directory is a **ready-to-run scaffold**: a user who *has* Gowin EDA
> (free, login/license-gated) can run it to get the real fit. The results table
> below is a **template to fill in** — do not treat any cell as measured until you
> run it.

---

## Why this exists (the yosys wall)

The repo's own structural gate `make synth-glm` only *elaborates* the product top
and runs `check -assert`; it deliberately does **not** emit a LUT count, because
**yosys 0.66 cannot map the FP8 datapaths through `abc -lut4`** — it times out on
`glm_matmul_fp8`'s accumulator banks. This is documented in
[`../docs/PHYSICAL_SKY130.md`](../docs/PHYSICAL_SKY130.md) ("FPGA resource fit
(ECP5) — partial, honest"), which was able to map only the *memory-system
controllers*: those **6 controllers alone are ~71,475 LUT4 ≈ 85% of an ECP5-85**,
so the full system needs a larger FPGA — but the *compute die's* mapped size
remains **unmeasured** because of the tooling wall.

**A vendor flow breaks that wall.** Gowin's `GowinSynthesis` (and, as a fallback,
a newer yosys + nextpnr-himbaechel) is a different mapper than the repo-baseline
yosys 0.66 `abc`, and can give the real LUT/DSP/BSRAM/Fmax numbers. That is the
entire reason this directory exists. *(If GowinSynthesis itself struggles on the
FP8 math, that outcome is itself a finding — record it.)*

## Target device

**Gowin GW5AT-138** — the **Sipeed Tang Mega 138K Pro** board (~138K LUT, on-board
DDR3). This is the mid-FPGA candidate referenced throughout
[`../docs/MINIATURIZATION.md`](../docs/MINIATURIZATION.md) (Phase E1 "gated on
Gowin EDA for the real LUT/DSP/BSRAM numbers on GW5AT-138"). The controllers'
~71,475 LUT4 anchor above is *why* size matters: the default config may or may not
fit 138K — which is exactly what D0.2 measures. If it overflows, the **compact**
config (below) and/or a larger FPGA are the levers.

---

## Directory contents

| File | What it is |
|------|-----------|
| [`gowin/build_gowin.tcl`](gowin/build_gowin.tcl) | **Primary flow.** Gowin `gw_sh` Tcl script: sets device, adds the 24 `src/` files + include dir, sets top, adds the SDC, runs synthesis (and optionally P&R), points at the resource/timing reports. `COMPACT=` and `FLOW=` env flags. |
| [`gowin/aipu.sdc`](gowin/aipu.sdc) | Timing constraints: `create_clock` for `host_clk` + `core_clk`, and `set_clock_groups -asynchronous` (the two domains cross only via `cdc_async_fifo`). Pin-assignment notes (pins go in a `.cst`, not the SDC). |
| [`gowin/glm_fp8_system_cdc_compact.v`](gowin/glm_fp8_system_cdc_compact.v) | Thin passthrough wrapper that fixes the 5 compact parameters (Gowin `gw_sh` can't override top params from Tcl the way yosys `-chparam` can). Used when `COMPACT=1`. Touches nothing in `src/`. |
| [`nextpnr/README.md`](nextpnr/README.md) | **Fallback flow.** Fully open-source path: yosys `synth_gowin` + `nextpnr-himbaechel` for GW5A, with the honest caveat that it can hit the same FP8/`abc` wall and that GW5A support is newer/approximate. |
| `gemm_harness.v`, `sm_harness.v` | Pre-existing synthesizable P&R harnesses for two *leaf* units (`gemm_systolic`, `softmax_unit`). They bury wide tile-memory ports so those units can be placed & routed standalone — the same pattern you'd need to fully P&R the whole top (see below). Not part of the GLM-5.2-FP8 top flow. |

---

## How to run

You need **Gowin EDA** installed and licensed, with `gw_sh` on your `PATH`. Run
from the **repo root** (paths in the script resolve relative to itself, so any CWD
works, but the root is cleanest):

```sh
# Default (committed) config, synthesis-only -> the RESOURCE FIT (LUT/DSP/BSRAM/FF)
gw_sh fpga/gowin/build_gowin.tcl

# Compact miniaturization config (byte-identical token, smaller area)
COMPACT=1 gw_sh fpga/gowin/build_gowin.tcl

# Attempt the FULL flow (synthesis + Place & Route + routed Fmax)
FLOW=all gw_sh fpga/gowin/build_gowin.tcl

# Both together
COMPACT=1 FLOW=all gw_sh fpga/gowin/build_gowin.tcl
```

**Before the first run, confirm the device part string** in
`gowin/build_gowin.tcl` (`set PART …`). It is a clearly-marked **placeholder**;
the comment there explains how to find the exact GW5AT-138 part number for the
Tang Mega 138K Pro (Gowin IDE device picker, `get_device_info`, or an existing
`.gprj`).

The open-source fallback is in [`nextpnr/README.md`](nextpnr/README.md).

### Full P&R vs synthesis-only (important)

- **`FLOW=syn` (default) — synthesis-only — is the D0.2 answer.** It reports
  LUT/DSP/BSRAM/registers with **no pin placement required**, so it always runs.
- **`FLOW=all` (full Place & Route)** additionally gives routed **Fmax**, but the
  raw top `glm_fp8_system_cdc` exposes **thousands of memory-side port bits**
  (e.g. `mem_resp_data` = `DDR_NCH*DDR_DATA_W` = up to 1024 bits, `logits` =
  `VOCAB*16` = 4096 bits, `h_state` = 2048 bits, …). No package has that many
  user I/O, so P&R will **fail on I/O** unless those wide ports are first **buried
  inside a bring-up harness** — exactly what `gemm_harness.v` / `sm_harness.v` do
  for their leaf units. Writing that top-level harness (drive the host interface;
  model DDR5/Flash/loader responses internally as BRAM/regs; expose only a
  handful of pins) is a follow-on task; until then, use `FLOW=syn` for the fit.

---

## What to measure and record

Per config (default and compact), from the reports under `./impl/`:

| Metric | Where (Gowin) | Notes |
|--------|---------------|-------|
| **LUT** (LUT4 / ALU) | `impl/gwsynthesis/*_syn.rpt*` | primary area number |
| **Registers (FF)** | same | flops used |
| **BSRAM** (block RAM) | same | KV ring, FIFOs, caches, weight staging |
| **DSP / MULT** | same | FP8 matmul multipliers, if inferred |
| **Utilization %** | same | vs GW5AT-138 totals — does it fit? |
| **Fmax per clock** | `impl/pnr/*timing*` (only `FLOW=all`) | `host_clk` and `core_clk` separately |

Cross-check against the yosys structural cell count from `make synth-glm` /
`make synth-glm-compact` (flattened cell histogram) for a sanity comparison.

---

## Results table — **TEMPLATE, fill in after running**

> All cells are `TBD` — **nothing here is measured.** Fill each in from your
> Gowin run and note the tool version + exact part string used.

**Environment**

| Field | Value |
|-------|-------|
| Gowin EDA version | `TBD` |
| Device part string | `TBD` (confirm from IDE / `get_device_info`) |
| Board | Sipeed Tang Mega 138K Pro (GW5AT-138) |
| Flow | `TBD` (syn / all) |
| Date | `TBD` |

**Fit — DEFAULT config** (PE_N=4, DDR_NCH=4, KV_RESIDENT=16, EFIFO_DEPTH=16, CACHE_SLOTS=4)

| Resource | Used | Available (GW5AT-138) | Util % |
|----------|------|-----------------------|--------|
| LUT | `TBD` | ~138K | `TBD` |
| Registers (FF) | `TBD` | `TBD` | `TBD` |
| BSRAM | `TBD` | `TBD` | `TBD` |
| DSP / MULT | `TBD` | `TBD` | `TBD` |
| Fmax `host_clk` | `TBD` MHz | target 100 MHz | — |
| Fmax `core_clk` | `TBD` MHz | target ~67 MHz | — |
| **Fits GW5AT-138?** | `TBD (yes/no)` | | |

**Fit — COMPACT config** (PE_N=2, DDR_NCH=2, KV_RESIDENT=8, EFIFO_DEPTH=8, CACHE_SLOTS=2 — byte-identical token)

| Resource | Used | Available (GW5AT-138) | Util % |
|----------|------|-----------------------|--------|
| LUT | `TBD` | ~138K | `TBD` |
| Registers (FF) | `TBD` | `TBD` | `TBD` |
| BSRAM | `TBD` | `TBD` | `TBD` |
| DSP / MULT | `TBD` | `TBD` | `TBD` |
| Fmax `host_clk` | `TBD` MHz | target 100 MHz | — |
| Fmax `core_clk` | `TBD` MHz | target ~67 MHz | — |
| **Fits GW5AT-138?** | `TBD (yes/no)` | | |

**Decision (D0.2 gate):** from these numbers, pick the FPGA class and record
whether the default fits, whether compact is needed, or whether a larger FPGA is
required — then propagate to size / thermal / BOM / price in
[`../docs/USBC_PRODUCT_PLAN.md`](../docs/USBC_PRODUCT_PLAN.md) §6–§8.

---

## Caveats (honest)

- **Un-run scaffold.** Nothing here has been executed; Gowin isn't installed in
  the authoring environment. The numbers come from *you* running Gowin.
- **The yosys `abc`/`-lut4` wall** on the FP8 datapaths is *why* a vendor flow is
  needed ([`../docs/PHYSICAL_SKY130.md`](../docs/PHYSICAL_SKY130.md)). Gowin's
  synthesizer may still struggle on the FP8 math — if so, that's a finding to
  record, not a script bug.
- **Part string is a placeholder.** Confirm the exact GW5AT-138 package/speed/
  grade for your board before running (comment in `build_gowin.tcl`).
- **Board pinout is user-provided.** Pin locations live in a board-specific
  `.cst`; only commented placeholders are given (`aipu.sdc`). Not needed for the
  synthesis-only resource fit.
- **Full P&R needs a bring-up harness** to bury the wide memory-side ports; the
  synthesis-only fit does not. See "Full P&R vs synthesis-only" above.
- **`gw_sh` command spellings vary by Gowin version.** The script targets the
  common `set_device`/`add_file`/`set_option`/`run` API and flags every spot you
  may need to adjust.

## References

- [`../docs/USBC_PRODUCT_PLAN.md`](../docs/USBC_PRODUCT_PLAN.md) — Phase **D0.2**
  FPGA fit (the gating unknown), §6 FPGA class, §8 BOM/pricing.
- [`../docs/MINIATURIZATION.md`](../docs/MINIATURIZATION.md) — the **compact**
  config (L0), and Phase E1 "gated on Gowin EDA … on GW5AT-138".
- [`../docs/PHYSICAL_SKY130.md`](../docs/PHYSICAL_SKY130.md) — the yosys ECP5
  partial fit and the ~71,475 LUT4 controllers anchor / `abc` wall.
- Repo `Makefile` targets: `synth-glm`, `synth-glm-compact`, `sim-glm-compact`
  (the structural + byte-identical-token gates this fit builds on).
