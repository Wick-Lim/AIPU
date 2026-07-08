# `fpga/` — XCKU3P fit (Vivado) for the GLM-5.2 **Q4_K** accelerator

The **[PENDING] hardware-fit gate**: turn the verified Q4_K RTL into a **real, routed
FPGA fit** — actual LUT / DSP / BRAM utilization and routed **Fmax** on a Kintex
UltraScale+ **XCKU3P** — the number iverilog (sim) and yosys (`check-assert`,
structural) cannot give. Getting this number sets the FPGA class → the box's size,
thermal budget, BOM, and per-seat price.

> **Track note.** This is the **current Q4_K / XCKU3P / Vivado** flow. A prior scaffold
> targeted **FP8 on Gowin GW5AT-138** (`gowin/`, `nextpnr/`); that referenced the FP8
> datapath now removed from `main` (preserved on branch `fp8`) and is superseded here.

## What it synthesizes
`glm_q4k_system_cdc` — the whole 2-clock product top (compute die `glm_model_q4k` +
memory system + CDC) — in the **compact** config (result-invariant resource params:
`PE_N 4→2, DDR_NCH 4→2, KV_RESIDENT 16→8, EFIFO_DEPTH 16→8, CACHE_SLOTS 4→2`). The
decoded token is byte-identical to the default config (proven in sim); only the
parallelism/capacity shrinks to target a small dev-board part.

## Files
| File | What |
|---|---|
| `synth_ku3p.tcl` | Vivado batch: read the 24 Q4_K sources → `synth_design` (compact generics) → place → route → utilization + timing reports |
| `constraints.xdc` | `create_clock` for `host_clk` + `core_clk` + async CDC groups (timing/CDC only — **pin locations are board-specific**, add from your board's master XDC) |
| `run_fit.sh` | runs the tcl in the Vivado Docker container (installs the tool's runtime libs, sources `settings64.sh`) |
| `out/` | reports land here (`util_synth.rpt`, `util_routed.rpt`, `timing.rpt`, checkpoints) |

## Run
Vivado ML must be installed into the `xilinx_install` docker volume first. Then from
the repo root:
```bash
bash fpga/run_fit.sh
```

### License (one-time, free)
Vivado ML **2026.1 refuses to launch without a valid license** — not even a trivial
batch script (verified: a bare `puts` script errors `valid license was not found`).
The **free "Vivado ML Standard"** edition covers XCKU3P (KU3P/KU5P are the free-tier
UltraScale+ parts), but you must generate a $0 license once:
1. <https://www.xilinx.com/getlicense> → sign in → **Vivado ML Standard** (Node-Locked, free).
2. Host ID: `run_fit.sh` pins the container MAC to `02:42:c0:a8:64:02`, so give the
   FlexLM host id **`0242c0a86402`** when the portal asks.
3. Save the generated `Xilinx.lic` to `/Users/Shared/xilinx/Xilinx.lic` (or export
   `XILINX_LICENSE=/path/to/Xilinx.lic`), then re-run `bash fpga/run_fit.sh`.

`run_fit.sh` auto-mounts the `.lic` and sets `XILINXD_LICENSE_FILE`; the fixed MAC
keeps the node-locked license valid across container runs.

## Adjust for your board
- **Part**: edit `PART` in `synth_ku3p.tcl` (package/speed grade, e.g. `xcku3p-ffvb676-2-e`).
- **Fmax sweep**: tighten `core_clk` period in `constraints.xdc`; achieved Fmax = `1/(period − WNS)` (WNS from `out/timing.rpt`).
- **Pins**: add `set_property PACKAGE_PIN`/`IOSTANDARD` from your dev board's master XDC.

## Honest notes
- **First real fit** — expect Vivado to surface synthesis-specific issues the structural
  yosys gate did not (width/timing/inference warnings); iterate.
- **Wide top-level ports**: `glm_q4k_system_cdc` exposes many memory-/logits-side bits
  (`logits`=VOCAB·16, `h_state`, DDR/Flash buses). Full P&R may fail on **I/O count** on a
  real package unless those wide ports are buried in a bring-up harness (model the memory
  responses internally, expose only the host pins). Until that harness exists, **synth-only**
  (`report_utilization` after `synth_design`) gives the resource fit without pin placement;
  routed Fmax needs the harness. `run_fit.sh` attempts full P&R — if it fails on I/O or OOM,
  fall back to synth-only.
- **RAM**: compact-system P&R is memory-heavy; if `route_design` OOMs on the Docker
  allocation, raise Docker Desktop memory or run synth-only first.

## Results — TEMPLATE, fill in after running
| Resource | Used | Avail (XCKU3P) | Util % |
|---|---|---|---|
| CLB LUT | `TBD` | ~162K | `TBD` |
| CLB Register (FF) | `TBD` | ~325K | `TBD` |
| Block RAM (36Kb) | `TBD` | ~360 | `TBD` |
| DSP48E2 | `TBD` | **600** | `TBD` |
| Fmax `core_clk` | `TBD` MHz | (target from constraints.xdc) | — |
| Fmax `host_clk` | `TBD` MHz | — | — |
| **Fits XCKU3P?** | `TBD` | | |
