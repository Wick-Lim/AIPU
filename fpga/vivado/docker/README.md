# Vivado-in-Docker — KU3P routed fit

Runs the authoritative routed fit (LUT/FF/DSP/BRAM/URAM + Fmax) for the product
FPGA (**Kintex UltraScale+ XCKU3P**) via AMD/Xilinx Vivado in a Linux container.
Fills the `[PENDING — needs Vivado]` rows in [`../../../docs/PART_SELECTION.md`](../../../docs/PART_SELECTION.md)
and the "Routed fit + Fmax · Vivado · next" chip in the public site §04.

## Why Docker
Vivado is **Linux/Windows only — no macOS build**. This Intel Mac runs the Linux
x86_64 Vivado natively in Docker. The image is **slim** (runtime libs only); the
~100 GB Vivado install lives on the **host** and is mounted in — so the image
stays small and the install persists.

## ⚠️ Disk (the current blocker on this Mac)
Vivado is **~100 GB** installed (a UltraScale+-only device-limited install is
~30–50 GB). The Docker host needs that much free for the Vivado dir.
- **This Mac has ~22 GB free (98 % full)** and the Docker VM ~28 GB → **won't fit.**
- **Realistic paths:** (a) a **cloud Linux Docker host** (free tier VM + disk), or
  (b) **free ~100 GB on the Mac** first, then run locally.

The scaffold is host-agnostic — identical `docker build` / `run_docker.sh` on a
cloud box or a freed-up Mac. (On a dedicated cloud VM you can also skip Docker and
run `vivado -mode batch -source fpga/vivado/synth_ku3p.tcl` directly.)

## One-time setup (you do these — AMD needs a login)
1. **Install Vivado on the host.** Use AMD's Unified Installer → **Vivado ML
   Standard (FREE, covers KU3P)**. Pick a device-limited install (UltraScale+ only)
   to keep it smaller. Note the path, e.g. `~/Xilinx/Vivado/2024.2` (must contain
   `settings64.sh`).
2. **License.** KU3P is free (ML Standard). If AMD prompts, generate a free license
   at <https://www.xilinx.com/getlicense> and export `XILINXD_LICENSE_FILE`.
3. **Build the image:** `docker build -t aipu-vivado fpga/vivado/docker`

## Run
```
VIVADO_DIR=~/Xilinx/Vivado/2024.2 fpga/vivado/run_docker.sh          # compact config
CFG=default VIVADO_DIR=~/Xilinx/Vivado/2024.2 fpga/vivado/run_docker.sh   # full config
PART=xcku3p-ffvb676-2-e VIVADO_DIR=... fpga/vivado/run_docker.sh     # confirm your board's exact part
```
Reports land in `fpga/vivado/out/` (`util_*` = utilization, `timing_*` = WNS/TNS →
Fmax). Copy the numbers into `docs/PART_SELECTION.md` and the site §04.

## The flow itself
[`../synth_ku3p.tcl`](../synth_ku3p.tcl) reads the 24 GLM_CDC_SRCS, applies the
compact-config generics (PE_N=2 / DDR_NCH=2 / KV_RESIDENT=8 / EFIFO_DEPTH=8 /
CACHE_SLOTS=2 — byte-identical to the committed config), then `synth_design →
opt/place/route → report_utilization + report_timing_summary`.
