# Gowin vendor flow in Docker (Linux x86_64) — the burnable-bitstream path

**Why this exists.** The open flow (`fpga/nextpnr/run.sh`, oss-cad-suite) runs on
this Mac and gives a resource-fit estimate, but **nextpnr-himbaechel's GW5A clock
routing is incomplete** — every *clocked* design fails to route on GW5AST-138C
(a single flip-flop already fails; combinational designs route + pack to `.fs`
fine). Our accelerator is entirely clocked, so the open flow cannot emit a
burnable clocked bitstream for the target.

**The fix is the vendor tool.** Gowin's own `gw_sh` (GowinSynthesis + P&R) fully
supports GW5A clock spines → real routed LUT/DSP/BSRAM + Fmax + a burnable `.fs`.
Gowin EDA is Linux/Windows only; this Intel Mac runs the **Linux x86_64** build
**natively** in Docker (no emulation — the host is x86_64).

Confirmed target: **`GW5AST-LV138PG484AC1/I0`** (Tang Mega 138K Pro, package
PBGA484A — verified against the Gowin pinout `GW5AST-138C/PBGA484A.json`).

---

## One-time setup (you do these — Gowin requires a login, scripts can't fetch it)

### 1. Get Gowin EDA (Linux, Education = free, GW5A-capable)
Download from <https://www.gowinsemi.com/en/support/download_eda/> — the
**Education** edition, **Linux** tarball (e.g. `Gowin_V1.9.11.xx_Education_linux.tar.gz`).
Save it as:
```
fpga/gowin/docker/gowin_linux.tar.gz
```

### 2. License (node-locked to the container's MAC)
GW5A Education needs a license tied to a MAC address. We **pin the container MAC**
so it stays valid (`LOCK_MAC` in `run_docker.sh`, default `02:42:ac:11:00:99` — any
locally-administered MAC works). Request a GW5A Education license for **exactly that
MAC** from Gowin, and save it as:
```
fpga/gowin/docker/gwlicense.lic
```
> To use a different MAC, set `LOCK_MAC=xx:xx:xx:xx:xx:xx` when running and license
> that one instead. The container MAC (not the Mac's en0) is what matters.

### 3. Build the image
```sh
docker build -t aipu-gowin fpga/gowin/docker
```
(The Dockerfile installs the Qt/X runtime libs `gw_sh` needs and unpacks Gowin to
`/opt/gowin`.)

> `.gitignore` these — never commit them: `gowin_linux.tar.gz`, `gwlicense.lic`
> (license file + vendor binary are not ours to redistribute).

---

## Run

```sh
fpga/gowin/run_docker.sh                 # default config, synth + P&R  -> routed fit + Fmax + .fs
COMPACT=1 fpga/gowin/run_docker.sh       # compact (byte-identical) config
FLOW=syn  fpga/gowin/run_docker.sh       # synthesis-only (resource fit; no pins needed)
```

Outputs land under `./impl/`; the wrapper auto-extracts LUT/DSP/BSRAM/FF + Fmax and
points at the `.fs`. Record the numbers in [`../../README.md`](../README.md).

### Pins before a real board flash
`FLOW=all` (P&R) needs pin constraints. `fpga/gowin/aipu.sdc` has the clock
constraints; **pin *locations* go in a `.cst`** matched to the **Tang Mega 138K Pro**
board (its clock crystal / LED / UART pins). The valid package pins for PBGA484A are
in the Gowin pinout JSON (`.../device/GW5AST-138C/PBGA484A.json`, `PIN_DATA` entries
with `"TYPE":"I/O"`). A bitstream with wrong pins configures but does nothing useful,
so finalize the `.cst` against the board's schematic before flashing.

### Flash to the board (later, when you have it)
```sh
openFPGALoader -b tangmega138kpro impl/pnr/*.fs      # or Gowin Programmer
```

---

## What this de-risks

- **The clock-routing wall** (open-flow blocker) → gone; vendor P&R routes GW5A clocks.
- **Real routed fit + Fmax** → the D0.2 answer (FPGA class → BOM → per-seat price).
- **A burnable `.fs`** → everything up to physically flashing a board is done here.

The only remaining physical steps: a real Tang Mega 138K Pro board, board-matched
pins, and (for a functional demo) weight provisioning + the P1.1 fidelity gate.
