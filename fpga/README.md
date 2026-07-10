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
| `synth_ku3p.tcl` | Vivado batch: read the 24 Q4_K sources → **(1)** `synth_design` of the raw product top `glm_q4k_system_cdc` → `util_synth.rpt` (pure resource fit, no pins); **(2)** `synth_design` + place + route of `bringup_harness` → `util_routed.rpt` + `timing.rpt` (routed Fmax) |
| `bringup_harness.v` | SYNTHESIZABLE P&R harness: wraps the exact product top but buries its thousands of wide memory-side ports (DDR/flash/KV/dequant buses + `logits`) behind an on-chip LFSR/CRC, so I/O collapses to 5 clock/control pins + 1 output → **routable**. Non-constant LFSR drivers keep the datapath from being pruned, so the routed fit is real. Verified: iverilog elaborates + yosys `hierarchy -check`/`check -assert` clean (no comb loop from the registered feedback) |
| `constraints.xdc` | `create_clock` for `host_clk` + `core_clk` + async CDC groups (timing/CDC only — **pin locations are board-specific**, add from your board's master XDC). Same clock/reset port names on both the system and the harness, so it applies to either top |
| `run_fit.sh` | runs the tcl in the Vivado Docker container (pins the container MAC for the license hostid, mounts `Xilinx.lic`, installs the tool's runtime libs, sources `settings64.sh`) |
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

> The portal now names the free tier **"Vivado Basic Tier License, Node Locked"**
> (the ML-Standard/WebPACK successor); it grants `Vivado_Synthesis` +
> `Vivado_Implementation` + `Vivado_Simulation`, which covers this whole flow.

### Docker Desktop resource footguns (learned the hard way)
Docker Desktop **validates** `settings.json` at boot. A resource value it rejects
(memory or disk **exceeding what the host can back** — e.g. a 150 GB VM disk on a
111 GB-free host) does not get clamped: Docker **factory-resets every resource
setting** (mem → 8 GB, disk → 60 GB) **and recreates the VM disk — wiping all
volumes**, including the 57 GB Vivado install. Rules that keep it stable:
1. Edit `settings.json` only with Docker **fully stopped** (wait for
   `com.docker.backend` to exit) — a shutdown flush overwrites live edits.
2. Keep `memoryMiB` ≥ 32768 (the compact-chip synth OOM-kills below that even
   with `maxThreads 4`) and `diskSizeMiB` comfortably **under** host free space.
3. Keep the extracted installer + `install_config.txt` (`/Users/Shared/xilinx_setup/`)
   — if the volume is ever wiped, the reinstall is one ~1 h batch command
   (`xsetup -a XilinxEULA,3rdPartyEULA -b Install -c install_config.txt`).

### Docker crash workaround (handled automatically by `run_fit.sh`)
With a valid license, Vivado 2026.1 in Docker then **aborts at launch** with
`realloc(): invalid pointer` — the FlexLM checkout (`libXil_lmgr11.so`) scans
network devices through `libudev`, and Vivado's bundled `libtcmalloc.so.4`
(loaded as `NEEDED` by the main binary, pure malloc interposition — zero direct
`tc_*` references) mixes allocators with glibc-internal allocations inside that
scan. Verified on Ubuntu 20.04 and 22.04 images; `LD_PRELOAD`/`/etc/ld.so.preload`
shims do **not** help because lmgr `dlopen`s libudev directly. The fix that works:
bind-mount an **empty stub** `.so` over the bundled `libtcmalloc.so.4` (removes the
interposition; everything runs on plain glibc). `run_fit.sh` builds and mounts the
stub automatically. Feature libs additionally need `libpixman-1-0 libcairo2
libpango-1.0-0 libglib2.0-0 libfreetype6 libfontconfig1` (also installed by the
script).

## Adjust for your board
- **Part**: edit `PART` in `synth_ku3p.tcl` (package/speed grade, e.g. `xcku3p-ffvb676-2-e`).
- **Fmax sweep**: tighten `core_clk` period in `constraints.xdc`; achieved Fmax = `1/(period − WNS)` (WNS from `out/timing.rpt`).
- **Pins**: add `set_property PACKAGE_PIN`/`IOSTANDARD` from your dev board's master XDC.

## Honest notes
- **First real fit** — expect Vivado to surface synthesis-specific issues the structural
  yosys gate did not (width/timing/inference warnings); iterate.
- **Wide top-level ports** (resolved): `glm_q4k_system_cdc` exposes thousands of
  memory-/logits-side bits (`logits`=VOCAB·16, `h_state`, DDR/Flash/KV/dequant buses).
  Full P&R would fail on **I/O count** on a real package. `bringup_harness.v` buries all
  of them behind an on-chip LFSR/CRC (expose only host pins), so the tcl's step (2) routes
  the harness for the real Fmax; step (1) still reports pure-product resources synth-only.
- **RAM**: compact-system P&R is memory-heavy; if `route_design` OOMs on the Docker
  allocation, raise Docker Desktop memory or run synth-only first.

## Results — MEASURED (Vivado ML 2026.1, 2026-07, compact config + `ACT_HW=1`)
Product top `glm_q4k_system_cdc`, `xcku3p-ffvb676-2-e`, synth-only utilization
(`util_synth.rpt`); harness P&R **completed** (place + route clean), timing from
`timing.rpt`.

| Resource | Used | Avail (XCKU3P) | Util % |
|---|---|---|---|
| CLB LUT | **141,710** (138,242 logic + 3,468 LUTRAM) | 162,720 | **87.1%** |
| CLB Register (FF) | 99,597 | 325,440 | 30.6% |
| Block RAM (36Kb) | 0 | 360 | 0% (all storage in LUTRAM/FF) |
| DSP48E2 | 421 | 1,368 | 30.8% |
| **Fits XCKU3P?** | **YES** — places and routes | | |
| Routed WNS `core_clk` @5ns | **−93.1 ns** | | |
| **Achieved Fmax `core_clk`** | **~10.2 MHz** (1/98.4ns) | 200 MHz target | |
| Hold (WHS) | +0.010 ns (met) | | |

**Fit history (the iteration the README predicted):**
1. Round 1–2: `mla_attn_q4k` static part-select bounds (Synth 8-524, fixed
   result-invariantly) → Docker VM OOM at Technology Mapping (fixed: 32 GB VM).
2. Round 3: synthesized **220.8K → 216.1K LUTs post-opt = 136% of KU3P**, DRC
   UTLZ-1 at placement. Hierarchical report: the three `glm_act` instances were
   **84.2K LUTs (38% of the chip)** — each lane replicates the full 6-stage fp32
   exp/recip pipeline (router sigmoid alone 48.3K > all of attention 43.8K).
3. Round 4: **`ACT_HW=1`** (lane-serialized `glm_act`, result-invariant — same
   1155-test golden passes bit-exact) → **141.7K LUTs, fits at 87.1%**, full P&R.

**Honest read of the Fmax number.** The RTL was written bit-exact-first: fp32
ops are combinational `glm_fp.vh` function chains, so single-cycle cones run
deep — the critical path is `mla_attn_q4k → rope_interleave_unit y_out` with
**382 logic levels** (9 serial DSP multiplies + 72 CARRY8), 98.4 ns; 34K
endpoints fail the 5 ns target. ~10 MHz still decodes bit-exact tokens (fine
for a first bring-up correctness demo), but reaching the 200 MHz-class clock
the tok/s roofline assumes is a **pipelining campaign**: latency-only
repipelining of the deep fp32 cones (rope, softmax, rmsnorm, dequant+MAC),
stage by stage, re-proving bit-exactness with the same goldens each time — the
`glm_act` S2a/S2b split and `glm_fp_pipe` are the in-repo pattern for exactly
this transformation.
