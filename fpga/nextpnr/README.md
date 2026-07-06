# Open-source fallback flow — yosys `synth_gowin` + nextpnr-himbaechel (GW5A)

This is the **fully open-source alternative** to the Gowin `gw_sh` vendor flow in
[`../gowin/`](../gowin/). Use it if you do **not** have Gowin EDA, or to
cross-check the vendor numbers. It targets the same part — Gowin **GW5AT-138**
(Sipeed Tang Mega 138K Pro) — via the [YosysHQ/apicula](https://github.com/YosysHQ/apicula)
bitstream documentation and the **himbaechel** nextpnr architecture.

> **MEASURED STATUS (run on this Mac via oss-cad-suite).**
> - ✅ **Toolchain works end-to-end** — `synth_gowin → nextpnr-himbaechel → gowin_pack`
>   produced a real **`.fs` bitstream** for a *combinational* test design on GW5AST-138C.
> - ✅ **Device + pins confirmed** — part `GW5AST-LV138PG484AC1/I0` (Tang Mega 138K Pro,
>   PBGA484A); the 454 package pins came from the Gowin pinout `GW5AST-138C/PBGA484A.json`.
> - ✅ **Placement/fit works for clocked designs** — nextpnr reports device utilisation +
>   **Fmax** during placement *before* routing (a blinky: LUT4 39/138240, 482 MHz).
> - ❌ **GW5A CLOCK ROUTING is incomplete in nextpnr-himbaechel** — every *clocked* design
>   fails to route (even a single flip-flop: "Failed to route net 'clk...' to CLK sink").
>   Our accelerator is entirely clocked, so **this open flow cannot emit a burnable clocked
>   `.fs`** for GW5A yet. The fix is the **vendor Gowin flow** ([`../gowin/docker/`](../gowin/docker/)),
>   whose P&R fully supports GW5A clock spines. Use THIS flow for the resource-fit estimate
>   (synth `stat` + nextpnr placement util/Fmax); use the vendor flow for the routed `.fs`.
>
> - **The FP8/`abc` wall is retired.** `synth_gowin` maps the FP8 datapath by
>   inferring hardware **DSPs** (not LUT-mapping every multiply), and the O(NB²)
>   dequant that blocked whole-system synth is **fixed** (O(1), bit-exact). So
>   step 1 (synth) now completes and gives a real LUT/DSP/BSRAM estimate. A
>   **recent** yosys is still wanted — it infers **BSRAM** for the O(NB) `accx`
>   block-accumulator (the repo-baseline 0.66 flattens it to registers); oss-cad-suite
>   ships one, plus the GW5A BRAM map (`brams_map_gw5a.v`).
> - **GW5AT-138 IS in apycula** as **`GW5AST-138C`** (chipdb present:
>   `chipdb-GW5AST-138C.bin`), so nextpnr routing is possible — but GW5A timing
>   models are **newer/approximate**, so any Fmax here is a rough estimate, not a
>   sign-off. The **Gowin vendor flow stays authoritative**; this is the
>   here-and-now cross-check.
>
> Just run [`run.sh`](run.sh) (see below); the sketch that follows documents what
> it does.

## Quick start (`run.sh`)

```sh
# 1. Install oss-cad-suite (macOS x86_64 / Linux) -- recent yosys + nextpnr-himbaechel + apycula:
URL=$(gh api repos/YosysHQ/oss-cad-suite-build/releases/latest \
      --jq '.assets[]|select(.name|test("darwin-x64")).browser_download_url')   # darwin-x64 for Intel mac
curl -L "$URL" -o oss.tgz && tar xzf oss.tgz -C $HOME                            # -> ~/oss-cad-suite

# 2. Run the fit (from the repo root):
fpga/nextpnr/run.sh                 # compact config, synth-only -> LUT/DSP/BSRAM
fpga/nextpnr/run.sh default         # default (committed) config
fpga/nextpnr/run.sh compact pnr     # + attempt nextpnr routing (GW5AST-138C, approximate Fmax)
```

Outputs land in `fpga/nextpnr/out/` (`stat_*.txt` = the resource fit; `*.json` = netlist).
**Device string for step 2:** the Tang Mega 138K Pro's GW5AT-138 is **`GW5AST-138C`** in
apycula/himbaechel.

## Prerequisites

- **yosys** (as new as you can get — a recent build is more likely to clear the
  FP8 `abc` wall than the repo-baseline 0.66) with the `synth_gowin` pass.
- **nextpnr-himbaechel** built with the **gowin** uarch, plus the **apicula**
  chip database (the himbaechel gowin chipdb for the GW5A family).
- The GW5AT-138 device name string for himbaechel (analogous to the vendor part;
  confirm against `nextpnr-himbaechel --uarch gowin --help` device list — GW5A
  device naming in apicula differs from the vendor `GW5AT-LV138...` string).

## Flow sketch

Run from the **repo root** so the `-I src` include path and `src/…` paths resolve.

```sh
# ---- 1. Synthesis (yosys -> Gowin primitives -> JSON netlist) --------------
#   -I src : resolve `include "glm_fp.vh" / "glm_fp_pipe_lat.vh" / "fp8_e4m3.vh"`
#   Source list = the 24 files of GLM_CDC_SRCS (see the repo Makefile and
#   ../gowin/build_gowin.tcl). Top = glm_fp8_system_cdc.
yosys -p '
  read_verilog -sv -I src \
    src/glm_fp8_system_cdc.v src/glm_fp8_system.v src/cdc_async_fifo.v \
    src/reset_sync.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v \
    src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v \
    src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v \
    src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v \
    src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
    src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v \
    src/glm_fp_pipe.v ;
  synth_gowin -top glm_fp8_system_cdc -json glm_fp8.json ;
  stat
'

# ---- 2. Place & Route (nextpnr-himbaechel) ---------------------------------
#   NOTE: full P&R needs pin constraints (.cst) AND the wide memory-side ports
#   buried in a harness (thousands of I/O bits otherwise -- see ../README.md
#   "Full P&R vs synthesis-only" and ../gemm_harness.v for the pattern).
#   For a RESOURCE ESTIMATE, `stat` from step 1 already reports the cell/LUT
#   counts -- that is the fit answer; step 2 is only for routed Fmax.
nextpnr-himbaechel --uarch gowin \
    --device <GW5AT-138-HIMBAECHEL-DEVICE-NAME> \
    --json glm_fp8.json \
    --vopt cst=fpga/gowin/aipu.cst \
    --report glm_fp8_report.json
#   (Consult `nextpnr-himbaechel --uarch gowin --help` for the exact --device
#    string and constraint flags for the GW5A family in your build.)
```

### Compact config

To synthesize the **compact** miniaturization config (PE_N=2, DDR_NCH=2,
KV_RESIDENT=8, EFIFO_DEPTH=8, CACHE_SLOTS=2 — byte-identical token, smaller
area; see [`../../docs/MINIATURIZATION.md`](../../docs/MINIATURIZATION.md)), yosys
can override parameters directly with `-chparam` (no wrapper needed here, unlike
the Gowin flow):

```sh
yosys -p '
  read_verilog -sv -I src <the 24 src/ files above> ;
  chparam -set PE_N 2 -set DDR_NCH 2 -set KV_RESIDENT 8 \
          -set EFIFO_DEPTH 8 -set CACHE_SLOTS 2 glm_fp8_system_cdc ;
  synth_gowin -top glm_fp8_system_cdc -json glm_fp8_compact.json ;
  stat
'
```

(This mirrors the repo Makefile `synth-glm-compact` target, which does the same
`-chparam` overrides for the structural elaboration check.)

## What to record

Even if only **step 1** (`synth_gowin` + `stat`) completes, capture:

- **LUT** (Gowin `LUT4` / ALU cells), **DFF/registers**, **BSRAM** (block RAM),
  **DSP/MULT** — from the `stat` cell histogram.
- If step 2 routes: **Max Frequency per clock** (host_clk, core_clk) from the
  nextpnr timing report (remember: GW5A timing is approximate here).

Put the numbers in the results table in [`../README.md`](../README.md) and note
which flow produced them (vendor Gowin vs this open flow), so the two can be
compared.
