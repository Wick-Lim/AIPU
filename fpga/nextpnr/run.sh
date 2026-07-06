#!/usr/bin/env bash
#============================================================================
# fpga/nextpnr/run.sh -- open-source Gowin fit on THIS machine (oss-cad-suite)
#----------------------------------------------------------------------------
# The macOS-native / no-license alternative to the Gowin `gw_sh` vendor flow
# (../gowin/build_gowin.tcl).  Uses the recent yosys + nextpnr-himbaechel +
# apycula bundled in oss-cad-suite.
#
#   Step 1 (synth) ALWAYS runs and gives the RESOURCE FIT estimate
#     (LUT / DFF / BSRAM / DSP) via `synth_gowin` + `stat`.  A recent yosys
#     infers BSRAM for the O(NB) `accx` block-accumulator (yosys 0.66 flattened
#     it to registers); confirm a non-trivial BSRAM count in the stat.
#   Step 2 (pnr) is OPTIONAL and gives routed Fmax -- but GW5A support in
#     apycula/himbaechel is newer/approximate and may not have GW5AT-138.  It
#     needs the wide memory-side ports buried in a harness (see ../README.md).
#
# USAGE (from anywhere in the repo):
#     fpga/nextpnr/run.sh                 # compact config, synth-only (default)
#     fpga/nextpnr/run.sh default         # default (committed) config, synth-only
#     fpga/nextpnr/run.sh compact pnr     # compact + attempt nextpnr routing
#     OSS_CAD_SUITE=/path fpga/nextpnr/run.sh   # if oss-cad-suite is elsewhere
#============================================================================
set -uo pipefail

# --- repo root ---
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
cd "$ROOT" || exit 1

# --- locate + load oss-cad-suite (recent yosys + nextpnr-himbaechel) ---
OSS="${OSS_CAD_SUITE:-$HOME/oss-cad-suite}"
if [ ! -f "$OSS/environment" ]; then
    echo "ERROR: oss-cad-suite not found at '$OSS'."
    echo "  Install it (macOS x86_64), then re-run:"
    echo "    URL=\$(gh api repos/YosysHQ/oss-cad-suite-build/releases/latest --jq '.assets[]|select(.name|test(\"darwin-x64\")).browser_download_url')"
    echo "    curl -L \"\$URL\" -o oss.tgz && tar xzf oss.tgz -C \$HOME"
    echo "  or set OSS_CAD_SUITE=/path/to/oss-cad-suite"
    exit 1
fi
# shellcheck disable=SC1091
source "$OSS/environment"
echo "yosys: $(yosys --version 2>/dev/null | head -1)"

CFG="${1:-compact}"          # compact | default
STEP="${2:-synth}"           # synth   | pnr
OUT="fpga/nextpnr/out"; mkdir -p "$OUT"

# --- the 24 GLM_CDC_SRCS files (must match the Makefile / build_gowin.tcl) ---
SRCS="src/glm_fp8_system_cdc.v src/glm_fp8_system.v src/cdc_async_fifo.v \
src/reset_sync.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v \
src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v \
src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v \
src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v \
src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v \
src/glm_fp_pipe.v"

# --- compact config = the 5 result-invariant param overrides ---
if [ "$CFG" = compact ]; then
    CHP="chparam -set PE_N 2 -set DDR_NCH 2 -set KV_RESIDENT 8 -set EFIFO_DEPTH 8 -set CACHE_SLOTS 2 glm_fp8_system_cdc;"
    JSON="$OUT/glm_fp8_compact.json"
else
    CHP=""
    JSON="$OUT/glm_fp8_default.json"
fi

echo "=================================================================="
echo " Open-source Gowin fit -- config=$CFG  step=$STEP  top=glm_fp8_system_cdc"
echo "=================================================================="

# ============================================================================
# STEP 1: Gowin synthesis -> resource stat + JSON netlist   (the FIT estimate)
#
#   We run synth_gowin's OWN pass script, MINUS the `share` pass.  `share` is a
#   SAT-based resource-sharing OPTIMIZATION that does not scale on this wide FP8
#   datapath (it SAT-solves over thousands of muxes -- the pass that hung the
#   whole-top run past ~28 min).  Dropping it costs a little area optimization but
#   the FIT estimate stays valid (a hair larger, if anything).  Everything else --
#   flatten, DSP techmap, memory/BRAM inference, alumacc, abc9 LUT map -- is the
#   standard synth_gowin flow.  (Pass list from `help synth_gowin`.)
# ============================================================================
#   Strategy: synth_gowin -run begin:coarse (safe), then the coarse section BY
#   HAND minus `share`, then synth_gowin -run map_ram: (lets the tool do the
#   tricky memory/gate/FF mapping correctly).  Only the coarse passes are
#   hand-written (exactly per `help synth_gowin`), so nothing fragile is guessed.
yosys -ql "$OUT/synth_${CFG}.log" -p "
  read_verilog -sv -I src $SRCS ;
  ${CHP}
  synth_gowin -top glm_fp8_system_cdc -run begin:coarse ;
  # ---- coarse section, WITHOUT 'share' (the SAT pass that doesn't scale) ----
  proc ; check ; flatten ; tribuf -logic ; deminout ;
  opt_expr ; opt_clean ; check ; opt -nodffe -nosdff ; fsm ; opt ;
  wreduce ; peepopt ; opt_clean ;
  alumacc ; opt ; memory -nomap ; opt_clean ;
  # ---- let synth_gowin finish: memory->BRAM, gates->LUT (abc9), FF map, etc ----
  synth_gowin -top glm_fp8_system_cdc -run map_ram: -json $JSON ;
  tee -o $OUT/stat_${CFG}.txt stat
"
RC=$?
echo "--- synth rc=$RC ; full log: $OUT/synth_${CFG}.log ---"
if [ $RC -ne 0 ] || [ ! -s "$OUT/stat_${CFG}.txt" ]; then
    echo "synth did not complete cleanly -- inspect $OUT/synth_${CFG}.log"
    exit $RC
fi
echo ""
echo "=== RESOURCE FIT ($CFG) -- record these in fpga/README.md ==="
grep -iE "LUT|ALU|DFF|MULT|DSP|BSRAM|SDPB|Number of cells" "$OUT/stat_${CFG}.txt" | grep -vE "Warning|Replacing" || cat "$OUT/stat_${CFG}.txt"
echo ""
echo ">> #1 CHECK: is 'accx' in BSRAM (SDPB/BSRAM cells present)?  If it shows as"
echo ">>          LUTs/DFFs instead, the fit reads wrong-huge -- see ../README.md."

# ============================================================================
# STEP 2 (optional): nextpnr-himbaechel routing -> Fmax   (GW5A = approximate)
# ============================================================================
if [ "$STEP" = pnr ]; then
    echo ""
    echo "=== STEP 2: nextpnr-himbaechel (GW5A support is newer/approximate) ==="
    echo "Devices this build knows (grep GW5 for the GW5AT-138 string):"
    nextpnr-himbaechel --uarch gowin --help 2>&1 | grep -iE "device|GW5" | head -20 || true
    echo ""
    echo "Then route with the confirmed device string, e.g.:"
    echo "  nextpnr-himbaechel --uarch gowin --device <GW5AT-138 string> \\"
    echo "     --json $JSON --report $OUT/report_${CFG}.json"
    echo "(needs the wide memory-side ports buried in a harness + a .cst -- ../README.md.)"
fi

echo ""
echo "DONE.  Netlist: $JSON   Stat: $OUT/stat_${CFG}.txt"
