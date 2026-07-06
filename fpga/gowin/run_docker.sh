#!/usr/bin/env bash
#============================================================================
# fpga/gowin/run_docker.sh -- run the Gowin vendor flow in a Linux container
#----------------------------------------------------------------------------
# Runs `gw_sh fpga/gowin/build_gowin.tcl` inside the `aipu-gowin` image (Linux
# x86_64 Gowin EDA), which -- unlike open nextpnr-himbaechel -- fully supports
# GW5A clock routing, so it produces REAL routed LUT/DSP/BSRAM + Fmax + a
# burnable .fs for the GW5AST-138 (Tang Mega 138K Pro).
#
# PREREQUISITES (one-time, YOU do these -- Gowin needs a login):
#   1. Put the Gowin EDA Linux tarball at  fpga/gowin/docker/gowin_linux.tar.gz
#   2. Pick the container MAC below (LOCK_MAC), get a GW5A Education license for
#      THAT MAC from Gowin, save it at  fpga/gowin/docker/gwlicense.lic
#   3. Build the image:   docker build -t aipu-gowin fpga/gowin/docker
#
# THEN:
#     fpga/gowin/run_docker.sh                # default config, synth+P&R
#     COMPACT=1 fpga/gowin/run_docker.sh      # compact config
#     FLOW=syn  fpga/gowin/run_docker.sh      # synth-only (resource fit; no pins)
#============================================================================
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

IMG="${IMG:-aipu-gowin}"
# Pin the container MAC to the one the Gowin license was issued for.  The license
# here was node-locked to this Mac's en0 (Ethernet) MAC, so force the container to
# present THAT MAC -- otherwise the license check fails (containers get a random MAC).
#   en0 (Ethernet) of this machine:
LOCK_MAC="${LOCK_MAC:-50:1F:C6:5B:D9:86}"

COMPACT="${COMPACT:-0}"
FLOW="${FLOW:-all}"     # all = synth+P&R (routed Fmax + .fs) | syn = synth-only fit

# Sanity: image present?
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "ERROR: image '$IMG' not found. Build it first:"
    echo "    docker build -t $IMG fpga/gowin/docker"
    echo "  (needs fpga/gowin/docker/gowin_linux.tar.gz + gwlicense.lic -- see the Dockerfile header)"
    exit 1
fi

echo "=================================================================="
echo " Gowin vendor flow in Docker  (GW5AST-138 / Tang Mega 138K Pro)"
echo "   image=$IMG  mac=$LOCK_MAC  COMPACT=$COMPACT  FLOW=$FLOW"
echo "   part = GW5AST-LV138PG484AC1/I0  (confirmed via apycula pinout)"
echo "=================================================================="

# Run the tcl flow inside the container; mount the repo at /work.
docker run --rm \
    --platform linux/amd64 \
    --mac-address "$LOCK_MAC" \
    -e COMPACT="$COMPACT" -e FLOW="$FLOW" \
    -v "$ROOT":/work \
    -w /work \
    "$IMG" \
    -lc 'gw_sh fpga/gowin/build_gowin.tcl'
RC=$?

echo ""
echo "=== Gowin flow rc=$RC.  Extracting the fit from ./impl/ ==="
# Pull LUT/DSP/BSRAM/FF + Fmax out of the vendor reports (paths per build_gowin.tcl).
SYN=$(ls -t "$ROOT"/impl/gwsynthesis/*_syn_rpt* "$ROOT"/impl/gwsynthesis/*.rpt* 2>/dev/null | head -1)
PNR=$(ls -t "$ROOT"/impl/pnr/*.rpt* 2>/dev/null | head -1)
TIM=$(ls -t "$ROOT"/impl/pnr/*timing* "$ROOT"/impl/pnr/*tr.html 2>/dev/null | head -1)
FS=$(ls -t "$ROOT"/impl/pnr/*.fs 2>/dev/null | head -1)

[ -n "$SYN" ] && { echo "--- synthesis resource ($SYN) ---"; grep -iE "LUT|Register|BSRAM|DSP|SSRAM|ALU|Utilization" "$SYN" 2>/dev/null | head -20; }
[ -n "$PNR" ] && { echo "--- P&R utilization ($PNR) ---"; grep -iE "LUT|Register|BSRAM|DSP|Utilization|IO " "$PNR" 2>/dev/null | head -20; }
[ -n "$TIM" ] && { echo "--- timing / Fmax ($TIM) ---"; grep -iE "Max Freq|Frequency|Clock|MHz" "$TIM" 2>/dev/null | head -10; }
[ -n "$FS" ] && echo ">> BITSTREAM: $FS  ($(du -h "$FS" | cut -f1))  <- flash this to the board"

echo ""
echo "Record the numbers in fpga/README.md.  If FLOW=all produced a .fs, that is"
echo "the burnable bitstream (finalize pins in fpga/gowin/aipu.cst against the board first)."
exit $RC
