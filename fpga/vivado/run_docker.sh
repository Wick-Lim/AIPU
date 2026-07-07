#!/usr/bin/env bash
#============================================================================
# fpga/vivado/run_docker.sh -- run the KU3P routed fit in a Linux container
#----------------------------------------------------------------------------
# Runs `vivado -mode batch -source fpga/vivado/synth_ku3p.tcl` inside the slim
# `aipu-vivado` image, with a HOST Vivado install mounted at /opt/Xilinx. Produces
# the real routed LUT/FF/DSP/BRAM/URAM + Fmax on the XCKU3P (the product part).
#
# PREREQUISITES (one-time, YOU do these -- AMD needs a login):
#   1. Install Vivado on the HOST (Vivado ML Standard = FREE, covers KU3P), e.g.
#      to ~/Xilinx/Vivado/2024.2  (a UltraScale+-only install keeps it smaller).
#   2. Build the image:   docker build -t aipu-vivado fpga/vivado/docker
#   3. Ensure the Docker HOST has ~100 GB free for the Vivado dir (this Mac has
#      ~22 GB -> use a cloud Linux Docker host, or free disk first).
#
# THEN:
#     VIVADO_DIR=~/Xilinx/Vivado/2024.2 fpga/vivado/run_docker.sh
#     CFG=default  VIVADO_DIR=... fpga/vivado/run_docker.sh   # full config
#     PART=xcku3p-ffvb676-2-e VIVADO_DIR=... fpga/vivado/run_docker.sh
#============================================================================
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

IMG="${IMG:-aipu-vivado}"
CFG="${CFG:-compact}"                       # compact | default
PART="${PART:-xcku3p-ffvb676-2-e}"          # confirm vs your board's exact part
VIVADO_DIR="${VIVADO_DIR:-}"                # HOST path to Vivado/<ver> (has settings64.sh)
LICENSE="${XILINXD_LICENSE_FILE:-}"         # optional free-license file/port@host

if ! docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "ERROR: image '$IMG' not found. Build it first:"
    echo "    docker build -t $IMG fpga/vivado/docker"
    exit 1
fi
if [ -z "$VIVADO_DIR" ] || [ ! -f "$VIVADO_DIR/settings64.sh" ]; then
    echo "ERROR: set VIVADO_DIR to a host Vivado install (must contain settings64.sh)."
    echo "    e.g.  VIVADO_DIR=~/Xilinx/Vivado/2024.2 $0"
    echo "  Vivado ML Standard is free and covers KU3P; install it on the host first."
    exit 1
fi

# license passthrough (KU3P is free; a license may still be required by AMD)
LIC_ARGS=()
if [ -n "$LICENSE" ]; then LIC_ARGS=(-e "XILINXD_LICENSE_FILE=$LICENSE"); fi

echo "== Vivado fit in Docker: part=$PART cfg=$CFG =="
echo "   Vivado (host) : $VIVADO_DIR  -> mounted at /opt/Xilinx/Vivado/cur"
echo "   repo          : $ROOT        -> mounted at /work"

docker run --rm --platform=linux/amd64 \
    -v "$ROOT":/work \
    -v "$VIVADO_DIR":/opt/Xilinx/Vivado/cur:ro \
    "${LIC_ARGS[@]}" \
    "$IMG" -lc "
        source /opt/Xilinx/Vivado/cur/settings64.sh 2>/dev/null || \
          { echo 'settings64.sh not found in mounted Vivado dir'; exit 1; } ;
        cd /work ;
        vivado -mode batch -source fpga/vivado/synth_ku3p.tcl -tclargs $PART $CFG
    "
RC=$?
echo "--- vivado rc=$RC ; reports in fpga/vivado/out/ ---"
[ $RC -eq 0 ] && echo "OK: copy util_*/timing_* numbers into docs/PART_SELECTION.md + site §04."
exit $RC
