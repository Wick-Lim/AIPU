#!/usr/bin/env bash
# ============================================================================
# run_fit.sh -- run the XCKU3P fit (fpga/synth_ku3p.tcl) inside the Vivado Docker
#   container, after Vivado is installed into the `xilinx_install` docker volume.
#
#   Usage:  bash fpga/run_fit.sh
#   Output: fpga/out/{util_synth,util_routed,timing,design_analysis}.rpt + the
#           FIT SUMMARY on stdout.
#
#   ---- LICENSE (read this first) ----
#   Vivado ML 2026.1 will NOT launch -- not even a trivial batch script -- without
#   a valid license. The free "Vivado ML Standard" edition covers the XCKU3P
#   (Kintex UltraScale+ KU3P/KU5P are the free-tier UltraScale+ parts), but you
#   must GENERATE a (free, $0) license from your AMD account, once:
#
#     1. https://www.xilinx.com/getlicense  (or Vivado > Help > Manage License)
#     2. Sign in with your AMD account -> "Vivado ML Standard" (Node-Locked, free)
#     3. When it asks for a HOST ID, give the FIXED container hostid this script
#        pins:   MAC 02:42:c0:a8:64:02   ->   FlexLM hostid  0242c0a86402
#        (This script always launches the container with that MAC via
#         --mac-address, so a node-locked license stays valid across runs.)
#     4. Download the generated  Xilinx.lic  and put it at  $LIC_HOST  below
#        (default /Users/Shared/xilinx/Xilinx.lic). Then re-run this script.
#
#   The Vivado *tool* also needs a few runtime libs (the installer did not). We
#   install them into a throwaway layer at run time (libtinfo5 from the focal pkg,
#   since 22.04 ships libtinfo6).
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VOL=xilinx_install
FIXMAC=02:42:c0:a8:64:02                       # -> FlexLM hostid 0242c0a86402
LIC_HOST="${XILINX_LICENSE:-/Users/Shared/xilinx/Xilinx.lic}"
VIVADO_LIB=/opt/Xilinx/2026.1/Vivado/lib/lnx64.o
STUB_HOST=/Users/Shared/xilinx/libtcmalloc_stub.so.4

# ---- tcmalloc-stub workaround (REQUIRED in Docker; verified 2026-07) ----
# Vivado's bundled libtcmalloc.so.4 interposes malloc/realloc globally. During
# FlexLM license checkout, lmgr's libudev device scan then mixes tcmalloc and
# glibc-internal allocations -> "realloc(): invalid pointer" abort at launch.
# The binary has ZERO direct tc_* symbol references (interposition-only), so an
# EMPTY stub .so bind-mounted over the bundled lib removes the interposition and
# everything runs on plain glibc. (Vivado's own bin/vivado already disables its
# other custom allocator the same way: "CR-1074520 ... use the system default".)
if [[ ! -f "$STUB_HOST" ]]; then
  echo "building tcmalloc stub -> $STUB_HOST"
  docker run --rm -v "$(dirname "$STUB_HOST")":/out ubuntu:22.04 bash -c '
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq --no-install-recommends gcc libc6-dev >/dev/null 2>&1
    echo "int __xil_tcmalloc_stub;" > /tmp/stub.c
    gcc -shared -fPIC -Wl,-soname,libtcmalloc.so.4 -o /out/'"$(basename "$STUB_HOST")"' /tmp/stub.c'
fi

# ---- license preflight: mount the .lic if present, else tell the user how ----
LIC_ARGS=()
LIC_ENV=""
if [[ -f "$LIC_HOST" ]]; then
  LIC_ARGS=(-v "$LIC_HOST":/lic/Xilinx.lic:ro)
  LIC_ENV="export XILINXD_LICENSE_FILE=/lic/Xilinx.lic;"
  echo "license: mounting $LIC_HOST  (hostid must be 0242c0a86402)"
else
  cat >&2 <<EOF
--------------------------------------------------------------------------
NO LICENSE FOUND at: $LIC_HOST
Vivado ML 2026.1 refuses to launch without one. Generate the FREE
"Vivado ML Standard" node-locked license for FlexLM hostid  0242c0a86402
at https://www.xilinx.com/getlicense , save the Xilinx.lic to that path
(or set XILINX_LICENSE=/path/to/Xilinx.lic), and re-run. See the header of
this script for the full walkthrough. Running anyway will fail at launch.
--------------------------------------------------------------------------
EOF
fi

docker run --rm \
  --mac-address "$FIXMAC" \
  -v ${VOL}:/opt/Xilinx:ro \
  -v "$STUB_HOST":"$VIVADO_LIB"/libtcmalloc.so.4:ro \
  -v "${REPO}":/work \
  "${LIC_ARGS[@]}" \
  -w /work \
  ubuntu:22.04 bash -euo pipefail -c "
    ${LIC_ENV}
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    # Vivado runtime deps; libtinfo5 is the classic one 22.04 lacks (grab from focal).
    apt-get install -y -qq --no-install-recommends \
      libx11-6 libxext6 libxrender1 libxtst6 libxi6 libncurses5 locales wget ca-certificates \
      libpixman-1-0 libcairo2 libpango-1.0-0 libglib2.0-0 libfreetype6 libfontconfig1 >/dev/null 2>&1 || true
    if ! ldconfig -p | grep -q libtinfo.so.5; then
      wget -q http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb -O /tmp/t.deb 2>/dev/null \
        && dpkg -i /tmp/t.deb >/dev/null 2>&1 || true
    fi
    locale-gen en_US.UTF-8 >/dev/null 2>&1 || true
    export LANG=en_US.UTF-8
    SETTINGS=\$(find /opt/Xilinx -name settings64.sh -path '*Vivado*' 2>/dev/null | head -1)
    echo \"sourcing: \$SETTINGS\"
    source \"\$SETTINGS\"
    echo \"XILINXD_LICENSE_FILE=\${XILINXD_LICENSE_FILE:-<unset>}\"
    vivado -version
    echo '==== running fit ===='
    mkdir -p /work/fpga/out
    vivado -mode batch -source fpga/synth_ku3p.tcl -nojournal -log /work/fpga/out/vivado.log 2>&1
  "
