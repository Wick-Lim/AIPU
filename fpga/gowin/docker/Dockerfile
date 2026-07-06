# ============================================================================
# Dockerfile -- Gowin EDA (Linux x86_64) headless container for the AIPU fit
# ----------------------------------------------------------------------------
# WHY: nextpnr-himbaechel's GW5A CLOCK routing is incomplete (any clocked design
#   fails to route on GW5AST-138C), so the open flow can't emit a burnable clocked
#   .fs for our target.  Gowin's own gw_sh (GowinSynthesis + P&R) fully supports
#   GW5A clock spines -> real routing + Fmax + bitstream.  Gowin EDA is Linux/
#   Windows only; this Intel Mac runs the Linux x86_64 build natively in Docker.
#
# YOU MUST PROVIDE (Gowin requires a login; scripts can't fetch it):
#   1. The Gowin EDA Linux tarball, e.g. Gowin_V1.9.11.03_Education_linux.tar.gz
#      from https://www.gowinsemi.com/en/support/download_eda/  (Education = free,
#      GW5A-capable).  Put it next to this Dockerfile as:  gowin_linux.tar.gz
#   2. A GW5A license.  Education needs a node-locked license tied to THIS
#      CONTAINER's MAC.  Pin the container MAC (docker run --mac-address ...) so
#      the license stays valid; get that MAC's license from Gowin, save it as
#      gwlicense.lic next to this Dockerfile.  (See run_docker.sh.)
#
# BUILD:  (from fpga/gowin/docker/)
#     docker build -t aipu-gowin .
# RUN:    ../run_docker.sh            (wraps docker run + the tcl flow)
# ============================================================================
FROM --platform=linux/amd64 ubuntu:22.04

# Gowin EDA is a 32/64-bit Qt app; these are the usual runtime deps for the
# headless gw_sh (no X needed for `run syn`, but libs must resolve).
RUN dpkg --add-architecture i386 && apt-get update && apt-get install -y --no-install-recommends \
      libqt5widgets5 libqt5gui5 libqt5core5a libqt5network5 libqt5xml5 \
      libglib2.0-0 libfontconfig1 libfreetype6 libx11-6 libxext6 libxrender1 \
      libgl1 libglu1-mesa libusb-1.0-0 libtinfo5 \
      ca-certificates tar gzip bash coreutils grep sed gawk && \
    rm -rf /var/lib/apt/lists/*

# Unpack Gowin EDA into /opt/gowin (tarball you provide, next to this Dockerfile).
WORKDIR /opt/gowin
COPY gowin_linux.tar.gz /tmp/gowin_linux.tar.gz
RUN tar xzf /tmp/gowin_linux.tar.gz -C /opt/gowin --strip-components=0 && \
    rm /tmp/gowin_linux.tar.gz && \
    ( find /opt/gowin -name gw_sh -type f | head -1 ) > /opt/gw_sh_path.txt && \
    echo "gw_sh at: $(cat /opt/gw_sh_path.txt)"

# License (node-locked to the container MAC; you provide gwlicense.lic).
COPY gwlicense.lic /opt/gowin/gwlicense.lic
ENV GOWIN_LICENSE_FILE=/opt/gowin/gwlicense.lic
# Some versions read GOWINHOME / gowin.lic env; set both defensively.
ENV GOWINHOME=/opt/gowin

# Put gw_sh on PATH (resolve the actual IDE/bin dir at build time).
RUN ln -sf "$(cat /opt/gw_sh_path.txt)" /usr/local/bin/gw_sh || true

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
