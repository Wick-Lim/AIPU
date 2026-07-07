# ============================================================================
# Dockerfile -- Vivado (Linux x86_64) headless container for the KU3P routed fit
# ----------------------------------------------------------------------------
# WHY: the product FPGA is a Kintex UltraScale+ XCKU3P (DDR4 + PCIe). Vivado gives
#   the AUTHORITATIVE routed fit -- LUT/FF/DSP/BRAM/URAM utilization + Fmax --
#   that locks the device + fills docs/PART_SELECTION.md and the site §04. Vivado
#   is Linux/Windows only (no macOS); this Intel Mac runs the Linux x86_64 build
#   natively in Docker.
#
# DISK: Vivado is ~100 GB installed (KU3P-only minimal ~30-50 GB). This image does
#   NOT bake Vivado in -- it stays SLIM and MOUNTS a host Vivado install at
#   /opt/Xilinx (see run_docker.sh). So the Docker HOST needs ~100 GB free for the
#   Vivado dir. (This particular Mac has only ~22 GB free -> run on a cloud Linux
#   Docker host, or free up disk first.)
#
# YOU MUST PROVIDE (AMD requires a login; scripts can't fetch it):
#   1. Install Vivado on the HOST into a directory, e.g. ~/Xilinx, using AMD's
#      Unified Installer (Vivado ML Standard = FREE, covers KU3P). A device-limited
#      install (UltraScale+ only) keeps it smaller.
#   2. License: KU3P is FREE (ML Standard). If AMD asks, generate a free license at
#      https://www.xilinx.com/getlicense and point XILINXD_LICENSE_FILE at it
#      (run_docker.sh passes it through).
#
# BUILD:  docker build -t aipu-vivado fpga/vivado/docker
# RUN:    VIVADO_DIR=~/Xilinx/Vivado/2024.2 fpga/vivado/run_docker.sh
# ============================================================================
FROM --platform=linux/amd64 ubuntu:22.04

# Vivado's headless synth/P&R still dlopen's X/font libs; these are the usual
# runtime deps for a GUI-less `vivado -mode batch`. Vivado bundles its own JRE.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libtinfo5 libncurses5 libx11-6 libxext6 libxrender1 libxtst6 libxi6 \
      libxft2 libfontconfig1 libfreetype6 libstdc++6 zlib1g \
      locales ca-certificates xvfb bash coreutils grep sed gawk && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
# Vivado is MOUNTED at /opt/Xilinx by run_docker.sh (not baked in); settings64.sh
# from the mounted install puts vivado on PATH at run time.
WORKDIR /work
ENTRYPOINT ["/bin/bash"]
