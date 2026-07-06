#============================================================================
# build_gowin.tcl  --  Gowin `gw_sh` batch flow for the GLM-5.2-FP8 accelerator
#----------------------------------------------------------------------------
# PURPOSE (Phase D0.2 "FPGA fit" -- docs/USBC_PRODUCT_PLAN.md)
#   Take the product top `glm_fp8_system_cdc` (2-clock CDC wrapper around the
#   verified GLM-5.2-FP8 compute box) through the GOWIN vendor flow to get the
#   REAL fit on a Gowin GW5AT-138 (Sipeed Tang Mega 138K Pro):
#     LUT / DSP / BSRAM(block RAM) / registers  +  achieved Fmax per clock.
#   This is the #1 unknown that gates FPGA class -> size / thermal / BOM / price.
#
# WHY A VENDOR FLOW (updated -- the earlier yosys walls are retired):
#   1. yosys `synth_gowin` (NOT `abc -lut4`) DOES map the FP8 datapath -- it infers
#      hardware DSPs (MULT18X18/MULT9X9) for the multiplies, so glm_matmul_fp8 maps
#      (leaf @ KMAX=256: ~17.8K LUT-eq + 20 DSP).  The old "abc-lut4 times out" wall
#      is broken (docs/FPGA_DEMO_PLAN.md).
#   2. The whole-system elaboration hang was a REAL area bug -- glm_matmul_fp8's
#      dequant was an O(NB^2) unrolled fold (NB=ceil(KMAX/BLK); NB=128 @ KMAX=16384).
#      It is FIXED: O(NB^2)->O(1) sequential fold, bit-exact (matmul 224/224,
#      bitacc 14/14+argmax, model argmax 4/31/20).  The design now elaborates at
#      product KMAX.
#   So the vendor flow is no longer needed to "get a LUT count at all" -- it is
#   needed for the ROUTED numbers yosys 0.66 can't give: placed LUT/DSP/BSRAM
#   utilization + real per-clock Fmax, correct BRAM inference for the O(NB) `accx`
#   block-accumulator memory, and a mapper that doesn't hit yosys 0.66's SAT-based
#   `SHARE` pass (the remaining yosys scalability limit at full NB).  GowinSynthesis
#   handles all of these.  (If it still struggles on the FP8 math, record it -- see
#   fpga/README.md caveats.)
#
# HOW TO RUN (from the repo root, with Gowin EDA installed & licensed):
#     gw_sh fpga/gowin/build_gowin.tcl                 # default config, synth-only
#     COMPACT=1 gw_sh fpga/gowin/build_gowin.tcl       # compact miniaturized config
#     FLOW=all  gw_sh fpga/gowin/build_gowin.tcl       # attempt full Place & Route
#     COMPACT=1 FLOW=all gw_sh fpga/gowin/build_gowin.tcl
#
#   Outputs land under ./impl/ (synthesis + P&R reports).  See the puts at the
#   end for exactly which report file holds the LUT/DSP/BSRAM/FF/Fmax numbers.
#
# NOTE ON `gw_sh`: exact Tcl command spellings vary slightly across Gowin EDA
#   versions.  This script targets the common `set_device` / `add_file` /
#   `set_option` / `run` API.  If a command name differs in your version, check
#   the "Gowin Software User Guide" (gw_sh chapter) and adjust; the comments mark
#   every spot you may need to touch.
#============================================================================

# ============================================================================
# 0. USER CONFIG -- edit these three, then the flags via environment variables.
# ============================================================================

# --- DEVICE PART STRING --------------------------------------------------- #
#   *** PLACEHOLDER -- VERIFY BEFORE RUNNING. ***
#   GW5AT-138 comes in several packages/speed grades; the Tang Mega 138K Pro
#   uses a specific one.  The string below is a best-guess for that board; the
#   trailing package/grade fields (FCG676 / speed / temp) MUST be confirmed.
#   HOW TO FIND THE EXACT STRING:
#     * Gowin IDE -> New Project -> "Select Device": pick Series=GW5AT,
#       Device=GW5AT-138, then the Package/Speed row that matches your board;
#       the IDE shows the full "Part Number" (that exact string goes here).
#     * Or in gw_sh:  `get_device_info` after a `set_device`, or consult the
#       Gowin "GW5AT series Product Brief" / device pin table.
#     * Or read it off an existing Tang Mega 138K Pro Gowin project's .gprj.
set PART        "GW5AT-LV138FCG676AC"   ;# <-- CONFIRM THIS (package/speed/grade)
set DEVICE_NAME "GW5AT-138"             ;# device series / family name

# --- CONFIG SELECT (env COMPACT) ------------------------------------------ #
#   0 = default / committed slice config (PE_N=4, DDR_NCH=4, KV_RESIDENT=16,
#                                         EFIFO_DEPTH=16, CACHE_SLOTS=4)
#   1 = compact FPGA-miniaturization config (PE_N=2, DDR_NCH=2, KV_RESIDENT=8,
#                                            EFIFO_DEPTH=8, CACHE_SLOTS=2)
#       -- byte-identical token, smaller area (docs/MINIATURIZATION.md L0).
if {[info exists ::env(COMPACT)]} { set COMPACT $::env(COMPACT) } else { set COMPACT 0 }

# --- FLOW SELECT (env FLOW) ----------------------------------------------- #
#   syn = synthesis-only  -> gives the RESOURCE FIT (LUT/DSP/BSRAM/FF). Always
#         runs; needs NO pin assignments. THIS is the D0.2 answer.
#   all = full flow (synthesis + Place & Route + timing) -> adds real routed
#         Fmax, BUT the raw top has thousands of memory-side port bits that
#         exceed any package's user-I/O count, so P&R will fail on I/O unless
#         you first wrap the design in a bring-up harness that buries those wide
#         ports (see fpga/README.md "Full P&R vs synthesis-only").  Provided for
#         completeness / for the harnessed case.
if {[info exists ::env(FLOW)]} { set FLOW $::env(FLOW) } else { set FLOW "syn" }

# ============================================================================
# 1. PATHS  (resolve everything relative to THIS script -> repo root).
# ============================================================================
set SCRIPT_DIR [file dirname [file normalize [info script]]]      ;# .../fpga/gowin
set REPO_ROOT  [file normalize [file join $SCRIPT_DIR .. ..]]     ;# repo root
set SRC_DIR    [file join $REPO_ROOT src]
set SDC_FILE   [file join $SCRIPT_DIR aipu.sdc]
set COMPACT_WRAP [file join $SCRIPT_DIR glm_fp8_system_cdc_compact.v]

puts "=================================================================="
puts " GLM-5.2-FP8  Gowin fit  (Phase D0.2 -- docs/USBC_PRODUCT_PLAN.md)"
puts "   repo root : $REPO_ROOT"
puts "   device    : $DEVICE_NAME  part=$PART   (VERIFY the part string!)"
puts "   config    : [expr {$COMPACT ? {COMPACT (PE_N=2,DDR_NCH=2,KVR=8,EFIFO=8,SLOTS=2)} : {DEFAULT (PE_N=4,DDR_NCH=4,KVR=16,EFIFO=16,SLOTS=4)}}]"
puts "   flow      : [expr {$FLOW eq {all} ? {FULL P&R (syn+pnr+timing)} : {SYNTHESIS-ONLY (resource fit)}}]"
puts "=================================================================="

# ============================================================================
# 2. SOURCE FILE LIST
#   This MUST match the Makefile `GLM_CDC_SRCS` variable (24 files, all under
#   src/).  Include files (glm_fp.vh, glm_fp_pipe_lat.vh, fp8_e4m3.vh, ...) are
#   pulled in via the include path set below; do NOT add .vh files here.
# ============================================================================
set GLM_CDC_SRCS {
    glm_fp8_system_cdc.v
    glm_fp8_system.v
    cdc_async_fifo.v
    reset_sync.v
    glm_model_fp8.v
    ddr5_xbar.v
    weight_loader.v
    expert_cache_pf.v
    expert_cache_ctrl.v
    kv_cache_pager.v
    glm_decoder_block_fp8.v
    mla_attn_fp8.v
    swiglu_expert_fp8.v
    moe_router_fp8.v
    glm_matmul_fp8.v
    rmsnorm_unit.v
    rope_interleave_unit.v
    glm_softmax.v
    dsa_indexer.v
    topk_select.v
    glm_act.v
    glm_matmul_pipe.v
    sampler.v
    glm_fp_pipe.v
}

# ============================================================================
# 3. DEVICE
# ============================================================================
# `set_device` selects the target part.  Signature is typically:
#     set_device -name <SERIES> <PART_NUMBER>
# (Some Gowin versions also accept a `-device_version <A|B|C>` grade flag.)
set_device -name $DEVICE_NAME $PART

# ============================================================================
# 4. ADD FILES
# ============================================================================
foreach f $GLM_CDC_SRCS {
    add_file -type verilog [file join $SRC_DIR $f]
}

# Compact config: add the passthrough wrapper (fpga/gowin/glm_fp8_system_cdc_compact.v)
# and retarget the top to it.  Nothing in src/ changes.
if {$COMPACT} {
    add_file -type verilog $COMPACT_WRAP
    set TOP_MODULE "glm_fp8_system_cdc_compact"
} else {
    set TOP_MODULE "glm_fp8_system_cdc"
}

# Timing constraints (2 async clocks -- see aipu.sdc).
add_file -type sdc $SDC_FILE

# ============================================================================
# 5. OPTIONS
# ============================================================================
set_option -top_module   $TOP_MODULE
# The RTL uses SystemVerilog-2012+ constructs ($clog2, generate, packed params);
# the repo builds it with iverilog -g2012 and yosys read_verilog -sv.  Match that:
set_option -verilog_std  sysv2017
# Include dir for `include "glm_fp.vh" / "glm_fp_pipe_lat.vh" / "fp8_e4m3.vh"`, etc.
set_option -include_path  $SRC_DIR

# (Optional quality knobs -- uncomment/tune after a first baseline run.)
# set_option -synthesis_tool     gowinsynthesis   ;# default on GW5A
# set_option -timing_driven      1
# set_option -retiming           1                ;# can lift Fmax on deep FP8 pipes
# set_option -gen_text_timing_rpt 1               ;# text timing report

# ============================================================================
# 6. RUN
# ============================================================================
if {$FLOW eq "all"} {
    puts ">> Running FULL flow: synthesis + place & route + timing (run all)"
    puts ">> (If this aborts on I/O / package pins, that is the expected wide-port"
    puts ">>  wall -- switch to FLOW=syn for the resource fit, or use a harness.)"
    run all
} else {
    puts ">> Running SYNTHESIS ONLY (run syn) -- resource fit, no pin placement"
    run syn
}

# ============================================================================
# 7. WHERE THE NUMBERS ARE
#   gw_sh writes reports under ./impl/ .  Exact filenames vary by version, but:
#     * Synthesis resource summary  : impl/gwsynthesis/*_syn.rpt.html (or .rpt)
#         -> LUT / register / BSRAM / DSP counts + utilization %.
#     * Place & Route report (FLOW=all): impl/pnr/*.rpt.html
#         -> placed/routed resource utilization for the chosen package.
#     * Timing report (FLOW=all)    : impl/pnr/*.timing.html / *_tr.html
#         -> Max Frequency per clock (host_clk, core_clk) + critical paths.
#   RECORD these in the fpga/README.md results table.
# ============================================================================
puts "=================================================================="
puts " DONE ($FLOW).  Look under:  [file join $REPO_ROOT impl]"
puts "   synthesis fit : impl/gwsynthesis/*_syn.rpt*   (LUT/DSP/BSRAM/FF)"
if {$FLOW eq "all"} {
    puts "   P&R fit       : impl/pnr/*.rpt*             (routed utilization)"
    puts "   timing/Fmax   : impl/pnr/*timing*           (per-clock Max Freq)"
}
puts "   -> Fill these into the results table in fpga/README.md"
puts "=================================================================="
