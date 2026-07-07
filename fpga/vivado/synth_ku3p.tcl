#============================================================================
# fpga/vivado/synth_ku3p.tcl -- routed FIT + Fmax on the REAL target part
#   (Kintex UltraScale+ XCKU3P) at the compact config, via AMD/Xilinx Vivado.
#----------------------------------------------------------------------------
# WHY: the open Gowin flow targets the wrong device (GW5A, DDR3, no DSP infer);
#   the product FPGA is a KU3P-class part (DDR4 + PCIe). This script produces the
#   AUTHORITATIVE routed fit -- LUT / FF / DSP / BRAM / URAM utilization + Fmax --
#   the one thing that locks the FPGA device + fills docs/PART_SELECTION.md.
#
# WHERE: Vivado is Linux/Windows only (NOT macOS). Run on a Linux/Windows box (or
#   a cloud instance). KU3P is covered by the FREE Vivado ML Standard license
#   (confirmed: the Alibaba-cloud KU3P community runs it on WebPACK/Standard).
#
# USAGE:
#   vivado -mode batch -source fpga/vivado/synth_ku3p.tcl
#   # override the part/config if needed:
#   vivado -mode batch -source fpga/vivado/synth_ku3p.tcl -tclargs xcku3p-ffvb676-2-e default
#
# OUTPUT: fpga/vivado/out/{util_*.rpt, timing_*.rpt} -- copy the numbers into
#   docs/PART_SELECTION.md (the "[PENDING -- needs Vivado]" rows) and the site §04.
#============================================================================

# ---- args: PART, CFG (compact|default) ----
set PART [expr {$argc >= 1 ? [lindex $argv 0] : "xcku3p-ffvb676-2-e"}]
set CFG  [expr {$argc >= 2 ? [lindex $argv 1] : "compact"}]

set ROOT [file normalize [file join [file dirname [info script]] .. ..]]
set OUT  [file join $ROOT fpga vivado out]
file mkdir $OUT
puts "== Vivado fit: part=$PART cfg=$CFG top=glm_fp8_system_cdc =="

# ---- the 24 GLM_CDC_SRCS (must match Makefile GLM_CDC_SRCS / build_gowin.tcl) ----
set SRCS {
  glm_fp8_system_cdc.v glm_fp8_system.v cdc_async_fifo.v reset_sync.v
  glm_model_fp8.v ddr5_xbar.v weight_loader.v expert_cache_pf.v
  expert_cache_ctrl.v kv_cache_pager.v glm_decoder_block_fp8.v mla_attn_fp8.v
  swiglu_expert_fp8.v moe_router_fp8.v glm_matmul_fp8.v rmsnorm_unit.v
  rope_interleave_unit.v glm_softmax.v dsa_indexer.v topk_select.v
  glm_act.v glm_matmul_pipe.v sampler.v glm_fp_pipe.v
}
foreach f $SRCS { read_verilog -sv [file join $ROOT src $f] }
set_property include_dirs [file join $ROOT src] [current_fileset]

# ---- compact config = the 5 result-invariant param overrides (byte-identical) ----
if {$CFG eq "compact"} {
  set GEN {PE_N=2 DDR_NCH=2 KV_RESIDENT=8 EFIFO_DEPTH=8 CACHE_SLOTS=2}
} else {
  set GEN {}
}

# ---- synth -> utilization (LUT / FF / DSP / BRAM / URAM) ----
synth_design -top glm_fp8_system_cdc -part $PART {*}[expr {[llength $GEN] ? "-generic $GEN" : ""}]
report_utilization      -file [file join $OUT util_synth_$CFG.rpt]
report_utilization -hierarchical -file [file join $OUT util_hier_$CFG.rpt]

# ---- place + route -> real Fmax (post-route timing) ----
# (Comment out place/route for a fast synth-only utilization pass.)
opt_design
place_design
route_design
report_utilization     -file [file join $OUT util_routed_$CFG.rpt]
report_timing_summary  -file [file join $OUT timing_$CFG.rpt]

puts "== DONE. Reports in $OUT :"
puts "   util_synth_$CFG.rpt   -- LUT/FF/DSP/BRAM/URAM after synth"
puts "   util_routed_$CFG.rpt  -- after place+route"
puts "   timing_$CFG.rpt       -- WNS/TNS -> Fmax"
puts "== Copy these into docs/PART_SELECTION.md and the site §04. =="
