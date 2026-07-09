# ============================================================================
# synth_ku3p.tcl -- Vivado batch synthesis + place & route + FIT report for the
#   GLM-5.2 Q4_K product top `glm_q4k_system_cdc` on a Kintex UltraScale+ XCKU3P.
#
#   This is the [PENDING] FPGA-fit gate that iverilog/yosys cannot give: the REAL
#   routed LUT/DSP/BRAM utilization + the routed Fmax (timing closure) on the actual
#   part. Run headless (non-project mode), from the repo root:
#
#     vivado -mode batch -source fpga/synth_ku3p.tcl
#
#   TWO reports, because the product top and a routable design are different things:
#     (1) util_synth.rpt  -- synth-only resources of the RAW product top
#         `glm_q4k_system_cdc`. No pins needed (report_utilization after synth), so
#         this is the honest resource fit of the shipping module.
#     (2) util_routed.rpt + timing.rpt -- a FULL place & route of `bringup_harness`,
#         which wraps the exact same product top but buries its thousands of wide
#         memory-side ports (DDR/flash/KV/dequant buses + VOCAB*16 logits) behind an
#         on-chip LFSR/CRC so the I/O count is routable. THIS gives the routed Fmax.
#         Its utilization == the product's + a negligible 256-FF LFSR + XOR tree.
#
#   Both synthesize the COMPACT (result-invariant resource) config -- half the matmul
#   PE array / DDR channels, smaller KV ring / expert FIFO / expert cache -- which is
#   what targets a small dev-board part; the decoded token is byte-identical to the
#   default config (proven in sim), only the parallelism/capacity shrinks.
#
#   Reports land in fpga/out/. Adjust PART (speed grade / package) + the clock period
#   in fpga/constraints.xdc to your board.
# ============================================================================

# cap synth multithreading: 7 forked processes inflate peak memory past the
# Docker VM during Technology Mapping (observed OOM-kill); 4 threads fits.
set_param general.maxThreads 4

set REPO [pwd]
set SRC  $REPO/src
set OUT  $REPO/fpga/out
file mkdir $OUT

# ---- target part: XCKU3P. Change the package/speed grade to your board's device
#      (e.g. -1/-2/-3 speed, -ffvb676 / other package). ----
set PART xcku3p-ffvb676-2-e

# ---- the Q4_K system hierarchy (== Makefile GLM_Q4K_CDC_SRCS) ----
set SRCS {
  glm_q4k_system_cdc glm_q4k_system cdc_async_fifo reset_sync glm_model_q4k
  ddr5_xbar weight_loader_q4k expert_cache_pf expert_cache_ctrl kv_cache_pager
  glm_decoder_block_q4k mla_attn_q4k swiglu_expert_q4k moe_router_q4k glm_matmul_q4k
  rmsnorm_unit rope_interleave_unit glm_softmax dsa_indexer topk_select glm_act
  glm_matmul_pipe sampler glm_fp_pipe
}
foreach m $SRCS { read_verilog -sv $SRC/$m.v }
read_verilog -sv $REPO/fpga/bringup_harness.v
read_xdc $REPO/fpga/constraints.xdc

# compact generics (PE_N 4->2, DDR_NCH 4->2, KV_RESIDENT 16->8, EFIFO_DEPTH 16->8,
# CACHE_SLOTS 4->2). The .vh includes (glm_fp.vh, q4k.vh, q4k_mixed.vh,
# glm_fp_pipe_lat.vh) live in src/ next to the .v, so Vivado resolves them from the
# including file's dir; -include_dirs makes it explicit.
set GEN {-generic PE_N=2 -generic DDR_NCH=2 -generic KV_RESIDENT=8 \
         -generic EFIFO_DEPTH=8 -generic CACHE_SLOTS=2 -generic ACT_HW=1}

# ============================================================================
# (1) SYNTH-ONLY resource fit of the RAW product top (no pins required)
# ============================================================================
synth_design -top glm_q4k_system_cdc -part $PART -include_dirs $SRC {*}$GEN
report_utilization -file $OUT/util_synth.rpt
write_checkpoint  -force $OUT/post_synth_system.dcp
puts "================= SYNTH-ONLY util (glm_q4k_system_cdc, compact) ========="
puts [report_utilization -return_string]

# ============================================================================
# (2) FULL place & route of the routable bring-up harness -> routed Fmax
# ============================================================================
synth_design -top bringup_harness -part $PART -include_dirs $SRC {*}$GEN
write_checkpoint  -force $OUT/post_synth_harness.dcp
opt_design
place_design
route_design
report_utilization    -hierarchical -file $OUT/util_routed.rpt
report_timing_summary -max_paths 10 -file $OUT/timing.rpt
report_design_analysis -file $OUT/design_analysis.rpt
write_checkpoint -force $OUT/post_route.dcp

# ---- headline numbers to stdout ----
puts "================= FIT SUMMARY -- bringup_harness(glm_q4k_system_cdc) on $PART (compact) ==="
puts [report_utilization -return_string]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "Worst setup slack (WNS) on core_clk 5.000 ns target: $wns ns"
puts "  -> achieved Fmax(core_clk) ~= 1 / (5.000 - WNS) GHz  (see fpga/out/timing.rpt)"
puts "  NB: routed util includes a negligible 256-FF LFSR + output XOR tree (the"
puts "      bring-up harness); util_synth.rpt is the pure product resource number."
puts "========================================================================="
