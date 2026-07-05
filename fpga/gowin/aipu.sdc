//============================================================================
// aipu.sdc  --  Timing constraints for glm_fp8_system_cdc (GLM-5.2-FP8 top)
//----------------------------------------------------------------------------
// Target: Gowin GW5AT-138 (Sipeed Tang Mega 138K Pro) via `gw_sh` (GowinSynthesis
//         + Place & Route).  Gowin consumes SDC-style timing constraints; add
//         this file to the project with `add_file -type sdc aipu.sdc` (done by
//         build_gowin.tcl).
//
// WHY TWO CLOCKS:
//   glm_fp8_system_cdc is a REAL 2-clock design (see src/glm_fp8_system_cdc.v):
//     * host_clk : USB-C device domain (host-facing start/prompt -> busy/done/token)
//     * core_clk : compute-die domain  (glm_model_fp8 + memory system + loaders)
//   The two domains are ASYNCHRONOUS (unrelated frequency & phase).  Every signal
//   that crosses does so ONLY through a cdc_async_fifo (gray-coded pointers +
//   2-FF synchronizers) or an explicit 2-FF synchronizer -- there is NO
//   combinational path between the domains.  Therefore the correct constraint is
//   to declare BOTH clocks and mark them mutually ASYNCHRONOUS so the timer does
//   NOT try to close (nonexistent) cross-domain paths.  Omitting this makes the
//   tool report bogus cross-domain setup/hold failures.
//============================================================================

// ---------------------------------------------------------------------------
// CLOCK DEFINITIONS  (periods are TARGETS to tune after the first fit run)
//   host_clk : 10.0 ns  -> 100 MHz  (USB-C device-side logic; modest)
//   core_clk : 15.0 ns  ->  66.7 MHz (compute die; the FP8 matmul datapath is
//              deep, so start conservative and tighten once you see the report)
//   After the first P&R, read the achieved Fmax per clock and re-target these
//   periods (e.g. drop core_clk toward the reported max, or relax if it fails).
// ---------------------------------------------------------------------------
create_clock -name host_clk -period 10.0 [get_ports {host_clk}]
create_clock -name core_clk -period 15.0 [get_ports {core_clk}]

// ---------------------------------------------------------------------------
// ASYNCHRONOUS CLOCK GROUPS
//   Mark host_clk and core_clk as asynchronous.  This is the SDC equivalent of
//   false-pathing every host<->core crossing; it is CORRECT here because all
//   crossings are gray-FIFO / 2-FF synchronized (proven in the RTL + formal CDC
//   checks).  Do NOT remove this -- without it the timer invents impossible
//   cross-domain paths.
// ---------------------------------------------------------------------------
set_clock_groups -asynchronous \
    -group [get_clocks {host_clk}] \
    -group [get_clocks {core_clk}]

// ---------------------------------------------------------------------------
// (OPTIONAL) Belt-and-suspenders false paths on the 2-FF synchronizer inputs.
//   set_clock_groups above already covers these.  If you prefer explicit false
//   paths (or your Gowin version handles the async groups differently), you can
//   instead false-path the specific crossing nets.  Left commented; tune to the
//   synthesized net names shown in the timing report if needed:
// set_false_path -from [get_clocks {core_clk}] -to [get_clocks {host_clk}]
// set_false_path -from [get_clocks {host_clk}] -to [get_clocks {core_clk}]

// ---------------------------------------------------------------------------
// I/O TIMING (placeholders -- tune to the real board once pins are assigned).
//   The host interface is source-synchronous to host_clk; the memory-side ports
//   (DDR5 fabric, Flash, weight-loader, GDDR6 stubs) belong to core_clk and, on
//   a real board, are driven by their own controller/PHY IP.  Until the board
//   integration is defined, leave I/O delays unconstrained (internal Fmax is the
//   D0.2 answer we want first) or set nominal budgets like:
// set_input_delay  -clock host_clk 2.0 [get_ports {start prompt_tok* start_pos* s_len*}]
// set_output_delay -clock host_clk 2.0 [get_ports {busy done next_tok* tok_valid}]
// ---------------------------------------------------------------------------

//============================================================================
// PIN ASSIGNMENTS -- NOTE: these do NOT go in the SDC.
//----------------------------------------------------------------------------
//   In the Gowin flow, physical PIN LOCATIONS and I/O standards live in a
//   ".cst" (physical Constraints) file, NOT this ".sdc" (timing) file.  The
//   pinout is board-specific (Tang Mega 138K Pro has its own pin map), so it is
//   left for the user to fill in.  Create fpga/gowin/aipu.cst and add it with
//   `add_file -type cst aipu.cst`.  Example .cst syntax (adapt pin names to the
//   Tang Mega 138K Pro schematic / Gowin device pin table):
//
//     // clocks
//     IO_LOC  "host_clk" <PIN>;
//     IO_PORT "host_clk" IO_TYPE=LVCMOS33;
//     IO_LOC  "core_clk" <PIN>;
//     IO_PORT "core_clk" IO_TYPE=LVCMOS33;
//     // resets
//     IO_LOC  "host_rst" <PIN>;
//     IO_LOC  "core_rst" <PIN>;
//     // ... (host interface + any brought-out debug pins) ...
//
//   IMPORTANT: this design has THOUSANDS of memory-side port bits (e.g.
//   mem_resp_data is DDR_NCH*DDR_DATA_W wide, logits is VOCAB*16, etc.).  Those
//   CANNOT all become real package pins -- no package has that many user I/O.
//   For a full Place & Route you must bury the wide memory-side ports inside a
//   bring-up harness (see fpga/gemm_harness.v / fpga/sm_harness.v for the
//   pattern, and fpga/README.md "Full P&R vs synthesis-only").  For the D0.2
//   RESOURCE FIT, run synthesis-only (FLOW=syn), which needs NO pin assignments.
//============================================================================
