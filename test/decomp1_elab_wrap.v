`timescale 1ns/1ps
//============================================================================
// test/decomp1_elab_wrap.v
//   DECOMP=1 ELABORATION wrapper for glm_q4k_system.
//
//   The shipped default is DECOMP=0, and the leaf round-trip gate (make
//   weight-decomp) proves weight_decomp itself bit-exact.  But NO gate ever
//   ELABORATED glm_q4k_system with DECOMP=1, so the entire `g_wpath` else-branch
//   (the compressed-image fetch FSM, the weight_decomp instantiation wired to
//   the SYSTEM parameters WD_*/RECON_DEPTH, and the byte-reassembly -> recon RAM
//   -> loader-refill path in src/glm_q4k_system.v ~lines 790-911) was never even
//   type/width checked.  A width/part-select/$clog2 bug in that branch would only
//   surface the day someone flips DECOMP on.
//
//   THIS IS AN ELABORATION STUDY, NOT A SIMULATION (mirrors
//   test/full_config_elab_wrap.v).  All ports except clk/rst are intentionally
//   left dangling: no stimulus, no golden, no run.  iverilog -tnull elaborates
//   (full type/width/parameter check) and exits; a functional DECOMP=1 system sim
//   is a separate, much larger task (it needs a full compressed backing image of
//   a Q4_K super-block driven through the whole memory system).
//
//   Build (elaboration only, from repo root):
//     iverilog -g2012 -I src -tnull -pfileline=1 test/decomp1_elab_wrap.v \
//       <GLM_Q4K_SYS_SRCS>
//============================================================================
module decomp1_elab_wrap (input wire clk, input wire rst);

    // Instantiate the system top with DECOMP=1 (the only non-default override);
    // every other parameter keeps its committed-slice default.  Only clk/rst are
    // connected -- this is a pure elaboration of the DECOMP=1 generate branch.
    glm_q4k_system #(
        .DECOMP(1)
    ) u_sys (
        .clk(clk),
        .rst(rst)
    );

endmodule
