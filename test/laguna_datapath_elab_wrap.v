`timescale 1ns/1ps
//============================================================================
// test/laguna_datapath_elab_wrap.v
//   ELABORATION study for the Laguna-S-2.1 GQA attention datapath (branch
//   laguna-s-2.1, `make laguna-datapath-elab`).
//
//   A GQA attention layer composes from main's ALREADY-VERIFIED leaf blocks:
//     * glm_matmul_q4k   -- the Q4_K GEMM for the Q/K/V/O projections
//     * rmsnorm_unit     -- q/k per-head RMSNorm (LEN=head_dim=128, eps=1e-6)
//                           and the pre-attn / post-attn layer norms
//     * glm_softmax      -- the attention-score softmax
//   This wrapper instantiates each at LAGUNA's attention shape so iverilog
//   -tnull (and yosys) type/width-check the parameterization at the real
//   head_dim=128 / eps=1e-6 / window scale -- catching width overflow, $clog2
//   edges, and out-of-range part-selects the small MoE slice cannot surface.
//
//   THIS IS ELABORATION, NOT A SIM.  Data ports are intentionally left floating
//   (no stimulus, no golden).  The FUNCTIONAL Laguna attention golden is the
//   numpy reference tools/laguna_attn_ref.py (`make laguna-attn`); the bit-exact
//   RTL orchestrator that streams these leaves is the remaining silicon-adjacent
//   work, scoped in docs/LAGUNA_S21.md SS6.
//============================================================================
`include "full_laguna_s21.vh"

module laguna_datapath_elab_wrap;
    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    // Laguna eps = 1e-6 (fp32); GLM's rmsnorm_unit defaults to 1e-5 -> override.
    localparam [31:0] LAGUNA_EPS = 32'h358637BD;   // 1e-6

    // ---- q/k per-head RMSNorm : LEN = head_dim = 128, eps = 1e-6 ----
    rmsnorm_unit #(.LEN(`LAGUNA_HEAD_DIM), .LANES(4), .EPS(LAGUNA_EPS))
        u_qk_norm (.clk(clk), .rst(rst));

    // ---- Q/K/V/O projection GEMM : Q4_K core, 256-elem super-blocks ----
    glm_matmul_q4k #(.PE_M(1), .PE_N(`LAGUNA_PE_N), .KMAX(256))
        u_proj (.clk(clk), .rst(rst));

    // ---- attention-score softmax : one sliding window (512) wide ----
    glm_softmax #(.LEN(`LAGUNA_SLIDING_WIN), .LANES(2))
        u_softmax (.clk(clk), .rst(rst));

    // elaboration-only: exercise reset then finish (no functional run).
    initial begin
        #20 rst = 1'b0;
        #20 $display("laguna_datapath_elab: leaves elaborated at head_dim=%0d, eps=1e-6, window=%0d",
                     `LAGUNA_HEAD_DIM, `LAGUNA_SLIDING_WIN);
        $finish;
    end
endmodule
