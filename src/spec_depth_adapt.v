`timescale 1ns/1ps
//============================================================================
// spec_depth_adapt.v -- ADAPTIVE DRAFT-DEPTH POLICY for spec_decode_seq
//                       (docs/R3_APPLIANCE_SPEC.md section 9 -- runtime-
//                        variable K: deep drafts HURT at low accept rates)
//----------------------------------------------------------------------------
// WHY
//   The optimal speculative draft depth is a function of the acceptance rate
//   r: expected tokens/pass at depth k is 1 + r + r^2 + ... + r^k, but every
//   extra draft position costs an MTP-chain step and a wider verify batch,
//   and at low r the deep drafts are almost always thrown away.  So the
//   compile-time DRAFT_K stays the MAXIMUM depth, and this module picks the
//   RUNTIME per-pass depth k_cur in [1 .. DRAFT_K] from the observed accept
//   results (spec_decode_seq's pass_done/pass_acc/pass_dep taps).
//
// POLICY (deliberately tiny -- a saturating streak counter, exactly
//   re-derivable in software; test/spec_depth_adapt_tb.v models it 1:1):
//     reset          : k_cur = 1, streak = 0    (conservative cold start)
//     each pass_done : full  := (pass_dep != 0) && (pass_acc == pass_dep)
//                      early := (pass_acc <  pass_dep)
//       * full  (every scanned draft accepted):
//             streak == THRESH-1 -> k_cur <= min(k_cur+1, DRAFT_K); streak <= 0
//             else               -> streak <= streak + 1
//         (i.e. THRESH consecutive fully-accepted passes raise the depth by 1)
//       * early (a draft was rejected before the end of the scan):
//             k_cur <= max(k_cur-1, 1); streak <= 0
//       * pass_dep == 0 (empty batch): no evidence either way -> hold state
//
// OUTPUT-INVARIANCE (the repo's moat -- spec==greedy):
//   k_cur only changes HOW MANY tokens are drafted/scanned per pass.  The
//   accept/reject rule and every committed token live in spec_decode_seq,
//   which commits ONLY the model's own greedy argmaxes (truth_vec) for ANY
//   k_cur value -- so ANY depth schedule this module produces (or a buggy /
//   adversarial one) is output-invariant BY CONSTRUCTION: the committed
//   stream equals greedy decode regardless; the policy affects tokens/pass
//   throughput only.  Proven in test/spec_depth_adapt_tb.v (forced schedules,
//   closed loop vs an independent software model, K = 2/3/4/6/8).
//
// TIMING: spec_decode_seq pulses pass_done on its m_1 commit cycle (one cycle
//   after pass_valid); k_cur updates on the following edge -- >= K cycles
//   before the next legal pass (pass_valid spacing >= K+1), so the depth the
//   orchestrator samples for a pass is always stable.
//
// DISCIPLINE: synchronous active-high reset, every output registered, no
//   latch, no combinational loop, deterministic -- pure integer control logic.
//============================================================================
module spec_depth_adapt #(
    parameter integer DRAFT_K = 2,   // compile-time MAXIMUM depth (>= 1)
    parameter integer THRESH  = 2,   // consecutive full-accept passes per raise (>= 1)
    // ---- derived (do NOT override) -- same width as spec_decode_seq ports ----
    parameter integer DKW     = (DRAFT_K <= 1) ? 1 : $clog2(DRAFT_K + 1)
)(
    input  wire            clk,
    input  wire            rst,        // sync, active-high
    // ---- per-pass observation (wire to spec_decode_seq's pass_* taps) ----
    input  wire            pass_done,  // 1-cycle pulse: a verify pass consumed
    input  wire [DKW-1:0]  pass_acc,   // p     : drafts accepted that pass
    input  wire [DKW-1:0]  pass_dep,   // nd_eff: drafts actually scanned
    // ---- depth for the NEXT pass (wire to spec_decode_seq.k_cur) ----
    output reg  [DKW-1:0]  k_cur       // in [1 .. DRAFT_K]; valid from reset
);
    // streak counts 0 .. THRESH-1 (it is consumed on reaching THRESH-1)
    localparam integer   SW    = (THRESH <= 2) ? 1 : $clog2(THRESH);
    localparam integer   TM1I  = THRESH - 1;
    localparam integer   ONEI  = 1;
    localparam [SW-1:0]  T_M1  = TM1I[SW-1:0];       // THRESH-1, SW-bit (exact)
    localparam [DKW-1:0] K_DKW = DRAFT_K[DKW-1:0];   // K, DKW-bit (exact)
    localparam [DKW-1:0] ONE_K = ONEI[DKW-1:0];

    reg [SW-1:0] streak;   // consecutive fully-accepted passes so far

    wire dep_nz = (pass_dep != {DKW{1'b0}});
    wire full   = dep_nz && (pass_acc == pass_dep);  // all scanned drafts hit
    wire early  = (pass_acc < pass_dep);             // rejected before the end

    always @(posedge clk) begin
        if (rst) begin
            k_cur  <= ONE_K;
            streak <= {SW{1'b0}};
        end else begin
            k_cur  <= k_cur;      // hold unless a completed pass updates it
            streak <= streak;
            if (pass_done) begin
                if (full) begin
                    if (streak == T_M1) begin
                        streak <= {SW{1'b0}};
                        if (k_cur < K_DKW)
                            k_cur <= k_cur + 1'b1;   // raise after THRESH fulls
                    end else
                        streak <= streak + 1'b1;
                end else if (early) begin
                    streak <= {SW{1'b0}};
                    if (k_cur > ONE_K)
                        k_cur <= k_cur - 1'b1;       // back off on early reject
                end
                // pass_dep == 0 (empty batch): no evidence -> hold
            end
        end
    end
endmodule
