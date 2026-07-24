`timescale 1ns/1ps
//============================================================================
// test/laguna_config_check.v
//   Phase-1 CONFIG-CONSISTENCY gate for the Laguna-S-2.1 port (branch
//   laguna-s-2.1).  Includes configs/full_laguna_s21.vh and proves the encoded
//   config is internally consistent with the locked layer / head / expert
//   counts from poolside/Laguna-S-2.1 config.json.
//
//   THIS IS AN ELABORATION + INTEGER-ASSERT gate -- there is NO Laguna datapath
//   yet.  It exists so a transcription error in the config header (a wrong count,
//   a broken per-layer rule, a non-divisible group) fails LOUDLY at Phase 1,
//   before any RTL is sized against these macros.
//
//   Build/run:
//     iverilog -g2012 -I configs -o build/laguna_config_check test/laguna_config_check.v
//     vvp build/laguna_config_check        ->  "ALL <N> TESTS PASSED"
//   (wired as `make laguna-config-check`.)
//============================================================================
`include "full_laguna_s21.vh"

// Injection build (`make laguna-config-check` step 2): override the full-layer
// rule to a WRONG schedule (full iff i%3==0 -> 16 full / 32 sliding).  The
// layer-count asserts must catch this and the gate must FAIL, proving they are
// load-bearing.  Default build leaves the header's rule untouched.
`ifdef LAGUNA_INJECT_BADSCHED
  `undef LAGUNA_IS_FULL
  `define LAGUNA_IS_FULL(i) ((((i) % 3) == 0))
`endif

module laguna_config_check;

    integer tests = 0;
    integer errors = 0;

    // -- derive the per-layer schedule from the header's RULES and count it,
    //    so we test the RULES, not a hand-copied array.
    integer i;
    integer n_full;      // full-attention layers
    integer n_swa;       // sliding-window layers
    integer n_dense;     // dense-FFN layers
    integer n_moe;       // MoE layers
    integer qh;

    task check; input cond; input [8*48-1:0] name; begin
        tests = tests + 1;
        if (!cond) begin
            errors = errors + 1;
            $display("FAIL: %0s", name);
        end
    end endtask

    initial begin
        n_full = 0; n_swa = 0; n_dense = 0; n_moe = 0;

        for (i = 0; i < `LAGUNA_L; i = i + 1) begin
            if (`LAGUNA_IS_FULL(i))  n_full  = n_full  + 1; else n_swa = n_swa + 1;
            if (`LAGUNA_IS_DENSE(i)) n_dense = n_dense + 1; else n_moe = n_moe + 1;

            // every layer's Q-head count is divisible by KV heads (valid GQA grouping)
            qh = `LAGUNA_QHEADS(i);
            check((qh % `LAGUNA_KV_HEADS) == 0, "Q heads not divisible by KV heads (bad GQA group)");
            // Q-head count == the LITERAL 48/72 for its type (RHS pins literals, not the
            //   same QHEADS_FULL/SWA macros, so a wrong QHEADS_FULL value is caught).
            check(qh == (`LAGUNA_IS_FULL(i) ? 48 : 72),
                  "per-layer Q head count wrong for attention type (vs literal 48/72)");
            // POSITION, not just count: only layer 0 may be the dense-FFN layer.
            if (i != 0) check(!`LAGUNA_IS_DENSE(i), "only layer 0 may be dense-FFN");
        end

        // ---- layer-layout totals match the locked config ----
        check(`LAGUNA_L == 48,           "num_hidden_layers != 48");
        check(n_full  == `LAGUNA_N_FULL, "full-attention layer count != LAGUNA_N_FULL");
        check(n_swa   == `LAGUNA_N_SWA,  "sliding-window layer count != LAGUNA_N_SWA");
        check(n_full  == 12,             "full-attention layers != 12");
        check(n_swa   == 36,             "sliding-window layers != 36");
        check(n_dense == `LAGUNA_N_DENSE, "dense-FFN layer count != LAGUNA_N_DENSE");
        check(n_dense == 1,              "dense-FFN layers != 1 (layer 0 only)");
        check(n_moe   == 47,             "MoE layers != 47");

        // ---- schedule POSITION anchors (config.json fixes layer 0 = full-attn + dense;
        //      a right-count / wrong-offset rule -- e.g. full iff i%4==1 -- passes the
        //      count asserts above but fails these) ----
        check(`LAGUNA_IS_FULL(0)  == 1, "layer 0 must be full-attention");
        check(`LAGUNA_IS_DENSE(0) == 1, "layer 0 must be the dense-FFN layer");
        check(`LAGUNA_IS_FULL(1)  == 0, "layer 1 must be sliding-window");
        check(`LAGUNA_IS_FULL(4)  == 1, "layer 4 must be full-attention (period 4)");

        // ---- attention dims ----
        check(`LAGUNA_HEAD_DIM == 128,   "head_dim != 128");
        check(`LAGUNA_KV_HEADS == 8,     "num_key_value_heads != 8");
        check(`LAGUNA_QHEADS_FULL == 48, "full-layer Q heads != 48");
        check(`LAGUNA_QHEADS_SWA  == 72, "sliding-layer Q heads != 72");
        check((`LAGUNA_QHEADS_FULL / `LAGUNA_KV_HEADS) == 6, "full-layer GQA group != 6");
        check((`LAGUNA_QHEADS_SWA  / `LAGUNA_KV_HEADS) == 9, "sliding-layer GQA group != 9");
        check(`LAGUNA_QHEADS_MAX == `LAGUNA_QHEADS_SWA, "QHEADS_MAX must bound the larger (sliding) count");

        // ---- MoE / FFN ----
        check(`LAGUNA_N_EXPERT == 256,   "num_experts != 256");
        check(`LAGUNA_TOPK == 10,        "num_experts_per_tok != 10");
        check(`LAGUNA_TOPK <= `LAGUNA_N_EXPERT, "top-k exceeds expert count");
        check(`LAGUNA_N_SHARED == 1,     "shared experts != 1");
        check(`LAGUNA_INTER_MOE == 1024, "moe_intermediate_size != 1024");
        check(`LAGUNA_INTER_SHARED == 1024, "shared_expert_intermediate_size != 1024");
        check(`LAGUNA_INTER_DENSE == 12288, "dense intermediate_size != 12288");
        check(`LAGUNA_RSCALE == 32'h40200000, "routed_scaling_factor fp32 != 2.5");

        // ---- context / position width ----
        check((1 << `LAGUNA_POSW) == 1048576, "2^POSW must equal the 1,048,576 context exactly");
        check(`LAGUNA_SLIDING_WIN == 512, "sliding_window != 512");
        check(`LAGUNA_SLIDING_WIN < (1 << `LAGUNA_POSW), "sliding window must be < context");

        // ---- RoPE (dual) ----
        check(`LAGUNA_ROPE_FULL_THETA == 500000, "full-attn rope_theta != 500000");
        check(`LAGUNA_ROPE_SWA_THETA  == 10000,  "sliding rope_theta != 10000");
        check(`LAGUNA_ROPE_FULL_TYPE_YARN == 1,  "full-attn rope must be YaRN");
        check(`LAGUNA_ROPE_SWA_TYPE_YARN  == 0,  "sliding rope must be plain");
        // YaRN scaling scalars -- load-bearing for the Phase-4 RoPE unit; pin each
        //   literal so a transcription slip (factor 128->28, beta_fast 32->3, ...) fails.
        check(`LAGUNA_YARN_FACTOR      == 128,   "yarn factor != 128");
        check(`LAGUNA_YARN_ORIG_MAXPOS == 8192,  "yarn original_max_position_embeddings != 8192");
        check(`LAGUNA_YARN_BETA_SLOW   == 1,     "yarn beta_slow != 1");
        check(`LAGUNA_YARN_BETA_FAST   == 32,    "yarn beta_fast != 32");
        // partial rotary: full 0.5 (1/2), sliding 1.0 (1/1)
        check((`LAGUNA_ROPE_FULL_PARTIAL_NUM * 2) == `LAGUNA_ROPE_FULL_PARTIAL_DEN, "full partial_rotary != 0.5");
        check(`LAGUNA_ROPE_SWA_PARTIAL_NUM == `LAGUNA_ROPE_SWA_PARTIAL_DEN, "sliding partial_rotary != 1.0");

        // ---- misc locked scalars ----
        check(`LAGUNA_MODEL_DIM == 3072, "hidden_size != 3072");
        check(`LAGUNA_VOCAB == 100352,   "vocab_size != 100352");
        check((`LAGUNA_VOCAB % `LAGUNA_LM_TN) == 0, "VOCAB must be divisible by LM_TN");
        check(`LAGUNA_RMS_EPS_1EN == 6,  "rms_norm_eps exponent != 6 (1e-6)");
        check(`LAGUNA_OUT_GATE_PERHEAD == 1, "per-head output gating flag must be set");
        check(`LAGUNA_NORM_TOPK == 1,    "norm_topk_prob must be set");

        // ---- SLICE self-consistency (the future-TB shape must obey the same rules) ----
        check((`LAGUNA_SLICE_QHEADS_FULL % `LAGUNA_SLICE_KV_HEADS) == 0, "slice full Q not divisible by slice KV");
        check((`LAGUNA_SLICE_QHEADS_SWA  % `LAGUNA_SLICE_KV_HEADS) == 0, "slice sliding Q not divisible by slice KV");
        check((`LAGUNA_SLICE_VOCAB % `LAGUNA_LM_TN) == 0, "slice VOCAB not divisible by LM_TN");
        check(`LAGUNA_SLICE_TOPK <= `LAGUNA_SLICE_N_EXPERT, "slice top-k exceeds slice expert count");
        check(`LAGUNA_SLICE_SLIDING_WIN < (1 << `LAGUNA_SLICE_POSW), "slice window >= slice context");
        check((`LAGUNA_SLICE_L % 4) == 0, "slice L not a whole full/sliding period (exercise both types)");

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "laguna_config_check had mismatches");
        end
        $display("ALL %0d TESTS PASSED  (Laguna-S-2.1 config internally consistent with the locked layer/head/expert counts)", tests);
        $finish;
    end

endmodule
