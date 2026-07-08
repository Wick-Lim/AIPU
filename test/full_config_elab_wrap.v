`timescale 1ns/1ps
//============================================================================
// test/full_config_elab_wrap.v
//   FULL-CONFIG ELABORATION wrapper for glm_model_q4k (PRODUCT_ROADMAP P1.2).
//
//   Instantiates the compute-die top `glm_model_q4k` with EVERY parameter
//   overridden to the REAL 753B GLM-5.2 (UD-Q4_K_XL) production shape so an
//   elaboration-only tool (iverilog -tnull / yosys hierarchy -check) type/width
//   checks the parameterization at true scale.  This catches full-scale RTL
//   issues -- width overflow, out-of-range part-selects, $clog2 edges, negative
//   replication counts -- that the small committed SLICE (MODEL_DIM=128, ...)
//   cannot surface.
//
//   THIS IS AN ELABORATION STUDY, NOT A SIMULATION.  All model ports except
//   clk/rst are intentionally left dangling: no stimulus, no golden, no run.
//   A full-config functional sim is intractable (the LM-head GEMV alone streams
//   MODEL_DIM(6144) x VOCAB(154880) ~ 2.4e8 K-beats PER TOKEN; the 256-expert
//   MoE runs into the billions of cycles).  See docs/FULL_CONFIG_ELAB.md.
//
//   The real shape is sourced from configs/full_glm52.vh (every value cited to
//   config.json of zai-org/GLM-5.2 / docs/ACCEL_GLM52.md).  Q_LORA/KV_LORA
//   = 2048/512 are the safetensors-CONFIRMED ranks (docs/REAL_CKPT_VALIDATION.md).
//
//   ASSUMPTIONS FLAGGED (not free model config -- see docs/FULL_CONFIG_ELAB.md):
//     * S_MAX = 8  -- the attention scratch (scores/probs/vstore in mla_attn_q4k)
//       is sized by S_MAX; the real 1M context lives in the POSW=20 position
//       field, NOT in S_MAX.  S_MAX is the latent-ring / KV scratch depth and is
//       kept modest for a tractable elaboration (decoupling window from context
//       is a separate task, B7).  S_MAX sizes counters/scratch only; the
//       datapath-width parameterization under study is independent of it.
//     * INTER_DENSE = 12288  -- GLM-5.2 dense-front (layers 0..N_DENSE-1) FFN
//       intermediate_size (config.json intermediate_size; docs/ACCEL_GLM52.md).
//       Distinct from moe_intermediate_size (INTER_MOE=2048).
//
//   Build (elaboration only, from repo root):
//     iverilog -g2012 -I src -I configs -tnull -pfileline=1 \
//       test/full_config_elab_wrap.v \
//       src/glm_model_q4k.v src/glm_decoder_block_q4k.v src/mla_attn_q4k.v \
//       src/swiglu_expert_q4k.v src/moe_router_q4k.v src/glm_matmul_q4k.v \
//       src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v \
//       src/dsa_indexer.v src/topk_select.v src/glm_act.v \
//       src/glm_matmul_pipe.v src/glm_fp_pipe.v
//============================================================================
`include "full_glm52.vh"

module full_config_elab_wrap (input wire clk, input wire rst);

    glm_model_q4k #(
        .MODEL_DIM  (`GLM52_MODEL_DIM),    // 6144    hidden_size
        .L          (`GLM52_L),            // 78      num_hidden_layers
        .N_DENSE    (`GLM52_N_DENSE),      // 3       first_k_dense_replace
        .VOCAB      (`GLM52_VOCAB),        // 154880  vocab_size
        .H_HEADS    (`GLM52_H_HEADS),      // 64      num_attention_heads
        .NOPE       (`GLM52_NOPE),         // 192     qk_nope_head_dim
        .ROPE       (`GLM52_ROPE),         // 64      qk_rope_head_dim
        .V_DIM      (`GLM52_V_DIM),        // 256     v_head_dim
        .Q_LORA     (`GLM52_Q_LORA),       // 2048    q_lora_rank (confirmed)
        .KV_LORA    (`GLM52_KV_LORA),      // 512     kv_lora_rank
        .S_MAX      (`GLM52_S_MAX),        // 8       latent-ring depth (FLAGGED)
        .TOPK_ATTN  (`GLM52_TOPK_ATTN),    // 2048    index_topk (DSA budget)
        .THETA      (`GLM52_THETA),        // 8000000 rope_theta
        .PE_N       (`GLM52_PE_N),         // 4       matmul output-lane tile
        .POSW       (`GLM52_POSW),         // 20      2^20 >= 1M context
        .N_EXPERT   (`GLM52_N_EXPERT),     // 256     n_routed_experts
        .TOPK       (`GLM52_TOPK),         // 8       num_experts_per_tok
        .INTER_MOE  (`GLM52_INTER_MOE),    // 2048    moe_intermediate_size
        .INTER_DENSE(`GLM52_INTER_DENSE),  // 12288   intermediate_size (FLAGGED)
        .RSCALE     (`GLM52_RSCALE),       // 2.5     routed_scaling_factor
        .TN         (`GLM52_TN),           // 4       swiglu output-tile
        .BLK        (`GLM52_BLK),          // 128     weight_block_size
        .LM_TN      (`GLM52_LM_TN),        // 4       LM-head GEMV tile (VOCAB%LM_TN==0)
        .PE_M       (`GLM52_PE_M)          // 1       query-token batch B
    ) u_full (
        .clk(clk), .rst(rst)
        // all remaining ports intentionally left unconnected -- elaboration only
    );

endmodule
