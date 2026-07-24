//============================================================================
// configs/full_laguna_s21.vh  --  REAL Laguna-S-2.1 full-model configuration
//----------------------------------------------------------------------------
// PURPOSE
//   Single source of truth for the *production* Laguna-S-2.1 shape (the base
//   model of unsloth/Laguna-S-2.1-GGUF : UD-Q4_K_XL), carried as `define
//   LAGUNA_* macros alongside a small-but-faithful RTL SLICE, mirroring
//   configs/full_glm52.vh.  This LOCKS the config in code; the Laguna RTL
//   (Phases 3-6, branch laguna-s-2.1) will override its module parameters WITH
//   these macros to target the real checkpoint.
//
//   Every value is cited to its source:
//     [cfg] = config.json field of poolside/Laguna-S-2.1 (arch LagunaForCausalLM)
//             archived at docs/laguna_s21_config.json
//     [doc] = docs/LAGUNA_S21.md
//
//   NO RTL consumes this header yet -- the Laguna datapath is unbuilt.  What
//   DOES consume it today is test/laguna_config_check.v (`make laguna-config-check`),
//   an elaboration+assert gate that proves the encoded config is internally
//   consistent with the locked layer/head/expert counts (no datapath needed).
//============================================================================
`ifndef FULL_LAGUNA_S21_VH
`define FULL_LAGUNA_S21_VH

// ---- top-level model dims ----
`define LAGUNA_MODEL_DIM   3072      // [cfg] hidden_size                 (slice 128)
`define LAGUNA_L           48        // [cfg] num_hidden_layers           (slice 8)
`define LAGUNA_N_DENSE     1         // [cfg] mlp_only_layers=[0] -> layer 0 dense FFN, 1..47 MoE  (slice 1)
`define LAGUNA_VOCAB       100352    // [cfg] vocab_size                  (slice 256)
`define LAGUNA_RMS_EPS_1EN 6         // [cfg] rms_norm_eps = 1e-6  (exponent; GLM was 1e-5)

// ---- GQA attention (head count is PER-LAYER; see the layer rules below) ----
`define LAGUNA_HEAD_DIM    128       // [cfg] head_dim                    (slice 16)
`define LAGUNA_KV_HEADS    8         // [cfg] num_key_value_heads (constant across layers)  (slice 2)
`define LAGUNA_QHEADS_FULL 48        // [cfg] num_attention_heads on full-attention layers  (group 48/8=6)  (slice 4)
`define LAGUNA_QHEADS_SWA  72        // [cfg] num_attention_heads on sliding layers          (group 72/8=9)  (slice 6)
`define LAGUNA_QHEADS_MAX  72        // max over layers = sizing bound for Q-side scratch/lanes
// attention_bias = false [cfg].  No MLA latent compression, no DSA indexer (that is GLM).

// ---- sliding-window attention + per-layer layout ----
//   layer_types = [full, sliding, sliding, sliding] repeated 12x  ->  12 full + 36 sliding.
//   Equivalently: layer i is FULL-attention iff (i % 4 == 0), else SLIDING (window 512).
`define LAGUNA_SLIDING_WIN 512       // [cfg] sliding_window
`define LAGUNA_N_FULL      12        // count of full-attention layers   (derived; asserted by the check)
`define LAGUNA_N_SWA       36        // count of sliding-window layers    (derived; asserted by the check)
// function-like macros -- the per-layer schedule rules (i in 0..LAGUNA_L-1):
`define LAGUNA_IS_FULL(i)  ((((i) % 4) == 0))                 // full-attention layer?
`define LAGUNA_IS_DENSE(i) (((i) == 0))                       // dense-FFN (non-MoE) layer?
`define LAGUNA_QHEADS(i)   (`LAGUNA_IS_FULL(i) ? `LAGUNA_QHEADS_FULL : `LAGUNA_QHEADS_SWA)

// ---- dual RoPE (per attention type) ----
//   full-attention layers : YaRN scaling, PARTIAL rotary 0.5
`define LAGUNA_ROPE_FULL_THETA   500000    // [cfg] rope_parameters.full_attention.rope_theta
`define LAGUNA_ROPE_FULL_TYPE_YARN 1        // rope_type = "yarn"
`define LAGUNA_YARN_FACTOR       128        // [cfg] factor
`define LAGUNA_YARN_ORIG_MAXPOS  8192       // [cfg] original_max_position_embeddings
`define LAGUNA_YARN_BETA_SLOW    1          // [cfg] beta_slow
`define LAGUNA_YARN_BETA_FAST    32         // [cfg] beta_fast
// attention_factor = 1.4852030263919618 [cfg] -- carried as a bf16/fp32 constant in the RoPE unit (Phase 4).
`define LAGUNA_ROPE_FULL_PARTIAL_NUM 1      // partial_rotary_factor 0.5 == 1/2 (num/den, integer-exact)
`define LAGUNA_ROPE_FULL_PARTIAL_DEN 2
//   sliding layers : plain rope, FULL rotary (1.0)
`define LAGUNA_ROPE_SWA_THETA    10000      // [cfg] rope_parameters.sliding_attention.rope_theta
`define LAGUNA_ROPE_SWA_TYPE_YARN 0         // rope_type = "default" (plain)
`define LAGUNA_ROPE_SWA_PARTIAL_NUM 1       // partial_rotary_factor 1.0 == 1/1
`define LAGUNA_ROPE_SWA_PARTIAL_DEN 1

// ---- per-head output gating (softplus), ALL 48 layers ----
`define LAGUNA_OUT_GATE_PERHEAD  1          // [cfg] gating = per-head ; gating_types all 'per_head'
// exact softplus form (scale/bias, pre/post-V) is [VERIFY] against modeling_laguna.py (Phase 5).

// ---- MoE / FFN ----
`define LAGUNA_N_EXPERT    256       // [cfg] num_experts                 (slice 8)
`define LAGUNA_TOPK        10        // [cfg] num_experts_per_tok         (slice 3)
`define LAGUNA_N_SHARED    1         // [derived] shared_expert_intermediate_size present -> exactly 1 shared expert (Qwen-MoE convention; count not an explicit config field) [VERIFY modeling_laguna.py]  (slice 1)
`define LAGUNA_INTER_MOE   1024      // [cfg] moe_intermediate_size (per routed expert)  (slice 16)
`define LAGUNA_INTER_SHARED 1024     // [cfg] shared_expert_intermediate_size            (slice 16)
`define LAGUNA_INTER_DENSE 12288     // [cfg] intermediate_size (layer-0 dense FFN)      (slice 64)
`define LAGUNA_NORM_TOPK   1         // [cfg] norm_topk_prob = true (renormalize top-k gate weights)
`define LAGUNA_RSCALE      32'h40200000 // [cfg] moe_routed_scaling_factor = 2.5 (fp32)  (slice 2.5)
// moe_router_logit_softcapping = 0.0 [cfg] -> no logit softcap.

// ---- context / position width ----
//   1M-token context: POSW must cover >= 1,048,576 positions.  2^20 == 1,048,576 exactly.
`define LAGUNA_POSW        20        // [cfg] max_position_embeddings = 1048576  (slice 20)

// ---- weight quantization (UD-Q4_K_XL: Q4_K + Q6_K + Q8_0 + F16 mix) ----
//   Dequant is FORMAT-level and model-agnostic -- inherited from main unchanged.
//   256-elem Q4_K super-blocks; tools/gguf_crosscheck.py runs AS-IS on the Laguna
//   GGUF (no code change) to re-seal the dequant on the real bytes:
//     python3 tools/gguf_crosscheck.py <Laguna UD-Q4_K_XL .gguf> <llamacpp_dir>

// ---- hardware tiling knobs (microarch, not model config) ----
`define LAGUNA_PE_N        4         // attention/matmul output-lane tile width  (slice 4)
`define LAGUNA_TN          4         // swiglu output-tile width                  (slice 4)
`define LAGUNA_LM_TN       4         // LM-head GEMV tile width (VOCAB % LM_TN==0; 100352 % 4 == 0) (slice 4)
`define LAGUNA_PE_M        1         // query-token batch B (1 == committed datapath)

// ============================================================================
// SLICE reference values -- the small-but-faithful shape future Laguna TBs use.
//   Chosen to exercise BOTH patterns: layer 0 = full-attn + dense-FFN; layers
//   1..3 = sliding + MoE; layer 4 = full-attn + MoE; ... (SLICE_L=8 = two full
//   periods -> 2 full / 6 sliding / 1 dense / 7 MoE).  Q heads differ by type
//   (full=SLICE 4, sliding=SLICE 6; KV=SLICE 2 -> groups 2 / 3).  Tune when the
//   Laguna RTL lands; these are placeholders, not a committed datapath yet.
// ============================================================================
`define LAGUNA_SLICE_MODEL_DIM   128
`define LAGUNA_SLICE_L           8
`define LAGUNA_SLICE_N_DENSE     1
`define LAGUNA_SLICE_VOCAB       256
`define LAGUNA_SLICE_HEAD_DIM    16
`define LAGUNA_SLICE_KV_HEADS    2
`define LAGUNA_SLICE_QHEADS_FULL 4
`define LAGUNA_SLICE_QHEADS_SWA  6
`define LAGUNA_SLICE_SLIDING_WIN 4
`define LAGUNA_SLICE_N_EXPERT    8
`define LAGUNA_SLICE_TOPK        3
`define LAGUNA_SLICE_INTER_MOE   16
`define LAGUNA_SLICE_INTER_DENSE 64
`define LAGUNA_SLICE_POSW        20

`endif // FULL_LAGUNA_S21_VH
