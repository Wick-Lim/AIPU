# GLM-5.2-FP8 accelerator -- Verilog build / simulation / synth / formal
#
#   make all        -> the GLM prove-it gate: unittests + synth-glm + formal
#   make unittests  -> build + run EVERY per-unit TB (ALL N TESTS PASSED each)
#   make synth-glm  -> yosys elaborate + `check -assert` the whole product top
#   make formal     -> BMC of the memory-system controllers
#   make lint       -> verilator --lint-only on the GLM top (diagnostic; not yet
#                      -Wall clean -- informational, NOT part of `all`)
#   make coverage   -> verilator --coverage-line/-toggle structural coverage of
#                      the clean-verilating TB subset (per-module + merged report)
#   make host-test  -> host-side runtime scaffold self-test (host/test_aipu.py)
#   make clean      -> remove build artifacts and the generated VCDs

IVERILOG  ?= iverilog
VVP       ?= vvp
VERILATOR ?= verilator
YOSYS     ?= yosys

BUILD_DIR  := build
IFLAGS := -g2012 -Wall -I src

.PHONY: all unittests q4k spec-slow cache-study formal formal-ind bitacc lint host-test synth-glm synth-glm-q4k synth-glm-compact sim-glm-compact cdc coverage clean

# `all` is the GLM-5.2-FP8 prove-it gate (main's product): every per-unit TB, the
# whole-chip structural sign-off, and the memory-controller formal proofs.
all: unittests synth-glm formal


# Build + run every per-unit TB.  attention_unit additionally needs softmax_unit.
unittests:
	@mkdir -p $(BUILD_DIR)
	@# expert_predictor_tb reads the generated routing trace (tools/glm_trace.hex);
	@# regenerate it so `unittests` is self-contained on a fresh clone (deterministic seed).
	@python3 tools/route_trace.py --dump >/dev/null
	@# ---- GLM-5.2 datapath units (bf16/fp32, fp32-golden verified) ----
	@# (the old scalar-TPU per-unit TBs were removed with legacy/.)
	@# rmsnorm_unit: bf16 in/out, fp32 reduce + rsqrt; the FP numerics foundation.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/rmsnorm_unit_sim test/rmsnorm_unit_tb.v src/rmsnorm_unit.v
	@printf '[%s] ' "rmsnorm_unit"; $(VVP) $(BUILD_DIR)/rmsnorm_unit_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: rmsnorm_unit"; exit 1; }
	@# topk_select: top-K of N fp32 scores (DSA top-2048, MoE router top-8), ref-sort golden.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/topk_select_sim test/topk_select_tb.v src/topk_select.v
	@printf '[%s] ' "topk_select"; $(VVP) $(BUILD_DIR)/topk_select_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: topk_select"; exit 1; }
	@# glm_matmul: bf16 x bf16 -> fp32-accum -> bf16 GEMM workhorse (QKV/O/FFN/experts/LM head).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_sim test/glm_matmul_tb.v src/glm_matmul.v
	@printf '[%s] ' "glm_matmul"; $(VVP) $(BUILD_DIR)/glm_matmul_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul"; exit 1; }
	@# glm_act: bf16 sigmoid + silu (MoE router gating + SwiGLU experts), fp32 internal.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_act_sim test/glm_act_tb.v src/glm_act.v
	@printf '[%s] ' "glm_act"; $(VVP) $(BUILD_DIR)/glm_act_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_act"; exit 1; }
	@# rope_interleave_unit: decoupled interleaved RoPE (MLA q_rope/k_rope, DSA indexer),
	@# fp32 angle table theta=8e6 to 1M positions. (slow TB: full position sweep.)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/rope_sim test/rope_interleave_unit_tb.v src/rope_interleave_unit.v
	@printf '[%s] ' "rope_interleave_unit"; $(VVP) $(BUILD_DIR)/rope_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: rope_interleave_unit"; exit 1; }
	@# glm_fp_pipe: pipelined FP modules (mul/add/mac/rsqrt/exp), bit-exact vs glm_fp.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp_pipe_sim test/glm_fp_pipe_tb.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp_pipe"; $(VVP) $(BUILD_DIR)/glm_fp_pipe_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp_pipe"; exit 1; }
	@# glm_matmul_pipe: high-fmax bf16 GEMM on pipelined MACs (L-way interleaved accumulate).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_pipe_sim test/glm_matmul_pipe_tb.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_matmul_pipe"; $(VVP) $(BUILD_DIR)/glm_matmul_pipe_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul_pipe"; exit 1; }
	@# fp8_e4m3: GLM-5.2-FP8 E4M3 primitives (decode/encode-RNE/mul), exhaustive 256x256 vs fp64.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/fp8_e4m3_sim test/fp8_e4m3_tb.v
	@printf '[%s] ' "fp8_e4m3"; $(VVP) $(BUILD_DIR)/fp8_e4m3_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: fp8_e4m3"; exit 1; }
	@# glm_matmul_fp8: FP8 E4M3 GEMM (4x4 mantissa mult + fp32 accumulate + block scale -> bf16).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_fp8_sim test/glm_matmul_fp8_tb.v src/glm_matmul_fp8.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_matmul_fp8"; $(VVP) $(BUILD_DIR)/glm_matmul_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul_fp8"; exit 1; }
	@# swiglu_expert_fp8: FP8 SwiGLU expert (gate/up/down via glm_matmul_fp8 + bf16 silu*up tail).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/swiglu_expert_fp8_sim test/swiglu_expert_fp8_tb.v src/swiglu_expert_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "swiglu_expert_fp8"; $(VVP) $(BUILD_DIR)/swiglu_expert_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert_fp8"; exit 1; }
	@# swiglu_expert_fp8 PE_M batch: B rows per ONE weight fetch (== B single runs, bit-exact) -- the batching keystone.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/swiglu_expert_fp8_pem_sim test/swiglu_expert_fp8_pem_tb.v src/swiglu_expert_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "swiglu_expert_fp8(PE_M batch)"; $(VVP) $(BUILD_DIR)/swiglu_expert_fp8_pem_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert_fp8_pem"; exit 1; }
	@# mla_attn_fp8: MLA attention, 7 weight projections via glm_matmul_fp8, bf16 attention/rope/norm/softmax/dsa.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_sim test/mla_attn_fp8_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8"; exit 1; }
	@# moe_router_fp8: MoE router, W_g GEMV via glm_matmul_fp8, bf16 sigmoid/topk/renorm*2.5 tail.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/moe_router_fp8_sim test/moe_router_fp8_tb.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router_fp8"; $(VVP) $(BUILD_DIR)/moe_router_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router_fp8"; exit 1; }
	@# PE_M batch: moe_router_fp8 + mla_attn_fp8 process B rows per ONE weight fetch (== B single runs, bit-exact).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/moe_router_fp8_pem_sim test/moe_router_fp8_pem_tb.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router_fp8(PE_M)"; $(VVP) $(BUILD_DIR)/moe_router_fp8_pem_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router_fp8_pem"; exit 1; }
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_pem_sim test/mla_attn_fp8_pem_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(PE_M)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_pem_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_pem"; exit 1; }
	@# mla_attn_fp8 per-position: each PE_M row at its OWN query position (per-row RoPE) == its single-token run (product-grade batched decode).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_ppos_sim test/mla_attn_fp8_ppos_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(per-position)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_ppos_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_ppos"; exit 1; }
	@# mla_attn_fp8 per-row s_len: each PE_M row at its OWN causal extent (score mask before softmax) == its single-token run at that s_len (shared KV prefix).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_pslen_sim test/mla_attn_fp8_pslen_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(per-slen)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_pslen_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_pslen"; exit 1; }
	@# mla_attn_fp8 sparse+per-row oracle (task B2): batched row === PE_M=1 reference on that row's
	@# own (x,pos,s_len) with S_MAX>TOPK, + fetch-sharing (one weight/kc fetch per distinct key).
	@# The distinct-x sparse compare is the pending-B6 oracle (guarded by SPARSE_XFAIL, default off).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_sparse_perrow_sim test/mla_attn_fp8_sparse_perrow_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(sparse-perrow)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_sparse_perrow_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_sparse_perrow"; exit 1; }
	@# mla_attn_fp8 MULTI-SEQ (PER_ROW_SEQ=1, A2): B DIFFERENT sequences batched -- each row attends
	@# its OWN KV window (routed by kc_seq) yet SHARES the query-side weight fetch. Bit-exact vs
	@# per-seq PE_M=1 goldens + sequence-routing + weight amortization (ms beats < 2x single).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_multiseq_sim test/mla_attn_fp8_multiseq_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(multi-seq)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_multiseq_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_multiseq"; exit 1; }
	@# mla_attn_fp8 MULTI-SEQ + DSA_REAL_IDX=1 (A2): SPARSE query-DEPENDENT selection with the
	@# PER-SEQUENCE kidx_buf pre-fetch -- each row scores against its OWN sequence's index vectors.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_multiseq_dsareal_sim test/mla_attn_fp8_multiseq_dsareal_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(multi-seq DSA-real)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_multiseq_dsareal_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_multiseq_dsareal"; exit 1; }
	@# mla_attn_fp8 SWIN-vs-S_MAX decouple (task B7): attention scratch sized by SWIN (top-K window),
	@# key indices still span S_MAX. Proves SWIN=8 out === SWIN=64(=S_MAX) out bit-exact at S_MAX=64.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_swin_decouple_sim test/mla_attn_fp8_swin_decouple_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(swin-decouple)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_swin_decouple_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_swin_decouple"; exit 1; }
	@# glm_decoder_block_fp8: one full GLM-5.2-FP8 decoder layer (mla_attn_fp8 + moe_router_fp8 + swiglu_expert_fp8).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_decoder_block_fp8_sim test/glm_decoder_block_fp8_tb.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_decoder_block_fp8"; $(VVP) $(BUILD_DIR)/glm_decoder_block_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_decoder_block_fp8"; exit 1; }
	@# glm_decoder_block_fp8 UNION-SKIP: PE_M>1 MoE loop fetches ONLY the union of experts any row selected (< N_EXPERT), byte-identical vs PE_M=1 per row.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_decoder_block_fp8_union_sim test/glm_decoder_block_fp8_union_tb.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_decoder_block_fp8(union-skip)"; $(VVP) $(BUILD_DIR)/glm_decoder_block_fp8_union_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_decoder_block_fp8_union"; exit 1; }
	@# glm_model_fp8: FULL GLM-5.2-FP8 forward pass (embed -> L FP8 layers -> norm -> LM head -> next-token).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_fp8_sim test/glm_model_fp8_tb.v src/glm_model_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_model_fp8"; $(VVP) $(BUILD_DIR)/glm_model_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_fp8"; exit 1; }
	@# glm_model_fp8 PE_M batch: the WHOLE forward pass processes B token-rows per ONE weight fetch (== B single runs, per-row argmax).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_fp8_pem_sim test/glm_model_fp8_pem_tb.v src/glm_model_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_model_fp8(PE_M batch)"; $(VVP) $(BUILD_DIR)/glm_model_fp8_pem_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_fp8_pem"; exit 1; }
	@# glm_model_fp8 MULTI-SEQ (PER_ROW_SEQ=1, A2): FULL model batches B DIFFERENT sequences --
	@# each row attends its OWN KV window (kc_seq routed model->decoder->mla->cache), per-row
	@# argmax/logits bit-exact vs per-seq PE_M=1 goldens, query-side weight stream shared.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_fp8_multiseq_sim test/glm_model_fp8_multiseq_tb.v src/glm_model_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_model_fp8(multi-seq)"; $(VVP) $(BUILD_DIR)/glm_model_fp8_multiseq_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_fp8_multiseq"; exit 1; }
	@# glm_model_fp8 MULTI-SEQ SCALE-UP (A2): B=4 DIFFERENT sequences batched in ONE forward --
	@# all 4 rows per-row bit-exact vs per-seq PE_M=1; dense + sparse; query-side weights shared (~52%).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_fp8_multiseq4_sim test/glm_model_fp8_multiseq4_tb.v src/glm_model_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_model_fp8(multi-seq B=4)"; $(VVP) $(BUILD_DIR)/glm_model_fp8_multiseq4_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_fp8_multiseq4"; exit 1; }
	@# mtp_head_fp8: FP8 multi-token-prediction head (W_proj+decoder_block_fp8 FP8, bf16 LM head + norms).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mtp_head_fp8_sim test/mtp_head_fp8_tb.v src/mtp_head_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "mtp_head_fp8"; $(VVP) $(BUILD_DIR)/mtp_head_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mtp_head_fp8"; exit 1; }
	@# mtp_head_fp8 PE_M batch: B MTP token-rows per ONE weight fetch (per-row bit-exact vs PE_M=1; dense all-port weight-share).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mtp_head_fp8_pem_sim test/mtp_head_fp8_pem_tb.v src/mtp_head_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "mtp_head_fp8(PE_M batch)"; $(VVP) $(BUILD_DIR)/mtp_head_fp8_pem_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mtp_head_fp8_pem"; exit 1; }
	@# spec_decode_top: MTP speculative-decode loop (glm_model_fp8 + mtp_head_fp8 + spec_decode_seq); spec==greedy.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_decode_top_sim test/spec_decode_top_tb.v src/spec_decode_top.v src/glm_model_fp8.v src/mtp_head_fp8.v src/spec_decode_seq.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "spec_decode_top"; $(VVP) $(BUILD_DIR)/spec_decode_top_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_decode_top"; exit 1; }
	@# spec_batched_top / spec_chain_top run the FULL model forward many times (K x
	@#   scenarios) and are minutes-long in iverilog -- moved to `make spec-slow` so the
	@#   fast `unittests` path completes (same rationale as `bcov`). They still gate on
	@#   spec==greedy; run them via `make spec-slow`.
	@# glm_fp8_soc: TOP-LEVEL SoC -- compute die + expert_cache_pf + kv_cache_pager + Flash arbiter == standalone model.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_soc_sim test/glm_fp8_soc_tb.v src/glm_fp8_soc.v src/glm_model_fp8.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_soc"; $(VVP) $(BUILD_DIR)/glm_fp8_soc_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_soc"; exit 1; }
	@# glm_fp8_soc_ms: MULTI-SEQ batched top (A2) -- host FSM prefills B seqs into a real NSEQ-window
	@# kv_cache_pager, runs ONE PE_M=B forward, commits B tokens; each row bit-exact vs per-seq PE_M=1.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_soc_ms_sim test/glm_fp8_soc_ms_tb.v src/glm_fp8_soc_ms.v src/glm_model_fp8.v src/kv_cache_pager.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_soc_ms(multi-seq)"; $(VVP) $(BUILD_DIR)/glm_fp8_soc_ms_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_soc_ms"; exit 1; }
	@# glm_fp8_soc_ms(N_STEPS=2): CONTINUOUS-BATCHING decode loop -- ONE start decodes N_STEPS tokens
	@# per sequence (feed argmax back, extent/pos grow, decode-token KV written to kv_mem and attended),
	@# each row's step-k token bit-exact vs a standalone PE_M=1 model decoding that sequence alone N steps.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_soc_ms_loop_sim test/glm_fp8_soc_ms_loop_tb.v src/glm_fp8_soc_ms.v src/glm_model_fp8.v src/kv_cache_pager.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_soc_ms(decode-loop)"; $(VVP) $(BUILD_DIR)/glm_fp8_soc_ms_loop_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_soc_ms_loop"; exit 1; }
	@# glm_fp8_system: PRODUCTION top -- compute + ddr5_xbar (multichannel) + weight_loader DMA + cache + pager == standalone.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_system_sim test/glm_fp8_system_tb.v src/glm_fp8_system.v src/weight_decomp.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_system"; $(VVP) $(BUILD_DIR)/glm_fp8_system_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_system"; exit 1; }
	@# glm_fp8_system(DECOMP=1): weight_decomp (order-0 lossless FP8) in the Flash->loader refill path (task C9).
	@# Proves the die consumes DECOMPRESSED FP8 codes from a COMPRESSED backing image, token-identical + per-beat bit-exact.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_system_decomp_sim test/glm_fp8_system_decomp_tb.v src/glm_fp8_system.v src/weight_decomp.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_system(decomp)"; $(VVP) $(BUILD_DIR)/glm_fp8_system_decomp_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_system_decomp"; exit 1; }
	@# glm_fp8_system(LOOPBACK=1) (task C8-full): the die's aw_col FP8 lanes sourced from the
	@# ddr5_xbar returned read data (loop closed) == the combinational-stub token, bit-exact.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_system_loopback_sim test/glm_fp8_system_loopback_tb.v src/glm_fp8_system.v src/weight_decomp.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_system(loopback)"; $(VVP) $(BUILD_DIR)/glm_fp8_system_loopback_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_system_loopback"; exit 1; }
	@# glm_fp8_system_cdc: 2-clock wrapper (host/USB <-> compute via cdc_async_fifo) == standalone across async clocks.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_system_cdc_sim test/glm_fp8_system_cdc_tb.v src/glm_fp8_system_cdc.v src/glm_fp8_system.v src/cdc_async_fifo.v src/reset_sync.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_system_cdc"; $(VVP) $(BUILD_DIR)/glm_fp8_system_cdc_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_system_cdc"; exit 1; }
	@# expert_cache_ctrl: MoE expert-weight HBM cache controller (tag/LRU/miss-DMA), single-package system PoC.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_cache_ctrl_sim test/expert_cache_ctrl_tb.v src/expert_cache_ctrl.v
	@printf '[%s] ' "expert_cache_ctrl"; $(VVP) $(BUILD_DIR)/expert_cache_ctrl_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: expert_cache_ctrl"; exit 1; }
	@# expert_predictor: per-(layer,expert) frequency/locality prefetch predictor w/ confidence threshold (fine-grained MoE).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_predictor_sim test/expert_predictor_tb.v src/expert_predictor.v
	@printf '[%s] ' "expert_predictor"; $(VVP) $(BUILD_DIR)/expert_predictor_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: expert_predictor"; exit 1; }
	@# spec_decode_seq: MTP speculative-decode controller (draft/verify/accept-reject; eff tok/pass = 1+alpha).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_decode_seq_sim test/spec_decode_seq_tb.v src/spec_decode_seq.v
	@printf '[%s] ' "spec_decode_seq"; $(VVP) $(BUILD_DIR)/spec_decode_seq_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_decode_seq"; exit 1; }
	@# spec_decode_seq K>1: multi-token draft (DRAFT_K=1/2/3), spec==greedy exact + eff-tok/pass vs alpha.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_decode_seq_k_sim test/spec_decode_seq_k_tb.v src/spec_decode_seq.v
	@printf '[%s] ' "spec_decode_seq(K>1)"; $(VVP) $(BUILD_DIR)/spec_decode_seq_k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_decode_seq_k"; exit 1; }
	@# kv_cache_pager: MLA latent-KV ring cache (append + DSA-gather + Flash overflow), single-module system.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/kv_cache_pager_sim test/kv_cache_pager_tb.v src/kv_cache_pager.v
	@printf '[%s] ' "kv_cache_pager"; $(VVP) $(BUILD_DIR)/kv_cache_pager_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_cache_pager"; exit 1; }
	@# kv_cache_pager(ECC=1) (task C6-full): lane-partitioned SECDED on the real ring --
	@# single-bit corrected, double-bit detected, across the ragged 768-bit / 100-bit rows.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/kv_cache_pager_ecc_sim test/kv_cache_pager_ecc_tb.v src/kv_cache_pager.v src/ecc_secded.v
	@printf '[%s] ' "kv_cache_pager(ECC)"; $(VVP) $(BUILD_DIR)/kv_cache_pager_ecc_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_cache_pager_ecc"; exit 1; }
	@# kv_cache_pager(NSEQ>1) (P1.3 per-row KV): NSEQ independent ring windows -- no
	@# cross-seq slot collision, independent per-seq eviction, flash_seq cold keying.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/kv_cache_pager_multiseq_sim test/kv_cache_pager_multiseq_tb.v src/kv_cache_pager.v
	@printf '[%s] ' "kv_cache_pager(NSEQ>1)"; $(VVP) $(BUILD_DIR)/kv_cache_pager_multiseq_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_cache_pager_multiseq"; exit 1; }
	@# ddr5_xbar: N-channel banked DDR5 read fabric (address striping -> ~Nx aggregate bandwidth).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/ddr5_xbar_sim test/ddr5_xbar_tb.v src/ddr5_xbar.v
	@printf '[%s] ' "ddr5_xbar"; $(VVP) $(BUILD_DIR)/ddr5_xbar_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: ddr5_xbar"; exit 1; }
	@# flash_xbar: N-channel banked Flash read fabric -- deep per-channel outstanding queue hides NAND latency (~QDEPTH x), banking ~N x.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/flash_xbar_sim test/flash_xbar_tb.v src/flash_xbar.v
	@printf '[%s] ' "flash_xbar"; $(VVP) $(BUILD_DIR)/flash_xbar_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: flash_xbar"; exit 1; }
	@# weight_loader: checkpoint FP8+block-scale memory image -> glm_matmul_fp8 pull stream (loader-fed == direct-fed).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/weight_loader_sim test/weight_loader_tb.v src/weight_loader.v src/glm_matmul_fp8.v src/glm_fp_pipe.v
	@printf '[%s] ' "weight_loader"; $(VVP) $(BUILD_DIR)/weight_loader_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: weight_loader"; exit 1; }
	@# boot_loader: power-up Flash->DDR5 resident-set (hot weights) DMA + ready handshake (chip must load model first).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/boot_loader_sim test/boot_loader_tb.v src/boot_loader.v
	@printf '[%s] ' "boot_loader"; $(VVP) $(BUILD_DIR)/boot_loader_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: boot_loader"; exit 1; }
	@# ecc_secded: (72,64) SECDED ECC for the DDR5/Flash path -- exhaustive single-correct + double-detect.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/ecc_secded_sim test/ecc_secded_tb.v src/ecc_secded.v
	@printf '[%s] ' "ecc_secded"; $(VVP) $(BUILD_DIR)/ecc_secded_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: ecc_secded"; exit 1; }
	@# clk_en_ctrl: work-driven clock-enable gating (die idles ~75% Flash-bound -> ~73% idle-power gated; never gates active work).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/clk_en_ctrl_sim test/clk_en_ctrl_tb.v src/clk_en_ctrl.v
	@printf '[%s] ' "clk_en_ctrl"; $(VVP) $(BUILD_DIR)/clk_en_ctrl_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: clk_en_ctrl"; exit 1; }
	@# clk_throttle: DVFS/eco frequency prescaler (die runs f/div, byte-identical peak-power cap) + its clk_en_ctrl throttle path.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/clk_throttle_sim test/clk_throttle_tb.v src/clk_throttle.v src/clk_en_ctrl.v
	@printf '[%s] ' "clk_throttle"; $(VVP) $(BUILD_DIR)/clk_throttle_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: clk_throttle"; exit 1; }
	@# swiglu_expert: SwiGLU FFN expert (gate/up/down GEMM + silu*up), dense + MoE modes.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/swiglu_expert_sim test/swiglu_expert_tb.v src/swiglu_expert.v src/glm_matmul_pipe.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "swiglu_expert"; $(VVP) $(BUILD_DIR)/swiglu_expert_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert"; exit 1; }
	@# moe_router: GEMV + sigmoid + top-K + renormalize-then-scale (GLM-5.2 MoE gating).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/moe_router_sim test/moe_router_tb.v src/moe_router.v src/glm_matmul_pipe.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router"; $(VVP) $(BUILD_DIR)/moe_router_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router"; exit 1; }
	@# glm_softmax: numerically-stable bf16 softmax (MLA attention), full-denominator sum.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_softmax_sim test/glm_softmax_tb.v src/glm_softmax.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_softmax"; $(VVP) $(BUILD_DIR)/glm_softmax_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_softmax"; exit 1; }
	@# dsa_indexer: DSA/IndexShare sparse-attention indexer (index-score + top-K + dense fallback).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/dsa_indexer_sim test/dsa_indexer_tb.v src/dsa_indexer.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "dsa_indexer"; $(VVP) $(BUILD_DIR)/dsa_indexer_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: dsa_indexer"; exit 1; }
	@# mla_attn: MLA latent attention orchestrator (low-rank Q/KV + RoPE + DSA + softmax + *V + O proj).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_sim test/mla_attn_tb.v src/mla_attn.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn"; $(VVP) $(BUILD_DIR)/mla_attn_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn"; exit 1; }
	@# sampler: temperature + top-k/top-p + softmax + multinomial(LFSR) token sampling.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/sampler_sim test/sampler_tb.v src/sampler.v src/topk_select.v src/glm_softmax.v src/glm_fp_pipe.v
	@printf '[%s] ' "sampler"; $(VVP) $(BUILD_DIR)/sampler_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: sampler"; exit 1; }
	@# glm_decoder_block: ONE full GLM-5.2 decoder layer (rmsnorm+mla_attn+residual+rmsnorm+FFN+residual).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_decoder_block_sim test/glm_decoder_block_tb.v src/glm_decoder_block.v src/rmsnorm_unit.v src/mla_attn.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/swiglu_expert.v src/moe_router.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_decoder_block"; $(VVP) $(BUILD_DIR)/glm_decoder_block_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_decoder_block"; exit 1; }
	@# glm_model: FULL GLM-5.2 forward pass (embed -> 6 layers dense/MoE -> norm -> LM head -> next-token logits).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_sim test/glm_model_tb.v src/glm_model.v src/glm_decoder_block.v src/rmsnorm_unit.v src/mla_attn.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/swiglu_expert.v src/moe_router.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_model"; $(VVP) $(BUILD_DIR)/glm_model_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model"; exit 1; }
	@# mtp_head: GLM-5.2 multi-token-prediction head (t+2 speculative; num_nextn_predict_layers=1).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mtp_head_sim test/mtp_head_tb.v src/mtp_head.v src/glm_decoder_block.v src/rmsnorm_unit.v src/mla_attn.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/swiglu_expert.v src/moe_router.v src/glm_fp_pipe.v
	@printf '[%s] ' "mtp_head"; $(VVP) $(BUILD_DIR)/mtp_head_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mtp_head"; exit 1; }
	@# AXI4-Lite MASTER DMA engine vs an AXI slave-memory BFM.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/axi_master_dma_sim test/axi_master_dma_tb.v src/axi_master_dma.v
	@printf '[%s] ' "axi_master_dma"; $(VVP) $(BUILD_DIR)/axi_master_dma_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: axi_master_dma"; exit 1; }
	@# Async CDC FIFO across two unrelated clocks (7ns vs 11ns).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/cdc_async_fifo_sim test/cdc_async_fifo_tb.v src/cdc_async_fifo.v
	@printf '[%s] ' "cdc_async_fifo"; $(VVP) $(BUILD_DIR)/cdc_async_fifo_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: cdc_async_fifo"; exit 1; }
	@# ---- P2 productization building blocks (ECC / reset-CDC / DFT-MBIST / power-ICG) ----
	@# ecc_mem_wrap: SECDED-protected synchronous RAM (encode on write, decode+correct/detect on read) -- exhaustive single-correct + double-detect.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/ecc_mem_wrap_sim test/ecc_mem_wrap_tb.v src/ecc_mem_wrap.v src/ecc_secded.v
	@printf '[%s] ' "ecc_mem_wrap"; $(VVP) $(BUILD_DIR)/ecc_mem_wrap_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: ecc_mem_wrap"; exit 1; }
	@# reset_sync: async-assert / sync-deassert reset synchronizer (CDC signoff) -- immediate assert, STAGES-edge clean deassert.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/reset_sync_sim test/reset_sync_tb.v src/reset_sync.v
	@printf '[%s] ' "reset_sync"; $(VVP) $(BUILD_DIR)/reset_sync_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: reset_sync"; exit 1; }
	@# mbist_ctrl: March C- memory BIST for a single-port SRAM (good RAM -> pass; stuck-at-0 cell -> fail + fail_addr).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mbist_ctrl_sim test/mbist_ctrl_tb.v src/mbist_ctrl.v
	@printf '[%s] ' "mbist_ctrl"; $(VVP) $(BUILD_DIR)/mbist_ctrl_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mbist_ctrl"; exit 1; }
	@# icg_cell: glitch-free integrated clock gate (low-phase enable latch + AND) -- turns clk_en into a real gated clock with no runt pulses.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/icg_cell_sim test/icg_cell_tb.v src/icg_cell.v
	@printf '[%s] ' "icg_cell"; $(VVP) $(BUILD_DIR)/icg_cell_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: icg_cell"; exit 1; }
	@# clk_gate_cluster (task C7): icg_cell + clk_en_ctrl gating a real leaf -- gated == free-running (bit-exact),
	@# idle => clock frozen, scan_enable => transparent, req=1 => enable=1 safety invariant.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/clk_gate_cluster_sim test/clk_gate_cluster_tb.v src/clk_gate_cluster.v src/icg_cell.v src/clk_en_ctrl.v src/clk_gate_leaf.v
	@printf '[%s] ' "clk_gate_cluster"; $(VVP) $(BUILD_DIR)/clk_gate_cluster_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: clk_gate_cluster"; exit 1; }
	@# kv_ecc_ring (task C6): lane-partitioned SECDED ring for wide (ragged, non-64-aligned) KV rows --
	@# single-bit corrected, double-bit detected, across the ragged final lane.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/kv_ecc_ring_sim test/kv_ecc_ring_tb.v src/kv_ecc_ring.v src/ecc_secded.v
	@printf '[%s] ' "kv_ecc_ring"; $(VVP) $(BUILD_DIR)/kv_ecc_ring_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_ecc_ring"; exit 1; }
	@# ---- Q4_K local-device track (GGUF UD-Q4_K_XL bring-up; both tracks live -- see docs/Q4K_SYSTEM_PLAN.md) ----
	@$(MAKE) --no-print-directory q4k
	@echo "unittests: all per-unit TBs passed"

# Q4_K local-device sub-gate (docs/Q4K_SYSTEM_PLAN.md 4.1): the four verified Q4_K unit TBs
# (q4k_prim / glm_matmul_q4k / swiglu_expert_q4k / moe_router_q4k), bit-exact to ggml goldens.
# Wired into `unittests` above; runnable standalone as `make q4k`. Expected: 18/160/240/40.
q4k:
	@mkdir -p $(BUILD_DIR)
	@# q4k_prim: fp16->fp32 + get_scale_min_k4 primitives (q4k.vh) vs ggml golden.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/q4k_prim_sim test/q4k_prim_tb.v
	@printf '[%s] ' "q4k_prim"; $(VVP) $(BUILD_DIR)/q4k_prim_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: q4k_prim"; exit 1; }
	@# glm_matmul_q4k: Q4_K GEMM core, bit-exact to ggml dequantize_row_q4_K.
	@python3 tools/q4k_matmul_gen.py >/dev/null            # -> build/q4k_vec.txt
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_q4k_sim test/glm_matmul_q4k_tb.v src/glm_matmul_q4k.v
	@printf '[%s] ' "glm_matmul_q4k"; $(VVP) $(BUILD_DIR)/glm_matmul_q4k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul_q4k"; exit 1; }
	@# swiglu_expert_q4k: MoE expert (gate/up/down + silu) on the Q4_K core.
	@python3 tools/swiglu_q4k_gen.py >/dev/null            # -> build/swiglu_q4k_vec.txt
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/swiglu_expert_q4k_sim test/swiglu_expert_q4k_tb.v \
	    src/swiglu_expert_q4k.v src/glm_matmul_q4k.v src/glm_act.v
	@printf '[%s] ' "swiglu_expert_q4k"; $(VVP) $(BUILD_DIR)/swiglu_expert_q4k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert_q4k"; exit 1; }
	@# moe_router_q4k: gating GEMV -> sigmoid -> top-K -> renorm on the Q4_K core.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/moe_router_q4k_sim test/moe_router_q4k_tb.v \
	    src/moe_router_q4k.v src/glm_matmul_q4k.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router_q4k"; $(VVP) $(BUILD_DIR)/moe_router_q4k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router_q4k"; exit 1; }
	@echo "q4k: all four Q4_K unit TBs passed"

# Single-package system study: regenerate the calibrated GLM-scale routing traces, then run
# the cache hit-rate + batching + prefetch sims through the real RTL (these need the generated
# tools/*.hex, so they are kept out of the self-contained `unittests`). See docs/SYSTEM_SINGLE_PACKAGE.md.
cache-study:
	@python3 tools/route_trace.py --dump >/dev/null
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_cache_batch tools/glm_cache_batch_tb.v src/expert_cache_ctrl.v
	@printf '[%s] ' "cache-batch"; $(VVP) $(BUILD_DIR)/glm_cache_batch | grep -E 'ALL [0-9]+ TESTS PASSED|BATCHING LEVER' \
	    || { echo "FAILED: cache-batch"; exit 1; }
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_cache_pf test/expert_cache_pf_tb.v src/expert_cache_pf.v src/expert_cache_ctrl.v
	@printf '[%s] ' "expert_cache_pf"; $(VVP) $(BUILD_DIR)/expert_cache_pf | grep -E 'ALL [0-9]+ TESTS PASSED|stall cut' \
	    || { echo "FAILED: expert_cache_pf"; exit 1; }
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_cache_pf_policy test/expert_cache_pf_policy_tb.v src/expert_cache_pf.v
	@printf '[%s] ' "cache-policy(LRU vs FREQ)"; $(VVP) $(BUILD_DIR)/expert_cache_pf_policy | grep -E 'ALL [0-9]+ TESTS PASSED|hit-rate|POLICY' \
	    || { echo "FAILED: expert_cache_pf_policy"; exit 1; }
	@# weight_decomp: lossless FP8-expert decompressor (Flash bytes down -> effective Flash_BW + energy). Needs the python vector.
	@python3 tools/fp8_gen.py gen scratchpad/wd_vec.txt >/dev/null
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/weight_decomp test/weight_decomp_tb.v src/weight_decomp.v
	@printf '[%s] ' "weight_decomp"; $(VVP) $(BUILD_DIR)/weight_decomp | grep -E 'ALL [0-9]+ TESTS PASSED|RATIO' \
	    || { echo "FAILED: weight_decomp"; exit 1; }
	@# weight_decomp2: context-modeled (order-1) lossless FP8 decompressor -- 1.42x on locality-bearing weights (1.13x over order-0).
	@python3 tools/fp8_ctxpack.py gen scratchpad/wd2_vec.txt >/dev/null
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/weight_decomp2 test/weight_decomp2_tb.v src/weight_decomp2.v
	@printf '[%s] ' "weight_decomp2(ctx)"; $(VVP) $(BUILD_DIR)/weight_decomp2 | grep -E 'ALL [0-9]+ TESTS PASSED|RATIO' \
	    || { echo "FAILED: weight_decomp2"; exit 1; }
	@# expert_prefetch_top: predictor-driven deep prefetch -- MEASURED no-op at real cache size (see docs/IMPROVEMENT_PLAN.md 2.3).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_prefetch_top test/expert_prefetch_top_tb.v src/expert_prefetch_top.v src/expert_predictor.v src/expert_cache_pf.v src/expert_cache_ctrl.v
	@printf '[%s] ' "expert_prefetch_top"; $(VVP) $(BUILD_DIR)/expert_prefetch_top | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: expert_prefetch_top"; exit 1; }
	@# flash_layout: offline expert->Flash-channel placement so flash_xbar's stripe spreads a token's top-8 (~39%->55% of 8x peak BW).
	@printf '[%s] ' "flash_layout"; python3 tools/flash_layout.py | grep -E 'OPTIMIZED|peakBW' | head -2 \
	    || { echo "FAILED: flash_layout"; exit 1; }
	@# batched_moe: expert-grouped batched MoE -- B tokens, union of experts fetched once each (aggregate-throughput lever).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/batched_moe test/batched_moe_tb.v src/batched_moe.v src/swiglu_expert_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "batched_moe"; $(VVP) $(BUILD_DIR)/batched_moe | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: batched_moe"; exit 1; }
	@printf '[%s] ' "batched_moe(amortize)"; python3 tools/batched_moe_amortize.py | grep -E '256|amort' | head -2 \
	    || { echo "FAILED: batched_moe_amortize"; exit 1; }
	@echo "cache-study: batching + prefetch + policy + decomp + predictor-prefetch + flash-layout + batched-moe sims passed (see docs/SYSTEM_SINGLE_PACKAGE.md)"

# P1.3c: batched_moe FULL B-COVERAGE.  batched_moe_tb.v (in cache-study) proves the
# expert-grouped batch identity at ONE width B=4; this sweeps it across widths
# {1,2,3,5,8} and routing patterns {same,distinct,random,overlap}, re-proving per
# width that batched_moe(PE_M=B) == B independent PE_M=1 expert runs, BIT-EXACT, with
# every union expert fetched exactly once.  B is a compile-time port width, so the
# parametrized TB is recompiled per (B,PATTERN) via +define.  Kept SEPARATE from
# `unittests`/`all` because each iverilog elaboration of the nested MoE datapath is
# minutes-long (esp. B=8).  See test/batched_moe_bcov_tb.v.
BCOV_SRC := src/batched_moe.v src/swiglu_expert_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/glm_fp_pipe.v
# spec-slow: the speculative-decode top harnesses. They run the full model forward
#   many times (K x accept/reject/mixed scenarios) -> minutes-long in iverilog, so they
#   are kept OUT of `unittests`; run explicitly. Both gate on spec==greedy (safety).
spec-slow:
	@mkdir -p $(BUILD_DIR)
	@# spec_batched_top: batched-verify -- K+1 draft positions in ONE PE_M=K+1 model weight-load; spec==greedy.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_batched_top_sim test/spec_batched_top_tb.v src/spec_batched_top.v src/glm_model_fp8.v src/spec_decode_seq.v src/mtp_head_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "spec_batched_top"; $(VVP) $(BUILD_DIR)/spec_batched_top_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_batched_top"; exit 1; }
	@# spec_chain_top (task B8): K-step MTP chain; committed stream == greedy rollout EXACT (spec==greedy).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_chain_top_sim test/spec_chain_top_tb.v src/spec_chain_top.v src/mtp_head_fp8.v src/glm_model_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/spec_decode_seq.v src/glm_fp_pipe.v
	@printf '[%s] ' "spec_chain_top"; $(VVP) $(BUILD_DIR)/spec_chain_top_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_chain_top"; exit 1; }
	@echo "spec-slow: spec_batched_top + spec_chain_top passed (spec==greedy)"

bcov:
	@mkdir -p $(BUILD_DIR)
	@set -e; for bp in 1:2:11 2:1:22 3:3:33 5:0:55 8:3:44; do \
	   B=$${bp%%:*}; rest=$${bp#*:}; P=$${rest%%:*}; S=$${rest##*:}; \
	   $(IVERILOG) $(IFLAGS) -DBCOV_B=$$B -DBCOV_PAT=$$P -DBCOV_SEED=$$S \
	     -o $(BUILD_DIR)/bcov_$$B test/batched_moe_bcov_tb.v $(BCOV_SRC); \
	   printf '[batched_moe B=%s] ' "$$B"; \
	   $(VVP) $(BUILD_DIR)/bcov_$$B | grep -E 'ALL [0-9]+ TESTS PASSED' \
	     || { echo "FAILED: batched_moe B=$$B"; exit 1; }; \
	 done
	@echo "bcov: batched_moe B-coverage passed for B in {1,2,3,5,8} (batched == per-row, bit-exact)"

# Formal (bounded model checking) of the memory-system controllers via yosys write_smt2 +
# yosys-smtbmc -s z3.  Each harness test/formal/<dut>_fv.v instantiates the committed controller
# read-only and asserts safety properties.  The mandatory `async2sync; chformal -lower` lowers
# yosys $check cells so the asserts are NOT silently dropped (a vacuous-pass trap); the assert
# count > 0 guard re-checks non-vacuity per model.  See docs/FORMAL.md.  Bounds kept modest for a
# routine run; deeper bounds (e.g. expert_cache_pf K=55) are in docs/FORMAL.md.
FV_DIR := scratchpad
define run_bmc   # $(1)=dut name  $(2)=extra read deps  $(3)=extra yosys (e.g. chparam)  $(4)=bound K
	@yosys -p "read_verilog -sv -formal -I src src/$(1).v $(2) test/formal/$(1)_fv.v; $(3) \
	          prep -top $(1)_fv -flatten; memory_map; async2sync; chformal -lower; \
	          write_smt2 -wires $(FV_DIR)/$(1)_fv.smt2" > $(FV_DIR)/$(1)_fv_build.log 2>&1 \
	    || { echo "FAILED(build): $(1)"; cat $(FV_DIR)/$(1)_fv_build.log; exit 1; }
	@test `grep -ic assert $(FV_DIR)/$(1)_fv.smt2` -gt 0 \
	    || { echo "FAILED(vacuous: 0 assertions in smt2): $(1)"; exit 1; }
	@yosys-smtbmc -s z3 -t $(4) $(FV_DIR)/$(1)_fv.smt2 > $(FV_DIR)/$(1)_fv_bmc.log 2>&1 \
	    && printf '[formal %-16s] PASSED  K=%s  (%s asserts)\n' "$(1)" "$(4)" "`grep -ic assert $(FV_DIR)/$(1)_fv.smt2`" \
	    || { echo "FAILED(BMC counterexample): $(1)"; tail -20 $(FV_DIR)/$(1)_fv_bmc.log; exit 1; }
endef

# Named-harness BMC (harness basename != dut name, and a custom read-source list).
# Used for kv_cache_pager_ecc_fv (proves the ECC=1 lane-SECDED ring datapath).
define run_bmc_named   # $(1)=harness basename  $(2)=read sources  $(3)=bound K
	@yosys -p "read_verilog -sv -formal -I src $(2) test/formal/$(1).v; \
	          prep -top $(1) -flatten; memory_map; async2sync; chformal -lower; \
	          write_smt2 -wires $(FV_DIR)/$(1).smt2" > $(FV_DIR)/$(1)_build.log 2>&1 \
	    || { echo "FAILED(build): $(1)"; cat $(FV_DIR)/$(1)_build.log; exit 1; }
	@test `grep -ic assert $(FV_DIR)/$(1).smt2` -gt 0 \
	    || { echo "FAILED(vacuous: 0 assertions): $(1)"; exit 1; }
	@yosys-smtbmc -s z3 -t $(3) $(FV_DIR)/$(1).smt2 > $(FV_DIR)/$(1)_bmc.log 2>&1 \
	    && printf '[formal %-20s] PASSED  K=%s  (%s asserts)\n' "$(1)" "$(3)" "`grep -ic assert $(FV_DIR)/$(1).smt2`" \
	    || { echo "FAILED(BMC counterexample): $(1)"; tail -20 $(FV_DIR)/$(1)_bmc.log; exit 1; }
endef

formal:
	@mkdir -p $(FV_DIR)
	$(call run_bmc,ddr5_xbar,,,12)
	$(call run_bmc,flash_xbar,,,12)
	$(call run_bmc,boot_loader,,,16)
	$(call run_bmc,clk_throttle,,,16)
	$(call run_bmc,spec_decode_seq,,,20)
	$(call run_bmc,kv_cache_pager,,,16)
	$(call run_bmc,expert_cache_pf,src/expert_cache_ctrl.v,chparam -set PF_ENABLE 0 expert_cache_pf_fv;,20)
	@# kv_cache_pager ECC=1 datapath (task C6-full followup): the lane-SECDED ring preserves
	@# encode-decode identity + window/in-bounds + no-false-alarm (single-lane; see harness header).
	$(call run_bmc_named,kv_cache_pager_ecc_fv,src/kv_cache_pager.v src/ecc_secded.v,12)
	@echo "formal: 5 controllers + the ECC=1 pager datapath BMC-proven (no counterexample); see docs/FORMAL.md"

# UNBOUNDED proof via temporal k-INDUCTION (yosys-smtbmc -i): base case + induction
# step => the asserts hold on ALL reachable states, not just the first K cycles.
# The step needs the design's reachable state space pinned; harnesses add
# STRENGTHENING INVARIANT asserts (over the DUT's primary I/O + harness shadow
# regs -- this yosys build has no internal observability) until the step closes.
define run_kind  # $(1)=dut name  $(2)=ind-harness basename  $(3)=K  $(4)=extra yosys (e.g. connect-bind)
	@yosys -p "read_verilog -sv -formal -I src src/$(1).v test/formal/$(2).v; \
	          prep -top $(2) -flatten; $(4) memory_map; async2sync; chformal -lower; \
	          write_smt2 -wires $(FV_DIR)/$(2).smt2" > $(FV_DIR)/$(2)_build.log 2>&1 \
	    || { echo "FAILED(build): $(2)"; cat $(FV_DIR)/$(2)_build.log; exit 1; }
	@test `grep -ic assert $(FV_DIR)/$(2).smt2` -gt 0 \
	    || { echo "FAILED(vacuous: 0 assertions): $(2)"; exit 1; }
	@yosys-smtbmc -s z3 -i -t $(3) $(FV_DIR)/$(2).smt2 > $(FV_DIR)/$(2)_kind.log 2>&1 \
	    && printf '[k-induction %-20s] PROVEN UNBOUNDED  K=%s  (%s asserts)\n' "$(2)" "$(3)" "`grep -ic assert $(FV_DIR)/$(2).smt2`" \
	    || { echo "FAILED(induction step): $(2)"; tail -20 $(FV_DIR)/$(2)_kind.log; exit 1; }
endef

# flash_xbar response-FIFO / outstanding proof needs the DUT's OWN per-channel counters
# (u_dut.outst[c], u_dut.cnt[c]) in the inductive hypothesis.  yosys 0.66 cannot reference
# them from Verilog (no hierarchical refs), so the harness declares `(* keep *)` UNDRIVEN
# probe wires and we wire them to the flattened DUT registers post-flatten with `connect`.
# The TRAILING SPACE before each `;` is load-bearing: it terminates the escaped bracketed
# id \u_dut.outst[0] (otherwise [0] is parsed as a bit-select of a non-existent wire).
FLASH_IND_CONN := connect -set \dut_outst0 \u_dut.outst[0] ; connect -set \dut_outst1 \u_dut.outst[1] ; connect -set \dut_cnt0 \u_dut.cnt[0] ; connect -set \dut_cnt1 \u_dut.cnt[1] ;
# ddr5_xbar response-FIFO proof (task C5) uses the same connect-bind trick to reach
# the DUT's per-channel response-FIFO occupancy counters cnt[0..N_CH-1] (N_CH=2 slice).
DDR5_IND_CONN := connect -set \dut_cnt0 \u_dut.cnt[0] ; connect -set \dut_cnt1 \u_dut.cnt[1] ;
formal-ind:
	@mkdir -p $(FV_DIR)
	$(call run_kind,boot_loader,boot_loader_ind_fv,8)
	$(call run_kind,kv_cache_pager,kv_cache_pager_ind_fv,16)
	$(call run_kind,spec_decode_seq,spec_decode_seq_ind_fv,2)
	$(call run_kind,ddr5_xbar,ddr5_xbar_ind_fv,12,$(DDR5_IND_CONN))
	$(call run_kind,flash_xbar,flash_xbar_ind_fv,3,$(FLASH_IND_CONN))
	@echo "formal-ind: boot_loader done-gate proven UNBOUNDED; kv_cache_pager append/gather in-bounds + window invariants proven UNBOUNDED; spec_decode_seq token-accounting equality + per-cycle modular increment bounds + step-form (non-decreasing-except-wrap) monotonicity proven UNBOUNDED (k-induction K=2); ddr5_xbar request-path routing safety (exclusive one-hot routing / banked-channel selection / ready coherence / payload integrity) proven UNBOUNDED + response-FIFO no-overflow/underflow (cnt[c]<=RESP_QD conservation form, inflight<=CAP) proven UNBOUNDED via connect-bound internal cnt[] counters (task C5) -- tag-issued stays BOUNDED (FIFO-content data-invariant; the FIFO is a 2-D memory cell, not connect-bindable); strict unsigned monotonicity stays BOUNDED (32-bit counter wrap); flash_xbar per-channel-queue no-overflow (cnt[c]<=QDEPTH) + outstanding<=N_CH*QDEPTH (P3) + inflight<=outstanding (P1a/P1b) proven UNBOUNDED via connect-bound internal counters (k-induction K=3) -- tag-issued (P2) stays BOUNDED (needs FIFO-content data-invariant; FIFO is a 2-D memory cell, not connect-bindable) -- see docs/FORMAL.md"

# Real-checkpoint bit-accuracy: prove glm_matmul_fp8 computes the GLM-5.2-FP8 FP8 contract exactly
# as the real inference engine (argmax-preserving vs fp32-accumulate + float64 ground truth), and
# that the HF safetensors -> RTL bridge round-trips.  See docs/BIT_ACCURACY.md (+ the full-model GPU procedure).
bitacc:
	@python3 tools/glm_fp8_ref.py | grep -qE 'SELFTEST PASS' || { echo "FAILED: glm_fp8_ref"; exit 1; }
	@printf '[%s] ' "glm_fp8_ref"; echo "SELFTEST PASS"
	@python3 tools/ckpt_pack.py | grep -qE 'SELFTEST PASS' || { echo "FAILED: ckpt_pack"; exit 1; }
	@printf '[%s] ' "ckpt_pack(HF safetensors->RTL)"; echo "round-trip PASS"
	@python3 test/bit_accuracy_gen.py >/dev/null
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/bit_accuracy test/bit_accuracy_tb.v src/glm_matmul_fp8.v src/glm_fp_pipe.v
	@printf '[%s] ' "bit_accuracy(RTL==real-contract)"; $(VVP) $(BUILD_DIR)/bit_accuracy | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: bit_accuracy"; exit 1; }
	@python3 test/bit_accuracy_check.py | grep -E 'ARGMAX-PRESERVED' || { echo "FAILED: bit_accuracy_check"; exit 1; }
	@# glm_fp8_contract: vectorized (torch/numpy/pure-python) our-FP8 contract, bit-exact to glm_fp8_ref -- the GPU-scale model for the Modal P1.1 real-checkpoint gate.
	@python3 tools/glm_fp8_contract.py | grep -qE 'ALL [0-9]+ TESTS PASSED' || { echo "FAILED: glm_fp8_contract"; exit 1; }
	@printf '[%s] ' "glm_fp8_contract"; echo "== glm_fp8_ref (bit-exact, vectorized)"
	@python3 test/modal_validate_test.py | grep -qE 'ALL [0-9]+ TESTS PASSED' || { echo "FAILED: modal_validate compare logic"; exit 1; }
	@printf '[%s] ' "modal_validate(compare logic)"; echo "PASS"
	@echo "bitacc: glm_matmul_fp8 == real GLM-5.2-FP8 FP8 contract, argmax preserved; Modal P1.1 app ready (docs/MODAL_VALIDATE.md)"

# Verilator lint of the GLM product top (diagnostic, NOT part of `all`): the GLM
# datapath is not yet -Wall clean (width/pin warnings under review), so this is
# informational.  The fidelity gate is `unittests` + `synth-glm` + `formal`.
lint:
	$(VERILATOR) --lint-only -Wall -Isrc --top-module glm_fp8_system_cdc $(GLM_CDC_SRCS)

# Host software scaffold (D2): OpenAI-compatible server + device protocol (stdlib).
host-test:
	@printf '[host] '; python3 host/test_aipu.py | tail -1

# ---- Whole-chip structural gate for the GLM-5.2-FP8 product top (task C1) ----
# This gate elaborates the
# ENTIRE product hierarchy -- the 2-clock chip top `glm_fp8_system_cdc` and every
# GLM compute + memory-system + CDC leaf beneath it -- and runs `check -assert`,
# which FAILS (non-zero exit) on any unresolved hierarchy, combinational loop,
# multiple driver, or inferred latch anywhere in the GLM datapath.  This is the
# first structural sign-off of the actual product top (nothing gated it before).
# `stat` prints the flattened gate-level cell count for the whole chip.
GLM_CDC_SRCS := src/glm_fp8_system_cdc.v src/glm_fp8_system.v src/cdc_async_fifo.v \
	src/reset_sync.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v \
	src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v \
	src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v \
	src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v \
	src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
	src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v \
	src/glm_fp_pipe.v

synth-glm:
	$(YOSYS) -q -p "read_verilog -sv -I src $(GLM_CDC_SRCS); \
	                hierarchy -top glm_fp8_system_cdc -check; proc; opt; check -assert; stat"

# ---- Whole-chip structural gate for the GLM-5.2 Q4_K product top (plan §1.4) ----
# SIBLING of `synth-glm`: the same structural elaboration + `check -assert` sign-off,
# but on the Q4_K two-clock chip top `glm_q4k_system_cdc` and the *_q4k.v hierarchy
# beneath it (the compute die glm_model_q4k + weight_loader_q4k, with the byte-
# agnostic memory-system / CDC leaves shared unchanged).  Fails (non-zero exit) on
# any unresolved hierarchy, combinational loop, multiple driver, or inferred latch
# anywhere in the Q4_K datapath.  `synth-glm` (FP8) is left UNTOUCHED.
GLM_Q4K_CDC_SRCS := src/glm_q4k_system_cdc.v src/glm_q4k_system.v src/cdc_async_fifo.v \
	src/reset_sync.v src/glm_model_q4k.v src/ddr5_xbar.v src/weight_loader_q4k.v \
	src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v \
	src/glm_decoder_block_q4k.v src/mla_attn_q4k.v src/swiglu_expert_q4k.v \
	src/moe_router_q4k.v src/glm_matmul_q4k.v src/rmsnorm_unit.v \
	src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
	src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v \
	src/glm_fp_pipe.v

synth-glm-q4k:
	$(YOSYS) -q -p "read_verilog -sv -I src $(GLM_Q4K_CDC_SRCS); \
	                hierarchy -top glm_q4k_system_cdc -check; proc; opt; check -assert; stat"

# ---- COMPACT SYNTH VARIANT (FPGA miniaturization; Tang Mega 138K / GW5AT-138) --
# Same structural elaboration+sign-off as `synth-glm`, but overrides the five
# RESULT-INVARIANT resource parameters (capacity / parallelism / bandwidth -- NOT
# the math) to their compact values so the elaborated hierarchy is SMALLER (half
# the matmul PE array, half the DDR5 crossbar, smaller KV ring / expert FIFO /
# expert cache).  The decoded TOKEN is BYTE-IDENTICAL to the default config --
# proven functionally by `make sim-glm-compact` (token == the committed run).
#   PE_N 4->2  DDR_NCH 4->2  KV_RESIDENT 16->8(>=S_MAX)  EFIFO_DEPTH 16->8  CACHE_SLOTS 4->2
# NOTE: yosys 0.66 cannot map the FP8 datapaths through ABC, so this target does
# NOT emit a LUT count; it elaborates + `check -assert`s the compact hierarchy and
# `stat`s the flattened cell count.  The area reduction is BY CONSTRUCTION (fewer
# PE columns / channels / ring+FIFO+cache entries); the byte-identical token is the
# verified invariant (see docs/OPERATION_FLOW.md "Compact config").
synth-glm-compact:
	$(YOSYS) -q -p "read_verilog -sv -I src $(GLM_CDC_SRCS); \
	                hierarchy -top glm_fp8_system_cdc -check \
	                  -chparam PE_N 2 -chparam DDR_NCH 2 -chparam KV_RESIDENT 8 \
	                  -chparam EFIFO_DEPTH 8 -chparam CACHE_SLOTS 2; \
	                proc; opt; check -assert; stat"

# ---- COMPACT FUNCTIONAL VERIFY (token byte-identical at the compact config) ----
# Runs the system TB (test/glm_fp8_system_tb.v) TWICE and asserts the compact
# config decodes the SAME token as the committed default: (1) default -P (committed
# slice) and (2) the compact -P overrides.  Both must print ALL 3 TESTS PASSED and
# the per-test `tok=` lines must match.  This is the functional proof behind
# `synth-glm-compact`.
sim-glm-compact:
	@mkdir -p $(BUILD_DIR)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_sys_committed_sim test/glm_fp8_system_tb.v \
	    src/glm_fp8_system.v src/weight_decomp.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v \
	    src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v \
	    src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v \
	    src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v \
	    src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@$(IVERILOG) $(IFLAGS) \
	    -Pglm_fp8_system_tb.PE_N=2 -Pglm_fp8_system_tb.DDR_NCH=2 -Pglm_fp8_system_tb.KV_RESIDENT=8 \
	    -Pglm_fp8_system_tb.EFIFO_DEPTH=8 -Pglm_fp8_system_tb.CACHE_SLOTS=2 \
	    -o $(BUILD_DIR)/glm_sys_compact_sim test/glm_fp8_system_tb.v \
	    src/glm_fp8_system.v src/weight_decomp.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v \
	    src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v \
	    src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v \
	    src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v \
	    src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@$(VVP) $(BUILD_DIR)/glm_sys_committed_sim | grep -E 'PASS\[|ALL [0-9]+ TESTS PASSED' > $(BUILD_DIR)/glm_committed.tok \
	    || { echo "FAILED: committed config did not pass"; exit 1; }
	@$(VVP) $(BUILD_DIR)/glm_sys_compact_sim   | grep -E 'PASS\[|ALL [0-9]+ TESTS PASSED' > $(BUILD_DIR)/glm_compact.tok \
	    || { echo "FAILED: compact config did not pass"; exit 1; }
	@echo "==== COMMITTED (PE_N=4 DDR_NCH=4 KV_RESIDENT=16 EFIFO_DEPTH=16 CACHE_SLOTS=2) ===="; cat $(BUILD_DIR)/glm_committed.tok
	@echo "==== COMPACT   (PE_N=2 DDR_NCH=2 KV_RESIDENT=8  EFIFO_DEPTH=8  CACHE_SLOTS=2) ===="; cat $(BUILD_DIR)/glm_compact.tok
	@grep -oE 'tok=[0-9]+' $(BUILD_DIR)/glm_committed.tok > $(BUILD_DIR)/glm_committed.toks
	@grep -oE 'tok=[0-9]+' $(BUILD_DIR)/glm_compact.tok   > $(BUILD_DIR)/glm_compact.toks
	@diff -q $(BUILD_DIR)/glm_committed.toks $(BUILD_DIR)/glm_compact.toks >/dev/null \
	    && echo "PASS: compact token stream is BYTE-IDENTICAL to the committed config" \
	    || { echo "FAILED: compact token differs from committed"; exit 1; }

# ---- CDC structural sign-off for the 2-clock product top (task C8) ----------
# Asserts every host_clk<->core_clk crossing in glm_fp8_system_cdc flows through a
# recognized synchronizer (async FIFO / 2-FF / reset_sync) and that no raw
# multi-bit register is captured across the boundary.  A targeted structural
# checker (not a commercial CDC tool -- see tools/cdc_check.py header for limits);
# constraints/glm_fp8_system_cdc.sdc carries the matching false-path/async SDC.
cdc:
	@python3 tools/cdc_check.py src/glm_fp8_system_cdc.v \
	    || { echo "FAILED: cdc structural sign-off"; exit 1; }


# ---- Structural code coverage (Verilator --coverage-line/-toggle) ---------
# Verilate + run the SUBSET of behavioral TBs that build+run cleanly under
# Verilator 5.x (--binary), each with --coverage-line --coverage-toggle, then
# report per-module line/toggle/branch coverage of the module's OWN source and
# a merged design-source summary.  Artifacts land in build/cov/ (gitignored);
# the merged database is build/cov/merged.dat.  This is STRUCTURAL coverage of
# the committed slice config -- the bit-fidelity proof remains the byte-
# identical iverilog suite (`make unittests`).  Scope + the TBs that don't
# verilate (and why) are documented in docs/COVERAGE.md.
coverage:
	@command -v $(VERILATOR) >/dev/null 2>&1 || { echo "coverage: needs verilator (5.x)"; exit 1; }
	@mkdir -p $(BUILD_DIR)/cov
	@VERILATOR="$(VERILATOR)" bash tools/cov_run.sh

clean:
	rm -rf $(BUILD_DIR)
	rm -f *.vcd
