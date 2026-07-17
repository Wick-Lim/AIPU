`timescale 1ns/1ps
//============================================================================
// glm_q4k_intra_batch_verify_tb.v
//   5b-sys SYSTEM-LEVEL position-accurate BATCHED-VERIFY oracle.
//----------------------------------------------------------------------------
// WHAT IS PROVEN (self-checking; $fatal on any mismatch):
//
//   ONE PE_M=K+1 batched pass through glm_q4k_system
//     (SELF_KV=1, INTRA_CAUSAL=1, PER_ROW_POS=1)
//   over K+1 tokens at consecutive positions 0..K -- row r decodes position r and
//   attends positions 0..r-1, which are the CURRENT-token keys of the earlier rows
//   0..r-1 computed IN THAT SAME PASS (intra-batch causal attention, through the
//   whole multi-layer model) -- reproduces, BIT-EXACT on the FULL per-row logit vector AND
//   the per-row argmax, the SERIAL single-row (PE_M=1) glm_q4k_system decode chain
//   of the same K+1 tokens:
//
//     serial decode r drives glm_q4k_system(PE_M=1, SELF_KV=1) with token r at
//     position r, s_len=r.  SELF_KV=1 makes the pager APPEND each decode's committed
//     latent (per layer) and GATHER it back for the next decode, so decode r reads,
//     for each of the 6 layers, the latents decodes 0..r-1 wrote at positions 0..r-1
//     -- the die-internal KV write-back.  This is the exact serial reference the
//     batched intra-causal pass must equal.
//
//   The reference is the SAME production module at PE_M=1 (INTRA_CAUSAL=0) -- NOT a
//   hand-rolled parallel model that could share the batched path's bug.  The batch
//   MUST reproduce the serial chain because a batched intra key i (a virtual cache
//   key at causal index s_reg+i, latent ckv_cur[i]/krope_cur[i]) is BYTE-IDENTICAL
//   to the pager row the serial chain wrote at position i, and both flow through the
//   identical RMSNorm/W_uk/W_uv/DSA/RoPE path (mla-intra proves this at the leaf;
//   self-kv-l6 proves the serial pager round-trip; THIS gate proves they COMPOSE in
//   the full system: model -> decoder -> mla, per (layer,position), for every row).
//
//   Positions start at 0 (empty shared prefix), so EVERY causal key a row attends is
//   an INTRA-BATCH key -- the batched path is maximally exercised, and the shared KV
//   prefix is 0 (the batched pass gathers nothing; all K+1-row context is intra).
//
//   K = 1, 2, 3  (separate PE_M=2/3/4 batched instances -- literally "one PE_M=K+1
//   batched pass" per K).  Prints "ALL <N> TESTS PASSED"; $fatal on any mismatch/X.
//
// INJECTION (compile-time, NEVER set in the normal build):
//   -D INJECT_SHARED_POS : drive the batched instance's pos_vec with ONE shared
//     position for every row (all rows rope at position 0) instead of the per-row
//     0..K.  Intra-batch causal indexing is then wrong (row j's query + intra keys
//     rope at pos 0, not pos j) -> the batched rows 1..K DIVERGE from the serial
//     chain and this oracle MUST FAIL.  (Reverting to per-row pos_vec re-passes.)
//============================================================================
module glm_q4k_intra_batch_verify_tb;

    // ---- L=3 slice (2 dense + 1 MoE): multi-layer, BOTH layer types, cross-layer
    //   intra-key propagation -- a runnable system sim.  (The identical harness at
    //   L=6/N_DENSE=3, the full self-kv-l6 slice depth, ALSO prints ALL 9 TESTS
    //   PASSED -- the composition is layer-count independent; self-kv-l6 separately
    //   proves the pager's per-(layer,pos) keying at L=6.) ----
    localparam integer MODEL_DIM = 16;
    localparam integer L         = 3;
    localparam integer N_DENSE   = 2;
    localparam integer VOCAB     = 16;
    localparam integer H_HEADS   = 2;
    localparam integer NOPE      = 4;
    localparam integer ROPE      = 4;
    localparam integer V_DIM     = 4;
    localparam integer Q_LORA    = 8;
    localparam integer KV_LORA   = 8;
    localparam integer S_MAX     = 8;
    localparam integer TOPK_ATTN = 8;      // dense (max extent K=3 < 8): DSA a no-op; INTRA is the lever
    localparam integer THETA     = 8000000;
    localparam integer PE_N      = 2;
    localparam integer POSW      = 20;
    localparam integer N_EXPERT  = 4;
    localparam integer TOPK      = 2;
    localparam integer INTER_MOE = 16;
    localparam integer INTER_DENSE = 32;
    localparam [31:0] RSCALE     = 32'h40200000;
    localparam integer TN        = 4;
    localparam integer BLK       = 128;
    localparam integer LM_TN     = 4;
    // batched attention union-slot scratch: min(PE_M*TOPK,S_MAX) bound -> S_MAX covers
    //   every K here (leaf asserts SWIN >= that bound for INTRA_CAUSAL divergent rows).
    localparam integer SWINB     = S_MAX;
    // ---- memory system ----
    localparam integer CACHE_SLOTS = 4;
    localparam integer FLASH_LAT   = 8;
    localparam integer KV_CTX      = 1024;
    localparam integer KV_RESIDENT = 16;
    localparam integer EFIFO_DEPTH = 16;
    // ---- DDR5 fabric + loader ----
    localparam integer DDR_NCH     = 4;
    localparam integer DDR_ADDR_W  = 32;
    localparam integer DDR_DATA_W  = 256;
    localparam integer DDR_TAG_W   = 8;
    localparam integer DDR_ROW_LAT = 10;
    localparam integer DDR_RESP_QD = 4;
    localparam integer WL_KMAX     = 256;
    localparam integer WL_ADDR_W   = 24;
    localparam integer LOADER_KLEN = MODEL_DIM;

    // ---- derived (mirror glm_q4k_system / glm_model_q4k) ----
    localparam integer QK_DIM = NOPE + ROPE;
    localparam integer IDXW   = (S_MAX<=1)?1:$clog2(S_MAX);
    localparam integer HQK    = H_HEADS*QK_DIM;
    localparam integer HNOPE  = H_HEADS*NOPE;
    localparam integer HV     = H_HEADS*V_DIM;
    localparam integer EIDXW  = (N_EXPERT<=1)?1:$clog2(N_EXPERT);
    localparam integer A_KMAX = (MODEL_DIM>Q_LORA)?
                       ((MODEL_DIM>KV_LORA)?((MODEL_DIM>HV)?MODEL_DIM:HV):((KV_LORA>HV)?KV_LORA:HV))
                     :((Q_LORA>KV_LORA)?((Q_LORA>HV)?Q_LORA:HV):((KV_LORA>HV)?KV_LORA:HV));
    localparam integer A_OMAX = (HQK>MODEL_DIM)?
                       ((HQK>HNOPE)?((HQK>HV)?HQK:HV):((HNOPE>HV)?HNOPE:HV))
                     :((MODEL_DIM>HNOPE)?((MODEL_DIM>HV)?MODEL_DIM:HV):((HNOPE>HV)?HNOPE:HV));
    localparam integer A_NGMAX = (A_OMAX+PE_N-1)/PE_N;
    localparam integer A_GRPW  = (A_NGMAX<=1)?1:$clog2(A_NGMAX);
    localparam integer A_KCW   = (A_KMAX <=1)?1:$clog2(A_KMAX);
    localparam integer FF_KMAX_D = (INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM;
    localparam integer FF_KMAX_M = (INTER_MOE >MODEL_DIM)?INTER_MOE :MODEL_DIM;
    localparam integer FF_GWD = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN+1);
    localparam integer FF_KWD = $clog2(FF_KMAX_D+1);
    localparam integer R_KW   = $clog2(FF_KMAX_M+1);
    localparam integer A_NSB    = (A_KMAX   +255)/256;
    localparam integer FF_NSB_D = (FF_KMAX_D+255)/256;
    localparam integer R_NSB    = (FF_KMAX_M+255)/256;
    localparam integer LAYW   = (L<=1)?1:$clog2(L);
    localparam integer TOKW   = (VOCAB<=1)?1:$clog2(VOCAB);
    localparam integer DIMW   = (MODEL_DIM<=1)?1:$clog2(MODEL_DIM);
    localparam integer NVTILE = VOCAB/LM_TN;
    localparam integer VTW    = (NVTILE<=1)?1:$clog2(NVTILE);
    localparam integer ROW_BITS = (KV_LORA+ROPE)*16;
    localparam integer KVPOSW   = (KV_CTX<=1)?1:$clog2(KV_CTX);
    localparam integer CSLOTW   = (CACHE_SLOTS<=1)?1:$clog2(CACHE_SLOTS);
    localparam integer WL_PE_N  = PE_N;
    localparam integer WL_DATA_W= 256;
    localparam integer CH_SEL_W = (DDR_NCH<=1)?1:$clog2(DDR_NCH);
    localparam integer KMAX_TB  = 3;       // largest batch tested (K=3 -> PE_M=4 rows)

    reg clk; initial clk = 1'b0; always #5 clk = ~clk;
    reg rst;
    integer test_count, errors;

    //========================================================================
    // deterministic stimulus generators (identical to self-kv-l6 TB -> the SAME
    //   weight image for the batched and serial instances)
    //========================================================================
    function automatic integer f_h; input integer seed; begin
        f_h = (seed*2654435761)^(seed<<13)^(seed*40503);
    end endfunction
    function automatic [15:0] gen_bf16; input integer seed;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h = f_h(seed); s = h[3]; e = 8'd124 + {6'b0,h[5:4]}; m = h[12:6];
        gen_bf16 = {s,e,m};
    end endfunction
    function automatic [15:0] gen_fp16; input integer seed;
        reg [4:0] e; reg [9:0] m; integer h; begin
        h = f_h(seed); e = 5'd12 + {4'b0,h[4]}; m = h[14:5];
        gen_fp16 = {1'b0,e,m};
    end endfunction
    function automatic [3:0] gen_q4; input integer seed; integer h; begin
        h = f_h(seed); gen_q4 = h[11:8];
    end endfunction
    function automatic [31:0] gen_s32; input integer seed; begin
        gen_s32 = f_h(seed*97 + 5);
    end endfunction
    function automatic [3:0] f_awq; input integer ly; input integer sel;
        input integer fo; input integer kk; begin
        f_awq = gen_q4(ly*7919 + sel*104729 + fo*611953 + kk*13 + 101);
    end endfunction
    function automatic [15:0] f_awd; input integer ly; input integer sel; input integer fo; begin
        f_awd = gen_fp16(ly*7919 + sel*104729 + fo*611953 + 211);
    end endfunction
    function automatic [15:0] f_awdm; input integer ly; input integer sel; input integer fo; begin
        f_awdm = gen_fp16(ly*7919 + sel*104729 + fo*611953 + 307);
    end endfunction
    function automatic [3:0] f_rwq; input integer ly; input integer e; input integer kk; begin
        f_rwq = gen_q4(ly*7919 + e*350377 + kk*13 + 401);
    end endfunction
    function automatic [3:0] f_fwq; input integer ly; input integer sel; input integer shr;
        input integer eidx; input integer fo; input integer kk; begin
        f_fwq = gen_q4(ly*7919 + sel*104729 + shr*15485863 + eidx*350377 + fo*611953 + kk*13 + 503);
    end endfunction

    //========================================================================
    // Per-instance I/O declaration (PE_M-INDEPENDENT ports only -- the weight/KV/
    //   Flash/DDR5/loader interface is a SHARED single-fetch stream, width identical
    //   at every PE_M).  Prefix P (with trailing underscore) => P``em_val etc.
    //========================================================================
    `define SYSIO(P) \
        wire P``busy, P``done; wire [TOKW-1:0] P``next_tok; wire P``tok_valid; \
        wire P``em_req; wire [TOKW-1:0] P``em_tok; wire [DIMW-1:0] P``em_idx; reg [15:0] P``em_val; \
        wire [LAYW-1:0] P``db_layer; wire P``idx_fresh; wire [LAYW-1:0] P``idx_win; \
        wire P``gn_req, P``gn_which; wire [DIMW-1:0] P``gn_idx; reg [15:0] P``gn_val; \
        wire P``aw_req; wire [3:0] P``aw_sel; wire [A_GRPW-1:0] P``aw_grp; wire [A_KCW-1:0] P``aw_k; \
        reg [PE_N*4-1:0] P``aw_q; reg [16*PE_N*A_NSB-1:0] P``aw_d, P``aw_dmin; reg [96*PE_N*A_NSB-1:0] P``aw_scales; \
        wire P``rw_req; wire [R_KW-1:0] P``rw_k; \
        reg [4*N_EXPERT-1:0] P``rw_q; reg [16*N_EXPERT*R_NSB-1:0] P``rw_d, P``rw_dmin; reg [96*N_EXPERT*R_NSB-1:0] P``rw_scales; \
        wire P``fw_req; wire [1:0] P``fw_sel; wire [FF_GWD-1:0] P``fw_grp; wire [FF_KWD-1:0] P``fw_k; \
        wire P``fw_shared; wire [EIDXW-1:0] P``fw_eidx; \
        reg [4*TN-1:0] P``fw_q, P``fw_q_up; \
        reg [16*TN*FF_NSB_D-1:0] P``fw_d_g, P``fw_dmin_g, P``fw_d_u, P``fw_dmin_u; \
        reg [96*TN*FF_NSB_D-1:0] P``fw_scales_g, P``fw_scales_u; \
        wire P``fn_req; wire [DIMW-1:0] P``fn_idx; reg [15:0] P``fn_val; \
        wire P``lw_req; wire [VTW-1:0] P``lw_vtile; wire [DIMW-1:0] P``lw_k; reg [LM_TN*16-1:0] P``lw_col; \
        wire P``kc_req; wire [IDXW-1:0] P``kc_idx; \
        wire [KVPOSW-1:0] P``kv_row_sel; \
        wire P``flash_req, P``flash_is_expert; wire [EIDXW-1:0] P``flash_expert_id; \
        wire [KVPOSW-1:0] P``flash_row_idx; reg [ROW_BITS-1:0] P``flash_row; reg P``flash_done; \
        wire P``mdl_busy; \
        wire [DDR_NCH-1:0] P``mem_req_valid; wire [DDR_NCH-1:0] P``mem_req_ready; \
        wire [DDR_NCH*DDR_ADDR_W-1:0] P``mem_req_addr; wire [DDR_NCH*DDR_TAG_W-1:0] P``mem_req_tag; \
        reg [DDR_NCH-1:0] P``mem_resp_valid; wire [DDR_NCH-1:0] P``mem_resp_ready; \
        reg [DDR_NCH*DDR_DATA_W-1:0] P``mem_resp_data; reg [DDR_NCH*DDR_TAG_W-1:0] P``mem_resp_tag; \
        wire P``wl_mem_en; wire [WL_ADDR_W-1:0] P``wl_mem_addr; reg [WL_DATA_W-1:0] P``wl_mem_data;

    //========================================================================
    // Per-instance responder + memory stubs (PE_M-independent).  Same weight image
    //   for every instance -> the batched and serial die see IDENTICAL weights.
    //========================================================================
    `define SYSRESP(P) \
        integer P``t, P``sb, P``re, P``ft; \
        always @* P``em_val = gen_bf16(P``em_tok*MODEL_DIM + P``em_idx + 7001); \
        always @* P``fn_val = gen_bf16(P``fn_idx + 7207); \
        always @* P``gn_val = gen_bf16(P``db_layer*1024 + P``gn_which*512 + P``gn_idx + 7411); \
        always @* for (P``t=0;P``t<LM_TN;P``t=P``t+1) \
            P``lw_col[16*P``t+:16] = gen_bf16((P``lw_vtile*LM_TN+P``t)*MODEL_DIM + P``lw_k + 7603); \
        always @* begin \
            for (P``t=0;P``t<PE_N;P``t=P``t+1) begin \
                P``aw_q[4*P``t+:4] = f_awq(P``db_layer, P``aw_sel, P``aw_grp*PE_N+P``t, P``aw_k); \
                for (P``sb=0;P``sb<A_NSB;P``sb=P``sb+1) begin \
                    P``aw_d   [16*(P``sb*PE_N+P``t)+:16] = f_awd (P``db_layer, P``aw_sel, P``aw_grp*PE_N+P``t); \
                    P``aw_dmin[16*(P``sb*PE_N+P``t)+:16] = f_awdm(P``db_layer, P``aw_sel, P``aw_grp*PE_N+P``t); \
                    P``aw_scales[96*(P``sb*PE_N+P``t)   +:32] = gen_s32(P``db_layer*7919+P``aw_sel*104729+(P``aw_grp*PE_N+P``t)*611953+601); \
                    P``aw_scales[96*(P``sb*PE_N+P``t)+32+:32] = gen_s32(P``db_layer*7919+P``aw_sel*104729+(P``aw_grp*PE_N+P``t)*611953+602); \
                    P``aw_scales[96*(P``sb*PE_N+P``t)+64+:32] = gen_s32(P``db_layer*7919+P``aw_sel*104729+(P``aw_grp*PE_N+P``t)*611953+603); \
                end end end \
        always @* begin \
            for (P``re=0;P``re<N_EXPERT;P``re=P``re+1) begin \
                P``rw_q[4*P``re+:4] = f_rwq(P``db_layer, P``re, P``rw_k); \
                for (P``sb=0;P``sb<R_NSB;P``sb=P``sb+1) begin \
                    P``rw_d   [16*(P``sb*N_EXPERT+P``re)+:16] = gen_fp16(P``db_layer*7919+P``re*350377+421); \
                    P``rw_dmin[16*(P``sb*N_EXPERT+P``re)+:16] = gen_fp16(P``db_layer*7919+P``re*350377+431); \
                    P``rw_scales[96*(P``sb*N_EXPERT+P``re)   +:32] = gen_s32(P``db_layer*7919+P``re*350377+441); \
                    P``rw_scales[96*(P``sb*N_EXPERT+P``re)+32+:32] = gen_s32(P``db_layer*7919+P``re*350377+442); \
                    P``rw_scales[96*(P``sb*N_EXPERT+P``re)+64+:32] = gen_s32(P``db_layer*7919+P``re*350377+443); \
                end end end \
        always @* begin \
            for (P``ft=0;P``ft<TN;P``ft=P``ft+1) begin \
                P``fw_q   [4*P``ft+:4] = f_fwq(P``db_layer, P``fw_sel, P``fw_shared, P``fw_eidx, P``fw_grp*TN+P``ft, P``fw_k); \
                P``fw_q_up[4*P``ft+:4] = f_fwq(P``db_layer, 3,         P``fw_shared, P``fw_eidx, P``fw_grp*TN+P``ft, P``fw_k); \
                for (P``sb=0;P``sb<FF_NSB_D;P``sb=P``sb+1) begin \
                    P``fw_d_g   [16*(P``sb*TN+P``ft)+:16] = gen_fp16(P``db_layer*7919+P``fw_sel*104729+P``fw_shared*15485863+P``fw_eidx*350377+(P``fw_grp*TN+P``ft)*611953+521); \
                    P``fw_dmin_g[16*(P``sb*TN+P``ft)+:16] = gen_fp16(P``db_layer*7919+P``fw_sel*104729+P``fw_shared*15485863+P``fw_eidx*350377+(P``fw_grp*TN+P``ft)*611953+531); \
                    P``fw_d_u   [16*(P``sb*TN+P``ft)+:16] = gen_fp16(P``db_layer*7919+3*104729+P``fw_shared*15485863+P``fw_eidx*350377+(P``fw_grp*TN+P``ft)*611953+521); \
                    P``fw_dmin_u[16*(P``sb*TN+P``ft)+:16] = gen_fp16(P``db_layer*7919+3*104729+P``fw_shared*15485863+P``fw_eidx*350377+(P``fw_grp*TN+P``ft)*611953+531); \
                    P``fw_scales_g[96*(P``sb*TN+P``ft)   +:32] = gen_s32(P``db_layer*7919+P``fw_sel*104729+P``fw_shared*15485863+P``fw_eidx*350377+(P``fw_grp*TN+P``ft)*611953+541); \
                    P``fw_scales_g[96*(P``sb*TN+P``ft)+32+:32] = gen_s32(P``db_layer*7919+P``fw_sel*104729+P``fw_shared*15485863+P``fw_eidx*350377+(P``fw_grp*TN+P``ft)*611953+542); \
                    P``fw_scales_g[96*(P``sb*TN+P``ft)+64+:32] = gen_s32(P``db_layer*7919+P``fw_sel*104729+P``fw_shared*15485863+P``fw_eidx*350377+(P``fw_grp*TN+P``ft)*611953+543); \
                    P``fw_scales_u[96*(P``sb*TN+P``ft)   +:32] = gen_s32(P``db_layer*7919+3*104729+P``fw_shared*15485863+P``fw_eidx*350377+(P``fw_grp*TN+P``ft)*611953+541); \
                    P``fw_scales_u[96*(P``sb*TN+P``ft)+32+:32] = gen_s32(P``db_layer*7919+3*104729+P``fw_shared*15485863+P``fw_eidx*350377+(P``fw_grp*TN+P``ft)*611953+542); \
                    P``fw_scales_u[96*(P``sb*TN+P``ft)+64+:32] = gen_s32(P``db_layer*7919+3*104729+P``fw_shared*15485863+P``fw_eidx*350377+(P``fw_grp*TN+P``ft)*611953+543); \
                end end end \
        assign P``mem_req_ready = {DDR_NCH{1'b1}}; \
        always @* begin P``mem_resp_valid = {DDR_NCH{1'b0}}; P``mem_resp_data = {(DDR_NCH*DDR_DATA_W){1'b0}}; P``mem_resp_tag = {(DDR_NCH*DDR_TAG_W){1'b0}}; end \
        always @* P``flash_row = {ROW_BITS{1'b0}}; \
        reg [31:0] P``fl_timer; reg P``fl_active; reg P``prev_freq; \
        always @(posedge clk) begin \
            if (rst) begin P``fl_timer<=32'd0; P``fl_active<=1'b0; P``flash_done<=1'b0; P``prev_freq<=1'b0; end \
            else begin \
                P``flash_done <= 1'b0; \
                if (!P``fl_active) begin if (P``flash_req && !P``prev_freq) begin P``fl_active<=1'b1; P``fl_timer<=FLASH_LAT[31:0]; end end \
                else begin if (P``fl_timer <= 32'd1) begin P``flash_done<=1'b1; P``fl_active<=1'b0; end else P``fl_timer <= P``fl_timer - 32'd1; end \
                P``prev_freq <= P``flash_req; \
            end end \
        always @(posedge clk) P``wl_mem_data <= {WL_DATA_W{1'b0}};

    //========================================================================
    // SERIAL reference : ONE glm_q4k_system at PE_M=1, SELF_KV=1 (INTRA off).
    //   Driven K+1 times per K-case (positions 0..K) -- the pager builds the KV.
    //========================================================================
    reg              s_start;  reg [TOKW-1:0] s_prompt_tok; reg [POSW-1:0] s_start_pos; reg [IDXW:0] s_s_len;
    wire [VOCAB*16-1:0] s_logits;  wire [TOKW-1:0] s_argmax_o;
    `SYSIO(s_)
    glm_q4k_system #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .SWIN(SWINB), .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN),
        .PE_M(1), .PER_ROW_POS(0), .INTRA_CAUSAL(0), .DSA_REAL_IDX(1),
        .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX),
        .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH), .SELF_KV(1),
        .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W),
        .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD),
        .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN)
    ) sys1 (
        .clk(clk), .rst(rst),
        .start(s_start), .prompt_tok(s_prompt_tok), .start_pos(s_start_pos), .s_len(s_s_len),
        .pos_vec(s_start_pos), .s_len_vec(s_s_len), .seq_vec(1'b0),
        .busy(s_busy), .done(s_done), .next_tok(s_next_tok), .tok_valid(s_tok_valid),
        .logits(s_logits),
        .em_req(s_em_req), .em_tok(s_em_tok), .em_idx(s_em_idx), .em_val(s_em_val),
        .db_layer(s_db_layer), .idx_fresh(s_idx_fresh), .idx_win(s_idx_win),
        .gn_req(s_gn_req), .gn_which(s_gn_which), .gn_idx(s_gn_idx), .gn_val(s_gn_val),
        .aw_req(s_aw_req), .aw_sel(s_aw_sel), .aw_grp(s_aw_grp), .aw_k(s_aw_k),
        .aw_q(s_aw_q), .aw_d(s_aw_d), .aw_dmin(s_aw_dmin), .aw_scales(s_aw_scales),
        .rw_req(s_rw_req), .rw_k(s_rw_k),
        .rw_q(s_rw_q), .rw_d(s_rw_d), .rw_dmin(s_rw_dmin), .rw_scales(s_rw_scales),
        .fw_req(s_fw_req), .fw_sel(s_fw_sel), .fw_grp(s_fw_grp), .fw_k(s_fw_k),
        .fw_shared(s_fw_shared), .fw_eidx(s_fw_eidx),
        .fw_q(s_fw_q), .fw_q_up(s_fw_q_up),
        .fw_d_g(s_fw_d_g), .fw_dmin_g(s_fw_dmin_g), .fw_scales_g(s_fw_scales_g),
        .fw_d_u(s_fw_d_u), .fw_dmin_u(s_fw_dmin_u), .fw_scales_u(s_fw_scales_u),
        .fn_req(s_fn_req), .fn_idx(s_fn_idx), .fn_val(s_fn_val),
        .lw_req(s_lw_req), .lw_vtile(s_lw_vtile), .lw_k(s_lw_k), .lw_col(s_lw_col),
        .kc_ckv({(KV_LORA*16){1'b0}}), .kc_krope({(ROPE*16){1'b0}}),
        .kc_req(s_kc_req), .kc_idx(s_kc_idx),
        .kv_row_sel(s_kv_row_sel), .kv_row_in({ROW_BITS{1'b0}}),
        .flash_req(s_flash_req), .flash_is_expert(s_flash_is_expert),
        .flash_expert_id(s_flash_expert_id), .flash_row_idx(s_flash_row_idx),
        .flash_done(s_flash_done), .flash_row(s_flash_row),
        .pf_valid(1'b0), .pf_expert_id({EIDXW{1'b0}}),
        .mem_req_valid(s_mem_req_valid), .mem_req_ready(s_mem_req_ready),
        .mem_req_addr(s_mem_req_addr), .mem_req_tag(s_mem_req_tag),
        .mem_resp_valid(s_mem_resp_valid), .mem_resp_ready(s_mem_resp_ready),
        .mem_resp_data(s_mem_resp_data), .mem_resp_tag(s_mem_resp_tag),
        .wl_mem_en(s_wl_mem_en), .wl_mem_addr(s_wl_mem_addr), .wl_mem_data(s_wl_mem_data),
        .decomp_tbl_we(1'b0), .decomp_tbl_sel(1'b0), .decomp_tbl_addr(9'h000), .decomp_tbl_wdata(10'h000),
        .argmax_o(s_argmax_o), .h_state(), .mdl_busy(s_mdl_busy),
        .ec_resp_valid(), .ec_hit(), .ec_resp_slot(), .ec_busy(),
        .ec_hit_count(), .ec_miss_count(), .ec_demand_stall_cycles(), .ec_pf_issued(), .ec_pf_hit(),
        .kv_row_valid(), .kv_row_out(), .kv_busy(),
        .kv_lat_row(), .kv_lat_valid(), .kv_lat_row_all(), .kv_lat_valid_all(),
        .kv_append_count(), .kv_resident_lo(), .kv_overflowed(), .ec_dropped(),
        .xbar_req_count(), .xbar_resp_count(), .xbar_resp_valid(), .xbar_resp_data(),
        .loader_busy(), .loader_done_count(), .loader_beat_count(),
        .loader_w_q(), .loader_in_valid()
    );
    `SYSRESP(s_)

    //========================================================================
    // BATCHED instances : glm_q4k_system at PE_M=B (B=2,3,4) with SELF_KV=1,
    //   INTRA_CAUSAL=1, PER_ROW_POS=1.  Driven ONCE per K-case over positions 0..K.
    //========================================================================
    `define BATCH_SYS(NM,B) \
        reg  NM``_start; reg [B*TOKW-1:0] NM``_token_id; reg [POSW-1:0] NM``_pos; reg [IDXW:0] NM``_slen; \
        reg  [POSW*B-1:0] NM``_pos_vec; reg [(IDXW+1)*B-1:0] NM``_slen_vec; \
        wire [B*VOCAB*16-1:0] NM``_logits; wire [B*TOKW-1:0] NM``_argmax; \
        `SYSIO(NM``_) \
        glm_q4k_system #( \
            .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB), \
            .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM), \
            .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN), \
            .SWIN(SWINB), .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK), \
            .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), \
            .BLK(BLK), .LM_TN(LM_TN), \
            .PE_M(B), .PER_ROW_POS(1), .INTRA_CAUSAL(1), .DSA_REAL_IDX(1), \
            .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX), \
            .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH), .SELF_KV(1), \
            .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W), \
            .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD), \
            .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN) \
        ) NM ( \
            .clk(clk), .rst(rst), \
            .start(NM``_start), .prompt_tok(NM``_token_id), .start_pos(NM``_pos), .s_len(NM``_slen), \
            .pos_vec(NM``_pos_vec), .s_len_vec(NM``_slen_vec), .seq_vec({($clog2(B)*B){1'b0}}), \
            .busy(NM``_busy), .done(NM``_done), .next_tok(NM``_next_tok), .tok_valid(NM``_tok_valid), \
            .logits(NM``_logits), \
            .em_req(NM``_em_req), .em_tok(NM``_em_tok), .em_idx(NM``_em_idx), .em_val(NM``_em_val), \
            .db_layer(NM``_db_layer), .idx_fresh(NM``_idx_fresh), .idx_win(NM``_idx_win), \
            .gn_req(NM``_gn_req), .gn_which(NM``_gn_which), .gn_idx(NM``_gn_idx), .gn_val(NM``_gn_val), \
            .aw_req(NM``_aw_req), .aw_sel(NM``_aw_sel), .aw_grp(NM``_aw_grp), .aw_k(NM``_aw_k), \
            .aw_q(NM``_aw_q), .aw_d(NM``_aw_d), .aw_dmin(NM``_aw_dmin), .aw_scales(NM``_aw_scales), \
            .rw_req(NM``_rw_req), .rw_k(NM``_rw_k), \
            .rw_q(NM``_rw_q), .rw_d(NM``_rw_d), .rw_dmin(NM``_rw_dmin), .rw_scales(NM``_rw_scales), \
            .fw_req(NM``_fw_req), .fw_sel(NM``_fw_sel), .fw_grp(NM``_fw_grp), .fw_k(NM``_fw_k), \
            .fw_shared(NM``_fw_shared), .fw_eidx(NM``_fw_eidx), \
            .fw_q(NM``_fw_q), .fw_q_up(NM``_fw_q_up), \
            .fw_d_g(NM``_fw_d_g), .fw_dmin_g(NM``_fw_dmin_g), .fw_scales_g(NM``_fw_scales_g), \
            .fw_d_u(NM``_fw_d_u), .fw_dmin_u(NM``_fw_dmin_u), .fw_scales_u(NM``_fw_scales_u), \
            .fn_req(NM``_fn_req), .fn_idx(NM``_fn_idx), .fn_val(NM``_fn_val), \
            .lw_req(NM``_lw_req), .lw_vtile(NM``_lw_vtile), .lw_k(NM``_lw_k), .lw_col(NM``_lw_col), \
            .kc_ckv({(KV_LORA*16){1'b0}}), .kc_krope({(ROPE*16){1'b0}}), \
            .kc_req(NM``_kc_req), .kc_idx(NM``_kc_idx), \
            .kv_row_sel(NM``_kv_row_sel), .kv_row_in({ROW_BITS{1'b0}}), \
            .flash_req(NM``_flash_req), .flash_is_expert(NM``_flash_is_expert), \
            .flash_expert_id(NM``_flash_expert_id), .flash_row_idx(NM``_flash_row_idx), \
            .flash_done(NM``_flash_done), .flash_row(NM``_flash_row), \
            .pf_valid(1'b0), .pf_expert_id({EIDXW{1'b0}}), \
            .mem_req_valid(NM``_mem_req_valid), .mem_req_ready(NM``_mem_req_ready), \
            .mem_req_addr(NM``_mem_req_addr), .mem_req_tag(NM``_mem_req_tag), \
            .mem_resp_valid(NM``_mem_resp_valid), .mem_resp_ready(NM``_mem_resp_ready), \
            .mem_resp_data(NM``_mem_resp_data), .mem_resp_tag(NM``_mem_resp_tag), \
            .wl_mem_en(NM``_wl_mem_en), .wl_mem_addr(NM``_wl_mem_addr), .wl_mem_data(NM``_wl_mem_data), \
            .decomp_tbl_we(1'b0), .decomp_tbl_sel(1'b0), .decomp_tbl_addr(9'h000), .decomp_tbl_wdata(10'h000), \
            .argmax_o(NM``_argmax), .h_state(), .mdl_busy(NM``_mdl_busy), \
            .ec_resp_valid(), .ec_hit(), .ec_resp_slot(), .ec_busy(), \
            .ec_hit_count(), .ec_miss_count(), .ec_demand_stall_cycles(), .ec_pf_issued(), .ec_pf_hit(), \
            .kv_row_valid(), .kv_row_out(), .kv_busy(), \
            .kv_lat_row(), .kv_lat_valid(), .kv_lat_row_all(), .kv_lat_valid_all(), \
            .kv_append_count(), .kv_resident_lo(), .kv_overflowed(), .ec_dropped(), \
            .xbar_req_count(), .xbar_resp_count(), .xbar_resp_valid(), .xbar_resp_data(), \
            .loader_busy(), .loader_done_count(), .loader_beat_count(), \
            .loader_w_q(), .loader_in_valid() \
        ); \
        `SYSRESP(NM``_)

    `BATCH_SYS(b2, 2)
    `BATCH_SYS(b3, 3)
    `BATCH_SYS(b4, 4)

    //========================================================================
    // stimulus + capture
    //========================================================================
    // the K+1 verify tokens for a K-case (arbitrary distinct tokens < VOCAB).
    reg [TOKW-1:0] tks [0:KMAX_TB];
    initial begin tks[0]=4'd5; tks[1]=4'd9; tks[2]=4'd3; tks[3]=4'd11; end

    // captured per-row batched results, and per-decode serial results.
    reg [VOCAB*16-1:0] blog [0:KMAX_TB];   reg [TOKW-1:0] barg [0:KMAX_TB];
    reg [VOCAB*16-1:0] slog [0:KMAX_TB];   reg [TOKW-1:0] sarg [0:KMAX_TB];

    integer gi;

    // drive one PE_M=1 serial decode of token tk at position ps (s_len=ps).
    task serial_decode; input [TOKW-1:0] tk; input integer ps; output [VOCAB*16-1:0] olog;
                        output [TOKW-1:0] oarg; begin
        s_prompt_tok = tk; s_start_pos = ps[POSW-1:0]; s_s_len = ps[IDXW:0];
        @(negedge clk); s_start = 1'b1;
        @(negedge clk); s_start = 1'b0;
        wait (s_tok_valid === 1'b1);
        @(negedge clk);
        olog = s_logits; oarg = s_argmax_o;
        repeat (6) @(negedge clk);   // drain H_DONE -> H_IDLE before the next decode
    end endtask

    // compare a captured batched pass's rows 0..K against the serial chain slog/sarg.
    integer errs_here, kk2, r2;
    task check_rows; input integer K; input [255:0] label; begin
        errs_here = 0;
        for (r2 = 0; r2 <= K; r2 = r2 + 1) begin
            test_count = test_count + 1;
            if (^blog[r2] === 1'bx || ^barg[r2] === 1'bx) begin
                $display("FAIL[%0s] K=%0d row%0d batched logits/argmax X/Z", label, K, r2);
                errs_here = errs_here + 1;
            end else if (blog[r2] !== slog[r2]) begin
                $display("FAIL[%0s] K=%0d row%0d (pos=%0d) LOGITS batch != serial", label, K, r2, r2);
                errs_here = errs_here + 1;
            end else if (barg[r2] !== sarg[r2]) begin
                $display("FAIL[%0s] K=%0d row%0d (pos=%0d) ARGMAX batch=%0d serial=%0d",
                         label, K, r2, r2, barg[r2], sarg[r2]);
                errs_here = errs_here + 1;
            end else
                $display("  PASS[%0s] K=%0d row%0d (tok=%0d pos=%0d): logits+argmax(=%0d) === serial",
                         label, K, r2, tks[r2], r2, barg[r2]);
        end
        errors = errors + errs_here;
        $display("    (%0s: K=%0d, ONE PE_M=%0d batched pass over positions 0..%0d == serial PE_M=1 decodes 0..%0d)",
                 label, K, K+1, K, K);
    end endtask

    integer pr;
    initial begin
        errors = 0; test_count = 0;
        rst = 1'b1; s_start = 1'b0; b2_start = 1'b0; b3_start = 1'b0; b4_start = 1'b0;
        s_prompt_tok = {TOKW{1'b0}}; s_start_pos = {POSW{1'b0}}; s_s_len = {(IDXW+1){1'b0}};
        b2_token_id = 0; b3_token_id = 0; b4_token_id = 0;
        b2_pos = 0; b3_pos = 0; b4_pos = 0; b2_slen = 0; b3_slen = 0; b4_slen = 0;
        b2_pos_vec = 0; b3_pos_vec = 0; b4_pos_vec = 0;
        b2_slen_vec = 0; b3_slen_vec = 0; b4_slen_vec = 0;

        // ---- ONE reset: every instance starts with an EMPTY pager ----
        repeat (4) @(negedge clk); rst = 1'b0; @(negedge clk);

        // ---- ONE serial reference chain: decode tokens 0..KMAX at positions 0..KMAX.
        //   SELF_KV=1 builds sys1's pager as it goes -> decode r reads positions 0..r-1
        //   (per layer).  The prefixes 0..K cover every K case.  Each decode uses the
        //   GIVEN draft token (not autoregressive) -- exactly what row r of the batch
        //   verifies.
        for (kk2 = 0; kk2 <= KMAX_TB; kk2 = kk2 + 1)
            serial_decode(tks[kk2], kk2, slog[kk2], sarg[kk2]);

        // ---- per-row positions for the batched instances : row r ropes at position r.
        //   The batched instances share ONE reset (empty pager); their pass runs at
        //   shared s_len=0, so INTRA_CAUSAL supplies EVERY causal key (positions 0..r-1
        //   are the current-token keys of rows 0..r-1) -- the batched pager is never
        //   read, only (harmlessly) appended, so no per-pass reset is needed.
        for (pr = 0; pr < 2; pr = pr + 1) b2_pos_vec[POSW*pr +: POSW] =
`ifdef INJECT_SHARED_POS
            {POSW{1'b0}};   // INJECT: every row shares position 0 -> intra indexing wrong
`else
            pr[POSW-1:0];
`endif
        for (pr = 0; pr < 3; pr = pr + 1) b3_pos_vec[POSW*pr +: POSW] =
`ifdef INJECT_SHARED_POS
            {POSW{1'b0}};
`else
            pr[POSW-1:0];
`endif
        for (pr = 0; pr < 4; pr = pr + 1) b4_pos_vec[POSW*pr +: POSW] =
`ifdef INJECT_SHARED_POS
            {POSW{1'b0}};
`else
            pr[POSW-1:0];
`endif

        // ================= K=1 : ONE PE_M=2 batched pass (positions 0..1) =========
        b2_pos = {POSW{1'b0}}; b2_slen = {(IDXW+1){1'b0}};
        b2_token_id[TOKW*0 +: TOKW] = tks[0];
        b2_token_id[TOKW*1 +: TOKW] = tks[1];
        @(negedge clk); b2_start = 1'b1; @(negedge clk); b2_start = 1'b0;
        wait (b2_done === 1'b1); @(negedge clk);
        for (r2 = 0; r2 <= 1; r2 = r2 + 1) begin
            blog[r2] = b2_logits[VOCAB*16*r2 +: VOCAB*16];
            barg[r2] = b2_argmax [TOKW*r2 +: TOKW];
        end
        check_rows(1, "K1_PEM2");

        // NOTE on the INJECTION: K=1 row 1 attends exactly ONE intra key, and softmax
        //   over a single key is 1.0, so row 1's output is INDEPENDENT of its query
        //   position -- K=1 is a WEAK injection target.  The injection FIRST bites at
        //   K>=2 (row 2 attends 2 keys, whose softmax weights depend on the RoPE
        //   positions).  So the injection runs the FULL K=1,2,3 (rows 2 and 3 diverge).

        // ================= K=2 : ONE PE_M=3 batched pass (positions 0..2) =========
        b3_pos = {POSW{1'b0}}; b3_slen = {(IDXW+1){1'b0}};
        b3_token_id[TOKW*0 +: TOKW] = tks[0];
        b3_token_id[TOKW*1 +: TOKW] = tks[1];
        b3_token_id[TOKW*2 +: TOKW] = tks[2];
        @(negedge clk); b3_start = 1'b1; @(negedge clk); b3_start = 1'b0;
        wait (b3_done === 1'b1); @(negedge clk);
        for (r2 = 0; r2 <= 2; r2 = r2 + 1) begin
            blog[r2] = b3_logits[VOCAB*16*r2 +: VOCAB*16];
            barg[r2] = b3_argmax [TOKW*r2 +: TOKW];
        end
        check_rows(2, "K2_PEM3");

        // ================= K=3 : ONE PE_M=4 batched pass (positions 0..3) =========
        b4_pos = {POSW{1'b0}}; b4_slen = {(IDXW+1){1'b0}};
        b4_token_id[TOKW*0 +: TOKW] = tks[0];
        b4_token_id[TOKW*1 +: TOKW] = tks[1];
        b4_token_id[TOKW*2 +: TOKW] = tks[2];
        b4_token_id[TOKW*3 +: TOKW] = tks[3];
        @(negedge clk); b4_start = 1'b1; @(negedge clk); b4_start = 1'b0;
        wait (b4_done === 1'b1); @(negedge clk);
        for (r2 = 0; r2 <= 3; r2 = r2 + 1) begin
            blog[r2] = b4_logits[VOCAB*16*r2 +: VOCAB*16];
            barg[r2] = b4_argmax [TOKW*r2 +: TOKW];
        end
        check_rows(3, "K3_PEM4");

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, test_count);
            $fatal(1, "glm_q4k_intra_batch_verify_tb: batched intra-causal system != serial decode chain");
        end
        $display("ALL %0d TESTS PASSED  (glm_q4k_system PE_M=K+1 INTRA_CAUSAL=1 PER_ROW_POS=1 SELF_KV=1 batched verify == serial PE_M=1 SELF_KV=1 decode chain, full-logit bit-exact + argmax, K=1,2,3)", test_count);
        $finish;
    end

    initial begin
        #40000000; $display("FAIL: global timeout"); $fatal(1, "timeout");
    end

endmodule
