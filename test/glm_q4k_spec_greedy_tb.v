`timescale 1ns/1ps
//============================================================================
// glm_q4k_spec_greedy_tb.v
//   5c  spec==greedy BINDING test for the composed speculating top
//       glm_q4k_spec_system (glm_q4k_system PE_M=K+1 + spec_decode_seq + KV write-back).
//----------------------------------------------------------------------------
// WHAT IS PROVEN (self-checking; $fatal on any mismatch):
//
//   The composed speculating top's COMMITTED TOKEN STREAM is IDENTICAL to a plain
//   PE_M=1 greedy decode of the same prompt -- for K=1,2,3 AND across a range of
//   draft accept rates (ALL-ACCEPT, ALL-REJECT, MIXED) -- with only the token
//   COUNT-per-pass changing.  This is the load-bearing spec==greedy invariant.
//
//   GREEDY REFERENCE (independent): a single PE_M=1, SELF_KV=1 glm_q4k_system,
//   decoded AUTOREGRESSIVELY -- g[0]=prompt; decode g[i] at position i (s_len=i)
//   -> gstream[i]=argmax; g[i+1]=gstream[i].  SELF_KV=1 builds its pager as it
//   goes, so decode i reads the KV of positions 0..i-1 (per layer).  gstream[] is
//   the model's greedy rollout.
//
//   SPECULATING DUT: glm_q4k_spec_system at DRAFT_K=K (PE_M=K+1), driven ONE pass
//   at a time so the TB controls the committed length t exactly and can present
//   ANY accept pattern.  The DUT's per-pass KV WRITE-BACK appends the committed
//   prefix's latents to its pager, so pass i+1 reads the true shared prefix 0..t-1.
//   We assert, beat-for-beat and X-free, that the DUT's committed stream EQUALS
//   gstream[0..NCHK-1] for every (K, pattern) -- SAME tokens, different #passes.
//
//   Per pass at committed length t with K drafts:
//     correct draft d_j = gstream[t+j-1]  (=> accepted: d_j == m_j)
//     wrong   draft d_j = gstream[t+j-1] ^ 1 (guaranteed != model token => reject)
//   ALL-ACCEPT: every d correct (p=K, K+1 tokens/pass).
//   ALL-REJECT: d_1 wrong (p=0, 1 token/pass).
//   MIXED     : d_1 correct, d_2.. wrong (p=1) for K>=2; K=1 alternates accept/reject.
//
// INJECTION (compile-time -DINJECT_RAW_DRAFT, NEVER in the normal build):
//   glm_q4k_spec_system feeds the RAW DRAFTS as spec_decode_seq's truth_vec, so the
//   loop COMMITS raw drafts instead of the model's argmaxes.  Under ALL-REJECT the
//   drafts are deliberately WRONG, so the committed stream diverges from greedy at
//   the first token -> this oracle MUST FAIL.  (Reverting re-passes.)  This is the
//   exact thing spec_decode_seq must prevent.
//============================================================================
module glm_q4k_spec_greedy_tb;

    // ---- L=3 slice (2 dense + 1 MoE), dense attention (S<=TOPK_ATTN => DSA no-op),
    //   S_MAX=16 so a continuing loop has room (extent t+K <= S_MAX). ----
    localparam integer MODEL_DIM = 16;
    localparam integer L         = 2;      // 1 dense + 1 MoE (both layer types; KV_NSEQ=L)
    localparam integer N_DENSE   = 1;
    localparam integer VOCAB     = 16;
    localparam integer H_HEADS   = 2;
    localparam integer NOPE      = 4;
    localparam integer ROPE      = 4;
    localparam integer V_DIM     = 4;
    localparam integer Q_LORA    = 8;
    localparam integer KV_LORA   = 8;
    localparam integer S_MAX     = 8;
    localparam integer TOPK_ATTN = 8;      // dense: S<=TOPK_ATTN always -> DSA no-op
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
    localparam integer SWINB     = S_MAX;
    localparam integer CACHE_SLOTS = 4;
    localparam integer FLASH_LAT   = 8;
    localparam integer KV_CTX      = 1024;
    localparam integer KV_RESIDENT = 16;
    localparam integer EFIFO_DEPTH = 16;
    localparam integer DDR_NCH     = 4;
    localparam integer DDR_ADDR_W  = 32;
    localparam integer DDR_DATA_W  = 256;
    localparam integer DDR_TAG_W   = 8;
    localparam integer DDR_ROW_LAT = 10;
    localparam integer DDR_RESP_QD = 4;
    localparam integer WL_KMAX     = 256;
    localparam integer WL_ADDR_W   = 24;
    localparam integer LOADER_KLEN = MODEL_DIM;

    // ---- derived (mirror glm_q4k_system) ----
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
    localparam integer WL_DATA_W= 256;

    reg clk; initial clk = 1'b0; always #5 clk = ~clk;
    reg rst;
    integer test_count, errors;

    //========================================================================
    // deterministic stimulus generators (IDENTICAL to glm_q4k_intra_batch_verify_tb
    //   -> the SAME weight image for the greedy reference and every spec DUT).
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
    // Per-instance MEMORY interface wires (PE_M-independent: one shared single-fetch
    //   stream, width identical at every PE_M).  Prefix P => P``em_val etc.
    //========================================================================
    `define MEMIO(P) \
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
        wire [DDR_NCH-1:0] P``mem_req_valid; wire [DDR_NCH-1:0] P``mem_req_ready; \
        wire [DDR_NCH*DDR_ADDR_W-1:0] P``mem_req_addr; wire [DDR_NCH*DDR_TAG_W-1:0] P``mem_req_tag; \
        reg [DDR_NCH-1:0] P``mem_resp_valid; wire [DDR_NCH-1:0] P``mem_resp_ready; \
        reg [DDR_NCH*DDR_DATA_W-1:0] P``mem_resp_data; reg [DDR_NCH*DDR_TAG_W-1:0] P``mem_resp_tag; \
        wire P``wl_mem_en; wire [WL_ADDR_W-1:0] P``wl_mem_addr; reg [WL_DATA_W-1:0] P``wl_mem_data;

    //========================================================================
    // Per-instance responder + memory stubs (identical formula => identical weights).
    //========================================================================
    `define MEMRESP(P) \
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
    // GREEDY REFERENCE : ONE glm_q4k_system at PE_M=1, SELF_KV=1 (INTRA off).
    //========================================================================
    reg              s_start;  reg [TOKW-1:0] s_prompt_tok; reg [POSW-1:0] s_start_pos; reg [IDXW:0] s_s_len;
    wire             s_busy, s_done, s_tok_valid; wire [TOKW-1:0] s_next_tok;
    wire [VOCAB*16-1:0] s_logits;  wire [TOKW-1:0] s_argmax_o;
    // greedy reference : per-position committed token + FULL logit vector (the
    //   full-logit binding target -- catches KV/position bugs even if argmax is stable).
    reg [VOCAB*16-1:0] slog [0:31];
    `MEMIO(s_)
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
    ) sref (
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
        .ext_append_valid(1'b0), .ext_append_row({ROW_BITS{1'b0}}), .ext_append_seq(1'b0),
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
        .argmax_o(s_argmax_o), .h_state(), .mdl_busy(),
        .ec_resp_valid(), .ec_hit(), .ec_resp_slot(), .ec_busy(),
        .ec_hit_count(), .ec_miss_count(), .ec_demand_stall_cycles(), .ec_pf_issued(), .ec_pf_hit(),
        .kv_row_valid(), .kv_row_out(), .kv_busy(),
        .kv_lat_row(), .kv_lat_valid(), .kv_lat_row_all(), .kv_lat_valid_all(),
        .kv_append_count(), .kv_resident_lo(), .kv_overflowed(), .ec_dropped(),
        .xbar_req_count(), .xbar_resp_count(), .xbar_resp_valid(), .xbar_resp_data(),
        .loader_busy(), .loader_done_count(), .loader_beat_count(),
        .loader_w_q(), .loader_in_valid()
    );
    `MEMRESP(s_)

    //========================================================================
    // SPECULATING DUTs : glm_q4k_spec_system at DRAFT_K=1,2,3 (PE_M=2,3,4).
    //   ONE broadcast control bus routed to the DUT selected by `dsel`; the others
    //   see start=0 (idle).  Committed streams OR-combined (only one runs at a time).
    //========================================================================
    reg  [1:0]        dsel;             // 1,2,3 : which DUT is being driven
    reg               g_start;
    reg  [TOKW-1:0]   g_prompt;
    reg  [POSW-1:0]   g_startpos;
    reg  [15:0]       g_npass;
    reg  [3*TOKW-1:0] g_draft;          // up to K=3 drafts (low K*TOKW used)
    reg  [1:0]        g_ndraft;

    `define SPEC_DUT(NM,KK) \
        wire NM``_busy, NM``_done; \
        wire NM``_cv; wire [TOKW-1:0] NM``_ct; wire NM``_acc; \
        wire [31:0] NM``_tot, NM``_mp, NM``_ac, NM``_rj, NM``_wl; \
        wire [(KK+1)*VOCAB*16-1:0] NM``_lg; \
        `MEMIO(NM``_) \
        glm_q4k_spec_system #( \
            .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB), \
            .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM), \
            .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN), \
            .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK), \
            .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), \
            .BLK(BLK), .LM_TN(LM_TN), .DRAFT_K(KK), \
            .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX), \
            .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH), \
            .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W), \
            .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD), \
            .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN) \
        ) NM ( \
            .clk(clk), .rst(rst), \
            .start((dsel == KK) ? g_start : 1'b0), \
            .prompt_tok(g_prompt), .start_pos(g_startpos), .num_passes(g_npass), \
            .draft_in(g_draft[KK*TOKW-1:0]), .n_draft(g_ndraft), \
            .busy(NM``_busy), .done(NM``_done), \
            .commit_valid(NM``_cv), .commit_tok(NM``_ct), .accepted(NM``_acc), \
            .total_tokens(NM``_tot), .main_passes(NM``_mp), .accepts(NM``_ac), .rejects(NM``_rj), \
            .weight_loads(NM``_wl), .logits(NM``_lg), .argmax_o(), \
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
            .mdl_busy(), .ec_hit_count(), .ec_miss_count(), .kv_append_count() \
        ); \
        `MEMRESP(NM``_)

    `SPEC_DUT(k1, 1)
    `SPEC_DUT(k2, 2)
    `SPEC_DUT(k3, 3)

    // ---- committed-stream capture (only the running DUT commits) ----
    wire        act_cv  = k1_cv | k2_cv | k3_cv;
    wire [TOKW-1:0] act_ct = k1_cv ? k1_ct : (k2_cv ? k2_ct : k3_ct);
    wire        act_done = (dsel==2'd1)? k1_done : (dsel==2'd2)? k2_done : k3_done;
    // the running DUT's per-row logits, zero-padded to the K=3 (4-row) width.
    wire [4*VOCAB*16-1:0] act_lg = (dsel==2'd1) ? {{(2*VOCAB*16){1'b0}}, k1_lg}
                                 : (dsel==2'd2) ? {{(1*VOCAB*16){1'b0}}, k2_lg}
                                 :                 k3_lg;
    // 5d : the running DUT's HARDWARE amortization counters (weight_loads counts one
    //   die weight-load per outer pass at S_LAUNCH; total_tokens counts committed tokens).
    //   Reading these -- not the tb's own pass count -- makes A_eff a MEASURED number.
    wire [31:0] act_wl  = (dsel==2'd1)? k1_wl  : (dsel==2'd2)? k2_wl  : k3_wl;
    wire [31:0] act_tot = (dsel==2'd1)? k1_tot : (dsel==2'd2)? k2_tot : k3_tot;
    reg  [4*VOCAB*16-1:0] dlog;   // latched at done : this pass's per-row logits

    reg  [TOKW-1:0] cstr [0:63];
    reg  [15:0]     ccnt;
    always @(posedge clk) begin
        if (rst) ccnt <= 16'd0;
        else if (act_cv) begin cstr[ccnt[5:0]] <= act_ct; ccnt <= ccnt + 16'd1; end
    end

    //========================================================================
    // greedy rollout + per-config driving
    //========================================================================
    localparam integer NMAX = 5;       // greedy tokens computed (covers drafts up to t+K-1)
    localparam integer NCHK = 3;       // committed tokens compared per config
    reg [TOKW-1:0] gstream [0:NMAX-1]; // greedy rollout (the reference stream)
    reg [TOKW-1:0] gstream_prompt;     // the initial prompt token (row 0 of pass 0)

    integer di;

    // one PE_M=1 autoregressive greedy decode step: token `tk` @ position `ps`, s_len=ps.
    task greedy_step; input [TOKW-1:0] tk; input integer ps; output [TOKW-1:0] oarg; begin
        s_prompt_tok = tk; s_start_pos = ps[POSW-1:0]; s_s_len = ps[IDXW:0];
        @(negedge clk); s_start = 1'b1;
        @(negedge clk); s_start = 1'b0;
        wait (s_tok_valid === 1'b1);
        @(negedge clk);
        oarg = s_argmax_o;
        slog[ps] = s_logits;            // the greedy reference's full logit vector @ position ps
        repeat (6) @(negedge clk);
    end endtask

    // pulse a global reset (clears every pager + FSM + the capture counter).
    task do_reset; begin
        rst = 1'b1; g_start = 1'b0; s_start = 1'b0; dsel = 2'd0;
        repeat (4) @(negedge clk); rst = 1'b0; @(negedge clk);
    end endtask

    // run ONE spec pass on DUT `which` (DRAFT_K=kk): commit length t, cur_tok ct,
    //   drafts dv (kk*TOKW).  Returns nothing; the capture array grows by (p+1).
    task run_pass; input [1:0] which; input integer kk; input [TOKW-1:0] ct;
                   input integer t; input [3*TOKW-1:0] dv; begin
        dsel       = which;
        g_prompt   = ct;
        g_startpos = t[POSW-1:0];
        g_npass    = 16'd1;
        g_draft    = dv;
        g_ndraft   = kk[1:0];
        @(negedge clk); g_start = 1'b1;
        @(negedge clk); g_start = 1'b0;
        wait (act_done === 1'b1);
        dlog = act_lg;               // latch this pass's per-row logits (stable at done)
        @(negedge clk);
        repeat (4) @(negedge clk);   // let the final commit beats settle
    end endtask

    // ---- pattern codes ---- 0=ALL_ACCEPT 1=ALL_REJECT 2=MIXED
    // build the K-draft vector for a pass at committed length t.
    function [3*TOKW-1:0] mk_drafts;
        input integer kk; input integer t; input integer pat; input integer passno;
        integer j; reg [3*TOKW-1:0] d; reg wrong; begin
            d = {3*TOKW{1'b0}};
            for (j = 1; j <= kk; j = j + 1) begin
                wrong = 1'b0;
                if (pat == 1) wrong = 1'b1;                       // ALL-REJECT: every draft wrong
                else if (pat == 2) begin                          // MIXED
                    if (kk == 1) wrong = (passno % 2 == 1);       //   K=1: alternate accept/reject
                    else         wrong = (j >= 2);                //   K>=2: accept d_1, reject rest
                end
                // correct draft d_j = gstream[t+j-1]; wrong => flip a bit (guaranteed !=)
                d[(j-1)*TOKW +: TOKW] = wrong ? (gstream[t+j-1] ^ 1'b1)
                                              : gstream[t+j-1];
            end
            mk_drafts = d;
        end
    endfunction

    integer t, nb0, cfgpass, kk, pat, which, ci;
    reg [TOKW-1:0] curtok;
    reg [255:0] pname;

    // run a full config : DUT `which` (DRAFT_K=kk), pattern `pat`, from the prompt.
    task run_config; input [1:0] which; input integer kk; input integer pat; begin
        do_reset;
        t = 0; curtok = gstream_prompt; cfgpass = 0;
        while (ccnt < NCHK[15:0]) begin
            nb0 = ccnt;
            run_pass(which, kk, curtok, t, mk_drafts(kk, t, pat, cfgpass));
            // verify the beats committed THIS pass match greedy at t.. :
            //   (1) TOKEN == greedy token, and (2) FULL LOGIT VECTOR of the committed
            //   batch row (== the model's own row at position ci) == the greedy
            //   reference's logit vector at position ci -- the latter binds the KV
            //   write-back / per-row position bit-exactly even when the argmax is stable.
            for (ci = nb0; ci < ccnt; ci = ci + 1) begin
                test_count = test_count + 1;
                if (^cstr[ci[5:0]] === 1'bx) begin
                    $display("FAIL[%0s K=%0d] committed[%0d] is X", pname, kk, ci);
                    errors = errors + 1;
                end else if (cstr[ci[5:0]] !== gstream[ci]) begin
                    $display("FAIL[%0s K=%0d] committed token[%0d]=%0d != greedy=%0d",
                             pname, kk, ci, cstr[ci[5:0]], gstream[ci]);
                    errors = errors + 1;
                end else if (^dlog[(ci-nb0)*VOCAB*16 +: VOCAB*16] === 1'bx) begin
                    $display("FAIL[%0s K=%0d] committed row logits[%0d] X", pname, kk, ci);
                    errors = errors + 1;
                end else if (dlog[(ci-nb0)*VOCAB*16 +: VOCAB*16] !== slog[ci]) begin
                    $display("FAIL[%0s K=%0d] committed row FULL-LOGIT[pos=%0d] != greedy ref (KV/position diverged)",
                             pname, kk, ci);
                    errors = errors + 1;
                end
            end
            t       = ccnt;
            curtok  = cstr[(ccnt-1) & 6'h3f];
            cfgpass = cfgpass + 1;
            if (cfgpass > 30) begin
                $display("FAIL[%0s K=%0d] runaway passes", pname, kk); errors=errors+1;
                disable run_config;
            end
        end
        // 5d -- MEASURE A_eff from the DUT's OWN hardware counters (not the tb's).
        //   (a) faithfulness: the die pulses exactly ONE weight-load per outer pass, so the
        //       hardware weight_loads counter must equal the tb's actual pass count.  If the
        //       batch secretly re-loaded weights per row, act_wl would exceed cfgpass and
        //       this fails -- this is the load-bearing "K+1 verified in ONE load" evidence.
        if (act_wl !== cfgpass) begin
            $display("FAIL[%0s K=%0d] hw weight_loads=%0d != actual passes=%0d (amortization counter unfaithful)",
                     pname, kk, act_wl, cfgpass); errors = errors + 1;
        end
        //   (b) the hardware committed-token counter must equal what the scoreboard captured.
        if (act_tot !== ccnt) begin
            $display("FAIL[%0s K=%0d] hw total_tokens=%0d != captured commits=%0d",
                     pname, kk, act_tot, ccnt); errors = errors + 1;
        end
        //   (c) CEILING: under ALL-ACCEPT every pass commits exactly K+1 tokens for its ONE
        //       weight-load -- the maximum amortization the K+1 batch can extract.  A_eff = K+1.
        if (pat == 0 && act_tot !== cfgpass*(kk+1)) begin
            $display("FAIL[%0s K=%0d] ALL-ACCEPT ceiling: %0d tokens over %0d loads != K+1 per load",
                     pname, kk, act_tot, act_wl); errors = errors + 1;
        end
        //   A_eff = tokens per weight-load (HW-measured); bytes/token = 25.50 GB / A_eff
        //   (PE_M=1 no-spec = 25.50 GB/token; speculation amortizes ONE load over A_eff tokens).
        $display("  PASS[%0s K=%0d]: committed %0d tokens == greedy over %0d weight-load(s)  [A_eff %0d.%02d tok/load  =>  %0d.%02d GB/token]",
                 pname, kk, act_tot, act_wl,
                 (act_tot / act_wl), ((act_tot*100/act_wl) % 100),
                 (2550*act_wl/act_tot)/100, (2550*act_wl/act_tot)%100);
        $fflush;
    end endtask

    initial begin
        errors = 0; test_count = 0;
        rst = 1'b1; s_start = 1'b0; g_start = 1'b0; dsel = 2'd0;
        s_prompt_tok = 0; s_start_pos = 0; s_s_len = 0;
        g_prompt = 0; g_startpos = 0; g_npass = 0; g_draft = 0; g_ndraft = 0;
        gstream_prompt = 4'd5;                                // arbitrary prompt token < VOCAB
        repeat (6) @(negedge clk); rst = 1'b0; @(negedge clk);

        // ---- build the greedy rollout gstream[0..NMAX-1] (autoregressive) ----
        greedy_step(gstream_prompt, 0, gstream[0]);
        $display("  [greedy] step 0 -> %0d", gstream[0]); $fflush;
        for (di = 1; di < NMAX; di = di + 1) begin
            greedy_step(gstream[di-1], di, gstream[di]);
            $display("  [greedy] step %0d -> %0d", di, gstream[di]); $fflush;
        end
        $write("greedy rollout gstream =");
        for (di = 0; di < NMAX; di = di + 1) $write(" %0d", gstream[di]);
        $write("\n"); $fflush;
        // sanity : consecutive tokens distinct enough that the accept schedules exercise
        //   real divergence points (not required for correctness, informative only).

        // ================= K x {ALL-ACCEPT, ALL-REJECT, MIXED} =================
        //   Default build runs the K=2 column (fast correctness check incl. the
        //   partial-accept-at-t>0 KV write-back hazard).  -DSPEC_FULL runs the whole
        //   K=1,2,3 x 3-pattern matrix (the shipped gate).
        pname = "ACCEPT"; run_config(2'd2, 2, 0);
        pname = "REJECT"; run_config(2'd2, 2, 1);
        pname = "MIXED";  run_config(2'd2, 2, 2);
`ifdef SPEC_FULL
        pname = "ACCEPT"; run_config(2'd1, 1, 0);
        pname = "REJECT"; run_config(2'd1, 1, 1);
        pname = "MIXED";  run_config(2'd1, 1, 2);
        pname = "ACCEPT"; run_config(2'd3, 3, 0);
        pname = "REJECT"; run_config(2'd3, 3, 1);
        pname = "MIXED";  run_config(2'd3, 3, 2);
`endif

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, test_count);
            $fatal(1, "glm_q4k_spec_greedy_tb: committed stream != greedy (spec==greedy broken)");
        end
        $display("ALL %0d TESTS PASSED  (glm_q4k_spec_system committed stream == PE_M=1 greedy decode, K=1,2,3 x {ACCEPT,REJECT,MIXED}; KV write-back + accept/reject verified; A_eff MEASURED from hw weight_loads counter -- ALL-ACCEPT hits the K+1/load ceiling)", test_count);
        $finish;
    end

    initial begin
        #200000000; $display("FAIL: global timeout"); $fatal(1, "timeout");
    end

endmodule
