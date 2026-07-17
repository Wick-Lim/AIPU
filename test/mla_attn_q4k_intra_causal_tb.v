`timescale 1ns/1ps
//============================================================================
// mla_attn_q4k_intra_causal_tb.v
//   5b-leaf LEAF ORACLE for INTRA-BATCH CAUSAL MLA attention (the new
//   INTRA_CAUSAL feature of mla_attn_q4k).
//----------------------------------------------------------------------------
// WHAT THIS PROVES  (a DUT-vs-DUT, same-arithmetic, EXACT === oracle):
//
//   A batched PE_M=B pass with INTRA_CAUSAL=1 over B tokens at CONSECUTIVE
//   causal positions p..p+B-1 -- row r decodes position p+r and attends
//   positions 0..p+r-1, where 0..p-1 are supplied as SHARED cached keys and
//   p..p+r-1 are the CURRENT-token keys of the earlier rows 0..r-1 computed in
//   THAT SAME pass -- reproduces, BIT-EXACT on the FULL MODEL_DIM output, the
//   SERIAL single-row (PE_M=1) chain of the same B tokens:
//
//     the serial chain decodes token r at position p+r with s_len=p+r over a
//     cache holding 0..p-1 (the shared prefix) PLUS p..p+r-1 which are the
//     latents the EARLIER serial decodes 0..r-1 committed (each decode's OWN
//     kv_lat_row is appended to the cache for the next) -- i.e. exactly the
//     die-internal KV write-back the system performs between serial steps.
//
//   The reference is the SAME module at PE_M=1 (INTRA_CAUSAL=0): NOT a
//   hand-rolled parallel reference that could share the batched path's bug.
//   The batch MUST reproduce the serial chain because the batched intra key i
//   (a virtual cache key at index s_reg+i whose latent is ckv_cur[i] and whose
//   roped k_rope is krope_cur[i]) is BYTE-IDENTICAL to the cache key the serial
//   chain wrote at position p+i: both are x_i*W_dkv / rope(x_i*W_kr, p+i) and
//   flow through the identical RMSNorm+W_uk (score) / W_uv (value) / DSA-index.
//
//   Also asserted: the WIDENED egress kv_lat_row_all[r] === serial decode r's
//   kv_lat_row (the per-row write-back 5b-sys will append).
//
//   Regimes covered per B: DENSE (every row's extent <= TOPK), SPARSE (every
//   row's extent > TOPK -> real per-row DSA over the COMBINED pager+intra key
//   set), and MIXED (dense prefix rows, sparse tail rows in one pass).
//
//   S_MAX=8, TOPK=4  ->  a row's combined extent 1..4 is dense, 5..8 sparse.
//   B = 2, 3, 4.   Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch/X.
//
// INJECTION HOOKS (compile-time, NEVER set in the normal build):
//   -D INTRA_INJECT_NOMASK   : drop the per-row causal mask (row r attends its
//                              OWN current token) -> this oracle MUST FAIL.
//   -D INTRA_INJECT_SKIPNORM : skip RMSNorm on the intra key's latent before
//                              W_uk -> this oracle MUST FAIL.
//============================================================================
module mla_attn_q4k_intra_causal_tb;
    // ---- small slice: S_MAX(8) > TOPK(4) so sparse selection is exercised ----
    localparam integer MODEL_DIM = 16;
    localparam integer H_HEADS   = 2;
    localparam integer NOPE      = 4;
    localparam integer ROPE      = 4;
    localparam integer V_DIM     = 4;
    localparam integer Q_LORA    = 8;
    localparam integer KV_LORA   = 8;
    localparam integer S_MAX     = 8;
    localparam integer TOPK      = 4;
    localparam integer THETA     = 8000000;
    localparam integer PE_N      = 2;
    localparam integer POSW      = 20;
    localparam integer BLK       = 128;
    localparam integer QK_DIM    = NOPE + ROPE;
    localparam integer IDXW      = (S_MAX <= 1) ? 1 : $clog2(S_MAX);
    localparam integer HQK       = H_HEADS*QK_DIM;
    localparam integer HNOPE     = H_HEADS*NOPE;
    localparam integer HV        = H_HEADS*V_DIM;
    localparam integer KVR       = (KV_LORA+ROPE)*16;   // packed latent row width

    localparam integer KMAX = (MODEL_DIM>Q_LORA)?
                 ((MODEL_DIM>KV_LORA)?((MODEL_DIM>HV)?MODEL_DIM:HV)
                                     :((KV_LORA>HV)?KV_LORA:HV))
                :((Q_LORA>KV_LORA)?((Q_LORA>HV)?Q_LORA:HV)
                                  :((KV_LORA>HV)?KV_LORA:HV));
    localparam integer OMAX = (HQK>MODEL_DIM)?
                 ((HQK>HNOPE)?((HQK>HV)?HQK:HV):((HNOPE>HV)?HNOPE:HV))
                :((MODEL_DIM>HNOPE)?((MODEL_DIM>HV)?MODEL_DIM:HV)
                                   :((HNOPE>HV)?HNOPE:HV));
    localparam integer NGMAX = (OMAX + PE_N - 1)/PE_N;
    localparam integer GRPW  = (NGMAX<=1)?1:$clog2(NGMAX);
    localparam integer KCW   = (KMAX<=1)?1:$clog2(KMAX);
    localparam integer NSB   = (KMAX + 255)/256;   // = 1 here

    // batched SWIN: q-dependent + intra-divergent per-row selection can use up to
    //   min(PE_M*TOPK, S_MAX) distinct union keys -> S_MAX(8) covers every B here.
    localparam integer SWINB = S_MAX;

    localparam [15:0] D_FP16  = 16'h1C00;   // d    = 2^-8
    localparam [15:0] DM_FP16 = 16'h2800;   // dmin = 2^-5

    reg clk=1'b0, rst=1'b1;
    always #5 clk = ~clk;

    // ================= shared stimulus ROMs (Q4_K codes + bf16 cache) =========
    reg [3:0]  Wq_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [3:0]  Wq_uq  [0:HQK-1][0:Q_LORA-1];
    reg [3:0]  Wq_dkv [0:KV_LORA-1][0:MODEL_DIM-1];
    reg [3:0]  Wq_kr  [0:ROPE-1][0:MODEL_DIM-1];
    reg [3:0]  Wq_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [3:0]  Wq_uv  [0:HV-1][0:KV_LORA-1];
    reg [3:0]  Wq_o   [0:MODEL_DIM-1][0:HV-1];
    reg [95:0] Sc_dq  [0:Q_LORA-1];
    reg [95:0] Sc_uq  [0:HQK-1];
    reg [95:0] Sc_dkv [0:KV_LORA-1];
    reg [95:0] Sc_kr  [0:ROPE-1];
    reg [95:0] Sc_uk  [0:HNOPE-1];
    reg [95:0] Sc_uv  [0:HV-1];
    reg [95:0] Sc_o   [0:MODEL_DIM-1];

    localparam integer BMAX = 4;                       // max batch tested
    reg [15:0] xr  [0:BMAX-1][0:MODEL_DIM-1];          // per-row token activations
    // ONE shared KV window: positions 0..p-1 are the random prefix; positions
    //   p.. are FILLED by the serial chain (each decode's committed latent).
    reg [15:0] CKV [0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP [0:S_MAX-1][0:ROPE-1];

    // deterministic stimulus generators (same hashing style as the sparse TB).
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h = (seed*2654435761) ^ (seed<<13) ^ (seed*40503);
        s = h[3];
        if (band==1) e = 8'd125 + h[6:4];
        else         e = 8'd124 + h[5:4];
        m = h[12:6];
        gen_bf16 = {s, e, m};
    end endfunction
    function [3:0] gen_q4; input integer seed; integer h; begin
        h = (seed*2246822519) ^ (seed<<11) ^ (seed*3266489917);
        gen_q4 = h[6:3];
    end endfunction
    function [95:0] gen_sc96; input integer seed;
        reg [31:0] a, b, c; begin
        a = (seed*2654435761) ^ (seed<<7)  ^ 32'h9E3779B9;
        b = (seed*2246822519) ^ (seed<<13) ^ 32'h85EBCA6B;
        c = (seed*3266489917) ^ (seed<<3)  ^ 32'hC2B2AE35;
        gen_sc96 = {a, b, c};
    end endfunction

    reg stim_bump = 1'b0;    // wake the responder blocks after array writes

    integer ii,jj,kk,sc;
    task build_stimulus; input integer seed0; input integer band; input integer p; begin
        sc = seed0;
        for (ii=0;ii<Q_LORA;ii=ii+1) for (jj=0;jj<MODEL_DIM;jj=jj+1) begin Wq_dq[ii][jj]=gen_q4(sc); sc=sc+1; end
        for (ii=0;ii<HQK;ii=ii+1)    for (jj=0;jj<Q_LORA;jj=jj+1)    begin Wq_uq[ii][jj]=gen_q4(sc); sc=sc+1; end
        for (ii=0;ii<KV_LORA;ii=ii+1)for (jj=0;jj<MODEL_DIM;jj=jj+1) begin Wq_dkv[ii][jj]=gen_q4(sc); sc=sc+1; end
        for (ii=0;ii<ROPE;ii=ii+1)   for (jj=0;jj<MODEL_DIM;jj=jj+1) begin Wq_kr[ii][jj]=gen_q4(sc); sc=sc+1; end
        for (ii=0;ii<HNOPE;ii=ii+1)  for (jj=0;jj<KV_LORA;jj=jj+1)   begin Wq_uk[ii][jj]=gen_q4(sc); sc=sc+1; end
        for (ii=0;ii<HV;ii=ii+1)     for (jj=0;jj<KV_LORA;jj=jj+1)   begin Wq_uv[ii][jj]=gen_q4(sc); sc=sc+1; end
        for (ii=0;ii<MODEL_DIM;ii=ii+1)for (jj=0;jj<HV;jj=jj+1)      begin Wq_o[ii][jj]=gen_q4(sc);  sc=sc+1; end
        for (ii=0;ii<Q_LORA;ii=ii+1)   begin Sc_dq[ii] =gen_sc96(sc); sc=sc+1; end
        for (ii=0;ii<HQK;ii=ii+1)      begin Sc_uq[ii] =gen_sc96(sc); sc=sc+1; end
        for (ii=0;ii<KV_LORA;ii=ii+1)  begin Sc_dkv[ii]=gen_sc96(sc); sc=sc+1; end
        for (ii=0;ii<ROPE;ii=ii+1)     begin Sc_kr[ii] =gen_sc96(sc); sc=sc+1; end
        for (ii=0;ii<HNOPE;ii=ii+1)    begin Sc_uk[ii] =gen_sc96(sc); sc=sc+1; end
        for (ii=0;ii<HV;ii=ii+1)       begin Sc_uv[ii] =gen_sc96(sc); sc=sc+1; end
        for (ii=0;ii<MODEL_DIM;ii=ii+1)begin Sc_o[ii]  =gen_sc96(sc); sc=sc+1; end
        // DISTINCT per-row token activations.
        for (kk=0;kk<BMAX;kk=kk+1)
            for (ii=0;ii<MODEL_DIM;ii=ii+1) begin xr[kk][ii]=gen_bf16(sc,band); sc=sc+1; end
        // KV window: fill the WHOLE array (positions >= p are overwritten by the
        //   serial chain before they are read; batched never reads >= p).  Only
        //   the prefix 0..p-1 is load-bearing for the batched pass.
        for (ii=0;ii<S_MAX;ii=ii+1) begin
            for (jj=0;jj<KV_LORA;jj=jj+1) begin CKV[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
            for (jj=0;jj<ROPE;jj=jj+1)    begin KRP[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
        end
        stim_bump = ~stim_bump;
    end endtask

    // ================= shared Q4_K weight responder (pure function) ============
    localparam integer WRW = PE_N*(96*NSB + 16*NSB + 16*NSB + 4);
    function [WRW-1:0] w_resp; input [3:0] sel; input [GRPW-1:0] grp; input [KCW-1:0] k;
        reg [PE_N*4-1:0]       q;
        reg [16*PE_N*NSB-1:0]  d, dm;
        reg [96*PE_N*NSB-1:0]  s96;
        integer t, cidx; begin
        q = {PE_N*4{1'b0}}; d = {16*PE_N*NSB{1'b0}};
        dm = {16*PE_N*NSB{1'b0}}; s96 = {96*PE_N*NSB{1'b0}};
        for (t=0;t<PE_N;t=t+1) begin
            cidx = grp*PE_N + t;
            case (sel)
            4'd0: if (cidx < Q_LORA)   begin q[4*t+:4]=Wq_dq [cidx][k]; s96[96*t+:96]=Sc_dq [cidx]; d[16*t+:16]=D_FP16; dm[16*t+:16]=DM_FP16; end
            4'd1: if (cidx < HQK)      begin q[4*t+:4]=Wq_uq [cidx][k]; s96[96*t+:96]=Sc_uq [cidx]; d[16*t+:16]=D_FP16; dm[16*t+:16]=DM_FP16; end
            4'd2: if (cidx < KV_LORA)  begin q[4*t+:4]=Wq_dkv[cidx][k]; s96[96*t+:96]=Sc_dkv[cidx]; d[16*t+:16]=D_FP16; dm[16*t+:16]=DM_FP16; end
            4'd3: if (cidx < ROPE)     begin q[4*t+:4]=Wq_kr [cidx][k]; s96[96*t+:96]=Sc_kr [cidx]; d[16*t+:16]=D_FP16; dm[16*t+:16]=DM_FP16; end
            4'd4: if (cidx < HNOPE)    begin q[4*t+:4]=Wq_uk [cidx][k]; s96[96*t+:96]=Sc_uk [cidx]; d[16*t+:16]=D_FP16; dm[16*t+:16]=DM_FP16; end
            4'd5: if (cidx < HV)       begin q[4*t+:4]=Wq_uv [cidx][k]; s96[96*t+:96]=Sc_uv [cidx]; d[16*t+:16]=D_FP16; dm[16*t+:16]=DM_FP16; end
            4'd6: if (cidx < MODEL_DIM)begin q[4*t+:4]=Wq_o  [cidx][k]; s96[96*t+:96]=Sc_o  [cidx]; d[16*t+:16]=D_FP16; dm[16*t+:16]=DM_FP16; end
            default: q[4*t+:4] = 4'h0;
            endcase
        end
        w_resp = {s96, dm, d, q};
    end endfunction

    // ================= shared KV-cache responder (pure function) ===============
    //   returns {kc_krope, kc_ckv} of key `idx` from the ONE window.
    localparam integer KCRW = (ROPE + KV_LORA)*16;
    function [KCRW-1:0] kc_resp; input [IDXW-1:0] idx;
        reg [KV_LORA*16-1:0] cv; reg [ROPE*16-1:0] kr; integer c; begin
        cv = {KV_LORA*16{1'b0}}; kr = {ROPE*16{1'b0}};
        for (c=0;c<KV_LORA;c=c+1) cv[16*c +:16] = CKV[idx][c];
        for (c=0;c<ROPE;c=c+1)    kr[16*c +:16] = KRP[idx][c];
        kc_resp = {kr, cv};
    end endfunction

    // ===========================================================================
    //  BATCHED DUTs : PE_M = 2 / 3 / 4, INTRA_CAUSAL=1, per-row pos, real DSA.
    // ===========================================================================
    // -- macro-free per-instance boilerplate (mirrors the sparse TB style) --
    // B=2
    localparam integer B2=2, SEQW2=(B2<=1)?1:$clog2(B2);
    reg  d2_start; wire d2_busy, d2_done;
    reg  [POSW-1:0] d2_pos; reg [POSW*B2-1:0] d2_pos_vec;
    reg  [IDXW:0] d2_slen; reg [(IDXW+1)*B2-1:0] d2_slen_vec;
    reg  [MODEL_DIM*16*B2-1:0] d2_xvec; wire [MODEL_DIM*16*B2-1:0] d2_out;
    wire d2_wreq; wire [3:0] d2_wsel; wire [GRPW-1:0] d2_wgrp; wire [KCW-1:0] d2_wk;
    reg  [PE_N*4-1:0] d2_wq; reg [16*PE_N*NSB-1:0] d2_wd,d2_wdmin; reg [96*PE_N*NSB-1:0] d2_wscales;
    wire d2_kcreq; wire [IDXW-1:0] d2_kcidx; wire [SEQW2-1:0] d2_kcseq;
    reg  [KV_LORA*16-1:0] d2_kcckv; reg [ROPE*16-1:0] d2_kckrope; reg d2_kcvalid;
    wire [B2*KVR-1:0] d2_lat_all; wire [B2-1:0] d2_lat_valid_all;
    mla_attn_q4k #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK(TOPK),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .BLK(BLK), .PE_M(B2), .SWIN(SWINB),
        .PER_ROW_POS(1), .DSA_REAL_IDX(1), .INTRA_CAUSAL(1)) dut2 (
        .clk(clk), .rst(rst), .start(d2_start), .busy(d2_busy), .done(d2_done),
        .pos(d2_pos), .pos_vec(d2_pos_vec), .s_len(d2_slen), .s_len_vec(d2_slen_vec),
        .seq_vec({SEQW2*B2{1'b0}}), .x_vec(d2_xvec),
        .w_req(d2_wreq), .w_sel(d2_wsel), .w_grp(d2_wgrp), .w_k(d2_wk),
        .w_q(d2_wq), .w_d(d2_wd), .w_dmin(d2_wdmin), .w_scales(d2_wscales),
        .kc_req(d2_kcreq), .kc_idx(d2_kcidx), .kc_seq(d2_kcseq),
        .kc_ckv(d2_kcckv), .kc_krope(d2_kckrope), .kc_valid(d2_kcvalid), .out(d2_out),
        .kv_lat_row(), .kv_lat_valid(),
        .kv_lat_row_all(d2_lat_all), .kv_lat_valid_all(d2_lat_valid_all));

    // B=3
    localparam integer B3=3, SEQW3=(B3<=1)?1:$clog2(B3);
    reg  d3_start; wire d3_busy, d3_done;
    reg  [POSW-1:0] d3_pos; reg [POSW*B3-1:0] d3_pos_vec;
    reg  [IDXW:0] d3_slen; reg [(IDXW+1)*B3-1:0] d3_slen_vec;
    reg  [MODEL_DIM*16*B3-1:0] d3_xvec; wire [MODEL_DIM*16*B3-1:0] d3_out;
    wire d3_wreq; wire [3:0] d3_wsel; wire [GRPW-1:0] d3_wgrp; wire [KCW-1:0] d3_wk;
    reg  [PE_N*4-1:0] d3_wq; reg [16*PE_N*NSB-1:0] d3_wd,d3_wdmin; reg [96*PE_N*NSB-1:0] d3_wscales;
    wire d3_kcreq; wire [IDXW-1:0] d3_kcidx; wire [SEQW3-1:0] d3_kcseq;
    reg  [KV_LORA*16-1:0] d3_kcckv; reg [ROPE*16-1:0] d3_kckrope; reg d3_kcvalid;
    wire [B3*KVR-1:0] d3_lat_all; wire [B3-1:0] d3_lat_valid_all;
    mla_attn_q4k #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK(TOPK),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .BLK(BLK), .PE_M(B3), .SWIN(SWINB),
        .PER_ROW_POS(1), .DSA_REAL_IDX(1), .INTRA_CAUSAL(1)) dut3 (
        .clk(clk), .rst(rst), .start(d3_start), .busy(d3_busy), .done(d3_done),
        .pos(d3_pos), .pos_vec(d3_pos_vec), .s_len(d3_slen), .s_len_vec(d3_slen_vec),
        .seq_vec({SEQW3*B3{1'b0}}), .x_vec(d3_xvec),
        .w_req(d3_wreq), .w_sel(d3_wsel), .w_grp(d3_wgrp), .w_k(d3_wk),
        .w_q(d3_wq), .w_d(d3_wd), .w_dmin(d3_wdmin), .w_scales(d3_wscales),
        .kc_req(d3_kcreq), .kc_idx(d3_kcidx), .kc_seq(d3_kcseq),
        .kc_ckv(d3_kcckv), .kc_krope(d3_kckrope), .kc_valid(d3_kcvalid), .out(d3_out),
        .kv_lat_row(), .kv_lat_valid(),
        .kv_lat_row_all(d3_lat_all), .kv_lat_valid_all(d3_lat_valid_all));

    // B=4
    localparam integer B4=4, SEQW4=(B4<=1)?1:$clog2(B4);
    reg  d4_start; wire d4_busy, d4_done;
    reg  [POSW-1:0] d4_pos; reg [POSW*B4-1:0] d4_pos_vec;
    reg  [IDXW:0] d4_slen; reg [(IDXW+1)*B4-1:0] d4_slen_vec;
    reg  [MODEL_DIM*16*B4-1:0] d4_xvec; wire [MODEL_DIM*16*B4-1:0] d4_out;
    wire d4_wreq; wire [3:0] d4_wsel; wire [GRPW-1:0] d4_wgrp; wire [KCW-1:0] d4_wk;
    reg  [PE_N*4-1:0] d4_wq; reg [16*PE_N*NSB-1:0] d4_wd,d4_wdmin; reg [96*PE_N*NSB-1:0] d4_wscales;
    wire d4_kcreq; wire [IDXW-1:0] d4_kcidx; wire [SEQW4-1:0] d4_kcseq;
    reg  [KV_LORA*16-1:0] d4_kcckv; reg [ROPE*16-1:0] d4_kckrope; reg d4_kcvalid;
    wire [B4*KVR-1:0] d4_lat_all; wire [B4-1:0] d4_lat_valid_all;
    mla_attn_q4k #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK(TOPK),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .BLK(BLK), .PE_M(B4), .SWIN(SWINB),
        .PER_ROW_POS(1), .DSA_REAL_IDX(1), .INTRA_CAUSAL(1)) dut4 (
        .clk(clk), .rst(rst), .start(d4_start), .busy(d4_busy), .done(d4_done),
        .pos(d4_pos), .pos_vec(d4_pos_vec), .s_len(d4_slen), .s_len_vec(d4_slen_vec),
        .seq_vec({SEQW4*B4{1'b0}}), .x_vec(d4_xvec),
        .w_req(d4_wreq), .w_sel(d4_wsel), .w_grp(d4_wgrp), .w_k(d4_wk),
        .w_q(d4_wq), .w_d(d4_wd), .w_dmin(d4_wdmin), .w_scales(d4_wscales),
        .kc_req(d4_kcreq), .kc_idx(d4_kcidx), .kc_seq(d4_kcseq),
        .kc_ckv(d4_kcckv), .kc_krope(d4_kckrope), .kc_valid(d4_kcvalid), .out(d4_out),
        .kv_lat_row(), .kv_lat_valid(),
        .kv_lat_row_all(d4_lat_all), .kv_lat_valid_all(d4_lat_valid_all));

    // ===========================================================================
    //  PE_M=1 SERIAL REFERENCE (same module; INTRA off; re-run per chain step).
    // ===========================================================================
    reg  r1_start; wire r1_busy, r1_done;
    reg  [POSW-1:0] r1_pos; reg [IDXW:0] r1_slen;
    reg  [MODEL_DIM*16-1:0] r1_xvec; wire [MODEL_DIM*16-1:0] r1_out;
    wire r1_wreq; wire [3:0] r1_wsel; wire [GRPW-1:0] r1_wgrp; wire [KCW-1:0] r1_wk;
    reg  [PE_N*4-1:0] r1_wq; reg [16*PE_N*NSB-1:0] r1_wd,r1_wdmin; reg [96*PE_N*NSB-1:0] r1_wscales;
    wire r1_kcreq; wire [IDXW-1:0] r1_kcidx; wire r1_kcseq;
    reg  [KV_LORA*16-1:0] r1_kcckv; reg [ROPE*16-1:0] r1_kckrope; reg r1_kcvalid;
    wire [KVR-1:0] r1_lat; wire r1_lat_valid;
    mla_attn_q4k #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK(TOPK),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .BLK(BLK), .PE_M(1), .SWIN(SWINB),
        .DSA_REAL_IDX(1), .INTRA_CAUSAL(0)) ref1 (
        .clk(clk), .rst(rst), .start(r1_start), .busy(r1_busy), .done(r1_done),
        .pos(r1_pos), .pos_vec(r1_pos), .s_len(r1_slen), .s_len_vec(r1_slen),
        .seq_vec(1'b0), .x_vec(r1_xvec),
        .w_req(r1_wreq), .w_sel(r1_wsel), .w_grp(r1_wgrp), .w_k(r1_wk),
        .w_q(r1_wq), .w_d(r1_wd), .w_dmin(r1_wdmin), .w_scales(r1_wscales),
        .kc_req(r1_kcreq), .kc_idx(r1_kcidx), .kc_seq(r1_kcseq),
        .kc_ckv(r1_kcckv), .kc_krope(r1_kckrope), .kc_valid(r1_kcvalid), .out(r1_out),
        .kv_lat_row(r1_lat), .kv_lat_valid(r1_lat_valid),
        .kv_lat_row_all(), .kv_lat_valid_all());

    // ================= combinational responders (one per instance) =============
    always @(d2_wsel or d2_wgrp or d2_wk or stim_bump) {d2_wscales,d2_wdmin,d2_wd,d2_wq}=w_resp(d2_wsel,d2_wgrp,d2_wk);
    always @(d3_wsel or d3_wgrp or d3_wk or stim_bump) {d3_wscales,d3_wdmin,d3_wd,d3_wq}=w_resp(d3_wsel,d3_wgrp,d3_wk);
    always @(d4_wsel or d4_wgrp or d4_wk or stim_bump) {d4_wscales,d4_wdmin,d4_wd,d4_wq}=w_resp(d4_wsel,d4_wgrp,d4_wk);
    always @(r1_wsel or r1_wgrp or r1_wk or stim_bump) {r1_wscales,r1_wdmin,r1_wd,r1_wq}=w_resp(r1_wsel,r1_wgrp,r1_wk);

    // KV responders read the ONE shared window; chain_bump wakes them after appends.
    reg chain_bump = 1'b0;
    always @(d2_kcidx or stim_bump or chain_bump) {d2_kckrope,d2_kcckv}=kc_resp(d2_kcidx);
    always @(d3_kcidx or stim_bump or chain_bump) {d3_kckrope,d3_kcckv}=kc_resp(d3_kcidx);
    always @(d4_kcidx or stim_bump or chain_bump) {d4_kckrope,d4_kcckv}=kc_resp(d4_kcidx);
    always @(r1_kcidx or stim_bump or chain_bump) {r1_kckrope,r1_kcckv}=kc_resp(r1_kcidx);

    // kc_valid one cycle after kc_req (registered).
    always @(posedge clk) begin
        if (rst) begin d2_kcvalid<=1'b0; d3_kcvalid<=1'b0; d4_kcvalid<=1'b0; r1_kcvalid<=1'b0; end
        else begin d2_kcvalid<=d2_kcreq; d3_kcvalid<=d3_kcreq; d4_kcvalid<=d4_kcreq; r1_kcvalid<=r1_kcreq; end
    end

    // ===========================================================================
    //  DRIVERS + CAPTURE
    // ===========================================================================
    integer i, r;
    // captured batched results (out + per-row committed latent).
    reg [15:0] bout [0:BMAX-1][0:MODEL_DIM-1];
    reg [KVR-1:0] blat [0:BMAX-1];
    // captured serial-chain results.
    reg [15:0] rout [0:BMAX-1][0:MODEL_DIM-1];
    reg [KVR-1:0] rlat [0:BMAX-1];

    // run the B=2 batched pass over positions p..p+1 (extents p..p+1).
    task run_b2; input integer p; begin
        d2_pos = p[POSW-1:0];
        d2_pos_vec[POSW*0 +: POSW] = p[POSW-1:0];
        d2_pos_vec[POSW*1 +: POSW] = (p+1);
        d2_slen = p[IDXW:0];
        d2_slen_vec[(IDXW+1)*0 +: (IDXW+1)] = p[IDXW:0];
        d2_slen_vec[(IDXW+1)*1 +: (IDXW+1)] = p[IDXW:0];   // shared cached extent = p
        for (i=0;i<MODEL_DIM;i=i+1) begin
            d2_xvec[16*(MODEL_DIM*0+i)+:16]=xr[0][i];
            d2_xvec[16*(MODEL_DIM*1+i)+:16]=xr[1][i];
        end
        @(negedge clk); d2_start=1'b1; @(negedge clk); d2_start=1'b0;
        wait (d2_done==1'b1); @(negedge clk);
        for (r=0;r<B2;r=r+1) begin
            for (i=0;i<MODEL_DIM;i=i+1) bout[r][i]=d2_out[16*(MODEL_DIM*r+i)+:16];
            blat[r]=d2_lat_all[r*KVR +: KVR];
        end
    end endtask

    task run_b3; input integer p; begin
        d3_pos = p[POSW-1:0];
        d3_pos_vec[POSW*0 +: POSW] = p[POSW-1:0];
        d3_pos_vec[POSW*1 +: POSW] = (p+1);
        d3_pos_vec[POSW*2 +: POSW] = (p+2);
        d3_slen = p[IDXW:0];
        d3_slen_vec[(IDXW+1)*0 +: (IDXW+1)] = p[IDXW:0];
        d3_slen_vec[(IDXW+1)*1 +: (IDXW+1)] = p[IDXW:0];
        d3_slen_vec[(IDXW+1)*2 +: (IDXW+1)] = p[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) begin
            d3_xvec[16*(MODEL_DIM*0+i)+:16]=xr[0][i];
            d3_xvec[16*(MODEL_DIM*1+i)+:16]=xr[1][i];
            d3_xvec[16*(MODEL_DIM*2+i)+:16]=xr[2][i];
        end
        @(negedge clk); d3_start=1'b1; @(negedge clk); d3_start=1'b0;
        wait (d3_done==1'b1); @(negedge clk);
        for (r=0;r<B3;r=r+1) begin
            for (i=0;i<MODEL_DIM;i=i+1) bout[r][i]=d3_out[16*(MODEL_DIM*r+i)+:16];
            blat[r]=d3_lat_all[r*KVR +: KVR];
        end
    end endtask

    task run_b4; input integer p; begin
        d4_pos = p[POSW-1:0];
        d4_pos_vec[POSW*0 +: POSW] = p[POSW-1:0];
        d4_pos_vec[POSW*1 +: POSW] = (p+1);
        d4_pos_vec[POSW*2 +: POSW] = (p+2);
        d4_pos_vec[POSW*3 +: POSW] = (p+3);
        d4_slen = p[IDXW:0];
        d4_slen_vec[(IDXW+1)*0 +: (IDXW+1)] = p[IDXW:0];
        d4_slen_vec[(IDXW+1)*1 +: (IDXW+1)] = p[IDXW:0];
        d4_slen_vec[(IDXW+1)*2 +: (IDXW+1)] = p[IDXW:0];
        d4_slen_vec[(IDXW+1)*3 +: (IDXW+1)] = p[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) begin
            d4_xvec[16*(MODEL_DIM*0+i)+:16]=xr[0][i];
            d4_xvec[16*(MODEL_DIM*1+i)+:16]=xr[1][i];
            d4_xvec[16*(MODEL_DIM*2+i)+:16]=xr[2][i];
            d4_xvec[16*(MODEL_DIM*3+i)+:16]=xr[3][i];
        end
        @(negedge clk); d4_start=1'b1; @(negedge clk); d4_start=1'b0;
        wait (d4_done==1'b1); @(negedge clk);
        for (r=0;r<B4;r=r+1) begin
            for (i=0;i<MODEL_DIM;i=i+1) bout[r][i]=d4_out[16*(MODEL_DIM*r+i)+:16];
            blat[r]=d4_lat_all[r*KVR +: KVR];
        end
    end endtask

    // ONE serial reference decode at (x=xr[row], pos, s_len); append its latent.
    integer c;
    task ref_decode; input integer row; input integer pos; input integer slen; begin
        r1_pos  = pos[POSW-1:0];
        r1_slen = slen[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) r1_xvec[16*i+:16]=xr[row][i];
        @(negedge clk); r1_start=1'b1; @(negedge clk); r1_start=1'b0;
        wait (r1_done==1'b1); @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) rout[row][i]=r1_out[16*i+:16];
        rlat[row]=r1_lat;
        // APPEND the committed latent to the cache at position `pos` for the next
        //   serial step (die-internal KV write-back).  Layout: low KV_LORA*16 =
        //   c_kv, high ROPE*16 = k_rope -- EXACTLY kv_lat_row / kv_lat_row_all.
        for (c=0;c<KV_LORA;c=c+1) CKV[pos][c]=r1_lat[16*c +: 16];
        for (c=0;c<ROPE;c=c+1)    KRP[pos][c]=r1_lat[KV_LORA*16 + 16*c +: 16];
        chain_bump = ~chain_bump;   // wake the KV responders on the new entry
    end endtask

    // serial chain of B decodes at positions p..p+B-1 (each appends its latent).
    task serial_chain; input integer B; input integer p; integer rr; begin
        for (rr=0; rr<B; rr=rr+1) ref_decode(rr, p+rr, p+rr);
    end endtask

    // ===========================================================================
    //  CHECKS (exact, X-aware)
    // ===========================================================================
    integer errors, test_count, fails;
    task check_all; input integer B; input integer p; input [256*8-1:0] label; begin
        for (r=0;r<B;r=r+1) begin
            // (1) full-output bit-exact: batched row r === serial decode r.
            fails=0;
            for (i=0;i<MODEL_DIM;i=i+1) begin
                if (^bout[r][i] === 1'bx) begin
                    $display("FAIL[%0s] B=%0d p=%0d row%0d out[%0d] X/Z", label, B, p, r, i);
                    fails=fails+1;
                end else if (bout[r][i] !== rout[r][i]) begin
                    $display("FAIL[%0s] B=%0d p=%0d row%0d out[%0d] batch=%h serial=%h",
                             label, B, p, r, i, bout[r][i], rout[r][i]);
                    fails=fails+1;
                end
            end
            test_count=test_count+1;
            if (fails==0) $display("  PASS[%0s] B=%0d p=%0d row%0d (pos=%0d ext=%0d) out === serial",
                                   label, B, p, r, p+r, p+r);
            else errors=errors+fails;
            // (2) widened egress: batched kv_lat_row_all[r] === serial decode r latent.
            test_count=test_count+1;
            if (^blat[r] === 1'bx) begin
                $display("FAIL[%0s] B=%0d p=%0d row%0d kv_lat_row_all X/Z", label, B, p, r);
                errors=errors+1;
            end else if (blat[r] !== rlat[r]) begin
                $display("FAIL[%0s] B=%0d p=%0d row%0d kv_lat_row_all batch=%h serial=%h",
                         label, B, p, r, blat[r], rlat[r]);
                errors=errors+1;
            end else
                $display("  PASS[%0s] B=%0d p=%0d row%0d kv_lat_row_all === serial latent", label, B, p, r);
        end
    end endtask

    // run one full case: batched pass, then the serial chain, then compare.
    task run_case; input integer B; input integer seed0; input integer band; input integer p;
                   input [256*8-1:0] label; begin
        build_stimulus(seed0, band, p);
        if      (B==2) run_b2(p);
        else if (B==3) run_b3(p);
        else           run_b4(p);
        serial_chain(B, p);
        check_all(B, p, label);
        $display("    (%0s: B=%0d pos=%0d..%0d extents=%0d..%0d %s band=%0d)",
                 label, B, p, p+B-1, p, p+B-1,
                 ((p+B-1)<=TOPK)?"DENSE":((p>TOPK)?"SPARSE":"MIXED"), band);
    end endtask

    initial begin
        errors=0; test_count=0;
        d2_start=0; d3_start=0; d4_start=0; r1_start=0;
        d2_pos=0; d2_pos_vec=0; d2_slen=0; d2_slen_vec=0; d2_xvec=0;
        d3_pos=0; d3_pos_vec=0; d3_slen=0; d3_slen_vec=0; d3_xvec=0;
        d4_pos=0; d4_pos_vec=0; d4_slen=0; d4_slen_vec=0; d4_xvec=0;
        r1_pos=0; r1_slen=0; r1_xvec=0;
        @(negedge clk); rst=1'b0; @(negedge clk);

        // ---------------- B=2 ----------------
        run_case(2,   11, 0, 1, "B2_dense_p1");     // extents 1,2   (dense)
        run_case(2,  123, 0, 2, "B2_dense_p2");     // extents 2,3   (dense)
        run_case(2,  321, 1, 4, "B2_mixed_p4");     // extents 4,5   (dense row0, sparse row1)
        run_case(2,  909, 0, 5, "B2_sparse_p5");    // extents 5,6   (sparse)

        // ---------------- B=3 ----------------
        run_case(3,   55, 0, 1, "B3_dense_p1");     // extents 1,2,3 (dense)
        run_case(3, 1010, 1, 3, "B3_mixed_p3");     // extents 3,4,5 (dense,dense,sparse)
        run_case(3,  202, 0, 5, "B3_sparse_p5");    // extents 5,6,7 (sparse)

        // ---------------- B=4 ----------------
        run_case(4,  777, 0, 1, "B4_dense_p1");     // extents 1,2,3,4 (dense)
        run_case(4, 1313, 1, 3, "B4_mixed_p3");     // extents 3,4,5,6 (mixed)
        run_case(4,  500, 0, 5, "B4_sparse_p5");    // extents 5,6,7,8 (sparse; row3 all keys)

        if (errors==0) $display("ALL %0d TESTS PASSED", test_count);
        else begin
            $display("FAILED: %0d errors over %0d checks", errors, test_count);
            $fatal(1, "mla_attn_q4k intra-causal leaf oracle failed");
        end
        $finish;
    end

    initial begin
        #2000000000;
        $display("FAIL: timeout (done never asserted)");
        $fatal(1, "timeout");
    end
endmodule
