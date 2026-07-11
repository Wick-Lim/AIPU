`timescale 1ns/1ps
//============================================================================
// mla_attn_q4k_sparse_perrow_tb.v
//   PE_M batching oracle for mla_attn_q4k (the Q4_K PRODUCT attention) that
//   combines SPARSE DSA selection (S_MAX > TOPK) with PER-ROW query position,
//   PER-ROW causal extent AND PER-ROW KV windows (PER_ROW_SEQ).
//----------------------------------------------------------------------------
// WHAT THIS PROVES (a DUT-vs-DUT, same-arithmetic, EXACT === oracle -- the
// Q4_K sibling of the fp8 track's mla_attn_fp8_sparse_perrow_tb.v):
//
//   (1) PER-ROW BIT-EXACT EQUIVALENCE -- one batched DUT (PE_M=B=3,
//       PER_ROW_POS=1, PER_ROW_SLEN=1, DSA_REAL_IDX=1) against, for the SAME
//       shared Q4_K weight ROMs / d,dmin,scales / KV cache, ONE PE_M=1
//       reference re-run per row on THAT row's own (x_r, pos_r, s_len_r):
//       batched row r's MODEL_DIM bf16 `out` === the PE_M=1 run (X/Z-aware,
//       exact ===).  Covered regimes:
//         * DENSE fallback (max extent <= TOPK): per-row extents, per-row pos.
//         * SPARSE (max extent > TOPK), ALL-EQUAL-x shared extent.
//         * SPARSE, DISTINCT-x, per-row extents/pos (the B6 per-row
//           q-DEPENDENT DSA selection, asserted UNCONDITIONALLY -- on main the
//           per-row dsa_indexer re-run has LANDED, so batched row r must be
//           bit-exact to its standalone decode even when rows select
//           DIFFERENT key sets; a hierarchical probe of sel_list_r/sel_cnt_r
//           additionally asserts the selections really DO diverge, i.e. the
//           divergent path is LIVE, not vacuously folded).
//   (2) FETCH SHARING -- the batch fetches each distinct weight/key ONCE:
//         * shared-selection regimes: batched w_req/kc_req beat counts ===
//           ONE PE_M=1 run covering the shared max extent (EXACT equality);
//         * divergent-selection regime: max_single <= batch < sum_of_rows
//           (union fetched once -- strictly cheaper than B standalone runs).
//   (3) DENSE-vs-SPARSE CROSS-CHECK -- when the sparse machine's selection
//       covers the FULL causal window (extent <= TOPK: dense fallback keeps
//       keys 0..S-1), its output must be BIT-IDENTICAL to a genuinely DENSE
//       machine (TOPK=S_MAX=8, never sparse, DSA_REAL_IDX=0) on the same
//       inputs -- pinning that the DSA gather machinery and the SWIN-sized
//       scratch padding are numeric no-ops when nothing is actually dropped.
//   (4) PER-ROW KV WINDOWS (PER_ROW_SEQ=1) -- a second batched DUT where each
//       row r attends its OWN sequence window (seq_vec={2,1,0}, kc_seq routes
//       each key fetch): batched row r === a PE_M=1 run served ONLY row r's
//       window, dense AND sparse extents; kc beats == SUM of the rows' own
//       fetches (windows share no key) while w beats stay SHARED (< sum).
//       Per the RTL contract PER_ROW_SEQ requires DSA_REAL_IDX=0,
//       S_MAX >= PE_M*TOPK (union depth) and SWIN >= PE_M*TOPK (scratch),
//       so the multi-seq instances run TOPK_SEQ=2 (3*2=6 <= S_MAX=8).
//
// NOT COVERED HERE (precisely):
//   * PER_ROW_SEQ combined with DSA_REAL_IDX=1 (q-dependent selection across
//     per-row windows) -- the RTL itself excludes it ("for now" the kidx_buf
//     prefetch is not per-seq); nothing to test until that lands.
//   * numeric-vs-numpy goldens -- this TB is a batching/selection ORACLE
//     (DUT-vs-DUT ===); the Q4_K numeric path itself is golden-gated by
//     glm_matmul_q4k_tb / glm_model_q4k_full_tb.
//
//   S_MAX=8, TOPK=4  ->  extent 1..4 is dense, 5..8 is sparse.
//
//   Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X.
//============================================================================
module mla_attn_q4k_sparse_perrow_tb;
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

    // ---- derived sizing identical to the DUT (for w_q / w_d / w_scales) ----
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
    localparam integer NSB   = (KMAX + 255)/256;   // Q4_K super-blocks along K (=1 here)

    localparam integer PE_M  = 3;
    localparam integer SEQW  = (PE_M<=1)?1:$clog2(PE_M);

    // batched sparse DUT: per-row q-DEPENDENT DSA selection (DSA_REAL_IDX=1) can
    // select up to min(PE_M*TOPK, S_MAX) DISTINCT union keys -> size SWIN for it.
    localparam integer SWIN_TB = (PE_M*TOPK < S_MAX) ? PE_M*TOPK : S_MAX;   // = 8
    // PER_ROW_SEQ DUT: rows never share a key -> worst-case no-dedup union.
    // The RTL contract REQUIRES S_MAX >= PE_M*TOPK (union depth) and
    // SWIN >= PE_M*TOPK (scratch), so the multi-seq instances run a SMALLER
    // top-K budget: TOPK_SEQ=2 -> PE_M*TOPK_SEQ = 6 <= S_MAX = 8.
    localparam integer TOPK_SEQ = 2;
    localparam integer SWIN_SEQ = PE_M*TOPK_SEQ;                            // = 6

    // ---- Q4_K super-block scale/min headers (fp16), shared by every matrix ----
    //   d = 2^-8, dmin = 2^-5: w = (d*sc)*q - (dmin*m), sc,m in 0..63, q in 0..15
    //   -> weights land in a tame, MIXED-SIGN band (~[-2,+2] typical).
    localparam [15:0] D_FP16  = 16'h1C00;
    localparam [15:0] DM_FP16 = 16'h2800;

    // ---- clock / reset ----
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
    // per-COLUMN 96-bit packed 6-bit (sc,m) super-block scales (any bit pattern
    // decodes to valid 0..63 fields -> deterministic hashed values are fine).
    reg [95:0] Sc_dq  [0:Q_LORA-1];
    reg [95:0] Sc_uq  [0:HQK-1];
    reg [95:0] Sc_dkv [0:KV_LORA-1];
    reg [95:0] Sc_kr  [0:ROPE-1];
    reg [95:0] Sc_uk  [0:HNOPE-1];
    reg [95:0] Sc_uv  [0:HV-1];
    reg [95:0] Sc_o   [0:MODEL_DIM-1];
    // PE_M per-sequence KV windows: window w's cached latents / roped keys.
    //   The shared-context instances (dutM/ref1/refD) always read window 0.
    reg [15:0] CKV3 [0:PE_M-1][0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP3 [0:PE_M-1][0:S_MAX-1][0:ROPE-1];
    reg [15:0] xr   [0:PE_M-1][0:MODEL_DIM-1];   // per-row token activations

    // deterministic stimulus generators (same hashing style as committed TBs).
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

    // stim_bump: toggled after every build_stimulus so the responder blocks
    //   re-evaluate.  (The responders call FUNCTIONS that read the stimulus
    //   arrays; @* cannot see array references hidden inside a function, so
    //   without this the value computed at t=0 -- all-X arrays -- would be
    //   served stale for any fetch whose address never changes, e.g. key 0.)
    reg stim_bump = 1'b0;

    integer ii,jj,kk,sc;
    task build_stimulus; input integer seed0; input integer band; begin
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
        // DISTINCT per-row token activations (rows differ in x by default).
        for (kk=0;kk<PE_M;kk=kk+1)
            for (ii=0;ii<MODEL_DIM;ii=ii+1) begin xr[kk][ii]=gen_bf16(sc,band); sc=sc+1; end
        // PE_M DISTINCT KV windows (shared-context cases read window 0 only).
        for (kk=0;kk<PE_M;kk=kk+1) begin
            for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<KV_LORA;jj=jj+1) begin CKV3[kk][ii][jj]=gen_bf16(sc,band); sc=sc+1; end
            for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<ROPE;jj=jj+1)    begin KRP3[kk][ii][jj]=gen_bf16(sc,band); sc=sc+1; end
        end
        stim_bump = ~stim_bump;    // force the responder blocks to re-evaluate
    end endtask

    // force ALL rows to share row-0's x (all-equal-x regime).
    task equalize_x; integer r,c; begin
        for (r=1;r<PE_M;r=r+1) for (c=0;c<MODEL_DIM;c=c+1) xr[r][c]=xr[0][c];
    end endtask

    // ================= shared Q4_K weight responder (pure function) ============
    //   returns {w_scales, w_dmin, w_d, w_q} for a (w_sel, w_grp, w_k) beat.
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
    //   returns {kc_krope, kc_ckv} of window `sq`, key `idx`.
    localparam integer KCRW = (ROPE + KV_LORA)*16;
    function [KCRW-1:0] kc_resp; input integer sq; input [IDXW-1:0] idx;
        reg [KV_LORA*16-1:0] cv; reg [ROPE*16-1:0] kr; integer c; begin
        cv = {KV_LORA*16{1'b0}}; kr = {ROPE*16{1'b0}};
        for (c=0;c<KV_LORA;c=c+1) cv[16*c +:16] = CKV3[sq][idx][c];
        for (c=0;c<ROPE;c=c+1)    kr[16*c +:16] = KRP3[sq][idx][c];
        kc_resp = {kr, cv};
    end endfunction

    // ===========================================================================
    //  BATCHED DUT : PE_M=3, per-row pos + per-row extent + per-row DSA (B6).
    // ===========================================================================
    reg                      d_start;
    wire                     d_busy, d_done;
    reg  [POSW-1:0]          d_pos;
    reg  [POSW*PE_M-1:0]     d_pos_vec;
    reg  [IDXW:0]            d_slen;
    reg  [(IDXW+1)*PE_M-1:0] d_slen_vec;
    reg  [MODEL_DIM*16*PE_M-1:0] d_xvec;
    wire [MODEL_DIM*16*PE_M-1:0] d_out;
    wire                     d_wreq; wire [3:0] d_wsel;
    wire [GRPW-1:0]          d_wgrp; wire [KCW-1:0] d_wk;
    reg  [PE_N*4-1:0]        d_wq;
    reg  [16*PE_N*NSB-1:0]   d_wd, d_wdmin;
    reg  [96*PE_N*NSB-1:0]   d_wscales;
    wire                     d_kcreq; wire [IDXW-1:0] d_kcidx; wire [SEQW-1:0] d_kcseq;
    reg  [KV_LORA*16-1:0]    d_kcckv; reg [ROPE*16-1:0] d_kckrope; reg d_kcvalid;

    mla_attn_q4k #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(PE_M), .SWIN(SWIN_TB),
               .PER_ROW_POS(1), .PER_ROW_SLEN(1), .DSA_REAL_IDX(1)) dutM (
        .clk(clk), .rst(rst), .start(d_start), .busy(d_busy), .done(d_done),
        .pos(d_pos), .pos_vec(d_pos_vec), .s_len(d_slen), .s_len_vec(d_slen_vec),
        .seq_vec({SEQW*PE_M{1'b0}}), .x_vec(d_xvec),
        .w_req(d_wreq), .w_sel(d_wsel), .w_grp(d_wgrp), .w_k(d_wk),
        .w_q(d_wq), .w_d(d_wd), .w_dmin(d_wdmin), .w_scales(d_wscales),
        .kc_req(d_kcreq), .kc_idx(d_kcidx), .kc_seq(d_kcseq),
        .kc_ckv(d_kcckv), .kc_krope(d_kckrope), .kc_valid(d_kcvalid), .out(d_out)
    );

    // ===========================================================================
    //  PE_M=1 SPARSE REFERENCE (same machine; re-run once per row on row inputs)
    // ===========================================================================
    reg                      r1_start;
    wire                     r1_busy, r1_done;
    reg  [POSW-1:0]          r1_pos;
    reg  [IDXW:0]            r1_slen;
    reg  [MODEL_DIM*16-1:0]  r1_xvec;
    wire [MODEL_DIM*16-1:0]  r1_out;
    wire                     r1_wreq; wire [3:0] r1_wsel;
    wire [GRPW-1:0]          r1_wgrp; wire [KCW-1:0] r1_wk;
    reg  [PE_N*4-1:0]        r1_wq;
    reg  [16*PE_N*NSB-1:0]   r1_wd, r1_wdmin;
    reg  [96*PE_N*NSB-1:0]   r1_wscales;
    wire                     r1_kcreq; wire [IDXW-1:0] r1_kcidx; wire r1_kcseq;
    reg  [KV_LORA*16-1:0]    r1_kcckv; reg [ROPE*16-1:0] r1_kckrope; reg r1_kcvalid;

    mla_attn_q4k #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(1), .SWIN(SWIN_TB),
               .DSA_REAL_IDX(1)) ref1 (
        .clk(clk), .rst(rst), .start(r1_start), .busy(r1_busy), .done(r1_done),
        .pos(r1_pos), .pos_vec(r1_pos), .s_len(r1_slen), .s_len_vec(r1_slen),
        .seq_vec(1'b0), .x_vec(r1_xvec),
        .w_req(r1_wreq), .w_sel(r1_wsel), .w_grp(r1_wgrp), .w_k(r1_wk),
        .w_q(r1_wq), .w_d(r1_wd), .w_dmin(r1_wdmin), .w_scales(r1_wscales),
        .kc_req(r1_kcreq), .kc_idx(r1_kcidx), .kc_seq(r1_kcseq),
        .kc_ckv(r1_kcckv), .kc_krope(r1_kckrope), .kc_valid(r1_kcvalid), .out(r1_out)
    );

    // ===========================================================================
    //  PE_M=1 DENSE ORACLE (TOPK=S_MAX=8: never sparse, attends ALL keys 0..S-1)
    // ===========================================================================
    reg                      rd_start;
    wire                     rd_busy, rd_done;
    reg  [POSW-1:0]          rd_pos;
    reg  [IDXW:0]            rd_slen;
    reg  [MODEL_DIM*16-1:0]  rd_xvec;
    wire [MODEL_DIM*16-1:0]  rd_out;
    wire                     rd_wreq; wire [3:0] rd_wsel;
    wire [GRPW-1:0]          rd_wgrp; wire [KCW-1:0] rd_wk;
    reg  [PE_N*4-1:0]        rd_wq;
    reg  [16*PE_N*NSB-1:0]   rd_wd, rd_wdmin;
    reg  [96*PE_N*NSB-1:0]   rd_wscales;
    wire                     rd_kcreq; wire [IDXW-1:0] rd_kcidx; wire rd_kcseq;
    reg  [KV_LORA*16-1:0]    rd_kcckv; reg [ROPE*16-1:0] rd_kckrope; reg rd_kcvalid;

    mla_attn_q4k #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(S_MAX), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(1), .SWIN(SWIN_TB),
               .DSA_REAL_IDX(0)) refD (
        .clk(clk), .rst(rst), .start(rd_start), .busy(rd_busy), .done(rd_done),
        .pos(rd_pos), .pos_vec(rd_pos), .s_len(rd_slen), .s_len_vec(rd_slen),
        .seq_vec(1'b0), .x_vec(rd_xvec),
        .w_req(rd_wreq), .w_sel(rd_wsel), .w_grp(rd_wgrp), .w_k(rd_wk),
        .w_q(rd_wq), .w_d(rd_wd), .w_dmin(rd_wdmin), .w_scales(rd_wscales),
        .kc_req(rd_kcreq), .kc_idx(rd_kcidx), .kc_seq(rd_kcseq),
        .kc_ckv(rd_kcckv), .kc_krope(rd_kckrope), .kc_valid(rd_kcvalid), .out(rd_out)
    );

    // ===========================================================================
    //  BATCHED MULTI-SEQUENCE DUT : PER_ROW_SEQ=1 (per-row KV windows).
    //    RTL contract: DSA_REAL_IDX=0 and SWIN >= PE_M*TOPK.
    // ===========================================================================
    reg                      s_start;
    wire                     s_busy, s_done;
    reg  [POSW-1:0]          s_pos;
    reg  [POSW*PE_M-1:0]     s_pos_vec;
    reg  [IDXW:0]            s_slen;
    reg  [(IDXW+1)*PE_M-1:0] s_slen_vec;
    reg  [MODEL_DIM*16*PE_M-1:0] s_xvec;
    wire [MODEL_DIM*16*PE_M-1:0] s_out;
    wire                     s_wreq; wire [3:0] s_wsel;
    wire [GRPW-1:0]          s_wgrp; wire [KCW-1:0] s_wk;
    reg  [PE_N*4-1:0]        s_wq;
    reg  [16*PE_N*NSB-1:0]   s_wd, s_wdmin;
    reg  [96*PE_N*NSB-1:0]   s_wscales;
    wire                     s_kcreq; wire [IDXW-1:0] s_kcidx; wire [SEQW-1:0] s_kcseq;
    reg  [KV_LORA*16-1:0]    s_kcckv; reg [ROPE*16-1:0] s_kckrope; reg s_kcvalid;

    mla_attn_q4k #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK_SEQ), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(PE_M), .SWIN(SWIN_SEQ),
               .PER_ROW_POS(1), .PER_ROW_SLEN(1), .PER_ROW_SEQ(1),
               .DSA_REAL_IDX(0)) dutS (
        .clk(clk), .rst(rst), .start(s_start), .busy(s_busy), .done(s_done),
        .pos(s_pos), .pos_vec(s_pos_vec), .s_len(s_slen), .s_len_vec(s_slen_vec),
        .seq_vec({2'd2, 2'd1, 2'd0}), .x_vec(s_xvec),
        .w_req(s_wreq), .w_sel(s_wsel), .w_grp(s_wgrp), .w_k(s_wk),
        .w_q(s_wq), .w_d(s_wd), .w_dmin(s_wdmin), .w_scales(s_wscales),
        .kc_req(s_kcreq), .kc_idx(s_kcidx), .kc_seq(s_kcseq),
        .kc_ckv(s_kcckv), .kc_krope(s_kckrope), .kc_valid(s_kcvalid), .out(s_out)
    );

    // ===========================================================================
    //  PE_M=1 PER-WINDOW REFERENCE for dutS (same TOPK/SWIN/DSA config; the TB
    //    reg ref_seq selects WHICH window its cache responder serves).
    // ===========================================================================
    reg                      rs_start;
    wire                     rs_busy, rs_done;
    reg  [POSW-1:0]          rs_pos;
    reg  [IDXW:0]            rs_slen;
    reg  [MODEL_DIM*16-1:0]  rs_xvec;
    wire [MODEL_DIM*16-1:0]  rs_out;
    wire                     rs_wreq; wire [3:0] rs_wsel;
    wire [GRPW-1:0]          rs_wgrp; wire [KCW-1:0] rs_wk;
    reg  [PE_N*4-1:0]        rs_wq;
    reg  [16*PE_N*NSB-1:0]   rs_wd, rs_wdmin;
    reg  [96*PE_N*NSB-1:0]   rs_wscales;
    wire                     rs_kcreq; wire [IDXW-1:0] rs_kcidx; wire rs_kcseq;
    reg  [KV_LORA*16-1:0]    rs_kcckv; reg [ROPE*16-1:0] rs_kckrope; reg rs_kcvalid;
    integer ref_seq;

    mla_attn_q4k #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK_SEQ), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(1), .SWIN(SWIN_SEQ),
               .DSA_REAL_IDX(0)) refS (
        .clk(clk), .rst(rst), .start(rs_start), .busy(rs_busy), .done(rs_done),
        .pos(rs_pos), .pos_vec(rs_pos), .s_len(rs_slen), .s_len_vec(rs_slen),
        .seq_vec(1'b0), .x_vec(rs_xvec),
        .w_req(rs_wreq), .w_sel(rs_wsel), .w_grp(rs_wgrp), .w_k(rs_wk),
        .w_q(rs_wq), .w_d(rs_wd), .w_dmin(rs_wdmin), .w_scales(rs_wscales),
        .kc_req(rs_kcreq), .kc_idx(rs_kcidx), .kc_seq(rs_kcseq),
        .kc_ckv(rs_kcckv), .kc_krope(rs_kckrope), .kc_valid(rs_kcvalid), .out(rs_out)
    );

    // ================= combinational responders (one per instance) =============
    //   EXPLICIT sensitivity (ports + stim_bump), NOT @*: the stimulus arrays are
    //   read inside w_resp/kc_resp and @* cannot see through a function call, so
    //   each block must also wake on stim_bump (arrays only change in
    //   build_stimulus, which toggles it).
    always @(d_wsel  or d_wgrp  or d_wk  or stim_bump) {d_wscales,  d_wdmin,  d_wd,  d_wq}  = w_resp(d_wsel,  d_wgrp,  d_wk);
    always @(r1_wsel or r1_wgrp or r1_wk or stim_bump) {r1_wscales, r1_wdmin, r1_wd, r1_wq} = w_resp(r1_wsel, r1_wgrp, r1_wk);
    always @(rd_wsel or rd_wgrp or rd_wk or stim_bump) {rd_wscales, rd_wdmin, rd_wd, rd_wq} = w_resp(rd_wsel, rd_wgrp, rd_wk);
    always @(s_wsel  or s_wgrp  or s_wk  or stim_bump) {s_wscales,  s_wdmin,  s_wd,  s_wq}  = w_resp(s_wsel,  s_wgrp,  s_wk);
    always @(rs_wsel or rs_wgrp or rs_wk or stim_bump) {rs_wscales, rs_wdmin, rs_wd, rs_wq} = w_resp(rs_wsel, rs_wgrp, rs_wk);

    // shared-context instances read window 0; dutS routes by its kc_seq output;
    // refS serves the TB-selected window ref_seq.
    always @(d_kcidx  or stim_bump)             {d_kckrope,  d_kcckv}  = kc_resp(0,       d_kcidx);
    always @(r1_kcidx or stim_bump)             {r1_kckrope, r1_kcckv} = kc_resp(0,       r1_kcidx);
    always @(rd_kcidx or stim_bump)             {rd_kckrope, rd_kcckv} = kc_resp(0,       rd_kcidx);
    always @(s_kcidx  or s_kcseq or stim_bump)  {s_kckrope,  s_kcckv}  = kc_resp(s_kcseq, s_kcidx);
    always @(rs_kcidx or ref_seq or stim_bump)  {rs_kckrope, rs_kcckv} = kc_resp(ref_seq, rs_kcidx);

    // answer every cache pull with kc_valid one cycle after kc_req (registered).
    always @(posedge clk) begin
        if (rst) begin
            d_kcvalid<=1'b0; r1_kcvalid<=1'b0; rd_kcvalid<=1'b0;
            s_kcvalid<=1'b0; rs_kcvalid<=1'b0;
        end else begin
            d_kcvalid<=d_kcreq; r1_kcvalid<=r1_kcreq; rd_kcvalid<=rd_kcreq;
            s_kcvalid<=s_kcreq; rs_kcvalid<=rs_kcreq;
        end
    end

    // ================= fetch-beat counters (w_req / kc_req cycles) =============
    reg [31:0] bw_cnt, bkc_cnt, rw_cnt, rkc_cnt, b2w_cnt, b2kc_cnt, rsw_cnt, rskc_cnt;
    always @(posedge clk) begin
        if (d_start)  begin bw_cnt<=0;  bkc_cnt<=0;  end
        else          begin if (d_wreq)  bw_cnt <=bw_cnt +1'b1; if (d_kcreq)  bkc_cnt <=bkc_cnt +1'b1; end
        if (r1_start) begin rw_cnt<=0;  rkc_cnt<=0;  end
        else          begin if (r1_wreq) rw_cnt <=rw_cnt +1'b1; if (r1_kcreq) rkc_cnt <=rkc_cnt +1'b1; end
        if (s_start)  begin b2w_cnt<=0; b2kc_cnt<=0; end
        else          begin if (s_wreq)  b2w_cnt<=b2w_cnt+1'b1; if (s_kcreq)  b2kc_cnt<=b2kc_cnt+1'b1; end
        if (rs_start) begin rsw_cnt<=0; rskc_cnt<=0; end
        else          begin if (rs_wreq) rsw_cnt<=rsw_cnt+1'b1; if (rs_kcreq) rskc_cnt<=rskc_cnt+1'b1; end
    end

    // ===========================================================================
    //  DRIVERS
    // ===========================================================================
    integer i;
    // batched shared-context run: per-row pos p0..p2, per-row extents s0..s2.
    reg [31:0] bw_run, bkc_run;
    task run_dutM; input integer p0; input integer p1; input integer p2;
                   input integer s0; input integer s1; input integer s2; begin
        d_pos     = p0[POSW-1:0];                          // row 0 uses scalar pos
        d_pos_vec[POSW*0 +: POSW] = p0[POSW-1:0];          // row-0 slice unused
        d_pos_vec[POSW*1 +: POSW] = p1[POSW-1:0];
        d_pos_vec[POSW*2 +: POSW] = p2[POSW-1:0];
        d_slen    = s0[IDXW:0];                            // row 0 uses scalar s_len
        d_slen_vec[(IDXW+1)*0 +: (IDXW+1)] = s0[IDXW:0];   // row-0 slice unused
        d_slen_vec[(IDXW+1)*1 +: (IDXW+1)] = s1[IDXW:0];
        d_slen_vec[(IDXW+1)*2 +: (IDXW+1)] = s2[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) begin
            d_xvec[16*(MODEL_DIM*0 + i) +:16] = xr[0][i];
            d_xvec[16*(MODEL_DIM*1 + i) +:16] = xr[1][i];
            d_xvec[16*(MODEL_DIM*2 + i) +:16] = xr[2][i];
        end
        @(negedge clk); d_start=1'b1; @(negedge clk); d_start=1'b0;
        wait (d_done==1'b1); @(negedge clk);
        bw_run = bw_cnt; bkc_run = bkc_cnt;
    end endtask

    // batched multi-sequence run (dutS): row r attends window r (seq_vec={2,1,0}).
    reg [31:0] b2w_run, b2kc_run;
    task run_dutS; input integer p0; input integer p1; input integer p2;
                   input integer s0; input integer s1; input integer s2; begin
        s_pos     = p0[POSW-1:0];
        s_pos_vec[POSW*0 +: POSW] = p0[POSW-1:0];
        s_pos_vec[POSW*1 +: POSW] = p1[POSW-1:0];
        s_pos_vec[POSW*2 +: POSW] = p2[POSW-1:0];
        s_slen    = s0[IDXW:0];
        s_slen_vec[(IDXW+1)*0 +: (IDXW+1)] = s0[IDXW:0];
        s_slen_vec[(IDXW+1)*1 +: (IDXW+1)] = s1[IDXW:0];
        s_slen_vec[(IDXW+1)*2 +: (IDXW+1)] = s2[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) begin
            s_xvec[16*(MODEL_DIM*0 + i) +:16] = xr[0][i];
            s_xvec[16*(MODEL_DIM*1 + i) +:16] = xr[1][i];
            s_xvec[16*(MODEL_DIM*2 + i) +:16] = xr[2][i];
        end
        @(negedge clk); s_start=1'b1; @(negedge clk); s_start=1'b0;
        wait (s_done==1'b1); @(negedge clk);
        b2w_run = b2w_cnt; b2kc_run = b2kc_cnt;
    end endtask

    // PE_M=1 sparse reference on one row's (xr[row], pos p, s_len s); window 0.
    reg [15:0] ro [0:MODEL_DIM-1];
    reg [31:0] rw_run, rkc_run;
    task run_ref; input integer row; input integer p; input integer s; begin
        r1_pos  = p[POSW-1:0];
        r1_slen = s[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) r1_xvec[16*i +:16] = xr[row][i];
        @(negedge clk); r1_start=1'b1; @(negedge clk); r1_start=1'b0;
        wait (r1_done==1'b1); @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) ro[i] = r1_out[16*i +:16];
        rw_run = rw_cnt; rkc_run = rkc_cnt;
    end endtask

    // PE_M=1 DENSE oracle run (TOPK=S_MAX machine); window 0.
    reg [15:0] rdo [0:MODEL_DIM-1];
    task run_refD; input integer row; input integer p; input integer s; begin
        rd_pos  = p[POSW-1:0];
        rd_slen = s[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) rd_xvec[16*i +:16] = xr[row][i];
        @(negedge clk); rd_start=1'b1; @(negedge clk); rd_start=1'b0;
        wait (rd_done==1'b1); @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) rdo[i] = rd_out[16*i +:16];
    end endtask

    // PE_M=1 per-window reference run (dutS oracle): serve window `sq` only.
    reg [15:0] rso [0:MODEL_DIM-1];
    reg [31:0] rsw_run, rskc_run;
    task run_refS; input integer row; input integer p; input integer s; input integer sq; begin
        ref_seq = sq;
        rs_pos  = p[POSW-1:0];
        rs_slen = s[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) rs_xvec[16*i +:16] = xr[row][i];
        @(negedge clk); rs_start=1'b1; @(negedge clk); rs_start=1'b0;
        wait (rs_done==1'b1); @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) rso[i] = rs_out[16*i +:16];
        rsw_run = rsw_cnt; rskc_run = rskc_cnt;
    end endtask

    // ===========================================================================
    //  CHECKS (exact, X-aware)
    // ===========================================================================
    integer errors, test_count, fails;
    // batched (dutM) row `row` === last-captured ref1 output ro[].
    task check_row; input integer row; input [256*8-1:0] label; begin
        fails=0;
        for (i=0;i<MODEL_DIM;i=i+1) begin
            if (^d_out[16*(MODEL_DIM*row+i) +:16] === 1'bx) begin
                $display("FAIL[%0s] row%0d out[%0d] X/Z", label, row, i);
                fails=fails+1;
            end else if (d_out[16*(MODEL_DIM*row+i) +:16] !== ro[i]) begin
                $display("FAIL[%0s] row%0d out[%0d] dut=%h ref=%h", label, row, i,
                         d_out[16*(MODEL_DIM*row+i) +:16], ro[i]);
                fails=fails+1;
            end
        end
        test_count=test_count+1;
        if (fails==0) $display("  PASS[%0s] row%0d (exact match)", label, row);
        else errors=errors+fails;
    end endtask

    // batched (dutS) row `row` === last-captured refS output rso[].
    task check_rowS; input integer row; input [256*8-1:0] label; begin
        fails=0;
        for (i=0;i<MODEL_DIM;i=i+1) begin
            if (^s_out[16*(MODEL_DIM*row+i) +:16] === 1'bx) begin
                $display("FAIL[%0s] seq-row%0d out[%0d] X/Z", label, row, i);
                fails=fails+1;
            end else if (s_out[16*(MODEL_DIM*row+i) +:16] !== rso[i]) begin
                $display("FAIL[%0s] seq-row%0d out[%0d] dut=%h ref=%h", label, row, i,
                         s_out[16*(MODEL_DIM*row+i) +:16], rso[i]);
                fails=fails+1;
            end
        end
        test_count=test_count+1;
        if (fails==0) $display("  PASS[%0s] seq-row%0d (exact match, own KV window)", label, row);
        else errors=errors+fails;
    end endtask

    // DENSE-vs-SPARSE cross: refD output rdo[] === ref1 output ro[] (both PE_M=1;
    // valid ONLY when the sparse machine's selection covers the full window S<=TOPK).
    task check_cross_dense; input integer row; input [256*8-1:0] label; begin
        fails=0;
        for (i=0;i<MODEL_DIM;i=i+1) begin
            if (^rdo[i] === 1'bx) begin
                $display("FAIL[%0s] cross row%0d out[%0d] X/Z (dense oracle)", label, row, i);
                fails=fails+1;
            end else if (rdo[i] !== ro[i]) begin
                $display("FAIL[%0s] cross row%0d out[%0d] sparse=%h dense=%h", label, row, i,
                         ro[i], rdo[i]);
                fails=fails+1;
            end
        end
        test_count=test_count+1;
        if (fails==0) $display("  PASS[%0s] row%0d sparse(TOPK=%0d) === dense(TOPK=%0d) full-window", label, row, TOPK, S_MAX);
        else errors=errors+fails;
    end endtask

    // FETCH SHARING (shared-selection regimes): batched beat counts == ONE ref
    //   run over the shared max extent (rw_run/rkc_run captured at s=smax first).
    task check_fetch_share; input [256*8-1:0] label; begin
        test_count=test_count+1;
        if (bw_run !== rw_run) begin
            $display("FAIL[%0s] w_req beats: batch=%0d != 1-ref=%0d (weight fetch not shared)",
                     label, bw_run, rw_run);
            errors=errors+1;
        end else if (bkc_run !== rkc_run) begin
            $display("FAIL[%0s] kc_req beats: batch=%0d != 1-ref=%0d (key fetch not shared)",
                     label, bkc_run, rkc_run);
            errors=errors+1;
        end else
            $display("  PASS[%0s] fetch-share: w=%0d kc=%0d (one fetch per distinct key/weight)",
                     label, bw_run, bkc_run);
    end endtask

    function integer imax3; input integer a; input integer b; input integer c;
        integer mx; begin mx=a; if (b>mx) mx=b; if (c>mx) mx=c; imax3=mx; end
    endfunction

    // FETCH SHARING under genuine per-row divergence: the batch fetches the
    //   UNION once -> max_single <= batch < sum_of_rows for both streams.
    reg [31:0] sxw [0:PE_M-1];
    reg [31:0] sxkc[0:PE_M-1];
    task check_fetch_share_union; input [256*8-1:0] label; begin
        test_count = test_count + 1;
        if (!(bw_run < (sxw[0]+sxw[1]+sxw[2]) && bw_run >= imax3(sxw[0],sxw[1],sxw[2]))) begin
            $display("FAIL[%0s] w_req beats: batch=%0d not in [%0d,%0d) (weight fetch not shared)",
                     label, bw_run, imax3(sxw[0],sxw[1],sxw[2]), sxw[0]+sxw[1]+sxw[2]);
            errors = errors + 1;
        end else if (!(bkc_run < (sxkc[0]+sxkc[1]+sxkc[2]) && bkc_run >= imax3(sxkc[0],sxkc[1],sxkc[2]))) begin
            $display("FAIL[%0s] kc_req beats: batch=%0d not in [%0d,%0d) (key fetch not shared)",
                     label, bkc_run, imax3(sxkc[0],sxkc[1],sxkc[2]), sxkc[0]+sxkc[1]+sxkc[2]);
            errors = errors + 1;
        end else
            $display("  PASS[%0s] fetch-share(union): w=%0d in [%0d,%0d)  kc=%0d in [%0d,%0d)",
                     label, bw_run, imax3(sxw[0],sxw[1],sxw[2]), sxw[0]+sxw[1]+sxw[2],
                     bkc_run, imax3(sxkc[0],sxkc[1],sxkc[2]), sxkc[0]+sxkc[1]+sxkc[2]);
    end endtask

    // PER_ROW_SEQ fetch discipline: rows share NO key (disjoint windows) -> the
    //   batch's kc beats == SUM of the rows' own fetches, while the weight
    //   stream stays SHARED: max_single <= batch_w < sum_of_rows.
    reg [31:0] ssw [0:PE_M-1];
    reg [31:0] sskc[0:PE_M-1];
    task check_fetch_seq; input [256*8-1:0] label; begin
        test_count = test_count + 1;
        if (b2kc_run !== (sskc[0]+sskc[1]+sskc[2])) begin
            $display("FAIL[%0s] kc_req beats: batch=%0d != sum-of-rows=%0d (disjoint windows)",
                     label, b2kc_run, sskc[0]+sskc[1]+sskc[2]);
            errors = errors + 1;
        end else if (!(b2w_run < (ssw[0]+ssw[1]+ssw[2]) && b2w_run >= imax3(ssw[0],ssw[1],ssw[2]))) begin
            $display("FAIL[%0s] w_req beats: batch=%0d not in [%0d,%0d) (weight fetch not shared)",
                     label, b2w_run, imax3(ssw[0],ssw[1],ssw[2]), ssw[0]+ssw[1]+ssw[2]);
            errors = errors + 1;
        end else
            $display("  PASS[%0s] fetch(seq): kc=%0d == sum %0d; w=%0d in [%0d,%0d)",
                     label, b2kc_run, sskc[0]+sskc[1]+sskc[2],
                     b2w_run, imax3(ssw[0],ssw[1],ssw[2]), ssw[0]+ssw[1]+ssw[2]);
    end endtask

    // DIVERGENCE PROOF: with real query-dependent index vectors the batched
    //   DUT's PER-ROW DSA selection sel_list_r[r] must actually DIFFER across
    //   rows whose queries differ -- i.e. the per-row B6 path is LIVE, and the
    //   bit-exact row compares above were NOT vacuous shared-selection folds.
    integer dvr, dvs; reg dv_diff;
    task check_divergence; input [256*8-1:0] label; begin
        test_count = test_count + 1;
        for (dvr=0; dvr<PE_M; dvr=dvr+1) begin
            $write("    [%0s] row%0d sel_cnt=%0d sel_list=", label, dvr, dutM.sel_cnt_r[dvr]);
            for (dvs=0; dvs<TOPK; dvs=dvs+1) $write(" %0d", dutM.sel_list_r[dvr][dvs]);
            $write("\n");
        end
        dv_diff = 1'b0;
        for (dvr=1; dvr<PE_M; dvr=dvr+1) begin
            if (dutM.sel_cnt_r[dvr] !== dutM.sel_cnt_r[0]) dv_diff = 1'b1;
            for (dvs=0; dvs<TOPK; dvs=dvs+1)
                if (dutM.sel_list_r[dvr][dvs] !== dutM.sel_list_r[0][dvs]) dv_diff = 1'b1;
        end
        if (dv_diff)
            $display("  PASS[%0s] per-row DSA selection is q-DEPENDENT (rows select DIFFERENT key sets)", label);
        else begin
            $display("FAIL[%0s] per-row DSA selection IDENTICAL across distinct-x rows (not q-dependent)", label);
            errors = errors + 1;
        end
    end endtask

    // ---------------------------------------------------------------------------
    //  CASE DRIVERS
    // ---------------------------------------------------------------------------
    integer smax_c;
    // shared-selection case: per-row bit-exact + EXACT fetch-share equality.
    task case_match; input integer seed0; input integer band; input integer eqx;
                     input integer p0; input integer p1; input integer p2;
                     input integer s0; input integer s1; input integer s2;
                     input [256*8-1:0] label; begin
        build_stimulus(seed0, band);
        if (eqx) equalize_x;
        smax_c = imax3(s0,s1,s2);
        run_dutM(p0, p1, p2, s0, s1, s2);
        // (1) per-row bit-exact equivalence
        run_ref(0, p0, s0); check_row(0, label);
        run_ref(1, p1, s1); check_row(1, label);
        run_ref(2, p2, s2); check_row(2, label);
        // (2) fetch sharing vs ONE reference run covering the shared max extent
        run_ref(0, p0, smax_c);
        check_fetch_share(label);
        $display("    (%0s: pos=(%0d,%0d,%0d) S=(%0d,%0d,%0d) eqx=%0d band=%0d %s)",
                 label, p0, p1, p2, s0, s1, s2, eqx, band,
                 (smax_c<=TOPK)?"DENSE":"SPARSE");
    end endtask

    // DENSE-vs-SPARSE full-window cross case (all extents <= TOPK): batched rows
    //   === per-row sparse refs === per-row DENSE(TOPK=S_MAX) oracle, bitwise.
    task case_cross; input integer seed0; input integer band;
                     input integer p; input integer s0; input integer s1; input integer s2;
                     input [256*8-1:0] label; begin
        build_stimulus(seed0, band);
        smax_c = imax3(s0,s1,s2);
        run_dutM(p, p, p, s0, s1, s2);
        run_ref(0, p, s0); check_row(0, label); run_refD(0, p, s0); check_cross_dense(0, label);
        run_ref(1, p, s1); check_row(1, label); run_refD(1, p, s1); check_cross_dense(1, label);
        run_ref(2, p, s2); check_row(2, label); run_refD(2, p, s2); check_cross_dense(2, label);
        run_ref(0, p, smax_c);
        check_fetch_share(label);
        $display("    (%0s: pos=%0d S=(%0d,%0d,%0d) full-window cross)", label, p, s0, s1, s2);
    end endtask

    // distinct-x SPARSE case: per-row q-DEPENDENT selection (B6, LANDED on the
    //   Q4_K product) -- batched row r must STILL be bit-exact to its standalone
    //   run; selections must diverge; the union is fetched once (bounds).
    task case_sparse; input integer seed0; input integer band;
                      input integer p0; input integer p1; input integer p2;
                      input integer s0; input integer s1; input integer s2;
                      input [256*8-1:0] label; begin
        build_stimulus(seed0, band);           // distinct x (NOT equalized)
        run_dutM(p0, p1, p2, s0, s1, s2);
        // (0) PROVE the divergent path is live: per-row selection is q-dependent.
        check_divergence(label);
        // (1) per-row BIT-EXACT equivalence under divergent selections.
        run_ref(0, p0, s0); check_row(0, label); sxw[0]=rw_run; sxkc[0]=rkc_run;
        run_ref(1, p1, s1); check_row(1, label); sxw[1]=rw_run; sxkc[1]=rkc_run;
        run_ref(2, p2, s2); check_row(2, label); sxw[2]=rw_run; sxkc[2]=rkc_run;
        // (2) fetch-sharing under divergence (union fetched once; bounds vs 3 runs).
        check_fetch_share_union(label);
        $display("    (%0s: pos=(%0d,%0d,%0d) S=(%0d,%0d,%0d) distinct-x SPARSE)",
                 label, p0, p1, p2, s0, s1, s2);
    end endtask

    // PER_ROW_SEQ case: row r attends its OWN KV window r (distinct x/pos/extent).
    task case_seq; input integer seed0; input integer band;
                   input integer p0; input integer p1; input integer p2;
                   input integer s0; input integer s1; input integer s2;
                   input [256*8-1:0] label; begin
        build_stimulus(seed0, band);
        run_dutS(p0, p1, p2, s0, s1, s2);
        run_refS(0, p0, s0, 0); check_rowS(0, label); ssw[0]=rsw_run; sskc[0]=rskc_run;
        run_refS(1, p1, s1, 1); check_rowS(1, label); ssw[1]=rsw_run; sskc[1]=rskc_run;
        run_refS(2, p2, s2, 2); check_rowS(2, label); ssw[2]=rsw_run; sskc[2]=rskc_run;
        check_fetch_seq(label);
        $display("    (%0s: pos=(%0d,%0d,%0d) S=(%0d,%0d,%0d) PER_ROW_SEQ %s)",
                 label, p0, p1, p2, s0, s1, s2,
                 (imax3(s0,s1,s2)<=TOPK_SEQ)?"DENSE":"SPARSE");
    end endtask

    initial begin
        errors=0; test_count=0; ref_seq=0;
        d_start=1'b0; r1_start=1'b0; rd_start=1'b0; s_start=1'b0; rs_start=1'b0;
        d_pos=0; d_pos_vec=0; d_slen=0; d_slen_vec=0; d_xvec=0;
        r1_pos=0; r1_slen=0; r1_xvec=0;
        rd_pos=0; rd_slen=0; rd_xvec=0;
        s_pos=0; s_pos_vec=0; s_slen=0; s_slen_vec=0; s_xvec=0;
        rs_pos=0; rs_slen=0; rs_xvec=0;
        bw_cnt=0; bkc_cnt=0; rw_cnt=0; rkc_cnt=0;
        b2w_cnt=0; b2kc_cnt=0; rsw_cnt=0; rskc_cnt=0;
        @(negedge clk); rst=1'b0; @(negedge clk);

        // ---------------- shared-context batching oracle ----------------
        // A. DENSE fallback (max extent <= TOPK=4), DISTINCT x, SHARED pos/extent.
        case_match(  11, 0, 0,   7,  7,  7, 3, 3, 3, "denseA_distinctx_shared");
        case_match( 123, 0, 0,   0,  0,  0, 4, 4, 4, "denseB_distinctx_S4");
        // B. DENSE, DISTINCT x, PER-ROW DISTINCT extents (per-row causal mask).
        case_match(  55, 0, 0,   9,  9,  9, 2, 4, 3, "denseC_distinctx_perrowS");
        // C. DENSE, DISTINCT x, PER-ROW pos AND PER-ROW extent together.
        case_match(1010, 1, 0, 250, 13, 77, 3, 1, 4, "denseD_distinctx_perrow_pos_slen");
        // D. SPARSE (max extent > TOPK), ALL-EQUAL x, SHARED pos/extent -> exact.
        case_match( 321, 0, 1,   5,  5,  5, 6, 6, 6, "sparseE_equalx_S6");
        case_match( 909, 1, 1, 100,100,100, 8, 8, 8, "sparseF_equalx_S8");
        // E. PE_M=1-fold: ALL-EQUAL x, tiny SHARED dense extent (degenerate batch).
        case_match( 777, 0, 1,  42, 42, 42, 2, 2, 2, "foldG_equalx_S2");

        // ---------------- dense-vs-sparse full-window cross-check ----------------
        // extents 2,3,4 <= TOPK: the sparse machine keeps keys 0..S-1 (the FULL
        // window) -> must be bit-identical to the TOPK=S_MAX dense machine.
        case_cross( 500, 0,  13, 2, 3, 4, "crossH_fullwindow_dense_equiv");

        // ---------------- distinct-x SPARSE: per-row q-dependent DSA (B6) --------
        case_sparse( 202, 0,  63, 63, 63, 6, 5, 7, "sparseX_distinctx_S657");
        case_sparse(1313, 1,  17, 29, 41, 8, 6, 5, "sparseY_distinctx_perrowpos");

        // ---------------- PER_ROW_SEQ: per-row KV windows (TOPK_SEQ=2) ----------
        case_seq(2024, 0,   9, 21, 33, 2, 1, 2, "seqZ_dense_perrow_windows");
        case_seq(4242, 1,   5,  6,  7, 6, 5, 7, "seqW_sparse_perrow_windows");

        if (errors==0) begin
            $display("ALL %0d TESTS PASSED", test_count);
        end else begin
            $display("FAILED: %0d errors over %0d checks", errors, test_count);
            $fatal(1, "mla_attn_q4k sparse/per-row batching TB failed");
        end
        $finish;
    end

    initial begin
        #2000000000;
        $display("FAIL: timeout (done never asserted)");
        $fatal(1, "timeout");
    end
endmodule
