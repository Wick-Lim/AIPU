`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// mla_attn_fp8_multiseq_tb.v -- PER_ROW_SEQ (multi-sequence batched attention)
//   equivalence + weight-share TB for mla_attn_fp8 (GLM-5.2 MLA decode, FP8).
//----------------------------------------------------------------------------
// GOAL: prove PER_ROW_SEQ=1 batches B DIFFERENT sequences through ONE PE_M=B
//   wrapper such that:
//     (1) BIT-EXACT per-row equivalence -- row r's MODEL_DIM bf16 `out` equals an
//         independent PE_M=1 run on row r's own activation x AGAINST ROW r's OWN
//         KV WINDOW (seq r's c_kv/k_rope), routed via kc_seq.  Compared X-aware.
//     (2) SEQUENCE ROUTING -- two rows with the SAME activation x but DIFFERENT
//         sequences produce DIFFERENT outputs (each attends its own KV window),
//         and each still matches its own single-seq golden.
//     (3) WEIGHT-BW AMORTIZATION -- the batched PE_M=2 multi-seq run asserts w_req
//         on STRICTLY FEWER cycles than two independent PE_M=1 runs: the B
//         sequences SHARE the (seq-independent) query-side projection fetches
//         (W_dq/W_uq/W_dkv/W_kr/W_o).  This is the batching bandwidth win.
//
//   The KV cache responder is keyed by the DUT's kc_seq output -> CKV_ms[seq][idx]
//   / KRP_ms[seq][idx].  A wrong kc_seq would fetch the wrong window and (1)/(2)
//   would fail.  S<=TOPK (dense DSA fallback) so each row selects keys 0..S-1 of
//   ITS OWN sequence; the per-row-slot union tags each with its seq.
//
//   Row 0 is always sequence 0 (like pos row-0 = scalar pos); rows 1.. take their
//   seq_vec slice.  Requires SWIN >= PE_M*TOPK and S_MAX >= PE_M*TOPK (union room).
//
//   Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X.
//============================================================================
module mla_attn_fp8_multiseq_dsareal_tb;
    // ---- slice params (match mla_attn_fp8_pem_tb, but S_MAX/SWIN sized for 2 seqs) ----
    localparam integer MODEL_DIM = 16;
    localparam integer H_HEADS   = 2;
    localparam integer NOPE      = 4;
    localparam integer ROPE      = 4;
    localparam integer V_DIM     = 4;
    localparam integer Q_LORA    = 8;
    localparam integer KV_LORA   = 8;
    localparam integer TOPK      = 2;
    localparam integer PE_M      = 2;
    localparam integer NSEQ      = 2;                 // sequences in this TB
    localparam integer S_MAX     = PE_M*TOPK;         // >= union room (4)
    localparam integer SWIN_MS   = PE_M*TOPK;         // multi-seq scratch depth (4)
    localparam integer THETA     = 8000000;
    localparam integer PE_N      = 2;
    localparam integer POSW      = 20;
    localparam integer BLK       = 128;
    localparam integer QK_DIM    = NOPE + ROPE;
    localparam integer IDXW      = (S_MAX <= 1) ? 1 : $clog2(S_MAX);
    localparam integer SEQW      = (PE_M  <= 1) ? 1 : $clog2(PE_M);
    localparam integer HQK       = H_HEADS*QK_DIM;
    localparam integer HNOPE     = H_HEADS*NOPE;
    localparam integer HV        = H_HEADS*V_DIM;

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
    localparam integer NB    = (KMAX + BLK - 1)/BLK;

    `include "glm_fp.vh"
    `include "fp8_e4m3.vh"

    reg clk=1'b0, rst=1'b1;
    always #5 clk = ~clk;

    reg  [POSW-1:0] pos;
    reg  [IDXW:0]   s_len;

    // ================= shared weight ROMs (seq-INDEPENDENT) + block scales =========
    reg [7:0] W_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_uq  [0:HQK-1][0:Q_LORA-1];
    reg [7:0] W_dkv [0:KV_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_kr  [0:ROPE-1][0:MODEL_DIM-1];
    reg [7:0] W_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [7:0] W_uv  [0:HV-1][0:KV_LORA-1];
    reg [7:0] W_o   [0:MODEL_DIM-1][0:HV-1];
    reg [15:0] S_dq, S_uq, S_dkv, S_kr, S_uk, S_uv, S_o;
    // PER-SEQUENCE KV windows: CKV_ms[seq][key][d], KRP_ms[seq][key][d]
    reg [15:0] CKV_ms [0:NSEQ-1][0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP_ms [0:NSEQ-1][0:S_MAX-1][0:ROPE-1];

    reg [15:0] xrow  [0:1][0:MODEL_DIM-1];             // two token activations

    // ---- deterministic stimulus generators (same hashes as the pem TB) ----
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h = (seed*2654435761) ^ (seed<<13) ^ (seed*40503);
        s = h[3];
        if (band==1) e = 8'd125 + h[6:4];
        else         e = 8'd124 + h[5:4];
        m = h[12:6];
        gen_bf16 = {s, e, m};
    end endfunction
    function [7:0] gen_fp8; input integer seed; reg sg; reg [3:0] e; reg [2:0] m;
        integer h; begin
        h = (seed*2246822519) ^ (seed<<11) ^ (seed*3266489917);
        sg = h[2]; e = 4'd5 + h[4:3]; m = h[7:5];
        gen_fp8 = {sg, e, m};
    end endfunction

    integer i,j,sc,rw,sq;
    task build_stimulus; input integer seed0; input integer band; begin
        sc = seed0;
        for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[i][j]=gen_fp8(sc);  sc=sc+1; end
        S_dq=16'h3F80; S_uq=16'h3F00; S_dkv=16'h3F80; S_kr=16'h3F80;
        S_uk=16'h4000; S_uv=16'h3F00; S_o=16'h3F80;
        for (rw=0; rw<2; rw=rw+1)
            for (i=0;i<MODEL_DIM;i=i+1) begin xrow[rw][i]=gen_bf16(sc,band); sc=sc+1; end
        // DISTINCT KV per sequence (own seed band region) -> windows truly differ
        for (sq=0; sq<NSEQ; sq=sq+1) begin
            for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV_ms[sq][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP_ms[sq][i][j]=gen_bf16(sc,band); sc=sc+1; end
        end
    end endtask

    // ================= FP8 weight responder (macro, shared across seqs) =============
    `define WRESP(WSEL,WGRP,WK,WCOL,WSCALE)                                     \
        integer WCOL``_t; reg [15:0] WCOL``_ss;                                 \
        always @* begin                                                        \
            WCOL   = {PE_N*8{1'b0}};                                           \
            WSCALE = {16*PE_N*NB{1'b0}};                                       \
            case (WSEL)                                                        \
                4'd0: WCOL``_ss=S_dq;  4'd1: WCOL``_ss=S_uq;                   \
                4'd2: WCOL``_ss=S_dkv; 4'd3: WCOL``_ss=S_kr;                   \
                4'd4: WCOL``_ss=S_uk;  4'd5: WCOL``_ss=S_uv;                   \
                4'd6: WCOL``_ss=S_o;   default: WCOL``_ss=16'h3F80;            \
            endcase                                                            \
            for (WCOL``_t=0; WCOL``_t<PE_N; WCOL``_t=WCOL``_t+1) begin         \
                case (WSEL)                                                    \
                4'd0: if(WGRP*PE_N+WCOL``_t<Q_LORA)  WCOL[8*WCOL``_t+:8]=W_dq [WGRP*PE_N+WCOL``_t][WK]; \
                4'd1: if(WGRP*PE_N+WCOL``_t<HQK)     WCOL[8*WCOL``_t+:8]=W_uq [WGRP*PE_N+WCOL``_t][WK]; \
                4'd2: if(WGRP*PE_N+WCOL``_t<KV_LORA) WCOL[8*WCOL``_t+:8]=W_dkv[WGRP*PE_N+WCOL``_t][WK]; \
                4'd3: if(WGRP*PE_N+WCOL``_t<ROPE)    WCOL[8*WCOL``_t+:8]=W_kr [WGRP*PE_N+WCOL``_t][WK]; \
                4'd4: if(WGRP*PE_N+WCOL``_t<HNOPE)   WCOL[8*WCOL``_t+:8]=W_uk [WGRP*PE_N+WCOL``_t][WK]; \
                4'd5: if(WGRP*PE_N+WCOL``_t<HV)      WCOL[8*WCOL``_t+:8]=W_uv [WGRP*PE_N+WCOL``_t][WK]; \
                4'd6: if(WGRP*PE_N+WCOL``_t<MODEL_DIM)WCOL[8*WCOL``_t+:8]=W_o [WGRP*PE_N+WCOL``_t][WK]; \
                default: WCOL[8*WCOL``_t+:8]=8'h38;                            \
                endcase                                                        \
                WSCALE[16*WCOL``_t+:16]=WCOL``_ss;                             \
            end                                                                \
        end

    // ============ per-sequence cache responder (keyed by kc_seq) ============
    `define CRESP_MS(KSEQ,KIDX,KCKV,KKRP)                                       \
        integer KCKV``_d;                                                      \
        always @* begin                                                       \
            KCKV = {KV_LORA*16{1'b0}};  KKRP = {ROPE*16{1'b0}};               \
            for (KCKV``_d=0; KCKV``_d<KV_LORA; KCKV``_d=KCKV``_d+1)           \
                KCKV[16*KCKV``_d+:16]=CKV_ms[KSEQ][KIDX][KCKV``_d];           \
            for (KCKV``_d=0; KCKV``_d<ROPE; KCKV``_d=KCKV``_d+1)             \
                KKRP[16*KCKV``_d+:16]=KRP_ms[KSEQ][KIDX][KCKV``_d];           \
        end
    // golden PE_M=1 responder: fixed sequence g_seq (set per reference run)
    reg [SEQW-1:0] g_seq;
    `define CRESP_G(KIDX,KCKV,KKRP)                                            \
        integer KCKV``_d;                                                      \
        always @* begin                                                       \
            KCKV = {KV_LORA*16{1'b0}};  KKRP = {ROPE*16{1'b0}};               \
            for (KCKV``_d=0; KCKV``_d<KV_LORA; KCKV``_d=KCKV``_d+1)           \
                KCKV[16*KCKV``_d+:16]=CKV_ms[g_seq][KIDX][KCKV``_d];          \
            for (KCKV``_d=0; KCKV``_d<ROPE; KCKV``_d=KCKV``_d+1)             \
                KKRP[16*KCKV``_d+:16]=KRP_ms[g_seq][KIDX][KCKV``_d];          \
        end

    // ====================== DUT 1 : PE_M=1 golden (single seq g_seq) ======================
    reg                      start1;
    wire                     busy1, done1;
    reg  [MODEL_DIM*16-1:0]  x1;
    wire [MODEL_DIM*16-1:0]  out1;
    wire                     w_req1; wire [3:0] w_sel1; wire [GRPW-1:0] w_grp1;
    wire [KCW-1:0] w_k1; reg [PE_N*8-1:0] w_col1; reg [16*PE_N*NB-1:0] w_scale1;
    wire kc_req1; wire [IDXW-1:0] kc_idx1; wire [SEQW-1:0] kc_seq1;
    reg [KV_LORA*16-1:0] kc_ckv1; reg [ROPE*16-1:0] kc_krope1; reg kc_valid1;
    `WRESP(w_sel1,w_grp1,w_k1,w_col1,w_scale1)
    `CRESP_G(kc_idx1,kc_ckv1,kc_krope1)
    always @(posedge clk) if (rst) kc_valid1<=1'b0; else kc_valid1<=kc_req1;
    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM),.H_HEADS(H_HEADS),.NOPE(NOPE),.ROPE(ROPE),
        .V_DIM(V_DIM),.Q_LORA(Q_LORA),.KV_LORA(KV_LORA),.S_MAX(S_MAX),.TOPK(TOPK),
        .THETA(THETA),.PE_N(PE_N),.POSW(POSW),.BLK(BLK),.DSA_REAL_IDX(1),.PE_M(1)) dut1 (
        .clk(clk),.rst(rst),.start(start1),.busy(busy1),.done(done1),
        .pos(pos),.s_len(s_len),.x_vec(x1),
        .w_req(w_req1),.w_sel(w_sel1),.w_grp(w_grp1),.w_k(w_k1),.w_col(w_col1),.w_scale(w_scale1),
        .kc_req(kc_req1),.kc_idx(kc_idx1),.kc_seq(kc_seq1),.kc_ckv(kc_ckv1),.kc_krope(kc_krope1),.kc_valid(kc_valid1),
        .out(out1));

    // ====================== DUT MS : PE_M=2 multi-seq (PER_ROW_SEQ=1) ======================
    reg                       startM;
    wire                      busyM, doneM;
    reg  [MODEL_DIM*16*2-1:0] xM;
    reg  [SEQW*2-1:0]         seqM;             // per-row seq ids (row0 forced 0)
    wire [MODEL_DIM*16*2-1:0] outM;
    wire                      w_reqM; wire [3:0] w_selM; wire [GRPW-1:0] w_grpM;
    wire [KCW-1:0] w_kM; reg [PE_N*8-1:0] w_colM; reg [16*PE_N*NB-1:0] w_scaleM;
    wire kc_reqM; wire [IDXW-1:0] kc_idxM; wire [SEQW-1:0] kc_seqM;
    reg [KV_LORA*16-1:0] kc_ckvM; reg [ROPE*16-1:0] kc_kropeM; reg kc_validM;
    `WRESP(w_selM,w_grpM,w_kM,w_colM,w_scaleM)
    `CRESP_MS(kc_seqM,kc_idxM,kc_ckvM,kc_kropeM)     // <-- keyed by the DUT's kc_seq output
    always @(posedge clk) if (rst) kc_validM<=1'b0; else kc_validM<=kc_reqM;
    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM),.H_HEADS(H_HEADS),.NOPE(NOPE),.ROPE(ROPE),
        .V_DIM(V_DIM),.Q_LORA(Q_LORA),.KV_LORA(KV_LORA),.S_MAX(S_MAX),.TOPK(TOPK),
        .SWIN(SWIN_MS),.THETA(THETA),.PE_N(PE_N),.POSW(POSW),.BLK(BLK),.PE_M(2),
        .PER_ROW_SEQ(1),.DSA_REAL_IDX(1)) dutM (
        .clk(clk),.rst(rst),.start(startM),.busy(busyM),.done(doneM),
        .pos(pos),.s_len(s_len),.x_vec(xM),.seq_vec(seqM),
        .w_req(w_reqM),.w_sel(w_selM),.w_grp(w_grpM),.w_k(w_kM),.w_col(w_colM),.w_scale(w_scaleM),
        .kc_req(kc_reqM),.kc_idx(kc_idxM),.kc_seq(kc_seqM),.kc_ckv(kc_ckvM),.kc_krope(kc_kropeM),.kc_valid(kc_validM),
        .out(outM));

    // ================= w_req beat counters =================
    reg [31:0] cnt1, cntM, ref_sum;
    always @(posedge clk) begin
        if (start1) cnt1 <= 32'd0; else if (w_req1) cnt1 <= cnt1 + 1'b1;
        if (startM) cntM <= 32'd0; else if (w_reqM) cntM <= cntM + 1'b1;
    end

    integer errors, test_count, b;

    // capture a PE_M=1 golden (activation xrow[which] against sequence sqid) into gcap
    reg [MODEL_DIM*16-1:0] gcap;
    task cap_ref; input integer which; input integer sqid; begin
        @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) x1[16*i +:16]=xrow[which][i];
        g_seq = sqid[SEQW-1:0];
        start1=1'b1; @(negedge clk); start1=1'b0;
        wait (done1==1'b1); @(negedge clk);
        gcap = out1;
    end endtask

    task run_ms; input integer x0; input integer x1sel; input [SEQW*2-1:0] sv; begin
        @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) begin
            xM[16*(MODEL_DIM*0+i) +:16]=xrow[x0][i];
            xM[16*(MODEL_DIM*1+i) +:16]=xrow[x1sel][i];
        end
        seqM = sv;
        startM=1'b1; @(negedge clk); startM=1'b0;
        wait (doneM==1'b1); @(negedge clk);
    end endtask

    task cmp; input [255:0] label; input [MODEL_DIM*16-1:0] got, exp; begin
        for (b=0;b<MODEL_DIM*16;b=b+1)
            if (got[b]===1'bx || got[b]===1'bz) begin
                $display("FAIL[%0s]: out bit %0d is X/Z", label, b); errors=errors+1;
            end
        test_count=test_count+1;
        if (got !== exp) begin
            $display("FAIL[%0s]: got %h != exp %h", label, got, exp); errors=errors+1;
        end
    end endtask

    // assert two outputs DIFFER (sequence routing actually changed the result)
    task cmp_differ; input [255:0] label; input [MODEL_DIM*16-1:0] a, bb; begin
        test_count=test_count+1;
        if (a === bb) begin
            $display("FAIL[%0s]: outputs identical %h -- seq had no effect", label, a);
            errors=errors+1;
        end
    end endtask

    reg [MODEL_DIM*16-1:0] r0, r1, msr0, msr1;
    integer sd;
    task one_case; input integer seed0; input integer bnd; input integer pp; input integer ss; begin
        pos=pp[POSW-1:0]; s_len=ss[IDXW:0];
        build_stimulus(seed0, bnd);

        // ---- goldens: (x0,seq0), (x1,seq1), (x0,seq1) ----
        cap_ref(0,0); r0 = gcap;                     // x0 against seq0
        cap_ref(1,1); r1 = gcap;                     // x1 against seq1

        // ---- (1) BIT-EXACT: row0=(x0,seq0), row1=(x1,seq1) ----
        run_ms(0, 1, 2'b10);                         // seq_vec: row1 -> seq1
        cmp("row0==golden(x0,seq0)", outM[16*MODEL_DIM*0 +: 16*MODEL_DIM], r0);
        cmp("row1==golden(x1,seq1)", outM[16*MODEL_DIM*1 +: 16*MODEL_DIM], r1);
        // (3) weight amortization: batched beats < two separate single runs
        ref_sum = cnt1;                              // cnt1 holds the last golden (x1,seq1) run
        // (both golden runs have equal beats -> two-run total = 2*cnt1)
        if (!(cntM < (cnt1<<1))) begin
            $display("FAIL[wreq]: multi-seq beats %0d not < 2 single-run %0d", cntM, cnt1<<1);
            errors=errors+1;
        end
        test_count=test_count+1;

        // ---- (2) SEQUENCE ROUTING: SAME x for both rows, different seq ----
        cap_ref(0,0); msr0 = gcap;                   // x0 vs seq0
        cap_ref(0,1); msr1 = gcap;                   // x0 vs seq1
        run_ms(0, 0, 2'b10);                         // row0=(x0,seq0), row1=(x0,seq1)
        cmp("SR row0==golden(x0,seq0)", outM[16*MODEL_DIM*0 +: 16*MODEL_DIM], msr0);
        cmp("SR row1==golden(x0,seq1)", outM[16*MODEL_DIM*1 +: 16*MODEL_DIM], msr1);
        // rows must differ ONLY IF their per-seq goldens differ.  With the query-
        //   DEPENDENT DSA_REAL_IDX=1 selection two sequences can coincidentally
        //   yield the SAME result for a given x; the bit-exact per-seq checks above
        //   already prove routing, so only assert divergence when the goldens do.
        if (msr0 !== msr1)
            cmp_differ("SR rows differ (seq routed)",
                       outM[16*MODEL_DIM*0 +: 16*MODEL_DIM],
                       outM[16*MODEL_DIM*1 +: 16*MODEL_DIM]);

        // ---- same-seq under PER_ROW_SEQ=1 (both rows seq0; no dedup but correct) ----
        cap_ref(0,0); msr0 = gcap;
        cap_ref(1,0); msr1 = gcap;
        run_ms(0, 1, 2'b00);                         // row0=(x0,seq0), row1=(x1,seq0)
        cmp("SS row0==golden(x0,seq0)", outM[16*MODEL_DIM*0 +: 16*MODEL_DIM], msr0);
        cmp("SS row1==golden(x1,seq0)", outM[16*MODEL_DIM*1 +: 16*MODEL_DIM], msr1);

        if (errors==0)
            $display("  PASS[seed=%0d band=%0d pos=%0d S=%0d]  wreq: ms=%0d  1x=%0d  (2 runs=%0d)",
                     seed0, bnd, pp, ss, cntM, cnt1, cnt1<<1);
    end endtask

    initial begin
        errors=0; test_count=0;
        start1=0; startM=0; pos=0; s_len=0; x1=0; xM=0; seqM=0; cnt1=0; cntM=0; g_seq=0;
        @(negedge clk); rst=1'b0;

        // SPARSE (S > TOPK=2): DSA_REAL_IDX=1 pre-fetches EACH sequence's candidate
        //   keys and the per-row indexer selects TOP-K by the query -- validating the
        //   PER-SEQUENCE kidx_buf pre-fetch (each row scores against its own seq).
        one_case( 11, 0, 0, 4);   // S=4 pos=0 (RoPE identity)
        one_case(101, 0, 7, 4);   // S=4 pos>0
        one_case(211, 1, 4095, 3);// S=3 wide-range, large pos
        one_case(321, 0, 42, 3);  // S=3 pos>0

        if (errors==0) $display("ALL %0d TESTS PASSED", test_count);
        else begin
            $display("FAILED: %0d errors over %0d checks", errors, test_count);
            $fatal(1, "mla_attn_fp8 multi-seq TB failed");
        end
        $finish;
    end

    initial begin
        #800000000;
        $display("FAIL: timeout"); $fatal(1, "timeout");
    end
endmodule
