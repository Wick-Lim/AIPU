`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_decoder_block_fp8_union_tb.v -- UNION-SKIP proof for the PE_M>1 MoE loop
//----------------------------------------------------------------------------
// WHAT THIS PINS  (the Flash-bandwidth optimization added to the PE_M>1 path)
//   glm_decoder_block_fp8 at PE_M>1 evaluates the MoE routed experts by SCANNING
//   the expert axis and fetching ONLY the UNION of experts some batched row
//   selected (skipping -- never fetching -- experts no row picked).  This TB
//   proves BOTH halves of the contract at the slice (N_EXPERT=8, TOPK=2, PE_M=2):
//
//     (A) BYTE-IDENTICAL: row r of the PE_M=2 batched layer is BIT-EXACT (!==) to
//         a standalone PE_M=1 layer run on row r's own input (same pos/s_len/KV/
//         weights).  Skipped experts contributed gate=0 to every row, so the
//         per-row fp32 combine is unchanged.
//
//     (B) UNION-SKIP IS REAL: the number of routed-expert evaluations the PE_M=2
//         layer LAUNCHES (em_start pulses with fw_shared=0) equals the number of
//         DISTINCT experts the 2 rows selected (the union size) -- each union
//         member fetched exactly once, ascending index, no non-member ever
//         fetched.  For PE_M=2,TOPK=2 the union is <= 4 of 8, so the count is
//         STRICTLY LESS than the all-N_EXPERT=8 baseline (>=4 experts skipped).
//
//   Both DUTs read ONE shared weight ROM set through their own pull ports.  The
//   union is read back from the batched DUT's captured routing (dut_w.sel_e) and
//   the launch sequence is captured via dut_w.em_start / fw_shared / fw_eidx.
//   Prints "ALL <N> TESTS PASSED"; $fatal on any mismatch / bad union / timeout.
//============================================================================
module glm_decoder_block_fp8_union_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ================= slice config (== glm_decoder_block_fp8_tb) =============
    localparam integer MODEL_DIM  = 128;
    localparam integer H_HEADS    = 4;
    localparam integer NOPE       = 16;
    localparam integer ROPE       = 16;
    localparam integer V_DIM      = 32;
    localparam integer Q_LORA     = 64;
    localparam integer KV_LORA    = 32;
    localparam integer S_MAX      = 8;
    localparam integer TOPK_ATTN  = 8;
    localparam integer THETA      = 8000000;
    localparam integer PE_N       = 4;
    localparam integer POSW       = 20;
    localparam integer N_EXPERT   = 8;
    localparam integer TOPK       = 2;
    localparam integer INTER_MOE  = 64;
    localparam integer INTER_DENSE= 256;
    localparam [31:0]  RSCALE     = 32'h40200000;  // 2.5
    localparam integer TN         = 4;
    localparam integer BLK        = 128;
    localparam integer B          = 2;             // batched rows for the PE_M>1 DUT

    // ---- derived ----
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
    localparam integer A_NB    = (A_KMAX   +BLK-1)/BLK;
    localparam integer FF_NB_D = (FF_KMAX_D+BLK-1)/BLK;
    localparam integer R_NB    = (FF_KMAX_M+BLK-1)/BLK;

    integer test_count = 0;
    integer errors     = 0;

    // ================= shared WEIGHT ROMs (E4M3 codes + bf16 scales) ==========
    reg [15:0] G1 [0:MODEL_DIM-1];
    reg [15:0] G2 [0:MODEL_DIM-1];
    reg [7:0] W_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_uq  [0:HQK-1][0:Q_LORA-1];
    reg [7:0] W_dkv [0:KV_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_kr  [0:ROPE-1][0:MODEL_DIM-1];
    reg [7:0] W_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [7:0] W_uv  [0:HV-1][0:KV_LORA-1];
    reg [7:0] W_o   [0:MODEL_DIM-1][0:HV-1];
    reg [15:0] ScW_dq, ScW_uq, ScW_dkv, ScW_kr, ScW_uk, ScW_uv, ScW_o;
    reg [15:0] CKV [0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP [0:S_MAX-1][0:ROPE-1];
    reg [7:0]  Wg [0:MODEL_DIM-1][0:N_EXPERT-1];
    reg [15:0] ScWg;
    reg [7:0] Dg [0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [7:0] Du [0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [7:0] Dd [0:MODEL_DIM-1][0:INTER_DENSE-1];
    reg [15:0] ScDg [0:FF_NB_D-1];
    reg [15:0] ScDu [0:FF_NB_D-1];
    reg [15:0] ScDd [0:FF_NB_D-1];
    reg [7:0] Mg [0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] Mu [0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] Md [0:N_EXPERT-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [7:0] SHg [0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] SHu [0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] SHd [0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [15:0] ScMg [0:N_EXPERT-1], ScMu [0:N_EXPERT-1], ScMd [0:N_EXPERT-1];
    reg [15:0] ScSHg, ScSHu, ScSHd;

    // per-row inputs (B rows)
    reg [15:0] xin [0:B-1][0:MODEL_DIM-1];

    // ================= deterministic generators (== committed TBs) ============
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e=8'd125+h[6:4]; else e=8'd124+h[5:4];
        m=h[12:6]; gen_bf16={s,e,m};
    end endfunction
    function [7:0] gen_e4m3; input integer seed; input integer band;
        reg s; reg [3:0] e; reg [2:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e = 4'd7 + {3'b0,h[4]}; else e = 4'd6 + {3'b0,h[4]};
        m = h[12:10]; gen_e4m3 = {s,e,m};
    end endfunction
    function [15:0] gen_scale; input integer seed;
        reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*22229);
        e = 8'd122 + {7'b0,h[2]}; m = h[10:4]; gen_scale={1'b0,e,m};
    end endfunction

    integer i,j,e,sc;
    // build the shared weight set; the B row inputs are generated separately so
    // the two rows route to (potentially) different experts.
    task build_weights; input integer seed0; input integer band; begin
        sc=seed0;
        for (i=0;i<MODEL_DIM;i=i+1) begin G1[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin G2[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        ScW_dq=gen_scale(sc); sc=sc+1;  ScW_uq=gen_scale(sc); sc=sc+1;
        ScW_dkv=gen_scale(sc); sc=sc+1; ScW_kr=gen_scale(sc); sc=sc+1;
        ScW_uk=gen_scale(sc); sc=sc+1;  ScW_uv=gen_scale(sc); sc=sc+1;
        ScW_o=gen_scale(sc); sc=sc+1;
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<N_EXPERT;j=j+1) begin Wg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        ScWg=gen_scale(sc); sc=sc+1;
        for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Dg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Du[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_DENSE;j=j+1) begin Dd[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin ScDg[i]=gen_scale(sc); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin ScDu[i]=gen_scale(sc); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin ScDd[i]=gen_scale(sc); sc=sc+1; end
        for (e=0;e<N_EXPERT;e=e+1) begin
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mg[e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mu[e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin Md[e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScMg[e]=gen_scale(sc); sc=sc+1; ScMu[e]=gen_scale(sc); sc=sc+1; ScMd[e]=gen_scale(sc); sc=sc+1;
        end
        for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHu[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin SHd[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        ScSHg=gen_scale(sc); sc=sc+1; ScSHu=gen_scale(sc); sc=sc+1; ScSHd=gen_scale(sc); sc=sc+1;
    end endtask

    task build_rows; input integer seedA; input integer seedB; input integer band;
        integer s2; begin
        s2=seedA; for (i=0;i<MODEL_DIM;i=i+1) begin xin[0][i]=gen_bf16(s2,band); s2=s2+1; end
        s2=seedB; for (i=0;i<MODEL_DIM;i=i+1) begin xin[1][i]=gen_bf16(s2,band); s2=s2+1; end
    end endtask

    // ================= shared control =================
    reg  [POSW-1:0] pos;
    reg  [IDXW:0]   s_len;

    //========================================================================
    // BATCHED DUT (PE_M = B)  -- the union-skip path under test
    //========================================================================
    reg                       start_w, mode_w;
    wire                      busy_w, done_w;
    reg  [MODEL_DIM*16*B-1:0] x_vec_w;
    wire [MODEL_DIM*16*B-1:0] y_out_w;
    wire                      gn_req_w, gn_which_w; wire [$clog2(MODEL_DIM)-1:0] gn_idx_w; reg [15:0] gn_val_w;
    wire                      aw_req_w; wire [3:0] aw_sel_w; wire [A_GRPW-1:0] aw_grp_w; wire [A_KCW-1:0] aw_k_w;
    reg  [PE_N*8-1:0]         aw_col_w; reg [16*PE_N*A_NB-1:0] aw_scale_w;
    wire                      kc_req_w; wire [IDXW-1:0] kc_idx_w; reg [KV_LORA*16-1:0] kc_ckv_w; reg [ROPE*16-1:0] kc_krope_w; reg kc_valid_w;
    wire                      rw_req_w; wire [R_KW-1:0] rw_k_w; reg [8*N_EXPERT-1:0] rw_col_w; reg [16*N_EXPERT*R_NB-1:0] rw_scale_w;
    wire                      fw_req_w; wire [1:0] fw_sel_w; wire [FF_GWD-1:0] fw_grp_w; wire [FF_KWD-1:0] fw_k_w;
    wire                      fw_shared_w; wire [EIDXW-1:0] fw_eidx_w;
    reg  [8*TN-1:0]           fw_col_w, fw_col_up_w; reg [16*TN*FF_NB_D-1:0] fw_scale_g_w, fw_scale_u_w;

    glm_decoder_block_fp8 #(
        .MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX),
        .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW),
        .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), .BLK(BLK), .PE_M(B)
    ) dut_w (
        .clk(clk), .rst(rst), .start(start_w), .busy(busy_w), .done(done_w),
        .mode(mode_w), .pos(pos), .s_len(s_len), .x_vec(x_vec_w), .y_out(y_out_w),
        .gn_req(gn_req_w), .gn_which(gn_which_w), .gn_idx(gn_idx_w), .gn_val(gn_val_w),
        .aw_req(aw_req_w), .aw_sel(aw_sel_w), .aw_grp(aw_grp_w), .aw_k(aw_k_w),
        .aw_col(aw_col_w), .aw_scale(aw_scale_w),
        .kc_req(kc_req_w), .kc_idx(kc_idx_w), .kc_ckv(kc_ckv_w), .kc_krope(kc_krope_w), .kc_valid(kc_valid_w),
        .rw_req(rw_req_w), .rw_k(rw_k_w), .rw_col(rw_col_w), .rw_scale(rw_scale_w),
        .fw_req(fw_req_w), .fw_sel(fw_sel_w), .fw_grp(fw_grp_w), .fw_k(fw_k_w),
        .fw_shared(fw_shared_w), .fw_eidx(fw_eidx_w), .fw_col(fw_col_w), .fw_col_up(fw_col_up_w),
        .fw_scale_g(fw_scale_g_w), .fw_scale_u(fw_scale_u_w)
    );
    always @(posedge clk) begin if (rst) kc_valid_w <= 1'b0; else kc_valid_w <= kc_req_w; end
    /* verilator lint_off UNUSEDSIGNAL */
    wire _u_w = &{1'b0, busy_w, aw_req_w, fw_req_w, rw_req_w, gn_req_w};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // REFERENCE DUT (PE_M = 1) -- run once per row, y captured for bit-exact cmp
    //========================================================================
    reg                       start_r, mode_r;
    wire                      busy_r, done_r;
    reg  [MODEL_DIM*16-1:0]   x_vec_r;
    wire [MODEL_DIM*16-1:0]   y_out_r;
    wire                      gn_req_r, gn_which_r; wire [$clog2(MODEL_DIM)-1:0] gn_idx_r; reg [15:0] gn_val_r;
    wire                      aw_req_r; wire [3:0] aw_sel_r; wire [A_GRPW-1:0] aw_grp_r; wire [A_KCW-1:0] aw_k_r;
    reg  [PE_N*8-1:0]         aw_col_r; reg [16*PE_N*A_NB-1:0] aw_scale_r;
    wire                      kc_req_r; wire [IDXW-1:0] kc_idx_r; reg [KV_LORA*16-1:0] kc_ckv_r; reg [ROPE*16-1:0] kc_krope_r; reg kc_valid_r;
    wire                      rw_req_r; wire [R_KW-1:0] rw_k_r; reg [8*N_EXPERT-1:0] rw_col_r; reg [16*N_EXPERT*R_NB-1:0] rw_scale_r;
    wire                      fw_req_r; wire [1:0] fw_sel_r; wire [FF_GWD-1:0] fw_grp_r; wire [FF_KWD-1:0] fw_k_r;
    wire                      fw_shared_r; wire [EIDXW-1:0] fw_eidx_r;
    reg  [8*TN-1:0]           fw_col_r, fw_col_up_r; reg [16*TN*FF_NB_D-1:0] fw_scale_g_r, fw_scale_u_r;

    glm_decoder_block_fp8 #(
        .MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX),
        .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW),
        .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), .BLK(BLK), .PE_M(1)
    ) dut_r (
        .clk(clk), .rst(rst), .start(start_r), .busy(busy_r), .done(done_r),
        .mode(mode_r), .pos(pos), .s_len(s_len), .x_vec(x_vec_r), .y_out(y_out_r),
        .gn_req(gn_req_r), .gn_which(gn_which_r), .gn_idx(gn_idx_r), .gn_val(gn_val_r),
        .aw_req(aw_req_r), .aw_sel(aw_sel_r), .aw_grp(aw_grp_r), .aw_k(aw_k_r),
        .aw_col(aw_col_r), .aw_scale(aw_scale_r),
        .kc_req(kc_req_r), .kc_idx(kc_idx_r), .kc_ckv(kc_ckv_r), .kc_krope(kc_krope_r), .kc_valid(kc_valid_r),
        .rw_req(rw_req_r), .rw_k(rw_k_r), .rw_col(rw_col_r), .rw_scale(rw_scale_r),
        .fw_req(fw_req_r), .fw_sel(fw_sel_r), .fw_grp(fw_grp_r), .fw_k(fw_k_r),
        .fw_shared(fw_shared_r), .fw_eidx(fw_eidx_r), .fw_col(fw_col_r), .fw_col_up(fw_col_up_r),
        .fw_scale_g(fw_scale_g_r), .fw_scale_u(fw_scale_u_r)
    );
    always @(posedge clk) begin if (rst) kc_valid_r <= 1'b0; else kc_valid_r <= kc_req_r; end
    /* verilator lint_off UNUSEDSIGNAL */
    wire _u_r = &{1'b0, busy_r, aw_req_r, fw_req_r, rw_req_r, gn_req_r};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // BATCHED DUT responder (reads shared ROMs, keyed on dut_w pull ports)
    //========================================================================
    integer tw, rew, ftw, fow, cdw, obdw; reg [15:0] scaw;
    always @* gn_val_w = gn_which_w ? G2[gn_idx_w] : G1[gn_idx_w];
    always @(aw_sel_w or aw_grp_w or aw_k_w or start_w) begin
        aw_col_w = {PE_N*8{1'b0}}; aw_scale_w = {16*PE_N*A_NB{1'b0}};
        for (tw=0;tw<PE_N;tw=tw+1) case (aw_sel_w)
        4'd0: if (aw_grp_w*PE_N+tw<Q_LORA)   aw_col_w[8*tw+:8]=W_dq [aw_grp_w*PE_N+tw][aw_k_w];
        4'd1: if (aw_grp_w*PE_N+tw<HQK)      aw_col_w[8*tw+:8]=W_uq [aw_grp_w*PE_N+tw][aw_k_w];
        4'd2: if (aw_grp_w*PE_N+tw<KV_LORA)  aw_col_w[8*tw+:8]=W_dkv[aw_grp_w*PE_N+tw][aw_k_w];
        4'd3: if (aw_grp_w*PE_N+tw<ROPE)     aw_col_w[8*tw+:8]=W_kr [aw_grp_w*PE_N+tw][aw_k_w];
        4'd4: if (aw_grp_w*PE_N+tw<HNOPE)    aw_col_w[8*tw+:8]=W_uk [aw_grp_w*PE_N+tw][aw_k_w];
        4'd5: if (aw_grp_w*PE_N+tw<HV)       aw_col_w[8*tw+:8]=W_uv [aw_grp_w*PE_N+tw][aw_k_w];
        4'd6: if (aw_grp_w*PE_N+tw<MODEL_DIM)aw_col_w[8*tw+:8]=W_o  [aw_grp_w*PE_N+tw][aw_k_w];
        default: aw_col_w[8*tw+:8]=8'h0; endcase
        case (aw_sel_w) 4'd0:scaw=ScW_dq; 4'd1:scaw=ScW_uq; 4'd2:scaw=ScW_dkv; 4'd3:scaw=ScW_kr;
        4'd4:scaw=ScW_uk; 4'd5:scaw=ScW_uv; 4'd6:scaw=ScW_o; default:scaw=16'h3F80; endcase
        for (tw=0;tw<PE_N;tw=tw+1) aw_scale_w[16*tw+:16]=scaw;
    end
    always @* begin kc_ckv_w={KV_LORA*16{1'b0}}; kc_krope_w={ROPE*16{1'b0}};
        for (cdw=0;cdw<KV_LORA;cdw=cdw+1) kc_ckv_w[16*cdw+:16]=CKV[kc_idx_w][cdw];
        for (cdw=0;cdw<ROPE;cdw=cdw+1)    kc_krope_w[16*cdw+:16]=KRP[kc_idx_w][cdw]; end
    always @* begin rw_col_w={8*N_EXPERT{1'b0}}; rw_scale_w={16*N_EXPERT*R_NB{1'b0}};
        for (rew=0;rew<N_EXPERT;rew=rew+1) begin rw_col_w[8*rew+:8]=Wg[rw_k_w][rew]; rw_scale_w[16*rew+:16]=ScWg; end end
    always @(fw_grp_w or fw_k_w or fw_sel_w or fw_shared_w or fw_eidx_w or mode_w or start_w) begin
        fw_col_w={8*TN{1'b0}}; fw_col_up_w={8*TN{1'b0}}; fw_scale_g_w={16*TN*FF_NB_D{1'b0}}; fw_scale_u_w={16*TN*FF_NB_D{1'b0}};
        obdw=(fw_grp_w*TN)/BLK;
        for (ftw=0;ftw<TN;ftw=ftw+1) begin fow=fw_grp_w*TN+ftw;
            if (mode_w==1'b0) begin
                if (fw_sel_w==2'd2) begin if (fow<MODEL_DIM) fw_col_w[8*ftw+:8]=Dd[fow][fw_k_w]; end
                else begin if (fow<INTER_DENSE) begin fw_col_w[8*ftw+:8]=Dg[fow][fw_k_w]; fw_col_up_w[8*ftw+:8]=Du[fow][fw_k_w]; end end
            end else begin
                if (fw_shared_w) begin
                    if (fw_sel_w==2'd2) begin if (fow<MODEL_DIM) fw_col_w[8*ftw+:8]=SHd[fow][fw_k_w]; end
                    else if (fow<INTER_MOE) begin fw_col_w[8*ftw+:8]=SHg[fow][fw_k_w]; fw_col_up_w[8*ftw+:8]=SHu[fow][fw_k_w]; end
                end else begin
                    if (fw_sel_w==2'd2) begin if (fow<MODEL_DIM) fw_col_w[8*ftw+:8]=Md[fw_eidx_w][fow][fw_k_w]; end
                    else if (fow<INTER_MOE) begin fw_col_w[8*ftw+:8]=Mg[fw_eidx_w][fow][fw_k_w]; fw_col_up_w[8*ftw+:8]=Mu[fw_eidx_w][fow][fw_k_w]; end
                end end end
        for (ftw=0;ftw<TN;ftw=ftw+1) begin
            if (mode_w==1'b0) begin
                if (fw_sel_w==2'd2) begin fw_scale_g_w[16*(0*TN+ftw)+:16]=ScDd[0]; fw_scale_g_w[16*(1*TN+ftw)+:16]=ScDd[1]; end
                else begin fw_scale_g_w[16*(0*TN+ftw)+:16]=ScDg[obdw]; fw_scale_g_w[16*(1*TN+ftw)+:16]=ScDg[obdw];
                    fw_scale_u_w[16*(0*TN+ftw)+:16]=ScDu[obdw]; fw_scale_u_w[16*(1*TN+ftw)+:16]=ScDu[obdw]; end
            end else begin
                if (fw_shared_w) begin
                    if (fw_sel_w==2'd2) fw_scale_g_w[16*(0*TN+ftw)+:16]=ScSHd;
                    else begin fw_scale_g_w[16*(0*TN+ftw)+:16]=ScSHg; fw_scale_u_w[16*(0*TN+ftw)+:16]=ScSHu; end
                end else begin
                    if (fw_sel_w==2'd2) fw_scale_g_w[16*(0*TN+ftw)+:16]=ScMd[fw_eidx_w];
                    else begin fw_scale_g_w[16*(0*TN+ftw)+:16]=ScMg[fw_eidx_w]; fw_scale_u_w[16*(0*TN+ftw)+:16]=ScMu[fw_eidx_w]; end
                end end end
    end

    //========================================================================
    // REFERENCE DUT responder (reads SAME shared ROMs, keyed on dut_r ports)
    //========================================================================
    integer tr, rer, ftr, forr, cdr, obdr; reg [15:0] scar;
    always @* gn_val_r = gn_which_r ? G2[gn_idx_r] : G1[gn_idx_r];
    always @(aw_sel_r or aw_grp_r or aw_k_r or start_r) begin
        aw_col_r = {PE_N*8{1'b0}}; aw_scale_r = {16*PE_N*A_NB{1'b0}};
        for (tr=0;tr<PE_N;tr=tr+1) case (aw_sel_r)
        4'd0: if (aw_grp_r*PE_N+tr<Q_LORA)   aw_col_r[8*tr+:8]=W_dq [aw_grp_r*PE_N+tr][aw_k_r];
        4'd1: if (aw_grp_r*PE_N+tr<HQK)      aw_col_r[8*tr+:8]=W_uq [aw_grp_r*PE_N+tr][aw_k_r];
        4'd2: if (aw_grp_r*PE_N+tr<KV_LORA)  aw_col_r[8*tr+:8]=W_dkv[aw_grp_r*PE_N+tr][aw_k_r];
        4'd3: if (aw_grp_r*PE_N+tr<ROPE)     aw_col_r[8*tr+:8]=W_kr [aw_grp_r*PE_N+tr][aw_k_r];
        4'd4: if (aw_grp_r*PE_N+tr<HNOPE)    aw_col_r[8*tr+:8]=W_uk [aw_grp_r*PE_N+tr][aw_k_r];
        4'd5: if (aw_grp_r*PE_N+tr<HV)       aw_col_r[8*tr+:8]=W_uv [aw_grp_r*PE_N+tr][aw_k_r];
        4'd6: if (aw_grp_r*PE_N+tr<MODEL_DIM)aw_col_r[8*tr+:8]=W_o  [aw_grp_r*PE_N+tr][aw_k_r];
        default: aw_col_r[8*tr+:8]=8'h0; endcase
        case (aw_sel_r) 4'd0:scar=ScW_dq; 4'd1:scar=ScW_uq; 4'd2:scar=ScW_dkv; 4'd3:scar=ScW_kr;
        4'd4:scar=ScW_uk; 4'd5:scar=ScW_uv; 4'd6:scar=ScW_o; default:scar=16'h3F80; endcase
        for (tr=0;tr<PE_N;tr=tr+1) aw_scale_r[16*tr+:16]=scar;
    end
    always @* begin kc_ckv_r={KV_LORA*16{1'b0}}; kc_krope_r={ROPE*16{1'b0}};
        for (cdr=0;cdr<KV_LORA;cdr=cdr+1) kc_ckv_r[16*cdr+:16]=CKV[kc_idx_r][cdr];
        for (cdr=0;cdr<ROPE;cdr=cdr+1)    kc_krope_r[16*cdr+:16]=KRP[kc_idx_r][cdr]; end
    always @* begin rw_col_r={8*N_EXPERT{1'b0}}; rw_scale_r={16*N_EXPERT*R_NB{1'b0}};
        for (rer=0;rer<N_EXPERT;rer=rer+1) begin rw_col_r[8*rer+:8]=Wg[rw_k_r][rer]; rw_scale_r[16*rer+:16]=ScWg; end end
    always @(fw_grp_r or fw_k_r or fw_sel_r or fw_shared_r or fw_eidx_r or mode_r or start_r) begin
        fw_col_r={8*TN{1'b0}}; fw_col_up_r={8*TN{1'b0}}; fw_scale_g_r={16*TN*FF_NB_D{1'b0}}; fw_scale_u_r={16*TN*FF_NB_D{1'b0}};
        obdr=(fw_grp_r*TN)/BLK;
        for (ftr=0;ftr<TN;ftr=ftr+1) begin forr=fw_grp_r*TN+ftr;
            if (mode_r==1'b0) begin
                if (fw_sel_r==2'd2) begin if (forr<MODEL_DIM) fw_col_r[8*ftr+:8]=Dd[forr][fw_k_r]; end
                else begin if (forr<INTER_DENSE) begin fw_col_r[8*ftr+:8]=Dg[forr][fw_k_r]; fw_col_up_r[8*ftr+:8]=Du[forr][fw_k_r]; end end
            end else begin
                if (fw_shared_r) begin
                    if (fw_sel_r==2'd2) begin if (forr<MODEL_DIM) fw_col_r[8*ftr+:8]=SHd[forr][fw_k_r]; end
                    else if (forr<INTER_MOE) begin fw_col_r[8*ftr+:8]=SHg[forr][fw_k_r]; fw_col_up_r[8*ftr+:8]=SHu[forr][fw_k_r]; end
                end else begin
                    if (fw_sel_r==2'd2) begin if (forr<MODEL_DIM) fw_col_r[8*ftr+:8]=Md[fw_eidx_r][forr][fw_k_r]; end
                    else if (forr<INTER_MOE) begin fw_col_r[8*ftr+:8]=Mg[fw_eidx_r][forr][fw_k_r]; fw_col_up_r[8*ftr+:8]=Mu[fw_eidx_r][forr][fw_k_r]; end
                end end end
        for (ftr=0;ftr<TN;ftr=ftr+1) begin
            if (mode_r==1'b0) begin
                if (fw_sel_r==2'd2) begin fw_scale_g_r[16*(0*TN+ftr)+:16]=ScDd[0]; fw_scale_g_r[16*(1*TN+ftr)+:16]=ScDd[1]; end
                else begin fw_scale_g_r[16*(0*TN+ftr)+:16]=ScDg[obdr]; fw_scale_g_r[16*(1*TN+ftr)+:16]=ScDg[obdr];
                    fw_scale_u_r[16*(0*TN+ftr)+:16]=ScDu[obdr]; fw_scale_u_r[16*(1*TN+ftr)+:16]=ScDu[obdr]; end
            end else begin
                if (fw_shared_r) begin
                    if (fw_sel_r==2'd2) fw_scale_g_r[16*(0*TN+ftr)+:16]=ScSHd;
                    else begin fw_scale_g_r[16*(0*TN+ftr)+:16]=ScSHg; fw_scale_u_r[16*(0*TN+ftr)+:16]=ScSHu; end
                end else begin
                    if (fw_sel_r==2'd2) fw_scale_g_r[16*(0*TN+ftr)+:16]=ScMd[fw_eidx_r];
                    else begin fw_scale_g_r[16*(0*TN+ftr)+:16]=ScMg[fw_eidx_r]; fw_scale_u_r[16*(0*TN+ftr)+:16]=ScMu[fw_eidx_r]; end
                end end end
    end

    // ================= union-skip fetch-launch instrumentation ================
    // A routed-expert evaluation LAUNCH = em_start pulse with fw_shared=0.  We
    // sample at negedge (em_start/fw_shared are stable for the whole clk period
    // after their setting posedge).  fw_eidx at that instant is the launched id.
    integer routed_launches, shared_launches;
    reg [EIDXW-1:0] launched_eidx [0:N_EXPERT-1];
    reg counting;
    always @(negedge clk) if (counting) begin
        if (dut_w.em_start && !fw_shared_w) begin
            if (routed_launches < N_EXPERT) launched_eidx[routed_launches] = fw_eidx_w;
            routed_launches = routed_launches + 1;
        end
        if (dut_w.em_start && fw_shared_w) shared_launches = shared_launches + 1;
    end

    // ---- run the PE_M=1 reference on one row, capture y_out_r ----
    integer wd;
    reg [MODEL_DIM*16-1:0] ref_y [0:B-1];
    task run_ref_row; input integer r; begin
        for (i=0;i<MODEL_DIM;i=i+1) x_vec_r[16*i+:16]=xin[r][i];
        mode_r = 1'b1;
        @(negedge clk); start_r=1'b1; @(negedge clk); start_r=1'b0;
        wd=0; while (!done_r && wd<4000000) begin @(negedge clk); wd=wd+1; end
        if (!done_r) begin $display("FAIL: ref row %0d TIMEOUT", r); errors=errors+1; end
        @(negedge clk);
        ref_y[r] = y_out_r;
    end endtask

    // ---- union bookkeeping ----
    integer u, uu, union_size, seen_before, d, rowb, slot, r0;
    reg [EIDXW-1:0] union_list [0:2*TOPK-1];
    reg [15:0] dv, rv;

    // ---- run the PE_M=B batched DUT, then check union-skip + bit-exact ----
    task run_batched_and_check; input [255:0] label; begin
        test_count = test_count + 1;
        for (r0=0;r0<B;r0=r0+1) for (i=0;i<MODEL_DIM;i=i+1)
            x_vec_w[16*(MODEL_DIM*r0+i)+:16]=xin[r0][i];
        mode_w = 1'b1;
        routed_launches=0; shared_launches=0;
        @(negedge clk);
        counting=1'b1;
        start_w=1'b1; @(negedge clk); start_w=1'b0;
        wd=0; while (!done_w && wd<8000000) begin @(negedge clk); wd=wd+1; end
        counting=1'b0;
        if (!done_w) begin $display("FAIL[%0s]: batched TIMEOUT", label); errors=errors+1; disable run_batched_and_check; end
        @(negedge clk);

        // ---- distinct experts the B rows selected (the union), from sel_e ----
        union_size=0;
        for (rowb=0; rowb<B; rowb=rowb+1)
            for (slot=0; slot<TOPK; slot=slot+1) begin
                seen_before=0;
                for (uu=0; uu<union_size; uu=uu+1)
                    if (union_list[uu]===dut_w.sel_e[rowb][slot]) seen_before=1;
                if (!seen_before) begin union_list[union_size]=dut_w.sel_e[rowb][slot]; union_size=union_size+1; end
            end

        // CHECK 1: routed launches == union size (each member once, no non-member)
        if (routed_launches !== union_size) begin
            $display("FAIL[%0s]: routed fetches=%0d != union size=%0d", label, routed_launches, union_size);
            errors=errors+1;
        end
        // CHECK 2: launch sequence strictly ascending + every launched id is in union
        for (u=0; u<routed_launches; u=u+1) begin
            seen_before=0;
            for (uu=0; uu<union_size; uu=uu+1) if (launched_eidx[u]===union_list[uu]) seen_before=1;
            if (!seen_before) begin
                $display("FAIL[%0s]: launched expert %0d not in union", label, launched_eidx[u]);
                errors=errors+1;
            end
            if (u>0 && !(launched_eidx[u] > launched_eidx[u-1])) begin
                $display("FAIL[%0s]: launch order not strictly ascending (%0d then %0d)",
                         label, launched_eidx[u-1], launched_eidx[u]);
                errors=errors+1;
            end
        end
        // CHECK 3: exactly one shared expert launch
        if (shared_launches !== 1) begin
            $display("FAIL[%0s]: shared launches=%0d != 1", label, shared_launches); errors=errors+1;
        end
        // CHECK 4: union-skip actually saved fetches (strictly < N_EXPERT)
        if (!(union_size < N_EXPERT)) begin
            $display("FAIL[%0s]: union=%0d not < N_EXPERT=%0d (no skip)", label, union_size, N_EXPERT);
            errors=errors+1;
        end
        // CHECK 5: BYTE-IDENTICAL -- each batched row == its PE_M=1 reference run
        for (rowb=0; rowb<B; rowb=rowb+1)
            for (d=0; d<MODEL_DIM; d=d+1) begin
                dv=y_out_w[16*(MODEL_DIM*rowb+d)+:16];
                rv=ref_y[rowb][16*d+:16];
                if (^dv===1'bx) begin $display("FAIL[%0s]: row %0d elt %0d X", label, rowb, d); errors=errors+1; end
                else if (dv!==rv) begin
                    $display("FAIL[%0s]: row %0d elt %0d dut=%h != ref=%h", label, rowb, d, dv, rv); errors=errors+1;
                end
            end

        $display("UNION-SKIP OK[%0s]: PE_M=%0d evaluated %0d experts == distinct-selected, skipped %0d of %0d; both rows bit-exact vs PE_M=1",
                 label, B, routed_launches, N_EXPERT-union_size, N_EXPERT);
    end endtask

    task do_case; input integer seedW; input integer seedA; input integer seedB; input integer band;
        input [255:0] label; begin
        build_weights(seedW, band);
        build_rows(seedA, seedB, band);
        run_ref_row(0);
        run_ref_row(1);
        run_batched_and_check(label);
    end endtask

    // ---- watchdog ----
    initial begin #400000000; $display("FAIL: global timeout"); $fatal; end

    initial begin
        start_w=1'b0; start_r=1'b0; mode_w=1'b1; mode_r=1'b1; counting=1'b0;
        routed_launches=0; shared_launches=0;
        pos={POSW{1'b0}}; s_len={(IDXW+1){1'b0}};
        x_vec_w={MODEL_DIM*16*B{1'b0}}; x_vec_r={MODEL_DIM*16{1'b0}};
        rst=1'b1; repeat(4) @(negedge clk); rst=1'b0; @(negedge clk);

        // several (weights, rowA, rowB) so the router yields different unions
        pos=0;   s_len=1;     do_case(31000, 4000,  9000,  0, "MoE pos0 S1");
        pos=2;   s_len=2;     do_case(33000, 12000, 61000, 0, "MoE pos2 S2");
        pos=300; s_len=5;     do_case(70000, 5000,  88000, 0, "MoE pos300 S5");
        pos=128; s_len=S_MAX; do_case(95000, 21000, 44000, 1, "MoE pos128 Smax");

        if (errors!=0) begin
            $display("FAILED: %0d error(s) across %0d tests", errors, test_count);
            $fatal;
        end
        $display("ALL %0d TESTS PASSED", test_count);
        $finish;
    end
endmodule
