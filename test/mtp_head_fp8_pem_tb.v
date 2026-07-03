`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// mtp_head_fp8_pem_tb.v -- PE_M BATCHING equivalence + weight-share TB for the
//   GLM-5.2-FP8 Multi-Token-Prediction head (mtp_head_fp8.v).
//----------------------------------------------------------------------------
// GOAL (NOT an fp64 golden -- that is mtp_head_fp8_tb's job).  Prove the PE_M
//   widening of the head is a PURE THROUGHPUT TRANSFORM:
//     (1) BIT-EXACT per-row equivalence -- driving B distinct MTP token-rows
//         (h_t[r], emb_t1[r]) through ONE PE_M=B head yields, for EACH row r, the
//         SAME (logits[r], argmax[r], h_mtp[r]) as an independent PE_M=1 run on
//         row r's own inputs (same shared pos/s_len/KV/mode).  X/Z-aware bit
//         compare, checked in BOTH the DENSE and MoE FFN paths.
//     (2) WEIGHT-BW AMORTIZATION B->1 -- in the DENSE path (no per-row expert
//         divergence) the batched PE_M=B run asserts EVERY weight-fetch request
//         port (cn_req concat/final gamma, pw_req combine-proj, gn_req decoder
//         gamma, aw_req attn, kc_req cache, rw_req router, fw_req FFN, lw_req LM
//         head) on EXACTLY the same number of cycles as ONE PE_M=1 run -- the B
//         rows share the ONE weight fetch stream.  (In MoE the router picks
//         per-row experts, so fw_req/rw_req legitimately iterate more experts at
//         PE_M>1; the non-diverging streams still share -- we assert the full set
//         in DENSE, where it holds for every port including fw_req/rw_req.)
//
//   We instantiate ONE PE_M=B DUT and ONE PE_M=1 DUT over the SAME weight ROMs /
//   block scales / cache; the PE_M=1 DUT is run once PER ROW to build the
//   reference, then the PE_M=B DUT once.  S_len <= TOPK_ATTN (dense DSA fallback)
//   so the shared, row-0-driven attention key selection equals every row's own.
//
//   Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X / Z / timeout.
//============================================================================
module mtp_head_fp8_pem_tb;

    // ================= slice config (small-but-valid; PE_M smoke, fast) =========
    localparam integer MODEL_DIM  = 32;
    localparam integer VOCAB      = 16;
    localparam integer H_HEADS    = 2;
    localparam integer NOPE       = 4;
    localparam integer ROPE       = 4;
    localparam integer V_DIM      = 4;
    localparam integer Q_LORA     = 8;
    localparam integer KV_LORA    = 8;
    localparam integer S_MAX      = 2;
    localparam integer TOPK_ATTN  = 2;
    localparam integer THETA      = 8000000;
    localparam integer PE_N       = 2;
    localparam integer POSW       = 20;
    localparam integer N_EXPERT   = 4;
    localparam integer TOPK       = 2;
    localparam integer INTER_MOE  = 16;
    localparam integer INTER_DENSE= 32;
    localparam [31:0]  RSCALE     = 32'h40200000;  // 2.5
    localparam integer TN         = 2;
    localparam integer BLK        = 128;
    localparam integer LM_TN      = 2;
    localparam integer PROJ_TN    = 2;
    localparam integer B          = 3;             // PE_M batch under test

    // ---- derived (mirror DUT) ----
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
    localparam integer FF_NB_M = (FF_KMAX_M+BLK-1)/BLK;
    localparam integer R_NB    = (FF_KMAX_M+BLK-1)/BLK;
    localparam integer TOKW   = (VOCAB<=1)?1:$clog2(VOCAB);
    localparam integer DIMW   = (MODEL_DIM<=1)?1:$clog2(MODEL_DIM);
    localparam integer NVTILE = VOCAB/LM_TN;
    localparam integer VTW    = (NVTILE<=1)?1:$clog2(NVTILE);
    localparam integer CK     = 2*MODEL_DIM;
    localparam integer CKIW   = $clog2(CK);
    localparam integer NPTILE = MODEL_DIM/PROJ_TN;
    localparam integer PTW    = (NPTILE<=1)?1:$clog2(NPTILE);
    localparam integer PROJ_NB = (CK+BLK-1)/BLK;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    `include "glm_fp.vh"
    `include "fp8_e4m3.vh"

    // ================= shared WEIGHT ROMs (identical for both DUTs) =============
    reg [15:0] GA  [0:MODEL_DIM-1];         // head RMSNorm gamma (h_t)
    reg [15:0] GB  [0:MODEL_DIM-1];         // head RMSNorm gamma (emb)
    reg [15:0] GF  [0:MODEL_DIM-1];         // head final RMSNorm gamma
    reg [15:0] Wlm [0:VOCAB-1][0:MODEL_DIM-1];
    reg [7:0]  Wp  [0:MODEL_DIM-1][0:CK-1]; // FP8 W_proj[out][in]
    reg [15:0] ScWp[0:PROJ_NB-1];
    reg [15:0] G1 [0:MODEL_DIM-1];          // decoder pre-attn gamma
    reg [15:0] G2 [0:MODEL_DIM-1];          // decoder pre-ffn gamma
    reg [15:0] CKV [0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP [0:S_MAX-1][0:ROPE-1];
    reg [7:0] W_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_uq  [0:HQK-1][0:Q_LORA-1];
    reg [7:0] W_dkv [0:KV_LORA-1][0:MODEL_DIM-1];  // DEAD (cache overwrites)
    reg [7:0] W_kr  [0:ROPE-1][0:MODEL_DIM-1];     // DEAD (cache overwrites)
    reg [7:0] W_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [7:0] W_uv  [0:HV-1][0:KV_LORA-1];
    reg [7:0] W_o   [0:MODEL_DIM-1][0:HV-1];
    reg [15:0] ScW_dq, ScW_uq, ScW_dkv, ScW_kr, ScW_uk, ScW_uv, ScW_o;
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

    // ---- B distinct token rows ----
    reg [15:0] in_h [0:B-1][0:MODEL_DIM-1];
    reg [15:0] in_e [0:B-1][0:MODEL_DIM-1];

    // ================= deterministic stimulus generators =======================
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e=8'd125+h[6:4]; else e=8'd124+h[5:4];
        m=h[12:6];
        gen_bf16={s,e,m};
    end endfunction
    function [7:0] gen_e4m3; input integer seed; input integer band;
        reg s; reg [3:0] e; reg [2:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e = 4'd7 + {3'b0,h[4]}; else e = 4'd6 + {3'b0,h[4]};
        m = h[12:10];
        gen_e4m3 = {s,e,m};                    // e<=8 -> never the NaN slot
    end endfunction
    function [15:0] gen_scale; input integer seed;
        reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*22229);
        e = 8'd122 + {7'b0,h[2]};
        m = h[10:4];
        gen_scale={1'b0,e,m};
    end endfunction

    integer i,j,e2,ex,rw2,sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc=seed0;
        for (i=0;i<MODEL_DIM;i=i+1) begin GA[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin GB[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin GF[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<CK;j=j+1) begin Wp[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<PROJ_NB;i=i+1) begin ScWp[i]=gen_scale(sc); sc=sc+1; end
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Wlm[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin G1[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin G2[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        ScW_dq=gen_scale(sc); sc=sc+1; ScW_uq=gen_scale(sc); sc=sc+1;
        ScW_dkv=gen_scale(sc); sc=sc+1; ScW_kr=gen_scale(sc); sc=sc+1;
        ScW_uk=gen_scale(sc); sc=sc+1; ScW_uv=gen_scale(sc); sc=sc+1; ScW_o=gen_scale(sc); sc=sc+1;
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<N_EXPERT;j=j+1) begin Wg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        ScWg=gen_scale(sc); sc=sc+1;
        for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Dg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Du[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_DENSE;j=j+1) begin Dd[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin ScDg[i]=gen_scale(sc); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin ScDu[i]=gen_scale(sc); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin ScDd[i]=gen_scale(sc); sc=sc+1; end
        for (ex=0;ex<N_EXPERT;ex=ex+1) begin
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mg[ex][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mu[ex][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin Md[ex][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScMg[ex]=gen_scale(sc); sc=sc+1; ScMu[ex]=gen_scale(sc); sc=sc+1; ScMd[ex]=gen_scale(sc); sc=sc+1;
        end
        for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHu[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin SHd[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        ScSHg=gen_scale(sc); sc=sc+1; ScSHu=gen_scale(sc); sc=sc+1; ScSHd=gen_scale(sc); sc=sc+1;
        // B distinct token rows (each from its own seed band)
        for (rw2=0; rw2<B; rw2=rw2+1) begin
            for (i=0;i<MODEL_DIM;i=i+1) begin in_h[rw2][i]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) begin in_e[rw2][i]=gen_bf16(sc,band); sc=sc+1; end
        end
    end endtask

    // ================= a full combinational pull responder (macro) =============
    //   Each DUT gets its OWN copy keyed off ITS pull-request wires (suffix SUF).
    //   Reads the SHARED weight ROMs -> both DUTs see byte-identical weights.
    `define MTP_RESP(SUF)                                                          \
        always @* begin                                                            \
            case (cn_which``SUF)                                                    \
                2'd0:    cn_val``SUF = GA[cn_idx``SUF];                             \
                2'd1:    cn_val``SUF = GB[cn_idx``SUF];                             \
                default: cn_val``SUF = GF[cn_idx``SUF];                             \
            endcase                                                                \
        end                                                                        \
        integer pq``SUF, cd``SUF;                                                  \
        always @* begin                                                            \
            pw_col``SUF   = {PROJ_TN*8{1'b0}};                                      \
            pw_scale``SUF = {16*PROJ_TN*PROJ_NB{1'b0}};                             \
            for (pq``SUF=0; pq``SUF<PROJ_TN; pq``SUF=pq``SUF+1)                     \
                pw_col``SUF[8*pq``SUF +: 8] = Wp[pw_ptile``SUF*PROJ_TN + pq``SUF][pw_k``SUF]; \
            for (cd``SUF=0; cd``SUF<PROJ_NB; cd``SUF=cd``SUF+1)                     \
                for (pq``SUF=0; pq``SUF<PROJ_TN; pq``SUF=pq``SUF+1)                 \
                    pw_scale``SUF[16*(cd``SUF*PROJ_TN + pq``SUF) +: 16] = ScWp[cd``SUF]; \
        end                                                                        \
        always @* gn_val``SUF = gn_which``SUF ? G2[gn_idx``SUF] : G1[gn_idx``SUF];  \
        integer at``SUF; reg [15:0] sca``SUF;                                       \
        always @* begin                                                            \
            aw_col``SUF   = {PE_N*8{1'b0}};                                         \
            aw_scale``SUF = {16*PE_N*A_NB{1'b0}};                                   \
            for (at``SUF=0; at``SUF<PE_N; at``SUF=at``SUF+1) begin                  \
                case (aw_sel``SUF)                                                  \
                4'd0: if(aw_grp``SUF*PE_N+at``SUF<Q_LORA)  aw_col``SUF[8*at``SUF+:8]=W_dq [aw_grp``SUF*PE_N+at``SUF][aw_k``SUF]; \
                4'd1: if(aw_grp``SUF*PE_N+at``SUF<HQK)     aw_col``SUF[8*at``SUF+:8]=W_uq [aw_grp``SUF*PE_N+at``SUF][aw_k``SUF]; \
                4'd2: if(aw_grp``SUF*PE_N+at``SUF<KV_LORA) aw_col``SUF[8*at``SUF+:8]=W_dkv[aw_grp``SUF*PE_N+at``SUF][aw_k``SUF]; \
                4'd3: if(aw_grp``SUF*PE_N+at``SUF<ROPE)    aw_col``SUF[8*at``SUF+:8]=W_kr [aw_grp``SUF*PE_N+at``SUF][aw_k``SUF]; \
                4'd4: if(aw_grp``SUF*PE_N+at``SUF<HNOPE)   aw_col``SUF[8*at``SUF+:8]=W_uk [aw_grp``SUF*PE_N+at``SUF][aw_k``SUF]; \
                4'd5: if(aw_grp``SUF*PE_N+at``SUF<HV)      aw_col``SUF[8*at``SUF+:8]=W_uv [aw_grp``SUF*PE_N+at``SUF][aw_k``SUF]; \
                4'd6: if(aw_grp``SUF*PE_N+at``SUF<MODEL_DIM)aw_col``SUF[8*at``SUF+:8]=W_o [aw_grp``SUF*PE_N+at``SUF][aw_k``SUF]; \
                default: aw_col``SUF[8*at``SUF+:8]=8'h0;                            \
                endcase                                                            \
            end                                                                    \
            case (aw_sel``SUF)                                                      \
                4'd0: sca``SUF=ScW_dq; 4'd1: sca``SUF=ScW_uq; 4'd2: sca``SUF=ScW_dkv; 4'd3: sca``SUF=ScW_kr; \
                4'd4: sca``SUF=ScW_uk; 4'd5: sca``SUF=ScW_uv; 4'd6: sca``SUF=ScW_o; default: sca``SUF=16'h3F80; \
            endcase                                                                \
            for (at``SUF=0; at``SUF<PE_N; at``SUF=at``SUF+1) aw_scale``SUF[16*at``SUF+:16]=sca``SUF; \
        end                                                                        \
        integer kd``SUF;                                                           \
        always @* begin                                                            \
            kc_ckv``SUF   = {KV_LORA*16{1'b0}};                                     \
            kc_krope``SUF = {ROPE*16{1'b0}};                                        \
            for (kd``SUF=0;kd``SUF<KV_LORA;kd``SUF=kd``SUF+1) kc_ckv``SUF[16*kd``SUF+:16]   = CKV[kc_idx``SUF][kd``SUF]; \
            for (kd``SUF=0;kd``SUF<ROPE;kd``SUF=kd``SUF+1)    kc_krope``SUF[16*kd``SUF+:16] = KRP[kc_idx``SUF][kd``SUF]; \
        end                                                                        \
        always @(posedge clk) begin if (rst) kc_valid``SUF<=1'b0; else kc_valid``SUF<=kc_req``SUF; end \
        integer re``SUF;                                                           \
        always @* begin                                                            \
            rw_col``SUF   = {8*N_EXPERT{1'b0}};                                     \
            rw_scale``SUF = {16*N_EXPERT*R_NB{1'b0}};                               \
            for (re``SUF=0;re``SUF<N_EXPERT;re``SUF=re``SUF+1) begin                \
                rw_col``SUF[8*re``SUF+:8]    = Wg[rw_k``SUF][re``SUF];              \
                rw_scale``SUF[16*re``SUF+:16]= ScWg;                               \
            end                                                                    \
        end                                                                        \
        integer ft``SUF, fo``SUF;                                                  \
        always @* begin                                                            \
            fw_col``SUF     = {8*TN{1'b0}};      fw_col_up``SUF  = {8*TN{1'b0}};    \
            fw_scale_g``SUF = {16*TN*FF_NB_D{1'b0}}; fw_scale_u``SUF = {16*TN*FF_NB_D{1'b0}}; \
            for (ft``SUF=0;ft``SUF<TN;ft``SUF=ft``SUF+1) begin                      \
                fo``SUF = fw_grp``SUF*TN + ft``SUF;                                 \
                if (mode_q``SUF==1'b0) begin                                        \
                    if (fw_sel``SUF==2'd2) begin if (fo``SUF<MODEL_DIM) fw_col``SUF[8*ft``SUF+:8]=Dd[fo``SUF][fw_k``SUF]; end \
                    else if (fo``SUF<INTER_DENSE) begin fw_col``SUF[8*ft``SUF+:8]=Dg[fo``SUF][fw_k``SUF]; fw_col_up``SUF[8*ft``SUF+:8]=Du[fo``SUF][fw_k``SUF]; end \
                end else begin                                                     \
                    if (fw_shared``SUF) begin                                       \
                        if (fw_sel``SUF==2'd2) begin if (fo``SUF<MODEL_DIM) fw_col``SUF[8*ft``SUF+:8]=SHd[fo``SUF][fw_k``SUF]; end \
                        else if (fo``SUF<INTER_MOE) begin fw_col``SUF[8*ft``SUF+:8]=SHg[fo``SUF][fw_k``SUF]; fw_col_up``SUF[8*ft``SUF+:8]=SHu[fo``SUF][fw_k``SUF]; end \
                    end else begin                                                 \
                        if (fw_sel``SUF==2'd2) begin if (fo``SUF<MODEL_DIM) fw_col``SUF[8*ft``SUF+:8]=Md[fw_eidx``SUF][fo``SUF][fw_k``SUF]; end \
                        else if (fo``SUF<INTER_MOE) begin fw_col``SUF[8*ft``SUF+:8]=Mg[fw_eidx``SUF][fo``SUF][fw_k``SUF]; fw_col_up``SUF[8*ft``SUF+:8]=Mu[fw_eidx``SUF][fo``SUF][fw_k``SUF]; end \
                    end                                                            \
                end                                                                \
            end                                                                    \
            for (ft``SUF=0;ft``SUF<TN;ft``SUF=ft``SUF+1) begin                      \
                if (mode_q``SUF==1'b0) begin                                        \
                    if (fw_sel``SUF==2'd2) fw_scale_g``SUF[16*ft``SUF+:16]=ScDd[0]; \
                    else begin fw_scale_g``SUF[16*ft``SUF+:16]=ScDg[0]; fw_scale_u``SUF[16*ft``SUF+:16]=ScDu[0]; end \
                end else begin                                                     \
                    if (fw_shared``SUF) begin                                       \
                        if (fw_sel``SUF==2'd2) fw_scale_g``SUF[16*ft``SUF+:16]=ScSHd; \
                        else begin fw_scale_g``SUF[16*ft``SUF+:16]=ScSHg; fw_scale_u``SUF[16*ft``SUF+:16]=ScSHu; end \
                    end else begin                                                 \
                        if (fw_sel``SUF==2'd2) fw_scale_g``SUF[16*ft``SUF+:16]=ScMd[fw_eidx``SUF]; \
                        else begin fw_scale_g``SUF[16*ft``SUF+:16]=ScMg[fw_eidx``SUF]; fw_scale_u``SUF[16*ft``SUF+:16]=ScMu[fw_eidx``SUF]; end \
                    end                                                            \
                end                                                                \
            end                                                                    \
        end                                                                        \
        integer lq``SUF;                                                           \
        always @* begin                                                            \
            lw_col``SUF = {LM_TN*16{1'b0}};                                         \
            for (lq``SUF=0;lq``SUF<LM_TN;lq``SUF=lq``SUF+1) lw_col``SUF[16*lq``SUF+:16] = Wlm[lw_vtile``SUF*LM_TN + lq``SUF][lw_k``SUF]; \
        end

    // ---- mode_q per DUT: the responder needs the LATCHED mode (FFN path). The
    //      head latches `mode` at start; we mirror it by registering the driven
    //      `mode` on the same start pulse so the fw responder tracks the DUT. ----
    reg mode_q_s, mode_q_b;

    // ======================== DUT S : PE_M = 1 reference =======================
    reg                    start_s, mode_s;
    reg  [POSW-1:0]        pos_s;
    reg  [IDXW:0]          s_len_s;
    reg  [MODEL_DIM*16-1:0] h_t_s, emb_s;
    wire                   busy_s, done_s;
    wire [VOCAB*16-1:0]    logits_s;
    wire [TOKW-1:0]        argmax_s;
    wire [MODEL_DIM*16-1:0] h_mtp_s;
    wire                   cn_req_s;  wire [1:0] cn_which_s;  wire [DIMW-1:0] cn_idx_s;
    reg  [15:0]            cn_val_s;
    wire                   pw_req_s;  wire [PTW-1:0] pw_ptile_s;  wire [CKIW-1:0] pw_k_s;
    reg  [PROJ_TN*8-1:0]   pw_col_s;  reg [16*PROJ_TN*PROJ_NB-1:0] pw_scale_s;
    wire                   gn_req_s, gn_which_s;  wire [DIMW-1:0] gn_idx_s;
    reg  [15:0]            gn_val_s;
    wire                   aw_req_s;  wire [3:0] aw_sel_s;  wire [A_GRPW-1:0] aw_grp_s;  wire [A_KCW-1:0] aw_k_s;
    reg  [PE_N*8-1:0]      aw_col_s;  reg [16*PE_N*A_NB-1:0] aw_scale_s;
    wire                   kc_req_s;  wire [IDXW-1:0] kc_idx_s;
    reg  [KV_LORA*16-1:0]  kc_ckv_s;  reg [ROPE*16-1:0] kc_krope_s;  reg kc_valid_s;
    wire                   rw_req_s;  wire [R_KW-1:0] rw_k_s;
    reg  [8*N_EXPERT-1:0]  rw_col_s;  reg [16*N_EXPERT*R_NB-1:0] rw_scale_s;
    wire                   fw_req_s;  wire [1:0] fw_sel_s;  wire [FF_GWD-1:0] fw_grp_s;  wire [FF_KWD-1:0] fw_k_s;
    wire                   fw_shared_s;  wire [EIDXW-1:0] fw_eidx_s;
    reg  [8*TN-1:0]        fw_col_s, fw_col_up_s;  reg [16*TN*FF_NB_D-1:0] fw_scale_g_s, fw_scale_u_s;
    wire                   lw_req_s;  wire [VTW-1:0] lw_vtile_s;  wire [DIMW-1:0] lw_k_s;
    reg  [LM_TN*16-1:0]    lw_col_s;

    `MTP_RESP(_s)

    mtp_head_fp8 #(
        .MODEL_DIM(MODEL_DIM), .VOCAB(VOCAB), .H_HEADS(H_HEADS), .NOPE(NOPE),
        .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
        .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N),
        .PE_M(1), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), .BLK(BLK),
        .LM_TN(LM_TN), .PROJ_TN(PROJ_TN)
    ) dut_s (
        .clk(clk), .rst(rst), .start(start_s), .busy(busy_s), .done(done_s),
        .mode(mode_s), .pos(pos_s), .s_len(s_len_s), .h_t(h_t_s), .emb_t1(emb_s),
        .logits(logits_s), .argmax(argmax_s), .h_mtp(h_mtp_s),
        .cn_req(cn_req_s), .cn_which(cn_which_s), .cn_idx(cn_idx_s), .cn_val(cn_val_s),
        .pw_req(pw_req_s), .pw_ptile(pw_ptile_s), .pw_k(pw_k_s),
        .pw_col(pw_col_s), .pw_scale(pw_scale_s),
        .gn_req(gn_req_s), .gn_which(gn_which_s), .gn_idx(gn_idx_s), .gn_val(gn_val_s),
        .aw_req(aw_req_s), .aw_sel(aw_sel_s), .aw_grp(aw_grp_s), .aw_k(aw_k_s),
        .aw_col(aw_col_s), .aw_scale(aw_scale_s),
        .kc_req(kc_req_s), .kc_idx(kc_idx_s), .kc_ckv(kc_ckv_s), .kc_krope(kc_krope_s), .kc_valid(kc_valid_s),
        .rw_req(rw_req_s), .rw_k(rw_k_s), .rw_col(rw_col_s), .rw_scale(rw_scale_s),
        .fw_req(fw_req_s), .fw_sel(fw_sel_s), .fw_grp(fw_grp_s), .fw_k(fw_k_s),
        .fw_shared(fw_shared_s), .fw_eidx(fw_eidx_s),
        .fw_col(fw_col_s), .fw_col_up(fw_col_up_s),
        .fw_scale_g(fw_scale_g_s), .fw_scale_u(fw_scale_u_s),
        .lw_req(lw_req_s), .lw_vtile(lw_vtile_s), .lw_k(lw_k_s), .lw_col(lw_col_s)
    );
    always @(posedge clk) if (rst) mode_q_s<=1'b0; else if (start_s) mode_q_s<=mode_s;

    // ======================== DUT B : PE_M = B batch ===========================
    reg                    start_b, mode_b;
    reg  [POSW-1:0]        pos_b;
    reg  [IDXW:0]          s_len_b;
    reg  [MODEL_DIM*16*B-1:0] h_t_b, emb_b;
    wire                   busy_b, done_b;
    wire [VOCAB*16*B-1:0]  logits_b;
    wire [TOKW*B-1:0]      argmax_b;
    wire [MODEL_DIM*16*B-1:0] h_mtp_b;
    wire                   cn_req_b;  wire [1:0] cn_which_b;  wire [DIMW-1:0] cn_idx_b;
    reg  [15:0]            cn_val_b;
    wire                   pw_req_b;  wire [PTW-1:0] pw_ptile_b;  wire [CKIW-1:0] pw_k_b;
    reg  [PROJ_TN*8-1:0]   pw_col_b;  reg [16*PROJ_TN*PROJ_NB-1:0] pw_scale_b;
    wire                   gn_req_b, gn_which_b;  wire [DIMW-1:0] gn_idx_b;
    reg  [15:0]            gn_val_b;
    wire                   aw_req_b;  wire [3:0] aw_sel_b;  wire [A_GRPW-1:0] aw_grp_b;  wire [A_KCW-1:0] aw_k_b;
    reg  [PE_N*8-1:0]      aw_col_b;  reg [16*PE_N*A_NB-1:0] aw_scale_b;
    wire                   kc_req_b;  wire [IDXW-1:0] kc_idx_b;
    reg  [KV_LORA*16-1:0]  kc_ckv_b;  reg [ROPE*16-1:0] kc_krope_b;  reg kc_valid_b;
    wire                   rw_req_b;  wire [R_KW-1:0] rw_k_b;
    reg  [8*N_EXPERT-1:0]  rw_col_b;  reg [16*N_EXPERT*R_NB-1:0] rw_scale_b;
    wire                   fw_req_b;  wire [1:0] fw_sel_b;  wire [FF_GWD-1:0] fw_grp_b;  wire [FF_KWD-1:0] fw_k_b;
    wire                   fw_shared_b;  wire [EIDXW-1:0] fw_eidx_b;
    reg  [8*TN-1:0]        fw_col_b, fw_col_up_b;  reg [16*TN*FF_NB_D-1:0] fw_scale_g_b, fw_scale_u_b;
    wire                   lw_req_b;  wire [VTW-1:0] lw_vtile_b;  wire [DIMW-1:0] lw_k_b;
    reg  [LM_TN*16-1:0]    lw_col_b;

    `MTP_RESP(_b)

    mtp_head_fp8 #(
        .MODEL_DIM(MODEL_DIM), .VOCAB(VOCAB), .H_HEADS(H_HEADS), .NOPE(NOPE),
        .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
        .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N),
        .PE_M(B), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), .BLK(BLK),
        .LM_TN(LM_TN), .PROJ_TN(PROJ_TN)
    ) dut_b (
        .clk(clk), .rst(rst), .start(start_b), .busy(busy_b), .done(done_b),
        .mode(mode_b), .pos(pos_b), .s_len(s_len_b), .h_t(h_t_b), .emb_t1(emb_b),
        .logits(logits_b), .argmax(argmax_b), .h_mtp(h_mtp_b),
        .cn_req(cn_req_b), .cn_which(cn_which_b), .cn_idx(cn_idx_b), .cn_val(cn_val_b),
        .pw_req(pw_req_b), .pw_ptile(pw_ptile_b), .pw_k(pw_k_b),
        .pw_col(pw_col_b), .pw_scale(pw_scale_b),
        .gn_req(gn_req_b), .gn_which(gn_which_b), .gn_idx(gn_idx_b), .gn_val(gn_val_b),
        .aw_req(aw_req_b), .aw_sel(aw_sel_b), .aw_grp(aw_grp_b), .aw_k(aw_k_b),
        .aw_col(aw_col_b), .aw_scale(aw_scale_b),
        .kc_req(kc_req_b), .kc_idx(kc_idx_b), .kc_ckv(kc_ckv_b), .kc_krope(kc_krope_b), .kc_valid(kc_valid_b),
        .rw_req(rw_req_b), .rw_k(rw_k_b), .rw_col(rw_col_b), .rw_scale(rw_scale_b),
        .fw_req(fw_req_b), .fw_sel(fw_sel_b), .fw_grp(fw_grp_b), .fw_k(fw_k_b),
        .fw_shared(fw_shared_b), .fw_eidx(fw_eidx_b),
        .fw_col(fw_col_b), .fw_col_up(fw_col_up_b),
        .fw_scale_g(fw_scale_g_b), .fw_scale_u(fw_scale_u_b),
        .lw_req(lw_req_b), .lw_vtile(lw_vtile_b), .lw_k(lw_k_b), .lw_col(lw_col_b)
    );
    always @(posedge clk) if (rst) mode_q_b<=1'b0; else if (start_b) mode_q_b<=mode_b;

    // ================= per-port weight-fetch beat counters =====================
    // reset on the run's start pulse; count every cycle the req port is high.
    integer c_cn_s,c_pw_s,c_gn_s,c_aw_s,c_kc_s,c_rw_s,c_fw_s,c_lw_s;
    integer c_cn_b,c_pw_b,c_gn_b,c_aw_b,c_kc_b,c_rw_b,c_fw_b,c_lw_b;
    always @(posedge clk) begin
        if (start_s) begin c_cn_s<=0;c_pw_s<=0;c_gn_s<=0;c_aw_s<=0;c_kc_s<=0;c_rw_s<=0;c_fw_s<=0;c_lw_s<=0; end
        else begin
            if (cn_req_s) c_cn_s<=c_cn_s+1; if (pw_req_s) c_pw_s<=c_pw_s+1;
            if (gn_req_s) c_gn_s<=c_gn_s+1; if (aw_req_s) c_aw_s<=c_aw_s+1;
            if (kc_req_s) c_kc_s<=c_kc_s+1; if (rw_req_s) c_rw_s<=c_rw_s+1;
            if (fw_req_s) c_fw_s<=c_fw_s+1; if (lw_req_s) c_lw_s<=c_lw_s+1;
        end
        if (start_b) begin c_cn_b<=0;c_pw_b<=0;c_gn_b<=0;c_aw_b<=0;c_kc_b<=0;c_rw_b<=0;c_fw_b<=0;c_lw_b<=0; end
        else begin
            if (cn_req_b) c_cn_b<=c_cn_b+1; if (pw_req_b) c_pw_b<=c_pw_b+1;
            if (gn_req_b) c_gn_b<=c_gn_b+1; if (aw_req_b) c_aw_b<=c_aw_b+1;
            if (kc_req_b) c_kc_b<=c_kc_b+1; if (rw_req_b) c_rw_b<=c_rw_b+1;
            if (fw_req_b) c_fw_b<=c_fw_b+1; if (lw_req_b) c_lw_b<=c_lw_b+1;
        end
    end

    // ================= reference storage (per row) =============================
    reg [VOCAB*16-1:0]     ref_logits [0:B-1];
    reg [TOKW-1:0]         ref_argmax [0:B-1];
    reg [MODEL_DIM*16-1:0] ref_hmtp   [0:B-1];
    // frozen ONE-PE_M=1-run per-port beat counts (row 0 of the current mode)
    integer o_cn,o_pw,o_gn,o_aw,o_kc,o_rw,o_fw,o_lw;

    integer pass_cnt, fail_cnt;
    integer wd;

    // ---- run PE_M=1 on row `row`, capture ref outputs (+ counts on row 0) ----
    task run_ref; input integer row; begin
        @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) begin
            h_t_s[16*i +: 16] = in_h[row][i];
            emb_s[16*i +: 16] = in_e[row][i];
        end
        @(negedge clk);
        start_s = 1'b1; @(negedge clk); start_s = 1'b0;
        wd = 0;
        while (done_s !== 1'b1) begin
            @(negedge clk);
            wd = wd + 1;
            if (wd > 2000000) begin $display("FATAL: ref row %0d timeout", row); $fatal(1,"timeout"); end
        end
        ref_logits[row] = logits_s;
        ref_argmax[row] = argmax_s;
        ref_hmtp[row]   = h_mtp_s;
    end endtask

    // ---- run PE_M=B once (all rows), share the one weight stream ----
    task run_batch; begin
        @(negedge clk);
        for (rw2=0;rw2<B;rw2=rw2+1)
            for (i=0;i<MODEL_DIM;i=i+1) begin
                h_t_b[16*(MODEL_DIM*rw2+i) +: 16] = in_h[rw2][i];
                emb_b[16*(MODEL_DIM*rw2+i) +: 16] = in_e[rw2][i];
            end
        @(negedge clk);
        start_b = 1'b1; @(negedge clk); start_b = 1'b0;
        wd = 0;
        while (done_b !== 1'b1) begin
            @(negedge clk);
            wd = wd + 1;
            if (wd > 2000000) begin $display("FATAL: batch timeout"); $fatal(1,"timeout"); end
        end
    end endtask

    integer row, bt;
    reg [VOCAB*16-1:0]     gl_b;
    reg [MODEL_DIM*16-1:0] gh_b;
    reg [TOKW-1:0]         ga_b;

    task cmp_row; input integer row; begin
        // logits row slice
        gl_b = logits_b[VOCAB*16*row +: VOCAB*16];
        for (bt=0;bt<VOCAB*16;bt=bt+1)
            if (gl_b[bt]===1'bx || gl_b[bt]===1'bz) begin
                $display("FAIL: X/Z in PE_M=%0d logits row %0d bit %0d", B, row, bt);
                fail_cnt=fail_cnt+1;
            end
        if (gl_b !== ref_logits[row]) begin
            $display("FAIL: row %0d logits  batch=%h  ref=%h  (NOT bit-identical)", row, gl_b, ref_logits[row]);
            fail_cnt=fail_cnt+1;
        end else pass_cnt=pass_cnt+1;
        // argmax row slice
        ga_b = argmax_b[TOKW*row +: TOKW];
        if ((^ga_b)===1'bx) begin
            $display("FAIL: X/Z in PE_M=%0d argmax row %0d", B, row); fail_cnt=fail_cnt+1;
        end else if (ga_b !== ref_argmax[row]) begin
            $display("FAIL: row %0d argmax batch=%0d ref=%0d", row, ga_b, ref_argmax[row]);
            fail_cnt=fail_cnt+1;
        end else pass_cnt=pass_cnt+1;
        // h_mtp row slice
        gh_b = h_mtp_b[MODEL_DIM*16*row +: MODEL_DIM*16];
        for (bt=0;bt<MODEL_DIM*16;bt=bt+1)
            if (gh_b[bt]===1'bx || gh_b[bt]===1'bz) begin
                $display("FAIL: X/Z in PE_M=%0d h_mtp row %0d bit %0d", B, row, bt);
                fail_cnt=fail_cnt+1;
            end
        if (gh_b !== ref_hmtp[row]) begin
            $display("FAIL: row %0d h_mtp  batch=%h  ref=%h  (NOT bit-identical)", row, gh_b, ref_hmtp[row]);
            fail_cnt=fail_cnt+1;
        end else pass_cnt=pass_cnt+1;
    end endtask

    // ---- one scenario: build stimulus, run refs + batch, check equivalence ----
    task run_case; input integer seed0; input integer band; input integer pp; input integer ss; input integer md;
        begin
            build_stimulus(seed0, band);
            mode_s=md[0]; mode_b=md[0];
            pos_s=pp[POSW-1:0]; pos_b=pp[POSW-1:0];
            s_len_s=ss[IDXW:0]; s_len_b=ss[IDXW:0];
            for (row=0; row<B; row=row+1) run_ref(row);
            run_batch;
            for (row=0; row<B; row=row+1) cmp_row(row);
        end
    endtask

    initial begin
        pass_cnt=0; fail_cnt=0;
        start_s=0; start_b=0; mode_s=0; mode_b=0; pos_s=0; pos_b=0; s_len_s=0; s_len_b=0;
        h_t_s=0; emb_s=0; h_t_b=0; emb_b=0;
        rst=1'b1; repeat(4) @(negedge clk); rst=1'b0; @(negedge clk);

        // ================= DENSE path (mode=0): equivalence + FULL weight-share =
        run_case(1, 0, 0, 1, 0);      // S=1, pos=0
        // freeze ONE PE_M=1 run's per-port beat counts (row-2 ref just ran; counts
        // are structural per mode, so any single dense run is representative).
        o_cn=c_cn_s; o_pw=c_pw_s; o_gn=c_gn_s; o_aw=c_aw_s;
        o_kc=c_kc_s; o_rw=c_rw_s; o_fw=c_fw_s; o_lw=c_lw_s;

        // WEIGHT-SHARE: PE_M=B batch asserts EACH weight-fetch port the SAME #beats
        // as ONE PE_M=1 run (dense: NO per-row expert divergence -> every port,
        // incl. fw_req/rw_req, matches exactly; one fetch stream feeds all B rows).
        `define WCHK(NM,CB,C1) \
            if (CB !== C1) begin \
                $display("FAIL: weight port %0s beats differ: PE_M=%0d=%0d  PE_M=1=%0d", NM, B, CB, C1); \
                fail_cnt=fail_cnt+1; \
            end else pass_cnt=pass_cnt+1;
        `WCHK("cn(norm gamma)", c_cn_b, o_cn)
        `WCHK("pw(combine-proj)", c_pw_b, o_pw)
        `WCHK("gn(decoder gamma)", c_gn_b, o_gn)
        `WCHK("aw(attention)", c_aw_b, o_aw)
        `WCHK("kc(kv-cache)", c_kc_b, o_kc)
        `WCHK("rw(router)", c_rw_b, o_rw)
        `WCHK("fw(ffn expert)", c_fw_b, o_fw)
        `WCHK("lw(lm head)", c_lw_b, o_lw)
        if (fail_cnt==0)
            $display("WEIGHT-SHARE OK: PE_M=%0d run issued {cn=%0d pw=%0d gn=%0d aw=%0d kc=%0d rw=%0d fw=%0d lw=%0d} weight beats == ONE PE_M=1 run (same) -> %0d rows, 1 fetch stream",
                     B, c_pw_b, c_pw_b, c_gn_b, c_aw_b, c_kc_b, c_rw_b, c_fw_b, c_lw_b, B);

        run_case(400, 1, 7, 2, 0);    // dense, wide band, S=2, pos>0

        // ================= MoE path (mode=1): per-row equivalence ===============
        // (router picks per-row experts -> fw/rw legitimately iterate more experts
        //  at PE_M>1, so only the per-row-output equivalence is asserted here.)
        run_case(900, 0, 0, 2, 1);    // MoE, S=2, pos=0
        run_case(1300, 1, 5, 1, 1);   // MoE, wide band, S=1, pos>0

        if (fail_cnt != 0) begin
            $display("FAILED: %0d mismatch(es), %0d passed.", fail_cnt, pass_cnt);
            $fatal(1, "mtp_head_fp8 PE_M batching TB failed");
        end else begin
            $display("ALL %0d TESTS PASSED", pass_cnt);
        end
        $finish;
    end

    initial begin
        #2000000000;
        $display("FAIL: global timeout"); $fatal(1, "timeout");
    end
endmodule
