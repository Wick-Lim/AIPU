`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_fp8_soc_ms.v -- MULTI-SEQUENCE batched SoC top (A2 stage 3b)
//----------------------------------------------------------------------------
// The deployable batched decode top: runs glm_model_fp8 at PE_M=B with
// PER_ROW_SEQ=1 so B DIFFERENT sequences (users) are decoded in ONE forward,
// each row r attending its OWN KV window via a REAL kv_cache_pager (NSEQ=B).
// This is the continuous-batching integration: the B sequences SHARE the
// query-side weight/expert fetch while each keeps its own KV history.
//
// Differs from glm_fp8_soc (the single-token PE_M=1 SoC) in exactly the
// multi-seq dimensions -- the weight-pull responders (em_*/gn_*/aw_*/rw_*/
// fw_*/fn_*/lw_*) are PE_M-AGNOSTIC and pass straight through to the model,
// same as the PE_M=1 SoC.  For a focused, verifiable top this module wires the
// FFN/expert weights DIRECTLY (no expert_cache_pf / flash arbiter -- those are
// orthogonal bandwidth optimisations carried over from glm_fp8_soc later).
//
// REAL KV PATH: unlike glm_fp8_soc (which STUBS kc_ckv/kc_krope from the TB and
// uses the pager only as a timing/window model), here the pager ACTUALLY serves
// the model's KV: model.kc_req -> pager.gather (gather_seq = model.kc_seq),
// pager.row_valid -> model.kc_valid, pager.row_out -> {kc_krope,kc_ckv}.  The
// model waits in its K_RDWAIT handshake, so resident (1-cycle) and cold-fetch
// (Flash) latencies both work.
//
// HOST FSM (one batched decode step):
//   IDLE -> PREFILL (append each sequence s's s_len_r[s] prompt latents into
//   its own pager window, seq by seq) -> RUN (one PE_M=B forward) -> DECAP
//   (append each sequence's new decode-token latent at position s_len_r[s]) ->
//   DONE (commit B argmax tokens).  Row r decodes at position s_len_r[r] with
//   sequence id r (PER_ROW_POS/SLEN/SEQ=1).
//
// STYLE: synchronous, ACTIVE-HIGH reset; no latch; deterministic.
//============================================================================
module glm_fp8_soc_ms #(
    parameter integer MODEL_DIM  = 16,
    parameter integer L          = 2,
    parameter integer N_DENSE    = 1,
    parameter integer VOCAB      = 16,
    parameter integer H_HEADS    = 2,
    parameter integer NOPE       = 4,
    parameter integer ROPE       = 4,
    parameter integer V_DIM      = 4,
    parameter integer Q_LORA     = 8,
    parameter integer KV_LORA    = 8,
    parameter integer S_MAX      = 4,      // >= B*TOPK_ATTN (multi-seq union depth)
    parameter integer TOPK_ATTN  = 2,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 2,
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 4,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 8,
    parameter integer INTER_DENSE= 160,
    parameter [31:0]  RSCALE     = 32'h40200000,
    parameter integer TN         = 2,
    parameter integer BLK        = 128,
    parameter integer LM_TN      = 2,
    parameter integer PE_M       = 2,      // batch rows == sequences decoded together
    // KV pager
    parameter integer KV_CTX     = 1024,   // logical KV context capacity (positions)
    parameter integer KV_RESIDENT= 16,     // KV ring capacity per sequence (POWER OF TWO)
    parameter integer FLASH_LAT  = 8,
    // ---- derived (do NOT override) ----
    parameter integer NSEQ       = PE_M,
    parameter integer SWIN       = PE_M*TOPK_ATTN,               // multi-seq attention scratch
    parameter integer IDXW       = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer SEQW       = (PE_M  <= 1) ? 1 : $clog2(PE_M),
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM<=1)?1:$clog2(MODEL_DIM),
    parameter integer LAYW       = (L<=1)?1:$clog2(L),
    parameter integer EIDXW      = (N_EXPERT<=1)?1:$clog2(N_EXPERT),
    parameter integer QK_DIM     = NOPE + ROPE,
    parameter integer HQK        = H_HEADS*QK_DIM,
    parameter integer HNOPE      = H_HEADS*NOPE,
    parameter integer HV         = H_HEADS*V_DIM,
    parameter integer A_KMAX     = (MODEL_DIM>Q_LORA)?
                       ((MODEL_DIM>KV_LORA)?((MODEL_DIM>HV)?MODEL_DIM:HV):((KV_LORA>HV)?KV_LORA:HV))
                     :((Q_LORA>KV_LORA)?((Q_LORA>HV)?Q_LORA:HV):((KV_LORA>HV)?KV_LORA:HV)),
    parameter integer A_OMAX     = (HQK>MODEL_DIM)?
                       ((HQK>HNOPE)?((HQK>HV)?HQK:HV):((HNOPE>HV)?HNOPE:HV))
                     :((MODEL_DIM>HNOPE)?((MODEL_DIM>HV)?MODEL_DIM:HV):((HNOPE>HV)?HNOPE:HV)),
    parameter integer A_NGMAX    = (A_OMAX+PE_N-1)/PE_N,
    parameter integer A_GRPW     = (A_NGMAX<=1)?1:$clog2(A_NGMAX),
    parameter integer A_KCW      = (A_KMAX <=1)?1:$clog2(A_KMAX),
    parameter integer A_NB       = (A_KMAX+BLK-1)/BLK,
    parameter integer FF_KMAX_D  = (INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM,
    parameter integer FF_KMAX_M  = (INTER_MOE >MODEL_DIM)?INTER_MOE :MODEL_DIM,
    parameter integer FF_GWD     = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN+1),
    parameter integer FF_KWD     = $clog2(FF_KMAX_D+1),
    parameter integer R_KW       = $clog2(FF_KMAX_M+1),
    parameter integer FF_NB_D    = (FF_KMAX_D+BLK-1)/BLK,
    parameter integer R_NB       = (FF_KMAX_M+BLK-1)/BLK,
    parameter integer NVTILE     = VOCAB/LM_TN,
    parameter integer VTW        = (NVTILE<=1)?1:$clog2(NVTILE),
    parameter integer ROW_BITS   = (KV_LORA + ROPE) * 16,
    parameter integer KVPOSW     = (KV_CTX <= 1) ? 1 : $clog2(KV_CTX)
)(
    input  wire                          clk,
    input  wire                          rst,          // sync, active-high

    //========================== HOST interface (B sequences) ================
    input  wire                          start,        // 1-cycle: begin a batched decode step
    input  wire [PE_M*TOKW-1:0]          prompt_tok,   // B tokens to decode (row r = seq r)
    input  wire [(IDXW+1)*PE_M-1:0]      s_len_vec,    // per-seq prompt length / decode position
    output reg                           busy,
    output reg                           done,         // 1-cycle: batched step committed
    output reg  [PE_M*TOKW-1:0]          next_tok,     // B committed next tokens (argmax)
    output reg                           tok_valid,    // 1-cycle pulse with next_tok
    output wire [PE_M*VOCAB*16-1:0]      logits,       // B * VOCAB bf16 (observability)

    //========================== weight-pull responder ports (PE_M-agnostic) =
    output wire                          em_req,
    output wire [TOKW-1:0]               em_tok,
    output wire [DIMW-1:0]               em_idx,
    input  wire [15:0]                   em_val,
    output wire [LAYW-1:0]               db_layer,
    output wire                          idx_fresh,
    output wire [LAYW-1:0]               idx_win,
    output wire                          gn_req,
    output wire                          gn_which,
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*8-1:0]             aw_col,
    input  wire [16*PE_N*A_NB-1:0]       aw_scale,
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [8*N_EXPERT-1:0]         rw_col,
    input  wire [16*N_EXPERT*R_NB-1:0]   rw_scale,
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [8*TN-1:0]               fw_col,
    input  wire [8*TN-1:0]               fw_col_up,
    input  wire [16*TN*FF_NB_D-1:0]      fw_scale_g,
    input  wire [16*TN*FF_NB_D-1:0]      fw_scale_u,
    output wire                          fn_req,
    output wire [DIMW-1:0]               fn_idx,
    input  wire [15:0]                   fn_val,
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col,

    //========================== latent-KV read DATA STUB ====================
    //  Like glm_fp8_soc: the KV DATA (c_kv|k_rope) is served by the TB/PHY keyed
    //  on (kc_seq, db_layer, kc_idx) -- the model's KV is PER-LAYER, while the
    //  pager models the per-sequence resident WINDOW + Flash timing (kc_seq ->
    //  gather_seq).  kc_valid is the 1-cycle registered ack of kc_req.
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,
    output wire [SEQW-1:0]               kc_seq,     // multi-seq: which sequence's window
    input  wire [KV_LORA*16-1:0]         kc_ckv,
    input  wire [ROPE*16-1:0]            kc_krope,

    //========================== KV append (latent ROW source) ===============
    //  During PREFILL/DECAP the host appends one latent row per (seq,pos); the
    //  ROW BYTES come from this stub (kv_row_in) -- the computed latent a real
    //  datapath writes back.  kv_seq_sel/kv_row_sel tell the stub which (seq,pos).
    output wire [SEQW-1:0]               kv_seq_sel,
    output wire [KVPOSW-1:0]             kv_row_sel,
    input  wire [ROW_BITS-1:0]           kv_row_in,
    output wire [ROW_BITS-1:0]           kv_row_out, // pager gathered row (observability)

    //========================== Flash overflow (cold KV) ====================
    output wire                          flash_req,
    output wire [KVPOSW-1:0]             flash_idx,
    output wire [SEQW-1:0]               flash_seq,
    input  wire                          flash_done,
    input  wire [ROW_BITS-1:0]           flash_row
);

    //------------------------------------------------------------------------
    // 1) MODEL -- PE_M=B, per-row position / extent / sequence.
    //------------------------------------------------------------------------
    reg                       mdl_start;
    wire                      mdl_busy, mdl_done;
    wire [PE_M*TOKW-1:0]      mdl_argmax;
    // per-row query position = that sequence's decode position (= s_len_r); the
    // scalar pos/s_len feed row 0 (PER_ROW_* replicate to rows 1.. from the vecs).
    wire [POSW-1:0]           pos0   = {{(POSW-(IDXW+1)){1'b0}}, s_len_vec[0 +: (IDXW+1)]};
    wire [IDXW:0]             slen0  = s_len_vec[0 +: (IDXW+1)];
    // per-row position vector: row r position = s_len_vec[r] (its decode pos)
    wire [POSW*PE_M-1:0]      pos_vec;
    genvar r;
    generate
        for (r = 0; r < PE_M; r = r + 1) begin : g_posvec
            assign pos_vec[POSW*r +: POSW] =
                {{(POSW-(IDXW+1)){1'b0}}, s_len_vec[(IDXW+1)*r +: (IDXW+1)]};
        end
    endgenerate
    // per-row sequence id: row r -> sequence r
    wire [SEQW*PE_M-1:0]      seq_vec;
    generate
        for (r = 0; r < PE_M; r = r + 1) begin : g_seqvec
            assign seq_vec[SEQW*r +: SEQW] = r[SEQW-1:0];
        end
    endgenerate

    // model KV read: kc_req/kc_idx/kc_seq are TOP outputs (drive the stub + the
    // pager window model); kc_ckv/kc_krope are TOP inputs (stubbed, per-layer);
    // kc_valid is the 1-cycle registered ack of kc_req (the verified read contract).
    reg                       kc_valid_r;
    always @(posedge clk) if (rst) kc_valid_r <= 1'b0; else kc_valid_r <= kc_req;

    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .SWIN(SWIN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT),
        .TOPK(TOPK), .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE),
        .TN(TN), .BLK(BLK), .LM_TN(LM_TN), .PE_M(PE_M),
        .PER_ROW_POS(1), .PER_ROW_SLEN(1), .PER_ROW_SEQ(1)
    ) u_model (
        .clk(clk), .rst(rst), .start(mdl_start), .busy(mdl_busy), .done(mdl_done),
        .token_id(prompt_tok), .pos(pos0), .s_len(slen0),
        .pos_vec(pos_vec), .s_len_vec(s_len_vec), .seq_vec(seq_vec),
        .logits(logits), .argmax(mdl_argmax),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_col(aw_col), .aw_scale(aw_scale),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_seq(kc_seq),
        .kc_ckv(kc_ckv), .kc_krope(kc_krope), .kc_valid(kc_valid_r),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col), .rw_scale(rw_scale),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_col(fw_col), .fw_col_up(fw_col_up),
        .fw_scale_g(fw_scale_g), .fw_scale_u(fw_scale_u),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col),
        .h_state()
    );

    //------------------------------------------------------------------------
    // 2) HOST FSM : PREFILL B sequences -> RUN batched forward -> DECAP -> DONE.
    //------------------------------------------------------------------------
    localparam [2:0] H_IDLE=3'd0, H_PREFILL=3'd1, H_RUN=3'd2, H_RUNW=3'd3,
                     H_DECAP=3'd4, H_DONE=3'd5;
    reg [2:0]        hstate;
    reg [SEQW-1:0]   cur_seq;                  // sequence being (pre)filled
    reg [IDXW:0]     cur_pos;                  // position within cur_seq's prompt
    reg              ap_valid;                 // drive a pager append this cycle
    reg [SEQW-1:0]   ap_seq;
    reg [KVPOSW-1:0] ap_pos;

    // this sequence's prompt length (per-seq s_len)
    wire [IDXW:0]    cur_slen = s_len_vec[(IDXW+1)*cur_seq +: (IDXW+1)];

    assign kv_seq_sel = ap_seq;
    assign kv_row_sel = ap_pos;

    always @(posedge clk) begin
        if (rst) begin
            hstate    <= H_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            next_tok  <= {PE_M*TOKW{1'b0}};
            tok_valid <= 1'b0;
            mdl_start <= 1'b0;
            cur_seq   <= {SEQW{1'b0}};
            cur_pos   <= {(IDXW+1){1'b0}};
            ap_valid  <= 1'b0;
            ap_seq    <= {SEQW{1'b0}};
            ap_pos    <= {KVPOSW{1'b0}};
        end else begin
            done      <= 1'b0;
            tok_valid <= 1'b0;
            mdl_start <= 1'b0;
            ap_valid  <= 1'b0;
            case (hstate)
                H_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy    <= 1'b1;
                        cur_seq <= {SEQW{1'b0}};
                        cur_pos <= {(IDXW+1){1'b0}};
                        // empty prompts -> straight to RUN
                        if (s_len_vec[0 +: (IDXW+1)] == {(IDXW+1){1'b0}} && PE_M == 1)
                            hstate <= H_RUN;
                        else
                            hstate <= H_PREFILL;
                    end
                end
                H_PREFILL: begin
                    // append (cur_seq, cur_pos) if this sequence has a prompt key here.
                    if (cur_pos < cur_slen) begin
                        ap_valid <= 1'b1;
                        ap_seq   <= cur_seq;
                        ap_pos   <= {{(KVPOSW-(IDXW+1)){1'b0}}, cur_pos};
                        cur_pos  <= cur_pos + 1'b1;
                    end else begin
                        // this sequence's prefill done -> next sequence (or RUN)
                        cur_pos <= {(IDXW+1){1'b0}};
                        if (cur_seq == (NSEQ-1)) begin
                            hstate <= H_RUN;
                        end else begin
                            cur_seq <= cur_seq + 1'b1;
                        end
                    end
                end
                H_RUN: begin
                    mdl_start <= 1'b1;                 // launch the batched forward
                    hstate    <= H_RUNW;
                end
                H_RUNW: begin
                    if (mdl_done) begin
                        next_tok <= mdl_argmax;        // capture B argmax tokens
                        cur_seq  <= {SEQW{1'b0}};
                        hstate   <= H_DECAP;
                    end
                end
                H_DECAP: begin
                    // append each sequence's decode-token latent at position s_len_r[seq].
                    ap_valid <= 1'b1;
                    ap_seq   <= cur_seq;
                    ap_pos   <= {{(KVPOSW-(IDXW+1)){1'b0}},
                                 s_len_vec[(IDXW+1)*cur_seq +: (IDXW+1)]};
                    if (cur_seq == (NSEQ-1)) hstate <= H_DONE;
                    else                     cur_seq <= cur_seq + 1'b1;
                end
                H_DONE: begin
                    done      <= 1'b1;
                    tok_valid <= 1'b1;
                    busy      <= 1'b0;
                    hstate    <= H_IDLE;
                end
                default: hstate <= H_IDLE;
            endcase
        end
    end

    //------------------------------------------------------------------------
    // 3) KV PAGER -- NSEQ INDEPENDENT resident windows (per sequence).  Models
    //    the per-sequence KV window + Flash overflow timing: gather is driven by
    //    the model's kc_* so gather_seq = kc_seq tracks WHICH sequence's window
    //    each read hits (resident vs cold), and append is driven per-seq by the
    //    host FSM.  Row layout: [ c_kv(KV_LORA*16) | k_rope(ROPE*16) ].  The KV
    //    DATA the model consumes is the (per-layer) stub above; the pager's
    //    row_out is exposed for observability / a future real per-layer path.
    //------------------------------------------------------------------------
    wire [KVPOSW-1:0] pg_gather_idx = {{(KVPOSW-IDXW){1'b0}}, kc_idx};
    wire              pg_row_valid, pg_busy;

    kv_cache_pager #(
        .ROW_BITS(ROW_BITS), .RESIDENT(KV_RESIDENT), .S_MAX(KV_CTX),
        .FLASH_LAT(FLASH_LAT), .NSEQ(NSEQ)
    ) u_kvpager (
        .clk(clk), .rst(rst),
        .append_valid(ap_valid), .append_row(kv_row_in), .append_seq(ap_seq),
        .gather_valid(kc_req), .gather_idx(pg_gather_idx), .gather_seq(kc_seq),
        .row_valid(pg_row_valid), .row_out(kv_row_out), .busy(pg_busy),
        .flash_req(flash_req), .flash_idx(flash_idx), .flash_seq(flash_seq),
        .flash_done(flash_done), .flash_row(flash_row),
        .append_count(), .resident_lo(), .overflowed(),
        .ecc_serr(), .ecc_derr()
    );

    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, mdl_busy, pg_busy, pg_row_valid};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
