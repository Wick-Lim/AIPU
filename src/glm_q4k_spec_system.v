`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_q4k_spec_system.v  --  GLM-5.2-Q4_K CLOSED SPECULATIVE-DECODE SYSTEM  (5c)
//                            (docs/SPEC_COMPOSITION_DESIGN.md step 5c)
//----------------------------------------------------------------------------
// WHAT THIS IS
//   The composed SPECULATING top: it closes the batched speculative-decode loop
//   around the FULL memory-system die glm_q4k_system (PE_M=K+1, SELF_KV=1,
//   INTRA_CAUSAL=1, PER_ROW_POS=1 -- the 5b-sys position-accurate batched verify)
//   plus the committed accept/reject controller spec_decode_seq (batch g_kn path),
//   plus the KV WRITE-BACK of the committed prefix.  Its committed token STREAM is
//   a prefix of the model's greedy decode for ANY K and ANY draft accept rate
//   (the spec==greedy invariant), because it ONLY ever commits the model's own
//   argmaxes (truth_vec), never a raw draft.
//
//   RELATION TO spec_batched_top:  spec_batched_top wraps the bare glm_model_q4k
//   (KV/weights to TB stubs, SHARED-position verify, NO memory system, NO KV
//   write-back).  glm_q4k_spec_system wraps glm_q4k_system -- the REAL memory
//   system (weight_loader / expert_cache / ddr5_xbar / kv_cache_pager + the
//   die-internal per-(layer,pos) KV write-back) -- and runs a POSITION-ACCURATE
//   batched verify (INTRA_CAUSAL) with a CORRECT multi-token KV write-back across
//   passes, so the batched pass at committed length t reads the true shared prefix
//   0..t-1 from the pager.  That is the 5c deliverable.
//
//----------------------------------------------------------------------------
// THE LOOP  (one glm_q4k_system PE_M=K+1 weight-load per outer pass)
//   State: cur_tok @ committed length t (= pager append count per layer window);
//   the pager holds every layer's positions 0..t-1 (written by prior write-backs).
//   For pass i = 0 .. num_passes-1:
//     1. SETUP : latch this pass's K drafts {d_1..d_K} (draft_in) + n_draft.
//     2. LAUNCH: assemble the PE_M=K+1 verify batch  {cur_tok, d_1..d_K}, shared
//                s_len=t, per-row pos_vec=[t, t+1, .., t+K]; pulse glm_q4k_system
//                .start -> ONE weight-load.  The die reads the shared prefix 0..t-1
//                from the pager and supplies the intra-batch keys t..t+j-1 for row
//                j INTERNALLY (INTRA_CAUSAL).  ext_append_valid is held 0 for the
//                whole pass, so the pager stays STABLE at t (no per-pass append).
//     3. WAIT  : while the die runs, CAPTURE every layer's PE_M current-token
//                latents kv_lat_row_all into buf[db_layer][row] (keyed by db_layer
//                on kv_lat_valid_all -- the SAME (layer,valid) the internal SELF_KV
//                append uses).  On done: truth_q <= argmax_o = {m_1..m_{K+1}}.
//     4. FEED  : pulse spec_decode_seq.pass_valid with draft_vec={d_1..d_K},
//                truth_vec={m_1..m_{K+1}}, n_draft -> it commits m_1..m_{p+1}
//                (p = local longest-accepted-prefix).
//     5. WBACK : write back EXACTLY the committed prefix -- for every layer m in
//                0..L-1 and every committed row r in 0..p, drive ext_append_row =
//                buf[m][r], ext_append_seq = m: window m advances t..t+p (p+1 rows).
//                The REJECTED rows p+1..K are NEVER appended -> no phantom KV at
//                positions after a reject.  Advance cur_tok <= m_{p+1}, t <= t+p+1.
//   pass_valid pulses are many cycles apart (a full die pass + the write-back
//   between them), so spec_decode_seq's drain of m_2..m_{p+1} always finishes
//   before the next pass.
//
// WHY spec==greedy HOLDS (with the KV write-back):
//   accept d_j <=> d_j == m_j.  m_1 = argmax(cur_tok @ t) = the true greedy token
//   at t+1.  A row is accepted ONLY if its INPUT draft equals the true token there,
//   so an accepted row decoded the RIGHT token at its position -> its argmax m_{r+1}
//   IS the greedy token, and buf[m][r] IS the latent greedy would have written at
//   position t+r.  So the committed stream == greedy AND the pager after write-back
//   == greedy's pager -- for ANY draft accept rate; the drafts only gate HOW MANY
//   tokens commit per weight-load.
//
// DISCIPLINE: synchronous ACTIVE-HIGH reset; no latch; no combinational loop;
//   handshake-driven (each pass waits glm_q4k_system.done).  spec_decode_seq is the
//   committer (NEVER a raw draft); glm_q4k_system is UNCHANGED except the additive
//   KV_EXT_APPEND hook (byte-identical at its default).
//============================================================================
module glm_q4k_spec_system #(
    // ---- compute-die / slice config (mirrors glm_q4k_system) ----
    parameter integer MODEL_DIM  = 16,
    parameter integer L          = 3,
    parameter integer N_DENSE    = 2,
    parameter integer VOCAB      = 16,
    parameter integer H_HEADS    = 2,
    parameter integer NOPE       = 4,
    parameter integer ROPE       = 4,
    parameter integer V_DIM      = 4,
    parameter integer Q_LORA     = 8,
    parameter integer KV_LORA    = 8,
    parameter integer S_MAX      = 16,
    parameter integer TOPK_ATTN  = 16,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 2,
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 4,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 16,
    parameter integer INTER_DENSE= 32,
    parameter [31:0]  RSCALE     = 32'h40200000,
    parameter integer TN         = 4,
    parameter integer BLK        = 128,
    parameter integer LM_TN      = 4,
    // ---- K drafts per pass -> PE_M = K+1 verify rows ----
    parameter integer DRAFT_K    = 2,
    // ---- memory-system config (mirrors glm_q4k_system) ----
    parameter integer CACHE_SLOTS = 4,
    parameter integer FLASH_LAT   = 8,
    parameter integer KV_CTX      = 1024,
    parameter integer KV_RESIDENT = 16,
    parameter integer EFIFO_DEPTH = 16,
    parameter integer DDR_NCH     = 4,
    parameter integer DDR_ADDR_W  = 32,
    parameter integer DDR_DATA_W  = 256,
    parameter integer DDR_TAG_W   = 8,
    parameter integer DDR_ROW_LAT = 10,
    parameter integer DDR_RESP_QD = 4,
    parameter integer WL_KMAX     = 256,
    parameter integer WL_ADDR_W   = 24,
    parameter integer LOADER_KLEN = MODEL_DIM,
    // ====================================================================
    // derived (do NOT override) -- mirror glm_q4k_system's port-width derivations
    // ====================================================================
    parameter integer PE_M       = DRAFT_K + 1,            // batch = K+1 verify rows
    parameter integer SWIN       = S_MAX,                  // union scratch bound (>= min(PE_M*TOPK_ATTN,S_MAX))
    parameter integer QK_DIM     = NOPE + ROPE,
    parameter integer IDXW       = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer HQK        = H_HEADS * QK_DIM,
    parameter integer HNOPE      = H_HEADS * NOPE,
    parameter integer HV         = H_HEADS * V_DIM,
    parameter integer EIDXW      = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer A_KMAX     = (MODEL_DIM > Q_LORA) ?
                               ((MODEL_DIM > KV_LORA) ?
                                ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV))
                             : ((Q_LORA > KV_LORA) ?
                                ((Q_LORA > HV) ? Q_LORA : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV)),
    parameter integer A_OMAX     = (HQK > MODEL_DIM) ?
                               ((HQK > HNOPE) ?
                                 ((HQK > HV) ? HQK : HV)
                               : ((HNOPE > HV) ? HNOPE : HV))
                             : ((MODEL_DIM > HNOPE) ?
                                 ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                               : ((HNOPE > HV) ? HNOPE : HV)),
    parameter integer A_NGMAX    = (A_OMAX + PE_N - 1) / PE_N,
    parameter integer A_GRPW     = (A_NGMAX <= 1) ? 1 : $clog2(A_NGMAX),
    parameter integer A_KCW      = (A_KMAX  <= 1) ? 1 : $clog2(A_KMAX),
    parameter integer FF_GWD     = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN + 1),
    parameter integer FF_KMAX_D  = (INTER_DENSE > MODEL_DIM) ? INTER_DENSE : MODEL_DIM,
    parameter integer FF_KWD     = $clog2(FF_KMAX_D + 1),
    parameter integer FF_KMAX_M  = (INTER_MOE  > MODEL_DIM) ? INTER_MOE  : MODEL_DIM,
    parameter integer R_KW       = $clog2(FF_KMAX_M + 1),
    parameter integer A_NSB      = (A_KMAX    + 255) / 256,
    parameter integer FF_NSB_D   = (FF_KMAX_D + 255) / 256,
    parameter integer R_NSB      = (FF_KMAX_M + 255) / 256,
    parameter integer LAYW       = (L     <= 1) ? 1 : $clog2(L),
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    parameter integer ROW_BITS   = (KV_LORA + ROPE) * 16,
    parameter integer KVPOSW     = (KV_CTX <= 1) ? 1 : $clog2(KV_CTX),
    parameter integer KV_NSEQ    = L,                      // SELF_KV=1 -> L per-layer windows
    parameter integer KV_SEQW    = (KV_NSEQ <= 1) ? 1 : $clog2(KV_NSEQ),
    parameter integer CSLOTW     = (CACHE_SLOTS <= 1) ? 1 : $clog2(CACHE_SLOTS),
    parameter integer WL_DATA_W  = 256,
    // spec_decode_seq batch interface widths
    parameter integer DKW        = (DRAFT_K <= 1) ? 1 : $clog2(DRAFT_K + 1),
    parameter integer OCW        = $clog2(DRAFT_K + 2)     // 0..K+1 accepted-prefix width
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    // ---- loop control ----
    input  wire                          start,      // 1-cycle pulse: begin the loop
    input  wire [TOKW-1:0]               prompt_tok, // first token to decode from (row 0)
    input  wire [POSW-1:0]               start_pos,  // first committed length t (usually 0)
    input  wire [15:0]                   num_passes, // # batched-verify passes (>=1)
    // K drafts for THIS pass (TB-driven; d_1..d_K = positions t+1..t+K), latched at SETUP.
    input  wire [DRAFT_K*TOKW-1:0]       draft_in,   // draft_in[j*TOKW +: TOKW] = d_{j+1}
    input  wire [DKW-1:0]                n_draft,    // #valid drafts this pass (<=K)

    output reg                           busy,
    output reg                           done,       // 1-cycle pulse: loop complete

    // ---- committed-token stream + counters (straight from spec_decode_seq) ----
    output wire                          commit_valid,
    output wire [TOKW-1:0]               commit_tok,
    output wire                          accepted,
    output wire [PE_M*VOCAB*16-1:0]      logits,       // die per-row logits (verify: full-logit bind)
    output wire [PE_M*TOKW-1:0]          argmax_o,     // die per-row argmax {m_1..m_{K+1}}
    output wire [31:0]                   total_tokens,
    output wire [31:0]                   main_passes,
    output wire [31:0]                   accepts,
    output wire [31:0]                   rejects,
    output reg  [31:0]                   weight_loads,  // ONE die weight-load per pass

    // ====================================================================
    // glm_q4k_system MEMORY INTERFACE -- routed straight up (single shared die).
    // ====================================================================
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
    input  wire [PE_N*4-1:0]             aw_q,
    input  wire [16*PE_N*A_NSB-1:0]      aw_d,
    input  wire [16*PE_N*A_NSB-1:0]      aw_dmin,
    input  wire [96*PE_N*A_NSB-1:0]      aw_scales,
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [4*N_EXPERT-1:0]         rw_q,
    input  wire [16*N_EXPERT*R_NSB-1:0]  rw_d,
    input  wire [16*N_EXPERT*R_NSB-1:0]  rw_dmin,
    input  wire [96*N_EXPERT*R_NSB-1:0]  rw_scales,
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [4*TN-1:0]               fw_q,
    input  wire [4*TN-1:0]               fw_q_up,
    input  wire [16*TN*FF_NSB_D-1:0]     fw_d_g,
    input  wire [16*TN*FF_NSB_D-1:0]     fw_dmin_g,
    input  wire [96*TN*FF_NSB_D-1:0]     fw_scales_g,
    input  wire [16*TN*FF_NSB_D-1:0]     fw_d_u,
    input  wire [16*TN*FF_NSB_D-1:0]     fw_dmin_u,
    input  wire [96*TN*FF_NSB_D-1:0]     fw_scales_u,
    output wire                          fn_req,
    output wire [DIMW-1:0]               fn_idx,
    input  wire [15:0]                   fn_val,
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col,
    input  wire [KV_LORA*16-1:0]         kc_ckv,     // unused at SELF_KV=1 (die reads pager)
    input  wire [ROPE*16-1:0]            kc_krope,
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,
    // KV stub append port (observation only under SELF_KV=1; tie kv_row_in=0)
    output wire [KVPOSW-1:0]             kv_row_sel,
    input  wire [ROW_BITS-1:0]           kv_row_in,
    // Flash channel (KV NVMe spill path; not exercised while t < KV_RESIDENT)
    output wire                          flash_req,
    output wire                          flash_is_expert,
    output wire [EIDXW-1:0]              flash_expert_id,
    output wire [KVPOSW-1:0]             flash_row_idx,
    input  wire                          flash_done,
    input  wire [ROW_BITS-1:0]           flash_row,
    input  wire                          pf_valid,
    input  wire [EIDXW-1:0]              pf_expert_id,
    // DDR5 fabric channels
    output wire [DDR_NCH-1:0]            mem_req_valid,
    input  wire [DDR_NCH-1:0]            mem_req_ready,
    output wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr,
    output wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag,
    input  wire [DDR_NCH-1:0]            mem_resp_valid,
    output wire [DDR_NCH-1:0]            mem_resp_ready,
    input  wire [DDR_NCH*DDR_DATA_W-1:0] mem_resp_data,
    input  wire [DDR_NCH*DDR_TAG_W-1:0]  mem_resp_tag,
    // weight_loader staging memory
    output wire                          wl_mem_en,
    output wire [WL_ADDR_W-1:0]          wl_mem_addr,
    input  wire [WL_DATA_W-1:0]          wl_mem_data,

    // ---- observability ----
    output wire                          mdl_busy,
    output wire [31:0]                   ec_hit_count,
    output wire [31:0]                   ec_miss_count,
    output wire [KVPOSW-1:0]             kv_append_count // pager append head (layer 0)
);
    //========================================================================
    // spec loop state
    //========================================================================
    localparam integer K    = DRAFT_K;                    // actual drafts per pass
    localparam integer SEQK = (K >= 2) ? K : 2;           // spec_decode_seq batch depth (g_kn)
    localparam integer DKW_S= (SEQK <= 1) ? 1 : $clog2(SEQK + 1);
    localparam integer KVR  = ROW_BITS;
    localparam integer LW   = (L <= 1) ? 1 : $clog2(L);   // layer index width
    localparam integer PW   = (PE_M <= 1) ? 1 : $clog2(PE_M); // row index width (0..K)

    reg  [TOKW-1:0]           cur_tok;
    reg  [POSW-1:0]           t_pos;         // committed length = pager append count/layer
    reg  [DRAFT_K*TOKW-1:0]   draft_q;       // latched K drafts d_1..d_K
    reg  [DKW-1:0]            nd_q;           // latched n_draft
    reg  [PE_M*TOKW-1:0]      truth_q;       // captured argmaxes m_1..m_{K+1}
    reg  [15:0]               pass_idx, npass_q;

    reg                       sys_start;     // pulse glm_q4k_system.start
    reg                       seq_arm, seq_pass;
    reg                       ext_av;        // pager append valid (write-back)
    reg  [KVR-1:0]            ext_row;       // pager append row
    reg  [KV_SEQW-1:0]        ext_seq;       // pager append seq (= layer)

    // per-(layer,row) captured current-token latents (the write-back source)
    reg  [KVR-1:0]            buf_lat [0:L-1][0:PE_M-1];
    reg  [LW-1:0]             wb_m;          // write-back layer cursor
    reg  [PW-1:0]             wb_r;          // write-back row cursor (0..p)

    //========================================================================
    // PE_M=K+1 verify batch token vector : {cur_tok, d_1..d_K}, row-major.
    //========================================================================
    wire [PE_M*TOKW-1:0]      sys_token_id;
    assign sys_token_id[TOKW*0 +: TOKW] = cur_tok;
    genvar gj;
    generate
    for (gj = 0; gj < DRAFT_K; gj = gj + 1) begin : G_TOK
        assign sys_token_id[TOKW*(gj+1) +: TOKW] = draft_q[TOKW*gj +: TOKW];
    end
    endgenerate

    // per-row query positions : row r ropes at position t_pos + r (PER_ROW_POS=1).
    wire [POSW*PE_M-1:0]      sys_pos_vec;
    generate
    for (gj = 0; gj < PE_M; gj = gj + 1) begin : G_POS
        assign sys_pos_vec[POSW*gj +: POSW] = t_pos + gj[POSW-1:0];
    end
    endgenerate
    // shared causal extent = committed length t (row 0); INTRA supplies the rest.
    wire [IDXW:0]             sys_slen = t_pos[IDXW:0];

    //========================================================================
    // the FULL memory-system die at PE_M = K+1 (5b-sys shape), KV_EXT_APPEND=1 so
    //   THIS loop owns the pager append.  Every memory port routes straight up.
    //========================================================================
    wire                      sys_done;
    wire [PE_M*KVR-1:0]       sys_kv_lat_row_all;
    wire [PE_M-1:0]           sys_kv_lat_valid_all;

    glm_q4k_system #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .SWIN(SWIN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT),
        .TOPK(TOPK), .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE),
        .TN(TN), .BLK(BLK), .LM_TN(LM_TN),
        .PE_M(PE_M), .PER_ROW_POS(1), .PER_ROW_SLEN(0),
        .DSA_REAL_IDX(1), .INTRA_CAUSAL(1),
        .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX),
        .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH),
        .SELF_KV(1), .KV_EXT_APPEND(1),
        .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W),
        .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD),
        .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN)
    ) u_sys (
        .clk(clk), .rst(rst),
        .start(sys_start), .prompt_tok(sys_token_id),
        .start_pos(t_pos), .s_len(sys_slen),
        .pos_vec(sys_pos_vec), .s_len_vec({(IDXW+1)*PE_M{1'b0}}),
        .seq_vec({((PE_M<=1?1:$clog2(PE_M))*PE_M){1'b0}}),
        .busy(), .done(sys_done), .next_tok(), .tok_valid(),
        .logits(logits),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_q(aw_q), .aw_d(aw_d), .aw_dmin(aw_dmin), .aw_scales(aw_scales),
        .rw_req(rw_req), .rw_k(rw_k),
        .rw_q(rw_q), .rw_d(rw_d), .rw_dmin(rw_dmin), .rw_scales(rw_scales),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_q(fw_q), .fw_q_up(fw_q_up),
        .fw_d_g(fw_d_g), .fw_dmin_g(fw_dmin_g), .fw_scales_g(fw_scales_g),
        .fw_d_u(fw_d_u), .fw_dmin_u(fw_dmin_u), .fw_scales_u(fw_scales_u),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col),
        .kc_ckv(kc_ckv), .kc_krope(kc_krope), .kc_req(kc_req), .kc_idx(kc_idx),
        .kv_row_sel(kv_row_sel), .kv_row_in(kv_row_in),
        // ---- 5c external KV-append hook : THIS loop drives the pager append ----
        .ext_append_valid(ext_av), .ext_append_row(ext_row), .ext_append_seq(ext_seq),
        .flash_req(flash_req), .flash_is_expert(flash_is_expert),
        .flash_expert_id(flash_expert_id), .flash_row_idx(flash_row_idx),
        .flash_done(flash_done), .flash_row(flash_row),
        .pf_valid(pf_valid), .pf_expert_id(pf_expert_id),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .wl_mem_en(wl_mem_en), .wl_mem_addr(wl_mem_addr), .wl_mem_data(wl_mem_data),
        .decomp_tbl_we(1'b0), .decomp_tbl_sel(1'b0),
        .decomp_tbl_addr(9'h000), .decomp_tbl_wdata(10'h000),
        .argmax_o(argmax_o), .h_state(), .mdl_busy(mdl_busy),
        .ec_resp_valid(), .ec_hit(), .ec_resp_slot(), .ec_busy(),
        .ec_hit_count(ec_hit_count), .ec_miss_count(ec_miss_count),
        .ec_demand_stall_cycles(), .ec_pf_issued(), .ec_pf_hit(),
        .kv_row_valid(), .kv_row_out(), .kv_busy(),
        .kv_lat_row(), .kv_lat_valid(),
        .kv_lat_row_all(sys_kv_lat_row_all), .kv_lat_valid_all(sys_kv_lat_valid_all),
        .kv_append_count(kv_append_count), .kv_resident_lo(), .kv_overflowed(),
        .ec_dropped(),
        .xbar_req_count(), .xbar_resp_count(), .xbar_resp_valid(), .xbar_resp_data(),
        .loader_busy(), .loader_done_count(), .loader_beat_count(),
        .loader_w_q(), .loader_in_valid()
    );

    //========================================================================
    // spec_decode_seq (batch g_kn) : the COMMITTER.  Depth SEQK >= 2 so the batch
    //   path is always selected; K=1 (PE_M=2) presents n_draft=1 (only 1 draft
    //   scanned) with the unused high draft/truth slots zero-padded.  It ONLY ever
    //   commits truth_vec entries (m_1..m_{p+1}) -- never a raw draft.
    //========================================================================
    wire [SEQK*TOKW-1:0]     sk_draft = { {((SEQK-K)*TOKW){1'b0}}, draft_q };
`ifdef INJECT_RAW_DRAFT
    // INJECTION (compile-time; NEVER in the normal build): feed the RAW DRAFTS as the
    //   "truth" -- spec_decode_seq's scan then sees draft==truth everywhere (accepts
    //   all) and COMMITS the raw drafts d_1..d_K instead of the model's argmaxes.
    //   Because a draft can be WRONG (the REJECT/MIXED schedules feed d != the model
    //   token on purpose), the committed stream diverges from greedy AT THE FIRST wrong
    //   draft -> the spec==greedy TB MUST FAIL.  This is exactly what spec_decode_seq
    //   must prevent: commit the model's OWN argmaxes (truth_vec), NEVER a raw draft.
    wire [(SEQK+1)*TOKW-1:0] sk_truth =
        { {((SEQK+1-PE_M)*TOKW){1'b0}}, draft_q[(K-1)*TOKW +: TOKW], draft_q };
`else
    wire [(SEQK+1)*TOKW-1:0] sk_truth = { {((SEQK+1-PE_M)*TOKW){1'b0}}, truth_q };
`endif
    /* verilator lint_off WIDTHEXPAND */
    wire [DKW_S-1:0]         sk_ndraft = nd_q;
    /* verilator lint_on WIDTHEXPAND */

    spec_decode_seq #(.TOKW(TOKW), .DRAFT_K(SEQK)) u_seq (
        .clk(clk), .rst(rst), .start(seq_arm),
        .pass_valid(seq_pass),
        .verified_tok({TOKW{1'b0}}), .draft_tok({TOKW{1'b0}}), .draft_present(1'b0),
        .commit_valid(commit_valid), .commit_tok(commit_tok), .accepted(accepted),
        .total_tokens(total_tokens), .main_passes(main_passes),
        .accepts(accepts), .rejects(rejects),
        .draft_vec(sk_draft), .truth_vec(sk_truth), .n_draft(sk_ndraft),
        .k_cur({DKW_S{1'b0}}), .pass_done(), .pass_acc(), .pass_dep()
    );

    //========================================================================
    // LOCAL longest-accepted-prefix (cursor + write-back count) -- the SAME rule
    //   spec_decode_seq uses, over the REAL K drafts / K+1 truths.
    //========================================================================
    function automatic [OCW-1:0] acc_prefix;
        input [DRAFT_K*TOKW-1:0] dv;
        input [PE_M*TOKW-1:0]    tv;
        input [OCW-1:0]          ndi;
        integer       fj;
        reg           fb;
        reg [OCW-1:0] fp;
        begin
            fp = {OCW{1'b0}};
            fb = 1'b0;
            for (fj = 0; fj < DRAFT_K; fj = fj + 1) begin
                if (!fb && (fj < ndi) &&
                    (dv[fj*TOKW +: TOKW] == tv[fj*TOKW +: TOKW]))
                    fp = fp + 1'b1;
                else
                    fb = 1'b1;
            end
            acc_prefix = fp;
        end
    endfunction

    localparam [OCW-1:0] K_OCW = DRAFT_K[OCW-1:0];
    /* verilator lint_off WIDTHEXPAND */
    wire [OCW-1:0] nd_ext = nd_q;
    /* verilator lint_on WIDTHEXPAND */
    wire [OCW-1:0] nd_w   = (nd_ext > K_OCW) ? K_OCW : nd_ext;
    wire [OCW-1:0] pfx_w  = acc_prefix(draft_q, truth_q, nd_w);   // p (0..K)
    wire [TOKW-1:0] frontier_tok = truth_q[TOKW*pfx_w +: TOKW];   // m_{p+1}
    wire [PW-1:0]   pfx_pw = pfx_w[PW-1:0];                       // p, row-index width

    //========================================================================
    // capture : buf[db_layer][row] <= this layer's PE_M current-token latents,
    //   on every kv_lat_valid_all[0] pulse (once per layer, db_layer stable) --
    //   the SAME (layer,valid) the internal SELF_KV append would use.
    //========================================================================
    integer cb;
    always @(posedge clk) begin
        if (sys_kv_lat_valid_all[0]) begin
            for (cb = 0; cb < PE_M; cb = cb + 1)
                buf_lat[db_layer][cb] <= sys_kv_lat_row_all[cb*KVR +: KVR];
        end
    end

    //========================================================================
    // LOOP FSM
    //========================================================================
    localparam [2:0]
        S_IDLE  = 3'd0,
        S_SETUP = 3'd1,    // latch this pass's K drafts + n_draft
        S_LAUNCH= 3'd2,    // pulse the die start (ONE weight-load)
        S_WAIT  = 3'd3,    // die running; capture latents; wait done -> truth_q
        S_FEED  = 3'd4,    // pulse pass_valid -> spec_decode_seq commits m_1..m_{p+1}
        S_WBACK = 3'd5,    // write back the committed prefix (per layer, rows 0..p)
        S_DRAIN = 3'd6,    // let the final pass's commit beats drain
        S_DONE  = 3'd7;
    reg [2:0] state;
    reg [7:0] drain_cnt;

    integer bi, bj;
    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            sys_start    <= 1'b0;
            seq_arm      <= 1'b0;
            seq_pass     <= 1'b0;
            ext_av       <= 1'b0;
            ext_row      <= {KVR{1'b0}};
            ext_seq      <= {KV_SEQW{1'b0}};
            cur_tok      <= {TOKW{1'b0}};
            t_pos        <= {POSW{1'b0}};
            draft_q      <= {DRAFT_K*TOKW{1'b0}};
            nd_q         <= {DKW{1'b0}};
            truth_q      <= {PE_M*TOKW{1'b0}};
            pass_idx     <= 16'd0;
            npass_q      <= 16'd0;
            wb_m         <= {LW{1'b0}};
            wb_r         <= {PW{1'b0}};
            drain_cnt    <= 8'd0;
            weight_loads <= 32'd0;
            for (bi = 0; bi < L; bi = bi + 1)
                for (bj = 0; bj < PE_M; bj = bj + 1)
                    buf_lat[bi][bj] <= {KVR{1'b0}};
        end else begin
            // pulse defaults
            done      <= 1'b0;
            sys_start <= 1'b0;
            seq_arm   <= 1'b0;
            seq_pass  <= 1'b0;
            ext_av    <= 1'b0;

            case (state)
            //----------------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy     <= 1'b1;
                    cur_tok  <= prompt_tok;
                    t_pos    <= start_pos;
                    npass_q  <= num_passes;
                    pass_idx <= 16'd0;
                    seq_arm  <= 1'b1;            // arm the committer
                    state    <= S_SETUP;
                end
            end
            //---------------------------------------------------------------- latch drafts
            S_SETUP: begin
                draft_q <= draft_in;
                nd_q    <= n_draft;
                state   <= S_LAUNCH;
            end
            //---------------------------------------------------------------- launch die
            S_LAUNCH: begin
                sys_start    <= 1'b1;           // ONE weight-load for the K+1-row verify
                weight_loads <= weight_loads + 32'd1;
                state        <= S_WAIT;
            end
            //---------------------------------------------------------------- die wait
            //   ext_av held 0 -> pager stable at t; per-layer latents captured above.
            S_WAIT: begin
                if (sys_done) begin
                    truth_q <= argmax_o;        // {m_1..m_{K+1}}
                    state   <= S_FEED;
                end
            end
            //---------------------------------------------------------------- feed committer
            S_FEED: begin
                seq_pass <= 1'b1;               // spec_decode_seq commits m_1..m_{p+1}
                wb_m     <= {LW{1'b0}};
                wb_r     <= {PW{1'b0}};
                state    <= S_WBACK;
            end
            //---------------------------------------------------------------- KV write-back
            //   append EXACTLY the committed prefix : for every layer m, rows 0..p.
            //   window m advances t..t+p (p+1 appends); rejected rows never appended.
            S_WBACK: begin
                ext_av  <= 1'b1;
                ext_row <= buf_lat[wb_m][wb_r];
                ext_seq <= wb_m[KV_SEQW-1:0];
                if (wb_r == pfx_pw) begin
                    wb_r <= {PW{1'b0}};
                    if (wb_m == (L[LW-1:0]-1'b1)) begin
                        // write-back complete : advance the committed frontier
                        cur_tok <= frontier_tok;                       // m_{p+1}
                        t_pos   <= t_pos + {{(POSW-OCW){1'b0}}, pfx_w}  // + (p+1)
                                        + {{(POSW-1){1'b0}}, 1'b1};
                        if (pass_idx == npass_q - 16'd1) begin
                            drain_cnt <= 8'd0;
                            state     <= S_DRAIN;
                        end else begin
                            pass_idx <= pass_idx + 16'd1;
                            state    <= S_SETUP;
                        end
                    end else begin
                        wb_m <= wb_m + 1'b1;
                    end
                end else begin
                    wb_r <= wb_r + 1'b1;
                end
            end
            //---------------------------------------------------------------- drain
            S_DRAIN: begin
                if (drain_cnt == (DRAFT_K[7:0] + 8'd2)) state <= S_DONE;
                else drain_cnt <= drain_cnt + 8'd1;
            end
            //----------------------------------------------------------------
            S_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
/* verilator lint_on DECLFILENAME */
