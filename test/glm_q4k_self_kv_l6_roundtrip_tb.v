`timescale 1ns/1ps
//============================================================================
// glm_q4k_self_kv_l6_roundtrip_tb.v
//   KV latent WRITE-BACK round-trip proof at L>1 (KV_WRITEBACK_DESIGN.md step 3).
//   Sibling of glm_q4k_self_kv_roundtrip_tb.v (L=1); this one runs the SLICE DEPTH
//   L=6 (3 dense + 3 MoE layers) to prove PER-(LAYER,POSITION) KV keying.
//----------------------------------------------------------------------------
// WHAT IS PROVEN (self-checking; $fatal on any mismatch):
//
//   The DUT is glm_q4k_system with SELF_KV=1 at L=6.  Each token runs L=6 layers;
//   every layer m does its OWN KV gather (kc_req/kc_idx while db_layer==m) and its
//   OWN append (kv_lat_valid while db_layer==m).  glm_q4k_system keys the pager by
//   (layer,position) -- realised as NSEQ=L independent ring windows, seq=db_layer,
//   so layer m's position p is a DISTINCT row (window m) and layer m ONLY gathers
//   window-m rows.  Per token there are L appends (one per layer).
//
//   The REFERENCE is an INDEPENDENT standalone glm_model_q4k (same L=6, same
//   deterministic weights) whose KV is delivered by a TB SHADOW keyed by
//   (LAYER, position):  shadow[layer*S_MAX + pos].  The TB captures the reference's
//   OWN committed latent (r_kv_lat_row on r_kv_lat_valid) into
//   shadow[r_db_layer*S_MAX + ref_wr[r_db_layer]] and answers every reference gather
//   from shadow[r_db_layer*S_MAX + r_kc_idx].  So the reference's KV for (layer m,
//   token t) is exactly the latents layer m computed for tokens 0..t-1 -- the SAME
//   per-(layer,pos) round-trip, transported by an INDEPENDENT path (a TB 2-D array
//   keyed by (layer,pos)) instead of the pager's per-layer windows.
//
//   BINDING (every token): the system's committed next_tok === the reference's
//   committed token, AND the full pre-argmax logit vector logits === r_logits.
//   Both models compute identical latents (same weights / prompt / prior KV) so
//   they agree IFF the pager transport delivers, for every (layer,pos), the SAME
//   bit-identical KV the (layer,pos)-keyed TB shadow delivers.  If the pager mixed
//   layers (e.g. dropped the layer offset so layer m read layer-0's window), the
//   DUT would attend corrupted KV and the FULL-LOGIT binding would DIVERGE, while
//   the (layer,pos)-keyed reference stays correct -- see the INJECTION note.
//
//   Also per token: append_count of the last-layer window grew by +1 (the write-
//   back path, not the host stub), the committed token is X/Z-clean and === the
//   system internal argmax, and the pager served gather rows (die READ its KV).
//
// INJECTION SOUNDNESS (proves the check catches a LAYER-ALIASING bug):
//   Drop the layer term on the gather side in glm_q4k_system, i.e.
//       wire [KV_SEQW-1:0] pg_gather_seq = (SELF_KV != 0) ? db_layer[KV_SEQW-1:0]
//                                                         : {KV_SEQW{1'b0}};
//   ->  wire [KV_SEQW-1:0] pg_gather_seq = {KV_SEQW{1'b0}};   // BUG: all layers read window 0
//   Now every layer m>0 gathers layer-0's rows.  The (layer,pos)-keyed reference is
//   unaffected, so the FULL-LOGIT binding FAILS on the first token that reads real
//   KV at a layer m>0.  Revert after confirming.  (An off-by-one on the layer term,
//   or dropping it on the APPEND side, fails the same way.)
//
// STYLE: sync active-high reset; self-checking ($fatal on mismatch); final banner
//   "ALL %0d TESTS PASSED".
//============================================================================
module glm_q4k_self_kv_l6_roundtrip_tb;

    // ---- L=6 slice (3 dense + 3 MoE layers) ----
    parameter integer MODEL_DIM = 16;
    parameter integer L         = 6;    // KV_WRITEBACK step 3 scope: L>1 (slice depth)
    parameter integer N_DENSE   = 3;    // layers 0..2 dense, 3..5 MoE
    parameter integer VOCAB     = 16;
    parameter integer H_HEADS   = 2;
    parameter integer NOPE      = 4;
    parameter integer ROPE      = 4;
    parameter integer V_DIM     = 4;
    parameter integer Q_LORA    = 8;
    parameter integer KV_LORA   = 8;
    parameter integer S_MAX     = 8;
    parameter integer TOPK_ATTN = 8;
    localparam integer THETA     = 8000000;
    parameter integer PE_N      = 2;
    localparam integer POSW      = 20;
    parameter integer N_EXPERT  = 4;
    parameter integer TOPK      = 2;
    parameter integer INTER_MOE = 16;
    parameter integer INTER_DENSE = 32;
    localparam [31:0] RSCALE     = 32'h40200000;
    parameter integer TN        = 4;
    localparam integer BLK       = 128;
    parameter integer LM_TN     = 4;
    // ---- memory system ----
    localparam integer CACHE_SLOTS = 4;
    localparam integer FLASH_LAT   = 8;
    localparam integer KV_CTX      = 1024;
    localparam integer KV_RESIDENT = 16;   // power of two, >= S_MAX: every position resident
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

    reg clk; initial clk = 1'b0; always #5 clk = ~clk;
    reg rst;

    integer test_count, errors;

    //========================================================================
    // deterministic stimulus generators (pure functions of request address)
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
    function automatic [15:0] f_awd; input integer ly; input integer sel;
        input integer fo; begin
        f_awd = gen_fp16(ly*7919 + sel*104729 + fo*611953 + 211);
    end endfunction
    function automatic [15:0] f_awdm; input integer ly; input integer sel;
        input integer fo; begin
        f_awdm = gen_fp16(ly*7919 + sel*104729 + fo*611953 + 307);
    end endfunction
    function automatic [3:0] f_rwq; input integer ly; input integer e;
        input integer kk; begin
        f_rwq = gen_q4(ly*7919 + e*350377 + kk*13 + 401);
    end endfunction
    function automatic [3:0] f_fwq; input integer ly; input integer sel;
        input integer shr; input integer eidx; input integer fo; input integer kk;
        begin
        f_fwq = gen_q4(ly*7919 + sel*104729 + shr*15485863 + eidx*350377
                       + fo*611953 + kk*13 + 503);
    end endfunction

    //========================================================================
    // DUT host I/O
    //========================================================================
    reg                       start;
    reg  [TOKW-1:0]           prompt_tok;
    reg  [POSW-1:0]           start_pos;
    reg  [IDXW:0]             s_len;
    wire                      busy, done;
    wire [TOKW-1:0]           next_tok;
    wire                      tok_valid;
    wire [VOCAB*16-1:0]       logits;
    wire                      em_req;  wire [TOKW-1:0] em_tok;  wire [DIMW-1:0] em_idx;  reg [15:0] em_val;
    wire [LAYW-1:0]           db_layer;  wire idx_fresh;  wire [LAYW-1:0] idx_win;
    wire                      gn_req, gn_which;  wire [DIMW-1:0] gn_idx;  reg [15:0] gn_val;
    wire                      aw_req;  wire [3:0] aw_sel;  wire [A_GRPW-1:0] aw_grp;  wire [A_KCW-1:0] aw_k;
    reg  [PE_N*4-1:0]         aw_q;
    reg  [16*PE_N*A_NSB-1:0]  aw_d, aw_dmin;
    reg  [96*PE_N*A_NSB-1:0]  aw_scales;
    wire                      rw_req;  wire [R_KW-1:0] rw_k;
    reg  [4*N_EXPERT-1:0]         rw_q;
    reg  [16*N_EXPERT*R_NSB-1:0]  rw_d, rw_dmin;
    reg  [96*N_EXPERT*R_NSB-1:0]  rw_scales;
    wire                      fw_req;  wire [1:0] fw_sel;  wire [FF_GWD-1:0] fw_grp;  wire [FF_KWD-1:0] fw_k;
    wire                      fw_shared;  wire [EIDXW-1:0] fw_eidx;
    reg  [4*TN-1:0]           fw_q, fw_q_up;
    reg  [16*TN*FF_NSB_D-1:0] fw_d_g, fw_dmin_g, fw_d_u, fw_dmin_u;
    reg  [96*TN*FF_NSB_D-1:0] fw_scales_g, fw_scales_u;
    wire                      fn_req;  wire [DIMW-1:0] fn_idx;  reg [15:0] fn_val;
    wire                      lw_req;  wire [VTW-1:0] lw_vtile;  wire [DIMW-1:0] lw_k;  reg [LM_TN*16-1:0] lw_col;
    wire                      kc_req;  wire [IDXW-1:0] kc_idx;
    wire [KVPOSW-1:0]         kv_row_sel;
    wire                      flash_req, flash_is_expert;
    wire [EIDXW-1:0]          flash_expert_id;  wire [KVPOSW-1:0] flash_row_idx;
    reg  [ROW_BITS-1:0]       flash_row;  reg flash_done;   // driven by the FLASH PHY stub below
    wire [TOKW-1:0]           argmax_o;  wire [MODEL_DIM*16-1:0] h_state;  wire mdl_busy;
    wire                      ec_resp_valid, ec_hit;  wire [CSLOTW-1:0] ec_resp_slot;  wire ec_busy;
    wire [31:0]               ec_hit_count, ec_miss_count, ec_demand_stall_cycles, ec_pf_issued, ec_pf_hit;
    wire                      kv_row_valid;  wire [ROW_BITS-1:0] kv_row_out;  wire kv_busy;
    wire [KVPOSW-1:0]         kv_append_count, kv_resident_lo;  wire kv_overflowed;
    wire [ROW_BITS-1:0]       kv_lat_row;  wire kv_lat_valid;
    wire [31:0]               ec_dropped;
    wire [DDR_NCH-1:0]            mem_req_valid;
    wire [DDR_NCH-1:0]            mem_req_ready;
    wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr;
    wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag;
    reg  [DDR_NCH-1:0]            mem_resp_valid;
    wire [DDR_NCH-1:0]            mem_resp_ready;
    reg  [DDR_NCH*DDR_DATA_W-1:0] mem_resp_data;
    reg  [DDR_NCH*DDR_TAG_W-1:0]  mem_resp_tag;
    wire                      wl_mem_en;  wire [WL_ADDR_W-1:0] wl_mem_addr;  reg [WL_DATA_W-1:0] wl_mem_data;
    wire [31:0]               xbar_req_count, xbar_resp_count;
    wire                      xbar_resp_valid;  wire [DDR_DATA_W-1:0] xbar_resp_data;
    wire                      loader_busy;  wire [31:0] loader_done_count, loader_beat_count;
    wire [4*WL_PE_N-1:0]      loader_w_q;  wire loader_in_valid;

    glm_q4k_system #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN),
        .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX),
        .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH),
        .SELF_KV(1),                              // <== close the KV write-back loop
        .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W),
        .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD),
        .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN)
    ) dut (
        .clk(clk), .rst(rst),
        .start(start), .prompt_tok(prompt_tok), .start_pos(start_pos), .s_len(s_len),
        .busy(busy), .done(done), .next_tok(next_tok), .tok_valid(tok_valid),
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
        // SELF_KV=1: the model reads KV from the pager internally, so these stub
        //   inputs are UNUSED by the die (tied 0).  kv_row_in likewise (append
        //   comes from kv_lat_row internally).
        .kc_ckv({(KV_LORA*16){1'b0}}), .kc_krope({(ROPE*16){1'b0}}),
        .kc_req(kc_req), .kc_idx(kc_idx),
        .kv_row_sel(kv_row_sel), .kv_row_in({ROW_BITS{1'b0}}),
        .flash_req(flash_req), .flash_is_expert(flash_is_expert),
        .flash_expert_id(flash_expert_id), .flash_row_idx(flash_row_idx),
        .flash_done(flash_done), .flash_row(flash_row),
        .pf_valid(1'b0), .pf_expert_id({EIDXW{1'b0}}),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .wl_mem_en(wl_mem_en), .wl_mem_addr(wl_mem_addr), .wl_mem_data(wl_mem_data),
        .decomp_tbl_we(1'b0), .decomp_tbl_sel(1'b0),
        .decomp_tbl_addr(9'h000), .decomp_tbl_wdata(10'h000),
        .argmax_o(argmax_o), .h_state(h_state), .mdl_busy(mdl_busy),
        .ec_resp_valid(ec_resp_valid), .ec_hit(ec_hit), .ec_resp_slot(ec_resp_slot),
        .ec_busy(ec_busy), .ec_hit_count(ec_hit_count), .ec_miss_count(ec_miss_count),
        .ec_demand_stall_cycles(ec_demand_stall_cycles),
        .ec_pf_issued(ec_pf_issued), .ec_pf_hit(ec_pf_hit),
        .kv_row_valid(kv_row_valid), .kv_row_out(kv_row_out), .kv_busy(kv_busy),
        .kv_append_count(kv_append_count), .kv_resident_lo(kv_resident_lo),
        .kv_overflowed(kv_overflowed),
        .kv_lat_row(kv_lat_row), .kv_lat_valid(kv_lat_valid),
        .ec_dropped(ec_dropped),
        .xbar_req_count(xbar_req_count), .xbar_resp_count(xbar_resp_count),
        .xbar_resp_valid(xbar_resp_valid), .xbar_resp_data(xbar_resp_data),
        .loader_busy(loader_busy), .loader_done_count(loader_done_count),
        .loader_beat_count(loader_beat_count),
        .loader_w_q(loader_w_q), .loader_in_valid(loader_in_valid)
    );

    // ================= DUT weight responders (deterministic; from request) ======
    integer t, ft, re, sb;
    always @* em_val = gen_bf16(em_tok*MODEL_DIM + em_idx + 7001);
    always @* fn_val = gen_bf16(fn_idx + 7207);
    always @* gn_val = gen_bf16(db_layer*1024 + gn_which*512 + gn_idx + 7411);
    always @* begin
        for (t=0;t<LM_TN;t=t+1)
            lw_col[16*t+:16] = gen_bf16((lw_vtile*LM_TN+t)*MODEL_DIM + lw_k + 7603);
    end
    always @* begin
        for (t=0;t<PE_N;t=t+1) begin
            aw_q[4*t+:4] = f_awq(db_layer, aw_sel, aw_grp*PE_N+t, aw_k);
            for (sb=0;sb<A_NSB;sb=sb+1) begin
                aw_d   [16*(sb*PE_N+t)+:16] = f_awd (db_layer, aw_sel, aw_grp*PE_N+t);
                aw_dmin[16*(sb*PE_N+t)+:16] = f_awdm(db_layer, aw_sel, aw_grp*PE_N+t);
                aw_scales[96*(sb*PE_N+t)   +:32] = gen_s32(db_layer*7919+aw_sel*104729+(aw_grp*PE_N+t)*611953+601);
                aw_scales[96*(sb*PE_N+t)+32+:32] = gen_s32(db_layer*7919+aw_sel*104729+(aw_grp*PE_N+t)*611953+602);
                aw_scales[96*(sb*PE_N+t)+64+:32] = gen_s32(db_layer*7919+aw_sel*104729+(aw_grp*PE_N+t)*611953+603);
            end
        end
    end
    always @* begin
        for (re=0;re<N_EXPERT;re=re+1) begin
            rw_q[4*re+:4] = f_rwq(db_layer, re, rw_k);
            for (sb=0;sb<R_NSB;sb=sb+1) begin
                rw_d   [16*(sb*N_EXPERT+re)+:16] = gen_fp16(db_layer*7919+re*350377+421);
                rw_dmin[16*(sb*N_EXPERT+re)+:16] = gen_fp16(db_layer*7919+re*350377+431);
                rw_scales[96*(sb*N_EXPERT+re)   +:32] = gen_s32(db_layer*7919+re*350377+441);
                rw_scales[96*(sb*N_EXPERT+re)+32+:32] = gen_s32(db_layer*7919+re*350377+442);
                rw_scales[96*(sb*N_EXPERT+re)+64+:32] = gen_s32(db_layer*7919+re*350377+443);
            end
        end
    end
    always @* begin
        for (ft=0;ft<TN;ft=ft+1) begin
            fw_q   [4*ft+:4] = f_fwq(db_layer, fw_sel, fw_shared, fw_eidx, fw_grp*TN+ft, fw_k);
            fw_q_up[4*ft+:4] = f_fwq(db_layer, 3,      fw_shared, fw_eidx, fw_grp*TN+ft, fw_k);
            for (sb=0;sb<FF_NSB_D;sb=sb+1) begin
                fw_d_g   [16*(sb*TN+ft)+:16] = gen_fp16(db_layer*7919+fw_sel*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+521);
                fw_dmin_g[16*(sb*TN+ft)+:16] = gen_fp16(db_layer*7919+fw_sel*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+531);
                fw_d_u   [16*(sb*TN+ft)+:16] = gen_fp16(db_layer*7919+3*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+521);
                fw_dmin_u[16*(sb*TN+ft)+:16] = gen_fp16(db_layer*7919+3*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+531);
                fw_scales_g[96*(sb*TN+ft)   +:32] = gen_s32(db_layer*7919+fw_sel*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+541);
                fw_scales_g[96*(sb*TN+ft)+32+:32] = gen_s32(db_layer*7919+fw_sel*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+542);
                fw_scales_g[96*(sb*TN+ft)+64+:32] = gen_s32(db_layer*7919+fw_sel*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+543);
                fw_scales_u[96*(sb*TN+ft)   +:32] = gen_s32(db_layer*7919+3*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+541);
                fw_scales_u[96*(sb*TN+ft)+32+:32] = gen_s32(db_layer*7919+3*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+542);
                fw_scales_u[96*(sb*TN+ft)+64+:32] = gen_s32(db_layer*7919+3*104729+fw_shared*15485863+fw_eidx*350377+(fw_grp*TN+ft)*611953+543);
            end
        end
    end

    // ================= INDEPENDENT REFERENCE: standalone glm_model_q4k ==========
    //   Same deterministic weights; KV delivered by a TB SHADOW keyed by (LAYER,pos)
    //   that captures the reference's OWN committed latents (r_kv_lat_row) and
    //   answers gathers from shadow[layer*S_MAX + pos].
    reg                       r_start;
    wire                      r_busy, r_done;
    wire [TOKW-1:0]           r_argmax;
    wire [VOCAB*16-1:0]       r_logits;
    wire                      r_em_req;  wire [TOKW-1:0] r_em_tok;  wire [DIMW-1:0] r_em_idx;  reg [15:0] r_em_val;
    wire [LAYW-1:0]           r_db_layer;  wire r_idx_fresh;  wire [LAYW-1:0] r_idx_win;
    wire                      r_gn_req, r_gn_which;  wire [DIMW-1:0] r_gn_idx;  reg [15:0] r_gn_val;
    wire                      r_aw_req;  wire [3:0] r_aw_sel;  wire [A_GRPW-1:0] r_aw_grp;  wire [A_KCW-1:0] r_aw_k;
    reg  [PE_N*4-1:0]         r_aw_q;
    reg  [16*PE_N*A_NSB-1:0]  r_aw_d, r_aw_dmin;
    reg  [96*PE_N*A_NSB-1:0]  r_aw_scales;
    wire                      r_rw_req;  wire [R_KW-1:0] r_rw_k;
    reg  [4*N_EXPERT-1:0]         r_rw_q;
    reg  [16*N_EXPERT*R_NSB-1:0]  r_rw_d, r_rw_dmin;
    reg  [96*N_EXPERT*R_NSB-1:0]  r_rw_scales;
    wire                      r_fw_req;  wire [1:0] r_fw_sel;  wire [FF_GWD-1:0] r_fw_grp;  wire [FF_KWD-1:0] r_fw_k;
    wire                      r_fw_shared;  wire [EIDXW-1:0] r_fw_eidx;
    reg  [4*TN-1:0]           r_fw_q, r_fw_q_up;
    reg  [16*TN*FF_NSB_D-1:0] r_fw_d_g, r_fw_dmin_g, r_fw_d_u, r_fw_dmin_u;
    reg  [96*TN*FF_NSB_D-1:0] r_fw_scales_g, r_fw_scales_u;
    wire                      r_fn_req;  wire [DIMW-1:0] r_fn_idx;  reg [15:0] r_fn_val;
    wire                      r_lw_req;  wire [VTW-1:0] r_lw_vtile;  wire [DIMW-1:0] r_lw_k;  reg [LM_TN*16-1:0] r_lw_col;
    wire                      r_kc_req;  wire [IDXW-1:0] r_kc_idx;  wire r_kc_seq;
    reg  [KV_LORA*16-1:0]     r_kc_ckv;  reg [ROPE*16-1:0] r_kc_krope;
    reg                       r_kc_valid;
    wire [MODEL_DIM*16-1:0]   r_h_state;
    wire [ROW_BITS-1:0]       r_kv_lat_row;  wire r_kv_lat_valid;

    glm_model_q4k #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN)
    ) u_ref (
        .clk(clk), .rst(rst),
        .start(r_start), .busy(r_busy), .done(r_done),
        .token_id(prompt_tok), .pos(start_pos), .pos_vec({POSW{1'b0}}),
        .s_len_vec({(IDXW+1){1'b0}}), .seq_vec(1'b0), .s_len(s_len),
        .logits(r_logits), .argmax(r_argmax),
        .em_req(r_em_req), .em_tok(r_em_tok), .em_idx(r_em_idx), .em_val(r_em_val),
        .db_layer(r_db_layer), .idx_fresh(r_idx_fresh), .idx_win(r_idx_win),
        .gn_req(r_gn_req), .gn_which(r_gn_which), .gn_idx(r_gn_idx), .gn_val(r_gn_val),
        .aw_req(r_aw_req), .aw_sel(r_aw_sel), .aw_grp(r_aw_grp), .aw_k(r_aw_k),
        .aw_q(r_aw_q), .aw_d(r_aw_d), .aw_dmin(r_aw_dmin), .aw_scales(r_aw_scales),
        .kc_req(r_kc_req), .kc_idx(r_kc_idx), .kc_seq(r_kc_seq),
        .kc_ckv(r_kc_ckv), .kc_krope(r_kc_krope), .kc_valid(r_kc_valid),
        .rw_req(r_rw_req), .rw_k(r_rw_k),
        .rw_q(r_rw_q), .rw_d(r_rw_d), .rw_dmin(r_rw_dmin), .rw_scales(r_rw_scales),
        .fw_req(r_fw_req), .fw_sel(r_fw_sel), .fw_grp(r_fw_grp), .fw_k(r_fw_k),
        .fw_shared(r_fw_shared), .fw_eidx(r_fw_eidx),
        .fw_q(r_fw_q), .fw_q_up(r_fw_q_up),
        .fw_d_g(r_fw_d_g), .fw_dmin_g(r_fw_dmin_g), .fw_scales_g(r_fw_scales_g),
        .fw_d_u(r_fw_d_u), .fw_dmin_u(r_fw_dmin_u), .fw_scales_u(r_fw_scales_u),
        .fn_req(r_fn_req), .fn_idx(r_fn_idx), .fn_val(r_fn_val),
        .lw_req(r_lw_req), .lw_vtile(r_lw_vtile), .lw_k(r_lw_k), .lw_col(r_lw_col),
        .h_state(r_h_state),
        .kv_lat_row(r_kv_lat_row), .kv_lat_valid(r_kv_lat_valid)
    );

    integer rt, rft, rre, rsb;
    always @* r_em_val = gen_bf16(r_em_tok*MODEL_DIM + r_em_idx + 7001);
    always @* r_fn_val = gen_bf16(r_fn_idx + 7207);
    always @* r_gn_val = gen_bf16(r_db_layer*1024 + r_gn_which*512 + r_gn_idx + 7411);
    always @* begin
        for (rt=0;rt<LM_TN;rt=rt+1)
            r_lw_col[16*rt+:16] = gen_bf16((r_lw_vtile*LM_TN+rt)*MODEL_DIM + r_lw_k + 7603);
    end
    always @* begin
        for (rt=0;rt<PE_N;rt=rt+1) begin
            r_aw_q[4*rt+:4] = f_awq(r_db_layer, r_aw_sel, r_aw_grp*PE_N+rt, r_aw_k);
            for (rsb=0;rsb<A_NSB;rsb=rsb+1) begin
                r_aw_d   [16*(rsb*PE_N+rt)+:16] = f_awd (r_db_layer, r_aw_sel, r_aw_grp*PE_N+rt);
                r_aw_dmin[16*(rsb*PE_N+rt)+:16] = f_awdm(r_db_layer, r_aw_sel, r_aw_grp*PE_N+rt);
                r_aw_scales[96*(rsb*PE_N+rt)   +:32] = gen_s32(r_db_layer*7919+r_aw_sel*104729+(r_aw_grp*PE_N+rt)*611953+601);
                r_aw_scales[96*(rsb*PE_N+rt)+32+:32] = gen_s32(r_db_layer*7919+r_aw_sel*104729+(r_aw_grp*PE_N+rt)*611953+602);
                r_aw_scales[96*(rsb*PE_N+rt)+64+:32] = gen_s32(r_db_layer*7919+r_aw_sel*104729+(r_aw_grp*PE_N+rt)*611953+603);
            end
        end
    end
    always @* begin
        for (rre=0;rre<N_EXPERT;rre=rre+1) begin
            r_rw_q[4*rre+:4] = f_rwq(r_db_layer, rre, r_rw_k);
            for (rsb=0;rsb<R_NSB;rsb=rsb+1) begin
                r_rw_d   [16*(rsb*N_EXPERT+rre)+:16] = gen_fp16(r_db_layer*7919+rre*350377+421);
                r_rw_dmin[16*(rsb*N_EXPERT+rre)+:16] = gen_fp16(r_db_layer*7919+rre*350377+431);
                r_rw_scales[96*(rsb*N_EXPERT+rre)   +:32] = gen_s32(r_db_layer*7919+rre*350377+441);
                r_rw_scales[96*(rsb*N_EXPERT+rre)+32+:32] = gen_s32(r_db_layer*7919+rre*350377+442);
                r_rw_scales[96*(rsb*N_EXPERT+rre)+64+:32] = gen_s32(r_db_layer*7919+rre*350377+443);
            end
        end
    end
    always @* begin
        for (rft=0;rft<TN;rft=rft+1) begin
            r_fw_q   [4*rft+:4] = f_fwq(r_db_layer, r_fw_sel, r_fw_shared, r_fw_eidx, r_fw_grp*TN+rft, r_fw_k);
            r_fw_q_up[4*rft+:4] = f_fwq(r_db_layer, 3,        r_fw_shared, r_fw_eidx, r_fw_grp*TN+rft, r_fw_k);
            for (rsb=0;rsb<FF_NSB_D;rsb=rsb+1) begin
                r_fw_d_g   [16*(rsb*TN+rft)+:16] = gen_fp16(r_db_layer*7919+r_fw_sel*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+521);
                r_fw_dmin_g[16*(rsb*TN+rft)+:16] = gen_fp16(r_db_layer*7919+r_fw_sel*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+531);
                r_fw_d_u   [16*(rsb*TN+rft)+:16] = gen_fp16(r_db_layer*7919+3*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+521);
                r_fw_dmin_u[16*(rsb*TN+rft)+:16] = gen_fp16(r_db_layer*7919+3*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+531);
                r_fw_scales_g[96*(rsb*TN+rft)   +:32] = gen_s32(r_db_layer*7919+r_fw_sel*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+541);
                r_fw_scales_g[96*(rsb*TN+rft)+32+:32] = gen_s32(r_db_layer*7919+r_fw_sel*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+542);
                r_fw_scales_g[96*(rsb*TN+rft)+64+:32] = gen_s32(r_db_layer*7919+r_fw_sel*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+543);
                r_fw_scales_u[96*(rsb*TN+rft)   +:32] = gen_s32(r_db_layer*7919+3*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+541);
                r_fw_scales_u[96*(rsb*TN+rft)+32+:32] = gen_s32(r_db_layer*7919+3*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+542);
                r_fw_scales_u[96*(rsb*TN+rft)+64+:32] = gen_s32(r_db_layer*7919+3*104729+r_fw_shared*15485863+r_fw_eidx*350377+(r_fw_grp*TN+rft)*611953+543);
            end
        end
    end

    // ---- REFERENCE KV SHADOW keyed by (LAYER, position) ----
    //   shadow[layer*S_MAX + pos] = the latent the reference committed for that layer
    //   at position pos (r_kv_lat_row on r_kv_lat_valid).  r_db_layer is STABLE across
    //   the whole layer (registered; advances only on db_done), so it is the correct
    //   layer at BOTH the commit and the gather.  Per-layer write pointer ref_wr[m]
    //   advances once per token for layer m.  The gather answer UNPACKS the row the
    //   same way glm_q4k_system does (c_kv LOW, k_rope HIGH) -- an INDEPENDENT TB
    //   implementation of the pack convention, keyed by (layer,pos).
    reg [ROW_BITS-1:0] shadow [0:L*S_MAX-1];
    reg [KVPOSW-1:0]   ref_wr [0:L-1];
    integer shi;
    initial begin
        for (shi=0; shi<L*S_MAX; shi=shi+1) shadow[shi] = {ROW_BITS{1'b0}};
    end
    always @(posedge clk) begin
        if (rst) begin
            for (shi=0; shi<L; shi=shi+1) ref_wr[shi] <= {KVPOSW{1'b0}};
        end else if (r_kv_lat_valid) begin
            shadow[r_db_layer*S_MAX + ref_wr[r_db_layer]] <= r_kv_lat_row;
            ref_wr[r_db_layer] <= ref_wr[r_db_layer] + 1'b1;
        end
    end
    always @* begin
        r_kc_ckv   = shadow[r_db_layer*S_MAX + r_kc_idx][0          +: KV_LORA*16];
        r_kc_krope = shadow[r_db_layer*S_MAX + r_kc_idx][KV_LORA*16 +: ROPE*16];
    end
    always @(posedge clk) begin
        if (rst) r_kc_valid <= 1'b0;
        else     r_kc_valid <= r_kc_req;
    end
    reg [TOKW-1:0] r_tok_lat;  reg r_done_seen;
    always @(posedge clk) begin
        if (rst)            begin r_done_seen<=1'b0; r_tok_lat<={TOKW{1'b0}}; end
        else if (r_start)   r_done_seen<=1'b0;
        else if (r_done)    begin r_tok_lat<=r_argmax; r_done_seen<=1'b1; end
    end

    // ---- lint guard ----
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, busy, done, em_req, aw_req, fw_req, rw_req, gn_req, fn_req,
                     lw_req, idx_fresh, idx_win, mdl_busy, ec_resp_valid, ec_hit,
                     ec_resp_slot, ec_busy, ec_hit_count, ec_miss_count,
                     ec_demand_stall_cycles, ec_pf_issued, ec_pf_hit, ec_dropped,
                     kv_row_sel, kv_resident_lo, kv_overflowed, kv_busy, kv_lat_row,
                     kv_lat_valid, flash_expert_id, flash_row_idx, flash_is_expert,
                     h_state, gn_which, gn_idx, r_busy, r_em_req, r_aw_req, r_fw_req,
                     r_rw_req, r_gn_req, r_fn_req, r_lw_req, r_idx_fresh, r_idx_win,
                     r_logits, r_h_state, r_gn_which, r_kc_seq, loader_busy,
                     loader_done_count, loader_beat_count, loader_w_q, loader_in_valid,
                     xbar_req_count, xbar_resp_count, xbar_resp_valid, xbar_resp_data,
                     mem_req_tag, mem_req_addr, logits};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= DDR5 PER-CHANNEL MEMORY MODEL (X-clean; from perf TB) =====
    localparam integer NINF = 64;
    reg                  infv  [0:NINF-1];
    reg [DDR_TAG_W-1:0]  inftg [0:NINF-1];
    reg [DDR_ADDR_W-1:0] infad [0:NINF-1];
    reg [15:0]           inftm [0:NINF-1];
    integer ii, cc;
    assign mem_req_ready = {DDR_NCH{1'b1}};
    reg [31:0] presIdx [0:DDR_NCH-1];
    reg        presV   [0:DDR_NCH-1];
    function automatic [DDR_DATA_W-1:0] gen_beat;
        input [DDR_TAG_W-1:0] tg; input [DDR_ADDR_W-1:0] ad; integer ln; begin
        gen_beat = {DDR_DATA_W{1'b0}};
        for (ln=0; ln<DDR_DATA_W/16; ln=ln+1)
            gen_beat[16*ln+:16] = gen_bf16(({24'd0,tg}*7 + ad*3 + ln*5 + 1));
        end
    endfunction
    always @* begin
        mem_resp_valid = {DDR_NCH{1'b0}};
        mem_resp_data  = {(DDR_NCH*DDR_DATA_W){1'b0}};
        mem_resp_tag   = {(DDR_NCH*DDR_TAG_W){1'b0}};
        for (cc=0; cc<DDR_NCH; cc=cc+1) begin presV[cc]=1'b0; presIdx[cc]=32'd0; end
        for (cc=0; cc<DDR_NCH; cc=cc+1) begin
            for (ii=NINF-1; ii>=0; ii=ii-1) begin
                if (infv[ii] && (inftm[ii]==16'd0) &&
                    ((DDR_NCH==1) ? (cc==0)
                                  : (infad[ii][CH_SEL_W-1:0] == cc[CH_SEL_W-1:0]))) begin
                    presV[cc]=1'b1; presIdx[cc]=ii;
                end
            end
            if (presV[cc]) begin
                mem_resp_valid[cc] = 1'b1;
                mem_resp_data[cc*DDR_DATA_W +: DDR_DATA_W] =
                    gen_beat(inftg[presIdx[cc]], infad[presIdx[cc]]);
                mem_resp_tag[cc*DDR_TAG_W +: DDR_TAG_W] = inftg[presIdx[cc]];
            end
        end
    end
    integer freeslot; reg got_free;
    always @(posedge clk) begin
        if (rst) begin
            for (ii=0; ii<NINF; ii=ii+1) begin
                infv[ii]<=1'b0; inftg[ii]<={DDR_TAG_W{1'b0}};
                infad[ii]<={DDR_ADDR_W{1'b0}}; inftm[ii]<=16'd0;
            end
        end else begin
            for (ii=0; ii<NINF; ii=ii+1)
                if (infv[ii] && (inftm[ii]!=16'd0)) inftm[ii] <= inftm[ii]-16'd1;
            for (cc=0; cc<DDR_NCH; cc=cc+1)
                if (presV[cc] && mem_resp_ready[cc]) infv[presIdx[cc]] <= 1'b0;
            got_free=1'b0; freeslot=0;
            for (ii=NINF-1; ii>=0; ii=ii-1) if (!infv[ii]) begin got_free=1'b1; freeslot=ii; end
            for (cc=0; cc<DDR_NCH; cc=cc+1) begin
                if (mem_req_valid[cc] && mem_req_ready[cc] && got_free) begin
                    infv[freeslot]  <= 1'b1;
                    inftg[freeslot] <= mem_req_tag[cc*DDR_TAG_W +: DDR_TAG_W];
                    infad[freeslot] <= mem_req_addr[cc*DDR_ADDR_W +: DDR_ADDR_W];
                    inftm[freeslot] <= DDR_ROW_LAT[15:0];
                end
            end
        end
    end

    // ================= FLASH PHY STUB (answers cold pager gathers) ==============
    //   The empty-KV first token (s_len=0) makes each layer issue gathers the empty
    //   pager window cannot serve from the ring (cold); answer them (FLASH_LAT-cycle)
    //   so the die never hangs.  Their data is MASKED (u_cnt=0 -> zero attention) so
    //   the value is don't-care; both DUT and shadow-ref see zero context for that
    //   token.  Real (non-cold) gathers for tokens 1..N-1 come from the resident ring.
    wire [KVPOSW-1:0] flash_row_idx_w = flash_row_idx;
    reg [31:0]         fl_timer; reg fl_active; reg prev_freq;
    always @* flash_row = {ROW_BITS{1'b0}};
    wire _fri_unused = &{1'b0, flash_row_idx_w};
    always @(posedge clk) begin
        if (rst) begin fl_timer<=32'd0; fl_active<=1'b0; flash_done<=1'b0; prev_freq<=1'b0; end
        else begin
            flash_done <= 1'b0;
            if (!fl_active) begin
                if (flash_req && !prev_freq) begin fl_active<=1'b1; fl_timer<=FLASH_LAT[31:0]; end
            end else begin
                if (fl_timer <= 32'd1) begin flash_done<=1'b1; fl_active<=1'b0; end
                else fl_timer <= fl_timer - 32'd1;
            end
            prev_freq <= flash_req;
        end
    end

    // ================= LOADER STAGING-TIER RAM (latency-1 registered read) =======
    localparam integer STG_DEPTH = 2048;
    reg [WL_DATA_W-1:0] STAGE [0:STG_DEPTH-1];
    integer si, sj;
    initial for (si=0; si<STG_DEPTH; si=si+1)
        for (sj=0; sj<WL_DATA_W/16; sj=sj+1)
            STAGE[si][16*sj+:16] = gen_bf16(si*61 + sj*17 + 9);
    always @(posedge clk) begin
        if (wl_mem_en) wl_mem_data <= STAGE[wl_mem_addr[10:0]];
        else           wl_mem_data <= {WL_DATA_W{1'b0}};
    end

    //========================================================================
    // decode driver + per-token checks
    //========================================================================
    // gather-activity observer
    integer kv_rowvalid_cnt, krv_before;
    always @(posedge clk) if (!rst && kv_row_valid) kv_rowvalid_cnt = kv_rowvalid_cnt + 1;

    integer b;
    reg [KVPOSW-1:0] abefore;
    task run_token; input [TOKW-1:0] tk; input [POSW-1:0] ps; input integer SL;
        input [255:0] label; begin
        abefore = kv_append_count;
        prompt_tok = tk; start_pos = ps; s_len = SL[IDXW:0];
        @(negedge clk); start = 1'b1; r_start = 1'b1;
        @(negedge clk); start = 1'b0; r_start = 1'b0;
        wait (tok_valid === 1'b1);      // DUT (SELF_KV=1 pager loop) committed this token
        @(negedge clk);
        wait (r_done_seen === 1'b1);    // reference (shadow-fed model) committed this token
        repeat (20) @(negedge clk);     // drain

        test_count = test_count + 1;

        // (1) BINDING: SELF_KV=1 per-(layer,pos) pager round-trip token == the
        //   (layer,pos)-keyed shadow reference.
        if (next_tok !== r_tok_lat) begin
            $display("FAIL[%0s]: BINDING next_tok=%0d != shadow-ref token=%0d (SELF_KV L=6 round-trip diverged)",
                     label, next_tok, r_tok_lat);
            errors = errors + 1;
        end
        // (1b) BIT-EXACT LOGIT binding: the full pre-argmax logit vector must match
        //   the independent (layer,pos)-keyed reference.  Far more sensitive than
        //   argmax alone -- a layer alias (layer m reads window k!=m) perturbs the
        //   attention context and shifts logits even when the argmax happens not to
        //   move.  This is what makes the layer-aliasing INJECTION bite.
        if (logits !== r_logits) begin
            $display("FAIL[%0s]: BINDING logits != shadow-ref logits (SELF_KV L=6 per-(layer,pos) KV corrupted)", label);
            errors = errors + 1;
        end
        // (2) committed token X/Z-clean and == the system internal argmax
        for (b=0;b<TOKW;b=b+1) if (next_tok[b]===1'bx || next_tok[b]===1'bz) begin
            $display("FAIL[%0s]: next_tok bit %0d X/Z", label, b); errors=errors+1; end
        if (next_tok !== argmax_o) begin
            $display("FAIL[%0s]: next_tok=%0d != system internal argmax=%0d", label, next_tok, argmax_o);
            errors=errors+1; end
        // (3) the WRITE-BACK path advanced the (last-layer) window counter by one this
        //   token (not the host stub).  kv_append_count = count[db_layer]; between
        //   tokens db_layer holds L-1, whose window appends exactly once per token.
        if (kv_append_count !== abefore + 1'b1) begin
            $display("FAIL[%0s]: append_count=%0d expected %0d (last-layer window should grow by 1/token)",
                     label, kv_append_count, abefore + 1);
            errors=errors+1; end
        // (4) the pager actually served gather rows (except the first, empty-KV token)
        if (SL != 0 && kv_rowvalid_cnt <= krv_before) begin
            $display("FAIL[%0s]: pager produced no gather row_valid (die did not READ its KV)", label);
            errors=errors+1; end

        $display("PASS[%0s] tok=%0d(==ref) pos=%0d s_len=%0d append_count=%0d(+1) gathers+=%0d",
                 label, next_tok, ps, SL, kv_append_count, kv_rowvalid_cnt - krv_before);
        krv_before = kv_rowvalid_cnt;
    end endtask

    initial begin
        #16000000; $display("FAIL: global timeout"); $fatal(1, "timeout");
    end

    initial begin
        errors=0; test_count=0; kv_rowvalid_cnt=0; krv_before=0;
        rst=1'b1; start=1'b0; r_start=1'b0;
        prompt_tok={TOKW{1'b0}}; start_pos={POSW{1'b0}}; s_len={(IDXW+1){1'b0}};
        wl_mem_data={WL_DATA_W{1'b0}};
        repeat(4) @(negedge clk);
        rst=1'b0;
        @(negedge clk);

        // ---- self-attending decode at L=6: pos=t, s_len=t ----
        //   token 0 has EMPTY KV (s_len=0); each later token READS, FOR EACH of the 6
        //   layers, that layer's own rows written by earlier tokens (per-(layer,pos)),
        //   and appends its own row for each of the 6 layers.
        run_token(4'd5,     20'd0, 0, "tok0 s0 (empty KV, 6 layers)");
        run_token(next_tok, 20'd1, 1, "tok1 s1 (reads pos0 x6 layers)");
        run_token(next_tok, 20'd2, 2, "tok2 s2 (reads pos0,1 x6 layers)");
        run_token(next_tok, 20'd3, 3, "tok3 s3 (reads pos0..2 x6 layers)");
        run_token(next_tok, 20'd4, 4, "tok4 s4 (reads pos0..3 x6 layers)");
        run_token(next_tok, 20'd5, 5, "tok5 s5 (reads pos0..4 x6 layers)");

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d tests", errors, test_count);
            $fatal(1, "glm_q4k_self_kv_l6_roundtrip_tb: SELF_KV=1 L=6 per-(layer,pos) round-trip mismatch");
        end
        $display("ALL %0d TESTS PASSED  (glm_q4k_system SELF_KV=1 L=6 per-(layer,position) KV write-back round-trip == independent (layer,pos)-keyed shadow-fed glm_model_q4k, %0d-token self-attending decode; each of 6 layers attends ONLY its own KV)",
                 test_count, test_count);
        $finish;
    end

endmodule
