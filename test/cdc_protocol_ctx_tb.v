`timescale 1ns/1ps
//============================================================================
// cdc_protocol_ctx_tb.v -- self-checking TB for the PROTO_CTX=1 protocol
//   extension of glm_q4k_system_cdc (USAGE_GAPS §C, findings #19/#26).
//----------------------------------------------------------------------------
// WHAT THIS PROVES (the NEW behavior, default-OFF elsewhere)
//   The 2-clock host<->device top is instantiated with PROTO_CTX=1 and driven
//   across its TWO ASYNCHRONOUS clocks (host_clk = USB domain, core_clk =
//   compute domain).  The compute box is fed the SAME faithful weight/KV/Flash/
//   DDR5/loader memory backing used by the perf TB, so it produces REAL tokens.
//   Then:
//     (1) CONTEXT ROUND-TRIP / DEMUX: several token-generation requests are
//         issued, each tagged with a DISTINCT context/sequence id.  Every
//         emitted-token response must carry BACK the id of the request that
//         produced it (resp_ctx_id == the request's ctx), with resp_is_telem=0
//         and an X-clean token -- i.e. the host can multiplex N contexts and
//         demultiplex the returned tokens.
//     (2) TELEMETRY READBACK: an OP_TELEM request returns, through the SAME
//         response FIFO tagged resp_is_telem=1 and carrying its own ctx id, a
//         snapshot of the device counters {tokens, runs, done, stall}.  The TB
//         independently counts host-visible tokens / runs / completions and
//         checks the readback EXACTLY; the stall field is checked X-clean and
//         consistent (<= the live core stall counter).  Two readbacks taken at
//         different points must show the counters ADVANCING (guards a stub that
//         returns a constant).
//
// FALSIFIABILITY (this TB can fail):
//   * a wrapper that dropped/misrouted the ctx id -> ctx mismatch FAIL;
//   * a wrapper that returned a token for a telemetry request (or vice-versa)
//     -> resp_is_telem tag FAIL;
//   * a telemetry stub returning constant/zero counters -> exact-count FAIL and
//     the advancing-counter FAIL;
//   * an X on any response field -> X FAIL.
//
// CROSS-CHECK vs default: the SAME source at PROTO_CTX=0 is proven sequentially
//   equivalent to the pre-change top by `make cdc-protocol-equiv`.
//============================================================================
module cdc_protocol_ctx_tb;

    // ---- slice geometry (the perf TB's small-but-faithful, token-producing slice) ----
    localparam integer MODEL_DIM  = 16;
    localparam integer L          = 4;
    localparam integer N_DENSE    = 2;
    localparam integer VOCAB      = 16;
    localparam integer H_HEADS    = 2;
    localparam integer NOPE       = 4;
    localparam integer ROPE       = 4;
    localparam integer V_DIM      = 4;
    localparam integer Q_LORA     = 8;
    localparam integer KV_LORA    = 8;
    localparam integer S_MAX      = 4;
    localparam integer TOPK_ATTN  = 4;
    localparam integer THETA      = 8000000;
    localparam integer PE_N       = 2;
    localparam integer POSW       = 20;
    localparam integer N_EXPERT   = 4;
    localparam integer TOPK       = 2;
    localparam integer INTER_MOE  = 16;
    localparam integer INTER_DENSE= 32;
    localparam [31:0]  RSCALE     = 32'h40200000;
    localparam integer TN         = 4;
    localparam integer BLK        = 128;
    localparam integer LM_TN      = 4;
    // ---- memory system ----
    localparam integer CACHE_SLOTS = 4;
    localparam integer FLASH_LAT   = 8;
    localparam integer KV_CTX      = 1024;
    localparam integer KV_RESIDENT = 16;
    localparam integer EFIFO_DEPTH = 16;
    localparam integer RESIDENT    = 0;
    localparam integer DDR_NCH     = 4;
    localparam integer DDR_ADDR_W  = 32;
    localparam integer DDR_DATA_W  = 256;
    localparam integer DDR_TAG_W   = 8;
    localparam integer DDR_ROW_LAT = 10;
    localparam integer DDR_RESP_QD = 4;
    localparam integer WL_KMAX     = 256;
    localparam integer WL_ADDR_W   = 24;
    localparam integer LOADER_KLEN = MODEL_DIM;
    localparam integer REQ_AW      = 2;
    localparam integer TOK_AW      = 3;
    // ---- protocol-extension config (this TB drives PROTO_CTX=1) ----
    localparam integer PROTO_CTX   = 1;
    localparam integer CTX_W       = 8;
    localparam integer TELEM_W     = 32;

    // ---- derived (mirror glm_q4k_system_cdc / glm_model_q4k) ----
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
    // opcodes (must match the DUT localparams)
    localparam [1:0] OP_TOKEN = 2'd0;
    localparam [1:0] OP_TELEM = 2'd1;

    // ================= TWO ASYNCHRONOUS CLOCKS =================
    reg core_clk = 1'b0;  always #5 core_clk = ~core_clk;   // 10 ns compute domain
    reg host_clk = 1'b0;  always #7 host_clk = ~host_clk;   // 14 ns USB domain (async ratio)
    reg core_rst, host_rst;

    // ================= deterministic weight generators (perf-TB copy) =====
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
    function automatic [3:0] f_fwq; input integer ly; input integer sel;
        input integer shr; input integer eidx; input integer fo; input integer kk; begin
        f_fwq = gen_q4(ly*7919 + sel*104729 + shr*15485863 + eidx*350377 + fo*611953 + kk*13 + 503);
    end endfunction

    // ================= DUT host-side signals =================
    reg                     start;
    reg  [TOKW-1:0]         prompt_tok;
    reg  [POSW-1:0]         start_pos;
    reg  [IDXW:0]           s_len;
    reg  [CTX_W-1:0]        req_ctx_id;
    reg  [1:0]              req_opcode;
    wire                    busy, done;
    wire [TOKW-1:0]         next_tok;
    wire                    tok_valid;
    wire [CTX_W-1:0]        resp_ctx_id;
    wire                    resp_is_telem;
    wire [4*TELEM_W-1:0]    resp_telem;

    // ================= DUT core-side memory ports =================
    wire [VOCAB*16-1:0]     logits;
    wire                    em_req;  wire [TOKW-1:0] em_tok;  wire [DIMW-1:0] em_idx;  reg [15:0] em_val;
    wire [LAYW-1:0]         db_layer;  wire idx_fresh;  wire [LAYW-1:0] idx_win;
    wire                    gn_req, gn_which;  wire [DIMW-1:0] gn_idx;  reg [15:0] gn_val;
    wire                    aw_req;  wire [3:0] aw_sel;  wire [A_GRPW-1:0] aw_grp;  wire [A_KCW-1:0] aw_k;
    reg  [PE_N*4-1:0]       aw_q;
    reg  [16*PE_N*A_NSB-1:0] aw_d, aw_dmin;
    reg  [96*PE_N*A_NSB-1:0] aw_scales;
    wire                    rw_req;  wire [R_KW-1:0] rw_k;
    reg  [4*N_EXPERT-1:0]        rw_q;
    reg  [16*N_EXPERT*R_NSB-1:0] rw_d, rw_dmin;
    reg  [96*N_EXPERT*R_NSB-1:0] rw_scales;
    wire                    fw_req;  wire [1:0] fw_sel;  wire [FF_GWD-1:0] fw_grp;  wire [FF_KWD-1:0] fw_k;
    wire                    fw_shared;  wire [EIDXW-1:0] fw_eidx;
    reg  [4*TN-1:0]         fw_q, fw_q_up;
    reg  [16*TN*FF_NSB_D-1:0] fw_d_g, fw_dmin_g, fw_d_u, fw_dmin_u;
    reg  [96*TN*FF_NSB_D-1:0] fw_scales_g, fw_scales_u;
    wire                    fn_req;  wire [DIMW-1:0] fn_idx;  reg [15:0] fn_val;
    wire                    lw_req;  wire [VTW-1:0] lw_vtile;  wire [DIMW-1:0] lw_k;  reg [LM_TN*16-1:0] lw_col;
    wire                    kc_req;  wire [IDXW-1:0] kc_idx;  reg [KV_LORA*16-1:0] kc_ckv;  reg [ROPE*16-1:0] kc_krope;
    wire [KVPOSW-1:0]       kv_row_sel;  reg [ROW_BITS-1:0] kv_row_in;
    wire                    flash_req, flash_is_expert;
    wire [EIDXW-1:0]        flash_expert_id;  wire [KVPOSW-1:0] flash_row_idx;
    reg                     flash_done;  reg [ROW_BITS-1:0] flash_row;
    wire [TOKW-1:0]         argmax_o;  wire [MODEL_DIM*16-1:0] h_state;  wire mdl_busy;
    wire                    ec_resp_valid, ec_hit;  wire [CSLOTW-1:0] ec_resp_slot;  wire ec_busy;
    wire [31:0]             ec_hit_count, ec_miss_count, ec_demand_stall_cycles, ec_pf_issued, ec_pf_hit;
    wire                    kv_row_valid;  wire [ROW_BITS-1:0] kv_row_out;  wire kv_busy;
    wire [KVPOSW-1:0]       kv_append_count, kv_resident_lo;  wire kv_overflowed;
    wire [31:0]             ec_dropped;
    wire [DDR_NCH-1:0]            mem_req_valid;
    wire [DDR_NCH-1:0]            mem_req_ready;
    wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr;
    wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag;
    reg  [DDR_NCH-1:0]            mem_resp_valid;
    wire [DDR_NCH-1:0]            mem_resp_ready;
    reg  [DDR_NCH*DDR_DATA_W-1:0] mem_resp_data;
    reg  [DDR_NCH*DDR_TAG_W-1:0]  mem_resp_tag;
    wire                    wl_mem_en;  wire [WL_ADDR_W-1:0] wl_mem_addr;  reg [WL_DATA_W-1:0] wl_mem_data;
    wire [31:0]             xbar_req_count, xbar_resp_count;
    wire                    xbar_resp_valid;  wire [DDR_DATA_W-1:0] xbar_resp_data;
    wire                    loader_busy;  wire [31:0] loader_done_count, loader_beat_count;
    wire [4*WL_PE_N-1:0]    loader_w_q;  wire loader_in_valid;

    glm_q4k_system_cdc #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN),
        .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX),
        .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH), .RESIDENT(RESIDENT),
        .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W),
        .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD),
        .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN),
        .REQ_AW(REQ_AW), .TOK_AW(TOK_AW),
        .PROTO_CTX(PROTO_CTX), .CTX_W(CTX_W), .TELEM_W(TELEM_W)
    ) dut (
        .host_clk(host_clk), .host_rst(host_rst),
        .core_clk(core_clk), .core_rst(core_rst),
        .start(start), .prompt_tok(prompt_tok), .start_pos(start_pos), .s_len(s_len),
        .busy(busy), .done(done), .next_tok(next_tok), .tok_valid(tok_valid),
        .req_ctx_id(req_ctx_id), .req_opcode(req_opcode),
        .resp_ctx_id(resp_ctx_id), .resp_is_telem(resp_is_telem), .resp_telem(resp_telem),
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
        .flash_req(flash_req), .flash_is_expert(flash_is_expert),
        .flash_expert_id(flash_expert_id), .flash_row_idx(flash_row_idx),
        .flash_done(flash_done), .flash_row(flash_row),
        .pf_valid(1'b0), .pf_expert_id({EIDXW{1'b0}}),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .wl_mem_en(wl_mem_en), .wl_mem_addr(wl_mem_addr), .wl_mem_data(wl_mem_data),
        .argmax_o(argmax_o), .h_state(h_state), .mdl_busy(mdl_busy),
        .ec_resp_valid(ec_resp_valid), .ec_hit(ec_hit), .ec_resp_slot(ec_resp_slot),
        .ec_busy(ec_busy), .ec_hit_count(ec_hit_count), .ec_miss_count(ec_miss_count),
        .ec_demand_stall_cycles(ec_demand_stall_cycles),
        .ec_pf_issued(ec_pf_issued), .ec_pf_hit(ec_pf_hit),
        .kv_row_valid(kv_row_valid), .kv_row_out(kv_row_out), .kv_busy(kv_busy),
        .kv_append_count(kv_append_count), .kv_resident_lo(kv_resident_lo),
        .kv_overflowed(kv_overflowed), .ec_dropped(ec_dropped),
        .xbar_req_count(xbar_req_count), .xbar_resp_count(xbar_resp_count),
        .xbar_resp_valid(xbar_resp_valid), .xbar_resp_data(xbar_resp_data),
        .loader_busy(loader_busy), .loader_done_count(loader_done_count),
        .loader_beat_count(loader_beat_count),
        .loader_w_q(loader_w_q), .loader_in_valid(loader_in_valid)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, logits, em_req, aw_req, fw_req, rw_req, gn_req, fn_req,
                     lw_req, idx_fresh, idx_win, mdl_busy, ec_resp_valid, ec_hit,
                     ec_resp_slot, ec_busy, ec_hit_count, ec_miss_count, ec_pf_issued,
                     ec_pf_hit, kv_row_out, kv_busy, kv_resident_lo, flash_expert_id,
                     h_state, gn_which, kv_overflowed, kv_append_count, argmax_o,
                     xbar_req_count, xbar_resp_count, xbar_resp_valid, xbar_resp_data,
                     loader_busy, loader_done_count, loader_beat_count, loader_w_q,
                     loader_in_valid, mem_req_tag, ec_dropped};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= core-domain weight/KV responders (perf-TB backing) =====
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
    integer cd;
    always @* begin
        for (cd=0;cd<KV_LORA;cd=cd+1) kc_ckv  [16*cd+:16] = gen_bf16(db_layer*513 + kc_idx*67 + cd*7 + 8011);
        for (cd=0;cd<ROPE;cd=cd+1)    kc_krope[16*cd+:16] = gen_bf16(db_layer*771 + kc_idx*91 + cd*5 + 8101);
    end

    // ================= KV latent-ROW stub =================
    integer rr;
    always @* begin
        kv_row_in = {ROW_BITS{1'b0}};
        for (rr=0;rr<(KV_LORA+ROPE);rr=rr+1)
            kv_row_in[16*rr+:16] = gen_bf16(kv_row_sel*131 + rr*7 + 3);
    end

    // ================= FLASH PHY STUB (FLASH_LAT-cycle fetch) =================
    reg [31:0] fl_timer; reg fl_active; reg prev_req;
    always @(posedge core_clk) begin
        if (core_rst) begin
            fl_timer <= 32'd0; fl_active <= 1'b0; flash_done <= 1'b0; prev_req <= 1'b0;
        end else begin
            flash_done <= 1'b0;
            if (!fl_active) begin
                if (flash_req && !prev_req) begin fl_active <= 1'b1; fl_timer <= FLASH_LAT[31:0]; end
            end else begin
                if (fl_timer <= 32'd1) begin flash_done <= 1'b1; fl_active <= 1'b0; end
                else fl_timer <= fl_timer - 32'd1;
            end
            prev_req <= flash_req;
        end
    end
    integer cr;
    always @* begin
        flash_row = {ROW_BITS{1'b0}};
        for (cr=0;cr<(KV_LORA+ROPE);cr=cr+1)
            flash_row[16*cr+:16] = gen_bf16(flash_row_idx*977 + cr*13 + 1);
    end

    // ================= LOADER STAGING-TIER RAM (latency-1 registered read) =======
    localparam integer STG_DEPTH = 2048;
    reg [WL_DATA_W-1:0] STAGE [0:STG_DEPTH-1];
    integer si, sj;
    initial for (si=0; si<STG_DEPTH; si=si+1)
        for (sj=0; sj<WL_DATA_W/16; sj=sj+1)
            STAGE[si][16*sj+:16] = gen_bf16(si*61 + sj*17 + 9);
    always @(posedge core_clk) begin
        if (wl_mem_en) wl_mem_data <= STAGE[wl_mem_addr[10:0]];
        else           wl_mem_data <= {WL_DATA_W{1'b0}};
    end

    // ================= DDR5 PER-CHANNEL MEMORY MODEL =================
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
                    ((DDR_NCH==1) ? (cc==0) : (infad[ii][CH_SEL_W-1:0]==cc[CH_SEL_W-1:0]))) begin
                    presV[cc]=1'b1; presIdx[cc]=ii;
                end
            end
            if (presV[cc]) begin
                mem_resp_valid[cc] = 1'b1;
                mem_resp_data[cc*DDR_DATA_W +: DDR_DATA_W] = gen_beat(inftg[presIdx[cc]], infad[presIdx[cc]]);
                mem_resp_tag[cc*DDR_TAG_W +: DDR_TAG_W] = inftg[presIdx[cc]];
            end
        end
    end
    integer freeslot; reg got_free;
    always @(posedge core_clk) begin
        if (core_rst) begin
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
                    infv[freeslot]<=1'b1;
                    inftg[freeslot]<=mem_req_tag[cc*DDR_TAG_W +: DDR_TAG_W];
                    infad[freeslot]<=mem_req_addr[cc*DDR_ADDR_W +: DDR_ADDR_W];
                    inftm[freeslot]<=DDR_ROW_LAT[15:0];
                end
            end
        end
    end

    // ================= host-side observers + response capture =================
    integer errors;
    integer host_tok_cnt;   // host-visible TOKEN responses (is_telem==0)
    integer host_done_cnt;  // host-visible done pulses
    integer host_req_cnt;   // token-gen requests ACCEPTED into the request FIFO

    // count host-visible done pulses (used to serialize single-outstanding runs)
    always @(posedge host_clk) if (!host_rst && done) host_done_cnt = host_done_cnt + 1;

    // continuous X-cleanliness monitor on the response fields whenever a
    // response is presented (any tok_valid pulse).
    integer b;
    always @(posedge host_clk) if (!host_rst && tok_valid) begin
        for (b=0;b<CTX_W;b=b+1) if (resp_ctx_id[b]===1'bx||resp_ctx_id[b]===1'bz) begin
            $display("FAIL: resp_ctx_id X/Z @%0t", $time); errors=errors+1; b=CTX_W; end
        if (resp_is_telem===1'bx||resp_is_telem===1'bz) begin
            $display("FAIL: resp_is_telem X/Z @%0t", $time); errors=errors+1; end
        if (resp_is_telem===1'b0)
            for (b=0;b<TOKW;b=b+1) if (next_tok[b]===1'bx||next_tok[b]===1'bz) begin
                $display("FAIL: token X/Z @%0t", $time); errors=errors+1; b=TOKW; end
        else
            for (b=0;b<4*TELEM_W;b=b+1) if (resp_telem[b]===1'bx||resp_telem[b]===1'bz) begin
                $display("FAIL: resp_telem X/Z @%0t", $time); errors=errors+1; b=4*TELEM_W; end
    end

    // ---- captured response fields ----
    reg [TOKW-1:0]      cap_tok;
    reg [CTX_W-1:0]     cap_ctx;
    reg                 cap_is_telem;
    reg [4*TELEM_W-1:0] cap_telem;

    // Wait (on host_clk) for the next response pulse and latch its fields.
    // tok_valid, next_tok and the resp_* fields are all registered off the same
    // tok_rd_d, so they are jointly valid on the negedge of the pulse cycle.
    task get_response;
        begin
            forever begin
                @(negedge host_clk);
                if (tok_valid === 1'b1) begin
                    cap_tok      = next_tok;
                    cap_ctx      = resp_ctx_id;
                    cap_is_telem = resp_is_telem;
                    cap_telem    = resp_telem;
                    if (cap_is_telem === 1'b0) host_tok_cnt = host_tok_cnt + 1;
                    disable get_response;
                end
            end
        end
    endtask

    // Issue one request on host_clk (single rising edge of `start`).
    task issue; input [TOKW-1:0] tk; input [POSW-1:0] ps; input integer SL;
        input [CTX_W-1:0] ctx; input [1:0] op; begin
        @(negedge host_clk);
        prompt_tok = tk; start_pos = ps; s_len = SL[IDXW:0];
        req_ctx_id = ctx; req_opcode = op; start = 1'b1;
        if (op === OP_TOKEN) host_req_cnt = host_req_cnt + 1;
        @(negedge host_clk);
        start = 1'b0;
    end endtask

    // Wait until at least `n` done pulses have been seen (with timeout guard).
    task wait_done_ge; input integer n; integer g; begin
        g = 0;
        while (host_done_cnt < n) begin
            @(negedge host_clk); g = g + 1;
            if (g > 2000000) begin $display("FAIL: timeout waiting for done>=%0d (have %0d)", n, host_done_cnt); errors=errors+1; disable wait_done_ge; end
        end
    end endtask

    // ---- one full token-generation run: request(ctx) -> token response -> done ----
    task run_token; input [TOKW-1:0] tk; input [POSW-1:0] ps; input integer SL;
        input [CTX_W-1:0] ctx; integer dbefore; begin
        dbefore = host_done_cnt;
        issue(tk, ps, SL, ctx, OP_TOKEN);
        get_response;                       // capture the emitted token + its ctx
        if (cap_is_telem !== 1'b0) begin
            $display("FAIL[ctx %0h]: token run returned is_telem=1", ctx); errors=errors+1; end
        if (cap_ctx !== ctx) begin
            $display("FAIL[ctx %0h]: response ctx=%0h != request ctx=%0h (round-trip)", ctx, cap_ctx, ctx); errors=errors+1; end
        wait_done_ge(dbefore + 1);          // run completed
        repeat (40) @(negedge host_clk);    // drain CDC FIFOs / in-flight reads
        $display("PASS[token ctx=%0h] tok=%0d is_telem=%0b ctx_echo=%0h", ctx, cap_tok, cap_is_telem, cap_ctx);
    end endtask

    // ---- telemetry readback: request(OP_TELEM,ctx) -> snapshot response ----
    // Checks the exact deterministic counters {tokens,runs,done} against the
    // host-side tally and the stall field for X-clean + consistency.
    task telem_read; input [CTX_W-1:0] ctx;
        input integer exp_tok; input integer exp_run; input integer exp_done;
        reg [TELEM_W-1:0] f_tok, f_run, f_done, f_stall; begin
        issue(0, 0, 0, ctx, OP_TELEM);
        get_response;
        if (cap_is_telem !== 1'b1) begin
            $display("FAIL[telem ctx %0h]: readback returned is_telem=0 (token, not telemetry)", ctx); errors=errors+1; end
        if (cap_ctx !== ctx) begin
            $display("FAIL[telem ctx %0h]: response ctx=%0h != request ctx", ctx, cap_ctx); errors=errors+1; end
        f_tok   = cap_telem[0*TELEM_W +: TELEM_W];
        f_run   = cap_telem[1*TELEM_W +: TELEM_W];
        f_done  = cap_telem[2*TELEM_W +: TELEM_W];
        f_stall = cap_telem[3*TELEM_W +: TELEM_W];
        if (f_tok !== exp_tok) begin
            $display("FAIL[telem ctx %0h]: tokens=%0d expected %0d", ctx, f_tok, exp_tok); errors=errors+1; end
        if (f_run !== exp_run) begin
            $display("FAIL[telem ctx %0h]: runs=%0d expected %0d", ctx, f_run, exp_run); errors=errors+1; end
        if (f_done !== exp_done) begin
            $display("FAIL[telem ctx %0h]: done=%0d expected %0d", ctx, f_done, exp_done); errors=errors+1; end
        // stall snapshot was sampled in core domain BEFORE crossing; it must be
        // X-clean and cannot exceed the (monotonically increasing) live counter.
        if (f_stall > ec_demand_stall_cycles) begin
            $display("FAIL[telem ctx %0h]: stall snapshot %0d > live %0d", ctx, f_stall, ec_demand_stall_cycles); errors=errors+1; end
        $display("PASS[telem ctx=%0h] tokens=%0d runs=%0d done=%0d stall=%0d (live=%0d)",
                 ctx, f_tok, f_run, f_done, f_stall, ec_demand_stall_cycles);
    end endtask

    // ================= global timeout =================
    initial begin
        #400000000;
        $display("FAIL: global timeout"); $fatal;
    end

    integer test_count;
    reg [TELEM_W-1:0] stall_after2;
    initial begin
        errors=0; test_count=0;
        host_tok_cnt=0; host_done_cnt=0; host_req_cnt=0;
        start=1'b0; prompt_tok=0; start_pos=0; s_len=0; req_ctx_id=0; req_opcode=OP_TOKEN;
        flash_done=1'b0; wl_mem_data={WL_DATA_W{1'b0}};
        core_rst=1'b1; host_rst=1'b1;
        repeat (8) @(negedge core_clk);
        core_rst=1'b0; host_rst=1'b0;
        repeat (8) @(negedge host_clk);

        // ---- (1) CONTEXT ROUND-TRIP: 4 token runs, each a DISTINCT ctx id ----
        //   pos == s_len (the real decode relation); feed the returned token back.
        run_token(4'd7,     20'd1, 1, 8'hA1); test_count=test_count+1;
        run_token(cap_tok,  20'd2, 2, 8'hB2); test_count=test_count+1;

        // ---- (2a) TELEMETRY after 2 runs: counters must read exactly 2 ----
        telem_read(8'h5A, host_tok_cnt, host_req_cnt, host_done_cnt); test_count=test_count+1;
        stall_after2 = cap_telem[3*TELEM_W +: TELEM_W];

        // ---- more token runs with fresh ctx ids ----
        run_token(cap_tok,  20'd3, 3, 8'hC3); test_count=test_count+1;
        run_token(cap_tok,  20'd4, 4, 8'hD4); test_count=test_count+1;

        // ---- (2b) TELEMETRY after 4 runs: counters advanced to exactly 4 ----
        telem_read(8'hE7, host_tok_cnt, host_req_cnt, host_done_cnt); test_count=test_count+1;

        // ---- (2c) counters ADVANCED between the two readbacks (not a constant) ----
        if (cap_telem[0*TELEM_W +: TELEM_W] <= 2) begin
            $display("FAIL: telemetry token counter did not advance past first readback"); errors=errors+1; end
        if (cap_telem[3*TELEM_W +: TELEM_W] < stall_after2) begin
            $display("FAIL: telemetry stall counter went backwards"); errors=errors+1; end
        test_count=test_count+1;

        // ---- (3) protocol still alive after a readback: one more ctx round-trip ----
        run_token(cap_tok,  20'd1, 1, 8'h9F); test_count=test_count+1;

        if (errors!=0) begin
            $display("FAILED: %0d error(s) across %0d tests", errors, test_count);
            $fatal;
        end
        $display("[cdc_protocol_ctx] ALL %0d TESTS PASSED  (ctx-id round-trip across host<->core CDC + telemetry readback; PROTO_CTX=1)", test_count);
        $finish;
    end
endmodule
