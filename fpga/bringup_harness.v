// ============================================================================
// bringup_harness.v -- SYNTHESIZABLE place-and-route harness for the whole
//   GLM-5.2 Q4_K product top `glm_q4k_system_cdc`.
//
//   WHY THIS EXISTS
//   `glm_q4k_system_cdc` exposes its entire memory subsystem at the top level:
//   the DDR5 crossbar bus, the flash/expert-cache ring, the KV pager row bus, the
//   weight-loader DMA, and per-matmul dequant response buses (aw_/rw_/fw_scales
//   are 96*N*NSB bits each), plus a VOCAB*16 `logits` bus and dozens of telemetry
//   counters. On a REAL package that is thousands of I/O -- full place & route
//   fails on I/O count long before it fails on logic. `report_utilization` after
//   `synth_design` still gives the resource fit (no pins needed), but the ROUTED
//   Fmax -- the timing-closure number -- needs a design that actually fits the pkg.
//
//   WHAT IT DOES  (classic synthesis harness; same technique as the retired
//   sm_harness.v / gemm_harness.v used for their DUTs)
//     * Buries EVERY wide memory-side port inside the chip:
//         - every response INPUT (weights, KV rows, flash rows, DDR beats, ...)
//           is driven from a free-running 256-bit LFSR `r`. Non-constant drivers
//           keep the synthesizer from constant-folding the dequant/MAC datapath
//           away, so the fit is REAL (not a pruned shell).
//         - every request/telemetry/logits OUTPUT is XOR-reduced into a single
//           registered bit `crc_out`, and also fed back into the LFSR so the
//           request-generating logic can't be pruned either.
//     * Exposes only 5 clock/control pins + 1 output -> trivially routable I/O.
//
//   The token VALUES are meaningless here (the LFSR is not real weights) -- this
//   harness is for the FIT ONLY (utilization + routed Fmax). Functional proof is
//   the iverilog testbenches + the assembled glm_model_q4k golden. The registered
//   LFSR/CRC feedback means there are NO combinational loops (yosys check -assert
//   clean); the DUT's own logic is unchanged.
//
//   Defaults are the COMPACT (result-invariant resource) config the fit targets
//   -- PE_N=2, DDR_NCH=2, KV_RESIDENT=8, EFIFO_DEPTH=8, CACHE_SLOTS=2 -- passed
//   straight through to the DUT. Override the generics to fit the default config.
// ============================================================================
`default_nettype none

module bringup_harness #(
    // ---- compute-die (glm_model_q4k) slice config -- passed straight through --
    parameter integer MODEL_DIM  = 128,
    parameter integer L          = 6,
    parameter integer N_DENSE    = 3,
    parameter integer VOCAB      = 256,
    parameter integer H_HEADS    = 4,
    parameter integer NOPE       = 16,
    parameter integer ROPE       = 16,
    parameter integer V_DIM      = 32,
    parameter integer Q_LORA     = 64,
    parameter integer KV_LORA    = 32,
    parameter integer S_MAX      = 8,
    parameter integer TOPK_ATTN  = 8,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 2,     // compact (default cfg: 4)
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 8,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 64,
    parameter integer INTER_DENSE= 256,
    parameter [31:0]  RSCALE     = 32'h40200000,
    parameter integer TN         = 4,
    parameter integer BLK        = 128,
    parameter integer LM_TN      = 4,
    parameter integer ACT_HW     = 1,     // compact fit: serialize glm_act lanes (result-invariant; default cfg: 0=full)
    // ---- memory-system config ----
    parameter integer CACHE_SLOTS = 2,    // compact (default cfg: 4)
    parameter integer FLASH_LAT   = 8,
    parameter integer KV_CTX      = 1024,
    parameter integer KV_RESIDENT = 8,    // compact (default cfg: 16)
    parameter integer EFIFO_DEPTH = 8,    // compact (default cfg: 16)
    // ---- DDR5 fast-tier fabric (ddr5_xbar) config ----
    parameter integer DDR_NCH     = 2,    // compact (default cfg: 4)
    parameter integer DDR_ADDR_W  = 32,
    parameter integer DDR_DATA_W  = 256,
    parameter integer DDR_TAG_W   = 8,
    parameter integer DDR_ROW_LAT = 10,
    parameter integer DDR_RESP_QD = 4,
    // ---- weight_loader_q4k (matmul weight-pull DMA) config ----
    parameter integer WL_KMAX     = 256,
    parameter integer WL_ADDR_W   = 24,
    parameter integer LOADER_KLEN = MODEL_DIM,
    // ---- CDC FIFO depths ----
    parameter integer REQ_AW      = 2,
    parameter integer TOK_AW      = 3,
    // ====================================================================
    // derived (do NOT override) -- mirror glm_q4k_system_cdc's derivations so the
    // response-bus slice widths below match the DUT ports exactly.
    // ====================================================================
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
    parameter integer CSLOTW     = (CACHE_SLOTS <= 1) ? 1 : $clog2(CACHE_SLOTS),
    parameter integer WL_PE_N    = PE_N,
    parameter integer WL_DATA_W  = 256,
    // widest response bus fits in RRW bits (default cfg: mem_resp_data = 4*256).
    parameter integer RRW        = 2048
)(
    input  wire host_clk,
    input  wire host_rst,
    input  wire core_clk,
    input  wire core_rst,
    input  wire start,
    output wire crc_out
);
    // ---------------- entropy source: free-running LFSR + output feedback -------
    reg  [255:0] r;
    wire [RRW-1:0] rr = {(RRW/256){r}};          // tile the LFSR to the widest bus

    // ---------------- DUT output nets (every output -> folded into out_fold) ----
    wire                          busy, done, tok_valid;
    wire [TOKW-1:0]               next_tok;
    wire [VOCAB*16-1:0]           logits;
    wire                          em_req;
    wire [TOKW-1:0]               em_tok;
    wire [DIMW-1:0]               em_idx;
    wire [LAYW-1:0]               db_layer;
    wire                          idx_fresh;
    wire [LAYW-1:0]               idx_win;
    wire                          gn_req, gn_which;
    wire [DIMW-1:0]               gn_idx;
    wire                          aw_req;
    wire [3:0]                    aw_sel;
    wire [A_GRPW-1:0]             aw_grp;
    wire [A_KCW-1:0]              aw_k;
    wire                          rw_req;
    wire [R_KW-1:0]               rw_k;
    wire                          fw_req;
    wire [1:0]                    fw_sel;
    wire [FF_GWD-1:0]             fw_grp;
    wire [FF_KWD-1:0]             fw_k;
    wire                          fw_shared;
    wire [EIDXW-1:0]              fw_eidx;
    wire                          fn_req;
    wire [DIMW-1:0]               fn_idx;
    wire                          lw_req;
    wire [VTW-1:0]                lw_vtile;
    wire [DIMW-1:0]               lw_k;
    wire                          kc_req;
    wire [IDXW-1:0]              kc_idx;
    wire [KVPOSW-1:0]            kv_row_sel;
    wire                          flash_req, flash_is_expert;
    wire [EIDXW-1:0]              flash_expert_id;
    wire [KVPOSW-1:0]            flash_row_idx;
    wire [DDR_NCH-1:0]            mem_req_valid;
    wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr;
    wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag;
    wire [DDR_NCH-1:0]            mem_resp_ready;
    wire                          wl_mem_en;
    wire [WL_ADDR_W-1:0]          wl_mem_addr;
    wire [TOKW-1:0]               argmax_o;
    wire [MODEL_DIM*16-1:0]       h_state;
    wire                          mdl_busy;
    wire                          ec_resp_valid, ec_hit, ec_busy;
    wire [CSLOTW-1:0]            ec_resp_slot;
    wire [31:0]                   ec_hit_count, ec_miss_count, ec_demand_stall_cycles;
    wire [31:0]                   ec_pf_issued, ec_pf_hit;
    wire                          kv_row_valid, kv_busy, kv_overflowed;
    wire [ROW_BITS-1:0]           kv_row_out;
    wire [KVPOSW-1:0]            kv_append_count, kv_resident_lo;
    wire [31:0]                   ec_dropped, xbar_req_count, xbar_resp_count;
    wire                          xbar_resp_valid;
    wire [DDR_DATA_W-1:0]         xbar_resp_data;
    wire                          loader_busy, loader_in_valid;
    wire [31:0]                   loader_done_count, loader_beat_count;
    wire [4*WL_PE_N-1:0]          loader_w_q;

    // ---------------- fold EVERY output bit into one signal --------------------
    wire out_fold = ^{
        busy, done, next_tok, tok_valid, logits,
        em_req, em_tok, em_idx, db_layer, idx_fresh, idx_win,
        gn_req, gn_which, gn_idx,
        aw_req, aw_sel, aw_grp, aw_k,
        rw_req, rw_k,
        fw_req, fw_sel, fw_grp, fw_k, fw_shared, fw_eidx,
        fn_req, fn_idx,
        lw_req, lw_vtile, lw_k,
        kc_req, kc_idx, kv_row_sel,
        flash_req, flash_is_expert, flash_expert_id, flash_row_idx,
        mem_req_valid, mem_req_addr, mem_req_tag, mem_resp_ready,
        wl_mem_en, wl_mem_addr,
        argmax_o, h_state, mdl_busy,
        ec_resp_valid, ec_hit, ec_resp_slot, ec_busy,
        ec_hit_count, ec_miss_count, ec_demand_stall_cycles, ec_pf_issued, ec_pf_hit,
        kv_row_valid, kv_row_out, kv_busy, kv_append_count, kv_resident_lo, kv_overflowed,
        ec_dropped, xbar_req_count, xbar_resp_count, xbar_resp_valid, xbar_resp_data,
        loader_busy, loader_done_count, loader_beat_count, loader_w_q, loader_in_valid
    };

    // ---------------- LFSR advance (feedback from the DUT output fold) ----------
    //   Registered -> the r -> rr -> DUT -> out_fold -> r loop has NO comb cycle.
    always @(posedge core_clk) begin
        if (core_rst) r <= 256'd1;                       // nonzero seed
        else          r <= {r[254:0], r[255]^r[249]^r[242]^r[59]^out_fold};
    end

    reg crc_q;
    always @(posedge core_clk) begin
        if (core_rst) crc_q <= 1'b0;
        else          crc_q <= crc_q ^ out_fold ^ r[0];
    end
    assign crc_out = crc_q;

    // ---------------- the DUT: all response inputs driven from the LFSR ---------
    glm_q4k_system_cdc #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .ACT_HW(ACT_HW), .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT),
        .KV_CTX(KV_CTX), .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH),
        .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W),
        .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD),
        .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN),
        .REQ_AW(REQ_AW), .TOK_AW(TOK_AW)
    ) dut (
        .host_clk(host_clk), .host_rst(host_rst),
        .core_clk(core_clk), .core_rst(core_rst),
        // control / prompt (driven from the LFSR to keep the control path live)
        .start(start),
        .prompt_tok(rr[TOKW-1:0]), .start_pos(rr[POSW-1:0]), .s_len(rr[IDXW:0]),
        .busy(busy), .done(done), .next_tok(next_tok), .tok_valid(tok_valid),
        .logits(logits),
        // GDDR6 hot-weight stubs
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(rr[15:0]),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(rr[15:0]),
        // attention-weight dequant response
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_q(rr[PE_N*4-1:0]),
        .aw_d(rr[16*PE_N*A_NSB-1:0]), .aw_dmin(rr[16*PE_N*A_NSB-1:0]),
        .aw_scales(rr[96*PE_N*A_NSB-1:0]),
        // router-weight dequant response
        .rw_req(rw_req), .rw_k(rw_k),
        .rw_q(rr[4*N_EXPERT-1:0]),
        .rw_d(rr[16*N_EXPERT*R_NSB-1:0]), .rw_dmin(rr[16*N_EXPERT*R_NSB-1:0]),
        .rw_scales(rr[96*N_EXPERT*R_NSB-1:0]),
        // FFN-weight dequant response (gate + up)
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_q(rr[4*TN-1:0]), .fw_q_up(rr[4*TN-1:0]),
        .fw_d_g(rr[16*TN*FF_NSB_D-1:0]), .fw_dmin_g(rr[16*TN*FF_NSB_D-1:0]),
        .fw_scales_g(rr[96*TN*FF_NSB_D-1:0]),
        .fw_d_u(rr[16*TN*FF_NSB_D-1:0]), .fw_dmin_u(rr[16*TN*FF_NSB_D-1:0]),
        .fw_scales_u(rr[96*TN*FF_NSB_D-1:0]),
        // final-norm gamma stub
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(rr[15:0]),
        // lm-head column stub
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(rr[LM_TN*16-1:0]),
        // KV compressed-row read
        .kc_ckv(rr[KV_LORA*16-1:0]), .kc_krope(rr[ROPE*16-1:0]),
        .kc_req(kc_req), .kc_idx(kc_idx),
        .kv_row_sel(kv_row_sel), .kv_row_in(rr[ROW_BITS-1:0]),
        // flash (cold expert / KV) response
        .flash_req(flash_req), .flash_is_expert(flash_is_expert),
        .flash_expert_id(flash_expert_id), .flash_row_idx(flash_row_idx),
        .flash_done(rr[0]), .flash_row(rr[ROW_BITS-1:0]),
        // prefetch stream-in
        .pf_valid(rr[1]), .pf_expert_id(rr[EIDXW-1:0]),
        // DDR5 crossbar memory bus
        .mem_req_valid(mem_req_valid), .mem_req_ready(rr[DDR_NCH-1:0]),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(rr[DDR_NCH-1:0]), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(rr[DDR_NCH*DDR_DATA_W-1:0]), .mem_resp_tag(rr[DDR_NCH*DDR_TAG_W-1:0]),
        // weight-loader DMA
        .wl_mem_en(wl_mem_en), .wl_mem_addr(wl_mem_addr), .wl_mem_data(rr[WL_DATA_W-1:0]),
        // observability outputs (all folded)
        .argmax_o(argmax_o), .h_state(h_state), .mdl_busy(mdl_busy),
        .ec_resp_valid(ec_resp_valid), .ec_hit(ec_hit), .ec_resp_slot(ec_resp_slot),
        .ec_busy(ec_busy), .ec_hit_count(ec_hit_count), .ec_miss_count(ec_miss_count),
        .ec_demand_stall_cycles(ec_demand_stall_cycles), .ec_pf_issued(ec_pf_issued),
        .ec_pf_hit(ec_pf_hit),
        .kv_row_valid(kv_row_valid), .kv_row_out(kv_row_out), .kv_busy(kv_busy),
        .kv_append_count(kv_append_count), .kv_resident_lo(kv_resident_lo),
        .kv_overflowed(kv_overflowed),
        .ec_dropped(ec_dropped), .xbar_req_count(xbar_req_count),
        .xbar_resp_count(xbar_resp_count), .xbar_resp_valid(xbar_resp_valid),
        .xbar_resp_data(xbar_resp_data),
        .loader_busy(loader_busy), .loader_done_count(loader_done_count),
        .loader_beat_count(loader_beat_count), .loader_w_q(loader_w_q),
        .loader_in_valid(loader_in_valid)
    );
endmodule

`default_nettype wire
