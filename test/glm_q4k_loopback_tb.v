`timescale 1ns/1ps
//============================================================================
// glm_q4k_loopback_tb.v -- C8 LOOPBACK=1 BIT-EXACTNESS proof for glm_q4k_system.
//   (the proof TB referenced by src/glm_q4k_system.v:236 "See the new local
//    proof TB").
//----------------------------------------------------------------------------
// WHAT IT PROVES
//   The perf TB (test/glm_q4k_system_perf_tb.v) already proves
//       glm_q4k_system(LOOPBACK=0) committed next_tok == standalone glm_model_q4k
//   fed by the SAME deterministic weight functions (f_awq / f_rwq / f_fwq / ...).
//   This TB proves the SAME equality with LOOPBACK=1 -- i.e. when the die's
//   attention-weight Q4_K CODE lanes are physically routed OUT of the die as a
//   banked ddr5_xbar read (TAG_LBAW) and back IN through the fabric, instead of
//   being served same-cycle by the stub.  If the loopback round-trip is faithful,
//   the committed token stream is BIT-EXACT to the non-loopback (== reference)
//   stream: clock-gating a synchronous die is transparent, and the fed-back lanes
//   equal the stub lanes for the same key.
//
// HOW THE LOOPBACK RESPONDER WORKS (the ONLY change vs the perf TB harness)
//   At LOOPBACK=1 each attention-weight beat, the die encodes its current pull
//   key {db_layer,aw_sel,aw_grp,aw_k} into a DDR read address `cur_addr`:
//       cur_addr[3:0]                = aw_sel
//       cur_addr[AWA_K_LO  +: A_KCW]  = aw_k          (offsets packed end-to-end
//       cur_addr[AWA_G_LO  +: A_GRPW] = aw_grp         from the widths -- must
//       cur_addr[AWA_LY_LO +: LAYW]   = db_layer       mirror g_lb's localparams)
//       cur_addr[24 +: 8]             = 8'hA5   (TAG_LBAW address marker)
//   ddr5_xbar carries the FULL address to the per-channel memory unchanged and
//   returns the response paired with its tag.  This TB's DDR5 memory model, on
//   any outstanding read whose addr[24 +: 8]==8'hA5, decodes {layer,sel,grp,k}
//   and packs into the response data's low PE_N*4 bits the Q4_K code lanes
//       lane t = f_awq(layer, sel, grp*PE_N + t, k)
//   -- EXACTLY the bytes the stub `aw_q` presents for that key at LOOPBACK=0
//   (perf TB line ~407) and the reference `r_aw_q` presents (perf TB line ~523).
//   Every OTHER read keeps the perf TB's banked-read behaviour (gen_beat).
//   Only loopback reads carry the 8'hA5 marker in addr[31:24] (TAG_LOAD/SLOT/
//   HOT/EFILL all leave the top address bits zero), so the decode is unambiguous.
//
// SOUNDNESS (-DLBINJECT): the loopback-returned lane 0 is XOR-corrupted (bit 0
//   flipped on EVERY loopback beat).  If the die actually CONSUMES the fed-back
//   lanes, the corrupted attention weights move the committed argmax -> the
//   stream DIVERGES from the reference -> this TB FAILS (the pass banner is only
//   printed when errors==0).  The clean build MUST pass; the injected build MUST
//   NOT print the banner -- that is what proves the round-trip is load-bearing.
//
// STYLE: sync active-high reset; self-checking; on success prints exactly
//   "ALL %0d TESTS PASSED ...".  Mirrors the perf TB idioms.
//============================================================================
module glm_q4k_loopback_tb;

    // ---- small-but-faithful slice (the perf TB geometry, PE_N=2) ----
    parameter integer FLASH_LAT_CFG    = 8;
    parameter integer DDR_NCH_CFG      = 4;
    parameter integer CACHE_SLOTS_CFG  = 4;
    parameter integer N_EXPERT_CFG     = 4;
    parameter integer L_CFG            = 4;
    // EXPERT_STALL / RESIDENT held OFF: at LOOPBACK=1 the §9 loopback branch owns
    // die_clk (the EXPERT_STALL clock-gate lives only in the LOOPBACK==0 branch),
    // and RESIDENT=0 keeps expert refills on the single Flash channel.  Neither
    // is needed to exercise the loopback path.
    parameter integer EXPERT_STALL_CFG = 0;
    parameter integer RESIDENT_CFG     = 0;
    parameter integer DSA_REAL_IDX_CFG = 0;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ================= slice params (mirror the perf TB) =================
    parameter integer MODEL_DIM = 16;
    localparam integer L          = L_CFG;
    parameter integer N_DENSE   = 2;
    parameter integer VOCAB     = 16;
    parameter integer H_HEADS   = 2;
    parameter integer NOPE      = 4;
    parameter integer ROPE      = 4;
    parameter integer V_DIM     = 4;
    parameter integer Q_LORA    = 8;
    parameter integer KV_LORA   = 8;
    parameter integer S_MAX     = 4;
    parameter integer TOPK_ATTN = 4;
    localparam integer THETA      = 8000000;
    parameter  integer PE_N        = 2;
    localparam integer POSW       = 20;
    localparam integer N_EXPERT   = N_EXPERT_CFG;
    parameter integer TOPK      = 2;
    parameter integer INTER_MOE = 16;
    parameter integer INTER_DENSE= 32;
    localparam [31:0]  RSCALE     = 32'h40200000;
    parameter integer TN          = 4;
    localparam integer BLK        = 128;
    parameter  integer LM_TN       = 4;
    localparam integer CACHE_SLOTS = CACHE_SLOTS_CFG;
    localparam integer FLASH_LAT   = FLASH_LAT_CFG;
    localparam integer KV_CTX      = 1024;
    localparam integer KV_RESIDENT = 16;
    localparam integer EFIFO_DEPTH = 16;
    localparam integer DDR_NCH     = DDR_NCH_CFG;
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
    // aw loopback key offsets -- MUST mirror glm_q4k_system g_lb (packed end-to-end)
    localparam integer AWA_K_LO  = 4;
    localparam integer AWA_G_LO  = AWA_K_LO + A_KCW;
    localparam integer AWA_LY_LO = AWA_G_LO + A_GRPW;
    localparam integer ROW_BITS = (KV_LORA+ROPE)*16;
    localparam integer KVPOSW   = (KV_CTX<=1)?1:$clog2(KV_CTX);
    localparam integer CSLOTW   = (CACHE_SLOTS<=1)?1:$clog2(CACHE_SLOTS);
    localparam integer WL_PE_N  = PE_N;
    localparam integer WL_DATA_W= 256;
    localparam integer CH_SEL_W = (DDR_NCH<=1)?1:$clog2(DDR_NCH);

    // ================= deterministic stimulus generators ========================
    function automatic integer f_h; input integer seed; begin
        f_h = (seed*2654435761)^(seed<<13)^(seed*40503);
    end endfunction
    function automatic [15:0] gen_bf16; input integer seed;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h = f_h(seed);
        s = h[3];
        e = 8'd124 + {6'b0,h[5:4]};
        m = h[12:6];
        gen_bf16 = {s,e,m};
    end endfunction
    function automatic [15:0] gen_fp16; input integer seed;
        reg [4:0] e; reg [9:0] m; integer h; begin
        h = f_h(seed);
        e = 5'd12 + {4'b0,h[4]};
        m = h[14:5];
        gen_fp16 = {1'b0,e,m};
    end endfunction
    function automatic [3:0] gen_q4; input integer seed; integer h; begin
        h = f_h(seed);
        gen_q4 = h[11:8];
    end endfunction
    function automatic [31:0] gen_s32; input integer seed; begin
        gen_s32 = f_h(seed*97 + 5);
    end endfunction

    // ---- per-family responder value functions (shared DUT/ref) ----
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

    integer errors;

    //========================================================================
    // DUT (the Q4_K system) host I/O -- LOOPBACK=1
    //========================================================================
    reg                       start;
    reg  [TOKW-1:0]           prompt_tok;
    reg  [POSW-1:0]           start_pos;
    reg  [IDXW:0]             s_len;
    wire                      busy, done;
    wire [TOKW-1:0]           next_tok;
    wire                      tok_valid;
    wire [VOCAB*16-1:0]       logits;
    wire                      em_req;  wire [TOKW-1:0] em_tok;  wire [DIMW-1:0] em_idx;
    reg  [15:0]               em_val;
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
    wire                      kc_req;  wire [IDXW-1:0] kc_idx;  reg [KV_LORA*16-1:0] kc_ckv;  reg [ROPE*16-1:0] kc_krope;
    wire [KVPOSW-1:0]         kv_row_sel;  reg [ROW_BITS-1:0] kv_row_in;
    wire                      flash_req, flash_is_expert;
    wire [EIDXW-1:0]          flash_expert_id;  wire [KVPOSW-1:0] flash_row_idx;
    reg                       flash_done;  reg [ROW_BITS-1:0] flash_row;
    wire [TOKW-1:0]           argmax_o;  wire [MODEL_DIM*16-1:0] h_state;  wire mdl_busy;
    wire                      ec_resp_valid, ec_hit;  wire [CSLOTW-1:0] ec_resp_slot;  wire ec_busy;
    wire [31:0]               ec_hit_count, ec_miss_count, ec_demand_stall_cycles, ec_pf_issued, ec_pf_hit;
    wire                      kv_row_valid;  wire [ROW_BITS-1:0] kv_row_out;  wire kv_busy;
    wire [KVPOSW-1:0]         kv_append_count, kv_resident_lo;  wire kv_overflowed;
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
        .LOOPBACK(1),                                   // <-- the proof knob
        .EXPERT_STALL(EXPERT_STALL_CFG), .RESIDENT(RESIDENT_CFG),
        .DSA_REAL_IDX(DSA_REAL_IDX_CFG),
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
        .decomp_tbl_we(1'b0), .decomp_tbl_sel(1'b0),
        .decomp_tbl_addr(9'h000), .decomp_tbl_wdata(10'h000),
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

    // ================= system weight/KV responders (stub ports) =====
    //   NOTE: at LOOPBACK=1 the die's attention CODE lanes come from the xbar
    //   round-trip, NOT from this `aw_q` input (die_aw_q = lb_col_q).  aw_q is
    //   still driven (harmlessly) so the port matches the perf TB exactly; the
    //   d/dmin/scales stubs ARE still consumed same-cycle by the die.
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

    // ================= INDEPENDENT REFERENCE: standalone glm_model_q4k ==========
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

    glm_model_q4k #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .DSA_REAL_IDX(DSA_REAL_IDX_CFG)
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
        .h_state(r_h_state)
    );

    integer rt, rft, rre, rsb, rcd;
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
    always @* begin
        for (rcd=0;rcd<KV_LORA;rcd=rcd+1) r_kc_ckv  [16*rcd+:16] = gen_bf16(r_db_layer*513 + r_kc_idx*67 + rcd*7 + 8011);
        for (rcd=0;rcd<ROPE;rcd=rcd+1)    r_kc_krope[16*rcd+:16] = gen_bf16(r_db_layer*771 + r_kc_idx*91 + rcd*5 + 8101);
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
    wire _unused = &{1'b0, busy, em_req, aw_req, fw_req, rw_req, gn_req, fn_req,
                     lw_req, idx_fresh, idx_win, mdl_busy, ec_resp_valid, ec_hit,
                     ec_resp_slot, ec_busy, ec_pf_issued, ec_pf_hit, kv_row_out,
                     kv_busy, kv_resident_lo, flash_expert_id, h_state, gn_which,
                     done, kv_overflowed, r_busy, r_em_req, r_aw_req, r_fw_req,
                     r_rw_req, r_gn_req, r_fn_req, r_lw_req, r_idx_fresh,
                     r_idx_win, r_logits, r_h_state, r_gn_which, r_kc_seq,
                     loader_busy, mem_req_tag, ec_hit_count, ec_miss_count,
                     ec_demand_stall_cycles, ec_dropped, kv_append_count,
                     kv_row_valid, aw_q, xbar_resp_data, loader_w_q,
                     loader_in_valid, loader_done_count, loader_beat_count,
                     flash_is_expert, kc_req};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= KV latent-ROW stub =================
    integer rr;
    always @* begin
        kv_row_in = {ROW_BITS{1'b0}};
        for (rr=0;rr<(KV_LORA+ROPE);rr=rr+1)
            kv_row_in[16*rr+:16] = gen_bf16(kv_row_sel*131 + rr*7 + 3);
    end

    // ================= FLASH PHY STUB (FLASH_LAT-cycle fetch) =================
    reg [31:0] fl_timer; reg fl_active; reg prev_req;
    always @(posedge clk) begin
        if (rst) begin
            fl_timer <= 32'd0; fl_active <= 1'b0; flash_done <= 1'b0; prev_req <= 1'b0;
        end else begin
            flash_done <= 1'b0;
            if (!fl_active) begin
                if (flash_req && !prev_req) begin
                    fl_active <= 1'b1; fl_timer <= FLASH_LAT[31:0];
                end
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
    always @(posedge clk) begin
        if (wl_mem_en) wl_mem_data <= STAGE[wl_mem_addr[10:0]];
        else           wl_mem_data <= {WL_DATA_W{1'b0}};
    end

    // ================= DDR5 PER-CHANNEL MEMORY MODEL + C8 LOOPBACK RESPONDER ==
    //   Single in-flight table; each read completes DDR_ROW_LAT cycles later on
    //   its banked channel, held until mem_resp_ready accepts.  For NORMAL reads
    //   the returned beat is gen_beat(tag,addr) (perf TB behaviour).  For a C8
    //   LOOPBACK read -- addr[24 +: 8]==8'hA5 -- the beat's low PE_N*4 bits are
    //   the packed f_awq lanes decoded from the address (the die's aw code lanes).
    localparam [7:0] TAG_LBAW_MARK = 8'hA5;     // addr[24 +: 8] on a loopback read
    localparam integer NINF = 64;
    reg                  infv  [0:NINF-1];
    reg [DDR_TAG_W-1:0]  inftg [0:NINF-1];
    reg [DDR_ADDR_W-1:0] infad [0:NINF-1];
    reg [15:0]           inftm [0:NINF-1];   // remaining latency (0 => ready)
    integer ii, cc;
    integer lb_req_seen, lb_resp_seen;        // loopback-path activity counters

    assign mem_req_ready = {DDR_NCH{1'b1}};

    reg [31:0] presIdx [0:DDR_NCH-1];
    reg        presV   [0:DDR_NCH-1];

    // NORMAL banked-read beat (perf TB behaviour, X-clean deterministic).
    function automatic [DDR_DATA_W-1:0] gen_beat;
        input [DDR_TAG_W-1:0] tg; input [DDR_ADDR_W-1:0] ad; integer ln; begin
        gen_beat = {DDR_DATA_W{1'b0}};
        for (ln=0; ln<DDR_DATA_W/16; ln=ln+1)
            gen_beat[16*ln+:16] = gen_bf16(({24'd0,tg}*7 + ad*3 + ln*5 + 1));
        end
    endfunction

    // C8 LOOPBACK beat: decode {layer,sel,grp,k} from the marked address and
    // pack lane t = f_awq(layer,sel,grp*PE_N+t,k) into the low PE_N*4 bits.  The
    // high bits are kept X-clean (gen_beat base) since the xbar-resp X-monitor
    // checks the WHOLE beat; the die only consumes bits [PE_N*4-1:0].
    function automatic [DDR_DATA_W-1:0] gen_lb_beat;
        input [DDR_TAG_W-1:0] tg; input [DDR_ADDR_W-1:0] ad;
        integer ln; integer ly, sel, grp, kk;
        reg [DDR_DATA_W-1:0] b; begin
        b   = gen_beat(tg, ad);              // X-clean base for the unused high bits
        sel = ad[3:0];
        kk  = ad[AWA_K_LO  +: A_KCW];
        grp = ad[AWA_G_LO  +: A_GRPW];
        ly  = ad[AWA_LY_LO +: LAYW];
        for (ln=0; ln<PE_N; ln=ln+1)
            b[4*ln +: 4] = f_awq(ly, sel, grp*PE_N + ln, kk);
`ifdef LBINJECT
        // SOUNDNESS INJECTION: corrupt the fed-back lane 0 (flip bit 0) on EVERY
        // loopback beat.  If the die actually consumes these lanes, the committed
        // stream must DIVERGE from the reference -> this TB must FAIL.
        b[0] = ~b[0];
`endif
        gen_lb_beat = b;
        end
    endfunction

    // ---- combinational response presentation (oldest-ready per channel) ----
    always @* begin
        mem_resp_valid = {DDR_NCH{1'b0}};
        mem_resp_data  = {(DDR_NCH*DDR_DATA_W){1'b0}};
        mem_resp_tag   = {(DDR_NCH*DDR_TAG_W){1'b0}};
        for (cc=0; cc<DDR_NCH; cc=cc+1) begin
            presV[cc]   = 1'b0;
            presIdx[cc] = 32'd0;
        end
        for (cc=0; cc<DDR_NCH; cc=cc+1) begin
            for (ii=NINF-1; ii>=0; ii=ii-1) begin
                if (infv[ii] && (inftm[ii]==16'd0) &&
                    ((DDR_NCH==1) ? (cc==0)
                                  : (infad[ii][CH_SEL_W-1:0] == cc[CH_SEL_W-1:0]))) begin
                    presV[cc]   = 1'b1;
                    presIdx[cc] = ii;
                end
            end
            if (presV[cc]) begin
                mem_resp_valid[cc] = 1'b1;
                // C8 LOOPBACK decode when the address carries the TAG_LBAW marker;
                // every other read keeps the normal banked beat.
                if (infad[presIdx[cc]][24 +: 8] == TAG_LBAW_MARK)
                    mem_resp_data[cc*DDR_DATA_W +: DDR_DATA_W] =
                        gen_lb_beat(inftg[presIdx[cc]], infad[presIdx[cc]]);
                else
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
                infv[ii]  <= 1'b0; inftg[ii] <= {DDR_TAG_W{1'b0}};
                infad[ii] <= {DDR_ADDR_W{1'b0}}; inftm[ii] <= 16'd0;
            end
            lb_req_seen <= 0; lb_resp_seen <= 0;
        end else begin
            for (ii=0; ii<NINF; ii=ii+1)
                if (infv[ii] && (inftm[ii]!=16'd0)) inftm[ii] <= inftm[ii]-16'd1;
            for (cc=0; cc<DDR_NCH; cc=cc+1)
                if (presV[cc] && mem_resp_ready[cc]) begin
                    infv[presIdx[cc]] <= 1'b0;
                    if (infad[presIdx[cc]][24 +: 8] == TAG_LBAW_MARK)
                        lb_resp_seen <= lb_resp_seen + 1;   // a loopback beat drained
                end
            got_free = 1'b0; freeslot = 0;
            for (ii=NINF-1; ii>=0; ii=ii-1)
                if (!infv[ii]) begin got_free = 1'b1; freeslot = ii; end
            for (cc=0; cc<DDR_NCH; cc=cc+1) begin
                if (mem_req_valid[cc] && mem_req_ready[cc] && got_free) begin
                    infv[freeslot]  <= 1'b1;
                    inftg[freeslot] <= mem_req_tag[cc*DDR_TAG_W +: DDR_TAG_W];
                    infad[freeslot] <= mem_req_addr[cc*DDR_ADDR_W +: DDR_ADDR_W];
                    inftm[freeslot] <= DDR_ROW_LAT[15:0];
                    if (mem_req_addr[cc*DDR_ADDR_W+24 +: 8] == TAG_LBAW_MARK)
                        lb_req_seen <= lb_req_seen + 1;     // a loopback read captured
                end
            end
        end
    end

    // ================= continuous X monitor on the xbar resp beat ==========
    integer xmon_err;
    always @(posedge clk) if (!rst) begin
        if (xbar_resp_valid)
            if (^xbar_resp_data === 1'bx) begin
                $display("FAIL: xbar_resp_data X while valid @%0t", $time); xmon_err=xmon_err+1; end
        if (|mem_req_valid && (^mem_req_addr === 1'bx)) begin
            $display("FAIL: mem_req_addr X while valid @%0t", $time); xmon_err=xmon_err+1; end
    end

    // ================= per-token bit-exactness check =================
    integer test_count;

    task settle; input integer n; integer c; begin
        for (c=0;c<n;c=c+1) @(negedge clk);
    end endtask

    task run_token; input [TOKW-1:0] tk; input [POSW-1:0] ps; input integer SL;
        input [256*8-1:0] label; integer b; begin
        prompt_tok = tk; start_pos = ps; s_len = SL[IDXW:0];
        @(negedge clk); start = 1'b1; r_start = 1'b1;
        @(negedge clk); start = 1'b0; r_start = 1'b0;
        wait (tok_valid === 1'b1);
        @(negedge clk);
        wait (r_done_seen === 1'b1);
        settle(400);   // drain FIFO/cache/Flash + xbar in-flight reads

        test_count = test_count + 1;

        // (a) BINDING: LOOPBACK=1 system committed token == standalone reference
        if (next_tok !== r_tok_lat) begin
            $display("FAIL[%0s]: BINDING next_tok %0d != standalone ref %0d (LOOPBACK=1 diverged)",
                     label, next_tok, r_tok_lat);
            errors=errors+1; end
        // (a') X-cleanliness + internal-argmax consistency
        for (b=0;b<TOKW;b=b+1) if (next_tok[b]===1'bx || next_tok[b]===1'bz) begin
            $display("FAIL[%0s]: next_tok bit %0d X/Z", label, b); errors=errors+1; end
        for (b=0;b<VOCAB*16;b=b+1) if (logits[b]===1'bx || logits[b]===1'bz) begin
            $display("FAIL[%0s]: logits bit %0d X/Z", label, b); errors=errors+1; b=VOCAB*16; end
        if (next_tok !== argmax_o) begin
            $display("FAIL[%0s]: next_tok %0d != system internal argmax %0d", label, next_tok, argmax_o);
            errors=errors+1; end
        // (b) the loopback fabric path was actually exercised for this token
        if (xbar_req_count == 32'd0 || xbar_resp_count == 32'd0) begin
            $display("FAIL[%0s]: ddr5_xbar carried no banked read (req=%0d resp=%0d)",
                     label, xbar_req_count, xbar_resp_count); errors=errors+1; end

        if (xmon_err != 0) begin
            $display("FAIL[%0s]: %0d xbar/req monitor violations", label, xmon_err);
            errors=errors+1; end

        $display("PASS[%0s] tok=%0d(==ref %0d) | xbar req/resp=%0d/%0d | loopback reads/resps=%0d/%0d",
                 label, next_tok, r_tok_lat, xbar_req_count, xbar_resp_count,
                 lb_req_seen, lb_resp_seen);
    end endtask

    initial begin
        #400000000;
        $display("FAIL: global timeout"); $fatal;
    end

    initial begin
        errors=0; test_count=0; xmon_err=0;
        lb_req_seen=0; lb_resp_seen=0;
        rst=1'b1; start=1'b0; r_start=1'b0;
        prompt_tok={TOKW{1'b0}}; start_pos={POSW{1'b0}};
        s_len={(IDXW+1){1'b0}};
        wl_mem_data={WL_DATA_W{1'b0}};
        repeat(4) @(negedge clk);
        rst=1'b0;
        @(negedge clk);

        // ---- small decode sequence: cold token 0, then feed next_tok back ----
        //   (pos == s_len, the real-decode relation; s_len grows 1..S_MAX)
        run_token(4'd7,     20'd1, 1, "tok0 cold s1");
        run_token(next_tok, 20'd2, 2, "tok1 s2");
        run_token(next_tok, 20'd3, 3, "tok2 s3");
        run_token(next_tok, 20'd4, 4, "tok3 s4");

        // ---- global soundness gate: the loopback path MUST have round-tripped ----
        //   (proves the bindings above actually flowed weight bytes through the
        //   real ddr5_xbar, not a bypass).
        test_count = test_count + 1;
        if (lb_req_seen == 0 || lb_resp_seen == 0) begin
            $display("FAIL: C8 loopback path never exercised (lb_req=%0d lb_resp=%0d) -- the proof is vacuous",
                     lb_req_seen, lb_resp_seen);
            errors=errors+1;
        end else
            $display("PASS[loopback-exercised] lb_reads=%0d lb_resps=%0d routed through ddr5_xbar",
                     lb_req_seen, lb_resp_seen);

        if (errors!=0) begin
            $display("FAILED: %0d error(s) across %0d tests (LOOPBACK=1 not bit-exact to reference)",
                     errors, test_count);
            $fatal;
        end
        $display("ALL %0d TESTS PASSED  (glm_q4k_system LOOPBACK=1: die aw code lanes routed OUT and back IN through ddr5_xbar, committed stream BIT-EXACT to standalone glm_model_q4k reference, 4-token decode)",
                 test_count);
        $finish;
    end
endmodule
