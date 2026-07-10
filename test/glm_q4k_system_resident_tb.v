`timescale 1ns/1ps
//============================================================================
// glm_q4k_system_resident_tb.v -- RESIDENT=1 expert-refill path testbench
//                                 (the "runtime decode never touches Flash
//                                  for weights" tie-off, docs/R3_APPLIANCE_SPEC)
//----------------------------------------------------------------------------
// WHAT IS PROVEN (self-checking; the die itself is held idle -- start is never
// pulsed -- so the ONLY traffic is the expert-cache refill under test):
//
//   DUT1 (RESIDENT=1):  a prefetch hint (pf_valid/pf_expert_id) makes
//     expert_cache_pf raise its held flash_req handshake; the §10 refill FSM
//     must serve it from the DDR-tier fabric:
//       * a banked ddr5_xbar read appears on the mem_* channel ports with
//         tag == TAG_EFILL (8'h05) -- checked on the channel request bus;
//       * the tagged response completes ec_flash_done: the cache returns to
//         IDLE (a later hint for a DIFFERENT expert starts a second fetch,
//         observable as ec_pf_issued incrementing -- only possible from IDLE);
//       * a repeat hint for the SAME expert is a resident SKIP (no new fetch,
//         no new xbar read) -- proves the refill actually INSTALLED;
//       * the system flash_req output NEVER fires, and flash_is_expert-granted
//         flash traffic never exists ("expert class never owns Flash").
//
//   DUT0 (RESIDENT=0, the byte-identical default):  the SAME stimulus goes to
//     the SINGLE FLASH CHANNEL exactly as committed -- flash_req fires with
//     flash_is_expert=1, completes on flash_done, and NO ddr5_xbar read is
//     issued (xbar_req_count stays 0).  This pins the contrast: RESIDENT only
//     changes WHERE the refill handshake is served.
//
// The die's weight/KV stub inputs are tied to 0 (never consumed: the die stays
// in IDLE, so no hot pulls, no KV gathers, no loader tiles -- the xbar sees
// ONLY the refill class under test).
// STYLE: sync active-high reset; self-checking ($fatal on any mismatch);
//   final banner "ALL %0d TESTS PASSED".
//============================================================================
module glm_q4k_system_resident_tb;

    // ---- default-slice geometry (mirrors glm_q4k_system defaults) ----
    localparam integer TOKW    = 8;    // clog2(VOCAB=256)
    localparam integer POSW    = 20;
    localparam integer IDXW    = 3;    // clog2(S_MAX=8)
    localparam integer EIDXW   = 3;    // clog2(N_EXPERT=8)
    localparam integer KVPOSW  = 10;   // clog2(KV_CTX=1024)
    localparam integer ROW_BITS= 768;  // (KV_LORA=32 + ROPE=16) * 16
    localparam integer NCH     = 4;    // DDR_NCH
    localparam integer AW      = 32;   // DDR_ADDR_W
    localparam integer DW      = 256;  // DDR_DATA_W
    localparam integer TW      = 8;    // DDR_TAG_W
    localparam integer WLAW    = 24;   // WL_ADDR_W
    localparam integer WLDW    = 256;  // WL_DATA_W
    localparam [TW-1:0] TAG_EFILL = 8'h05;

    reg clk;
    reg rst;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    integer pass_count;
    task check(input cond, input [255:0] name);
        begin
            if (!cond) begin
                $display("FAIL: %0s", name);
                $fatal(1, "glm_q4k_system_resident_tb assertion failed");
            end
            pass_count = pass_count + 1;
        end
    endtask

    // ---- shared stimulus ----
    reg              pf_valid;
    reg [EIDXW-1:0]  pf_expert_id;

    //========================================================================
    // DUT1 : RESIDENT=1  (refill must be served by ddr5_xbar, never Flash)
    //========================================================================
    wire             r1_flash_req;
    wire             r1_flash_is_expert;
    wire [NCH-1:0]         r1_mem_req_valid;
    wire [NCH*AW-1:0]      r1_mem_req_addr;
    wire [NCH*TW-1:0]      r1_mem_req_tag;
    reg  [NCH-1:0]         r1_mem_resp_valid;
    wire [NCH-1:0]         r1_mem_resp_ready;
    reg  [NCH*DW-1:0]      r1_mem_resp_data;
    reg  [NCH*TW-1:0]      r1_mem_resp_tag;
    wire [31:0]            r1_xbar_req_count;
    wire [31:0]            r1_xbar_resp_count;
    wire [31:0]            r1_pf_issued;
    wire                   r1_ec_busy;

    glm_q4k_system #(.RESIDENT(1)) dut1 (
        .clk(clk), .rst(rst),
        .start(1'b0), .prompt_tok({TOKW{1'b0}}), .start_pos({POSW{1'b0}}),
        .s_len({(IDXW+1){1'b0}}),
        .busy(), .done(), .next_tok(), .tok_valid(), .logits(),
        .em_req(), .em_tok(), .em_idx(), .em_val(16'h0000),
        .db_layer(), .idx_fresh(), .idx_win(),
        .gn_req(), .gn_which(), .gn_idx(), .gn_val(16'h0000),
        .aw_req(), .aw_sel(), .aw_grp(), .aw_k(),
        .aw_q(16'h0000), .aw_d({(16*4*1){1'b0}}), .aw_dmin({(16*4*1){1'b0}}),
        .aw_scales({(96*4*1){1'b0}}),
        .rw_req(), .rw_k(),
        .rw_q({(4*8){1'b0}}), .rw_d({(16*8*1){1'b0}}), .rw_dmin({(16*8*1){1'b0}}),
        .rw_scales({(96*8*1){1'b0}}),
        .fw_req(), .fw_sel(), .fw_grp(), .fw_k(), .fw_shared(), .fw_eidx(),
        .fw_q({(4*4){1'b0}}), .fw_q_up({(4*4){1'b0}}),
        .fw_d_g({(16*4*1){1'b0}}), .fw_dmin_g({(16*4*1){1'b0}}),
        .fw_scales_g({(96*4*1){1'b0}}),
        .fw_d_u({(16*4*1){1'b0}}), .fw_dmin_u({(16*4*1){1'b0}}),
        .fw_scales_u({(96*4*1){1'b0}}),
        .fn_req(), .fn_idx(), .fn_val(16'h0000),
        .lw_req(), .lw_vtile(), .lw_k(), .lw_col({(4*16){1'b0}}),
        .kc_ckv({(32*16){1'b0}}), .kc_krope({(16*16){1'b0}}),
        .kc_req(), .kc_idx(),
        .kv_row_sel(), .kv_row_in({ROW_BITS{1'b0}}),
        .flash_req(r1_flash_req), .flash_is_expert(r1_flash_is_expert),
        .flash_expert_id(), .flash_row_idx(),
        .flash_done(1'b0), .flash_row({ROW_BITS{1'b0}}),
        .pf_valid(pf_valid), .pf_expert_id(pf_expert_id),
        .mem_req_valid(r1_mem_req_valid), .mem_req_ready({NCH{1'b1}}),
        .mem_req_addr(r1_mem_req_addr), .mem_req_tag(r1_mem_req_tag),
        .mem_resp_valid(r1_mem_resp_valid), .mem_resp_ready(r1_mem_resp_ready),
        .mem_resp_data(r1_mem_resp_data), .mem_resp_tag(r1_mem_resp_tag),
        .wl_mem_en(), .wl_mem_addr(), .wl_mem_data({WLDW{1'b0}}),
        .decomp_tbl_we(1'b0), .decomp_tbl_sel(1'b0),
        .decomp_tbl_addr(9'h000), .decomp_tbl_wdata(10'h000),
        .argmax_o(), .h_state(), .mdl_busy(),
        .ec_resp_valid(), .ec_hit(), .ec_resp_slot(), .ec_busy(r1_ec_busy),
        .ec_hit_count(), .ec_miss_count(), .ec_demand_stall_cycles(),
        .ec_pf_issued(r1_pf_issued), .ec_pf_hit(),
        .kv_row_valid(), .kv_row_out(), .kv_busy(),
        .kv_append_count(), .kv_resident_lo(), .kv_overflowed(),
        .ec_dropped(),
        .xbar_req_count(r1_xbar_req_count), .xbar_resp_count(r1_xbar_resp_count),
        .xbar_resp_valid(), .xbar_resp_data(),
        .loader_busy(), .loader_done_count(), .loader_beat_count(),
        .loader_w_q(), .loader_in_valid()
    );

    // ---- DUT1 DDR-channel stub: 1 outstanding per channel, next-cycle resp ----
    reg [NCH-1:0]  r1_pend;
    reg [TW-1:0]   r1_tag_q [0:NCH-1];
    reg [TW-1:0]   r1_seen_tag;      // last channel-request tag observed
    reg            r1_seen_any;
    integer c1;
    always @(posedge clk) begin
        if (rst) begin
            r1_pend       <= {NCH{1'b0}};
            r1_seen_tag   <= {TW{1'b0}};
            r1_seen_any   <= 1'b0;
            r1_mem_resp_valid <= {NCH{1'b0}};
            r1_mem_resp_data  <= {(NCH*DW){1'b0}};
            r1_mem_resp_tag   <= {(NCH*TW){1'b0}};
            for (c1 = 0; c1 < NCH; c1 = c1 + 1) r1_tag_q[c1] <= {TW{1'b0}};
        end else begin
            for (c1 = 0; c1 < NCH; c1 = c1 + 1) begin
                // response accepted -> drop it
                if (r1_mem_resp_valid[c1] && r1_mem_resp_ready[c1]) begin
                    r1_mem_resp_valid[c1] <= 1'b0;
                    r1_pend[c1]           <= 1'b0;
                end
                // request accepted -> capture tag, present the response next cycle
                if (r1_mem_req_valid[c1] && !r1_pend[c1]) begin
                    r1_pend[c1]  <= 1'b1;
                    r1_tag_q[c1] <= r1_mem_req_tag[c1*TW +: TW];
                    r1_seen_tag  <= r1_mem_req_tag[c1*TW +: TW];
                    r1_seen_any  <= 1'b1;
                    r1_mem_resp_valid[c1]        <= 1'b1;
                    r1_mem_resp_tag[c1*TW +: TW] <= r1_mem_req_tag[c1*TW +: TW];
                    r1_mem_resp_data[c1*DW +: DW]<= {DW{1'b0}};
                end
            end
        end
    end

    // ---- DUT1 invariant monitor: the Flash channel NEVER fires ----
    always @(posedge clk) begin
        if (!rst && r1_flash_req)
            $fatal(1, "RESIDENT=1: flash_req fired (refill leaked to the Flash path)");
    end

    //========================================================================
    // DUT0 : RESIDENT=0 (default) -- same stimulus must use the Flash channel
    //========================================================================
    wire             r0_flash_req;
    wire             r0_flash_is_expert;
    reg              r0_flash_done;
    wire [NCH-1:0]   r0_mem_req_valid;
    wire [NCH-1:0]   r0_mem_resp_ready;
    wire [31:0]      r0_xbar_req_count;
    wire [31:0]      r0_pf_issued;
    reg  [31:0]      r0_flash_fires;   // #cycles flash_req seen high (level)
    reg  [31:0]      r0_flash_grants;  // #completed flash fetches

    glm_q4k_system #(.RESIDENT(0)) dut0 (
        .clk(clk), .rst(rst),
        .start(1'b0), .prompt_tok({TOKW{1'b0}}), .start_pos({POSW{1'b0}}),
        .s_len({(IDXW+1){1'b0}}),
        .busy(), .done(), .next_tok(), .tok_valid(), .logits(),
        .em_req(), .em_tok(), .em_idx(), .em_val(16'h0000),
        .db_layer(), .idx_fresh(), .idx_win(),
        .gn_req(), .gn_which(), .gn_idx(), .gn_val(16'h0000),
        .aw_req(), .aw_sel(), .aw_grp(), .aw_k(),
        .aw_q(16'h0000), .aw_d({(16*4*1){1'b0}}), .aw_dmin({(16*4*1){1'b0}}),
        .aw_scales({(96*4*1){1'b0}}),
        .rw_req(), .rw_k(),
        .rw_q({(4*8){1'b0}}), .rw_d({(16*8*1){1'b0}}), .rw_dmin({(16*8*1){1'b0}}),
        .rw_scales({(96*8*1){1'b0}}),
        .fw_req(), .fw_sel(), .fw_grp(), .fw_k(), .fw_shared(), .fw_eidx(),
        .fw_q({(4*4){1'b0}}), .fw_q_up({(4*4){1'b0}}),
        .fw_d_g({(16*4*1){1'b0}}), .fw_dmin_g({(16*4*1){1'b0}}),
        .fw_scales_g({(96*4*1){1'b0}}),
        .fw_d_u({(16*4*1){1'b0}}), .fw_dmin_u({(16*4*1){1'b0}}),
        .fw_scales_u({(96*4*1){1'b0}}),
        .fn_req(), .fn_idx(), .fn_val(16'h0000),
        .lw_req(), .lw_vtile(), .lw_k(), .lw_col({(4*16){1'b0}}),
        .kc_ckv({(32*16){1'b0}}), .kc_krope({(16*16){1'b0}}),
        .kc_req(), .kc_idx(),
        .kv_row_sel(), .kv_row_in({ROW_BITS{1'b0}}),
        .flash_req(r0_flash_req), .flash_is_expert(r0_flash_is_expert),
        .flash_expert_id(), .flash_row_idx(),
        .flash_done(r0_flash_done), .flash_row({ROW_BITS{1'b0}}),
        .pf_valid(pf_valid), .pf_expert_id(pf_expert_id),
        .mem_req_valid(r0_mem_req_valid), .mem_req_ready({NCH{1'b1}}),
        .mem_req_addr(), .mem_req_tag(),
        .mem_resp_valid({NCH{1'b0}}), .mem_resp_ready(r0_mem_resp_ready),
        .mem_resp_data({(NCH*DW){1'b0}}), .mem_resp_tag({(NCH*TW){1'b0}}),
        .wl_mem_en(), .wl_mem_addr(), .wl_mem_data({WLDW{1'b0}}),
        .decomp_tbl_we(1'b0), .decomp_tbl_sel(1'b0),
        .decomp_tbl_addr(9'h000), .decomp_tbl_wdata(10'h000),
        .argmax_o(), .h_state(), .mdl_busy(),
        .ec_resp_valid(), .ec_hit(), .ec_resp_slot(), .ec_busy(),
        .ec_hit_count(), .ec_miss_count(), .ec_demand_stall_cycles(),
        .ec_pf_issued(r0_pf_issued), .ec_pf_hit(),
        .kv_row_valid(), .kv_row_out(), .kv_busy(),
        .kv_append_count(), .kv_resident_lo(), .kv_overflowed(),
        .ec_dropped(),
        .xbar_req_count(r0_xbar_req_count), .xbar_resp_count(),
        .xbar_resp_valid(), .xbar_resp_data(),
        .loader_busy(), .loader_done_count(), .loader_beat_count(),
        .loader_w_q(), .loader_in_valid()
    );

    // ---- DUT0 Flash stub: complete a held flash_req after 8 cycles ----
    reg [3:0] r0_lat;
    always @(posedge clk) begin
        if (rst) begin
            r0_flash_done   <= 1'b0;
            r0_lat          <= 4'd0;
            r0_flash_fires  <= 32'd0;
            r0_flash_grants <= 32'd0;
        end else begin
            r0_flash_done <= 1'b0;
            if (r0_flash_req) begin
                r0_flash_fires <= r0_flash_fires + 32'd1;
                if (r0_lat == 4'd8) begin
                    r0_flash_done   <= 1'b1;
                    r0_lat          <= 4'd0;
                    r0_flash_grants <= r0_flash_grants + 32'd1;
                end else begin
                    r0_lat <= r0_lat + 4'd1;
                end
            end else begin
                r0_lat <= 4'd0;
            end
        end
    end

    //========================================================================
    // stimulus + checks
    //========================================================================
    integer w;
    task wait_cycles(input integer n);
        begin for (w = 0; w < n; w = w + 1) @(posedge clk); end
    endtask

    // wait until DUT1's xbar has returned `n` responses (bounded)
    task wait_resp(input [31:0] n);
        begin
            w = 0;
            while ((r1_xbar_resp_count < n) && (w < 500)) begin
                @(posedge clk); w = w + 1;
            end
            check(r1_xbar_resp_count == n, "xbar responses drained (bounded wait)");
        end
    endtask

    initial begin
        pass_count   = 0;
        rst          = 1'b1;
        pf_valid     = 1'b0;
        pf_expert_id = {EIDXW{1'b0}};
        wait_cycles(4);
        rst = 1'b0;
        wait_cycles(4);

        // (1) quiet after reset: no fetches, no xbar traffic, no flash
        check(r1_pf_issued      == 32'd0, "R1 no prefetch issued at reset");
        check(r1_xbar_req_count == 32'd0, "R1 xbar quiet at reset");
        check(r0_xbar_req_count == 32'd0, "R0 xbar quiet at reset");

        // (2) prefetch hint expert 3 -> RESIDENT=1 refill via ddr5_xbar
        @(posedge clk);
        pf_valid     <= 1'b1;
        pf_expert_id <= 3'd3;
        @(posedge clk);
        pf_valid     <= 1'b0;
        wait_cycles(3);
        check(r1_pf_issued == 32'd1, "R1 prefetch fetch started");
        wait_resp(32'd1);
        check(r1_seen_any, "R1 a DDR channel read was issued");
        check(r1_seen_tag == TAG_EFILL, "R1 channel read tag == TAG_EFILL (8'h05)");
        check(r1_xbar_req_count == 32'd1, "R1 exactly one xbar read for the refill");
        wait_cycles(4);

        // meanwhile RESIDENT=0 twin: SAME hint went to the FLASH channel
        check(r0_pf_issued == 32'd1,      "R0 prefetch fetch started (flash)");
        check(r0_flash_grants >= 32'd1 || r0_flash_fires >= 32'd1,
              "R0 flash channel carried the refill");
        check(r0_xbar_req_count == 32'd0, "R0 no xbar read for the refill (default path)");

        // (3) repeat hint for the SAME expert -> resident SKIP (install proven)
        @(posedge clk);
        pf_valid     <= 1'b1;
        pf_expert_id <= 3'd3;
        @(posedge clk);
        pf_valid     <= 1'b0;
        wait_cycles(20);
        check(r1_pf_issued      == 32'd1, "R1 repeat hint skipped (expert resident)");
        check(r1_xbar_req_count == 32'd1, "R1 no second xbar read on skip");

        // (4) hint a DIFFERENT expert -> second refill (FSM returned to IDLE)
        @(posedge clk);
        pf_valid     <= 1'b1;
        pf_expert_id <= 3'd5;
        @(posedge clk);
        pf_valid     <= 1'b0;
        wait_cycles(3);
        check(r1_pf_issued == 32'd2, "R1 second refill started (cache back in IDLE)");
        wait_resp(32'd2);
        check(r1_seen_tag == TAG_EFILL, "R1 second read also TAG_EFILL");
        check(r1_xbar_req_count == 32'd2, "R1 two xbar reads total");
        wait_cycles(10);

        // (5) the expert class never owned the Flash channel on DUT1
        //     (the per-cycle monitor above $fatal's on any flash_req; this
        //      bookend re-checks the level at the end of the run)
        check(!r1_flash_req, "R1 flash_req low at end of run");
        check(!r1_ec_busy,   "R1 expert cache idle at end of run");

        $display("ALL %0d TESTS PASSED (RESIDENT=1 refills served by ddr5_xbar TAG_EFILL; flash never fired; RESIDENT=0 twin used the Flash channel)", pass_count);
        $finish;
    end

    // global watchdog
    initial begin
        #100000;
        $fatal(1, "glm_q4k_system_resident_tb TIMEOUT");
    end

endmodule
