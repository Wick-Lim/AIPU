`timescale 1ns/1ps
//============================================================================
// boot_loader_manifest_tb.v -- boot-time INTEGRITY + VERSION gate testbench
//   (USAGE_GAPS LOCK-IN-NOW A; findings #2/#3/#4/#35/#40)
//----------------------------------------------------------------------------
// Proves the fail-closed boot manifest gate added to src/boot_loader.v behind
// the DEFAULT-OFF `INTEGRITY` parameter: a partial / corrupt / wrong-version
// model image can NOT be DMA'd and silently released as a working model.
//
// Two boot_loaders run the SAME stimulus, side by side, via a generate pair:
//   g_bl[1] : INTEGRITY=1 (gate ON)  -- the new behaviour under test
//   g_bl[0] : INTEGRITY=0 (gate OFF) -- MUST be byte-identical to legacy:
//             it releases `done` for EVERY image (good or bad) and NEVER
//             asserts boot_fail / a nonzero err_code (a live inert-check).
//
// Each side has its own Flash-read latency pipe + back-pressured DDR5 write
// sink (independent LFSR stalls -> exercises the skid FIFO), and its own
// done-ever / mutual-exclusion monitors.  An INDEPENDENT golden CRC (folded
// in DDR5 write / retirement order, exactly the DUT's fold order) + a golden
// total-word count drive the manifest header, so the checks are real.
//
// Scenarios (gate ON must):
//   (a) GOOD image (magic/version/len/CRC all match) -> done rises, no fail.
//   (b) TRUNCATED image (declared total-length > words actually delivered)
//       -> boot_fail=ERR_LEN, done NEVER asserts (done_ever stays 0).
//   (c) WRONG-VERSION image -> boot_fail=ERR_VER, done never asserts.
//   (d) BAD-CRC image (one payload word differs) -> boot_fail=ERR_CRC, no done.
//   (e) BAD-MAGIC image (not a model image at all) -> boot_fail=ERR_MAGIC.
//   (f) GOOD re-run after a fail -> the engine recovers and releases (done).
// Gate OFF must, for the SAME (b)..(e) bad images, still assert done and stay
// inert (boot_fail low) -- i.e. behaviour identical to today.
//============================================================================
module boot_loader_manifest_tb;

    // ---- DUT geometry (defaults) ----
    localparam integer ADDR_W  = 32;
    localparam integer DATA_W  = 64;
    localparam integer SEG_MAX = 4;
    localparam integer BURST   = 8;
    localparam integer LEN_W   = 16;
    localparam integer SEGW    = 3;            // clog2(SEG_MAX+1)
    localparam integer PROG_W  = LEN_W+SEGW;   // 19
    localparam integer MAGIC_W = 32;
    localparam integer VER_W   = 16;
    localparam integer CRC_W   = 32;

    // ---- manifest constants (MUST match src/boot_loader.v defaults) ----
    localparam [31:0] MAGIC    = 32'h4D4F_444C;   // "MODL"
    localparam [31:0] VERSION  = 32'h0000_0001;
    localparam [31:0] CRC_POLY = 32'h04C1_1DB7;   // CRC-32 (IEEE 802.3)
    localparam [31:0] CRC_INIT = 32'hFFFF_FFFF;

    // ---- error classes (MUST match src/boot_loader.v) ----
    localparam [2:0] ERR_NONE=3'd0, ERR_MAGIC=3'd1, ERR_VER=3'd2,
                     ERR_LEN=3'd3,  ERR_CRC=3'd4;

    // ---- TB stub geometry ----
    localparam integer FLASH_LAT = 5;
    localparam integer FSIZE     = 1024;
    localparam integer DSIZE     = 1024;
    localparam integer MAXCYC    = 200000;

    integer pass_count = 0;
    integer errors     = 0;

    task chk(input cond, input [1023:0] msg);
        begin
            if (cond) pass_count = pass_count + 1;
            else begin
                errors = errors + 1;
                $display("ASSERT FAIL: %0s  (t=%0t)", msg, $time);
                $fatal(1, "boot_loader_manifest_tb assertion failed");
            end
        end
    endtask

    // ------------------------------------------------------------------
    // clock / reset / shared stimulus
    // ------------------------------------------------------------------
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;
    reg mon_en;

    reg                        start;
    reg  [SEGW-1:0]            seg_count;
    reg  [SEG_MAX*ADDR_W-1:0]  seg_flash_base;
    reg  [SEG_MAX*ADDR_W-1:0]  seg_ddr_base;
    reg  [SEG_MAX*LEN_W-1:0]   seg_len;
    reg  [MAGIC_W-1:0]         mf_magic;
    reg  [VER_W-1:0]           mf_version;
    reg  [PROG_W-1:0]          mf_len;
    reg  [CRC_W-1:0]           mf_crc;

    // shared Flash image (read-only; both sides read the identical bytes)
    reg [DATA_W-1:0] FlashMem [0:FSIZE-1];

    // ------------------------------------------------------------------
    // independent golden CRC fold -- MUST mirror the DUT's crc_fold exactly
    // (MSB-first over the DATA_W-bit word, IEEE-802.3 poly)
    // ------------------------------------------------------------------
    function [CRC_W-1:0] tb_crc_fold;
        input [CRC_W-1:0]  c0;
        input [DATA_W-1:0] d;
        integer b; reg [CRC_W-1:0] c;
        begin
            c = c0;
            for (b = DATA_W-1; b >= 0; b = b - 1)
                c = (c[CRC_W-1] ^ d[b]) ? ((c << 1) ^ CRC_POLY[CRC_W-1:0])
                                        :  (c << 1);
            tb_crc_fold = c;
        end
    endfunction

    // ------------------------------------------------------------------
    // TWO DUTs (gate OFF and gate ON) + per-instance stubs & monitors
    // ------------------------------------------------------------------
    genvar gi;
    generate for (gi = 0; gi < 2; gi = gi + 1) begin : g_bl
        localparam integer INTEG = gi;         // 0 = OFF (legacy), 1 = ON

        wire              flash_req;
        wire [ADDR_W-1:0] flash_addr;
        reg               flash_ready;
        wire              flash_rvalid;
        wire [DATA_W-1:0] flash_rdata;

        wire              ddr_we;
        wire [ADDR_W-1:0] ddr_addr;
        wire [DATA_W-1:0] ddr_wdata;
        reg               ddr_ready;

        wire              busy;
        wire              done;
        wire [PROG_W-1:0] words_done;
        wire              boot_fail;
        wire [2:0]        err_code;

        boot_loader #(
            .ADDR_W(ADDR_W), .DATA_W(DATA_W), .SEG_MAX(SEG_MAX),
            .BURST(BURST), .LEN_W(LEN_W), .INTEGRITY(INTEG),
            .MAGIC_W(MAGIC_W), .VER_W(VER_W), .CRC_W(CRC_W),
            .MAGIC(MAGIC), .VERSION(VERSION),
            .CRC_POLY(CRC_POLY), .CRC_INIT(CRC_INIT)
        ) dut (
            .clk(clk), .rst(rst),
            .start(start), .seg_count(seg_count),
            .seg_flash_base(seg_flash_base), .seg_ddr_base(seg_ddr_base),
            .seg_len(seg_len),
            .mf_magic(mf_magic), .mf_version(mf_version),
            .mf_len(mf_len), .mf_crc(mf_crc),
            .flash_req(flash_req), .flash_addr(flash_addr),
            .flash_ready(flash_ready), .flash_rvalid(flash_rvalid),
            .flash_rdata(flash_rdata),
            .ddr_we(ddr_we), .ddr_addr(ddr_addr), .ddr_wdata(ddr_wdata),
            .ddr_ready(ddr_ready),
            .busy(busy), .done(done), .words_done(words_done),
            .boot_fail(boot_fail), .err_code(err_code)
        );

        // ---- Flash read: IN-ORDER FLASH_LAT latency pipe over shared image ----
        wire              issue_fire = flash_req & flash_ready;
        wire [DATA_W-1:0] src_word   = FlashMem[flash_addr % FSIZE];
        reg               lat_val [0:FLASH_LAT-1];
        reg [DATA_W-1:0]  lat_dat [0:FLASH_LAT-1];
        integer p;
        always @(posedge clk) begin
            if (rst) begin
                for (p = 0; p < FLASH_LAT; p = p + 1) begin
                    lat_val[p] <= 1'b0; lat_dat[p] <= {DATA_W{1'b0}};
                end
            end else begin
                lat_val[0] <= issue_fire; lat_dat[0] <= src_word;
                for (p = 1; p < FLASH_LAT; p = p + 1) begin
                    lat_val[p] <= lat_val[p-1]; lat_dat[p] <= lat_dat[p-1];
                end
            end
        end
        assign flash_rvalid = lat_val[FLASH_LAT-1];
        assign flash_rdata  = lat_dat[FLASH_LAT-1];

        // ---- DDR5 write sink + retirement counter ----
        reg [DATA_W-1:0] DDRMem [0:DSIZE-1];
        integer          tb_writes;
        wire             write_fire = ddr_we & ddr_ready;
        integer          wa;
        always @(posedge clk) begin
            if (!rst && write_fire) begin
                wa = ddr_addr;
                if (wa < DSIZE) DDRMem[wa] <= ddr_wdata;
                tb_writes = tb_writes + 1;
            end
        end

        // ---- back-pressure LFSRs (seeded per side; maximal -> always progress) ----
        reg [15:0] lf1, lf2;
        always @(posedge clk) begin
            if (rst) begin
                lf1 <= 16'hACE1 ^ (gi ? 16'h1234 : 16'h0);
                lf2 <= 16'hBEEF ^ (gi ? 16'h5678 : 16'h0);
            end else begin
                lf1 <= {lf1[14:0], lf1[15]^lf1[13]^lf1[12]^lf1[10]};
                lf2 <= {lf2[14:0], lf2[15]^lf2[13]^lf2[12]^lf2[10]};
            end
        end
        always @* flash_ready = lf1[0] | lf1[1] | lf1[2];
        always @* ddr_ready   = lf2[0] | lf2[1];

        // ---- done_ever (proves "never released" on a fail) ----
        //   Cleared on the HONORED start (start & ~busy) -- exactly when the DUT
        //   drops its own `done` -- so a prior run's still-high `done` cannot
        //   leak into this run's "never released" evidence.
        wire honored_start = start & ~busy;
        reg  done_ever;
        always @(posedge clk) begin
            if (rst || honored_start) done_ever <= 1'b0;
            else if (done)            done_ever <= 1'b1;
        end

        // ---- continuous monitors ----
        always @(posedge clk) if (mon_en && !rst) begin
            // no-X on the live gate outputs
            chk(^{busy,done,boot_fail} !== 1'bx, "X on busy/done/boot_fail");
            chk(^err_code !== 1'bx,              "X on err_code");
            if (INTEG != 0) begin
                // fail-closed: released and failed are MUTUALLY EXCLUSIVE
                chk(!(done && boot_fail), "ON: done & boot_fail must be exclusive");
                // a raised boot_fail must carry a real (nonzero) error class
                if (boot_fail) chk(err_code !== ERR_NONE,
                                   "ON: boot_fail must carry a nonzero err_code");
            end else begin
                // gate OFF is INERT -- identical to legacy (never fails)
                chk(boot_fail === 1'b0, "OFF: boot_fail must stay 0 (inert)");
                chk(err_code  === ERR_NONE, "OFF: err_code must stay 0 (inert)");
            end
        end
    end endgenerate

    // ------------------------------------------------------------------
    // TB-side descriptor (unpacked) + golden builder
    // ------------------------------------------------------------------
    integer fbase [0:SEG_MAX-1];
    integer dbase [0:SEG_MAX-1];
    integer len_  [0:SEG_MAX-1];
    integer segc;
    integer total_words;
    reg [CRC_W-1:0] golden_crc;
    integer k, w;

    task build_descriptor;
        begin
            seg_flash_base = {(SEG_MAX*ADDR_W){1'b0}};
            seg_ddr_base   = {(SEG_MAX*ADDR_W){1'b0}};
            seg_len        = {(SEG_MAX*LEN_W){1'b0}};
            for (k = 0; k < SEG_MAX; k = k + 1) begin
                seg_flash_base[k*ADDR_W +: ADDR_W] = fbase[k][ADDR_W-1:0];
                seg_ddr_base  [k*ADDR_W +: ADDR_W] = dbase[k][ADDR_W-1:0];
                seg_len       [k*LEN_W  +: LEN_W ] = len_[k][LEN_W-1:0];
            end
            seg_count = segc[SEGW-1:0];

            // golden: total words + running CRC in DDR5 write / retirement order
            total_words = 0;
            golden_crc  = CRC_INIT[CRC_W-1:0];
            for (k = 0; k < segc; k = k + 1)
                for (w = 0; w < len_[k]; w = w + 1) begin
                    golden_crc  = tb_crc_fold(golden_crc,
                                              FlashMem[(fbase[k] + w) % FSIZE]);
                    total_words = total_words + 1;
                end
        end
    endtask

    // ------------------------------------------------------------------
    // run one scenario on BOTH sides; check gate ON verdict + gate OFF inertness
    //   expect_fail : ON must fail-close (no done); 0 => ON must release
    //   expect_err  : the ERR_* class ON must report when expect_fail
    // ------------------------------------------------------------------
    integer c, s;
    task run_scenario(input [511:0] name, input expect_fail, input [2:0] expect_err);
        begin
            // clear the two write counters (hierarchical into the gen pair)
            g_bl[0].tb_writes = 0;
            g_bl[1].tb_writes = 0;

            // power-on: 1-cycle start pulse (both DUTs latch together; the
            // honored start also clears each side's done_ever)
            @(negedge clk);
            start = 1'b1;
            @(posedge clk);
            @(negedge clk); start = 1'b0;
            // scramble live inputs to prove they were latched
            seg_flash_base = {(SEG_MAX*ADDR_W){1'b1}};
            seg_ddr_base   = {(SEG_MAX*ADDR_W){1'b1}};
            seg_len        = {(SEG_MAX*LEN_W){1'b1}};
            seg_count      = {SEGW{1'b1}};
            mf_magic       = {MAGIC_W{1'b1}};
            mf_version     = {VER_W{1'b1}};
            mf_len         = {PROG_W{1'b1}};
            mf_crc         = {CRC_W{1'b1}};

            // wait: OFF always releases; ON releases (good) or fails (bad)
            c = 0;
            while (!( g_bl[0].done &&
                     (g_bl[1].done || g_bl[1].boot_fail) )) begin
                @(posedge clk);
                c = c + 1;
                if (c > MAXCYC) chk(1'b0, "TIMEOUT waiting for settle");
            end
            // let levels settle
            repeat (4) @(posedge clk);

            // -------- gate OFF (g_bl[0]) == legacy, for ANY image --------
            chk(g_bl[0].done === 1'b1,      "OFF: done must assert (legacy)");
            chk(g_bl[0].busy === 1'b0,      "OFF: busy must drop");
            chk(g_bl[0].boot_fail === 1'b0, "OFF: never fails (inert)");
            chk(g_bl[0].tb_writes == total_words,
                "OFF: writes must equal delivered words");

            // -------- gate ON (g_bl[1]) verdict --------
            if (!expect_fail) begin
                chk(g_bl[1].done === 1'b1,          "ON good: done must assert");
                chk(g_bl[1].boot_fail === 1'b0,     "ON good: no boot_fail");
                chk(g_bl[1].err_code === ERR_NONE,  "ON good: err_code NONE");
                chk(g_bl[1].busy === 1'b0,          "ON good: busy dropped");
                chk(g_bl[1].words_done === total_words[PROG_W-1:0],
                    "ON good: words_done == total");
                chk(g_bl[1].tb_writes == total_words,
                    "ON good: delivered words == total");
            end else begin
                chk(g_bl[1].boot_fail === 1'b1,     "ON bad: boot_fail must assert");
                chk(g_bl[1].done === 1'b0,          "ON bad: done must NOT assert");
                chk(g_bl[1].done_ever === 1'b0,     "ON bad: done NEVER asserted");
                chk(g_bl[1].err_code === expect_err,"ON bad: err_code class");
                chk(g_bl[1].busy === 1'b0,          "ON bad: engine halted (busy low)");
            end

            // -------- levels are STEADY + no post-verdict writes (both sides) --
            s = g_bl[1].tb_writes;
            repeat (8) @(posedge clk);
            chk(g_bl[0].done === 1'b1,     "OFF: done stays steady");
            if (!expect_fail) chk(g_bl[1].done === 1'b1,
                                  "ON good: done stays steady");
            else begin
                chk(g_bl[1].boot_fail === 1'b1, "ON bad: boot_fail stays steady");
                chk(g_bl[1].done === 1'b0,      "ON bad: done stays low");
                chk(g_bl[1].done_ever === 1'b0, "ON bad: still never released");
            end
            chk(g_bl[1].tb_writes == s, "ON: no spurious writes after verdict");

            $display("  [%0s] PASS  segs=%0d words=%0d ON:%s%0d OFF:done  (settle=%0d cyc)",
                     name, segc, total_words,
                     expect_fail ? "FAIL err=" : "done err=",
                     expect_fail ? g_bl[1].err_code : g_bl[1].err_code, c);
        end
    endtask

    // convenience: preset a GOOD manifest matching the current golden
    task good_manifest;
        begin
            mf_magic   = MAGIC[MAGIC_W-1:0];
            mf_version = VERSION[VER_W-1:0];
            mf_len     = total_words[PROG_W-1:0];
            mf_crc     = golden_crc;
        end
    endtask

    // load a canonical 4-segment resident-set descriptor into fbase/dbase/len_
    task load_default_descriptor;
        begin
            fbase[0]=32'h0010; dbase[0]=32'h0100; len_[0]=13;
            fbase[1]=32'h0030; dbase[1]=32'h0180; len_[1]=0;   // zero-len middle
            fbase[2]=32'h0040; dbase[2]=32'h0200; len_[2]=8;
            fbase[3]=32'h0080; dbase[3]=32'h0300; len_[3]=20;
            segc = 4;
        end
    endtask

    // ------------------------------------------------------------------
    // stimulus
    // ------------------------------------------------------------------
    integer si;
    initial begin
        for (si = 0; si < FSIZE; si = si + 1)
            FlashMem[si] = {$random, $random};

        start=1'b0; mon_en=1'b0;
        seg_count={SEGW{1'b0}};
        seg_flash_base={(SEG_MAX*ADDR_W){1'b0}};
        seg_ddr_base  ={(SEG_MAX*ADDR_W){1'b0}};
        seg_len       ={(SEG_MAX*LEN_W){1'b0}};
        mf_magic={MAGIC_W{1'b0}}; mf_version={VER_W{1'b0}};
        mf_len={PROG_W{1'b0}};    mf_crc={CRC_W{1'b0}};

        rst=1'b1;
        repeat (4) @(posedge clk);
        @(negedge clk); rst=1'b0; mon_en=1'b1;
        @(posedge clk);
        chk(g_bl[0].done===1'b0 && g_bl[1].done===1'b0, "reset clears done");
        chk(g_bl[1].boot_fail===1'b0, "reset clears boot_fail");

        // ---- (a) GOOD image: everything matches -> ON releases ----
        load_default_descriptor;   build_descriptor;   good_manifest;
        run_scenario("a: GOOD image -> released", 1'b0, ERR_NONE);

        // ---- (b) TRUNCATED image: header declares MORE words than delivered ----
        //   (a partial DMA -- the classic silent-bad-model hazard) -> ERR_LEN
        load_default_descriptor;   build_descriptor;   good_manifest;
        mf_len = total_words[PROG_W-1:0] + 5'd7;   // header claims 7 more words
        run_scenario("b: TRUNCATED -> fail-closed", 1'b1, ERR_LEN);

        // ---- (c) WRONG VERSION: valid payload, unsupported format version ----
        load_default_descriptor;   build_descriptor;   good_manifest;
        mf_version = VERSION[VER_W-1:0] + 16'd1;
        run_scenario("c: WRONG-VERSION -> fail-closed", 1'b1, ERR_VER);

        // ---- (d) BAD CRC: payload corrupt (expected CRC won't match) ----
        load_default_descriptor;   build_descriptor;   good_manifest;
        mf_crc = golden_crc ^ 32'h0000_0001;       // single-bit CRC corruption
        run_scenario("d: BAD-CRC -> fail-closed", 1'b1, ERR_CRC);

        // ---- (e) BAD MAGIC: not a model image at all ----
        load_default_descriptor;   build_descriptor;   good_manifest;
        mf_magic = MAGIC[MAGIC_W-1:0] ^ 32'hFFFF_FFFF;
        run_scenario("e: BAD-MAGIC -> fail-closed", 1'b1, ERR_MAGIC);

        // ---- (f) GOOD re-run after a fail: engine recovers & releases ----
        //   (also a DIFFERENT single-segment geometry to vary the CRC/length)
        fbase[0]=32'h0200; dbase[0]=32'h0040; len_[0]=37;
        fbase[1]=0; dbase[1]=0; len_[1]=0;
        fbase[2]=0; dbase[2]=0; len_[2]=0;
        fbase[3]=0; dbase[3]=0; len_[3]=0;
        segc = 1;
        build_descriptor;   good_manifest;
        run_scenario("f: GOOD re-run after fail -> released", 1'b0, ERR_NONE);

        // ---- (g) EMPTY resident set (seg_count=0) with a matching empty manifest:
        //   len=0, CRC=seed -> ON must still release (well-defined boundary) ----
        fbase[0]=0; dbase[0]=0; len_[0]=0;
        segc = 0;
        build_descriptor;   good_manifest;   // total_words=0, golden_crc=CRC_INIT
        run_scenario("g: EMPTY image -> released", 1'b0, ERR_NONE);

        if (errors == 0)
            $display("ALL %0d TESTS PASSED", pass_count);
        else
            $fatal(1, "%0d ERRORS", errors);
        $finish;
    end

    // global watchdog
    initial begin
        #(20*MAXCYC);
        $fatal(1, "GLOBAL TIMEOUT");
    end

endmodule
