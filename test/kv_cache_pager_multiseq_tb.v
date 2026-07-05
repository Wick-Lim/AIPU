`timescale 1ns/1ps
//============================================================================
// kv_cache_pager_multiseq_tb.v  --  MULTI-SEQUENCE (NSEQ>1) KV ring pager
//----------------------------------------------------------------------------
// Proves the NSEQ>1 path of kv_cache_pager: NSEQ INDEPENDENT ring windows (one
// per batch lane / concurrent sequence), each with its OWN append counter and
// resident window, addressed at seq*RESIDENT+slot -- the KV-storage side of
// per-row (multi-seq) batched decode.  (The NSEQ=1 byte-identical path is
// covered by kv_cache_pager_tb.v; the formal harnesses prove the invariants.)
//
// INDEPENDENT X-aware golden model (flattened per (seq,pos)):
//   * sw_row[seq*S_MAX+p] : the opaque row appended at (seq, logical pos p).
//   * sw_count[seq]       : positions appended so far in that sequence.
//   * per-seq resident window = last RESIDENT positions of THAT sequence.
//   The Flash backing model returns sw_row[flash_seq*S_MAX+flash_idx] -- i.e.
//   it is KEYED BY THE DUT's flash_seq OUTPUT, so a wrong flash_seq returns the
//   wrong sequence's row and the value check FAILS.  That is the flash_seq test.
//
// KEY PROPERTIES CHECKED
//   (1) NO CROSS-SEQ SLOT COLLISION: different sequences at the SAME logical
//       position (hence the SAME ring slot pos%RESIDENT) hold DIFFERENT rows and
//       each gather returns its OWN sequence's row.  (If the seq*RESIDENT offset
//       were missing, seq 1's row would overwrite seq 0's slot -> caught.)
//   (2) INDEPENDENT EVICTION: overflowing one sequence past RESIDENT does NOT
//       evict another sequence's rows; each window (resident_lo/overflowed,
//       observed via the seq selects) slides on its OWN counter.
//   (3) flash_seq KEYING: two cold gathers at the SAME idx in different
//       sequences return each sequence's own (distinct) cold row.
//
//   Prints "ALL <N> TESTS PASSED" with zero failures; $fatal on any mismatch.
//============================================================================
module kv_cache_pager_multiseq_tb;

    localparam integer NSEQ      = 3;
    localparam integer ROW_BITS  = 64;
    localparam integer RESIDENT  = 4;     // power of two, PER SEQUENCE
    localparam integer S_MAX     = 32;
    localparam integer POSW      = 5;     // clog2(32)
    localparam integer SEQW      = 2;     // clog2(3)
    localparam integer FLASH_LAT = 5;

    reg                 clk, rst;
    reg                 append_valid;
    reg  [ROW_BITS-1:0] append_row;
    reg  [SEQW-1:0]     append_seq;
    reg                 gather_valid;
    reg  [POSW-1:0]     gather_idx;
    reg  [SEQW-1:0]     gather_seq;
    wire                row_valid;
    wire [ROW_BITS-1:0] row_out;
    wire                busy;
    wire                flash_req;
    wire [POSW-1:0]     flash_idx;
    wire [SEQW-1:0]     flash_seq;
    reg                 flash_done;
    reg  [ROW_BITS-1:0] flash_row;
    wire [POSW-1:0]     append_count;
    wire [POSW-1:0]     resident_lo;
    wire                overflowed;

    kv_cache_pager #(
        .ROW_BITS(ROW_BITS), .RESIDENT(RESIDENT), .S_MAX(S_MAX),
        .POSW(POSW), .FLASH_LAT(FLASH_LAT), .NSEQ(NSEQ)
    ) dut (
        .clk(clk), .rst(rst),
        .append_valid(append_valid), .append_row(append_row), .append_seq(append_seq),
        .gather_valid(gather_valid), .gather_idx(gather_idx), .gather_seq(gather_seq),
        .row_valid(row_valid), .row_out(row_out), .busy(busy),
        .flash_req(flash_req), .flash_idx(flash_idx), .flash_seq(flash_seq),
        .flash_done(flash_done), .flash_row(flash_row),
        .append_count(append_count), .resident_lo(resident_lo),
        .overflowed(overflowed),
        .ecc_serr(), .ecc_derr()
    );

    //------------------------------------------------------------------ clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    //------------------------------------------------ independent golden model
    reg [ROW_BITS-1:0] sw_row [0:NSEQ*S_MAX-1];
    integer            sw_count [0:NSEQ-1];
    integer            tests, errors;
    integer            k, s, p;
    integer            ridx, rlo;

    // unique opaque pattern per (sequence, logical position): a collision (same
    // ring slot, wrong seq) or a wrong flash_seq would return a different value.
    function [ROW_BITS-1:0] genrow(input integer seq, input integer pos);
        reg [7:0] s8, p8;
        begin
            s8 = seq[7:0]; p8 = pos[7:0];
            genrow = { 16'hBEEF, s8, p8, 16'hC0DE, s8, p8 };
        end
    endfunction

    //------------------------------------------------------- Flash DMA model
    // On a HELD flash_req, wait FLASH_LAT cycles then pulse flash_done once with
    // sw_row[flash_seq*S_MAX+flash_idx] -- KEYED BY THE DUT's flash_seq output.
    integer flash_cnt;
    reg     flash_active;
    always @(posedge clk) begin
        if (rst) begin
            flash_done   <= 1'b0;
            flash_row    <= {ROW_BITS{1'b0}};
            flash_active <= 1'b0;
            flash_cnt    <= 0;
        end else begin
            flash_done <= 1'b0;
            if (flash_req && !flash_active && !flash_done) begin
                flash_active <= 1'b1;
                flash_cnt    <= FLASH_LAT;
            end else if (flash_active) begin
                if (flash_cnt <= 1) begin
                    flash_done   <= 1'b1;
                    flash_row    <= sw_row[flash_seq*S_MAX + flash_idx];
                    flash_active <= 1'b0;
                end else begin
                    flash_cnt <= flash_cnt - 1;
                end
            end
        end
    end

    //----------------------------------------------------------------- tasks
    // append the next logical position of sequence `seq` (pos = its own count).
    task do_append(input integer seq);
        integer pp;
        begin
            pp = sw_count[seq];
            @(negedge clk);
            append_valid = 1'b1;
            append_seq   = seq[SEQW-1:0];
            append_row   = genrow(seq, pp);
            @(negedge clk);
            append_valid = 1'b0;
            sw_row[seq*S_MAX + pp] = genrow(seq, pp);
            sw_count[seq]          = sw_count[seq] + 1;
        end
    endtask

    // issue one gather on sequence `seq`; wait row_valid; check vs golden row.
    task do_gather(input integer seq, input [POSW-1:0] idx, input expect_cold);
        reg [ROW_BITS-1:0] got, exp;
        integer wd;
        begin
            @(negedge clk);
            gather_valid = 1'b1;
            gather_seq   = seq[SEQW-1:0];
            gather_idx   = idx;
            @(negedge clk);
            gather_valid = 1'b0;
            wd = 0;
            while (!row_valid) begin
                @(negedge clk);
                wd = wd + 1;
                if (wd > FLASH_LAT + 30) begin
                    $display("FAIL: gather seq=%0d idx=%0d timed out", seq, idx);
                    $fatal(1, "gather timeout");
                end
            end
            got   = row_out;
            exp   = sw_row[seq*S_MAX + idx];
            tests = tests + 1;
            if (got !== exp) begin
                errors = errors + 1;
                $display("FAIL: gather seq=%0d idx=%0d got=%h exp=%h (cold=%0d wd=%0d)",
                         seq, idx, got, exp, expect_cold, wd);
            end
            if (!expect_cold && wd > 1) begin
                errors = errors + 1;
                $display("FAIL: resident seq=%0d idx=%0d took %0d cyc (expected fast)",
                         seq, idx, wd);
            end
            if (expect_cold && wd < 2) begin
                errors = errors + 1;
                $display("FAIL: cold seq=%0d idx=%0d too fast (%0d) -- no Flash?",
                         seq, idx, wd);
            end
        end
    endtask

    // observe sequence `seq`'s append counter (append_count is comb on append_seq).
    task check_count(input integer seq, input integer exp);
        begin
            @(negedge clk);
            append_seq = seq[SEQW-1:0];
            #1;
            tests = tests + 1;
            if (append_count !== exp[POSW-1:0]) begin
                errors = errors + 1;
                $display("FAIL count: seq=%0d got=%0d exp=%0d",
                         seq, append_count, exp);
            end
        end
    endtask

    // observe sequence `seq`'s resident window (resident_lo/overflowed comb on gather_seq).
    task check_window(input integer seq, input integer exp_lo, input exp_over);
        begin
            @(negedge clk);
            gather_seq = seq[SEQW-1:0];
            #1;
            tests = tests + 1;
            if (resident_lo !== exp_lo[POSW-1:0] || overflowed !== exp_over) begin
                errors = errors + 1;
                $display("FAIL window: seq=%0d lo=%0d/%0d over=%0b/%0b",
                         seq, resident_lo, exp_lo, overflowed, exp_over);
            end
        end
    endtask

    //------------------------------------------------------------- stimulus
    initial begin
        append_valid = 0; append_row = 0; append_seq = 0;
        gather_valid = 0; gather_idx = 0; gather_seq = 0;
        tests = 0; errors = 0;
        for (s = 0; s < NSEQ; s = s + 1) sw_count[s] = 0;

        rst = 1'b1;
        repeat (3) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        //==================================================================
        // PHASE 1: NO CROSS-SEQ SLOT COLLISION.
        //   Append logical positions 0,1,2 to EACH of seq 0,1,2 -- same ring
        //   slots (0,1,2), different data.  Each gather must return its OWN seq.
        //==================================================================
        for (p = 0; p < 3; p = p + 1)
            for (s = 0; s < NSEQ; s = s + 1) do_append(s);   // interleaved
        for (s = 0; s < NSEQ; s = s + 1) begin
            check_count(s, 3);
            check_window(s, 0, 1'b0);                        // 3 < 4 -> no evict
        end
        // all resident, each returns its own row (slot-collision proof).
        for (s = 0; s < NSEQ; s = s + 1)
            for (p = 0; p < 3; p = p + 1)
                do_gather(s, p[POSW-1:0], 1'b0);

        //==================================================================
        // PHASE 2: INDEPENDENT EVICTION.
        //   Overflow seq 0 past RESIDENT (append pos 3..6 -> count0=7, resident
        //   [3..6], lo=3, cold 0..2).  seq 1 & 2 stay at count=3 (all resident).
        //==================================================================
        for (p = 3; p < 7; p = p + 1) do_append(0);
        check_count(0, 7);  check_window(0, 3, 1'b1);        // seq0 overflowed
        check_count(1, 3);  check_window(1, 0, 1'b0);        // seq1 untouched
        check_count(2, 3);  check_window(2, 0, 1'b0);        // seq2 untouched

        do_gather(0, 5'd6, 1'b0);    // seq0 newest resident
        do_gather(0, 5'd3, 1'b0);    // seq0 oldest resident
        do_gather(0, 5'd0, 1'b1);    // seq0 pos0 EVICTED -> cold via Flash
        // seq1 pos0 is STILL RESIDENT and returns seq1's row, NOT seq0's cold pos0.
        do_gather(1, 5'd0, 1'b0);
        do_gather(2, 5'd0, 1'b0);

        //==================================================================
        // PHASE 3: flash_seq KEYING.
        //   Overflow seq 2 as well (append pos 3..6 -> cold 0..2).  Now BOTH
        //   seq0 and seq2 have a COLD pos 0; the two cold gathers at idx 0 must
        //   return DIFFERENT rows -> the Flash fetch used the right flash_seq.
        //==================================================================
        for (p = 3; p < 7; p = p + 1) do_append(2);
        check_count(2, 7);  check_window(2, 3, 1'b1);
        do_gather(0, 5'd0, 1'b1);    // cold -> Flash returns seq0 pos0
        do_gather(2, 5'd0, 1'b1);    // cold -> Flash returns seq2 pos0 (distinct!)
        do_gather(0, 5'd1, 1'b1);    // cold seq0 pos1
        do_gather(2, 5'd2, 1'b1);    // cold seq2 pos2
        do_gather(1, 5'd2, 1'b0);    // seq1 still fully resident

        //==================================================================
        // PHASE 4: randomized interleaved multi-seq stream.
        //   Each step appends the next position of a random sequence (while it
        //   has room) or gathers a random already-appended index of a random
        //   sequence; the cold class is predicted by THAT sequence's own window.
        //==================================================================
        for (k = 0; k < 120; k = k + 1) begin
            s = $unsigned($random) % NSEQ;
            if (sw_count[s] < S_MAX-2 &&
                ($random % 3 != 0 || sw_count[s] < RESIDENT+1)) begin
                do_append(s);
            end else if (sw_count[s] > 0) begin
                ridx = $unsigned($random) % sw_count[s];
                rlo  = (sw_count[s] > RESIDENT) ? (sw_count[s] - RESIDENT) : 0;
                do_gather(s, ridx[POSW-1:0], (ridx < rlo) ? 1'b1 : 1'b0);
            end
        end

        //------------------------------------------------------------ tally
        if (errors != 0) begin
            $display("FAILED: %0d errors out of %0d checks", errors, tests);
            $fatal(1, "kv_cache_pager_multiseq_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end

    // safety net: never hang.
    initial begin
        #500000;
        $fatal(1, "TIMEOUT: kv_cache_pager_multiseq_tb did not finish");
    end

endmodule
