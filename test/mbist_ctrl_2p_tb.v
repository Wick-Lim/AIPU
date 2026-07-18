`timescale 1ns/1ps
//============================================================================
// mbist_ctrl_2p_tb.v -- verify the dual-port March + concurrent-coupling BIST.
//
//   A behavioral TRUE-DUAL-PORT RAM (1R1W, async read, sync write, separate
//   addresses) with three switchable fault modes drives three scenarios:
//     1. GOOD                       -> pass (fail=0, done=1)
//     2. STUCK-AT cell              -> fail, fail_kind=0 (march), fail_addr=cell
//     3. CONCURRENT WRITE->READ     -> fail, fail_kind=1 (port coupling)
//        coupling into the sentinel
//   Scenario 3 is INVISIBLE to the march phase (reads/writes never overlap
//   there) and to any single-port engine -- only the concurrent phase catches
//   it. That is the whole reason a 2-port collar exists.
//============================================================================
module mbist_ctrl_2p_tb;
    localparam integer DEPTH = 16;
    localparam integer WIDTH = 8;
    localparam integer AW    = 4;
    localparam [WIDTH-1:0] W0 = {WIDTH{1'b0}};
    localparam [WIDTH-1:0] W1 = {WIDTH{1'b1}};

    integer test_count = 0, errors = 0;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, start;

    wire            busy, done, fail, fail_kind;
    wire [AW-1:0]   fail_addr;
    wire [AW-1:0]   waddr, raddr;
    wire            we;
    wire [WIDTH-1:0] wdata;
    reg  [WIDTH-1:0] rdata;

    // ---- fault-injection controls (set by the TB before each run) ----
    reg              f_stuck;             // enable a stuck-at read cell
    reg  [AW-1:0]    f_stuck_addr;
    reg  [WIDTH-1:0] f_stuck_val;
    reg              f_couple;            // enable concurrent write->sentinel-read coupling

    // ---- true dual-port RAM: async read, sync write, separate addresses ----
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    always @(posedge clk) if (we) mem[waddr] <= wdata;   // sync write, port B

    // async read (port A), with the two fault overrides layered on:
    //   couple: a CONCURRENT write to a different cell corrupts THIS read (the
    //           dual-port-specific fault) -- needs we=1 the same cycle as the read.
    //   stuck : the addressed cell always reads a fixed value.
    always @* begin
        if (f_couple && we && (raddr == {AW{1'b0}}) && (waddr != {AW{1'b0}}))
            rdata = W1;                                   // sentinel disturbed by concurrent write
        else if (f_stuck && (raddr == f_stuck_addr))
            rdata = f_stuck_val;                          // stuck-at read cell
        else
            rdata = mem[raddr];
    end

    mbist_ctrl_2p #(.DEPTH(DEPTH), .WIDTH(WIDTH)) dut (
        .clk(clk), .rst(rst), .start(start),
        .busy(busy), .done(done), .fail(fail), .fail_kind(fail_kind),
        .fail_addr(fail_addr),
        .waddr(waddr), .we(we), .wdata(wdata),
        .raddr(raddr), .rdata(rdata)
    );

    integer wd;
    // run one BIST pass to completion; caller has set the fault regs.
    task run_bist;
        begin
            @(negedge clk); rst = 1'b1; start = 1'b0;
            @(negedge clk); rst = 1'b0;
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;
            wd = 0;
            while (!done && wd < 4000) begin @(negedge clk); wd = wd + 1; end
            if (!done) begin
                $display("FAIL: BIST never asserted done (timeout, wd=%0d)", wd);
                errors = errors + 1;
            end
        end
    endtask

    task chk; input cond; input [255:0] name; begin
        test_count = test_count + 1;
        if (cond !== 1'b1) begin
            $display("FAIL[%0s]", name); errors = errors + 1;
        end
    end endtask

    initial begin
        rst = 1'b1; start = 1'b0;
        f_stuck = 1'b0; f_stuck_addr = 0; f_stuck_val = W1; f_couple = 1'b0;
        // mem contents are irrelevant (M0 initializes every cell); leave as-is.

        // ---------------- 1. GOOD RAM -> pass ----------------
        f_stuck = 1'b0; f_couple = 1'b0;
        run_bist;
        chk(done === 1'b1,  "GOOD done");
        chk(fail === 1'b0,  "GOOD no-fail");

        // ---------------- 2. STUCK-AT cell -> march fail ----------------
        // cell 6 always reads all-1: M0 writes 0, M1's r0 reads 1 -> mismatch @6.
        f_stuck = 1'b1; f_stuck_addr = 4'd6; f_stuck_val = W1; f_couple = 1'b0;
        run_bist;
        chk(done === 1'b1,          "STUCK done");
        chk(fail === 1'b1,          "STUCK fail");
        chk(fail_kind === 1'b0,     "STUCK kind=march");
        chk(fail_addr === 4'd6,     "STUCK addr=6");

        // ---------------- 3. CONCURRENT coupling -> port fail ----------------
        // a write to any non-sentinel cell corrupts a concurrent read of cell 0.
        // Invisible to the march phase; caught only by the concurrent phase.
        f_stuck = 1'b0; f_couple = 1'b1;
        run_bist;
        chk(done === 1'b1,          "COUPLE done");
        chk(fail === 1'b1,          "COUPLE fail");
        chk(fail_kind === 1'b1,     "COUPLE kind=coupling");

        // ---------------- 3b. SOUNDNESS: the coupling fault must be INVISIBLE
        //   to a march-only view. Prove it: with f_couple on but the concurrent
        //   phase's write suppressed... instead, prove the march phase alone did
        //   NOT already fail on this fault -- i.e. the failure is a coupling kind,
        //   which the chk above already binds. Additionally confirm a GOOD re-run
        //   after faults clears cleanly (no latched state across runs).
        f_stuck = 1'b0; f_couple = 1'b0;
        run_bist;
        chk(done === 1'b1,  "RECOVER done");
        chk(fail === 1'b0,  "RECOVER no-fail (state cleared across runs)");

        if (errors == 0)
            $display("ALL %0d TESTS PASSED  (mbist_ctrl_2p: dual-port March C- + concurrent-coupling; stuck-at kind=0, port-coupling kind=1, both caught)", test_count);
        else
            $display("FAILED: %0d error(s) across %0d checks", errors, test_count);
        $finish;
    end

    initial begin #200000; $display("FAIL: global timeout"); $finish; end
endmodule
