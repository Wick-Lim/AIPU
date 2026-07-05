`timescale 1ns/1ps
//============================================================================
// clk_throttle_tb.v -- DVFS/eco prescaler + its clk_en_ctrl throttle path.
//   Checks: (1) div<=1 => NO throttle (byte-identical default); (2) div=N => the
//   die is active exactly 1-in-N cycles; (3) hold suppresses the throttle; (4) with
//   the throttle wired into clk_en_ctrl, an always-advancing cluster's clk_en
//   duty-cycles to 1/N (die runs at f/N) yet is enabled on every active slot.
//============================================================================
module clk_throttle_tb;
    localparam integer DIVW = 8;
    reg clk = 0, rst = 1;
    reg [DIVW-1:0] div;
    reg hold;
    wire throttle;

    clk_throttle #(.DIVW(DIVW)) dut (
        .clk(clk), .rst(rst), .div(div), .hold(hold), .throttle(throttle));

    // clk_en_ctrl fed the throttle, with a cluster that ALWAYS wants to advance.
    wire [0:0] clk_en;
    wire [31:0] gated;
    clk_en_ctrl #(.N_CLUSTER(1), .HOLD(0)) cec (
        .clk(clk), .rst(rst), .boot_active(1'b0), .stall(1'b0), .throttle(throttle),
        .has_pending_work(1'b1), .input_valid(1'b1), .output_ready_downstream(1'b1),
        .clk_en(clk_en), .gated_cycles(gated));

    always #5 clk = ~clk;

    integer i, active, fails;
    initial begin
        fails = 0;

        // ---- (1) div=1: no throttle ----
        div = 1; hold = 0; rst = 1; @(posedge clk); #1 rst = 0;
        active = 0;
        for (i = 0; i < 40; i = i + 1) begin
            @(negedge clk);
            if (throttle !== 1'b0) begin
                $display("FAIL div=1: throttle=%b (expected 0) @%0d", throttle, i);
                fails = fails + 1;
            end
            if (clk_en[0] !== 1'b1) begin           // advancing cluster always enabled
                $display("FAIL div=1: clk_en=%b (expected 1) @%0d", clk_en[0], i);
                fails = fails + 1;
            end
        end

        // ---- (2) div=4: die active exactly 1-in-4; clk_en tracks the active slot ----
        div = 4; hold = 0; rst = 1; @(posedge clk); #1 rst = 0;
        @(negedge clk);                             // settle to the active slot (cnt=0)
        active = 0;
        for (i = 0; i < 40; i = i + 1) begin
            @(negedge clk);
            if (!throttle) active = active + 1;
            // SAFETY: the advancing cluster is enabled EXACTLY on the active (throttle=0) slot
            if (clk_en[0] !== ~throttle) begin
                $display("FAIL div=4: clk_en=%b throttle=%b (want clk_en==~throttle) @%0d",
                         clk_en[0], throttle, i);
                fails = fails + 1;
            end
        end
        if (active != 10) begin                     // 40 cycles / 4 = 10 active slots
            $display("FAIL div=4: %0d active slots in 40 cycles (expected 10)", active);
            fails = fails + 1;
        end else
            $display("div=4: %0d/40 active (f/4), clk_en==~throttle every cycle -- OK", active);

        // ---- (3) hold=1: throttle suppressed even at div=4 ----
        div = 4; hold = 1; rst = 1; @(posedge clk); #1 rst = 0;
        for (i = 0; i < 12; i = i + 1) begin
            @(negedge clk);
            if (throttle !== 1'b0) begin
                $display("FAIL hold: throttle=%b (expected 0 while held) @%0d", throttle, i);
                fails = fails + 1;
            end
        end

        // ---- (4) div=3: 1-in-3 active ----
        div = 3; hold = 0; rst = 1; @(posedge clk); #1 rst = 0;
        @(negedge clk);
        active = 0;
        for (i = 0; i < 30; i = i + 1) begin
            @(negedge clk);
            if (!throttle) active = active + 1;
        end
        if (active != 10) begin                     // 30/3 = 10
            $display("FAIL div=3: %0d active in 30 (expected 10)", active);
            fails = fails + 1;
        end else
            $display("div=3: %0d/30 active (f/3) -- OK", active);

        if (fails == 0) $display("ALL 4 TESTS PASSED  (clk_throttle: f/N eco duty-cycle, clk_en==~throttle, hold-safe)");
        else            $display("clk_throttle_tb: %0d FAIL(s)", fails);
        $finish;
    end
endmodule
