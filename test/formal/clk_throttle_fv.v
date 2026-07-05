//============================================================================
// clk_throttle_fv.v -- formal harness for the DVFS/eco frequency prescaler.
//   Proves (SAFETY, via yosys-smtbmc):
//     P1  div <= 1  => throttle == 0     (the byte-identical / no-throttle guarantee)
//     P2  hold      => throttle == 0     (never throttle while the die is already idle)
//     P3  throttle is high for at most div-1 CONSECUTIVE cycles => the die gets an
//         active slot at least once every `div` cycles (NO STARVATION / deadlock).
//   Environment: reset at t=0, then rst held low; `div` held stable and bounded
//   (div <= 6, so a K=16 BMC covers the whole period) -- standard formal scoping.
//   DIVW is chparam'd small by the Makefile (run_bmc extra-yosys) for a small state.
//============================================================================
module clk_throttle_fv #(
    parameter integer DIVW = 4
) (
    input wire            clk,
    // free formal stimulus (DUT inputs)
    input wire            rst,
    input wire [DIVW-1:0] div,
    input wire            hold
);
    wire throttle;

    clk_throttle #(.DIVW(DIVW)) dut (
        .clk(clk), .rst(rst), .div(div), .hold(hold), .throttle(throttle));

    // cycle counter: constrain the reset + stability environment
    reg [4:0] cyc = 0;
    always @(posedge clk) if (cyc != 5'h1f) cyc <= cyc + 1'b1;

    // consecutive-high run length of `throttle` (P3 tracker)
    reg [DIVW:0] hi_run = 0;
    always @(posedge clk) begin
        if (rst)         hi_run <= 0;
        else if (throttle) hi_run <= hi_run + 1'b1;
        else             hi_run <= 0;
    end

    always @(posedge clk) begin
        // ---- environment assumptions ----
        if (cyc == 0) assume (rst);                 // reset at t=0
        else begin
            assume (!rst);                          // then free-run, no reset
            assume (div == $past(div));             // div held stable over the run
        end
        assume (div <= 6);                          // bounded divisor (K=16 covers it)

        // ---- properties (checked once out of reset) ----
        if (!rst && cyc != 0) begin
            if (div <= {{(DIVW-1){1'b0}}, 1'b1})    // div <= 1
                assert (throttle == 1'b0);          // P1: no throttle => byte-identical
            if (hold)
                assert (throttle == 1'b0);          // P2: parked while already idle
            // P3: never throttled for `div` consecutive cycles (active slot recurs)
            assert (hi_run < div || div == 0);
        end
    end
endmodule
