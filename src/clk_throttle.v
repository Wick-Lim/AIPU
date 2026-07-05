`timescale 1ns/1ps
//============================================================================
// clk_throttle.v  --  DVFS / eco duty-cycle prescaler for the FP8 die
//----------------------------------------------------------------------------
// PURPOSE
//   Realize the "compute is nearly free on a Flash-bound die" slack
//   (docs/LOW_POWER.md, MINIATURIZATION.md) as a CONCRETE, byte-identical knob:
//   run the compute die at an effective f/`div` by handing `clk_en_ctrl` a
//   `throttle` term that idles the die (div-1) of every `div` cycles. The single
//   active slot per window still advances the exact same state, so the decoded
//   token is UNCHANGED (throttle reuses clk_en_ctrl's proven stall-gate path).
//
// WHAT IT BUYS (honest -- see docs/LOW_POWER.md):
//   * PEAK POWER / thermal cap: dynamic power scales ~1/div (fewer active edges
//     per unit time) -> lets the die/USB-C box hold a lower power envelope (the
//     product plan's "eco/40 W mode"). This is the RTL-realizable part of DVFS.
//   * It does NOT reduce energy-per-token (J/token): the switching-event COUNT is
//     unchanged, only spread over more time. The J/token part of DVFS is VOLTAGE
//     scaling, which is a physical/vendor step (not RTL).
//   * FREE in throughput only while the die stays inside the Flash-stall shadow
//     (div <= the compute-slowdown budget, ~4-5x); past that, tok/s drops.
//
// USAGE
//   div <= 1  : throttle stays 0  -> NO throttle (default, byte-identical).
//   div  = N  : the die advances 1 cycle in N  -> effective f/N.
//   hold      : tie to (boot_active | stall) so the prescaler does NOT spend its
//               active slot while the die is already idle (keeps the duty cycle
//               honest); tie 0 if unused.
//
//   Synchronous ACTIVE-HIGH reset. No latch, no combinational loop (throttle is a
//   pure function of the counter + inputs).
//============================================================================
module clk_throttle #(
    parameter integer DIVW = 8               // divisor register width (max div = 2^DIVW-1)
) (
    input  wire            clk,
    input  wire            rst,              // synchronous, ACTIVE-HIGH
    input  wire [DIVW-1:0] div,              // eco divisor (die runs 1-in-div); <=1 = off
    input  wire            hold,             // 1 = die already idle: freeze + don't throttle
    output wire            throttle          // 1 = force the die idle this cycle
);
    localparam [DIVW-1:0] ONE = {{(DIVW-1){1'b0}}, 1'b1};

    // enabled only when div >= 2 (div 0/1 => no throttle)
    wire enabled = (div > ONE);

    // free-running mod-div counter; the die's ONE active slot is cnt == 0.
    reg [DIVW-1:0] cnt;
    always @(posedge clk) begin
        if (rst)                     cnt <= {DIVW{1'b0}};
        else if (!enabled || hold)   cnt <= {DIVW{1'b0}};   // frozen/parked at the active slot
        else if (cnt >= div - ONE)   cnt <= {DIVW{1'b0}};   // wrap after div-1
        else                         cnt <= cnt + ONE;
    end

    // throttle every cycle except the active slot; suppressed while enabled=0 or the
    // die is already idle (the external stall/boot gate covers those).
    assign throttle = enabled & ~hold & (cnt != {DIVW{1'b0}});
endmodule
