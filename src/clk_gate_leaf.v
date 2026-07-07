`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// clk_gate_leaf.v  --  a small self-contained clocked register bank used as the
//                      GATED LEAF in the clock-gating equivalence proof.
//----------------------------------------------------------------------------
// WHY THIS EXISTS
//   `clk_gate_cluster` proves that a datapath leaf clocked by the ICG-gated clock
//   produces bit-identical state to the same leaf on the free-running clock.  It
//   needs SOME representative synchronous leaf to gate; this is that leaf -- a
//   tiny 16 x 32-bit register file (three combinational read ports, one
//   synchronous write, hardwired-zero index 0, synchronous reset).  It is
//   deliberately self-contained (numeric parameter defaults, no `include) so the
//   GLM prove-it gate has NO dependency outside src/.  The clock-gating property
//   is independent of which leaf is used; this one is convenient and fully reset
//   (no X after reset), so the equivalence check is deterministic.
//
// INTERFACE (unchanged from the earlier register_file leaf, so the cluster + TB
//   instantiate it by name only):
//   clk, rst                       : clock, synchronous active-high reset.
//   read_addr1/2/3 [IDX_W-1:0]     : three independent combinational read indices.
//   write_addr     [IDX_W-1:0]     : write index (ignored when == 0).
//   write_data     [WORD_W-1:0]    : write payload.
//   write_enable                   : commit the write on the next posedge.
//   read_data1/2/3 [WORD_W-1:0]    : combinational read results (0 for index 0).
//============================================================================
module clk_gate_leaf #(
    parameter integer REGS   = 16,   // number of registers
    parameter integer IDX_W  = 4,    // register index width ($clog2(REGS))
    parameter integer WORD_W = 32    // scalar word width
) (
    input  wire              clk,
    input  wire              rst,
    input  wire [IDX_W-1:0]  read_addr1,
    input  wire [IDX_W-1:0]  read_addr2,
    input  wire [IDX_W-1:0]  read_addr3,
    input  wire [IDX_W-1:0]  write_addr,
    input  wire [WORD_W-1:0] write_data,
    input  wire              write_enable,
    output wire [WORD_W-1:0] read_data1,
    output wire [WORD_W-1:0] read_data2,
    output wire [WORD_W-1:0] read_data3
);
    // Architectural state.  registers[0] is hardwired-zero: never written (write
    // enable is gated) and reads of it are forced to 0 below.
    reg [WORD_W-1:0] registers [0:REGS-1];
    integer i;

    // ----- Combinational read ports with index-0 hardwired to zero -----
    assign read_data1 = (read_addr1 == {IDX_W{1'b0}}) ? {WORD_W{1'b0}}
                                                      : registers[read_addr1];
    assign read_data2 = (read_addr2 == {IDX_W{1'b0}}) ? {WORD_W{1'b0}}
                                                      : registers[read_addr2];
    assign read_data3 = (read_addr3 == {IDX_W{1'b0}}) ? {WORD_W{1'b0}}
                                                      : registers[read_addr3];

    // ----- Synchronous reset + synchronous, index-0-protected write -----
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < REGS; i = i + 1) begin
                registers[i] <= {WORD_W{1'b0}};
            end
        end else if (write_enable && (write_addr != {IDX_W{1'b0}})) begin
            registers[write_addr] <= write_data;
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
