`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// mbist_ctrl_2p.v  --  March C- + concurrent-access BIST for a TRUE-DUAL-PORT
//                       (1R1W, separate addresses) on-die RAM        (DFT / C7)
//----------------------------------------------------------------------------
// WHY THIS EXISTS (the 2-port collar gap)
//   The single-port engine `mbist_ctrl.v` drives ONE shared address / one access
//   per cycle.  The production stores that actually hold model state are NOT
//   single-port: `kv_cache_pager.ring` reads on a gather address WHILE writing on
//   an append address the same cycle, and `mla_attn_q4k.vstore_mem` writes on one
//   attention beat and reads on another (2-port at PE_M>1).  A single-port March
//   engine cannot exercise them without serializing the ports -- which HIDES the
//   one fault class unique to a dual-port array: a write on port B disturbing a
//   CONCURRENT read on port A.  This engine drives the two ports independently and
//   adds a concurrent write+read phase, so it tests what the single-port engine
//   structurally cannot.  (In silicon the memory compiler emits the real per-macro
//   dual-port BIST collar; this is the verified RTL reference for that collar --
//   see docs/P2_MEMORY_MAP.md sec 4.)
//
// PORT MODEL (matches ring / vstore: async read, sync write, separate addresses)
//   Write port : waddr / we / wdata  -- registered; the RAM latches on the edge.
//   Read  port : raddr / rdata       -- rdata = RAM[raddr], COMBINATIONAL; because
//                raddr is a REGISTERED output, rdata is valid the cycle AFTER raddr
//                is driven (same 1-deep read pipeline as mbist_ctrl.v).
//   A write on port B and a read on port A may be driven the SAME cycle.
//
// TWO PHASES
//   PHASE M -- March C- (10N), cells tested through the R/W port PAIR:
//        M0 up  : w0            M1 up  : r0,w1     M2 up  : r1,w0
//        M3 dn  : r0,w1         M4 dn  : r1,w0     M5 dn  : r0
//     each op is a read (port A) OR a write (port B); the storage array is
//     exercised through both physical ports.  A read that mismatches -> FAIL,
//     fail_kind=0 (STUCK/transition/coupling in the cell array), aborts.
//   PHASE C -- concurrent write+read (the dual-port-specific class).  After M
//     the array is all-0.  A protected SENTINEL cell (address 0) is held at 0;
//     for every aggressor a in 1..DEPTH-1 the engine, IN ONE CYCLE, WRITES all-1
//     to waddr=a AND READS raddr=0, and checks the sentinel still reads 0.  A
//     write that couples into the concurrent read of another cell -> FAIL,
//     fail_kind=1.  (Representative dual-port coupling check via a fixed victim;
//     the compiler's production collar sweeps full aggressor/victim pairs.)
//
// OUTPUTS (all registered; RAM strobes too -> glitch-free)
//   busy/done/fail : as mbist_ctrl.v (done is a latched LEVEL, not a pulse).
//   fail_addr      : aggressor (phase C) or failing read address (phase M).
//   fail_kind      : 0 = march (cell array), 1 = concurrent port coupling.
//
// CONVENTIONS (match mbist_ctrl.v): sync active-high reset clears ALL state;
//   every output registered and assigned on every path (no inferred latch);
//   no combinational loop; fully parameterized.
//============================================================================
module mbist_ctrl_2p #(
    parameter integer DEPTH = 16,   // # RAM cells (addresses)
    parameter integer WIDTH = 8,    // bits per word
    localparam integer AW = (DEPTH < 2) ? 1 : $clog2(DEPTH)
) (
    input  wire              clk,
    input  wire              rst,        // sync, active-high (ALL state)

    // ---- command / status ----
    input  wire              start,      // 1-cycle pulse: begin a run
    output reg               busy,
    output reg               done,       // LEVEL: finished (pass or fail)
    output reg               fail,       // LEVEL: latched on first failure
    output reg               fail_kind,  // 0 = march (cells), 1 = port coupling
    output reg  [AW-1:0]     fail_addr,

    // ---- write port (sync) ----
    output reg  [AW-1:0]     waddr,
    output reg               we,
    output reg  [WIDTH-1:0]  wdata,

    // ---- read port (async: rdata = RAM[raddr], valid the cycle after raddr) ----
    output reg  [AW-1:0]     raddr,
    input  wire [WIDTH-1:0]  rdata
);
    // ---- March C- program (identical encoding to mbist_ctrl.v) ----
    localparam integer NOPS = 10;
    localparam integer OPW  = (NOPS < 2) ? 1 : $clog2(NOPS);
    localparam [NOPS-1:0] DIR_UP = 10'b00_0001_1111;   // ops 0..4 sweep UP
    localparam [NOPS-1:0] IS_RD  = 10'b10_1010_1010;   // ops 1,3,5,7,9 read
    localparam [NOPS-1:0] PAT1   = 10'b00_1100_1100;   // ops 2,3,6,7 all-ones

    localparam [WIDTH-1:0] WORD0 = {WIDTH{1'b0}};
    localparam [WIDTH-1:0] WORD1 = {WIDTH{1'b1}};

    localparam integer  CW      = AW + 1;
    localparam [CW-1:0] C_ONE   = { {(CW-1){1'b0}}, 1'b1 };
    localparam [CW-1:0] C_ZERO  = { CW{1'b0} };
    localparam [CW-1:0] DEPTHC  = CW'(DEPTH);
    localparam [CW-1:0] DEPTHM1 = DEPTHC - C_ONE;

    // ---- run state ----
    // phase: 0 = MARCH, 1 = COUPLING, 2 = DONE
    localparam [1:0] P_MARCH = 2'd0, P_COUPLE = 2'd1, P_DONE = 2'd2;
    reg [1:0]    phase;
    reg [OPW:0]  opi;     // march op index (0..NOPS)
    reg [CW-1:0] cnt;     // addresses issued in the current march op
    reg          run;

    // one-deep pending-read record (read issued last cycle, compared now)
    reg          rd_pend;
    reg          rd_pat1;   // expected pattern (1 = all-ones)  [march phase]
    reg          rd_couple; // this pending read is a coupling (sentinel) read
    reg [AW-1:0] rd_addr;   // address for fail_addr

    // ---- combinational view of the march op about to drive this cycle ----
    wire           active   = run & (phase == P_MARCH) & (opi < OPW'(NOPS));
    wire [OPW-1:0] oidx     = opi[OPW-1:0];
    wire           cur_up   = active ? DIR_UP[oidx] : 1'b0;
    wire           cur_rd   = active ? IS_RD [oidx] : 1'b0;
    wire           cur_pat1 = active ? PAT1  [oidx] : 1'b0;
    wire [CW-1:0]  addr_c   = cur_up ? cnt : (DEPTHM1 - cnt);
    wire           has_addr = active & (cnt < DEPTHC);

    // coupling phase: aggressor sweep 1..DEPTH-1 (cnt counts 0..DEPTH-2)
    wire           cpl_active = run & (phase == P_COUPLE);
    wire [CW-1:0]  aggr       = cnt + C_ONE;              // aggressor address (>=1)
    wire           cpl_has    = cpl_active & (aggr < DEPTHC);

    // pending read compare: march expects rd_pat1; coupling expects WORD0 (sentinel)
    wire           mismatch = rd_pend &
                              (rdata !== (rd_couple ? WORD0
                                                    : (rd_pat1 ? WORD1 : WORD0)));

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0; done <= 1'b0; fail <= 1'b0; fail_kind <= 1'b0;
            fail_addr <= {AW{1'b0}};
            waddr <= {AW{1'b0}}; we <= 1'b0; wdata <= WORD0;
            raddr <= {AW{1'b0}};
            phase <= P_MARCH; opi <= {(OPW+1){1'b0}}; cnt <= C_ZERO; run <= 1'b0;
            rd_pend <= 1'b0; rd_pat1 <= 1'b0; rd_couple <= 1'b0; rd_addr <= {AW{1'b0}};
        end else if (start & ~busy) begin
            busy <= 1'b1; done <= 1'b0; fail <= 1'b0; fail_kind <= 1'b0;
            fail_addr <= {AW{1'b0}};
            waddr <= {AW{1'b0}}; we <= 1'b0; wdata <= WORD0;
            raddr <= {AW{1'b0}};
            phase <= P_MARCH; opi <= {(OPW+1){1'b0}}; cnt <= C_ZERO; run <= 1'b1;
            rd_pend <= 1'b0; rd_pat1 <= 1'b0; rd_couple <= 1'b0; rd_addr <= {AW{1'b0}};
        end else if (busy) begin
            we <= 1'b0;                                  // default; no inferred latch

            if (mismatch) begin
                // first failure : latch fail (+kind) + address, abort
                fail      <= 1'b1;
                fail_kind <= rd_couple;                  // 1 = coupling, 0 = march
                fail_addr <= rd_addr;
                run <= 1'b0; busy <= 1'b0; done <= 1'b1; rd_pend <= 1'b0;
            end else if (phase == P_MARCH) begin
                if (has_addr) begin
                    // issue the march op: read on port A, or write on port B
                    raddr   <= cur_rd ? addr_c[AW-1:0] : raddr;
                    waddr   <= ~cur_rd ? addr_c[AW-1:0] : waddr;
                    we      <= ~cur_rd;
                    wdata   <= cur_pat1 ? WORD1 : WORD0;
                    rd_pend  <= cur_rd;
                    rd_pat1  <= cur_pat1;
                    rd_couple<= 1'b0;
                    rd_addr  <= addr_c[AW-1:0];
                    if (cnt == DEPTHM1) begin
                        cnt <= C_ZERO;
                        opi <= opi + {{OPW{1'b0}}, 1'b1};
                    end else begin
                        cnt <= cnt + C_ONE;
                    end
                end else begin
                    // march program exhausted (final read compared clean this cycle)
                    // -> enter the concurrent coupling phase; sentinel cell 0 is 0.
                    rd_pend <= 1'b0;
                    phase   <= P_COUPLE;
                    cnt     <= C_ZERO;
                end
            end else if (phase == P_COUPLE) begin
                if (cpl_has) begin
                    // ONE cycle: WRITE all-1 to aggressor on port B AND READ the
                    // sentinel (cell 0) on port A.  rd compared next cycle == 0.
                    waddr    <= aggr[AW-1:0];
                    we       <= 1'b1;
                    wdata    <= WORD1;
                    raddr    <= {AW{1'b0}};              // sentinel address
                    rd_pend  <= 1'b1;
                    rd_couple<= 1'b1;
                    rd_addr  <= aggr[AW-1:0];            // the aggressor, for fail_addr
                    cnt      <= cnt + C_ONE;
                end else begin
                    // aggressor sweep done, last sentinel read compared clean
                    // -> the run passed.
                    rd_pend <= 1'b0;
                    run <= 1'b0; busy <= 1'b0; done <= 1'b1; phase <= P_DONE;
                end
            end
        end
    end

endmodule
/* verilator lint_on DECLFILENAME */
