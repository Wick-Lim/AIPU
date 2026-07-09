`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// rmsnorm_unit.v  --  GLM-5.2 RMSNorm, high-performance pipelined  (§2,§8.4)
//----------------------------------------------------------------------------
// FUNCTION
//   y[i] = x[i] * rsqrt( (1/LEN) * Σ x[i]^2 + eps ) * gamma[i]
//
//   This is UNIT #1 of the GLM-5.2 decoder (ACCEL_GLM52 §8.4) and the canonical
//   place the numerics contract is locked: x and gamma stream in as BF16, the
//   Σx^2 REDUCE is performed in FP32 (mandatory -- a LEN=6144 bf16 sum loses
//   precision), the rsqrt is FP32, and y is rounded back to BF16 on output.
//   All FP arithmetic comes from glm_fp.vh (bf16<->fp32, fp32_mul/add, rsqrt).
//
//----------------------------------------------------------------------------
// PARAMETERS
//   LEN    : vector length (default 128; designed to scale to 6144).
//   LANES  : elements consumed/produced PER CYCLE (default 4).  Throughput
//            knob: the Σx^2 reduce uses a LANES-wide FP32 ADDER TREE each cycle,
//            and the normalize pass emits LANES bf16 outputs/cycle.
//            LEN must be a multiple of LANES.
//   EPS    : the +eps added inside the rsqrt, as an FP32 bit pattern.
//            Default 32'h3727C5AC = 1e-5 (GLM-5.2 RMSNorm eps).
//
//----------------------------------------------------------------------------
// INTERFACE  (clean start/stream/done handshake, deterministic latency)
//   clk, rst (synchronous, active-high).
//   start            : 1-cycle pulse to begin a new vector.  Captured; the unit
//                      then drives in_req for the reduce pass.
//   --- REDUCE pass (the unit pulls x, LANES/cycle) ---
//   in_req           : high while the unit wants the next x beat.
//   x_in   [LANES*16-1:0] : LANES bf16 elements of x (lane j = x_in[16*j +:16]).
//   x_valid          : producer asserts when x_in holds the requested beat.
//   --- NORMALIZE pass (the unit pulls gamma, replays x from its buffer) ---
//   g_req            : high while the unit wants the next gamma beat.
//   gamma_in [LANES*16-1:0] : LANES bf16 elements of gamma.
//   g_valid          : producer asserts when gamma_in holds the requested beat.
//   --- output (streamed, LANES/cycle, in input order) ---
//   y_valid          : high when y_out holds a valid output beat.
//   y_out  [LANES*16-1:0] : LANES bf16 elements of y.
//   --- status ---
//   busy             : high from start until done.
//   done             : 1-cycle pulse when the whole vector has been emitted.
//
//   x and gamma are pulled by the unit (in_req / g_req) and the producer
//   answers with x_valid / g_valid -- a simple ready/valid where the unit is
//   the consumer.  This lets the surrounding datapath (tile_memory reader, DMA)
//   feed beats at its own rate while the unit stays correct and latch-free.
//
//----------------------------------------------------------------------------
// PIPELINE / LATENCY
//   NBEATS = LEN / LANES.
//   Phase A  REDUCE   : NBEATS beats; each beat squares LANES lanes (fp32_mul),
//            sums them in a 1-cycle LANES-wide fp32 adder tree, and accumulates
//            into a running FP32 sum.  x is simultaneously written to an
//            internal buffer (BUF) for replay.
//   Phase B  RSQRT    : RSQRT_LAT cycles -- compute mean=sum/LEN (mul by 1/LEN),
//            add eps, fp32_rsqrt -> scalar inv (combinational rsqrt, registered
//            in 1 cycle here for timing; RSQRT_LAT = 2 pipe cycles).
//   Phase C  NORMALIZE: NBEATS beats; each beat reads BUF (x) and gamma, forms
//            y = bf16( (x*inv) * gamma ) per lane and emits a y beat.
//
//   Total latency (start -> done), assuming the producer answers every req
//   with valid on the next cycle (no stalls):
//       L = NBEATS (reduce) + RSQRT_LAT + NBEATS (normalize) + 1
//         = 2*(LEN/LANES) + RSQRT_LAT + 1            [RSQRT_LAT = 2]
//   With stalls, add one cycle per stalled beat.  THROUGHPUT (back-to-back
//   vectors, no stall) = LEN/LANES + RSQRT_LAT + const cycles/vector, i.e.
//   ~LANES elements/cycle in each streaming pass.
//
//----------------------------------------------------------------------------
// CORRECTNESS / STYLE
//   * Σx^2 reduce + mean + eps + rsqrt ALL in FP32 (numerics contract).
//   * Output rounded to bf16 (round-to-nearest-even) via glm_fp.fp32_to_bf16.
//   * Synchronous active-high reset; every reg assigned on every path (no
//     inferred latch); no combinational loop (rsqrt is feed-forward through the
//     glm_fp functions, registered between phases).
//============================================================================
module rmsnorm_unit #(
    parameter integer LEN   = 128,
    parameter integer LANES = 4,
    parameter [31:0]  EPS   = 32'h3727C5AC   // 1e-5 in fp32
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   start,
    // reduce-pass x input (unit pulls)
    output reg                    in_req,
    input  wire [LANES*16-1:0]    x_in,
    input  wire                   x_valid,
    // normalize-pass gamma input (unit pulls)
    output reg                    g_req,
    input  wire [LANES*16-1:0]    gamma_in,
    input  wire                   g_valid,
    // output stream
    output reg                    y_valid,
    output reg  [LANES*16-1:0]    y_out,
    // status
    output reg                    busy,
    output reg                    done
);
    // ---- derived sizes ----
    localparam integer NBEATS = LEN / LANES;
    localparam integer BAW    = (NBEATS <= 1) ? 1 : $clog2(NBEATS);
    localparam [BAW:0] LAST_BEAT = (BAW+1)'(NBEATS-1); // sized last-beat compare

    // 1/LEN as an fp32 constant, computed at elaboration by a pure-integer
    // reciprocal (synthesizable; constant-folded -- no `real`, yosys-friendly).
    localparam [31:0] INV_LEN = recip_fp32(LEN);

    // ---- x replay buffer: NBEATS lines of LANES*16 bits ----
    reg [LANES*16-1:0] buf_mem [0:NBEATS-1];

    // ---- FSM ----
    localparam [2:0] S_IDLE=3'd0, S_REDUCE=3'd1, S_RWAIT=3'd6, S_RS0=3'd2,
                     S_RS0B=3'd7, S_RS1=3'd3, S_NORM=3'd4, S_DONE=3'd5;
    reg [2:0]        state;
    reg [BAW:0]      beat;          // beat counter (one extra bit for == NBEATS)
    reg [31:0]       sumsq;         // fp32 running Σx^2
    reg [31:0]       inv;           // fp32 rsqrt result (scale)
    reg [31:0]       mean;          // fp32 sumsq/LEN      (S_RS0)
    reg [31:0]       meps;          // fp32 mean+eps       (S_RS0B)
    // rsqrt sequencer scratch (S_RS1 runs fp32_rsqrt one op per cycle)
    reg [3:0]        rs_step;
    reg [31:0]       rs_xh, rs_y, rs_yy, rs_xyy, rs_t;
    localparam [31:0] FP_HALF = 32'h3F000000, FP_3HALF = 32'h3FC00000;

    // ---- REDUCE PIPE (repipelined for fmax -- bit-exact) ----
    // The old single-cycle cone did LANES squares + a LANES-1-deep LEFT-CHAIN of
    // fp32 adds + the accumulate, all between two edges.  Now: one mul stage
    // (all lanes parallel), then ONE chain-add per stage -- the SAME left-assoc
    // grouping sq[0]+sq[1])+sq[2])+... as before, so sumsq is bit-identical --
    // then the accumulate add on the final tap.  RLAT = 1 + (LANES-1).
    integer k;
    reg [31:0] sq [0:LANES-1];
    always @* begin
        for (k = 0; k < LANES; k = k + 1) begin : LANE_SQ
            sq[k] = fp32_mul(bf16_to_fp32(x_in[16*k +: 16]),
                             bf16_to_fp32(x_in[16*k +: 16]));
        end
    end
    localparam integer RLAT = 1 + (LANES - 1);
    reg [RLAT-1:0] rvp;                                  // reduce valid taps
    reg [31:0] red_sq  [0:LANES-1];                      // P0: squares
    generate
    if (LANES > 1) begin : g_redchain
        reg [31:0] red_acc [1:LANES-1];                  // acc after adding lane i
        reg [31:0] red_fwd [1:LANES-1][0:LANES-1];       // sq lanes still pending
        integer ki;
        always @(posedge clk) begin
            if (rst) begin
                for (ki = 1; ki <= LANES-1; ki = ki + 1) red_acc[ki] <= 32'b0;
            end else begin
                if (rvp[0]) begin
                    red_acc[1] <= fp32_add(red_sq[0], red_sq[1]);
                    for (ki = 2; ki < LANES; ki = ki + 1)
                        red_fwd[1][ki] <= red_sq[ki];
                end
                for (ki = 2; ki <= LANES-1; ki = ki + 1)
                    if (rvp[ki-1]) begin : CHAIN
                        integer km;
                        red_acc[ki] <= fp32_add(red_acc[ki-1], red_fwd[ki-1][ki]);
                        for (km = ki+1; km < LANES; km = km + 1)
                            red_fwd[ki][km] <= red_fwd[ki-1][km];
                    end
            end
        end
        // final chain value, consumed by the accumulate below
        wire [31:0] red_final = red_acc[LANES-1];
    end else begin : g_redchain
        wire [31:0] red_final = red_sq[0];
    end
    endgenerate

    // ---- combinational normalize of the current buffered beat ----
    // y_lane = bf16( (x_lane * inv) * gamma_lane )
    reg [LANES*16-1:0] norm_beat;
    reg [LANES*16-1:0] buf_rd;
    reg [31:0]         xs;
    always @* begin
        buf_rd    = buf_mem[beat[BAW-1:0]];
        norm_beat = {LANES*16{1'b0}};
        for (k = 0; k < LANES; k = k + 1) begin : LANE_NORM
            xs = fp32_mul(bf16_to_fp32(buf_rd[16*k +: 16]), inv);
            norm_beat[16*k +: 16] =
                fp32_to_bf16(fp32_mul(xs, bf16_to_fp32(gamma_in[16*k +: 16])));
        end
    end

    // ---- sequential control ----
    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            in_req  <= 1'b0;
            g_req   <= 1'b0;
            y_valid <= 1'b0;
            y_out   <= {LANES*16{1'b0}};
            busy    <= 1'b0;
            done    <= 1'b0;
            beat    <= {(BAW+1){1'b0}};
            sumsq   <= 32'b0;
            inv     <= 32'b0;
            mean    <= 32'b0;
            meps    <= 32'b0;
            rvp     <= {RLAT{1'b0}};
            rs_step <= 4'd0;
            rs_xh <= 32'b0; rs_y <= 32'b0; rs_yy <= 32'b0;
            rs_xyy <= 32'b0; rs_t <= 32'b0;
            for (k = 0; k < LANES; k = k + 1) red_sq[k] <= 32'b0;
        end else begin
            // defaults (every reg gets a value every cycle -> no latch)
            done    <= 1'b0;
            y_valid <= 1'b0;
            in_req  <= 1'b0;
            g_req   <= 1'b0;

            // ---- reduce pipe front + accumulate (state-independent taps) ----
            // shift-left form (no part-select) so RLAT==1 (LANES==1) elaborates
            rvp <= (rvp << 1) | {{(RLAT-1){1'b0}}, (state == S_REDUCE) && x_valid};
            if ((state == S_REDUCE) && x_valid)
                for (k = 0; k < LANES; k = k + 1) red_sq[k] <= sq[k];
            if (rvp[RLAT-1])
                sumsq <= fp32_add(sumsq, g_redchain.red_final);

            case (state)
                // -------------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy   <= 1'b1;
                        beat   <= {(BAW+1){1'b0}};
                        sumsq  <= 32'b0;
                        in_req <= 1'b1;          // request first x beat
                        state  <= S_REDUCE;
                    end
                end
                // -------------------------- REDUCE ---------------------
                //   beats stream through the reduce pipe (1/cycle); the
                //   accumulate happens on the pipe's final tap above, so the
                //   FSM only counts accepted beats then waits for the drain.
                S_REDUCE: begin
                    in_req <= 1'b1;              // keep asking until accepted
                    if (x_valid) begin
                        buf_mem[beat[BAW-1:0]] <= x_in;
                        if (beat == LAST_BEAT) begin
                            in_req <= 1'b0;
                            beat   <= {(BAW+1){1'b0}};
                            state  <= S_RWAIT;
                        end else begin
                            beat <= beat + 1'b1;
                        end
                    end
                end
                S_RWAIT:                          // drain: last accumulate is
                    if (rvp == {RLAT{1'b0}}) state <= S_RS0;   // one edge after
                // -------------------------- RSQRT ----------------------
                //   mean = sumsq/LEN and meps = mean+eps, one fp op per state
                //   (the old S_RS0 chained mul+add in one cycle), then the
                //   Quake rsqrt runs ONE op per cycle (rs_step 0..9) -- the
                //   exact fp32_rsqrt call sequence incl. its special-case
                //   early returns, so `inv` is bit-identical.
                S_RS0: begin
                    mean  <= fp32_mul(sumsq, INV_LEN);
                    state <= S_RS0B;
                end
                S_RS0B: begin
                    meps    <= fp32_add(mean, EPS);
                    rs_step <= 4'd0;
                    state   <= S_RS1;
                end
                S_RS1: begin
                    rs_step <= rs_step + 4'd1;
                    case (rs_step)
                        4'd0: begin
                            // fp32_rsqrt specials, tested on the same input:
                            if ((meps[30:23] == 8'hFF && meps[22:0] != 23'b0) ||
                                meps[31] == 1'b1 || meps[30:23] == 8'b0) begin
                                inv   <= 32'h7FC00000;       // nan (x<=0/nan)
                                g_req <= 1'b1;
                                state <= S_NORM;
                            end else if (meps[30:23] == 8'hFF) begin
                                inv   <= 32'h00000000;       // 1/sqrt(inf)=+0
                                g_req <= 1'b1;
                                state <= S_NORM;
                            end else begin
                                rs_xh <= fp32_mul(FP_HALF, meps);
                                rs_y  <= 32'h5F3759DF - (meps >> 1);
                            end
                        end
                        4'd1: rs_yy  <= fp32_mul(rs_y, rs_y);
                        4'd2: rs_xyy <= fp32_mul(rs_xh, rs_yy);
                        4'd3: rs_t   <= fp32_add(FP_3HALF, {rs_xyy[31]^1'b1, rs_xyy[30:0]});
                        4'd4: rs_y   <= fp32_mul(rs_y, rs_t);
                        4'd5: rs_yy  <= fp32_mul(rs_y, rs_y);
                        4'd6: rs_xyy <= fp32_mul(rs_xh, rs_yy);
                        4'd7: rs_t   <= fp32_add(FP_3HALF, {rs_xyy[31]^1'b1, rs_xyy[30:0]});
                        4'd8: rs_y   <= fp32_mul(rs_y, rs_t);
                        default: begin           // step 9: done
                            inv   <= rs_y;
                            g_req <= 1'b1;       // request first gamma beat
                            state <= S_NORM;
                        end
                    endcase
                end
                // ------------------------- NORMALIZE -------------------
                S_NORM: begin
                    g_req <= 1'b1;
                    if (g_valid) begin
                        y_out   <= norm_beat;
                        y_valid <= 1'b1;
                        if (beat == LAST_BEAT) begin
                            g_req <= 1'b0;
                            state <= S_DONE;
                        end else begin
                            beat <= beat + 1'b1;
                        end
                    end
                end
                // --------------------------- DONE ----------------------
                S_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    //------------------------------------------------------------------------
    // recip_fp32 : elaboration-time PURE-INTEGER reciprocal -> fp32 bit pattern
    // of 1/N for a positive integer N.  No `real` (so yosys's Verilog frontend
    // accepts it); evaluated once for the INV_LEN localparam and constant-folded
    // -- never inferred as logic.
    //
    // Method: long-divide  (1 << SH) / N  to get a 64-bit fixed-point quotient
    // with SH fractional bits, find its MSB (=> the binary exponent of 1/N),
    // extract 23 mantissa bits + guard, and round-to-nearest-even into fp32.
    //------------------------------------------------------------------------
    function automatic [31:0] recip_fp32(input integer N);
        localparam integer SH = 60;          // fractional bits of the quotient
        reg [63:0] q;                        // (1<<SH)/N
        // qs holds q normalized with the implicit leading 1 at bit 63; bit 63 is
        // therefore always 1 and intentionally never read (it is the implicit
        // bit, not stored in the mantissa) -- waive the unused-bit lint on it.
        /* verilator lint_off UNUSEDSIGNAL */
        reg [63:0] qs;
        /* verilator lint_on UNUSEDSIGNAL */
        integer    msb, i, shl;
        reg [7:0]  expo;
        reg [22:0] mant;
        reg        guard, sticky, roundup;
        reg [24:0] mant_r;
        begin
            if (N <= 0) recip_fp32 = 32'b0;
            else begin
                q = (64'd1 << SH) / {32'd0, N}; // value = q * 2^-SH = 1/N
                // find MSB position of q
                msb = -1;
                for (i = 0; i < 64; i = i + 1)
                    if (q[i]) msb = i;
                if (msb < 0) recip_fp32 = 32'b0;  // shouldn't happen for N>=1
                else begin
                    // 1/N = 1.m * 2^(msb - SH).  fp32 exponent = (msb-SH)+127.
                    // (in range [67,130] for N in [1, 2^60] -> 8-bit cast safe)
                    expo = 8'(((msb - SH) + 127));
                    // left-shift so the leading 1 falls OFF the top (bit 62+)
                    // and only the fraction remains, MSB-aligned at bit 61.
                    shl  = 63 - msb;            // put the leading 1 at bit 63
                    qs   = q << shl;            // qs[63]=implicit 1 (unread)
                    mant   = qs[62:40];         // 23 mantissa bits below the 1
                    guard  = qs[39];            // round bit
                    sticky = |qs[38:0];         // sticky OR of the rest
                    // round-to-nearest-even
                    roundup = guard & (sticky | mant[0]);
                    mant_r  = {1'b0, mant} + {24'b0, roundup};
                    if (mant_r[24]) begin
                        mant_r = mant_r >> 1;
                        expo   = expo + 8'd1;
                    end
                    recip_fp32 = {1'b0, expo, mant_r[22:0]};
                end
            end
        end
    endfunction
endmodule
