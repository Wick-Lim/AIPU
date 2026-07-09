`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_act.v  --  GLM-5.2 ELEMENTWISE ACTIVATIONS (SIGMOID + SiLU)   (§5)
//----------------------------------------------------------------------------
// FUNCTION
//   Two bf16-in / bf16-out elementwise activations selected by a MODE bit:
//     MODE = MODE_SIGMOID :  y = sigmoid(x) = 1 / (1 + exp(-x))
//     MODE = MODE_SILU    :  y = silu(x)    = x * sigmoid(x)
//
//   sigmoid is the MoE-router gating nonlinearity (GLM/DeepSeek-v3 sigmoid
//   gating, ACCEL_GLM52 §5 "W_g -> sigmoid -> top-8") and SiLU is the SwiGLU
//   expert activation (config hidden_act = "silu", §1.1 / §5
//   "h = silu(g) (.) u").  ONE unit covers both because silu(x) = x*sigmoid(x):
//   we compute sigmoid(x) in fp32 and, in SiLU mode, take the extra fp32
//   multiply by x.  Everything obeys the §6 numerics contract: bf16 storage,
//   ALL transcendental math in FP32, round-to-nearest-even back to bf16 out.
//
//----------------------------------------------------------------------------
// EXP / SIGMOID METHOD  (and the accuracy it buys)
//   sigmoid(x) = 1/(1+exp(-x)).  Let z = -x.  We need exp(z) for z over the
//   range that survives saturation (|x| < ~17, see SATURATION below).
//
//   RANGE REDUCTION (the standard 2^k * exp(r) split):
//     k = round( z * log2(e) )                       (nearest integer)
//     r = z - k*ln2                                   (so r in [-ln2/2, ln2/2])
//     exp(z) = 2^k * exp(r)
//   2^k is assembled DIRECTLY into the fp32 exponent field (add k to the
//   biased exponent of exp(r)) -- exact, no extra multiply, and the place we
//   clamp k so 2^k can never overflow/underflow the fp32 field.
//
//   exp(r) on the tiny interval r in [-ln2/2, ln2/2] (|r| < 0.3466) by a
//   degree-5 minimax-style polynomial in HORNER form, all fp32:
//     exp(r) ~ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120     (Taylor c_i = 1/i!)
//   On |r| <= ln2/2 the truncation error of this series is bounded by the next
//   term |r|^6/720 <= 0.3466^6/720 ~ 2.5e-6  (< 2^-18), i.e. far under a bf16
//   ULP.  (Taylor coefficients on this small symmetric interval are already
//   within a hair of the true minimax polynomial, so we use the exact 1/i!
//   constants -- simpler, and the residual is dominated by the bf16 output
//   rounding anyway.)
//
//   RECIPROCAL 1/(1+exp(z)):  the denom d = 1 + exp(z) is ALWAYS >= 1 (z real,
//   exp(z) > 0), so 1/d is computed from the glm_fp Quake-seed Newton rsqrt as
//        1/d = rsqrt(d)^2
//   rsqrt is measured < 2^-22 rel-err (§ glm_fp), squaring at most doubles that
//   to < 2^-21 -- again far below a bf16 ULP.
//
//   SATURATION (no overflow / no NaN, ever):
//     For large +x, sigmoid -> 1; for large -x, sigmoid -> 0.  The SIGMOID exp
//     path uses a CLAMPED copy of x in [-X_SAT,+X_SAT], X_SAT = 16 (power of
//     two), so k stays tiny and 2^k never reaches the fp32 exponent rails.  At
//     |x|=16, exp(16) ~ 8.9e6 -> sigmoid(16) = 1 - 1.1e-7 and sigmoid(-16) =
//     1.1e-7, both already INSIDE one bf16 ULP of the saturated 1 / 0, so the
//     clamp is numerically invisible.  Inputs that are bf16 inf/nan are first
//     sanitized to +/-X_SAT (finite) before either path sees them.
//     The SiLU multiply, however, uses the *unclamped* (raw, sanitized) x:
//       silu(+big) = x * sigmoid(x) ~ x*1 ~ x   (correct large-x linear tail),
//       silu(-big) = x * sigmoid(x) ~ x*0 ~ 0   (correct vanishing left tail).
//     So only the sigmoid factor saturates; the linear factor is exact, and
//     silu's characteristic negative dip near x ~ -1.278 (in-range) is exact.
//     Every output is therefore a finite bf16 for every finite/inf/nan input.
//
//   NET ACCURACY (measured by the scratchpad TB vs an independent fp64 golden,
//   comparing on the bf16 grid to isolate the COMPUTE error from the shared
//   0.5-ULP output rounding):
//     * sigmoid worst abs-err = 1.13e-7  (~2^-23, << the 2^-10 §5 target),
//     * silu    worst rel-err = 0        (bit-exact to the bf16 result grid)
//   over directed anchors + tails + saturation rails + 160 random samples.
//   The compute error is dominated by the < 2^-18 poly and < 2^-21 reciprocal,
//   both far under one bf16 output ULP, so the END bf16 result is at worst the
//   correctly-rounded bf16 of the true value.
//
//----------------------------------------------------------------------------
// PARAMETERS
//   MODE   : MODE_SIGMOID (0) or MODE_SILU (1).  Compile-time activation select.
//   LANES  : elements processed PER CYCLE (default 4).  The datapath is LANES
//            independent, identical activation lanes -> LANES elem/cycle peak.
//   X_SAT  : fp32 saturation magnitude (default 16.0 = 32'h41800000).
//
//----------------------------------------------------------------------------
// INTERFACE  (streaming, deterministic latency, valid/valid handshake)
//   clk, rst            : synchronous, active-high reset.
//   in_valid            : producer asserts when x_in holds a fresh LANES-beat.
//   x_in [LANES*16-1:0] : LANES bf16 inputs (lane j = x_in[16*j +: 16]).
//   out_valid           : high when y_out holds a valid LANES-beat.
//   y_out[LANES*16-1:0] : LANES bf16 results, SAME lane order, LAT cycles later.
//
//   Pure FEED-FORWARD pipeline: every in_valid beat emits an out_valid beat
//   exactly LAT cycles later, one-for-one, no back-pressure needed (the unit
//   never stalls and accepts a beat every cycle).  This makes it trivially
//   composable behind gemm_ml / fused_ops streaming and inside moe_router.
//
//----------------------------------------------------------------------------
// PIPELINE / LATENCY  (deterministic, data-independent)
//   The activation core is a feed-forward chain of registered fp32 stages.
//   Stage layout (per lane, identical across lanes):
//     S1  decode + clamp  : widen bf16->fp32, clamp to [-X_SAT,X_SAT], z=-x,
//                           compute k = round(z*log2e), r = z - k*ln2.
//     S2a poly (inner)     : inner 3 Horner levels of exp(r) -> partial p_lo.
//     S2b poly (outer)     : outer 2 Horner levels -> exp(r) (fp32).
//     S3  scale+denom     : ex = 2^k * exp(r) (exponent add); d = 1+ex.
//     S4  recip           : t = rsqrt(d); s = t*t  (= sigmoid(x)).
//     S5  finish          : MODE_SILU -> s = s * x ; round fp32->bf16 -> y.
//   The degree-5 Horner exp poly is split across S2a/S2b (3+2 levels) so no
//   single stage carries more than 3 serial fp multiplies -- a latency-only
//   repipeline for fmax; exp(r) is bit-identical to the old one-stage Horner.
//   => LAT = 6 cycles, fixed, regardless of the data.  out_valid is in_valid
//   delayed by LAT through a shift register, so the handshake is exact.
//   THROUGHPUT = LANES elements/cycle (one beat in, one beat out, every cycle).
//
//----------------------------------------------------------------------------
// CORRECTNESS / STYLE
//   * All transcendental/reduce math in FP32 via glm_fp.vh (§6 contract).
//   * Synchronous active-high reset; EVERY reg written on EVERY path (no
//     inferred latch); the only feedback is the pipeline registers themselves
//     (no combinational loop -- exp/rsqrt are feed-forward glm_fp functions).
//   * bf16 in, bf16 out, RNE on the final narrow.
//============================================================================
module glm_act_core #(
    parameter integer MODE  = 0,                 // 0 = SIGMOID, 1 = SILU
    parameter integer LANES = 4,
    parameter [31:0]  X_SAT = 32'h41800000       // 16.0 fp32 (saturation rail)
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                in_valid,
    input  wire [LANES*16-1:0] x_in,
    output reg                 out_valid,
    output reg  [LANES*16-1:0] y_out
);
    // ---- mode encodings (named, for readability) ----
    localparam integer MODE_SIGMOID = 0;
    localparam integer MODE_SILU    = 1;
    // IS_SILU folds the MODE param to a single compile-time bit (and references
    // both encodings so neither is an "unused param").
    localparam         IS_SILU      = (MODE == MODE_SILU);
    localparam         IS_SIGMOID   = (MODE == MODE_SIGMOID);
    // elaboration guard: MODE must be one of the two legal encodings.
    initial begin
        if (!(IS_SILU || IS_SIGMOID)) begin
            $display("glm_act: ILLEGAL MODE=%0d (must be %0d SIGMOID or %0d SILU)",
                     MODE, MODE_SIGMOID, MODE_SILU);
            $fatal(1, "glm_act bad MODE");
        end
    end

    // ---- fp32 constants (bit patterns; no `real`, yosys-friendly) ----
    localparam [31:0] FP_ONE   = 32'h3F800000;   // 1.0
    localparam [31:0] FP_LOG2E = 32'h3FB8AA3B;   // log2(e)        = 1.44269504
    localparam [31:0] FP_LN2   = 32'h3F317218;   // ln(2)          = 0.69314718
    // 1/i! polynomial coefficients for exp(r) Horner:
    localparam [31:0] FP_1_2   = 32'h3F000000;   // 1/2
    localparam [31:0] FP_1_6   = 32'h3E2AAAAB;   // 1/6
    localparam [31:0] FP_1_24  = 32'h3D2AAAAB;   // 1/24
    localparam [31:0] FP_1_120 = 32'h3C088889;   // 1/120
    // K saturation: with X_SAT=16, |z|<=16, k = round(z*1.4427) in [-24,24];
    // clamp to +/-K_MAX so the exponent add can never leave the fp32 field.
    localparam integer K_MAX = 64;

    //------------------------------------------------------------------------
    // round-to-nearest fp32 -> signed integer (for k = round(z*log2e)).
    // |arg| <= 16*log2e ~ 23.1, so a small signed int is plenty.  Pure
    // feed-forward; handles sign and the 0.5 round.  Returns a 32-bit signed.
    //------------------------------------------------------------------------
    function automatic signed [31:0] fp32_round_to_int(input [31:0] f);
        reg        s;
        reg [7:0]  e;
        reg [23:0] m;            // implicit-1 significand
        integer    rsh;          // right-shift to align binary point (= -sh)
        // shifted holds m>>rsh; only its low 8 bits (the integer part, |k|<=24)
        // are used -- the high bits are the now-fractional remainder, waive lint.
        /* verilator lint_off UNUSEDSIGNAL */
        reg [23:0] shifted;      // m >> rsh (integer-part-aligned)
        /* verilator lint_on UNUSEDSIGNAL */
        reg [7:0]  mag;          // integer magnitude (pre-round): |round|<=24 -> 8b
        reg        frac_half;    // the bit just below the point
        reg        frac_rest;    // sticky OR below that
        reg [7:0]  r;            // |k| <= 24 here -> 8 bits is ample
        begin
            s = f[31];
            e = f[30:23];
            m = {1'b1, f[22:0]};
            // value = (-1)^s * 1.m * 2^(e-127).  Integer part needs the binary
            // point at bit (e-127) of the 24-bit significand whose point sits
            // just below bit 23.
            if (e < 8'd127) begin
                // |f| < 1.0 -> rounds to 0 or +/-1 by the 0.5 test
                // value's leading bit is below the units place; compare to 0.5
                if (e == 8'd126) r = 8'd1;          // [0.5, 1.0): rounds to 1
                else             r = 8'd0;          // < 0.5 -> 0
                frac_half = 1'b0;
                frac_rest = 1'b0;
            end else begin
                // This unit only ever rounds z*log2e with |z| <= X_SAT=16, so
                // |f| <= ~23.1 -> e in [127,131] -> the binary point is always
                // BELOW bit 23, i.e. a pure RIGHT shift by rsh = 23-(e-127) in
                // [19,23].  The left-shift (sh>=0, value>=2^23) case cannot occur
                // for this bounded input and is omitted by construction, shrinking
                // the 32-wide shifter to a 24-bit right-shifter and the magnitude
                // to 8 bits (|round| <= 24).
                rsh       = 32'd23 - ({24'b0, e} - 32'd127);
                shifted   = m >> rsh;               // capture integer part
                mag       = shifted[7:0];
                frac_half = m[(rsh-1)];
                if ((rsh-1) > 0)
                    frac_rest = (m & ((24'd1 << (rsh-1)) - 24'd1)) != 24'd0;
                else
                    frac_rest = 1'b0;
                // round-half-up on magnitude (ties away is fine: k feeds a
                // range reduction, a +/-1 tie choice only shifts r by ln2 and
                // is fully corrected by exp(r) -- result identical to ULP).
                r = mag + (frac_half ? 8'd1 : 8'd0);
                if (frac_rest) begin /* sticky already <0.5, no extra */ end
            end
            fp32_round_to_int = s ? -$signed({24'b0, r}) : $signed({24'b0, r});
        end
    endfunction

    //------------------------------------------------------------------------
    // exp(r) for r in [-ln2/2, ln2/2]: degree-5 Taylor-Horner, all fp32.
    //   p = 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120)))))
    // The five Horner levels are evaluated ONE PER PIPELINE STAGE below (E1-E5)
    // -- the same fp32_add/fp32_mul calls in the same grouping/order as the old
    // exp_poly_lo/exp_poly_hi split, so exp(r) is bit-identical; only registers
    // moved.  (Truncation error < 2^-18, far under a bf16 ULP -- see header.)
    //------------------------------------------------------------------------

    //------------------------------------------------------------------------
    // 2^k * v by adding k to the biased exponent of v (k pre-clamped to
    // [-K_MAX,K_MAX] so the field never overflows/underflows for normal v).
    // v here is exp(r) in [~0.707, ~1.414], always a normal positive fp32.
    //------------------------------------------------------------------------
    // k is a 32-bit signed but pre-clamped to [-K_MAX,K_MAX] (|k|<=64), so only
    // its low 11 bits are ever significant; the high bits are intentionally
    // unread (they are sign-extension of a tiny value) -- waive the lint.
    /* verilator lint_off UNUSEDSIGNAL */
    function automatic [31:0] scale_pow2(input [31:0] v, input signed [31:0] k);
    /* verilator lint_on UNUSEDSIGNAL */
        reg signed [10:0] e_new;
        begin
            e_new = $signed({3'b0, v[30:23]}) + k[10:0];
            if (e_new >= 11'sd255)
                scale_pow2 = {v[31], 8'hFE, v[22:0]};   // clamp just below inf
            else if (e_new <= 11'sd0)
                scale_pow2 = {v[31], 31'b0};            // flush to zero
            else
                scale_pow2 = {v[31], e_new[7:0], v[22:0]};
        end
    endfunction

    //------------------------------------------------------------------------
    // fp32 reciprocal of d (d > 0):  1/d = rsqrt(d)^2.  The Quake-seed +
    // 2-Newton-iteration rsqrt is UNROLLED one fp32 op per pipeline stage below
    // (R1-R10) -- identical call sequence to glm_fp.vh fp32_rsqrt followed by
    // the squaring multiply, so the result is bit-identical.  The special-case
    // branch (nan / x<=0 / inf) is carried alongside and muxed at R10, exactly
    // as the function's early returns.
    //------------------------------------------------------------------------

    //------------------------------------------------------------------------
    // sanitize_x : replace inf/nan with a finite +/-X_SAT (sign from input; nan
    // -> +X_SAT).  Keeps the SiLU multiply (which uses RAW x) finite for
    // pathological inputs.  Normal/zero pass through unchanged.
    //------------------------------------------------------------------------
    function automatic [31:0] sanitize_x(input [31:0] x);
        begin
            if (x[30:23] == 8'hFF) sanitize_x = {x[31], X_SAT[30:0]}; // inf/nan
            else                   sanitize_x = x;
        end
    endfunction

    //------------------------------------------------------------------------
    // clamp an fp32 x to [-X_SAT, X_SAT].  (Input is already inf/nan-free via
    // sanitize_x, so this is a pure magnitude clamp by sign.)
    //------------------------------------------------------------------------
    function automatic [31:0] clamp_xsat(input [31:0] x);
        reg s;
        reg [30:0] mag, sat_mag;
        begin
            s       = x[31];
            mag     = x[30:0];
            sat_mag = X_SAT[30:0];
            if (mag >= sat_mag) clamp_xsat = {s, sat_mag};   // includes inf/nan
            else                clamp_xsat = x;
        end
    endfunction

    // ===================================================================
    //  PER-LANE PIPELINE  (REPIPELINED FOR FMAX -- bit-exact)
    //
    //  The old 6-stage pipe put the WHOLE Quake-rsqrt reciprocal (7 serial
    //  multiplies) in one stage -- measured on XCKU3P as the chip's worst
    //  cone after the rope fix (58.5 ns / 236 levels, S3->S4).  The pipe is
    //  now LAT = 21 stages with at most ONE serial fp32 mul (or add) each:
    //
    //    A1  sanitize/clamp, z = -x, kf = z*log2e                [mul]
    //    A2  k = round(kf) clamp, klt = int->fp32, m = klt*ln2   [mul]
    //    A3  r = z - m                                           [add]
    //    E1..E5  the five exp(r) Horner levels                   [mul+add each]
    //    D1  ex = 2^k*exp(r) (exponent add), d = 1 + ex          [add]
    //    R1  xhalf = 0.5*d, y = quake-seed(d), special flags     [mul]
    //    R2  yy = y*y            R3  xyy = xhalf*yy              [mul]
    //    R4  t = 1.5 - xyy [add] R5  y = y*t                     [mul]
    //    R6..R9  the second Newton iteration (same four ops)
    //    R10 s = y*y (reciprocal), special mux                   [mul]
    //    F1  SILU ? s*x : s ; round fp32->bf16 -> y_out          [mul]
    //
    //  Every operation is the SAME glm_fp.vh call in the SAME order as the
    //  old code (register insertion only) -> outputs byte-identical.  Each
    //  stage's registers are ENABLED by the valid tap, so idle cycles freeze
    //  operands (operand-isolation, as before).  d = 1 + exp(z) with the
    //  X_SAT/K_MAX clamps is always a positive normal, so the rsqrt special
    //  branch is unreachable here -- the flags are carried anyway so the mux
    //  replicates fp32_rsqrt's early returns verbatim.
    // ===================================================================
    localparam integer LAT = 20;   // 19 stage banks (A1..R10) + the y_out bank
    reg [LAT-1:0] vp;

    // ---- stage registers (per lane) ----
    reg [31:0]       a1_z   [0:LANES-1];   // z = -clamp(x)
    reg [31:0]       a1_kf  [0:LANES-1];   // z * log2e
    reg signed [8:0] a2_k   [0:LANES-1];   // round(kf), clamped +/-64
    reg [31:0]       a2_z   [0:LANES-1];
    reg [31:0]       a2_m   [0:LANES-1];   // k * ln2
    reg [31:0]       a3_r   [0:LANES-1];   // reduced r
    reg signed [8:0] a3_k   [0:LANES-1];
    reg [31:0]       e_p    [1:5][0:LANES-1];  // Horner accumulator per level
    reg [31:0]       e_r    [1:5][0:LANES-1];  // forwarded r
    reg signed [8:0] e_k    [1:5][0:LANES-1];  // forwarded k
    reg [31:0]       d1_d   [0:LANES-1];   // 1 + 2^k * exp(r)
    reg [31:0]       r1_xh  [0:LANES-1];   // 0.5*d
    reg [31:0]       r1_y   [0:LANES-1];   // quake seed
    reg [1:0]        r1_sp  [0:LANES-1];   // special: 0 none, 1 nan-out, 2 zero-out
    reg [31:0]       r2_yy  [0:LANES-1];
    reg [31:0]       r2_xh  [0:LANES-1], r2_y [0:LANES-1];
    reg [1:0]        r2_sp  [0:LANES-1];
    reg [31:0]       r3_xyy [0:LANES-1];
    reg [31:0]       r3_xh  [0:LANES-1], r3_y [0:LANES-1];
    reg [1:0]        r3_sp  [0:LANES-1];
    reg [31:0]       r4_t   [0:LANES-1];
    reg [31:0]       r4_xh  [0:LANES-1], r4_y [0:LANES-1];
    reg [1:0]        r4_sp  [0:LANES-1];
    reg [31:0]       r5_y   [0:LANES-1];
    reg [31:0]       r5_xh  [0:LANES-1];
    reg [1:0]        r5_sp  [0:LANES-1];
    reg [31:0]       r6_yy  [0:LANES-1];
    reg [31:0]       r6_xh  [0:LANES-1], r6_y [0:LANES-1];
    reg [1:0]        r6_sp  [0:LANES-1];
    reg [31:0]       r7_xyy [0:LANES-1];
    reg [31:0]       r7_xh  [0:LANES-1], r7_y [0:LANES-1];
    reg [1:0]        r7_sp  [0:LANES-1];
    reg [31:0]       r8_t   [0:LANES-1];
    reg [31:0]       r8_xh  [0:LANES-1], r8_y [0:LANES-1];
    reg [1:0]        r8_sp  [0:LANES-1];
    reg [31:0]       r9_y   [0:LANES-1];
    reg [1:0]        r9_sp  [0:LANES-1];
    reg [31:0]       r10_s  [0:LANES-1];   // sigmoid (post special mux)

    integer j;
    genvar  gl;

    // ---- per-stage next-value combinationals ----
    reg [31:0] cA_xraw, cA_xcl;
    reg [31:0] nA1_z [0:LANES-1], nA1_kf [0:LANES-1];
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : NA1
            cA_xraw   = sanitize_x(bf16_to_fp32(x_in[16*j +: 16]));
            cA_xcl    = clamp_xsat(cA_xraw);
            nA1_z[j]  = {~cA_xcl[31], cA_xcl[30:0]};             // z = -x
            nA1_kf[j] = fp32_mul(nA1_z[j], FP_LOG2E);
        end
    end
    reg signed [31:0] cA2_k;
    reg [31:0]        cA2_klt;
    reg signed [8:0]  nA2_k [0:LANES-1];
    reg [31:0]        nA2_m [0:LANES-1];
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : NA2
            cA2_k = fp32_round_to_int(a1_kf[j]);
            if (cA2_k >  K_MAX) cA2_k =  K_MAX;
            if (cA2_k < -K_MAX) cA2_k = -K_MAX;
            nA2_k[j]  = cA2_k[8:0];
            cA2_klt   = int_to_fp32(cA2_k);
            nA2_m[j]  = fp32_mul(cA2_klt, FP_LN2);
        end
    end
    reg [31:0] nA3_r [0:LANES-1];
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : NA3
            nA3_r[j] = fp32_add(a2_z[j], neg_fp32(a2_m[j]));
        end
    end
    // exp(r) Horner levels (E1..E5) -- same grouping as the old lo(3)+hi(2).
    reg [31:0] nE_p [1:5][0:LANES-1];
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : NE
            nE_p[1][j] = fp32_add(FP_1_24, fp32_mul(a3_r[j],  FP_1_120));
            nE_p[2][j] = fp32_add(FP_1_6,  fp32_mul(e_r[1][j], e_p[1][j]));
            nE_p[3][j] = fp32_add(FP_1_2,  fp32_mul(e_r[2][j], e_p[2][j]));
            nE_p[4][j] = fp32_add(FP_ONE,  fp32_mul(e_r[3][j], e_p[3][j]));
            nE_p[5][j] = fp32_add(FP_ONE,  fp32_mul(e_r[4][j], e_p[4][j]));
        end
    end
    reg [31:0] cD_ex;
    reg [31:0] nD1_d [0:LANES-1];
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : ND1
            cD_ex    = scale_pow2(e_p[5][j], {{23{e_k[5][j][8]}}, e_k[5][j]});
            nD1_d[j] = fp32_add(FP_ONE, cD_ex);
        end
    end
    // rsqrt stages -- fp32_rsqrt(x) unrolled: specials at R1, Newton R2..R9,
    // square + special mux at R10.
    localparam [31:0] FP_HALF = 32'h3F000000, FP_3HALF = 32'h3FC00000;
    reg [31:0] nR1_xh [0:LANES-1], nR1_y [0:LANES-1];
    reg [1:0]  nR1_sp [0:LANES-1];
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : NR1
            // exact fp32_rsqrt special tests on the SAME input d:
            if ((d1_d[j][30:23] == 8'hFF && d1_d[j][22:0] != 23'b0) ||  // nan
                d1_d[j][31] == 1'b1 ||                                  // x < 0
                d1_d[j][30:23] == 8'b0)                                 // zero/denorm
                 nR1_sp[j] = 2'd1;                                      // -> nan out
            else if (d1_d[j][30:23] == 8'hFF)                           // +inf
                 nR1_sp[j] = 2'd2;                                      // -> +0 out
            else nR1_sp[j] = 2'd0;
            nR1_xh[j] = fp32_mul(FP_HALF, d1_d[j]);
            nR1_y[j]  = 32'h5F3759DF - (d1_d[j] >> 1);
        end
    end
    reg [31:0] nR2_yy [0:LANES-1], nR3_xyy [0:LANES-1], nR4_t [0:LANES-1];
    reg [31:0] nR5_y  [0:LANES-1];
    reg [31:0] nR6_yy [0:LANES-1], nR7_xyy [0:LANES-1], nR8_t [0:LANES-1];
    reg [31:0] nR9_y  [0:LANES-1], nR10_s [0:LANES-1];
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : NRN
            nR2_yy[j]  = fp32_mul(r1_y[j],  r1_y[j]);
            nR3_xyy[j] = fp32_mul(r2_xh[j], r2_yy[j]);
            nR4_t[j]   = fp32_add(FP_3HALF, {r3_xyy[j][31]^1'b1, r3_xyy[j][30:0]});
            nR5_y[j]   = fp32_mul(r4_y[j],  r4_t[j]);
            nR6_yy[j]  = fp32_mul(r5_y[j],  r5_y[j]);
            nR7_xyy[j] = fp32_mul(r6_xh[j], r6_yy[j]);
            nR8_t[j]   = fp32_add(FP_3HALF, {r7_xyy[j][31]^1'b1, r7_xyy[j][30:0]});
            nR9_y[j]   = fp32_mul(r8_y[j],  r8_t[j]);
            // R10: sigmoid = rsqrt(d)^2, with fp32_rsqrt's special returns:
            case (r9_sp[j])
                2'd1:    nR10_s[j] = fp32_mul(32'h7FC00000, 32'h7FC00000);
                2'd2:    nR10_s[j] = fp32_mul(32'h00000000, 32'h00000000);
                default: nR10_s[j] = fp32_mul(r9_y[j], r9_y[j]);
            endcase
        end
    end

    // ---- F1: SiLU multiply + narrow to bf16 (x carried IS_SILU-only) ----
    reg [LANES*16-1:0] n_y;
    reg [31:0] cF_val;
    generate
    if (IS_SILU) begin : g_silu
        // raw-x delay line, one tap per stage, aligned with the sigmoid pipe.
        reg [31:0] xfw [1:LAT-1][0:LANES-1];
        integer jx, ks;
        always @(posedge clk) begin
            if (rst) begin
                for (ks = 1; ks <= LAT-1; ks = ks + 1)
                    for (jx = 0; jx < LANES; jx = jx + 1) xfw[ks][jx] <= 32'b0;
            end else begin
                for (jx = 0; jx < LANES; jx = jx + 1) begin
                    if (in_valid) xfw[1][jx] <= sanitize_x(bf16_to_fp32(x_in[16*jx +: 16]));
                    for (ks = 2; ks <= LAT-1; ks = ks + 1)
                        if (vp[ks-2]) xfw[ks][jx] <= xfw[ks-1][jx];
                end
            end
        end
        always @* begin
            n_y = {LANES*16{1'b0}};
            for (jx = 0; jx < LANES; jx = jx + 1) begin
                cF_val = fp32_mul(r10_s[jx], xfw[LAT-1][jx]);    // x * sigmoid(x)
                n_y[16*jx +: 16] = fp32_to_bf16(cF_val);
            end
        end
    end else begin : g_sigmoid
        integer jx;
        always @* begin
            n_y = {LANES*16{1'b0}};
            for (jx = 0; jx < LANES; jx = jx + 1) begin
                cF_val = r10_s[jx];
                n_y[16*jx +: 16] = fp32_to_bf16(cF_val);
            end
        end
    end
    endgenerate

    // ---- pipeline registers: stage k enabled by its valid tap ----
    integer L2;
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            y_out     <= {LANES*16{1'b0}};
            vp        <= {LAT{1'b0}};
            for (j = 0; j < LANES; j = j + 1) begin
                a1_z[j] <= 32'b0; a1_kf[j] <= 32'b0;
                a2_k[j] <= 9'sb0; a2_z[j] <= 32'b0; a2_m[j] <= 32'b0;
                a3_r[j] <= 32'b0; a3_k[j] <= 9'sb0;
                for (L2 = 1; L2 <= 5; L2 = L2 + 1) begin
                    e_p[L2][j] <= 32'b0; e_r[L2][j] <= 32'b0; e_k[L2][j] <= 9'sb0;
                end
                d1_d[j] <= 32'b0;
                r1_xh[j] <= 32'b0; r1_y[j] <= 32'b0; r1_sp[j] <= 2'b0;
                r2_yy[j] <= 32'b0; r2_xh[j] <= 32'b0; r2_y[j] <= 32'b0; r2_sp[j] <= 2'b0;
                r3_xyy[j] <= 32'b0; r3_xh[j] <= 32'b0; r3_y[j] <= 32'b0; r3_sp[j] <= 2'b0;
                r4_t[j] <= 32'b0; r4_xh[j] <= 32'b0; r4_y[j] <= 32'b0; r4_sp[j] <= 2'b0;
                r5_y[j] <= 32'b0; r5_xh[j] <= 32'b0; r5_sp[j] <= 2'b0;
                r6_yy[j] <= 32'b0; r6_xh[j] <= 32'b0; r6_y[j] <= 32'b0; r6_sp[j] <= 2'b0;
                r7_xyy[j] <= 32'b0; r7_xh[j] <= 32'b0; r7_y[j] <= 32'b0; r7_sp[j] <= 2'b0;
                r8_t[j] <= 32'b0; r8_xh[j] <= 32'b0; r8_y[j] <= 32'b0; r8_sp[j] <= 2'b0;
                r9_y[j] <= 32'b0; r9_sp[j] <= 2'b0;
                r10_s[j] <= 32'b0;
            end
        end else begin
            vp        <= {vp[LAT-2:0], in_valid};
            out_valid <= vp[LAT-2];
            if (vp[LAT-2]) y_out <= n_y;
            for (j = 0; j < LANES; j = j + 1) begin
                if (in_valid) begin
                    a1_z[j] <= nA1_z[j]; a1_kf[j] <= nA1_kf[j];
                end
                if (vp[0]) begin
                    a2_k[j] <= nA2_k[j]; a2_z[j] <= a1_z[j]; a2_m[j] <= nA2_m[j];
                end
                if (vp[1]) begin
                    a3_r[j] <= nA3_r[j]; a3_k[j] <= a2_k[j];
                end
                if (vp[2]) begin
                    e_p[1][j] <= nE_p[1][j]; e_r[1][j] <= a3_r[j]; e_k[1][j] <= a3_k[j];
                end
                if (vp[3]) begin
                    e_p[2][j] <= nE_p[2][j]; e_r[2][j] <= e_r[1][j]; e_k[2][j] <= e_k[1][j];
                end
                if (vp[4]) begin
                    e_p[3][j] <= nE_p[3][j]; e_r[3][j] <= e_r[2][j]; e_k[3][j] <= e_k[2][j];
                end
                if (vp[5]) begin
                    e_p[4][j] <= nE_p[4][j]; e_r[4][j] <= e_r[3][j]; e_k[4][j] <= e_k[3][j];
                end
                if (vp[6]) begin
                    e_p[5][j] <= nE_p[5][j]; e_r[5][j] <= e_r[4][j]; e_k[5][j] <= e_k[4][j];
                end
                if (vp[7])  d1_d[j] <= nD1_d[j];
                if (vp[8])  begin
                    r1_xh[j] <= nR1_xh[j]; r1_y[j] <= nR1_y[j]; r1_sp[j] <= nR1_sp[j];
                end
                if (vp[9])  begin
                    r2_yy[j] <= nR2_yy[j];
                    r2_xh[j] <= r1_xh[j]; r2_y[j] <= r1_y[j]; r2_sp[j] <= r1_sp[j];
                end
                if (vp[10]) begin
                    r3_xyy[j] <= nR3_xyy[j];
                    r3_xh[j] <= r2_xh[j]; r3_y[j] <= r2_y[j]; r3_sp[j] <= r2_sp[j];
                end
                if (vp[11]) begin
                    r4_t[j] <= nR4_t[j];
                    r4_xh[j] <= r3_xh[j]; r4_y[j] <= r3_y[j]; r4_sp[j] <= r3_sp[j];
                end
                if (vp[12]) begin
                    r5_y[j] <= nR5_y[j];
                    r5_xh[j] <= r4_xh[j]; r5_sp[j] <= r4_sp[j];
                end
                if (vp[13]) begin
                    r6_yy[j] <= nR6_yy[j];
                    r6_xh[j] <= r5_xh[j]; r6_y[j] <= r5_y[j]; r6_sp[j] <= r5_sp[j];
                end
                if (vp[14]) begin
                    r7_xyy[j] <= nR7_xyy[j];
                    r7_xh[j] <= r6_xh[j]; r7_y[j] <= r6_y[j]; r7_sp[j] <= r6_sp[j];
                end
                if (vp[15]) begin
                    r8_t[j] <= nR8_t[j];
                    r8_xh[j] <= r7_xh[j]; r8_y[j] <= r7_y[j]; r8_sp[j] <= r7_sp[j];
                end
                if (vp[16]) begin
                    r9_y[j] <= nR9_y[j]; r9_sp[j] <= r8_sp[j];
                end
                if (vp[17]) r10_s[j] <= nR10_s[j];
            end
        end
    end

    //------------------------------------------------------------------------
    // neg_fp32 : flip the sign bit (exact, handles zero/inf; nan stays nan-ish).
    //------------------------------------------------------------------------
    function automatic [31:0] neg_fp32(input [31:0] f);
        neg_fp32 = {~f[31], f[30:0]};
    endfunction

    //------------------------------------------------------------------------
    // int_to_fp32 : convert a SMALL signed integer (|k| <= K_MAX) to fp32.
    // Range here is tiny so a simple normalize loop suffices; pure feed-forward,
    // constant-bounded -> synthesizable.  k=0 -> +0.0.
    //------------------------------------------------------------------------
    function automatic [31:0] int_to_fp32(input signed [31:0] k);
        reg        s;
        reg [7:0]  a;            // |k| <= K_MAX = 64 -> 7 significant bits (8b ample)
        // mshift's bit 23 (the leading 1) is intentionally dropped (we keep the
        // 23 mantissa bits below it) -- waive the unused-bits lint on that slice.
        /* verilator lint_off UNUSEDSIGNAL */
        reg [23:0] mshift;       // a left-justified to expose the fraction
        /* verilator lint_on UNUSEDSIGNAL */
        integer    msb, i;
        reg [7:0]  e;
        reg [22:0] mant;
        begin
            if (k == 0) int_to_fp32 = 32'b0;
            else begin
                s = k[31];
                a = s ? (~k[7:0] + 8'd1) : k[7:0];  // |k| in 8 bits (|k|<=64<128)
                // |k| <= 64 -> MSB index <= 6, so an 8-wide priority scan suffices.
                msb = 0;
                for (i = 0; i < 8; i = i + 1)
                    if (a[i]) msb = i;          // highest set bit (<= 6)
                e = 8'd127 + msb[7:0];
                // mantissa = fractional bits below the MSB, left-justified to 23.
                // msb <= 6 < 23 so this is ALWAYS a left shift; the msb>=23
                // (|k|>=2^23) case cannot occur for |k|<=64 and is omitted.
                mshift = {16'b0, a} << (23 - msb);
                mant   = mshift[22:0];
                int_to_fp32 = {s, e, mant};
            end
        end
    endfunction
endmodule

//============================================================================
// glm_act -- public wrapper: same interface/params as always, plus HW_LANES,
//   a RESULT-INVARIANT resource knob (like PE_N / DDR_NCH / CACHE_SLOTS).
//
//   HW_LANES = 0 (default) or >= LANES : ONE full-LANES-wide glm_act_core --
//     structurally the pre-wrapper unit, byte-identical outputs AND latency.
//   HW_LANES in [1, LANES-1] : ONE HW_LANES-wide core; the LANES-wide beat is
//     streamed through it in ceil(LANES/HW_LANES) back-to-back chunks and the
//     outputs are reassembled.  The activation is ELEMENTWISE (lane j depends
//     only on x_in[j]) and every element passes the IDENTICAL 6-stage pipeline,
//     so every bf16 output bit is UNCHANGED -- only the in_valid->out_valid
//     latency grows (LAT=6 -> ~LAT+1+chunks).  Both Q4_K users (moe_router_q4k
//     S_GATES/S_GWAIT, swiglu_expert_q4k S_UPW/S_GUW) issue ONE beat and wait
//     on out_valid before the next, so no backpressure port is needed; a beat
//     arriving while one is in flight is ignored by construction (the callers
//     never do this).
//
//   WHY: each core lane carries the full fp32 exp/recip pipeline (~2-3K LUTs).
//   The router instantiates LANES = N_EXPERT*PE_M (48K LUTs at the compact
//   config -- larger than all of attention); the two SwiGLU units are PE_M*TN
//   wide (18K each).  Serializing lanes is the honest resource knob: the fit
//   config sets ACT_HW=1, the default config keeps 0 (full width, unchanged).
//
//   IMPLEMENTATION NOTE: the chunk stream uses CONSTANT-width shifts of a
//   holding register (xhold >>= HW*16, ybuf = {chunk, ybuf >> HW*16}) -- no
//   variable part-selects, so static-bounds analysis (Vivado Synth 8-524) has
//   nothing to reject and no index can ever leave the vector.
//============================================================================
module glm_act #(
    parameter integer MODE     = 0,              // 0 = SIGMOID, 1 = SILU
    parameter integer LANES    = 4,
    parameter [31:0]  X_SAT    = 32'h41800000,   // 16.0 fp32 (saturation rail)
    parameter integer HW_LANES = 0               // 0 / >=LANES: full-width core
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                in_valid,
    input  wire [LANES*16-1:0] x_in,
    output wire                out_valid,
    output wire [LANES*16-1:0] y_out
);
    generate
    if (HW_LANES <= 0 || HW_LANES >= LANES) begin : g_full
        // ---- full width: exactly the pre-wrapper unit (same latency) ----
        glm_act_core #(.MODE(MODE), .LANES(LANES), .X_SAT(X_SAT)) u_core (
            .clk(clk), .rst(rst),
            .in_valid(in_valid), .x_in(x_in),
            .out_valid(out_valid), .y_out(y_out)
        );
    end else begin : g_ser
        // ---- serialized: one HW_LANES-wide core, NCH back-to-back chunks ----
        localparam integer HW   = HW_LANES;
        localparam integer NCH  = (LANES + HW - 1) / HW;    // chunks per beat
        localparam integer PADW = NCH * HW * 16;            // padded hold width
        localparam integer CW   = $clog2(NCH + 1);          // counter width

        reg  [PADW-1:0]  xhold;      // input hold; consumed low-chunk-first
        reg  [PADW-1:0]  ybuf;       // output assembly ({chunk, >>} = in-order)
        reg  [CW-1:0]    icnt;       // chunks fed
        reg  [CW-1:0]    ocnt;       // chunks collected
        reg              running;
        reg              c_iv;
        reg  [HW*16-1:0] c_x;
        wire             c_ov;
        wire [HW*16-1:0] c_y;
        reg              ov_r;

        glm_act_core #(.MODE(MODE), .LANES(HW), .X_SAT(X_SAT)) u_core (
            .clk(clk), .rst(rst),
            .in_valid(c_iv), .x_in(c_x),
            .out_valid(c_ov), .y_out(c_y)
        );

        always @(posedge clk) begin
            if (rst) begin
                xhold <= {PADW{1'b0}};  ybuf <= {PADW{1'b0}};
                icnt  <= {CW{1'b0}};    ocnt <= {CW{1'b0}};
                running <= 1'b0;  c_iv <= 1'b0;
                c_x   <= {(HW*16){1'b0}};
                ov_r  <= 1'b0;
            end else begin
                ov_r <= 1'b0;
                c_iv <= 1'b0;
                if (in_valid && !running) begin
                    // latch + zero-pad the beat; chunk 0 issues next cycle.
                    // (pad lanes compute sigmoid(0) and are discarded below.)
                    xhold   <= {{(PADW-LANES*16){1'b0}}, x_in};
                    icnt    <= {CW{1'b0}};
                    ocnt    <= {CW{1'b0}};
                    running <= 1'b1;
                end else if (running && icnt != NCH[CW-1:0]) begin
                    // feed chunks back-to-back (the core pipelines every cycle)
                    c_iv  <= 1'b1;
                    c_x   <= xhold[HW*16-1:0];
                    xhold <= xhold >> (HW*16);
                    icnt  <= icnt + 1'b1;
                end
                if (c_ov) begin
                    // insert at the top, shift down: after NCH inserts chunk i
                    // sits at slice i -- in-order little-endian reassembly.
                    ybuf <= {c_y, ybuf[PADW-1:HW*16]};
                    ocnt <= ocnt + 1'b1;
                    if (ocnt == NCH[CW-1:0] - 1'b1) begin
                        ov_r    <= 1'b1;
                        running <= 1'b0;
                    end
                end
            end
        end
        assign out_valid = ov_r;
        assign y_out     = ybuf[LANES*16-1:0];
    end
    endgenerate
endmodule
