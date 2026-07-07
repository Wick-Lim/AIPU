`timescale 1ns/1ps
`include "glm_fp.vh"
`include "q4k.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_matmul_q4k.v  --  GLM-5.2 Q4_K-NATIVE GEMM datapath (local-device target)
//                       a DROP-IN sibling of glm_matmul_pipe.v / glm_matmul_fp8.v.
//----------------------------------------------------------------------------
// FUNCTION
//   C[M,N] = A[M,K] x W[K,N], computed in the OFFICIAL GGML Q4_K numerics so the
//   published `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL` weights run with NO
//   re-quantization -- bit-exact to ggml `dequantize_row_q4_K` (tools/q4k_ref.py).
//
//   * Weights W arrive as GGML Q4_K: per output column pj a super-block carries
//       - fp16  d[pj]      (super-block scale)
//       - fp16  dmin[pj]   (super-block min)
//       - 96b   scales[pj] (8x 6-bit block-scales + 8x 6-bit block-mins, packed)
//       - per K-beat, a 4-bit quant code w_q[pj] (the weight for row k, col pj).
//     The K-beat's sub-block b = k/32 selects (sc_b, m_b) via get_scale_min_k4;
//     the weight dequantizes EXACTLY to  w = (d*sc_b)*q - (dmin*m_b)  (fp32).
//   * Activations A arrive bf16 (same interface as glm_matmul_pipe).
//
// CONTRACT (bit-exact to tools/q4k_ref.py `matmul_q4k_col`):
//   out[pi][pj] = bf16( SUM_k fp32(a[pi][k]) * w_deq[k][pj] ), the fp32 products
//   sequentially fp32-accumulated in K (streaming) order, rounded to bf16 (RNE).
//   Same accumulate structure as the proven bf16 glm_matmul_pipe -- only the
//   weight source changes (Q4_K dequant instead of a bf16 weight).  All fp32 ops
//   are glm_fp.vh's IEEE fp32_mul / fp32_add (the datapath's canonical numerics).
//
// This is the CORRECT-FIRST reference core: the per-weight dequant + fp32 MAC are
//   combinational, one K-beat per cycle, the accumulator registered each beat (a
//   single-cycle sequential accumulate -- no hazard).  KMAX <= 256 = one Q4_K
//   super-block along K (the caller tiles larger K, as with glm_matmul_fp8).
//
// CONVENTIONS: synchronous ACTIVE-HIGH reset; NO latch; NO combinational loop
//   (the accumulator feedback rides the per-beat register).
//----------------------------------------------------------------------------
// HANDSHAKE  (mirrors glm_matmul_pipe; activation interface identical)
//   start     : 1-cycle pulse; latches k_len + the per-column super-block params.
//   k_len     : number of K beats this tile (<= KMAX <= 256).
//   in_valid  : a K-beat is presented (a_col + w_q).
//   a_col[pi] : bf16 A[pi][k]  (PE_M packed, 16b each).
//   w_q[pj]   : 4-bit Q4_K code W[k][pj] (PE_N packed, 4b each).
//   w_d/w_dmin: fp16 super-block scale/min per column (PE_N packed, 16b), at start.
//   w_scales  : 96b packed 6-bit scales/mins per column (PE_N packed), at start.
//   out_valid : C tile (PE_M x PE_N bf16) valid for 1 cycle.  busy high in flight.
//============================================================================
module glm_matmul_q4k #(
    parameter integer PE_M = 4,       // array rows (== tile M)
    parameter integer PE_N = 4,       // array cols (== tile N)
    parameter integer KMAX = 256      // max K per tile (one Q4_K super-block)
) (
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    input  wire                       start,      // begin a tile
    input  wire [$clog2(KMAX+1)-1:0]  k_len,      // number of K beats this tile

    // per-column Q4_K super-block params (latched at start).  A column spans
    // NSB = ceil(KMAX/256) super-blocks along K; super-block sb of column pj is at
    // index (pj*NSB + sb).  (KMAX=256 -> NSB=1, one super-block per column.)
    input  wire [16*PE_N*((KMAX+255)/256)-1:0] w_d,      // fp16 d    per (col, super-block)
    input  wire [16*PE_N*((KMAX+255)/256)-1:0] w_dmin,   // fp16 dmin per (col, super-block)
    input  wire [96*PE_N*((KMAX+255)/256)-1:0] w_scales, // 96b scales per (col, super-block)

    input  wire                       in_valid,   // a K-beat is presented
    input  wire [16*PE_M-1:0]         a_col,      // bf16 A[*][k], PE_M packed
    input  wire [ 4*PE_N-1:0]         w_q,        // 4-bit Q4_K codes W[k][*], PE_N packed

    output reg                        busy,
    output reg                        out_valid,  // C tile valid (1 cycle)
    output reg  [16*PE_M*PE_N-1:0]    c_out       // bf16 C[pi][pj] packed
);
    localparam integer KW  = $clog2(KMAX+1);
    localparam integer NSB = (KMAX + 255) / 256;   // super-blocks along K

    // ---- latched tile params ----
    reg [KW-1:0]              k_cnt;      // beats consumed
    reg [KW-1:0]              k_len_r;
    reg [16*PE_N*NSB-1:0]     d_r, dmin_r;
    reg [96*PE_N*NSB-1:0]     scales_r;

    // ---- accumulators (fp32) ----
    reg [31:0] acc [0:PE_M*PE_N-1];

    integer pi, pj, idx, col;
    reg [KW-1:0] sb;                // super-block = k / 256
    reg [2:0]  sub;                 // sub-block within super-block = (k%256)/32 (0..7)
    reg [11:0] sm;                  // {min6, scale6}
    reg [31:0] d1, m1, wdeq, aprod;

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0; out_valid <= 1'b0; k_cnt <= 0; k_len_r <= 0;
            for (idx = 0; idx < PE_M*PE_N; idx = idx + 1) acc[idx] <= 32'd0;
        end else begin
            out_valid <= 1'b0;

            if (start) begin
                busy    <= 1'b1;
                k_cnt   <= 0;
                k_len_r <= k_len;
                d_r     <= w_d;
                dmin_r  <= w_dmin;
                scales_r<= w_scales;
                for (idx = 0; idx < PE_M*PE_N; idx = idx + 1) acc[idx] <= 32'd0;
            end else if (busy && in_valid) begin
                sb  = k_cnt >> 8;                          // k / 256 (super-block)
                sub = k_cnt[7:5];                          // (k%256) / 32 (sub-block)
                // per-column dequant + per-cell fp32 MAC (sequential accumulate)
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    col  = pj*NSB + sb;                    // (col, super-block) index
                    sm   = q4k_scale_min({1'b0, sub}, scales_r[96*col +: 96]);
                    d1   = fp32_mul(fp16_to_fp32(d_r[16*col +: 16]),    u7_to_fp32({1'b0, sm[5:0]}));
                    m1   = fp32_mul(fp16_to_fp32(dmin_r[16*col +: 16]), u7_to_fp32({1'b0, sm[11:6]}));
                    // w = d1*q - m1   (subtract = add with m1 sign flipped)
                    wdeq = fp32_add(fp32_mul(d1, u7_to_fp32({3'd0, w_q[4*pj +: 4]})),
                                    {~m1[31], m1[30:0]});
                    for (pi = 0; pi < PE_M; pi = pi + 1) begin
                        aprod = fp32_mul(bf16_to_fp32(a_col[16*pi +: 16]), wdeq);
                        acc[pi*PE_N + pj] <= fp32_add(acc[pi*PE_N + pj], aprod);
                    end
                end
                k_cnt <= k_cnt + 1'b1;
                if (k_cnt + 1'b1 == k_len_r) begin
                    busy      <= 1'b0;
                    out_valid <= 1'b1;                     // acc is final NEXT cycle...
                end
            end
        end
    end

    // ---- output: round each fp32 accumulator to bf16 ----
    // out_valid is asserted the cycle the last beat's accumulate is registered; the
    // combinational read of acc[] below reflects the final sums on that same edge.
    always @(*) begin
        for (idx = 0; idx < PE_M*PE_N; idx = idx + 1)
            c_out[16*idx +: 16] = fp32_to_bf16(acc[idx]);
    end
endmodule
/* verilator lint_on DECLFILENAME */
