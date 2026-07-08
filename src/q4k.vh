`ifndef Q4K_VH
`define Q4K_VH
//============================================================================
// q4k.vh  --  GGML Q4_K PRIMITIVES  (the UD-Q4_K_XL local-device numerics)
//----------------------------------------------------------------------------
// PURPOSE
//   GLM-5.2 for a LOCAL DEVICE ships as GGUF k-quants (unsloth/GLM-5.2-GGUF :
//   UD-Q4_K_XL).  This header is the single canonical definition of the GGML
//   Q4_K weight numerics so the Q4_K-typed weights run with NO re-quantization --
//   bit-exact to the ggml Q4_K reference `dequantize_row_q4_K` (tools/q4k_ref.py
//   golden).  (The dynamic UD-Q4_K_XL mix also keeps some Q6_K/Q8_0/F16 tensors
//   NOT yet consumed by this Q4_K-only datapath.)  Every
//   function here is a SYNTHESIZABLE, PURELY COMBINATIONAL `function automatic`
//   (no state, no clocks), included via `include "q4k.vh"`.
//
// FORMAT  --  GGML Q4_K super-block (QK_K = 256 weights, 144 bytes)
//   ggml_half d;         // fp16 super-block SCALE
//   ggml_half dmin;      // fp16 super-block MIN
//   uint8_t   scales[12] // 8x 6-bit block-scales + 8x 6-bit block-mins, packed
//   uint8_t   qs[128]    // 256x 4-bit quant codes q in [0,15]
//   8 sub-blocks of 32 weights; sub-block b has 6-bit (sc_b, m_b).
//   DEQUANT per weight:  w = (d*sc_b)*q - (dmin*m_b)     (all fp32).
//   d,dmin are fp16->fp32; sc_b,m_b are 6-bit ints; q is a 4-bit int.
//
// API  (all `function automatic`, combinational)
//   fp16_to_fp32  (input [15:0] h)                 -> [31:0]   exact IEEE half->single
//   q4k_scale_min (input [3:0] j, input [95:0] sc) -> [11:0]   {min6, scale6} for sub-block j
//============================================================================

//----------------------------------------------------------------------------
// fp16_to_fp32 : exact IEEE-754 half -> single.  Handles signed zero, subnormal
//   (normalized into a fp32 normal), normal, infinity, and NaN.  Bit-identical
//   to numpy float16->float32 (the golden's GGML_FP16_TO_FP32).
function automatic [31:0] fp16_to_fp32(input [15:0] h);
    reg        s;
    reg [4:0]  e;
    reg [9:0]  m;
    reg [7:0]  fe;
    reg [22:0] fm;
    reg [9:0]  mm;
    integer    i, sh;
    begin
        s = h[15]; e = h[14:10]; m = h[9:0];
        if (e == 5'd0) begin
            if (m == 10'd0) begin
                fp16_to_fp32 = {s, 31'd0};                 // signed zero
            end else begin
                // subnormal: value = m * 2^-24 ; normalize so leading 1 is at bit 9
                mm = m; sh = 0;
                for (i = 0; i < 10; i = i + 1)
                    if (mm[9] == 1'b0) begin mm = mm << 1; sh = sh + 1; end
                fe = 8'd112 - sh[7:0];                      // fp32 exp = 112 - sh
                fm = {mm[8:0], 14'd0};                      // drop leading 1, 9 bits -> top
                fp16_to_fp32 = {s, fe, fm};
            end
        end else if (e == 5'd31) begin
            // infinity (m==0) or NaN (m!=0, quieted)
            fp16_to_fp32 = (m == 10'd0) ? {s, 8'hFF, 23'd0}
                                        : {s, 8'hFF, 1'b1, m[8:0], 13'd0};
        end else begin
            fe = {3'd0, e} + 8'd112;                        // e - 15 + 127
            fm = {m, 13'd0};
            fp16_to_fp32 = {s, fe, fm};
        end
    end
endfunction

//----------------------------------------------------------------------------
// q4k_scale_min : ggml get_scale_min_k4 -- unpack the j-th (0..7) 6-bit block
//   SCALE and MIN from the 12-byte `scales` super-block field.  Returns
//   {min6[5:0], scale6[5:0]}.  `sc` is the 96-bit scales field, byte i = sc[8*i +: 8].
function automatic [11:0] q4k_scale_min(input [3:0] j, input [95:0] sc);
    reg [7:0] b_j, b_jp4, b_jm4;
    reg [5:0] d6, m6;
    begin
        b_j   = sc[8*j       +: 8];    // scales[j]
        b_jp4 = sc[8*(j + 4) +: 8];    // scales[j+4]
        b_jm4 = sc[8*(j - 4) +: 8];    // scales[j-4]  (used only when j >= 4)
        if (j < 4) begin
            d6 = b_j[5:0];                        // scales[j]   & 63
            m6 = b_jp4[5:0];                       // scales[j+4] & 63
        end else begin
            d6 = {b_jm4[7:6], b_jp4[3:0]};         // (s[j+4]&0xF) | ((s[j-4]>>6)<<4)
            m6 = {b_j[7:6],   b_jp4[7:4]};         // (s[j+4]>>4)  | ((s[j]  >>6)<<4)
        end
        q4k_scale_min = {m6, d6};
    end
endfunction

//----------------------------------------------------------------------------
// u7_to_fp32 : exact unsigned-int (0..127) -> fp32.  Covers the 6-bit block
//   scale/min (0..63) and the 4-bit quant code (0..15) that multiply the fp16
//   scales in the Q4_K dequant.  Exact (numpy np.float32(int) equivalent).
function automatic [31:0] u7_to_fp32(input [6:0] x);
    integer  i, p;
    reg [29:0] sh;
    begin
        if (x == 7'd0) begin
            u7_to_fp32 = 32'd0;
        end else begin
            p = 0;
            for (i = 0; i < 7; i = i + 1) if (x[i]) p = i;   // highest set bit
            sh = {23'd0, x} << (23 - p);                     // leading 1 -> bit 23
            u7_to_fp32 = {1'b0, (8'd127 + p[7:0]), sh[22:0]};
        end
    end
endfunction

`endif // Q4K_VH
