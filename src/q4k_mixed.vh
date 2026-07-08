`ifndef Q4K_MIXED_VH
`define Q4K_MIXED_VH
`include "glm_fp.vh"   // fp32_mul : the shared IEEE-fp32 MAC numerics (== numpy)
`include "q4k.vh"      // fp16_to_fp32, s8_to_fp32 (+ the proven Q4_K primitives)
//============================================================================
// q4k_mixed.vh  --  MIXED-TYPE (Q6_K / Q8_0 / F16) DEQUANT PRIMITIVES
//----------------------------------------------------------------------------
// PURPOSE
//   The published unsloth/GLM-5.2-GGUF:UD-Q4_K_XL checkpoint is a DYNAMIC mix:
//   most tensors Q4_K, quality-sensitive ones Q6_K / Q8_0 / F16.  q4k.vh already
//   defines the Q4_K numerics; this header adds the per-weight dequant for the
//   THREE higher-precision types, BIT-EXACT to the ggml goldens in
//   tools/q4k_ref.py (dequantize_block_q6_K / q8_0 / fp16->fp32).
//
// DESIGN INVARIANT (unchanged from the Q4_K path)
//   Every type dequantizes to ONE fp32 weight `wdeq`, which feeds the SAME shared
//   fp32 MAC (fp32_mul/fp32_add, bf16 activations fed direct, bf16 RNE out).  A
//   tile/column is exactly one type, so the consumer selects the active decoder
//   with a cheap per-column w_type mux -- never on the MAC critical path.  All
//   functions here are SYNTHESIZABLE, PURELY COMBINATIONAL `function automatic`
//   (no state, no clocks, no latch, no comb loop), same idiom as q4k.vh.
//
// EXACT ggml NUMERICS mirrored (from tools/q4k_ref.py):
//   Q6_K : q = int8( ((ql&0xF)|((qh>>k&3)<<4)) - 32 )  (signed 6-bit),
//          w = ( d * f32(int8 sc[is]) ) * f32(q)        // (d*sc)*q, left-assoc
//          d fp16; sc INT8; scale index of y-position p is (p>>4).
//   Q8_0 : w = d * f32(int8 qs)                          // d fp16, no offset
//   F16  : w = fp16_to_fp32(raw16)                       // passthrough
//
// API  (all `function automatic`, combinational)
//   q6k_assemble(ql, qh, sel)        -> [7:0]   signed (code-32) from raw ql/qh
//   q6k_deq (d16, sc8, code6)        -> [31:0]  Q6_K wdeq from PRE-ASSEMBLED code
//   q6k_deq_raw(d16, sc8, ql, qh, s) -> [31:0]  Q6_K wdeq straight from raw ql/qh
//   q8_0_deq(d16, qs8)               -> [31:0]  Q8_0 wdeq
//   f16_deq (raw16)                  -> [31:0]  F16  wdeq (passthrough)
//============================================================================

//----------------------------------------------------------------------------
// q6k_assemble : ggml Q6_K 6-bit assemble + signed (-32) offset, from raw bytes.
//   `sel` (0..3) picks which of the 4 weights co-packed in one (ql,qh) pair maps
//   to y-offsets 0/32/64/96 (the caller presents the correct ql byte -- l vs
//   l+32 -- per sel, exactly as ggml's inner loop does):
//     nibble = sel[1] ? ql[7:4] : ql[3:0]      (low nibble for sel 0/1, high 2/3)
//     hi2    = (qh >> (2*sel)) & 3             (2-bit high field: shift 0/2/4/6)
//     code   = {hi2, nibble}  (0..63)  ;  return int8(code - 32)  (-32..31)
//   Returned as an 8-bit two's-complement value -> feed s8_to_fp32 directly.
function automatic [7:0] q6k_assemble(input [7:0] ql, input [7:0] qh, input [1:0] sel);
    reg [1:0] hi2;
    reg [3:0] nib;
    begin
        hi2 = (qh >> (2 * sel)) & 2'b11;                  // 2-bit reg truncates
        nib = sel[1] ? ql[7:4] : ql[3:0];
        q6k_assemble = {2'b00, hi2, nib} - 8'd32;         // {2'b00,hi2,nib}=0..63
    end
endfunction

//----------------------------------------------------------------------------
// q6k_deq : Q6_K per-weight dequant from a PRE-ASSEMBLED 6-bit code (0..63) --
//   the recommended stream path (the packer pre-assembles per-beat codes, just as
//   the Q4_K loader pre-reorders qs into per-beat codes).  code6 - 32 = signed q;
//   wdeq = (d * f32(sc)) * f32(q), grouped to match numpy's left-assoc d*sc*q.
function automatic [31:0] q6k_deq(input [15:0] d16, input [7:0] sc8, input [5:0] code6);
    reg [7:0] qsig;
    begin
        qsig     = {2'b00, code6} - 8'd32;                // int8(code-32), -32..31
        q6k_deq  = fp32_mul(fp32_mul(fp16_to_fp32(d16), s8_to_fp32(sc8)),
                            s8_to_fp32(qsig));
    end
endfunction

//----------------------------------------------------------------------------
// q6k_deq_raw : Q6_K per-weight dequant straight from raw ql/qh (assemble + arith
//   in one call) -- for a fully-general loader that streams raw ql/qh.  Identical
//   numerics to q6k_deq; only the code source (raw assemble vs pre-assembled)
//   differs.  `sel`/`ql` conventions as in q6k_assemble.
function automatic [31:0] q6k_deq_raw(input [15:0] d16, input [7:0] sc8,
                                      input [7:0] ql, input [7:0] qh, input [1:0] sel);
    reg [7:0] qsig;
    begin
        qsig        = q6k_assemble(ql, qh, sel);          // already int8(code-32)
        q6k_deq_raw = fp32_mul(fp32_mul(fp16_to_fp32(d16), s8_to_fp32(sc8)),
                               s8_to_fp32(qsig));
    end
endfunction

//----------------------------------------------------------------------------
// q8_0_deq : Q8_0 per-weight dequant.  w = d * f32(int8 qs)  (d fp16 per 32-weight
//   block, qs signed int8, no offset).
function automatic [31:0] q8_0_deq(input [15:0] d16, input [7:0] qs8);
    begin
        q8_0_deq = fp32_mul(fp16_to_fp32(d16), s8_to_fp32(qs8));
    end
endfunction

//----------------------------------------------------------------------------
// f16_deq : F16 passthrough.  w = fp16_to_fp32(raw16)  (no scale; the proven
//   q4k.vh fp16_to_fp32 handles signed zero / subnormal / normal / inf / nan).
//   NaN NOTE: fp16_to_fp32 CANONICALIZES NaN payloads (quiets signaling-NaN) per
//   the glm_fp.vh NaN policy -- so it diverges from numpy on sNaN payload bits
//   ONLY; all finite values and +/-inf match bit-exact.  Real GGUF weight tensors
//   are never NaN, so F16 weight consumption is unaffected by this canonicalization.
function automatic [31:0] f16_deq(input [15:0] raw16);
    begin
        f16_deq = fp16_to_fp32(raw16);
    end
endfunction

`endif // Q4K_MIXED_VH
