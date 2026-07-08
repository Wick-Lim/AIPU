#!/usr/bin/env python3
"""glm_model_q4k_ref.py -- ASSEMBLED numpy golden of the FULL glm_model_q4k forward.

WHAT THIS IS
  A numpy reference of one decode step of src/glm_model_q4k.v (embed -> L decoder
  layers [MLA attention + dense/MoE FFN] -> final RMSNorm -> LM head -> argmax),
  at the committed slice config.  It is COMPOSED from the proven operator semantics
  (glm_matmul_q4k / swiglu_expert_q4k / moe_router_q4k / mla_attn_q4k and the shared
  bf16 units rmsnorm_unit / rope_interleave_unit / glm_softmax / dsa_indexer /
  topk_select / glm_act) and REUSES tools/q4k_ref.py's fp32/bf16 primitives
  (dequant + bf16_round + fp32 matmul), so the numerics match the RTL bit-for-bit
  where the RTL is bit-exact and functionally where it is functional.

THE ONE INTENTIONAL DIVERGENCE FROM TODAY'S RTL  (Phase-1 finding)
  This golden APPLIES the MLA softmax scale 1/sqrt(qk_head_dim) to the q.K score
  before softmax (score = bf16(f32(bf16(dot)) * (1/sqrt(NOPE+ROPE)))), which the
  current src/mla_attn_q4k.v OMITS.  Driving glm_model_q4k against this golden is
  meant to EXPOSE that missing scale; the DUT is then fixed to match.  Everything
  else is a faithful bit-reproduction of the RTL datapath.

NUMERICS CONTRACT (reproduced from src/glm_fp.vh + src/glm_fp_pipe.v + src/q4k.vh)
  * fp32 mul/add            : numpy float32 (the proven tools/q4k_ref.py semantics).
  * fp32->bf16 (RNE)        : q4k_ref.bf16_round.
  * Q4_K weight dequant     : w = (d*sc_b)*q - (dmin*m_b), sub-block b = k//32,
                              reusing q4k_ref.get_scale_min_k4 (bit-exact to ggml).
  * Q4_K GEMM               : q4k_ref.matmul_q4k_col (sequential fp32 MAC in K order).
  * bf16 GEMM (score/LM head): glm_matmul_pipe's L=7-way interleaved partial sums
                              + 3-level add-tree (fp32 add is non-associative, so the
                              GROUPING is part of the defined numerics -- reproduced).
  * fp32_rsqrt              : Quake seed 0x5F3759DF + 2 Newton iters (glm_fp.vh).
  * exp (softmax)           : glm_exp_ref -- range-reduce + degree-5 Horner + 2^k fold
                              (src/glm_fp_pipe.v), 0 ULP to the pipelined fp32_exp_pipe.
  * silu / sigmoid          : glm_act poly (clamp, k=round(z*log2e), degree-5 Horner
                              exp, 1/d via rsqrt^2), src/glm_act.v.
  * RoPE                    : rope_interleave_unit -- Q48 turn ROM (elaboration integer
                              log2/exp2) + quadrant-fold fp32 Taylor cos/sin.
  * RMSNorm                 : Sx^2 fp32 reduce -> rsqrt(mean+eps) -> *gamma, bf16 out.

Run `python3 tools/glm_model_q4k_ref.py` for the self-test.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4k_ref
from q4k_ref import get_scale_min_k4, bf16_round, matmul_q4k_col

F32 = np.float32
U32 = np.uint32
U16 = np.uint16

# ===========================================================================
# 0.  bit helpers + fp32 primitives (src/glm_fp.vh contract)
# ===========================================================================
def f32_bits(x):
    return int(F32(x).view(U32))

def bits_f32(u):
    return F32(np.uint32(u & 0xFFFFFFFF).view(np.float32))

def fp16_to_f32(h):
    """exact IEEE half->single (q4k.vh fp16_to_fp32 == numpy)."""
    return F32(np.frombuffer(U16(h & 0xFFFF).tobytes(), dtype=np.float16)[0])

def f32_to_f16bits(x):
    return int(np.frombuffer(np.float16(x).tobytes(), dtype=U16)[0])

def bf16_bits(x):
    """fp32 value -> 16-bit bf16 pattern (RNE via q4k_ref.bf16_round)."""
    return (f32_bits(bf16_round(x)) >> 16) & 0xFFFF

def bf16_from_bits(u16):
    """16-bit bf16 pattern -> bf16-valued fp32."""
    return bits_f32((u16 & 0xFFFF) << 16)

def bf16(x):
    """round an fp32 to a bf16-valued fp32 (src/glm_fp.vh fp32_to_bf16)."""
    return bf16_round(x)

def b2f(x):
    """bf16-valued fp32 -> fp32 (glm_fp.vh bf16_to_fp32: lossless widen)."""
    return F32(x)

# fp32 mul/add: numpy float32 (identical to tools/q4k_ref.py's proven semantics).
def fmul(a, b):
    return F32(F32(a) * F32(b))

def fadd(a, b):
    return F32(F32(a) + F32(b))

def fneg(x):
    return bits_f32(f32_bits(x) ^ 0x80000000)

QNAN = bits_f32(0x7FC00000)

def fp32_rsqrt(x):
    """glm_fp.vh fp32_rsqrt: Quake magic seed + 2 Newton iters (y*(1.5-0.5*x*y*y))."""
    x = F32(x)
    u = f32_bits(x)
    exp = (u >> 23) & 0xFF
    man = u & 0x7FFFFF
    sign = (u >> 31) & 1
    if (exp == 0xFF and man != 0) or sign == 1 or exp == 0:      # nan / x<0 / zero(FTZ)
        return QNAN
    if exp == 0xFF and man == 0:                                  # +inf -> +0
        return F32(0.0)
    half = bits_f32(0x3F000000)
    three_half = bits_f32(0x3FC00000)
    xhalf = fmul(half, x)
    y = bits_f32((0x5F3759DF - (u >> 1)) & 0xFFFFFFFF)
    for _ in range(2):
        yy = fmul(y, y)
        xyy = fmul(xhalf, yy)
        t = fadd(three_half, fneg(xyy))         # 1.5 - 0.5*x*y*y
        y = fmul(y, t)
    return y

def fp32_gt(a, b):
    """glm_softmax / topk_select fp32 sign-magnitude '>' (NaN smallest, +0==-0)."""
    a = F32(a); b = F32(b)
    ua = f32_bits(a); ub = f32_bits(b)
    sa = (ua >> 31) & 1; sb = (ub >> 31) & 1
    ma = ua & 0x7FFFFFFF; mb = ub & 0x7FFFFFFF
    a_nan = ((ua >> 23) & 0xFF) == 0xFF and (ua & 0x7FFFFF) != 0
    b_nan = ((ub >> 23) & 0xFF) == 0xFF and (ub & 0x7FFFFF) != 0
    if a_nan:
        return False
    if b_nan:
        return True
    if ma == 0 and mb == 0:
        return False
    if sa != sb:
        return sa == 0
    if sa == 0:
        return ma > mb
    return ma < mb

def bf16_gt(a_u16, b_u16):
    """glm_model_q4k argmax compare: strict '>' on bf16 patterns (lower-index tie)."""
    sa = (a_u16 >> 15) & 1; sb = (b_u16 >> 15) & 1
    ma = a_u16 & 0x7FFF; mb = b_u16 & 0x7FFF
    if sa != sb:
        if ma == 0 and mb == 0:
            return False
        return sb == 1
    if sa == 0:
        return ma > mb
    return ma < mb

# ===========================================================================
# 1.  exp for softmax  (src/glm_fp_pipe.v glm_exp_ref, 0 ULP vs fp32_exp_pipe)
# ===========================================================================
_LN2 = bits_f32(0x3F317218)
_INV_LN2 = bits_f32(0x3FB8AA3B)
_C1 = bits_f32(0x3F000000); _C2 = bits_f32(0x3E2AAAAB)
_C3 = bits_f32(0x3D2AAAAB); _C4 = bits_f32(0x3C088889)
_ONE = bits_f32(0x3F800000)

def _fp32_to_int10_rne(f):
    """glm_fp_pipe.v fp32_to_int10_rne (bounded fp32 -> signed int, round-half-up)."""
    u = f32_bits(f)
    s = (u >> 31) & 1
    e = (u >> 23) & 0xFF
    m = 0x800000 | (u & 0x7FFFFF)          # 24-bit {1,frac}
    mag = 0
    if e >= 127:
        if e >= 150:
            mag = 511
        else:
            sh = 150 - e
            shifted = m >> sh
            mag = 511 if (shifted >> 10) != 0 else (shifted & 0x3FF)
            if (m >> (sh - 1)) & 1:
                mag += 1
    return -mag if s else mag

def _int10_to_fp32(iv):
    """glm_fp_pipe.v int10_to_fp32 (exact for |iv| < 512)."""
    if iv == 0:
        return F32(0.0)
    s = 1 if iv < 0 else 0
    mag = -iv if iv < 0 else iv
    msb = mag.bit_length() - 1
    e = 127 + msb
    frac = (mag << (23 - msb)) & 0x7FFFFF
    return bits_f32((s << 31) | (e << 23) | frac)

def glm_exp(x):
    """glm_exp_ref(x): range-reduce k=round(x/ln2), r=x-k*ln2, degree-5 Horner, 2^k fold."""
    xv = F32(x)
    kf = fmul(xv, _INV_LN2)
    ki = _fp32_to_int10_rne(kf)
    kln2 = fmul(_int10_to_fp32(ki), _LN2)
    r = fadd(xv, fneg(kln2))
    poly = fadd(_C3, fmul(_C4, r))
    poly = fadd(_C2, fmul(poly, r))
    poly = fadd(_C1, fmul(poly, r))
    poly = fadd(_ONE, fmul(poly, r))
    poly = fadd(_ONE, fmul(poly, r))
    pu = f32_bits(poly)
    e = (pu >> 23) & 0xFF
    new_e = e + ki                          # e>=0, ki signed
    if e == 0:
        return F32(0.0)
    if new_e >= 255:
        return bits_f32((pu & 0x80000000) | (0xFF << 23))     # overflow -> inf
    if new_e <= 0:
        return F32(0.0)                                       # underflow -> FTZ
    return bits_f32((pu & 0x80000000) | ((new_e & 0xFF) << 23) | (pu & 0x7FFFFF))

# ===========================================================================
# 2.  glm_act : silu / sigmoid  (src/glm_act.v, poly exp + rsqrt^2 reciprocal)
# ===========================================================================
_LOG2E = bits_f32(0x3FB8AA3B)
_ACT_LN2 = bits_f32(0x3F317218)
_X_SAT = bits_f32(0x41800000)               # 16.0
_K_MAX = 64

def _act_round_to_int(f):
    """glm_act.v fp32_round_to_int (|arg|<=~23, round-half-up)."""
    u = f32_bits(f)
    s = (u >> 31) & 1
    e = (u >> 23) & 0xFF
    m = 0x800000 | (u & 0x7FFFFF)
    if e < 127:
        r = 1 if e == 126 else 0
    else:
        rsh = 23 - (e - 127)
        shifted = m >> rsh
        mag = shifted & 0xFF
        frac_half = (m >> (rsh - 1)) & 1
        r = mag + (1 if frac_half else 0)
    return -r if s else r

def _act_int_to_fp32(k):
    """glm_act.v int_to_fp32 (|k| <= 64)."""
    if k == 0:
        return F32(0.0)
    s = 1 if k < 0 else 0
    a = -k if k < 0 else k
    msb = a.bit_length() - 1
    e = 127 + msb
    mant = (a << (23 - msb)) & 0x7FFFFF
    return bits_f32((s << 31) | (e << 23) | mant)

def _act_sanitize(x):
    u = f32_bits(x)
    if ((u >> 23) & 0xFF) == 0xFF:                      # inf/nan -> signed X_SAT
        return bits_f32((u & 0x80000000) | (f32_bits(_X_SAT) & 0x7FFFFFFF))
    return x

def _act_clamp(x):
    u = f32_bits(x)
    mag = u & 0x7FFFFFFF
    if mag >= (f32_bits(_X_SAT) & 0x7FFFFFFF):
        return bits_f32((u & 0x80000000) | (f32_bits(_X_SAT) & 0x7FFFFFFF))
    return x

def _act_exp_poly(r):
    """degree-5 Horner exp(r), |r|<=ln2/2 (glm_act exp_poly_lo+hi)."""
    p = fadd(bits_f32(0x3D2AAAAB), fmul(r, bits_f32(0x3C088889)))   # 1/24 + r/120
    p = fadd(bits_f32(0x3E2AAAAB), fmul(r, p))                      # 1/6  + r*p
    p = fadd(bits_f32(0x3F000000), fmul(r, p))                      # 1/2  + r*p
    p = fadd(_ONE, fmul(r, p))                                      # 1 + r*p
    p = fadd(_ONE, fmul(r, p))                                      # 1 + r*p  (full)
    return p

def _act_scale_pow2(v, k):
    """glm_act scale_pow2: 2^k * v by adding k to v's biased exponent (clamped)."""
    u = f32_bits(v)
    e_new = ((u >> 23) & 0xFF) + k
    sgn = u & 0x80000000
    if e_new >= 255:
        return bits_f32(sgn | (0xFE << 23) | (u & 0x7FFFFF))      # clamp below inf
    if e_new <= 0:
        return bits_f32(sgn)                                       # FTZ
    return bits_f32(sgn | ((e_new & 0xFF) << 23) | (u & 0x7FFFFF))

def _sigmoid_fp32(xbf):
    """glm_act sigmoid core -> fp32 s (unrounded)."""
    xcl = _act_clamp(_act_sanitize(b2f(xbf)))
    z = fneg(xcl)
    kf = fmul(z, _LOG2E)
    k = _act_round_to_int(kf)
    k = max(-_K_MAX, min(_K_MAX, k))
    r = fadd(z, fneg(fmul(_act_int_to_fp32(k), _ACT_LN2)))
    pr = _act_exp_poly(r)
    ex = _act_scale_pow2(pr, k)
    d = fadd(_ONE, ex)
    t = fp32_rsqrt(d)
    return fmul(t, t)                                              # 1/d = rsqrt(d)^2

def glm_sigmoid(xbf):
    return bf16(_sigmoid_fp32(xbf))

def glm_silu(xbf):
    s = _sigmoid_fp32(xbf)
    xraw = _act_sanitize(b2f(xbf))
    return bf16(fmul(s, xraw))

# ===========================================================================
# 3.  RoPE  (src/rope_interleave_unit.v: Q48 turn ROM + fp32 Taylor cos/sin)
# ===========================================================================
_INV_2PI_Q64 = 0x28BE60DB9391054A
_FP_PI_HALF = bits_f32(0x3FC90FDB)
_ROPE_S = [bits_f32(v) for v in (0x3F800000, 0xBE2AAAAB, 0x3C088889, 0xB9500D01, 0x3638EF1D)]
_ROPE_C = [bits_f32(v) for v in (0x3F800000, 0xBF000000, 0x3D2AAAAB, 0xBAB60B61, 0x37D00D01)]

_EXP2_TAB = {
    1: 0x016A09E667F3BCD0, 2: 0x01306FE0A31B7150, 3: 0x01172B83C7D517B0,
    4: 0x010B5586CF9890F0, 5: 0x01059B0D31585740, 6: 0x0102C9A3E7780610,
    7: 0x010163DA9FB33350, 8: 0x0100B1AFA5ABCBF0, 9: 0x010058C86DA1C0A0,
    10: 0x01002C605E2E8CF0, 11: 0x0100162F39040520, 12: 0x01000B175EFFDC70,
    13: 0x0100058BA01FBA00, 14: 0x010002C5CC37DA90, 15: 0x01000162E525EE00,
    16: 0x010000B172557760, 17: 0x01000058B91B5BD0, 18: 0x0100002C5C89D5F0,
    19: 0x010000162E43F500, 20: 0x0100000B1721BD00, 21: 0x010000058B90CF20,
    22: 0x01000002C5C863B0, 23: 0x0100000162E430E0, 24: 0x01000000B1721830,
    25: 0x0100000058B90C10, 26: 0x010000002C5C8600, 27: 0x01000000162E4300,
    28: 0x010000000B172180, 29: 0x01000000058B90C0, 30: 0x0100000002C5C860,
    31: 0x010000000162E430, 32: 0x0100000000B17210, 33: 0x010000000058B910,
    34: 0x01000000002C5C80, 35: 0x0100000000162E40, 36: 0x01000000000B1720,
    37: 0x0100000000058B90, 38: 0x010000000002C5D0, 39: 0x01000000000162E0,
    40: 0x010000000000B170, 41: 0x01000000000058C0, 42: 0x0100000000002C60,
    43: 0x0100000000001630, 44: 0x0100000000000B10, 45: 0x0100000000000590,
    46: 0x01000000000002C0, 47: 0x0100000000000160, 48: 0x01000000000000B0,
    49: 0x0100000000000060, 50: 0x0100000000000030, 51: 0x0100000000000010,
    52: 0x0100000000000010,
}

def _log2_q56(x):
    """rope_interleave_unit log2_q56: log2(x) as Q56 fixed-point (x>=1)."""
    ip = x.bit_length() - 1
    acc = ip << 56
    z = (x >> (ip - 56)) if ip >= 56 else (x << (56 - ip))
    for i in range(56):
        z = (z * z) >> 56
        if z >= (1 << 57):
            acc |= (1 << (55 - i))
            z >>= 1
    return acc

def _exp2_frac_q56(f):
    """rope_interleave_unit exp2_frac_q56: 2^f for f in [0,1) (Q56 -> Q56)."""
    r = 1 << 56
    for k in range(1, 57):
        if (f >> (56 - k)) & 1:
            tab = _EXP2_TAB.get(k, 1 << 56)
            r = (r * tab) >> 56
    return r

def _invf_q56(idx, rot_dim, theta):
    l2t = _log2_q56(theta)
    e_q56 = ((2 * idx) * l2t) // rot_dim
    E = (56 << 56) - e_q56                          # (56 - e) Q56, >0 in range
    P = E >> 56                                       # floor integer part
    fr = E & ((1 << 56) - 1)
    m = _exp2_frac_q56(fr)
    return (m << (P - 56)) if P >= 56 else (m >> (56 - P))

def _turn_per_pos(idx, rot_dim, theta, bfr=48):
    prod = _invf_q56(idx, rot_dim, theta) * _INV_2PI_Q64
    sh = 120 - bfr                                    # = 72
    prod += 1 << (sh - 1)                             # round to nearest
    return prod >> sh                                  # Q48 turns/pos

def _frac46_to_fp32(fbits):
    """rope_interleave_unit frac46_to_fp32: 46-bit fraction (value/2^46) -> fp32."""
    if fbits == 0:
        return F32(0.0)
    msb = fbits.bit_length() - 1                      # 0..45
    e = msb - 46
    norm = (fbits >> (msb - 23)) if msb >= 23 else (fbits << (23 - msb))
    m23 = norm & 0x7FFFFF
    return bits_f32(((e + 127) & 0xFF) << 23 | m23)

def _cossin_quad(th):
    """cos/sin of th in [0,pi/2) via fp32 Horner Taylor (u=th^2)."""
    u = fmul(th, th)
    sp = fadd(_ROPE_S[3], fmul(u, _ROPE_S[4]))
    sp = fadd(_ROPE_S[2], fmul(u, sp))
    sp = fadd(_ROPE_S[1], fmul(u, sp))
    sp = fadd(_ROPE_S[0], fmul(u, sp))
    sinv = fmul(th, sp)
    cp = fadd(_ROPE_C[3], fmul(u, _ROPE_C[4]))
    cp = fadd(_ROPE_C[2], fmul(u, cp))
    cp = fadd(_ROPE_C[1], fmul(u, cp))
    cosv = fadd(_ROPE_C[0], fmul(u, cp))
    return cosv, sinv

def _cossin_turn(frac):
    """cos/sin from a Q48 turn fraction: [47:46]=quadrant, [45:0] -> theta in [0,pi/2)."""
    q = (frac >> 46) & 3
    r = _frac46_to_fp32(frac & ((1 << 46) - 1))
    th = fmul(r, _FP_PI_HALF)
    cosv, sinv = _cossin_quad(th)
    if q == 0:
        return cosv, sinv
    if q == 1:
        return fneg(sinv), cosv
    if q == 2:
        return fneg(cosv), fneg(sinv)
    return sinv, fneg(cosv)

_turn_rom_cache = {}

def _turn_rom(rot_dim, theta):
    key = (rot_dim, theta)
    rom = _turn_rom_cache.get(key)
    if rom is None:
        rom = [_turn_per_pos(p, rot_dim, theta) for p in range(rot_dim // 2)]
        _turn_rom_cache[key] = rom
    return rom

def rope_apply(vec_bf, pos, rot_dim, theta):
    """Interleaved RoPE of a rot_dim bf16 vector at position `pos`.
       Pair p: (x0=vec[2p], x1=vec[2p+1]) rotated by pos*inv_freq[p]; bf16 out."""
    rom = _turn_rom(rot_dim, theta)
    out = [F32(0.0)] * rot_dim
    for p in range(rot_dim // 2):
        frac = (pos * rom[p]) & ((1 << 48) - 1)
        cosv, sinv = _cossin_turn(frac)
        x0 = b2f(vec_bf[2 * p]); x1 = b2f(vec_bf[2 * p + 1])
        y0 = fadd(fmul(x0, cosv), fneg(fmul(x1, sinv)))    # x0*cos - x1*sin
        y1 = fadd(fmul(x0, sinv), fmul(x1, cosv))          # x0*sin + x1*cos
        out[2 * p] = bf16(y0)
        out[2 * p + 1] = bf16(y1)
    return out

# ===========================================================================
# 4.  RMSNorm  (src/rmsnorm_unit.v, LANES=1: Sx^2 fp32 -> rsqrt(mean+eps) -> *gamma)
# ===========================================================================
_EPS = bits_f32(0x3727C5AC)                 # 1e-5

def _recip_fp32(n):
    """rmsnorm_unit recip_fp32(N): elaboration-exact fp32 bit pattern of 1/N."""
    SH = 60
    q = (1 << SH) // n
    msb = q.bit_length() - 1
    expo = (msb - SH) + 127
    qs = q << (63 - msb)
    mant = (qs >> 40) & 0x7FFFFF
    guard = (qs >> 39) & 1
    sticky = 1 if (qs & ((1 << 39) - 1)) else 0
    roundup = guard & (sticky | (mant & 1))
    mant_r = mant + roundup
    if mant_r >> 23:
        mant_r >>= 1
        expo += 1
    return bits_f32(((expo & 0xFF) << 23) | (mant_r & 0x7FFFFF))

def rmsnorm(x_bf, gamma_bf):
    """y[i] = bf16( (x[i]*inv) * gamma[i] ),  inv = rsqrt(mean(x^2)+eps)."""
    n = len(x_bf)
    inv_len = _recip_fp32(n)
    sumsq = F32(0.0)
    for xi in x_bf:                          # fp32 reduce in element order
        xf = b2f(xi)
        sumsq = fadd(sumsq, fmul(xf, xf))
    meps = fadd(fmul(sumsq, inv_len), _EPS)
    inv = fp32_rsqrt(meps)
    out = []
    for i in range(n):
        xs = fmul(b2f(x_bf[i]), inv)
        out.append(bf16(fmul(xs, b2f(gamma_bf[i]))))
    return out

# ===========================================================================
# 5.  matmuls
# ===========================================================================
def matmul_q4k(a_row, wdeq_cols):
    """out[o] = bf16( sum_k a[k]*wdeq_cols[o][k] ), sequential fp32 MAC (q4k_ref)."""
    return [matmul_q4k_col(a_row, wc) for wc in wdeq_cols]

_MM_L = 7                                    # glm_matmul_pipe L = FP_MAC_LAT

def matmul_bf16_dot(a_bf, w_bf):
    """One glm_matmul_pipe output = L=7-way interleaved partial sums + 3-level add tree."""
    ps = [F32(0.0)] * _MM_L
    for k in range(len(a_bf)):
        prod = fmul(b2f(a_bf[k]), b2f(w_bf[k]))
        lane = k % _MM_L
        ps[lane] = fadd(ps[lane], prod)
    a01 = fadd(ps[0], ps[1]); a23 = fadd(ps[2], ps[3])
    a45 = fadd(ps[4], ps[5]); a6 = fadd(ps[6], F32(0.0))
    b0 = fadd(a01, a23); b1 = fadd(a45, a6)
    return bf16(fadd(b0, b1))

# ===========================================================================
# 6.  Q4_K weight container
# ===========================================================================
class QW:
    """A Q4_K weight matrix: NOUT output columns x NIN reduction, each column a
       super-block-scaled set of 4-bit codes.  dequant reproduces the RTL exactly
       (sub-block b = k//32; reuses q4k_ref.get_scale_min_k4)."""
    __slots__ = ("nout", "nin", "codes", "d", "dmin", "scales", "_deq")

    def __init__(self, nout, nin, codes, d, dmin, scales):
        self.nout = nout          # output columns
        self.nin = nin            # reduction length K
        self.codes = codes        # codes[o][k] in 0..15
        self.d = d                # d[o] fp16 bits
        self.dmin = dmin          # dmin[o] fp16 bits
        self.scales = scales      # scales[o] = list of 12 bytes (one super-block, K<=256)
        self._deq = None

    def dequant_col(self, o):
        d = fp16_to_f32(self.d[o])
        mn = fp16_to_f32(self.dmin[o])
        s = self.scales[o]
        col = np.empty(self.nin, dtype=np.float32)
        for k in range(self.nin):
            sc, m = get_scale_min_k4(k // 32, s)          # sub-block b = k//32
            col[k] = F32(F32(d * F32(sc)) * F32(self.codes[o][k]) - F32(mn * F32(m)))
        return col

    def dequant(self):
        if self._deq is None:
            self._deq = [self.dequant_col(o) for o in range(self.nout)]
        return self._deq


def _mk_qw(rng, nout, nin):
    """Random Q4_K weight matrix with FULL per-column super-blocks (every sub-block
       populated -- exercises the whole K reduction).  Uses the real ggml Q4_K format
       (fp16 d/dmin + 8x 6-bit sub-block scale/min), but WELL-CONDITIONED: the
       per-weight dequant w = d*sc*q - dmin*m is kept small and roughly zero-centered
       (small `sc` slope + a centering `m` offset) so a deep (L=6) stack stays O(1) in
       bf16.  Large-scale weights blow the activations to ~1e7 where bf16's 8-bit
       mantissa rounds away all input-dependent signal (a degenerate fixed point that
       hides real divergences -- e.g. the MLA softmax scale) -- so this is NOT the
       committed TB's sub-block-0-only scheme, it is a healthier, fuller stimulus.
       The dequant/matmul are format-general; only the drawn magnitudes are tuned."""
    codes = [[int(v) for v in rng.integers(0, 16, nin)] for _ in range(nout)]
    d = [f32_to_f16bits(rng.uniform(0.004, 0.012)) for _ in range(nout)]
    # dmin == d and m ~= 7.5*sc so each sub-block w = d*sc*(q - ~7.5) is ZERO-CENTERED
    # (q in 0..15, mean 7.5) with a small slope -> zero-mean, O(0.1) weights that keep
    # a deep stack well-conditioned in bf16.  A little jitter on m keeps entropy.
    dmin = list(d)
    scales = []
    for _ in range(nout):
        sc6 = [int(v) for v in rng.integers(1, 5, 8)]                       # slope: d*sc ~ 0.004-0.06
        m6 = [min(63, max(0, int(round(7.5 * s)) + int(rng.integers(-1, 2)))) for s in sc6]
        scales.append(list(q4k_ref._pack_6bit_scales(sc6, m6)))
    return QW(nout, nin, codes, d, dmin, scales)


def _mk_bf16(rng, n, lo=-1.0, hi=1.0):
    return [bf16(F32(v)) for v in rng.uniform(lo, hi, n)]


def _mk_gamma(rng, n):
    """RMSNorm learned scale ~ 1 (positive, near unity) -- realistic + keeps the
       normalized stream sign-stable."""
    return [bf16(F32(v)) for v in rng.uniform(0.6, 1.4, n)]

# ===========================================================================
# 7.  config + weights
# ===========================================================================
class Config:
    def __init__(self, **kw):
        # committed slice = glm_model_q4k.v module defaults
        self.MODEL_DIM = 128
        self.L = 6
        self.N_DENSE = 3
        self.VOCAB = 256
        self.H_HEADS = 4
        self.NOPE = 16
        self.ROPE = 16
        self.V_DIM = 32
        self.Q_LORA = 64
        self.KV_LORA = 32
        self.S_MAX = 8
        self.TOPK_ATTN = 8
        self.THETA = 8000000
        self.PE_N = 4
        self.POSW = 20
        self.N_EXPERT = 8
        self.TOPK = 2
        self.INTER_MOE = 64
        self.INTER_DENSE = 256
        self.RSCALE_BITS = 0x40200000        # 2.5 fp32
        self.TN = 4
        self.LM_TN = 4
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def QK_DIM(self):
        return self.NOPE + self.ROPE

    @property
    def HQK(self):
        return self.H_HEADS * self.QK_DIM

    @property
    def HNOPE(self):
        return self.H_HEADS * self.NOPE

    @property
    def HV(self):
        return self.H_HEADS * self.V_DIM

    @property
    def SWIN(self):
        return min(self.S_MAX, self.TOPK_ATTN)

    @property
    def sm_scale_f32(self):
        """1/sqrt(qk_head_dim) as fp32 (Phase-1 MLA softmax scale)."""
        return bits_f32(f32_bits(F32(1.0) / np.sqrt(F32(self.QK_DIM))))


# slice config used by the full-model spec TB (test/spec_decode_top_tb.v)
SPEC_SLICE = dict(MODEL_DIM=16, L=2, N_DENSE=1, VOCAB=16, H_HEADS=2, NOPE=4, ROPE=4,
                  V_DIM=4, Q_LORA=8, KV_LORA=8, S_MAX=2, TOPK_ATTN=2, PE_N=2,
                  N_EXPERT=4, TOPK=2, INTER_MOE=8, INTER_DENSE=32)


class Weights:
    """All ROMs for one model, indexed by layer.  Built deterministically from a
       seed; the phase-3 TB serves these SAME ROMs (Q4_K super-blocks for the weight
       matrices, bf16 for embeddings/norms/LM-head/KV-cache)."""
    def __init__(self, cfg, seed=424242):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        c = cfg
        self.EMB = [_mk_bf16(rng, c.MODEL_DIM) for _ in range(c.VOCAB)]
        self.G1 = []; self.G2 = []
        self.W_dq = []; self.W_uq = []; self.W_uk = []; self.W_uv = []; self.W_o = []
        self.CKV = []; self.KRP = []
        self.Wg = []; self.Dg = []; self.Du = []; self.Dd = []
        self.Mg = []; self.Mu = []; self.Md = []
        self.SHg = []; self.SHu = []; self.SHd = []
        for _ in range(c.L):
            self.G1.append(_mk_gamma(rng, c.MODEL_DIM))
            self.G2.append(_mk_gamma(rng, c.MODEL_DIM))
            self.W_dq.append(_mk_qw(rng, c.Q_LORA, c.MODEL_DIM))
            self.W_uq.append(_mk_qw(rng, c.HQK, c.Q_LORA))
            self.W_uk.append(_mk_qw(rng, c.HNOPE, c.KV_LORA))
            self.W_uv.append(_mk_qw(rng, c.HV, c.KV_LORA))
            self.W_o.append(_mk_qw(rng, c.MODEL_DIM, c.HV))
            self.CKV.append([_mk_bf16(rng, c.KV_LORA) for _ in range(c.S_MAX)])
            self.KRP.append([_mk_bf16(rng, c.ROPE) for _ in range(c.S_MAX)])
            # router W_g stored k-major: Wg[k][e]  (a QW with nout=N_EXPERT columns)
            self.Wg.append(_mk_qw(rng, c.N_EXPERT, c.MODEL_DIM))
            # dense FFN
            self.Dg.append(_mk_qw(rng, c.INTER_DENSE, c.MODEL_DIM))
            self.Du.append(_mk_qw(rng, c.INTER_DENSE, c.MODEL_DIM))
            self.Dd.append(_mk_qw(rng, c.MODEL_DIM, c.INTER_DENSE))
            # MoE experts
            self.Mg.append([_mk_qw(rng, c.INTER_MOE, c.MODEL_DIM) for _ in range(c.N_EXPERT)])
            self.Mu.append([_mk_qw(rng, c.INTER_MOE, c.MODEL_DIM) for _ in range(c.N_EXPERT)])
            self.Md.append([_mk_qw(rng, c.MODEL_DIM, c.INTER_MOE) for _ in range(c.N_EXPERT)])
            self.SHg.append(_mk_qw(rng, c.INTER_MOE, c.MODEL_DIM))
            self.SHu.append(_mk_qw(rng, c.INTER_MOE, c.MODEL_DIM))
            self.SHd.append(_mk_qw(rng, c.MODEL_DIM, c.INTER_MOE))
        self.GF = _mk_gamma(rng, c.MODEL_DIM)
        self.Wlm = [_mk_bf16(rng, c.MODEL_DIM) for _ in range(c.VOCAB)]

# ===========================================================================
# 8.  operator goldens
# ===========================================================================
def moe_router(cfg, x_bf, Wg):
    """logits=x@W_g -> sigmoid -> top-K -> renorm-then-*SCALE.  Returns
       (sel_idx list, sel_weight bf16 list, gate_bf list) reproducing moe_router_q4k."""
    c = cfg
    wcols = Wg.dequant()                                     # column e = W_g[:,e]
    logit = matmul_q4k(x_bf, wcols)                          # N_EXPERT bf16 logits
    gate_bf = [glm_sigmoid(lg) for lg in logit]              # bf16 gates
    # top-K by fp32(gate), strict-greater lower-index tie-break, descending order.
    order = sorted(range(c.N_EXPERT),
                   key=lambda e: (f32_bits_key(b2f(gate_bf[e])), -e), reverse=True)
    sel = order[:c.TOPK]
    win_gate = [b2f(gate_bf[e]) for e in sel]                # fp32 winner gates
    # s = renorm add-tree (SUMLEV=clog2(TOPK) levels).  TOPK=2 -> one add.
    s = _sum_tree(win_gate, c.TOPK)
    scale = bits_f32(c.RSCALE_BITS)
    rr = fp32_rsqrt(s)
    rs = fmul(fmul(rr, rr), scale)                           # SCALE / s
    sel_weight = [bf16(fmul(win_gate[i], rs)) for i in range(c.TOPK)]
    return sel, sel_weight, gate_bf


def f32_bits_key(x):
    """A monotone integer key for fp32_gt ordering (so sorted() replicates the tree)."""
    u = f32_bits(x)
    # map to an order-preserving unsigned key (IEEE total order for finite/inf).
    return (u ^ 0x80000000) if (u >> 31) == 0 else (~u & 0xFFFFFFFF)


def _sum_tree(vals, topk):
    """Balanced fp32 add-tree of clog2(topk) levels, +0.0 pad -- moe_router_q4k."""
    import math
    sumlev = 1 if topk <= 1 else int(math.ceil(math.log2(topk)))
    npow = 1 << sumlev
    node = [vals[i] if i < len(vals) else F32(0.0) for i in range(npow)]
    for _ in range(sumlev):
        node = [fadd(node[2 * i], node[2 * i + 1]) for i in range(len(node) // 2)]
    return node[0]


def swiglu_expert(cfg, x_bf, Wgate, Wup, Wdown):
    """y = ( silu(x@W_gate) (.) (x@W_up) ) @ W_down  -- swiglu_expert_q4k."""
    gcols = Wgate.dequant(); ucols = Wup.dequant(); dcols = Wdown.dequant()
    gate = matmul_q4k(x_bf, gcols)                           # INTER bf16
    up = matmul_q4k(x_bf, ucols)                             # INTER bf16
    h = [bf16(fmul(b2f(glm_silu(gate[i])), b2f(up[i]))) for i in range(Wgate.nout)]
    return matmul_q4k(h, dcols)                              # HIDDEN bf16


def mla_attn(cfg, x_bf, pos, s_len, layer, W):
    """MLA attention for one query token (PE_M=1), DSA dense fallback (S<=TOPK_ATTN),
       DSA_REAL_IDX=0.  Reproduces mla_attn_q4k with the Phase-1 1/sqrt(QK_DIM) score
       scale APPLIED (which today's RTL omits)."""
    c = cfg
    NOPE, ROPE, QK = c.NOPE, c.ROPE, c.QK_DIM
    # 1. q_lora = x @ W_dq ; q_lora_n = rmsnorm(q_lora, gamma=1)
    qlora = matmul_q4k(x_bf, W.W_dq[layer].dequant())
    ones_q = [bf16_from_bits(0x3F80)] * c.Q_LORA
    qlora_n = rmsnorm(qlora, ones_q)
    # 2. q_full = q_lora_n @ W_uq
    qfull = matmul_q4k(qlora_n, W.W_uq[layer].dequant())
    # 3. q_rot : NOPE part copied, ROPE part roped at `pos` (per head)
    qrot = [[F32(0.0)] * QK for _ in range(c.H_HEADS)]
    for h in range(c.H_HEADS):
        for d in range(NOPE):
            qrot[h][d] = qfull[h * QK + d]
        rope_in = [qfull[h * QK + NOPE + i] for i in range(ROPE)]
        roped = rope_apply(rope_in, pos, ROPE, c.THETA)
        for i in range(ROPE):
            qrot[h][NOPE + i] = roped[i]
    # 4. DSA dense fallback (S_MAX<=TOPK_ATTN at slice): keys 0..s_len-1 in order.
    keys = list(range(s_len))
    # 5. per key: ckv_n=rmsnorm(c_kv[j],1); knope=ckv_n@W_uk; v=ckv_n@W_uv; k_rope cached
    ones_k = [bf16_from_bits(0x3F80)] * c.KV_LORA
    scores = [[bf16_from_bits(0xFF80)] * c.SWIN for _ in range(c.H_HEADS)]   # -inf pad
    vstore = [None] * len(keys)
    scale = c.sm_scale_f32
    for slot, j in enumerate(keys):
        ckv_n = rmsnorm(W.CKV[layer][j], ones_k)
        knope = matmul_q4k(ckv_n, W.W_uk[layer].dequant())       # HNOPE
        v_j = matmul_q4k(ckv_n, W.W_uv[layer].dequant())         # HV
        krope = W.KRP[layer][j]                                   # cached, already roped
        vstore[slot] = v_j
        for h in range(c.H_HEADS):
            k_vec = [knope[h * NOPE + d] for d in range(NOPE)] + [krope[d] for d in range(ROPE)]
            q_vec = qrot[h]
            raw = matmul_bf16_dot(q_vec, k_vec)                  # bf16 dot (7-way engine)
            # PHASE-1 FIX: score = bf16( f32(bf16(dot)) * (1/sqrt(QK_DIM)) )
            scores[h][slot] = bf16(fmul(b2f(raw), scale))
    # 6. softmax per head over SWIN slots (real keys + -inf pad); LEN=SWIN
    probs = [[F32(0.0)] * c.SWIN for _ in range(c.H_HEADS)]
    for h in range(c.H_HEADS):
        p = glm_softmax([scores[h][s] for s in range(c.SWIN)])
        for s in range(s_len):
            probs[h][s] = p[s]                                    # slots>=s_len forced 0
    # 7. context : ctx[h*V_DIM+d] = bf16( sum_slot probs[h][slot]*V[slot][h*V_DIM+d] )
    ctx = [F32(0.0)] * c.HV
    for h in range(c.H_HEADS):
        for d in range(c.V_DIM):
            acc = F32(0.0)
            for slot in range(len(keys)):
                acc = fadd(acc, fmul(b2f(probs[h][slot]),
                                     b2f(vstore[slot][h * c.V_DIM + d])))
            ctx[h * c.V_DIM + d] = bf16(acc)
    # 8. out = ctx @ W_o
    return matmul_q4k(ctx, W.W_o[layer].dequant())


def glm_softmax(x_bf):
    """glm_softmax.v: m=max, e=exp(x-m) (glm_exp), S=serial fp32 sum, p=e*(1/S),
       1/S = rsqrt(S)^2.  x_bf and result are bf16-valued fp32 lists."""
    n = len(x_bf)
    xf = [b2f(v) for v in x_bf]
    m = xf[0]
    for i in range(1, n):
        if fp32_gt(xf[i], m):
            m = xf[i]
    e = [glm_exp(fadd(xf[i], fneg(m))) for i in range(n)]         # exp(x_i - m)
    s = F32(0.0)
    for i in range(n):                                            # serial fp32 sum
        s = fadd(s, e[i])
    r = fp32_rsqrt(s)
    recip = fmul(r, r)                                            # 1/S = rsqrt(S)^2
    return [bf16(fmul(e[i], recip)) for i in range(n)]


def decoder_block(cfg, x_bf, pos, s_len, layer, mode_moe, W):
    """One decoder layer:  h = x + attn(rmsnorm(x)) ; y = h + FFN(rmsnorm(h))."""
    c = cfg
    nrm1 = rmsnorm(x_bf, W.G1[layer])
    attn = mla_attn(cfg, nrm1, pos, s_len, layer, W)
    h = [bf16(fadd(b2f(x_bf[i]), b2f(attn[i]))) for i in range(c.MODEL_DIM)]     # residual 1
    nrm2 = rmsnorm(h, W.G2[layer])
    if not mode_moe:
        fbuf = swiglu_expert(cfg, nrm2, W.Dg[layer], W.Du[layer], W.Dd[layer])
    else:
        sel, sel_w, _ = moe_router(cfg, nrm2, W.Wg[layer])
        facc = [F32(0.0)] * c.MODEL_DIM
        for slot in range(c.TOPK):                                # routed experts, slot order
            e = sel[slot]
            y_e = swiglu_expert(cfg, nrm2, W.Mg[layer][e], W.Mu[layer][e], W.Md[layer][e])
            g = b2f(sel_w[slot])
            for d in range(c.MODEL_DIM):
                facc[d] = fadd(facc[d], fmul(g, b2f(y_e[d])))
        y_sh = swiglu_expert(cfg, nrm2, W.SHg[layer], W.SHu[layer], W.SHd[layer])  # shared, wt 1
        for d in range(c.MODEL_DIM):
            facc[d] = fadd(facc[d], b2f(y_sh[d]))
        fbuf = [bf16(facc[d]) for d in range(c.MODEL_DIM)]
    y = [bf16(fadd(b2f(h[i]), b2f(fbuf[i]))) for i in range(c.MODEL_DIM)]         # residual 2
    return y


def model_forward(cfg, token_id, pos, s_len, W, capture=True):
    """FULL glm_model_q4k forward for one (token, pos).  Returns a dict of stage
       intermediates: x0 (post-embed), per-layer hidden, xn (final norm), logits, argmax."""
    c = cfg
    x = [W.EMB[token_id][d] for d in range(c.MODEL_DIM)]         # embed (bf16 lookup)
    out = {"x0": list(x), "layers": []}
    for layer in range(c.L):
        mode_moe = layer >= c.N_DENSE
        x = decoder_block(cfg, x, pos, s_len, layer, mode_moe, W)
        if capture:
            out["layers"].append(list(x))
    xn = rmsnorm(x, W.GF)                                        # final RMSNorm
    logits = [matmul_bf16_dot(xn, W.Wlm[v]) for v in range(c.VOCAB)]   # LM head (bf16 pipe)
    # argmax: strict-greater bf16 compare, lower-index tie-break, start at -inf.
    best_bits = 0xFF80
    arg = 0
    for v in range(c.VOCAB):
        lb = bf16_bits(logits[v])
        if bf16_gt(lb, best_bits):
            best_bits = lb
            arg = v
    out["xn"] = xn
    out["logits"] = logits
    out["logits_bits"] = [bf16_bits(l) for l in logits]
    out["argmax"] = arg
    return out

# ===========================================================================
# 9.  self-test
# ===========================================================================
def _selftest():
    rng = np.random.default_rng(0)
    fails = 0

    # (a) column dequant == q4k_ref.dequantize_block_q4_K on a full super-block.
    for _ in range(50):
        scs = [int(v) for v in rng.integers(0, 64, 8)]
        mns = [int(v) for v in rng.integers(0, 64, 8)]
        s12 = q4k_ref._pack_6bit_scales(scs, mns)
        codes = [int(v) for v in rng.integers(0, 16, 256)]
        d_h = f32_to_f16bits(rng.uniform(0.001, 0.06))
        dm_h = f32_to_f16bits(rng.uniform(0.0, 0.03))
        # pack codes -> ggml qs bytes: byte(k)=32*(k//64)+(k%32); nibble low if (k%64)<32
        qs = [0] * 128
        for k in range(256):
            byte = 32 * (k // 64) + (k % 32)
            if (k % 64) < 32:
                qs[byte] |= codes[k] & 0xF
            else:
                qs[byte] |= (codes[k] & 0xF) << 4
        y_ref = q4k_ref.dequantize_block_q4_K(d_h, dm_h, s12, qs)
        qw = QW(1, 256, [codes], [d_h], [dm_h], [list(s12)])
        y_mine = qw.dequant_col(0)
        if not np.array_equal(y_ref.view(np.uint32), y_mine.view(np.uint32)):
            fails += 1
    print(f"[selftest a] column-dequant vs q4k_ref.dequantize_block_q4_K : {'PASS' if fails==0 else 'FAIL'}")

    # (b) matmul_q4k self-consistency: a full Q4_K GEMV column vs q4k_ref.matmul_q4k_col.
    b_fail = 0
    for _ in range(20):
        K = int(rng.integers(16, 256))
        a = [bf16(F32(v)) for v in rng.uniform(-1.5, 1.5, K)]
        qw = _mk_qw(rng, 1, K)
        wc = qw.dequant_col(0)
        got = matmul_q4k(a, [wc])[0]
        exp = matmul_q4k_col(np.array([b2f(v) for v in a], dtype=np.float32), wc)
        if f32_bits(got) != f32_bits(exp):
            b_fail += 1
    fails += b_fail
    print(f"[selftest b] matmul_q4k column vs q4k_ref.matmul_q4k_col       : {'PASS' if b_fail==0 else 'FAIL'}")

    # (c) glm_exp vs a range-reduced numpy cross-check (accuracy, not bit-exact).
    cerr = 0.0
    for _ in range(200):
        x = float(rng.uniform(-30, 0))
        g = float(glm_exp(F32(x)))
        cerr = max(cerr, abs(g - np.exp(x)) / max(np.exp(x), 1e-30))
    print(f"[selftest c] glm_exp rel-err vs numpy over [-30,0]            : {cerr:.2e} (<2^-11={2**-11:.2e})")
    if cerr > 2 ** -11:
        fails += 1

    # (d) fp32_rsqrt: the classic Quake seed + EXACTLY 2 Newton iters (glm_fp.vh).
    #     True 2-iter accuracy is ~5e-6 rel (e2 ~= 1.5*e1^2, e1 ~= 1.7e-3), so the
    #     bound here confirms 2 iters ran -- NOT the glm_fp.vh header's optimistic
    #     "<2^-22" note (that would need a 3rd iter).  RTL bit-fidelity (this repro
    #     copies the exact ops) is proven against the DUT in phase 3, not here.
    rerr = 0.0
    for _ in range(200):
        x = float(rng.uniform(1e-3, 1e3))
        g = float(fp32_rsqrt(F32(x)))
        rerr = max(rerr, abs(g - 1.0 / np.sqrt(x)) / (1.0 / np.sqrt(x)))
    print(f"[selftest d] fp32_rsqrt rel-err vs 1/sqrt (2-iter Quake ~5e-6): {rerr:.2e} (<1e-5)")
    if rerr > 1e-5:
        fails += 1

    # (e) glm_softmax normalizes (sum of probs ~ 1) and is non-negative.
    sm_fail = 0
    for _ in range(30):
        n = int(rng.integers(2, 8))
        x = [bf16(F32(v)) for v in rng.uniform(-4, 4, n)]
        p = glm_softmax(x)
        s = float(sum(float(pi) for pi in p))
        if not (0.985 <= s <= 1.015) or any(float(pi) < 0 for pi in p):
            sm_fail += 1
    fails += sm_fail
    print(f"[selftest e] glm_softmax sum(probs) in [0.985,1.015]          : {'PASS' if sm_fail==0 else 'FAIL'}")

    # (f) MoE renorm invariant: sum of routed weights == SCALE (2.5), per moe_router_q4k.
    cfg = Config()
    W = Weights(cfg, seed=7)
    ren_err = 0.0
    ren_fail = 0
    for _ in range(40):
        x = [bf16(F32(v)) for v in rng.uniform(-1.5, 1.5, cfg.MODEL_DIM)]
        sel, sw, _ = moe_router(cfg, x, W.Wg[cfg.N_DENSE])
        tot = float(sum(float(b2f(w)) for w in sw))
        ren_err = max(ren_err, abs(tot - 2.5))
        if abs(tot - 2.5) > 0.05:
            ren_fail += 1
    fails += ren_fail
    print(f"[selftest f] MoE renorm sum(sel_weight) ~ 2.5 (max|dev|={ren_err:.4f}) : {'PASS' if ren_fail==0 else 'FAIL'}")

    # (g) end-to-end forward: runs, no NaN/Inf anywhere, argmax in range.
    def _finite(vs):
        for v in vs:
            u = f32_bits(v)
            if ((u >> 23) & 0xFF) == 0xFF:
                return False
        return True

    g_fail = 0
    prev_x0 = None
    for tok in (3, 17, 200):
        r = model_forward(cfg, tok, pos=3, s_len=2, W=W)
        ok = _finite(r["x0"]) and _finite(r["xn"]) and _finite(r["logits"])
        ok = ok and all(_finite(h) for h in r["layers"])
        ok = ok and (0 <= r["argmax"] < cfg.VOCAB)
        if not ok:
            g_fail += 1
        lv = [float(l) for l in r["logits"]]
        spread = max(lv) - min(lv)                     # genuine logit variation?
        diff = (0.0 if prev_x0 is None
                else max(abs(float(a) - float(b)) for a, b in zip(r["x0"], prev_x0)))
        prev_x0 = r["x0"]
        print(f"[selftest g] forward(token={tok:3d},pos=3,s_len=2): argmax={r['argmax']:3d} "
              f"logit_spread={spread:6.3f} x0_delta_vs_prev={diff:6.3f} "
              f"finite={'yes' if ok else 'NO'}")
    # the shared argmax across tokens is the documented VOCAB-slice fixed point
    # (spec_decode_top_tb: "collapses to a fixed point ... for essentially any
    # weights") -- GENUINE here: logits vary (nonzero spread) and the per-token
    # embeddings/hidden differ (nonzero x0 delta), so it is not a degenerate collapse.
    fails += g_fail

    # (h) end-to-end on the spec-TB slice too (small config path exercised).
    cfg2 = Config(**SPEC_SLICE)
    W2 = Weights(cfg2, seed=11)
    r2 = model_forward(cfg2, 3, pos=0, s_len=1, W=W2)
    h_ok = _finite(r2["logits"]) and (0 <= r2["argmax"] < cfg2.VOCAB)
    if not h_ok:
        fails += 1
    print(f"[selftest h] spec-slice forward(token=3): argmax={r2['argmax']} "
          f"finite={'yes' if h_ok else 'NO'}")

    print("-" * 68)
    print(f"SELF-TEST {'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _selftest() else 0)
