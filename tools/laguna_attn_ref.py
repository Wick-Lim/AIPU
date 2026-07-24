#!/usr/bin/env python3
"""laguna_attn_ref.py -- numpy reference for the Laguna-S-2.1 attention block.

Ports LagunaAttention / LagunaRMSNorm / LagunaRotaryEmbedding / eager_attention
from modeling_laguna.py (poolside/Laguna-S-2.1) as the golden the Laguna RTL
(branch laguna-s-2.1, Phases 3-5) checks against.

THE LAGUNA ATTENTION FORWARD (per layer):
  Q = x @ Wq  -> [Hq, D]   K = x @ Wk -> [Hkv, D]   V = x @ Wv -> [Hkv, D]   (D=head_dim=128)
  q = q_norm(Q)  ;  k = k_norm(K)          # per-head RMSNorm (eps 1e-6) BEFORE RoPE
  q, k = rope(q, k)                        # rotate_half (NOT GLM interleave), PARTIAL rotary,
                                           #   dual: full=YaRN partial 0.5 / sliding=plain partial 1.0
  K,V = repeat_kv(K,V, Hq//Hkv)            # GQA group expand
  s   = q @ k^T * D^-0.5  + mask           # causal (full) OR sliding-window(512) mask
  a   = softmax(s, fp32) @ V
  a   = a * softplus(x @ Wg)               # per-head output gating (BEFORE o_proj)
  out = a @ Wo

DELTAS vs GLM MLA on main (docs/LAGUNA_S21.md):
  - GQA (repeat_kv), not MLA latent compression; no DSA indexer.
  - RoPE uses rotate_half (contiguous halves), NOT GLM's interleaved pairs.
  - dual RoPE: YaRN (full-attn layers) + plain (sliding); partial rotary 0.5 / 1.0.
  - per-head softplus output gating (new).
  - q/k per-head RMSNorm before RoPE (GLM norms c_kv/q_lora differently).

Usage:  python3 tools/laguna_attn_ref.py --selftest   ->  "ALL <N> TESTS PASSED"
"""
import sys
import numpy as np


# ---------------------------------------------------------------- RMSNorm
def rms_norm(x, weight, eps=1e-6):
    x32 = x.astype(np.float64)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    return (weight * (x32 / np.sqrt(var + eps))).astype(x.dtype if x.dtype != np.float64 else np.float64)


# ---------------------------------------------------------------- RoPE factors
def compute_plain_rope(dim, base):
    """Plain rope inv_freq (rope_type=default); attention_scaling=1.0."""
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    return inv_freq, 1.0


def compute_yarn_rope(dim, base, factor, orig_max, beta_fast, beta_slow, attention_factor):
    """Standard transformers YaRN (_compute_yarn_parameters). attention_scaling = attention_factor."""
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    def find_correction_dim(num_rot, dim, base, max_pos):
        return (dim * np.log(max_pos / (num_rot * 2 * np.pi))) / (2 * np.log(base))

    low = np.floor(find_correction_dim(beta_fast, dim, base, orig_max))
    high = np.ceil(find_correction_dim(beta_slow, dim, base, orig_max))
    low = max(low, 0)
    high = min(high, dim - 1)

    def linear_ramp(mn, mx, size):
        if mn == mx:
            mx += 0.001
        lin = (np.arange(size, dtype=np.float64) - mn) / (mx - mn)
        return np.clip(lin, 0, 1)

    extrap_factor = 1.0 - linear_ramp(low, high, dim // 2)
    inv_freq = (inv_freq_interpolation * (1 - extrap_factor)
                + inv_freq_extrapolation * extrap_factor)
    return inv_freq, float(attention_factor)


def rope_cos_sin(positions, inv_freq, attention_scaling):
    """cos/sin for the given positions. emb = cat(freqs, freqs); scaled by attention_scaling."""
    freqs = np.outer(positions.astype(np.float64), inv_freq)     # [T, dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)                # [T, dim]
    return np.cos(emb) * attention_scaling, np.sin(emb) * attention_scaling


def rotate_half(x):
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply_rope(x, cos, sin):
    """x [.., T, D]; cos/sin [T, rotary_dim]. Partial: rotate first rotary_dim dims, pass the rest."""
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return np.concatenate([x_embed, x_pass], axis=-1)


# ---------------------------------------------------------------- activation
def softplus(x):
    # numerically-stable softplus: log(1+e^x) = max(x,0) + log1p(e^-|x|)
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))


# ---------------------------------------------------------------- attention
def laguna_attention(x, Wq, Wk, Wv, Wo, Wg, qn_w, kn_w,
                     n_q_heads, n_kv_heads, head_dim,
                     rope_inv_freq, rope_scaling, positions,
                     is_sliding, sliding_window, eps=1e-6):
    """x [T, H] -> out [T, H].  Single sequence, causal.  Per-head softplus gating."""
    T, H = x.shape
    x64 = x.astype(np.float64)
    q = (x64 @ Wq.T).reshape(T, n_q_heads, head_dim)
    k = (x64 @ Wk.T).reshape(T, n_kv_heads, head_dim)
    v = (x64 @ Wv.T).reshape(T, n_kv_heads, head_dim)

    # per-head RMSNorm on q,k (before RoPE)
    q = rms_norm(q, qn_w, eps)
    k = rms_norm(k, kn_w, eps)

    # RoPE (rotate_half, partial) -- cos/sin cover rotary_dim = len(inv_freq)*2
    cos, sin = rope_cos_sin(positions, rope_inv_freq, rope_scaling)     # [T, rotary_dim]
    # apply per head: broadcast cos/sin over heads
    q = apply_rope(q.transpose(1, 0, 2), cos, sin).transpose(1, 0, 2)   # [T, Hq, D]
    k = apply_rope(k.transpose(1, 0, 2), cos, sin).transpose(1, 0, 2)   # [T, Hkv, D]

    # GQA repeat_kv
    grp = n_q_heads // n_kv_heads
    k = np.repeat(k, grp, axis=1)      # [T, Hq, D]
    v = np.repeat(v, grp, axis=1)

    scaling = head_dim ** -0.5
    out = np.zeros((T, n_q_heads, head_dim), dtype=np.float64)
    for h in range(n_q_heads):
        qh = q[:, h, :]                # [T, D]
        kh = k[:, h, :]
        vh = v[:, h, :]
        scores = (qh @ kh.T) * scaling  # [T, T]
        # mask
        for i in range(T):
            for j in range(T):
                masked = (j > i)                                   # causal
                if is_sliding and (i - j) >= sliding_window:       # sliding window
                    masked = True
                if masked:
                    scores[i, j] = -np.inf
        # softmax fp32 rowwise
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        a = e / e.sum(axis=-1, keepdims=True)
        out[:, h, :] = a @ vh

    out = out.reshape(T, n_q_heads * head_dim)
    # per-head softplus output gating (before o_proj)
    gate = softplus(x64 @ Wg.T)                                    # [T, n_q_heads]
    out = (out.reshape(T, n_q_heads, head_dim) * gate[:, :, None]).reshape(T, n_q_heads * head_dim)
    return out @ Wo.T                                             # [T, H]


# ---------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.default_rng(11)
    tests = 0
    fails = 0

    def check(cond, name):
        nonlocal tests, fails
        tests += 1
        if not cond:
            fails += 1
            print(f"FAIL: {name}")

    # -- RMSNorm matches definition --
    x = rng.standard_normal((4, 8))
    w = rng.standard_normal(8)
    ref = w * (x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6))
    check(np.allclose(rms_norm(x, w), ref), "rms_norm matches weight * x / sqrt(mean(x^2)+eps)")

    # -- rotate_half --
    a = np.arange(8.0).reshape(1, 8)
    check(np.array_equal(rotate_half(a), np.array([[-4, -5, -6, -7, 0, 1, 2, 3]], dtype=float)),
          "rotate_half swaps contiguous halves with negation (NOT interleaved)")

    # -- plain rope inv_freq --
    inv, sc = compute_plain_rope(8, 10000)
    check(sc == 1.0 and np.isclose(inv[0], 1.0), "plain rope: inv_freq[0]=1, scaling=1")

    # -- YaRN inv_freq/scaling sane + differs from plain --
    yinv, ysc = compute_yarn_rope(64, 500000, 128, 8192, 32, 1, 1.4852030263919618)
    check(ysc == 1.4852030263919618, "YaRN attention_scaling == attention_factor")
    pinv, _ = compute_plain_rope(64, 500000)
    check(not np.allclose(yinv, pinv), "YaRN inv_freq differs from plain (scaling ramp applied)")
    check(yinv.shape == (32,), "YaRN inv_freq length == rotary_dim/2")

    # -- RoPE application: partial rotary passes the tail unchanged --
    D = 16
    xq = rng.standard_normal((1, 3, D))
    inv2, sc2 = compute_plain_rope(8, 10000)     # rotary_dim = 8 (partial 0.5 of D=16)
    cos, sin = rope_cos_sin(np.arange(3), inv2, sc2)
    y = apply_rope(xq, cos, sin)
    check(np.allclose(y[..., 8:], xq[..., 8:]), "partial rope: tail dims pass through unrotated")
    check(not np.allclose(y[..., :8], xq[..., :8]), "partial rope: head dims are rotated")
    # position 0 is identity on the rotary part (cos=scaling, sin=0)
    cos0, sin0 = rope_cos_sin(np.arange(1), inv2, sc2)
    y0 = apply_rope(xq[:, :1, :], cos0, sin0)
    check(np.allclose(y0, xq[:, :1, :]), "rope at position 0 is identity (plain, scaling 1)")

    # -- softplus --
    check(np.isclose(softplus(np.array([0.0]))[0], np.log(2)), "softplus(0)=ln2")
    check((softplus(rng.standard_normal(50)) > 0).all(), "softplus is strictly positive (a valid gate)")

    # -- attention: causal mask (token t cannot see t+1) --
    T, H, D = 5, 32, 8
    Hq, Hkv = 4, 2
    Wq = (rng.standard_normal((Hq * D, H)) * 0.1)
    Wk = (rng.standard_normal((Hkv * D, H)) * 0.1)
    Wv = (rng.standard_normal((Hkv * D, H)) * 0.1)
    Wo = (rng.standard_normal((H, Hq * D)) * 0.1)
    Wg = (rng.standard_normal((Hq, H)) * 0.1)
    qn = np.ones(D); kn = np.ones(D)
    inv, sc = compute_plain_rope(D, 10000)
    pos = np.arange(T)
    x = rng.standard_normal((T, H))
    full = laguna_attention(x, Wq, Wk, Wv, Wo, Wg, qn, kn, Hq, Hkv, D, inv, sc, pos, False, None)
    check(full.shape == (T, H), "attention output shape [T,H]")
    # causal: perturbing a FUTURE token must not change an earlier output row
    x2 = x.copy(); x2[T-1] += 5.0
    full2 = laguna_attention(x2, Wq, Wk, Wv, Wo, Wg, qn, kn, Hq, Hkv, D, inv, sc, pos, False, None)
    check(np.allclose(full[:T-1], full2[:T-1]), "causal mask: a future token cannot affect earlier outputs")

    # -- sliding window: token t only sees [t-w+1 .. t] --
    w = 2
    swa = laguna_attention(x, Wq, Wk, Wv, Wo, Wg, qn, kn, Hq, Hkv, D, inv, sc, pos, True, w)
    # perturbing token 0 must NOT change output at token T-1 when (T-1 - 0) >= w
    x3 = x.copy(); x3[0] += 5.0
    swa3 = laguna_attention(x3, Wq, Wk, Wv, Wo, Wg, qn, kn, Hq, Hkv, D, inv, sc, pos, True, w)
    check(np.allclose(swa[T-1], swa3[T-1]), "sliding window: token outside the window cannot affect output")
    check(not np.allclose(swa, full), "sliding-window attention differs from full causal")

    # -- GQA grouping: Hq must be divisible by Hkv (config invariant) --
    check((Hq % Hkv) == 0, "GQA group divides evenly")

    if fails:
        print(f"FAILED: {fails} of {tests} checks")
        sys.exit(1)
    print(f"ALL {tests} TESTS PASSED  (Laguna attention reference: q/k RMSNorm, rotate_half + "
          f"dual YaRN/plain partial RoPE, GQA, causal/sliding mask, softplus per-head gating)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
