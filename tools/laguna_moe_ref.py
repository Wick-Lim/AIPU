#!/usr/bin/env python3
"""laguna_moe_ref.py -- numpy reference for the Laguna-S-2.1 Sparse-MoE block.

Ports the EXACT forward of LagunaSparseMoeBlock / LagunaTopKRouter / LagunaExperts
/ LagunaMLP from modeling_laguna.py (poolside/Laguna-S-2.1) so the Laguna RTL
(branch laguna-s-2.1) has a golden to check against.  This is the block-level
NUMERIC ORDER; the Q4_K weight bit-exactness is covered separately by the reused
leaf gates (moe_router_q4k / swiglu_expert_q4k at Laguna dims).

THE LAGUNA MoE ORDER (from modeling_laguna.py):
  router:
    logits  = x @ Wg.T                      (fp32)
    (softcap 0.0 -> off)
    scores  = sigmoid(logits)               <-- SIGMOID scoring, NOT softmax
    sel     = topk(scores + bias, k)        bias = e_score_correction_bias = 0 in
                                            the GGUF (zero-init, no-op) -> topk by score
    w       = scores.gather(sel)            gather the UNBIASED scores
    if norm_topk_prob: w /= w.sum(-1)       <-- normalize the top-k weights
  experts (per selected e):  down( silu(gate(x)) * up(x) ) * w_e     (SwiGLU)
  block:
    shared     = MLP_shared(x)                                       (unscaled)
    routed_sum = Sigma_e experts_e
    out        = routed_sum * routed_scaling_factor(2.5)  +  shared  <-- SCALE the
                 SUMMED routed output, THEN add the UNSCALED shared expert.

KEY DELTA vs the GLM-5.2 router on main (docs/LAGUNA_S21.md):
  GLM folds the scale into each routed weight -- w_j = (gate_j / s) * 2.5, so
  Sum(w_j) = 2.5.  Laguna keeps w normalized (Sum = 1.0) and scales the routed
  SUM (then adds unscaled shared).  Algebraically identical, but the fp rounding
  order differs, so for bit-exactness to real Laguna the RTL uses the router's
  SCALE=1.0 mode (normalize only) and applies x2.5 in the decoder-block combine.
  demo_scale_order_delta() below exhibits the fp difference.

Usage:
  python3 tools/laguna_moe_ref.py --selftest    ->  "ALL <N> TESTS PASSED"
  (import laguna_moe_block for use as a golden generator.)
"""
import sys
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def laguna_router(x, Wg, top_k, norm_topk_prob=True, bias=None, softcap=0.0):
    """Returns (selected_experts[T,k], routing_weights[T,k]) -- Laguna order."""
    logits = (x.astype(np.float64) @ Wg.astype(np.float64).T)          # [T, E], fp
    if softcap and softcap > 0.0:
        logits = np.tanh(logits / softcap) * softcap
    scores = sigmoid(logits)                                            # sigmoid, not softmax
    sel_scores = scores if bias is None else scores + bias.reshape(1, -1)
    # top-k by (score + bias); bias is 0 in the GGUF so this is topk by score.
    sel = np.argsort(-sel_scores, axis=-1, kind="stable")[:, :top_k]    # [T, k]
    w = np.take_along_axis(scores, sel, axis=-1)                        # gather UNBIASED scores
    if norm_topk_prob:
        w = w / w.sum(axis=-1, keepdims=True)
    return sel, w


def laguna_mlp(x, Wg, Wu, Wd):
    """SwiGLU: down( silu(gate(x)) * up(x) ).  Used for both dense FFN and shared expert."""
    g = x.astype(np.float64) @ Wg.astype(np.float64).T
    u = x.astype(np.float64) @ Wu.astype(np.float64).T
    return (silu(g) * u) @ Wd.astype(np.float64).T


def laguna_moe_block(x, Wg_router, experts, shared, top_k, routed_scaling=2.5,
                     norm_topk_prob=True, bias=None):
    """Full LagunaSparseMoeBlock forward on x[T, H].
    experts = list of (Wg, Wu, Wd) per expert (E of them).
    shared  = (Wg, Wu, Wd) for the single shared expert.
    """
    T, H = x.shape
    sel, w = laguna_router(x, Wg_router, top_k, norm_topk_prob, bias)
    routed = np.zeros((T, H), dtype=np.float64)
    for t in range(T):
        for j in range(top_k):
            e = sel[t, j]
            Wgn, Wun, Wdn = experts[e]
            routed[t] += w[t, j] * laguna_mlp(x[t:t + 1], Wgn, Wun, Wdn)[0]
    shared_out = laguna_mlp(x, *shared)
    # SCALE the summed routed output, THEN add the UNSCALED shared expert.
    out = routed * routed_scaling + shared_out
    return out


# --------------------------------------------------------------------------
def demo_scale_order_delta():
    """Exhibit that fold-scale (GLM) != sum-then-scale (Laguna) in fp32 -- the
    reason the RTL router runs SCALE=1.0 and the combine applies x2.5."""
    rng = np.random.default_rng(0)
    T, H, E, k, I = 3, 16, 8, 3, 12
    x = rng.standard_normal((T, H)).astype(np.float32)
    Wg_router = rng.standard_normal((E, H)).astype(np.float32) * 0.1
    experts = [(rng.standard_normal((I, H)).astype(np.float32) * 0.1,
                rng.standard_normal((I, H)).astype(np.float32) * 0.1,
                rng.standard_normal((H, I)).astype(np.float32) * 0.1) for _ in range(E)]
    sel, w = laguna_router(x, Wg_router, k)
    # Laguna: sum then scale
    routed = np.zeros((T, H))
    for t in range(T):
        for j in range(k):
            routed[t] += w[t, j] * laguna_mlp(x[t:t+1], *experts[sel[t, j]])[0]
    laguna_way = (routed.astype(np.float32) * np.float32(2.5))
    # GLM: fold scale into the weight, per term
    glm = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        for j in range(k):
            wj = np.float32(w[t, j]) * np.float32(2.5)
            glm[t] += (wj * laguna_mlp(x[t:t+1], *experts[sel[t, j]])[0]).astype(np.float32)
    return laguna_way, glm


def _selftest():
    rng = np.random.default_rng(7)
    tests = 0
    fails = 0

    def check(cond, name):
        nonlocal tests, fails
        tests += 1
        if not cond:
            fails += 1
            print(f"FAIL: {name}")

    # -- router: sigmoid scoring, top-k, norm sums to 1 --
    T, H, E, k = 5, 32, 16, 10   # top-10 like Laguna
    x = rng.standard_normal((T, H)).astype(np.float32)
    Wg = (rng.standard_normal((E, H)) * 0.1).astype(np.float32)
    sel, w = laguna_router(x, Wg, k, norm_topk_prob=True)
    check(sel.shape == (T, k), "router selection shape [T,k]")
    check(w.shape == (T, k), "router weight shape [T,k]")
    check(np.allclose(w.sum(axis=-1), 1.0, atol=1e-9), "norm_topk_prob: weights sum to 1.0")
    check((sel >= 0).all() and (sel < E).all(), "selected expert indices in range")
    check(len(set(sel[0].tolist())) == k, "top-k selects k DISTINCT experts (row 0)")

    # -- bias = 0 is a genuine no-op vs no bias --
    sel0, w0 = laguna_router(x, Wg, k, bias=np.zeros(E, dtype=np.float32))
    check(np.array_equal(sel0, sel) and np.allclose(w0, w), "e_score_correction_bias=0 is a no-op")

    # -- top-k picks the k HIGHEST sigmoid scores --
    scores = sigmoid((x[0].astype(np.float64) @ Wg.astype(np.float64).T))
    topk_true = set(np.argsort(-scores)[:k].tolist())
    check(set(sel[0].tolist()) == topk_true, "top-k == the k highest sigmoid scores")

    # -- block: scale applies to routed sum only; shared is UNSCALED --
    I, Ish = 12, 12
    experts = [((rng.standard_normal((I, H)) * 0.1).astype(np.float32),
                (rng.standard_normal((I, H)) * 0.1).astype(np.float32),
                (rng.standard_normal((H, I)) * 0.1).astype(np.float32)) for _ in range(E)]
    shared = ((rng.standard_normal((Ish, H)) * 0.1).astype(np.float32),
              (rng.standard_normal((Ish, H)) * 0.1).astype(np.float32),
              (rng.standard_normal((H, Ish)) * 0.1).astype(np.float32))
    out25 = laguna_moe_block(x, Wg, experts, shared, k, routed_scaling=2.5)
    out10 = laguna_moe_block(x, Wg, experts, shared, k, routed_scaling=1.0)
    shared_only = laguna_mlp(x, *shared)
    # out(scale=s) - shared == s * routed_sum ; so (out25-shared) == 2.5*(out10-shared)
    lhs = out25 - shared_only
    rhs = 2.5 * (out10 - shared_only)
    check(np.allclose(lhs, rhs, atol=1e-6), "routed_scaling multiplies the routed sum, shared stays unscaled")
    check(not np.allclose(out25, out10), "scaling actually changes the routed contribution")

    # -- the GLM fold-scale vs Laguna sum-then-scale genuinely differ in fp32 --
    laguna_way, glm_way = demo_scale_order_delta()
    check(np.allclose(laguna_way, glm_way, atol=1e-2), "fold vs sum-scale are algebraically close")
    check(not np.array_equal(laguna_way, glm_way),
          "fold-scale (GLM) != sum-then-scale (Laguna) bit-for-bit -> RTL uses SCALE=1.0 + combine x2.5")

    if fails:
        print(f"FAILED: {fails} of {tests} checks")
        sys.exit(1)
    print(f"ALL {tests} TESTS PASSED  (Laguna MoE reference: sigmoid router, top-10, "
          f"norm_topk_prob, routed x2.5 + unscaled shared, scale-order delta shown)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
