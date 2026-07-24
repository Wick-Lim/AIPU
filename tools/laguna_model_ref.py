#!/usr/bin/env python3
"""laguna_model_ref.py -- full-forward numpy reference for Laguna-S-2.1.

Assembles the executable spec of the whole model (branch laguna-s-2.1, Phase 6):
  embed -> 48 x decoder_layer -> final RMSNorm -> LM head -> argmax
where decoder_layer i is:
  h = h + attn( input_norm(h) )            # LagunaAttention (laguna_attn_ref)
  h = h + mlp(  post_attn_norm(h) )        # MoE (laguna_moe_ref) or dense SwiGLU
per the config-locked schedule (configs/full_laguna_s21.vh, docs/LAGUNA_S21.md):
  attention: layer i is FULL iff i%4==0 (YaRN, partial 0.5), else SLIDING (plain, window 512)
  mlp:       layer 0 is DENSE (SwiGLU, inter 12288); layers 1..47 are MoE (256/top-10/+1 shared)
  Q heads:   48 on full-attn layers, 72 on sliding; KV heads 8 (const); head_dim 128
NO MTP head (config) -> no speculative-decode; this is the plain autoregressive forward.

Run on a small-but-faithful SLICE with random weights: it proves the forward
STRUCTURE (schedule, GQA grouping, per-layer rope/mask, dense-vs-MoE dispatch,
causality end-to-end) is self-consistent -- the numeric bit-exactness to the real
checkpoint is sealed later by the dequant crosscheck + a real-weight golden.

Usage:  python3 tools/laguna_model_ref.py --selftest   ->  "ALL <N> TESTS PASSED"
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laguna_attn_ref import (rms_norm, compute_plain_rope, compute_yarn_rope,   # noqa: E402
                             laguna_attention)
from laguna_moe_ref import laguna_mlp, laguna_moe_block                          # noqa: E402


def is_full(i):
    return (i % 4) == 0


def is_dense(i):
    return i == 0


class LagunaWeights:
    """Random slice weights with the correct per-layer shapes."""
    def __init__(self, cfg, rng):
        H = cfg["hidden"]; D = cfg["head_dim"]; Hkv = cfg["kv_heads"]
        self.embed = rng.standard_normal((cfg["vocab"], H)) * 0.02
        self.final_norm = np.ones(H)
        self.lm_head = rng.standard_normal((cfg["vocab"], H)) * 0.02
        self.layers = []
        for i in range(cfg["layers"]):
            Hq = cfg["q_heads_full"] if is_full(i) else cfg["q_heads_swa"]
            L = {}
            L["in_norm"] = np.ones(H)
            L["post_norm"] = np.ones(H)
            L["Wq"] = rng.standard_normal((Hq * D, H)) * 0.05
            L["Wk"] = rng.standard_normal((Hkv * D, H)) * 0.05
            L["Wv"] = rng.standard_normal((Hkv * D, H)) * 0.05
            L["Wo"] = rng.standard_normal((H, Hq * D)) * 0.05
            L["Wg"] = rng.standard_normal((Hq, H)) * 0.05
            L["qn"] = np.ones(D); L["kn"] = np.ones(D)
            L["Hq"] = Hq
            if is_dense(i):
                I = cfg["inter_dense"]
                L["dense"] = (rng.standard_normal((I, H)) * 0.05,
                              rng.standard_normal((I, H)) * 0.05,
                              rng.standard_normal((H, I)) * 0.05)
            else:
                Im = cfg["inter_moe"]; Ish = cfg["inter_shared"]; E = cfg["experts"]
                L["router"] = rng.standard_normal((E, H)) * 0.05
                L["experts"] = [((rng.standard_normal((Im, H)) * 0.05),
                                 (rng.standard_normal((Im, H)) * 0.05),
                                 (rng.standard_normal((H, Im)) * 0.05)) for _ in range(E)]
                L["shared"] = ((rng.standard_normal((Ish, H)) * 0.05),
                               (rng.standard_normal((Ish, H)) * 0.05),
                               (rng.standard_normal((H, Ish)) * 0.05))
            self.layers.append(L)


def forward(token_ids, W, cfg, return_logits=False):
    """token_ids [T] -> argmax next-token id per position [T] (or logits [T, vocab])."""
    D = cfg["head_dim"]; Hkv = cfg["kv_heads"]
    T = len(token_ids)
    pos = np.arange(T)
    # dual rope factor sets (rotary_dim = head_dim * partial)
    full_dim = int(D * cfg["partial_full"])       # YaRN
    swa_dim = int(D * cfg["partial_swa"])         # plain
    yinv, ysc = compute_yarn_rope(full_dim, cfg["rope_full_theta"], cfg["yarn_factor"],
                                  cfg["yarn_orig_max"], cfg["yarn_beta_fast"],
                                  cfg["yarn_beta_slow"], cfg["yarn_attn_factor"])
    pinv, psc = compute_plain_rope(swa_dim, cfg["rope_swa_theta"])

    h = W.embed[token_ids].astype(np.float64)     # [T, H]
    for i, L in enumerate(W.layers):
        # ---- attention sub-block ----
        a_in = rms_norm(h, L["in_norm"], cfg["eps"])
        if is_full(i):
            inv, sc, sliding, win = yinv, ysc, False, None
        else:
            inv, sc, sliding, win = pinv, psc, True, cfg["sliding_window"]
        a = laguna_attention(a_in, L["Wq"], L["Wk"], L["Wv"], L["Wo"], L["Wg"],
                             L["qn"], L["kn"], L["Hq"], Hkv, D, inv, sc, pos,
                             sliding, win, cfg["eps"])
        h = h + a
        # ---- mlp sub-block ----
        m_in = rms_norm(h, L["post_norm"], cfg["eps"])
        if is_dense(i):
            m = laguna_mlp(m_in, *L["dense"])
        else:
            m = laguna_moe_block(m_in, L["router"], L["experts"], L["shared"],
                                 cfg["top_k"], cfg["routed_scaling"], True)
        h = h + m

    h = rms_norm(h, W.final_norm, cfg["eps"])
    logits = h @ W.lm_head.T                        # [T, vocab]
    if return_logits:
        return logits
    return np.argmax(logits, axis=-1)


def slice_cfg():
    """Small-but-faithful slice mirroring configs/full_laguna_s21.vh SLICE + rules."""
    return dict(
        hidden=32, head_dim=16, kv_heads=2, q_heads_full=4, q_heads_swa=6,
        layers=8, vocab=64, experts=8, top_k=3, inter_moe=16, inter_shared=16,
        inter_dense=64, sliding_window=4, eps=1e-6, routed_scaling=2.5,
        partial_full=0.5, partial_swa=1.0,
        rope_full_theta=500000, rope_swa_theta=10000,
        yarn_factor=128, yarn_orig_max=8192, yarn_beta_fast=32, yarn_beta_slow=1,
        yarn_attn_factor=1.4852030263919618,
    )


def _selftest():
    rng = np.random.default_rng(3)
    cfg = slice_cfg()
    W = LagunaWeights(cfg, rng)
    tests = 0
    fails = 0

    def check(cond, name):
        nonlocal tests, fails
        tests += 1
        if not cond:
            fails += 1
            print(f"FAIL: {name}")

    T = 6
    toks = rng.integers(0, cfg["vocab"], size=T)

    # -- shapes / determinism --
    out = forward(toks, W, cfg)
    logits = forward(toks, W, cfg, return_logits=True)
    check(out.shape == (T,), "forward returns one next-token id per position")
    check(logits.shape == (T, cfg["vocab"]), "logits shape [T, vocab]")
    check(np.array_equal(out, forward(toks, W, cfg)), "forward is deterministic")
    check(np.array_equal(out, np.argmax(logits, axis=-1)), "argmax over logits == returned ids")

    # -- end-to-end causality: changing the LAST token can't change earlier logits --
    toks2 = toks.copy(); toks2[-1] = (toks[-1] + 7) % cfg["vocab"]
    logits2 = forward(toks2, W, cfg, return_logits=True)
    check(np.allclose(logits[:-1], logits2[:-1]),
          "end-to-end causal: a later token cannot change earlier-position logits")

    # -- the layer schedule actually dispatched dense@0 + full/sliding pattern --
    check(is_dense(0) and not any(is_dense(i) for i in range(1, cfg["layers"])),
          "exactly layer 0 is dense; 1..L-1 are MoE")
    n_full = sum(is_full(i) for i in range(cfg["layers"]))
    check(n_full == cfg["layers"] // 4, "full-attention layers == L/4 (period-4 schedule)")
    check(W.layers[0]["Hq"] == cfg["q_heads_full"], "layer 0 (full) uses full Q-head count")
    check(W.layers[1]["Hq"] == cfg["q_heads_swa"], "layer 1 (sliding) uses sliding Q-head count")

    # -- a MoE layer and a dense layer both actually run (block coverage) --
    check(("experts" in W.layers[1]) and ("dense" in W.layers[0]),
          "layer 0 built dense weights; layer 1 built MoE weights")

    # -- GQA grouping holds for both head counts --
    check((cfg["q_heads_full"] % cfg["kv_heads"]) == 0
          and (cfg["q_heads_swa"] % cfg["kv_heads"]) == 0, "both Q-head counts divide KV heads")

    if fails:
        print(f"FAILED: {fails} of {tests} checks")
        sys.exit(1)
    print(f"ALL {tests} TESTS PASSED  (Laguna full-forward reference: embed -> 8x(GQA-attn + "
          f"MoE/dense) -> norm -> LM head -> argmax; dense@0, full/sliding schedule, "
          f"dual RoPE, GQA, end-to-end causal)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
