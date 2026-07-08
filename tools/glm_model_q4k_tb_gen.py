#!/usr/bin/env python3
"""glm_model_q4k_tb_gen.py -- emit $readmemh vectors for test/glm_model_q4k_full_tb.v.

Drives the ASSEMBLED numpy golden (tools/glm_model_q4k_ref.py) at the committed
slice (Config() defaults: MODEL_DIM=128, L=6, N_DENSE=3, VOCAB=256, ...) with a
deterministic Weights(seed) set, runs the FULL forward for a handful of
(token,pos,s_len) cases, and writes:

  * every weight ROM as a FLAT hex memory ($readmemh-able), in the exact flat
    index order the TB's pull responders reconstruct;
  * per-case golden logits[VOCAB] (bf16 bits), argmax, and xn[MODEL_DIM]
    (final-RMSNorm = the DUT's h_state), for the TB to compare bit-exact.

The golden APPLIES the Phase-1 MLA softmax scale 1/sqrt(QK_DIM); the RTL DUT now
does too (src/mla_attn_q4k.v), so logits must match BIT-EXACT.

Run:  python3 tools/glm_model_q4k_tb_gen.py [outdir]   (default build/mq4k)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_model_q4k_ref as G

SEED = 424242
# (token, pos, s_len).  s_len in 2..S_MAX (multi-key attention -> scale-sensitive
# softmax); pos>0 (non-trivial RoPE).  distinct tokens exercise distinct embeddings.
CASES = [(3, 5, 4), (17, 2, 3), (200, 7, 6)]
# fast SPEC_SLICE smoke config (MODEL_DIM=16/L=2/VOCAB=16, S_MAX=2): seconds/forward.
CASES_SPEC = [(3, 1, 2), (7, 3, 2), (11, 0, 1)]


def sc96(scales12):
    """12 Q4_K scale bytes -> 96-bit little-endian value (byte i at bits [8i+7:8i])."""
    v = 0
    for i in range(12):
        v |= (int(scales12[i]) & 0xFF) << (8 * i)
    return v


def w4(x):  return f"{int(x) & 0xFFFF:04x}"
def w24(x): return f"{int(x) & ((1 << 96) - 1):024x}"

CODES_PER_WORD = 16   # pack 16 4-bit codes into one 64-bit $readmemh word


def pack_codes(nibbles):
    """flat list of 4-bit codes -> list of 64-bit words (16 nibbles/word, low first).
       code at flat index fi lives in word fi//16, bits [4*(fi%16) +: 4]."""
    words = []
    for w in range(0, len(nibbles), CODES_PER_WORD):
        acc = 0
        for i in range(CODES_PER_WORD):
            fi = w + i
            if fi < len(nibbles):
                acc |= (int(nibbles[fi]) & 0xF) << (4 * i)
        words.append(f"{acc:016x}")
    return words


class Emitter:
    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def _write(self, name, lines):
        with open(os.path.join(self.outdir, name), "w") as f:
            f.write("\n".join(lines) + "\n")

    def bf16_rom(self, name, flat_vals):
        self._write(name + ".hex", [w4(G.bf16_bits(v)) for v in flat_vals])

    def qw_rom(self, name, qws):
        """Emit codes/d/dmin/sc for a list of QW (already in flat layer[-expert] order).
           codes: flat nout*nin nibbles per qw, PACKED 16/word; d/dmin/sc: nout/qw."""
        codes, ds, dms, scs = [], [], [], []
        for qw in qws:
            for o in range(qw.nout):
                ds.append(w4(qw.d[o])); dms.append(w4(qw.dmin[o]))
                scs.append(w24(sc96(qw.scales[o])))
                for k in range(qw.nin):
                    codes.append(qw.codes[o][k])
        self._write(name + "_c.hex",  pack_codes(codes))
        self._write(name + "_d.hex",  ds)
        self._write(name + "_dm.hex", dms)
        self._write(name + "_sc.hex", scs)


def main():
    args = [a for a in sys.argv[1:]]
    spec = "--spec" in args
    args = [a for a in args if a != "--spec"]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dir = os.path.join(root, "build", "mq4k_s" if spec else "mq4k")
    outdir = args[0] if args else default_dir
    cfg = G.Config(**G.SPEC_SLICE) if spec else G.Config()
    cases = CASES_SPEC if spec else CASES
    W = G.Weights(cfg, seed=SEED)
    c = cfg
    em = Emitter(outdir)

    # ---- bf16 ROMs (flat) ----
    em.bf16_rom("emb", [W.EMB[v][d] for v in range(c.VOCAB) for d in range(c.MODEL_DIM)])
    em.bf16_rom("gf",  [W.GF[d] for d in range(c.MODEL_DIM)])
    em.bf16_rom("wlm", [W.Wlm[v][d] for v in range(c.VOCAB) for d in range(c.MODEL_DIM)])
    em.bf16_rom("g1",  [W.G1[l][d] for l in range(c.L) for d in range(c.MODEL_DIM)])
    em.bf16_rom("g2",  [W.G2[l][d] for l in range(c.L) for d in range(c.MODEL_DIM)])
    em.bf16_rom("ckv", [W.CKV[l][j][d] for l in range(c.L)
                        for j in range(c.S_MAX) for d in range(c.KV_LORA)])
    em.bf16_rom("krp", [W.KRP[l][j][d] for l in range(c.L)
                        for j in range(c.S_MAX) for d in range(c.ROPE)])

    # ---- Q4_K weight matrices (flat over layers, then experts) ----
    em.qw_rom("wdq", [W.W_dq[l] for l in range(c.L)])
    em.qw_rom("wuq", [W.W_uq[l] for l in range(c.L)])
    em.qw_rom("wuk", [W.W_uk[l] for l in range(c.L)])
    em.qw_rom("wuv", [W.W_uv[l] for l in range(c.L)])
    em.qw_rom("wo",  [W.W_o[l]  for l in range(c.L)])
    em.qw_rom("wg",  [W.Wg[l]   for l in range(c.L)])
    em.qw_rom("dg",  [W.Dg[l]   for l in range(c.L)])
    em.qw_rom("du",  [W.Du[l]   for l in range(c.L)])
    em.qw_rom("dd",  [W.Dd[l]   for l in range(c.L)])
    em.qw_rom("mg",  [W.Mg[l][e] for l in range(c.L) for e in range(c.N_EXPERT)])
    em.qw_rom("mu",  [W.Mu[l][e] for l in range(c.L) for e in range(c.N_EXPERT)])
    em.qw_rom("md",  [W.Md[l][e] for l in range(c.L) for e in range(c.N_EXPERT)])
    em.qw_rom("shg", [W.SHg[l]  for l in range(c.L)])
    em.qw_rom("shu", [W.SHu[l]  for l in range(c.L)])
    em.qw_rom("shd", [W.SHd[l]  for l in range(c.L)])

    # ---- stimulus + per-case golden outputs ----
    stim = []
    for ci, (tok, pos, slen) in enumerate(cases):
        r = G.model_forward(cfg, tok, pos, slen, W)
        # finite guard (the golden must not emit NaN/Inf; a saturated case would
        # hide the scale divergence -- see Phase-2 finding #1).
        for v in r["logits"]:
            u = G.f32_bits(v)
            assert ((u >> 23) & 0xFF) != 0xFF, f"case {ci}: non-finite logit"
        em._write(f"logits_{ci}.hex", [w4(b) for b in r["logits_bits"]])
        em._write(f"xn_{ci}.hex",     [w4(G.bf16_bits(x)) for x in r["xn"]])
        em._write(f"argmax_{ci}.hex", [f"{r['argmax']:x}"])
        stim.append(f"{tok:x} {pos:x} {slen:x}")
        spread = max(float(l) for l in r["logits"]) - min(float(l) for l in r["logits"])
        print(f"case {ci}: token={tok} pos={pos} s_len={slen} -> argmax={r['argmax']} "
              f"logit_spread={spread:.4f}")
    em._write("stim.hex", stim)
    with open(os.path.join(outdir, "ncase.hex"), "w") as f:
        f.write(f"{len(cases):x}\n")
    print(f"emitted {len(cases)} cases + weight ROMs to {outdir} "
          f"(slice={'SPEC' if spec else 'committed'}, MODEL_DIM={cfg.MODEL_DIM})")
    print(f"sm_scale_f32 (QK_DIM={cfg.QK_DIM}) = 0x{G.f32_bits(cfg.sm_scale_f32):08X}")


if __name__ == "__main__":
    main()
