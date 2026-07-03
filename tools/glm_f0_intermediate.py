#!/usr/bin/env python3
# ============================================================================
# glm_f0_intermediate.py -- TRACK F0 intermediate (2x) config cross-check.
#   Runs the independent numpy full-model reference at a 2x config
#   (MODEL_DIM=256, L=8, N_DENSE=4, N_EXPERT=16, INTER_MOE=128, INTER_DENSE=512,
#   VOCAB=128, S_MAX=4) that also forces multi-K-block FP8 scales
#   (A_NB=2 attn, R_NB=2 router, FF_NB_D=4 dense), and cross-checks it against
#   test/glm_model_fp8_2x_tb.v (fp64 golden + RTL DUT) on one vector.
# ============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import glm_full_ref_np as ref
from glm_f0_crosscheck import parse_dump, compare
from glm_f0_stagecmp import parse_stage, FIELDS

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP2 = os.path.join(HERE, "..", "build_f0", "tb_dump2.txt")
STAGE2 = os.path.join(HERE, "..", "build_f0", "tb_stage2.txt")

# ---- 2x config (MUST match test/glm_model_fp8_2x_tb.v localparams) ----
CFG = dict(MODEL_DIM=256, L=6, N_DENSE=3, VOCAB=128, H_HEADS=4, NOPE=16, ROPE=16,
           V_DIM=32, Q_LORA=64, KV_LORA=32, S_MAX=4, THETA=8000000, N_EXPERT=8,
           TOPK=2, INTER_MOE=128, INTER_DENSE=384, RSCALE=2.5, BLK=128)
VEC = dict(seed0=500, band=0, tok=7, tpos=5, Sg=4, label="2x tok7 pos5 Smax")


def main():
    ref.set_config(**CFG)
    print("numpy 2x config:", {k: getattr(ref, k) for k in
          ["MODEL_DIM", "L", "N_DENSE", "VOCAB", "N_EXPERT", "INTER_MOE",
           "INTER_DENSE", "S_MAX", "HQK", "FF_NB_D"]})
    W = ref.build_stimulus(VEC["seed0"], VEC["band"])
    stages = []
    logits, am, lr = ref.forward(W, VEC["tok"], VEC["tpos"], VEC["Sg"], stages=stages)
    print(f"[np 2x] {VEC['label']}: argmax={am}  best={lr[am]:.4f}  "
          f"sc_consumed={W.total_sc}")

    ready = (os.path.exists(DUMP2) and os.path.getsize(DUMP2) > 0 and
             os.path.exists(STAGE2) and os.path.getsize(STAGE2) > 0)
    tb = parse_stage(STAGE2) if ready else {}
    if not ready or (0, ref.L - 1) not in tb:
        print("RTL 2x dumps not complete yet. numpy side computed above; "
              "re-run this script after the RTL sim finishes to cross-check.")
        return

    # ---- per-layer stage cross-check vs golden ----
    first = None
    worst_stage = 0.0
    for st in stages:
        ly = st["layer"]
        g = tb[(0, ly)]
        for fld in FIELDS:
            npc = st[fld].astype(np.uint16); gc = g[fld].astype(np.uint16)
            mism = int(np.sum(npc != gc))
            if mism and first is None:
                first = (ly, fld, mism)
            ad = float(np.max(np.abs(ref.b2real_vec(npc) - ref.b2real_vec(gc))))
            worst_stage = max(worst_stage, ad)
    print("per-layer stage vs GOLDEN: first divergence =",
          first if first else "none (ALL %d layers x %d fields bit-exact)"
          % (len(stages), len(FIELDS)),
          "| worst |abs| =", worst_stage)

    # ---- logit cross-check vs golden and DUT ----
    dump = parse_dump(DUMP2)
    d = dump[0]
    ga, gr, gex, _ = compare(lr, logits, d["g_codes"], ref.VOCAB)
    da, dr, dex, _ = compare(lr, logits, d["d_codes"], ref.VOCAB)
    print(f"argmax  np={am}  golden={d['g_argmax']}  dut={d['dut_argmax']}")
    print(f"logits np-vs-GOLDEN : maxabs={ga:.5g} maxrel={gr:.5g} exact={gex}/{ref.VOCAB}")
    print(f"logits np-vs-DUT    : maxabs={da:.5g} maxrel={dr:.5g} exact={dex}/{ref.VOCAB}")
    ok = (am == d["g_argmax"] == d["dut_argmax"]) and (first is None) and (gex == ref.VOCAB)
    print("RESULT:", "PASS (numpy == golden bit-exact; argmax == golden == DUT)"
          if ok else "SEE ABOVE")


if __name__ == "__main__":
    main()
