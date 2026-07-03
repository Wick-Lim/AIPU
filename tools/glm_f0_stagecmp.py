#!/usr/bin/env python3
# Localise the first per-layer divergence between the numpy full-model ref and
# the Verilog fp64 golden (build_f0/tb_stage.txt).
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import glm_full_ref_np as ref

STAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build_f0", "tb_stage.txt")
FIELDS = ["nrm1", "attn", "h", "nrm2", "ffn", "x"]


def parse_stage(path):
    """-> dict[(call,layer)] = dict[field] -> np.uint16[MODEL_DIM]"""
    out = {}
    call = layer = None
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "STAGE":
                if call is not None:
                    out[(call, layer)] = _pack(rows)
                call = int(p[2]); layer = int(p[4]); rows = []
            else:
                rows.append([int(x, 16) for x in p[1:]])
    if call is not None:
        out[(call, layer)] = _pack(rows)
    return out


def _pack(rows):
    a = np.array(rows, dtype=np.uint16)      # [MODEL_DIM][6]
    return {FIELDS[i]: a[:, i] for i in range(len(FIELDS))}


def main():
    call = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tb = parse_stage(STAGE)
    t = ref.TESTS[call]
    W = ref.build_stimulus(t["seed0"], t["band"])
    stages = []
    ref.forward(W, t["tok"], t["tpos"], t["Sg"], stages=stages)
    print(f"compare call {call}: {t['label']}")
    print(f"{'layer':>5} {'field':>6} {'ncode_mismatch':>15} {'max|abs real|':>14} "
          f"{'worst j':>8}  (np_code/gold_code @ worst)")
    first = None
    for st in stages:
        ly = st["layer"]
        g = tb[(call, ly)]
        for fld in FIELDS:
            npc = st[fld].astype(np.uint16)
            gc = g[fld].astype(np.uint16)
            mism = int(np.sum(npc != gc))
            nr = ref.b2real_vec(npc); grr = ref.b2real_vec(gc)
            ad = np.abs(nr - grr)
            wj = int(np.argmax(ad))
            flag = "" if mism == 0 else " <--"
            if mism and first is None:
                first = (ly, fld)
            print(f"{ly:>5} {fld:>6} {mism:>15} {ad[wj]:>14.5g} {wj:>8}  "
                  f"{npc[wj]:04x}/{gc[wj]:04x}{flag}")
        print()
    print("FIRST DIVERGENCE:", first if first else "none (all stages match)")


if __name__ == "__main__":
    main()
