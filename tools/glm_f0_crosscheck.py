#!/usr/bin/env python3
# ============================================================================
# glm_f0_crosscheck.py -- TRACK F0 full-model fidelity cross-check.
#   Compares the independent numpy full-model reference (glm_full_ref_np.py)
#   against the committed RTL testbench outputs (build_f0/tb_dump.txt, produced
#   by test/glm_model_fp8_dump_tb.v): both the fp64 Verilog GOLDEN and the RTL
#   DUT, on the SAME committed test vectors.
# ============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_full_ref_np as ref

DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build_f0", "tb_dump.txt")


def parse_dump(path):
    """-> list of dicts: {test, tok, pos, S, band, g_argmax, dut_argmax,
                          g_codes[256], d_codes[256]}"""
    tests = []
    cur = None
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "TEST":
                if cur is not None:
                    tests.append(cur)
                cur = dict(test=int(p[1]), tok=int(p[3]), pos=int(p[5]), S=int(p[7]),
                           band=int(p[9]), g_argmax=int(p[11]), dut_argmax=int(p[13]),
                           g_codes={}, d_codes={})
            else:
                o = int(p[0]); cur["g_codes"][o] = int(p[1], 16); cur["d_codes"][o] = int(p[2], 16)
    if cur is not None:
        tests.append(cur)
    return tests


def compare(np_real, np_codes, ref_codes, vocab):
    """np_* : numpy result;  ref_codes : dict o->bf16 code.  Returns metrics."""
    max_abs = 0.0; max_rel = 0.0; worst_v = -1
    exact = 0
    TINY = 1e-3
    for o in range(vocab):
        rr = ref.b2real(ref_codes[o])
        nv = np_real[o]
        if np_codes[o] == ref_codes[o]:
            exact += 1
        ad = abs(nv - rr)
        denom = abs(rr) if abs(rr) > TINY else TINY
        rel = ad / denom
        if ad > max_abs:
            max_abs = ad; worst_v = o
        if ad > 1e-9 and rel > max_rel:
            max_rel = rel
    return max_abs, max_rel, exact, worst_v


def main():
    if not os.path.exists(DUMP):
        print("MISSING", DUMP); sys.exit(2)
    dump = parse_dump(DUMP)
    print("Loaded %d tests from TB dump" % len(dump))
    print("Running independent numpy full-model reference (3 vectors)...")
    npres = ref.run_all(verbose=False)

    all_am_ok = True
    print()
    hdr = ("test  tok  pos  S  band | argmax np/gold/dut | "
           "np-vs-GOLDEN: maxabs  maxrel  exact | np-vs-DUT: maxabs  maxrel  exact | "
           "gold-vs-dut argmax")
    print(hdr)
    print("-" * len(hdr))
    for i, d in enumerate(dump):
        n = npres[i]
        assert n["tok"] == d["tok"] and n["Sg"] == d["S"] and n["band"] == d["band"], \
            "vector mismatch between numpy TESTS and TB dump order"
        ga, gr, gex, gw = compare(n["logit_real"], n["logit_codes"], d["g_codes"], ref.VOCAB)
        da, dr, dex, dw = compare(n["logit_real"], n["logit_codes"], d["d_codes"], ref.VOCAB)
        am_np = n["argmax"]; am_g = d["g_argmax"]; am_d = d["dut_argmax"]
        am_line = f"{am_np:3d}/{am_g:3d}/{am_d:3d}"
        gold_dut = "MATCH" if am_g == am_d else "DIFFER"
        ok = (am_np == am_g == am_d)
        all_am_ok = all_am_ok and ok
        print(f"  {d['test']}   {d['tok']:3d}  {d['pos']:3d}  {d['S']}   {d['band']}  | "
              f"{am_line}        | "
              f"{ga:10.4g} {gr:9.4g}  {gex:3d}/256 | "
              f"{da:10.4g} {dr:9.4g}  {dex:3d}/256 | {gold_dut}")

    print()
    print("SUMMARY:")
    print("  argmax agreement (numpy == golden == DUT) on all %d vectors: %s"
          % (len(dump), "YES" if all_am_ok else "NO"))
    # overall worst
    wa_g = max(compare(npres[i]["logit_real"], npres[i]["logit_codes"], dump[i]["g_codes"], ref.VOCAB)[0]
               for i in range(len(dump)))
    wr_g = max(compare(npres[i]["logit_real"], npres[i]["logit_codes"], dump[i]["g_codes"], ref.VOCAB)[1]
               for i in range(len(dump)))
    wa_d = max(compare(npres[i]["logit_real"], npres[i]["logit_codes"], dump[i]["d_codes"], ref.VOCAB)[0]
               for i in range(len(dump)))
    wr_d = max(compare(npres[i]["logit_real"], npres[i]["logit_codes"], dump[i]["d_codes"], ref.VOCAB)[1]
               for i in range(len(dump)))
    print("  numpy vs GOLDEN  : worst |abs logit err| = %.5g   worst rel = %.5g" % (wa_g, wr_g))
    print("  numpy vs RTL-DUT : worst |abs logit err| = %.5g   worst rel = %.5g" % (wa_d, wr_d))
    print("  (logits are O(100); TB pass envelope is ABS_TOL=4.0, REL_TOL=0.8125)")
    print("RESULT:", "PASS" if all_am_ok else "FAIL")
    sys.exit(0 if all_am_ok else 1)


if __name__ == "__main__":
    main()
