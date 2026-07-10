#!/usr/bin/env python3
"""
gguf_crosscheck.py -- close the OLDEST open trust-table row: prove our ggml
reimplementation (tools/q4k_ref.py, the golden every RTL gate verifies
against) is BIT-EXACT to the REAL ggml/llama.cpp on REAL GGUF bytes.

Chain:  RTL == q4k_ref   (existing 1155-test + unit gates)
      + q4k_ref == ggml  (THIS script, on a real published GGUF's raw blocks)
      => RTL == the real GGUF dequant, end of the self-referential caveat.

Method: parse a real GGUF (gguf-py, from the llama.cpp repo), pull every
Q4_K / Q6_K tensor's RAW block bytes, dequantize each with
  (A) tools/q4k_ref.py            (our reimpl -- numpy)
  (B) llama.cpp's own dequantize_row_q4_K / _q6_K (dequant_dump, linked
      against the built libggml -- the exact code llama.cpp executes)
and compare fp32 outputs BITWISE (np.uint32 view equality, not tolerance).

usage: python3 tools/gguf_crosscheck.py <model.gguf> <llamacpp_dir>
"""
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import q4k_ref  # noqa: E402

Q4K_BS, Q6K_BS = 144, 210  # bytes per 256-weight super-block


def ref_dequant_q4k(raw: bytes) -> np.ndarray:
    out = []
    for off in range(0, len(raw), Q4K_BS):
        b = raw[off:off + Q4K_BS]
        d_h = int.from_bytes(b[0:2], "little")
        dmin_h = int.from_bytes(b[2:4], "little")
        scales = b[4:16]
        qs = b[16:144]
        out.append(q4k_ref.dequantize_block_q4_K(d_h, dmin_h, scales, qs))
    return np.concatenate(out).astype(np.float32)


def ref_dequant_q6k(raw: bytes) -> np.ndarray:
    out = []
    for off in range(0, len(raw), Q6K_BS):
        b = raw[off:off + Q6K_BS]
        ql = b[0:128]
        qh = b[128:192]
        sc = b[192:208]
        d_h = int.from_bytes(b[208:210], "little")
        out.append(q4k_ref.dequantize_block_q6_K(d_h, ql, qh, sc))
    return np.concatenate(out).astype(np.float32)


def main():
    gguf_path, llamacpp = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.join(llamacpp, "gguf-py"))
    from gguf import GGUFReader, GGMLQuantizationType as T

    dump = os.path.join(llamacpp, "dequant_dump")
    rd = GGUFReader(gguf_path)

    totals = {"Q4_K": [0, 0], "Q6_K": [0, 0]}   # tensors, weights
    mismatches = 0
    for t in rd.tensors:
        ttype = T(t.tensor_type).name
        if ttype not in ("Q4_K", "Q6_K"):
            continue
        raw = bytes(t.data.tobytes())
        kind = "q4_k" if ttype == "Q4_K" else "q6_k"
        ours = ref_dequant_q4k(raw) if ttype == "Q4_K" else ref_dequant_q6k(raw)

        rin, rout = "/tmp/xchk_in.bin", "/tmp/xchk_out.bin"
        with open(rin, "wb") as f:
            f.write(raw)
        subprocess.run([dump, kind, rin, rout], check=True,
                       capture_output=True)
        theirs = np.fromfile(rout, dtype=np.float32)

        same = np.array_equal(ours.view(np.uint32), theirs.view(np.uint32))
        totals[ttype][0] += 1
        totals[ttype][1] += ours.size
        if not same:
            mismatches += 1
            bad = np.nonzero(ours.view(np.uint32) != theirs.view(np.uint32))[0]
            print(f"MISMATCH {t.name} [{ttype}] {bad.size}/{ours.size} weights "
                  f"first@{bad[0]}: ours={ours[bad[0]]!r} ggml={theirs[bad[0]]!r}")
        else:
            print(f"ok  {t.name:<44} {ttype}  {ours.size:>9} weights bit-exact")

    print("\n==== SUMMARY ====")
    for k, (nt, nw) in totals.items():
        print(f"  {k}: {nt} tensors, {nw} weights")
    if mismatches == 0 and any(v[0] for v in totals.values()):
        print("CROSSCHECK PASSED: q4k_ref.py == llama.cpp/ggml, BIT-EXACT on real GGUF bytes")
        return 0
    print(f"CROSSCHECK FAILED: {mismatches} tensor(s) mismatched")
    return 1


if __name__ == "__main__":
    sys.exit(main())
