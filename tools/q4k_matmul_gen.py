#!/usr/bin/env python3
"""q4k_matmul_gen.py -- generate random Q4_K GEMM test tiles + golden outputs for
the glm_matmul_q4k RTL testbench. Uses the verified tools/q4k_ref.py golden, so the
RTL is proven bit-exact to ggml Q4_K dequant + the fp32 MAC contract.

Emits a whitespace/hex file (build/q4k_vec.txt) the Verilog TB reads via $fscanf.
Layout per test (PE_M rows, PE_N cols, K beats <=256, one super-block along K):
  k_len
  d[pj]      (PE_N x 4hex fp16)
  dmin[pj]   (PE_N x 4hex fp16)
  scales[pj] (PE_N x 24hex, 96-bit packed)
  for k in 0..K-1:  a[pi] (PE_M x 4hex bf16)   q[pj] (PE_N x 1hex 4-bit)
  c[pi*PE_N+pj] (PE_M*PE_N x 4hex bf16 golden)
First line of file: NTEST PE_M PE_N
"""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from q4k_ref import (get_scale_min_k4, _pack_6bit_scales, _f32_to_f16bits,
                     bf16_round, matmul_q4k_col)

def f16bits(x):  return int(np.frombuffer(np.float16(x).tobytes(), dtype=np.uint16)[0])
def bf16bits(x):
    v = bf16_round(x)
    return (int(np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]) >> 16) & 0xFFFF
def bf16_val(u16):  # bf16 bits -> fp32 value
    return np.float32(np.frombuffer(np.uint32(u16 << 16).tobytes(), dtype=np.float32)[0])

def dequant_col(d_h, dmin_h, scales, qcol):
    """dequantize a weight COLUMN (K entries) spanning NSB=ceil(K/256) super-blocks.
    d_h/dmin_h are lists (one per super-block); scales is a list of 12-byte arrays.
    Bit-exact to ggml (per-weight, super-block sb=k//256, sub-block (k%256)//32)."""
    out = np.empty(len(qcol), dtype=np.float32)
    for k, q in enumerate(qcol):
        sb = k // 256
        d  = np.float32(np.frombuffer(np.uint16(d_h[sb]).tobytes(),   dtype=np.float16)[0])
        mn = np.float32(np.frombuffer(np.uint16(dmin_h[sb]).tobytes(), dtype=np.float16)[0])
        sc, m = get_scale_min_k4((k % 256) // 32, scales[sb])
        d1 = d * np.float32(sc); m1 = mn * np.float32(m)
        out[k] = d1 * np.float32(q) - m1
    return out

def gen(ntest, PE_M, PE_N, seed=0):
    rng = np.random.default_rng(seed)
    lines = [f"{ntest} {PE_M} {PE_N}"]
    for t in range(ntest):
        K   = int(rng.choice([32, 64, 128, 200, 256, 288, 512, 600, 768]))
        NSB = (K + 255) // 256
        # per-(column, super-block) params, index (pj*NSB + sb)
        d_h  = [[_f32_to_f16bits(rng.uniform(0.003, 0.05)) for _ in range(NSB)] for _ in range(PE_N)]
        dm_h = [[_f32_to_f16bits(rng.uniform(0.0,  0.02))  for _ in range(NSB)] for _ in range(PE_N)]
        scales, sc96 = [], []
        for pj in range(PE_N):
            srow, s96row = [], []
            for _ in range(NSB):
                s = _pack_6bit_scales([int(v) for v in rng.integers(0,64,8)],
                                      [int(v) for v in rng.integers(0,64,8)])
                srow.append(s); s96row.append(sum(b << (8*i) for i, b in enumerate(s)))
            scales.append(srow); sc96.append(s96row)
        A = [[bf16bits(v) for v in rng.uniform(-1.5, 1.5, K)] for _ in range(PE_M)]
        Q = [[int(v) for v in rng.integers(0, 16, K)] for _ in range(PE_N)]
        wdeq = [dequant_col(d_h[pj], dm_h[pj], scales[pj], Q[pj]) for pj in range(PE_N)]
        Af   = [np.array([bf16_val(A[pi][k]) for k in range(K)], dtype=np.float32) for pi in range(PE_M)]
        C = []
        for pi in range(PE_M):
            for pj in range(PE_N):
                C.append(bf16bits(matmul_q4k_col(Af[pi], wdeq[pj])))
        # serialize (params ordered col-outer, super-block-inner = RTL index pj*NSB+sb)
        lines.append(f"{K} {NSB}")
        lines.append(" ".join(f"{d_h[pj][sb]:04x}"  for pj in range(PE_N) for sb in range(NSB)))
        lines.append(" ".join(f"{dm_h[pj][sb]:04x}" for pj in range(PE_N) for sb in range(NSB)))
        lines.append(" ".join(f"{sc96[pj][sb]:024x}" for pj in range(PE_N) for sb in range(NSB)))
        for k in range(K):
            row = [f"{A[pi][k]:04x}" for pi in range(PE_M)] + [f"{Q[pj][k]:01x}" for pj in range(PE_N)]
            lines.append(" ".join(row))
        lines.append(" ".join(f"{x:04x}" for x in C))
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    ntest = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    PE_M  = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    PE_N  = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    out   = sys.argv[4] if len(sys.argv) > 4 else "build/q4k_vec.txt"
    open(out, "w").write(gen(ntest, PE_M, PE_N))
    print(f"wrote {out}: {ntest} tests, PE_M={PE_M} PE_N={PE_N}")
