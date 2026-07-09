#!/usr/bin/env python3
"""
moe_trace_analyze.py -- turn the routed-expert traces (moe_trace_hf.py) into
the two product-design numbers the 100 tok/s roofline needs:

  (a) EOR      : consecutive-token expert overlap per layer (reuse floor).
  (b) U(K)     : K-token union factor -- an MTP verify pass of K tokens streams
                 |union(experts)| not K*topk; U(K) = mean |union| / topk.
  (c) h(cache) : BANDWIDTH hit rate vs cache capacity (fraction of the expert
                 pool resident), for three policies:
                   - LRU        (the literature baseline; known-bad for MoE)
                   - LFU-decay  (frequency with slow decay)
                   - STATIC-TOP (offline: most-frequent experts pinned -- the
                     "fit the device to the model" policy; needs only a
                     profiling pass, zero runtime bookkeeping)
                 Counted per (layer, expert) STREAM EVENT: a miss = that
                 expert's bytes cross the flash interface.  Prefetch is NOT
                 modeled -- this is the bandwidth-h, deliberately conservative.
  (d) tok/s    : the roofline table re-evaluated with the measured numbers.

Usage: tools/moevenv/bin/python tools/moe_trace_analyze.py build/moe_trace/<tag>
"""
import glob, json, os, sys
import numpy as np


def load(dirpath):
    out = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "trace_*.npz"))):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        out[meta["workload"]] = (z["ids"], meta)
    return out


def eor(ids):
    """consecutive-token expert overlap, averaged over layers x steps."""
    t, L, k = ids.shape
    if t < 2:
        return float("nan")
    a = ids[:-1]  # [t-1, L, k]
    b = ids[1:]
    ov = np.zeros(L)
    for l in range(L):
        m = 0.0
        for i in range(t - 1):
            m += len(np.intersect1d(a[i, l], b[i, l], assume_unique=False)) / k
        ov[l] = m / (t - 1)
    return ov  # per layer


def union_factor(ids, K):
    """mean |union over K consecutive tokens| / topk, per layer, then averaged."""
    t, L, k = ids.shape
    if t < K:
        return float("nan")
    tot, n = 0.0, 0
    for i in range(0, t - K + 1, K):          # disjoint windows = verify passes
        w = ids[i:i + K]                       # [K, L, k]
        for l in range(L):
            tot += len(np.unique(w[:, l, :])) / k
            n += 1
    return tot / n


def h_curve(ids, n_exp, cache_frac, policy):
    """bandwidth hit rate with a cache of (cache_frac * n_exp * n_layers) expert
    slots, shared across layers (keyed by (layer, expert)); per stream event."""
    t, L, k = ids.shape
    cap = max(1, int(round(cache_frac * n_exp * L)))
    hits, total = 0, 0
    if policy == "static":
        # offline profile on the FIRST half, evaluate on the SECOND half
        half = t // 2
        cnt = {}
        for i in range(half):
            for l in range(L):
                for e in ids[i, l]:
                    cnt[(l, int(e))] = cnt.get((l, int(e)), 0) + 1
        pinned = set(sorted(cnt, key=cnt.get, reverse=True)[:cap])
        for i in range(half, t):
            for l in range(L):
                for e in ids[i, l]:
                    total += 1
                    if (l, int(e)) in pinned:
                        hits += 1
        return hits / max(total, 1)
    # online policies
    from collections import OrderedDict
    lru = OrderedDict()
    freq = {}
    tick = 0
    for i in range(t):
        for l in range(L):
            for e in ids[i, l]:
                key = (l, int(e))
                total += 1
                tick += 1
                if policy == "lru":
                    if key in lru:
                        hits += 1
                        lru.move_to_end(key)
                    else:
                        lru[key] = True
                        if len(lru) > cap:
                            lru.popitem(last=False)
                elif policy == "lfu":
                    freq[key] = freq.get(key, 0) * 0.999 + 1.0
                    if key in lru:
                        hits += 1
                    else:
                        lru[key] = True
                        if len(lru) > cap:
                            victim = min(lru, key=lambda x: freq.get(x, 0.0))
                            del lru[victim]
    return hits / max(total, 1)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("build/moe_trace/*"))[-1]
    traces = load(d)
    print(f"== traces from {d} ==")
    GB_PER_TOKEN = 14.0     # GLM-5.2 routed-expert stream per token [EST]
    BW = {"NVMe x2": 14, "NVMe x4": 28, "32ch ONFI": 100, "64ch ONFI": 200}

    for wl, (ids, meta) in traces.items():
        n_exp = meta["n_experts"]
        t, L, k = ids.shape
        print(f"\n---- workload={wl}  tokens={t} layers={L} topk={k} experts={n_exp} ----")
        e = eor(ids)
        print(f"EOR (consecutive-token overlap): mean={np.nanmean(e):.3f} "
              f"min-layer={np.nanmin(e):.3f} max-layer={np.nanmax(e):.3f} "
              f"(random baseline={k/n_exp:.3f})")
        for K in (2, 4, 8):
            u = union_factor(ids, K)
            print(f"U({K}) union factor: {u:.2f}  (ideal 1.0 = perfect reuse; worst {K}.0)")
        print(f"{'cache frac':>10} | {'LRU':>6} | {'LFU':>6} | {'STATIC':>6}")
        for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
            row = [h_curve(ids, n_exp, frac, p) for p in ("lru", "lfu", "static")]
            print(f"{frac:>10.2f} | {row[0]:>6.3f} | {row[1]:>6.3f} | {row[2]:>6.3f}")

        # roofline with measured U and the best measured h at 20% cache
        h20 = max(h_curve(ids, n_exp, 0.20, p) for p in ("lfu", "static"))
        for K, A_eff in ((4, 3.0),):  # draft 4, accept ~3 (GLM-5 class)
            U = union_factor(ids, K)
            bytes_per_tok = GB_PER_TOKEN * (1 - h20) * U / A_eff
            print(f"\nroofline @ K={K} accept~{A_eff}, h(20% cache)={h20:.2f}, U={U:.2f}"
                  f" -> effective {bytes_per_tok:.2f} GB/token")
            for name, bw in BW.items():
                print(f"   {name:>10}: {bw / bytes_per_tok:6.1f} tok/s [EST]")


if __name__ == "__main__":
    main()
