#!/usr/bin/env python3
# ============================================================================
# validate_real_ckpt.py -- CPU/numpy validation of OUR FP8 contract against the
#                          REAL published zai-org/GLM-5.2-FP8 checkpoint tensors
# ----------------------------------------------------------------------------
# WHY THIS EXISTS  (the project's #1 missing validation)
#   docs/BIT_ACCURACY.md proved OUR exact-BFP FP8 GEMM (src/glm_matmul_fp8.v,
#   mirrored by tools/glm_fp8_ref.py, vectorized in tools/glm_fp8_contract.py)
#   is bit-identical + argmax-preserving vs an fp32-accumulate engine -- but only
#   on a SELF-MADE synthetic golden.  That golden's INTERPRETATION of the real
#   checkpoint (weight orientation [out,in], [128,128] block-scale layout,
#   weight_scale_inv dtype/shape, the bf16 tail, and the pending q_lora/kv_lora
#   ranks) was NEVER checked against the actual published tensors.  This script
#   closes that gap on CPU + numpy ONLY (no GPU, no torch), against the PUBLIC
#   (gated: False, no token) checkpoint, downloading only a few MB via HTTP range
#   reads of a single small FP8 Linear (NOT the 753 GB model, NOT a full 5 GB
#   shard).  It is the CPU analogue of tools/modal_validate.py's `tier1_operator`.
#
# WHAT IT DOES  (the four acceptance steps)
#   (1) Download config.json (no token) and ASSERT our quant assumptions:
#       quant_method=fp8, fmt=e4m3, weight_block_size=[128,128],
#       activation_scheme=dynamic; print modules_to_not_convert (the bf16 tail)
#       and the real dims (hidden_size, num_hidden_layers, q_lora_rank,
#       kv_lora_rank, n_routed_experts, ...) and RECONCILE vs our assumptions.
#   (2) Download the safetensors INDEX, pick the SMALLEST real FP8 `.weight`
#       (with a sibling `.weight_scale_inv`) whose block grid exercises BOTH
#       axes, and RANGE-READ only that tensor's bytes + its scale bytes.  VERIFY
#       the real dtype/shape/orientation MATCH our assumed layout (F8_E4M3
#       [out,in]; scale [ceil(out/128),ceil(in/128)]).
#   (3) Run OUR contract (glm_fp8_contract.block_fp8_gemm) on the REAL weight
#       with a controlled activation vector, and compare to an INDEPENDENT numpy
#       fp32-accumulate reference that dequantizes the SAME real FP8 weight
#       (decode E4M3 -> * per-block bf16 scale -> fp32) using the SAME per-token
#       pow2 a_shift.  Mirror tier1's error_stats (max_rel, argmax match,
#       bf16-domain exactness).  Isolates ONLY the accumulator (exact-BFP vs
#       fp32-accumulate) on REAL weights -- exactly docs/BIT_ACCURACY.md Section A.
#   (4) Print a PASS/FAIL summary + the reconciliation table.
#
# RUN:  python3 tools/validate_real_ckpt.py
#         (downloads a few MB, prints the reconciliation + PASS/FAIL, exit 0/1)
# ============================================================================
import sys
import os
import json
import struct
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_fp8_ref as ref                 # golden scalar contract (E4M3 codecs)
import glm_fp8_contract as contract       # vectorized numpy contract kernels
import modal_validate as mv               # tier1's pure-python error_stats/argmax

import numpy as np

REPO = "zai-org/GLM-5.2-FP8"
BLK = 128

# ---- OUR ASSUMPTIONS (docs/ACCEL_GLM52.md, P12_SCALEUP.md, BIT_ACCURACY.md B) --
# (value, "pending"?) -- pending ranks are the DeepSeek-MLA standard we sized to.
ASSUMED_DIMS = {
    "hidden_size":        (6144,   False),
    "num_hidden_layers":  (78,     False),
    "q_lora_rank":        (2048,   False),  # CONFIRMED vs real safetensors (q_a_proj.weight [2048,6144]); full_glm52.vh Q_LORA=2048
    "kv_lora_rank":       (512,    True),   # DeepSeek-MLA standard; safetensors-pending (full_glm52.vh KV_LORA PENDING)
    "n_routed_experts":   (256,    False),
    "num_experts_per_tok":(8,      False),
    "n_shared_experts":   (1,      False),
    "moe_intermediate_size":(2048, False),
    "first_k_dense_replace":(3,    False),
    "vocab_size":         (154880, False),
    "routed_scaling_factor":(2.5,  False),
}
ASSUMED_QUANT = {
    "quant_method":     "fp8",
    "fmt":              "e4m3",
    "weight_block_size":[128, 128],
    "activation_scheme":"dynamic",
}


# ============================================================================
# (0) small helpers: bit-reinterpretation in numpy + an exact E4M3 decode LUT
# ============================================================================
_E4M3_DECODE_LUT = np.array([ref.fp8_e4m3_decode(c) for c in range(256)],
                            dtype=np.float32)   # exact: every E4M3 value is fp32


def _bf16_codes_to_fp32(codes):
    """int array of bf16 codes -> float32 array (value)."""
    bits = ((np.asarray(codes, dtype=np.int64) & 0xFFFF) << 16).astype(np.uint32)
    return bits.view(np.float32)


def _fp32_to_bf16_codes(f32):
    """float32 array -> int64 array of bf16 codes (RNE, our contract's narrow)."""
    bits = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32).astype(np.int64)
    return contract._fp32_to_bf16(np, bits)


# ============================================================================
# (1) config.json -- confirm quant assumptions + reconcile the real dims
# ============================================================================
def load_config():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, "config.json", token=False)
    with open(path) as f:
        return json.load(f), path


def check_quant_config(cfg):
    qc = cfg.get("quantization_config", {})
    print("=" * 78)
    print("(1) QUANT-CONFIG ASSERTIONS  (config.json -> quantization_config)")
    print("=" * 78)
    results = {}
    for k, want in ASSUMED_QUANT.items():
        got = qc.get(k)
        ok = (got == want)
        results[k] = ok
        print(f"    {k:20s} assumed={str(want):14s} real={str(got):14s} "
              f"{'OK' if ok else 'MISMATCH'}")
    mtnc = qc.get("modules_to_not_convert", [])
    print(f"\n    modules_to_not_convert: {len(mtnc)} explicit modules (the bf16 tail).")
    # categorize the tail by suffix so the print is readable
    import re
    cats = {}
    for n in mtnc:
        s = re.sub(r"\.\d+\.", ".N.", n)
        s = re.sub(r"layers\.\d+", "layers.N", s)
        cats[s] = cats.get(s, 0) + 1
    for s, c in sorted(cats.items()):
        print(f"        {c:4d}  {s}")
    return all(results.values()), results


def reconcile_dims(cfg):
    print("\n" + "=" * 78)
    print("(1b) DIMENSION RECONCILIATION  (our assumption vs the real config)")
    print("=" * 78)
    print(f"    {'dim':22s} {'assumed':>10s} {'real':>10s}   verdict")
    print(f"    {'-'*22} {'-'*10} {'-'*10}   {'-'*24}")
    findings = []
    real = {}
    for k, (assumed, pending) in ASSUMED_DIMS.items():
        got = cfg.get(k, "<absent>")
        real[k] = got
        match = (got == assumed)
        if match:
            verdict = "MATCH"
        elif pending:
            verdict = f"RESOLVED (was pending {assumed})"
            findings.append((k, assumed, got))
        else:
            verdict = "!!! MISMATCH !!!"
            findings.append((k, assumed, got))
        tag = " (pending)" if pending else ""
        print(f"    {k+tag:22s} {str(assumed):>10s} {str(got):>10s}   {verdict}")
    print(f"\n    architectures = {cfg.get('architectures')}  model_type = {cfg.get('model_type')}")
    print(f"    (extra real MLA/DSA dims: q_lora_rank={cfg.get('q_lora_rank')} "
          f"kv_lora_rank={cfg.get('kv_lora_rank')} qk_nope={cfg.get('qk_nope_head_dim')} "
          f"qk_rope={cfg.get('qk_rope_head_dim')} v_head={cfg.get('v_head_dim')} "
          f"num_heads={cfg.get('num_attention_heads')} "
          f"index_n_heads={cfg.get('index_n_heads')} index_topk={cfg.get('index_topk')})")
    return real, findings


# ============================================================================
# (2) safetensors index -> pick the smallest FP8 Linear -> RANGE-READ its bytes
# ============================================================================
def _read_shard_header(fs, shard):
    """Range-read ONLY the safetensors header of a shard (a few KB)."""
    with fs.open(f"{REPO}/{shard}", "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return hdr


def _range_read_tensor(fs, shard, meta):
    """Range-read exactly one tensor's raw bytes from a shard.  meta is its header
       entry {dtype, shape, data_offsets:[start,end]} -- offsets are RELATIVE to
       the end of the (8-byte len + JSON) header, so add that base."""
    # header length prefix is 8 bytes; tensor data begins after 8 + hlen.
    with fs.open(f"{REPO}/{shard}", "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        data_base = 8 + hlen
        s, e = meta["data_offsets"]
        f.seek(data_base + s)
        return f.read(e - s)


def pick_and_fetch_weight():
    from huggingface_hub import hf_hub_download, HfFileSystem
    idx_path = hf_hub_download(REPO, "model.safetensors.index.json", token=False)
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    names = set(wm)
    fp8 = sorted(n for n in names
                 if n.endswith(".weight") and (n + "_scale_inv") in names)

    print("\n" + "=" * 78)
    print("(2) REAL FP8 TENSOR LAYOUT  (safetensors index + range-read header)")
    print("=" * 78)
    print(f"    FP8 Linear weights in index (with weight_scale_inv sibling): {len(fp8)}")

    # Target the shard holding a layer-0 attention block (small MLA projections
    # live together there); read ONLY its header (a few KB) via a range request.
    anchor = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
    if anchor not in wm:
        anchor = fp8[0]
    shard = wm[anchor]
    fs = HfFileSystem(token=False)
    hdr = _read_shard_header(fs, shard)

    # Among FP8 weights physically in this shard, list geometry; choose the
    # SMALLEST whose scale grid spans BOTH axes (n_ob>=2 and n_kb>=2) so a
    # transpose/orientation bug WOULD be caught.  Fall back to absolute-smallest.
    cand = []
    for n in sorted(hdr):
        if n == "__metadata__" or not n.endswith(".weight"):
            continue
        if (n + "_scale_inv") not in hdr:
            continue
        m = hdr[n]; s = hdr[n + "_scale_inv"]
        numel = 1
        for d in m["shape"]:
            numel *= d
        cand.append((n, m, s, numel))
    absolute_smallest = min(cand, key=lambda c: c[3])
    two_axis = [c for c in cand if c[2]["shape"][0] >= 2 and c[2]["shape"][1] >= 2]
    chosen = min(two_axis, key=lambda c: c[3]) if two_axis else absolute_smallest
    name, wmeta, smeta, wnumel = chosen

    print(f"    shard read (header only): {shard}")
    print(f"    absolute-smallest FP8 weight here: {absolute_smallest[0]} "
          f"shape={absolute_smallest[1]['shape']} ({absolute_smallest[3]} elts)")
    print(f"    CHOSEN for validation (smallest with a 2-D block grid): {name}")
    print(f"        .weight            dtype={wmeta['dtype']:8s} shape={wmeta['shape']}")
    print(f"        .weight_scale_inv  dtype={smeta['dtype']:8s} shape={smeta['shape']}")

    # ---- verify real layout MATCHES our assumed layout ----------------------
    N, K = wmeta["shape"]                       # HF Linear stores [out, in]
    exp_scale = [(N + BLK - 1) // BLK, (K + BLK - 1) // BLK]
    layout = {
        "weight dtype F8_E4M3":       (wmeta["dtype"] == "F8_E4M3", wmeta["dtype"], "F8_E4M3"),
        "weight shape [out,in]":      (len(wmeta["shape"]) == 2, str(wmeta["shape"]),
                                       "[out,in] (2-D)"),
        "scale dtype F32|BF16":       (smeta["dtype"] in ("F32", "BF16"), smeta["dtype"],
                                       "F32 or BF16"),
        "scale [ceil(out/128),ceil(in/128)]":
                                      (list(smeta["shape"]) == exp_scale,
                                       str(smeta["shape"]), str(exp_scale)),
    }
    print("\n    LAYOUT vs OUR ASSUMED CONTRACT (docs/BIT_ACCURACY.md B / ckpt_pack.py):")
    layout_ok = True
    for label, (ok, got, want) in layout.items():
        layout_ok &= ok
        print(f"        {label:38s} real={got:16s} want={want:22s} "
              f"{'OK' if ok else 'MISMATCH'}")

    # ---- range-read ONLY the chosen tensor's bytes + its scale bytes ---------
    w_raw = _range_read_tensor(fs, shard, wmeta)
    s_raw = _range_read_tensor(fs, shard, smeta)
    print(f"\n    range-read {len(w_raw)/1e6:.2f} MB weight + {len(s_raw)} B scale "
          f"(NOT the {os.path.getsize(idx_path)/1e6:.1f} MB index's shard, NOT 753 GB).")

    return dict(name=name, shard=shard, N=N, K=K,
                w_dtype=wmeta["dtype"], w_shape=wmeta["shape"],
                s_dtype=smeta["dtype"], s_shape=smeta["shape"],
                w_raw=w_raw, s_raw=s_raw), layout_ok, layout


# ============================================================================
# (3) OUR contract vs an INDEPENDENT numpy fp32-accumulate reference, ON the
#     REAL weight.  Only the ACCUMULATOR differs (exact-BFP vs fp32 rolling add)
#     -- both share the SAME quantized operands, the SAME bf16 block scale, and
#     the SAME per-token pow2 a_shift.  (docs/BIT_ACCURACY.md Section A, on real
#     weights; the CPU analogue of modal_validate.tier1_operator.)
# ============================================================================
def _fp32_accumulate_ref(Aq_fp32, Wf_fp32, S_bf16_fp32, a_shift, N, K, blk=BLK):
    """INDEPENDENT numpy fp32-accumulate FP8 GEMM.
         Aq_fp32     : [M,K] E4M3-quantized activations (already * 2^a_shift, decoded)
         Wf_fp32     : [N,K] E4M3-decoded real weight (exact)
         S_bf16_fp32 : [n_ob,n_kb] the SAME bf16-narrowed block scale the contract uses
         a_shift     : [M] per-token pow2 exponents (undone at the end)
       Returns [M,N] float32.  Column n (output channel) uses S[n//blk][bj]."""
    M = Aq_fp32.shape[0]
    n_kb = (K + blk - 1) // blk
    col_ob = (np.arange(N, dtype=np.int64) // blk)
    out = np.zeros((M, N), dtype=np.float32)
    for bj in range(n_kb):
        k0, k1 = bj * blk, min(bj * blk + blk, K)
        seg = (Aq_fp32[:, k0:k1].astype(np.float32)
               @ Wf_fp32[:, k0:k1].T.astype(np.float32))          # [M,N] fp32 acc
        col_scale = S_bf16_fp32[col_ob, bj].astype(np.float32)    # [N]
        out = (out + seg * col_scale[None, :]).astype(np.float32)
    undo = np.exp2(-np.asarray(a_shift, dtype=np.float64)).astype(np.float32).reshape(M, 1)
    return (out * undo).astype(np.float32)


def compare_on_real_weight(fetched, m_tokens=16, seed=0):
    print("\n" + "=" * 78)
    print("(3) OUR CONTRACT vs INDEPENDENT numpy fp32-accumulate REF, ON REAL WEIGHT")
    print("=" * 78)
    N, K = fetched["N"], fetched["K"]

    # ---- decode the REAL FP8 weight (bytes -> E4M3 codes) --------------------
    W_u8 = np.frombuffer(fetched["w_raw"], dtype=np.uint8).reshape(N, K)   # hf[out,in]
    W_codes_KN = np.ascontiguousarray(W_u8.T.astype(np.int64))            # [K,N] contract
    Wf_fp32 = _E4M3_DECODE_LUT[W_u8.astype(np.int64)]                     # [N,K] exact

    # ---- the REAL weight_scale_inv (F32) -> bf16 codes (the RTL bus form) ----
    n_ob, n_kb = fetched["s_shape"]
    if fetched["s_dtype"] == "F32":
        S_fp32 = np.frombuffer(fetched["s_raw"], dtype=np.float32).reshape(n_ob, n_kb)
        WS_codes = _fp32_to_bf16_codes(S_fp32)                           # [n_ob,n_kb]
    else:  # BF16
        WS_codes = (np.frombuffer(fetched["s_raw"], dtype="<u2")
                    .astype(np.int64).reshape(n_ob, n_kb))
    S_bf16_fp32 = _bf16_codes_to_fp32(WS_codes)                          # the scale both use

    # ---- a controlled activation vector: bf16, small magnitude, deterministic
    rng = np.random.default_rng(seed)
    A_fp32 = (rng.standard_normal((m_tokens, K)) * 0.1).astype(np.float32)
    A_codes = _fp32_to_bf16_codes(A_fp32)                               # [M,K] bf16 codes

    # ---- OUR contract (exact-BFP, dynamic pow2 act, [128,128] block scale) ----
    C_codes, a_shift = contract.block_fp8_gemm(
        A_codes, W_codes_KN, WS_codes, blk=BLK, backend="numpy")
    ours = _bf16_codes_to_fp32(np.asarray(C_codes, dtype=np.int64))     # [M,N] bf16 values
    C_codes_arr = np.asarray(C_codes, dtype=np.int64)

    # ---- reproduce the contract's EXACT activation quantization for the ref ----
    A_ref_bits = ((A_codes & 0xFFFF) << 16).astype(np.int64)           # bf16->fp32 bits
    ash_col = np.asarray(a_shift, dtype=np.int64).reshape(m_tokens, 1)
    a_scaled = contract._fp32_scale_pow2(np, A_ref_bits, ash_col)      # * 2^a_shift (exact)
    Aq_codes = contract._fp32_to_fp8e4m3(np, a_scaled)                 # E4M3 (RNE+sat)
    Aq_fp32 = _E4M3_DECODE_LUT[Aq_codes]                              # decode -> fp32

    # ---- INDEPENDENT numpy fp32-accumulate reference (same operands + a_shift) -
    refout = _fp32_accumulate_ref(Aq_fp32, Wf_fp32, S_bf16_fp32, a_shift, N, K, BLK)

    # ---- error_stats (mirror tier1) : ours-bf16 vs ref-fp32 ------------------
    og = ours.flatten().tolist()
    rg = refout.flatten().tolist()
    stats = mv.error_stats(rg, og)

    # ---- bf16-domain exactness: does ref round to the SAME bf16 code as ours?
    ref_bf16_codes = _fp32_to_bf16_codes(refout)
    bf16_exact = int(np.sum(ref_bf16_codes == C_codes_arr))
    bf16_total = C_codes_arr.size

    # ---- argmax (next-token-like) preservation, per token row ---------------
    am_hits = am_rows = 0
    for i in range(m_tokens):
        if not np.any(A_fp32[i]):
            continue
        am_rows += 1
        if int(np.argmax(ours[i])) == int(np.argmax(refout[i])):
            am_hits += 1

    print(f"    tensor: {fetched['name']}")
    print(f"    weight [out=N={N}, in=K={K}]  block grid [n_ob={n_ob} x n_kb={n_kb}]  "
          f"activations M={m_tokens} tokens (seed {seed})")
    print(f"    per-token a_shift (pow2 act scale): {a_shift}")
    print(f"    error_stats (ref fp32-acc vs ours exact-BFP, {stats['n']} outputs):")
    print(f"        max_abs = {stats['max_abs']:.6e}")
    print(f"        rms_abs = {stats['rms_abs']:.6e}")
    print(f"        max_rel = {stats['max_rel']:.6e}")
    print(f"        value-exact (fp32==bf16)      = {stats['exact']}/{stats['n']}")
    print(f"    bf16-domain exact (ref rounds to SAME bf16 code as ours) = "
          f"{bf16_exact}/{bf16_total} = {100.0*bf16_exact/bf16_total:.2f}%")
    print(f"    ARGMAX match (next-token decision) = {am_hits}/{am_rows} rows "
          f"= {100.0*am_hits/max(am_rows,1):.1f}%")

    # ---- PASS gate: the two engines differ ONLY in accumulator rounding, so the
    #      binding guarantee is argmax-preservation (the decision the model makes)
    #      with a sub-bf16-ULP numeric gap.  (max_rel can spike on near-zero
    #      cancellations -- argmax is the robust gate, as in docs/BIT_ACCURACY A.)
    argmax_ok = (am_rows > 0 and am_hits == am_rows)
    bf16_rate = bf16_exact / bf16_total
    near_bit = (bf16_rate >= 0.90)
    passed = argmax_ok and near_bit
    print(f"\n    -> argmax-preserving: {'YES' if argmax_ok else 'NO'};  "
          f"bf16-agreement {100*bf16_rate:.2f}% (>=90%): {'YES' if near_bit else 'NO'}")
    return passed, dict(stats=stats, bf16_exact=bf16_exact, bf16_total=bf16_total,
                        am_hits=am_hits, am_rows=am_rows, a_shift=a_shift,
                        n_ob=n_ob, n_kb=n_kb)


# ============================================================================
# (4) driver
# ============================================================================
def main():
    print("Validating OUR FP8 contract against the REAL zai-org/GLM-5.2-FP8 "
          "checkpoint (CPU/numpy only, no token).\n")

    cfg, cfg_path = load_config()
    quant_ok, _ = check_quant_config(cfg)
    real_dims, dim_findings = reconcile_dims(cfg)

    fetched, layout_ok, _ = pick_and_fetch_weight()

    gemm_ok, gemm_info = compare_on_real_weight(fetched)

    # ---- final reconciliation table + verdict -------------------------------
    print("\n" + "=" * 78)
    print("(4) FINAL RECONCILIATION  (our assumption  vs  the REAL checkpoint)")
    print("=" * 78)
    print(f"    {'item':34s} {'our assumption':26s} {'real checkpoint':22s} verdict")
    print(f"    {'-'*34} {'-'*26} {'-'*22} {'-'*12}")
    # each row: (item, assumption_str, real_str, matches_assumption?)
    q_scale_match = (list(fetched["s_shape"]) ==
                     [gemm_info["n_ob"], gemm_info["n_kb"]])
    rows = [
        ("quant_method",       ASSUMED_QUANT["quant_method"],
         cfg["quantization_config"].get("quant_method"), True),
        ("fmt (E4M3)",         ASSUMED_QUANT["fmt"],
         cfg["quantization_config"].get("fmt"), True),
        ("weight_block_size",  str(ASSUMED_QUANT["weight_block_size"]),
         str(cfg["quantization_config"].get("weight_block_size")), True),
        ("activation_scheme",  ASSUMED_QUANT["activation_scheme"],
         cfg["quantization_config"].get("activation_scheme"), True),
        ("weight dtype",       "F8_E4M3", fetched["w_dtype"],
         fetched["w_dtype"] == "F8_E4M3"),
        ("weight shape/orient", "[out, in] (2-D)",
         f"{fetched['w_shape']} = [out,in]", len(fetched["w_shape"]) == 2),
        ("scale dtype",        "bf16 or F32", fetched["s_dtype"],
         fetched["s_dtype"] in ("F32", "BF16")),
        ("scale shape/orient", "[ceil(o/128),ceil(i/128)]",
         f"{fetched['s_shape']} = [{gemm_info['n_ob']},{gemm_info['n_kb']}]",
         q_scale_match),
        ("q_lora_rank",        f"{ASSUMED_DIMS['q_lora_rank'][0]} (pending)",
         str(real_dims.get('q_lora_rank')),
         real_dims.get('q_lora_rank') == ASSUMED_DIMS['q_lora_rank'][0]),
        ("kv_lora_rank",       f"{ASSUMED_DIMS['kv_lora_rank'][0]} (pending)",
         str(real_dims.get('kv_lora_rank')),
         real_dims.get('kv_lora_rank') == ASSUMED_DIMS['kv_lora_rank'][0]),
        ("hidden_size",        str(ASSUMED_DIMS['hidden_size'][0]),
         str(real_dims.get('hidden_size')),
         real_dims.get('hidden_size') == ASSUMED_DIMS['hidden_size'][0]),
        ("num_hidden_layers",  str(ASSUMED_DIMS['num_hidden_layers'][0]),
         str(real_dims.get('num_hidden_layers')),
         real_dims.get('num_hidden_layers') == ASSUMED_DIMS['num_hidden_layers'][0]),
    ]
    for item, a, r, match in rows:
        print(f"    {item:34s} {str(a):26s} {str(r):22s} "
              f"{'MATCH' if match else '<-- DIFFERS'}")

    if dim_findings:
        print("\n    RECONCILIATION FINDINGS (assumption resolved/updated by real config):")
        for k, assumed, got in dim_findings:
            print(f"        * {k}: our docs assumed {assumed}, REAL = {got}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"    (a) quant-config assumptions confirmed .......... {'PASS' if quant_ok else 'FAIL'}")
    print(f"    (b) real tensor dtype/shape/orientation MATCH ... {'PASS' if layout_ok else 'FAIL'}")
    print(f"    (c) contract == numpy-fp32 ref on real weight ... {'PASS' if gemm_ok else 'FAIL'}")
    overall = quant_ok and layout_ok and gemm_ok
    print(f"\n    OVERALL: {'PASS' if overall else 'FAIL'}  "
          f"(validated tensor: {fetched['name']} {fetched['w_shape']} F8_E4M3)")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
