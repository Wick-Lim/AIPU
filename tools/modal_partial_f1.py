#!/usr/bin/env python3
# ============================================================================
# modal_partial_f1.py -- BUDGET-CAPPED multi-real-LAYER fidelity gate on Modal
# ----------------------------------------------------------------------------
# WHAT THIS IS (extends tools/modal_validate.py from OPERATOR -> ASSEMBLED LAYER)
#   tools/modal_validate.py::tier1_operator already proved, on REAL published
#   zai-org/GLM-5.2-FP8 weights, that OUR FP8 contract (glm_fp8_contract.
#   block_fp8_gemm) matches a reference fp32-accumulate FP8 GEMM per *single
#   Linear* (argmax-preserving, ~1 bf16 ULP -- see docs/REAL_CKPT_VALIDATION.md).
#   tools/glm_full_ref_np.py assembled a whole forward pass in numpy, but only on
#   the SYNTHETIC slice weights the TB drives.
#
#   THIS module closes the middle gap that neither covers: does OUR contract,
#   ASSEMBLED into a real decoder layer (RMSNorm -> MLA attn -> RMSNorm -> FFN,
#   residuals; FFN = dense SwiGLU for layers <3, MoE for layers >=3), still track
#   the reference across MULTIPLE real layers -- INCLUDING the dense->MoE
#   transition (first_k_dense_replace=3), a real plumbing risk point.  It does so
#   on a BUDGET: it downloads ONLY the shards for the first few real layers (not
#   the 753 GB model), caches them, runs a $1 SMOKE gate first (1 layer), then a
#   capped N-layer compare on the CHEAPEST GPU that fits.
#
# TWO COMPARE MODES (both reuse tier1's exact operator pairing + isolation):
#   mode="ffn"  (SOLID, default, T4/CPU-cheap): we ASSEMBLE the FFN block from the
#     real layer tensors ourselves (dense SwiGLU or 256-expert MoE + shared
#     expert), and for every FP8 Linear compare OUR contract vs the fp32-
#     accumulate reference with the SAME per-token a_shift -- exactly the
#     BIT_ACCURACY.md Sec.A accumulator isolation, now over an ASSEMBLED,
#     multi-GEMM real-weight FFN.  The router/top-k/SiLU nonlinearities are SHARED
#     across the two schemes (only the Linear numerics differ), so per-layer error
#     isolates the accumulator.  This ALWAYS runs -- it reads tensors only, never
#     instantiates the exotic GlmMoeDsa graph -- and covers the dense->MoE seam.
#     CAVEAT: it validates OUR-assembly vs OUR-fp32-ref; it does NOT re-derive
#     HF's exact 256-expert routing (that needs mode="layer").
#   mode="layer" (BEST-EFFORT "vs HF"): build the REAL HF GlmMoeDsaDecoderLayer
#     (trust_remote_code) for layer i WITHOUT loading the 753 GB model, monkeypatch
#     every FP8 Linear's forward to a scheme-selectable numeric (ours | fp32-acc),
#     run the SAME real layer twice, and compare the full-layer output (and, chained
#     over N layers + final norm + lm_head, the next-token argmax + logit rel-err).
#     This runs HF's REAL attention/DSA-indexer/gating/rope/norm/residual graph, so
#     it validates the ASSEMBLY STRUCTURE against HF -- the "vs HF" claim.
#     UNCERTAIN (documented): (a) GlmMoeDsa must be importable via transformers or
#     the repo's remote code; (b) NO budget GPU (T4/A100 are Turing/Ampere) has FP8
#     tensor cores, so we inject a *software* fp8 numeric into the linears rather
#     than run HF's native fp8 kernel -- true HF-fp8-KERNEL parity needs H100/H200
#     (that is tools/modal_validate.py::tier2 on H200:8, already written).  If any
#     of this fails, mode="auto" falls back to mode="ffn" and says so.
#
# ===========================  RUN PLAN  =====================================
#   (0) one-time auth (repo is PUBLIC/gated:False -> HF token OPTIONAL):
#         modal token new
#   (1) cache the first-6-layer shards into the Volume (CPU fn, NO GPU cost):
#         modal run tools/modal_partial_f1.py::download_layers --n-layers 6
#   (2) $1 SMOKE GATE -- 1 layer end-to-end, confirm the pipeline BEFORE spending:
#         modal run tools/modal_partial_f1.py --smoke 1
#       -> prints "SMOKE OK" + a cost estimate for the full run, or "SMOKE FAIL".
#   (3) the full N-layer compare (cheapest GPU that fits; T4 default):
#         modal run tools/modal_partial_f1.py --layers 6 --mode ffn
#         modal run tools/modal_partial_f1.py --layers 6 --mode auto --gpu a100
#
# ==========================  COST DISCIPLINE  ===============================
#   Rates (per task): T4 $0.59/hr, A100-80GB $2.50/hr, Modal CPU ~ $0.10-0.20/hr.
#   The DOWNLOAD is on a CPU function (never GPU) so no GPU credit is burned on I/O.
#   Per-step estimate (measured shard sizes ~5 GB each; ~1 MoE layer ~= 9-10 GB of
#   FP8, a dense layer ~= <1 GB; a shard packs several tensors):
#     | step             | GPU  | ~time   | ~$ (expected) | container-timeout CAP |
#     |------------------|------|---------|---------------|-----------------------|
#     | download_layers  | CPU  | 10-30m  | $0.03-0.20    | 2 h  (worst ~$0.40)   |
#     | smoke (1 layer)  | T4   | 3-8m    | $0.03-0.10    | 20 m (worst ~$0.20)   |
#     | compare ffn N=6  | T4   | 8-20m   | $0.10-0.25    | 60 m (worst ~$0.59)   |
#     | compare layer N=6| A100 | 15-35m  | $0.6-1.5      | 90 m (worst ~$3.75)   |
#   Expected TOTAL for the full plan (download + smoke + compare): ~$0.2-1.8.
#   WORST-CASE if every container hits its cap: ~$0.4 + $0.2 + $3.75 = ~$4.4,
#   comfortably under the $10 target and the $15 hard cap.  Every @app.function
#   sets an explicit `timeout=` so a hang cannot run away.  main() PRINTS the
#   up-front estimate before dispatching.
#   GPU rationale: our contract runs as CPU int64 GEMMs (tier1 verified this on
#   T4 already), and we software-dequant FP8 (no fp8 HW), so mode="ffn" needs NO
#   GPU at all (T4 is a convenience default; a single MoE layer's top-8 experts +
#   shared fit in <2 GB of working tensors).  mode="layer" builds the HF graph;
#   a MoE layer materialized dense is ~19 GB bf16 (> T4's 16 GB) so it defaults to
#   A100-80GB; dense-only layers fit T4.
#
# ==========================  SELF-VERIFY (here, no modal/torch)  ============
#   python3 -c "import ast; ast.parse(open('tools/modal_partial_f1.py').read())"
#   python3 tools/modal_partial_f1.py           # runs the pure-helper self-check
#   (Full GPU behavior is UNTESTED here -- no modal, no torch, no GPU. See the
#    "WHAT THE USER SHOULD WATCH FOR" block at the bottom.)
# ============================================================================
import sys
import os
import math

# Make the sibling tools/ importable both locally and inside the Modal container
# (mirrors glm_fp8_contract.py).  We REUSE modal_validate's proven helpers rather
# than re-derive them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# In the Modal container the sibling modules are mounted at /root/tools
# (image.add_local_dir(..., remote_path="/root/tools")), which is NOT the
# entrypoint's dir (/root); add it so `import modal_validate` resolves there too.
sys.path.insert(0, "/root/tools")

# Pure-python compare helpers + the in-container numeric refs -- all import-safe
# WITHOUT modal/torch (modal_validate ships a no-op shim; the numeric helpers only
# reference torch when CALLED, taking it as a parameter).
import modal_validate as mv
from modal_validate import (
    argmax, argmax_match_rate, topk_overlap, error_stats, summarize_gate,
    MODEL_ID, VOLUME_NAME, WEIGHTS_DIR, CKPT_DIR, HF_SECRET_NAME,
    _fp32_accumulate_fp8_gemm, _bf16_codes_to_torch,
)

# ----------------------------------------------------------------------------
# REAL GLM-5.2-FP8 architecture constants (from docs/REAL_CKPT_VALIDATION.md,
# confirmed against the real config.json).  Used for shard scoping + assembly.
# ----------------------------------------------------------------------------
HIDDEN            = 6144
Q_LORA            = 2048        # RESOLVED real value (docs guess was 1536)
KV_LORA           = 512
N_HIDDEN_LAYERS   = 78
N_DENSE           = 3           # first_k_dense_replace: layers 0..2 dense, 3+ MoE
N_ROUTED_EXPERTS  = 256
NUM_EXPERTS_TOK   = 8           # top-k
MOE_INTER         = 2048
VOCAB             = 154880
BLK               = 128
ROUTED_SCALE_DEF  = 2.5         # routed_scaling_factor fallback (read from config)

# GPU tiers (cheapest-that-fits).  See the cost table above.
GPU_SMOKE          = "T4"
GPU_COMPARE_T4     = "T4"
GPU_COMPARE_A100   = "A100-80GB"

# container-timeout CAPS (seconds) -- the runaway-cost guardrail
TO_DOWNLOAD = 2 * 60 * 60       # 2 h
TO_SMOKE    = 20 * 60           # 20 m
TO_COMPARE  = 90 * 60           # 90 m (A100 path); T4 path finishes well inside

# Default breadth: layers 0..5 = 3 dense + 3 MoE  -> covers the dense->MoE seam.
DEFAULT_N_LAYERS = 6

# ============================================================================
# MODAL APP  (built only when modal is importable; else the modal_validate shim).
# ============================================================================
try:
    import modal
    _HAS_MODAL = True
except Exception:
    modal = None
    _HAS_MODAL = False


if _HAS_MODAL:
    app = modal.App("glm52-fp8-partial-f1")

    _TOOLS_LOCAL = os.path.dirname(os.path.abspath(__file__))
    # "pip-install latest" transformers per the task; pin torch to a known-good
    # CUDA wheel.  accelerate/safetensors/hf_hub for the sharded load; numpy for
    # the contract's vectorized backend.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.5.1",
            "transformers",            # latest -- mode="layer" needs GlmMoeDsa
            "accelerate>=1.0",
            "safetensors>=0.4.5",
            "huggingface_hub>=0.26",
            "numpy>=1.26",
        )
        .add_local_dir(_TOOLS_LOCAL, remote_path="/root/tools")
    )
    # SHARE the same cache volume as modal_validate: if the user already pre-warmed
    # shards there (or ran tier1), we reuse them; partial fetches land in the SAME
    # CKPT_DIR and are additive.
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    # Repo is PUBLIC (gated:False) -> the secret is OPTIONAL.  from_name(...) with
    # a missing secret would raise at runtime, so we make it best-effort.
    try:
        hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
        _SECRETS = [hf_secret]
    except Exception:
        _SECRETS = []
else:
    app = mv._Shim()
    image = mv._Shim()
    volume = mv._Shim()
    _SECRETS = []


# ----------------------------------------------------------------------------
# in-container path + cache helpers
# ----------------------------------------------------------------------------
def _ensure_on_path():
    if "/root/tools" not in sys.path:
        sys.path.insert(0, "/root/tools")


def _index_path():
    for name in ("model.safetensors.index.json",
                 "model.safetensors.index.fp8.json"):
        p = os.path.join(CKPT_DIR, name)
        if os.path.exists(p):
            return p
    return os.path.join(CKPT_DIR, "model.safetensors.index.json")


def _load_weight_map():
    """{tensor_key -> shard_filename} from the safetensors index (cached)."""
    import json as _json
    with open(_index_path()) as f:
        return _json.load(f).get("weight_map", {})


def _layer_prefixes(layers):
    return tuple(f"model.layers.{int(i)}." for i in layers)


def _extras_keys():
    """Non-layer tensors used for the cumulative logit proxy (all bf16 tail)."""
    return ("model.norm.weight", "lm_head.weight")


# ============================================================================
# (1) PARTIAL DOWNLOAD  --  fetch ONLY the shards for the first N layers (+ the
#     bf16 tail for the logit proxy).  Runs on a CPU function (NO GPU cost),
#     idempotent, commits to the shared Volume.
# ============================================================================
def _partial_download(layers, want_extras=True):
    """Download config.json + index + any *.py modeling files + tokenizer, then
    ONLY the shards referenced by `model.layers.{i}.` for i in `layers` (plus the
    final norm + lm_head shards if want_extras).  Returns the set of shard files
    fetched.  Cached: hf_hub_download skips files already present."""
    import json as _json
    from huggingface_hub import hf_hub_download, list_repo_files

    os.makedirs(CKPT_DIR, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # --- always fetch the small metadata + remote-code modeling files ---
    small = []
    try:
        for fn in list_repo_files(MODEL_ID, token=token):
            base = os.path.basename(fn)
            if (fn.endswith((".json", ".py", ".txt", ".model"))
                    or base.startswith("tokenizer")):
                small.append(fn)
    except Exception as e:                      # offline / listing blocked
        print(f"[partial-dl] list_repo_files failed ({e}); using minimal set.")
        small = ["config.json", "model.safetensors.index.json"]
    for fn in small:
        try:
            hf_hub_download(repo_id=MODEL_ID, filename=fn, local_dir=CKPT_DIR,
                            token=token)
        except Exception as e:
            print(f"[partial-dl] skip {fn}: {e}")

    # --- resolve which shards hold the requested layers (+ extras) ---
    idx = _index_path()
    if not os.path.exists(idx):
        raise RuntimeError(f"no safetensors index at {idx}; cannot scope shards")
    weight_map = _json.load(open(idx)).get("weight_map", {})
    prefixes = _layer_prefixes(layers)
    wanted = set()
    for k, shard in weight_map.items():
        if k.startswith(prefixes):
            wanted.add(shard)
        elif want_extras and k in _extras_keys():
            wanted.add(shard)

    if not wanted:
        raise RuntimeError(f"no shards matched layers {list(layers)} in the index")

    print(f"[partial-dl] layers={list(layers)} -> {len(wanted)} shard(s): "
          f"{sorted(wanted)}")
    for shard in sorted(wanted):
        try:
            hf_hub_download(repo_id=MODEL_ID, filename=shard, local_dir=CKPT_DIR,
                            token=token)
        except Exception as e:
            print(f"[partial-dl] FAILED shard {shard}: {e}")
            raise
    try:
        volume.commit()
    except Exception:
        pass
    print(f"[partial-dl] done -> {CKPT_DIR}  ({len(wanted)} shard(s) cached)")
    return wanted


@app.function(image=image, volumes={WEIGHTS_DIR: volume}, secrets=_SECRETS,
              timeout=TO_DOWNLOAD)                    # CPU function -- NO gpu=
def download_layers(n_layers: int = DEFAULT_N_LAYERS):
    """Cache the shards for layers 0..n_layers-1 (+ bf16 tail) into the Volume.
    NO GPU: this is pure I/O, billed at the cheap CPU rate.  Idempotent."""
    _ensure_on_path()
    layers = list(range(int(n_layers)))
    shards = _partial_download(layers, want_extras=True)
    return dict(layers=layers, n_shards=len(shards), shards=sorted(shards))


# ============================================================================
# in-container tensor loading (reads the cached shards; NO 753 GB model load)
# ============================================================================
def _open_shard_cache():
    """A tiny lazy cache of safe_open handles keyed by shard filename."""
    from safetensors import safe_open
    handles = {}

    def get(shard):
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(CKPT_DIR, shard),
                                       framework="pt")
        return handles[shard]
    return get


def _load_layer_tensors(layer_idx, weight_map, opener):
    """Return {short_key -> torch.Tensor} for all `model.layers.{i}.` tensors that
    are actually present in the cached shards.  short_key strips the layer prefix
    (e.g. 'mlp.experts.3.up_proj.weight').  Missing shards are skipped (caller
    validates coverage)."""
    prefix = f"model.layers.{int(layer_idx)}."
    out = {}
    for k, shard in weight_map.items():
        if not k.startswith(prefix):
            continue
        if not os.path.exists(os.path.join(CKPT_DIR, shard)):
            continue
        try:
            out[k[len(prefix):]] = opener(shard).get_tensor(k)
        except Exception as e:
            print(f"[load] {k}: {e}")
    return out


def _get_tail_tensor(name, weight_map, opener):
    shard = weight_map.get(name)
    if shard is None or not os.path.exists(os.path.join(CKPT_DIR, shard)):
        return None
    try:
        return opener(shard).get_tensor(name)
    except Exception:
        return None


# ============================================================================
# in-container GEMM pair: OUR contract vs the fp32-accumulate reference, on ONE
# real FP8 Linear.  This is EXACTLY tier1_operator's packing + isolation, factored
# for reuse across the assembled FFN.  scheme in {"ours","gold"}.
# ============================================================================
def _gemm(torch, contract, A_bf16, W_fp8, S, scheme):
    """A_bf16 : [M,K] bf16 torch tensor. W_fp8 : [N,K] float8_e4m3fn (HF orient).
       S : [n_ob,n_kb] scale (fp32/bf16).  Returns [M,N] float32 torch tensor.
       Always runs OUR contract (to derive the per-token a_shift); 'gold' then
       re-runs the fp32-accumulate engine with that SAME a_shift (accumulator
       isolation), 'ours' returns the contract output directly."""
    A_codes = (A_bf16.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF)
    W_codes = (W_fp8.view(torch.uint8).to(torch.int64).t().contiguous())   # [K,N]
    S_bf16 = S.to(torch.bfloat16)
    WS_codes = (S_bf16.view(torch.int16).to(torch.int64) & 0xFFFF)
    C_codes, a_shift = contract.block_fp8_gemm(
        A_codes, W_codes, WS_codes, blk=BLK, backend="torch")
    if scheme == "ours":
        return _bf16_codes_to_torch(torch, C_codes)
    # 'gold': fp32-accumulate FP8 with the SAME a_shift  (ref param unused)
    return _fp32_accumulate_fp8_gemm(torch, None, A_bf16.to(torch.bfloat16),
                                     W_fp8, S, blk=BLK, a_shift=a_shift)


def _sib_scale(tensors, wkey):
    """The weight_scale_inv sibling of an FP8 weight key, else None (bf16 Linear)."""
    return tensors.get(wkey + "_scale_inv", tensors.get(
        wkey.replace(".weight", ".weight_scale_inv")))


# ============================================================================
# (3) ASSEMBLED FFN forward (mode="ffn"): dense SwiGLU (layer<3) or MoE (layer>=3)
#     computed in BOTH numeric schemes with SHARED nonlinearities, on REAL weights.
# ============================================================================
def _silu(torch, x):
    return x / (1.0 + torch.exp(-x))


def _linear_bf16(torch, x, W):
    """Plain bf16 Linear (the bf16-tail: router gate).  x:[M,K] W:[out,K]."""
    return x.to(torch.float32) @ W.to(torch.float32).t()


def _swiglu(torch, contract, x_bf16, wg, sg, wu, su, wd, sd, scheme):
    """down( silu(gate(x)) * up(x) ) for one expert, given real fp8 weights+scales.
       All matmuls via _gemm(scheme); SiLU/mul shared across schemes."""
    gate = _gemm(torch, contract, x_bf16, wg, sg, scheme)          # [M, inter]
    up = _gemm(torch, contract, x_bf16, wu, su, scheme)            # [M, inter]
    h = _silu(torch, gate) * up                                    # [M, inter]
    h_bf16 = h.to(torch.bfloat16)
    y = _gemm(torch, contract, h_bf16, wd, sd, scheme)            # [M, hidden]
    return y


def _find_dense_ffn(tensors):
    """Return (wg,sg,wu,su,wd,sd) for a dense layer, or None."""
    def grab(name):
        w = tensors.get(f"mlp.{name}.weight")
        if w is None:
            return None, None
        return w, _sib_scale(tensors, f"mlp.{name}.weight")
    wg, sg = grab("gate_proj")
    wu, su = grab("up_proj")
    wd, sd = grab("down_proj")
    if wg is None or wu is None or wd is None:
        return None
    return (wg, sg, wu, su, wd, sd)


def _find_experts(tensors):
    """{expert_idx -> (wg,sg,wu,su,wd,sd)} for MoE routed experts present."""
    import re
    pat = re.compile(r"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
    experts = {}
    for k in tensors:
        m = pat.match(k)
        if not m:
            continue
        e = int(m.group(1))
        experts.setdefault(e, {})[m.group(2)] = k
    out = {}
    for e, d in experts.items():
        if {"gate", "up", "down"} <= set(d):
            out[e] = (tensors[d["gate"]], _sib_scale(tensors, d["gate"]),
                      tensors[d["up"]],   _sib_scale(tensors, d["up"]),
                      tensors[d["down"]], _sib_scale(tensors, d["down"]))
    return out


def _find_shared(tensors):
    for base in ("mlp.shared_experts", "mlp.shared_expert"):
        wg = tensors.get(f"{base}.gate_proj.weight")
        wu = tensors.get(f"{base}.up_proj.weight")
        wd = tensors.get(f"{base}.down_proj.weight")
        if wg is not None and wu is not None and wd is not None:
            return (wg, _sib_scale(tensors, f"{base}.gate_proj.weight"),
                    wu, _sib_scale(tensors, f"{base}.up_proj.weight"),
                    wd, _sib_scale(tensors, f"{base}.down_proj.weight"))
    return None


def _moe_route(torch, x_bf16, gate_w, corr_bias, topk, routed_scale):
    """SHARED router: sigmoid scores, (score+correction_bias) selection top-k,
       original-score normalization * routed_scaling_factor.  Group-limited
       selection is intentionally OMITTED (documented) -- both schemes use the
       identical selection, so this does not affect the accumulator isolation.
       Returns (idx list, weight list) for the M==1 token."""
    logits = _linear_bf16(torch, x_bf16, gate_w)[0]               # [n_experts]
    scores = torch.sigmoid(logits)
    choose = scores + (corr_bias.to(torch.float32) if corr_bias is not None else 0.0)
    k = min(topk, scores.shape[0])
    sel = torch.topk(choose, k).indices.tolist()
    w = scores[torch.tensor(sel, dtype=torch.long)]
    w = w / (w.sum() + 1e-20) * routed_scale
    return sel, w.tolist()


def _ffn_forward(torch, contract, x_bf16, tensors, is_moe, cfg, scheme, selection):
    """Assembled FFN output [1,HIDDEN] float32 for the given numeric `scheme`.
       `selection` (idx,weights) is computed ONCE from the bf16 router and shared
       across schemes so both evaluate the identical experts."""
    if not is_moe:
        wg, sg, wu, su, wd, sd = _find_dense_ffn(tensors)
        return _swiglu(torch, contract, x_bf16, wg, sg, wu, su, wd, sd, scheme)

    experts = _find_experts(tensors)
    sel, weights = selection
    acc = torch.zeros(1, HIDDEN, dtype=torch.float32)
    for e, gw in zip(sel, weights):
        if e not in experts:
            continue
        wg, sg, wu, su, wd, sd = experts[e]
        y = _swiglu(torch, contract, x_bf16, wg, sg, wu, su, wd, sd, scheme)
        acc = acc + gw * y
    shared = _find_shared(tensors)
    if shared is not None:
        wg, sg, wu, su, wd, sd = shared
        acc = acc + _swiglu(torch, contract, x_bf16, wg, sg, wu, su, wd, sd, scheme)
    return acc


def _cfg_get(default_scale=ROUTED_SCALE_DEF):
    """Read routed_scaling_factor / topk from the cached config.json (best-effort)."""
    import json as _json
    routed_scale, topk = default_scale, NUM_EXPERTS_TOK
    try:
        cfg = _json.load(open(os.path.join(CKPT_DIR, "config.json")))
        routed_scale = float(cfg.get("routed_scaling_factor", default_scale))
        topk = int(cfg.get("num_experts_per_tok", NUM_EXPERTS_TOK))
    except Exception:
        pass
    return routed_scale, topk


def _synthetic_input(torch, seed):
    """Deterministic realistic bf16 hidden state [1,HIDDEN] (like tier1)."""
    torch.manual_seed(seed)
    return (torch.randn(1, HIDDEN, dtype=torch.float32) * 0.1).to(torch.bfloat16)


def _logit_proxy(torch, ffn_out, norm_w, lm_head_w):
    """final RMSNorm(no-eps-detail) + lm_head -> logits, for an argmax proxy.
       Returns a python list[VOCAB] or None if the tail wasn't downloaded."""
    if norm_w is None or lm_head_w is None:
        return None
    x = ffn_out.to(torch.float32)[0]
    rms = torch.rsqrt((x * x).mean() + 1e-5)
    xn = (x * rms) * norm_w.to(torch.float32)
    logits = xn @ lm_head_w.to(torch.float32).t()                 # [VOCAB]
    return logits.tolist()


def _compare_layer_ffn(torch, contract, layer_idx, tensors, cfg, tail, seed):
    """FFN-mode per-layer compare: OUR contract vs fp32-accumulate over the
       assembled real-weight FFN.  Returns a stats dict."""
    is_moe = (layer_idx >= N_DENSE)
    x = _synthetic_input(torch, seed)
    routed_scale, topk = cfg

    # shared routing selection (MoE only)
    selection = ([], [])
    if is_moe:
        gate_w = tensors.get("mlp.gate.weight")
        corr = tensors.get("mlp.gate.e_score_correction_bias")
        if gate_w is None:
            return dict(layer=layer_idx, error="no mlp.gate.weight (router) present")
        selection = _moe_route(torch, x, gate_w, corr, topk, routed_scale)

    out_ours = _ffn_forward(torch, contract, x, tensors, is_moe, cfg, "ours", selection)
    out_gold = _ffn_forward(torch, contract, x, tensors, is_moe, cfg, "gold", selection)

    og = out_ours.flatten().tolist()
    rg = out_gold.flatten().tolist()
    st = error_stats(rg, og)

    # argmax proxy through the bf16 tail (final norm + lm_head), if available
    norm_w, lm_head_w = tail
    lo = _logit_proxy(torch, out_ours, norm_w, lm_head_w)
    lg = _logit_proxy(torch, out_gold, norm_w, lm_head_w)
    am_ours = am_gold = None
    if lo is not None and lg is not None:
        am_ours, am_gold = argmax(lo), argmax(lg)

    return dict(
        layer=layer_idx, kind=("moe" if is_moe else "dense"),
        n_out=st["n"], bf16_exact=f"{st['exact']}/{st['n']}",
        max_abs=st["max_abs"], rms_abs=st["rms_abs"], max_rel=st["max_rel"],
        experts=(selection[0] if is_moe else None),
        argmax_ours=am_ours, argmax_gold=am_gold,
        argmax_match=(None if am_ours is None else int(am_ours == am_gold)),
    )


# ============================================================================
# mode="layer" (BEST-EFFORT "vs HF"): build the REAL HF decoder layer and run it
# twice with scheme-selectable FP8 Linear numerics.  Returns None if the arch/
# loader is unavailable (caller falls back to FFN mode).
# ============================================================================
# module-global the patched linear forwards read -- flips golden<->candidate
_LAYER_SCHEME = "ours"


def _try_build_hf_layer(torch, transformers, layer_idx, weight_map, opener):
    """Instantiate a single GlmMoeDsaDecoderLayer WITHOUT loading the 753 GB model,
       load its real weights, and monkeypatch every FP8 Linear's forward to a
       scheme-selectable numeric.  Raises on any unsupported step (caught by the
       caller).  Returns (layer_module, meta) or raises."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(CKPT_DIR, trust_remote_code=True)

    # get the decoder-layer class from the loaded model class's module
    from transformers import AutoModelForCausalLM
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    # navigate to the layer list (standard HF layout)
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None or layer_idx >= len(layers):
        raise RuntimeError("could not locate model.model.layers for GlmMoeDsa")
    layer = layers[layer_idx]

    # materialize on CPU (uninitialized), then load real tensors (strict=False so
    # missing/extra keys don't fail; fp8 linears' own .weight is overridden anyway)
    layer = layer.to_empty(device="cpu")
    tens = _load_layer_tensors(layer_idx, weight_map, opener)
    # split into bf16-tail (loadable into params) vs fp8 (patched)
    fp8_mods = _patch_fp8_linears(torch, layer, tens)
    _load_bf16_tail(torch, layer, tens)
    # Run the whole layer in float32 on CPU: the bf16 tail (norms) + float32 hidden
    # otherwise raise "mixed dtype (CPU): expect parameter to have scalar type of
    # Float".  .float() only casts float params/buffers (int index buffers survive);
    # the fp8-patched forwards ignore the module weight and preserve x.dtype.
    layer = layer.float()
    return layer, dict(patched=len(fp8_mods))


def _patch_fp8_linears(torch, layer, tens):
    """For every submodule whose '<name>.weight' has a '<name>.weight_scale_inv'
       sibling in `tens`, attach the real fp8 weight+scale and override forward to
       route through _gemm() under the current _LAYER_SCHEME.  Returns patched
       module names."""
    from modal_validate import _bf16_codes_to_torch as _codes  # noqa: F401
    import glm_fp8_contract as contract
    patched = []
    named = dict(layer.named_modules())
    for wkey in list(tens.keys()):
        if not wkey.endswith(".weight"):
            continue
        skey = wkey[:-len(".weight")] + ".weight_scale_inv"
        if skey not in tens:
            continue
        mod_name = wkey[:-len(".weight")]
        module = named.get(mod_name)
        if module is None:
            continue
        W = tens[wkey]                                   # [N,K] fp8
        S = tens[skey]                                   # [n_ob,n_kb]
        bias = getattr(module, "bias", None)

        def make_forward(W=W, S=S, bias=bias):
            def forward(x):
                orig = x.shape
                x2 = x.reshape(-1, orig[-1]).to(torch.bfloat16)
                y = _gemm(torch, contract, x2, W, S, _LAYER_SCHEME).to(x.dtype)
                y = y.reshape(*orig[:-1], y.shape[-1])
                if bias is not None:
                    y = y + bias.to(y.dtype)
                return y
            return forward
        module.forward = make_forward()
        patched.append(mod_name)
    return patched


def _load_bf16_tail(torch, layer, tens):
    """Load the non-fp8 (bf16) params -- norms, router gate, correction bias --
       into the layer's own parameters/buffers via load_state_dict(strict=False)."""
    sd = {}
    for k, v in tens.items():
        if k.endswith(".weight") and (k[:-len(".weight")] + ".weight_scale_inv") in tens:
            continue                                     # fp8 weight: skip (patched)
        if k.endswith(".weight_scale_inv"):
            continue
        sd[k] = v
    try:
        layer.load_state_dict(sd, strict=False, assign=True)
    except Exception as e:
        print(f"[layer] partial load_state_dict note: {e}")


def _compare_layers_hf(torch, transformers, layer_indices, weight_map, opener,
                       tail, seed):
    """Chain the real HF layers, running the whole chain twice (gold then ours),
       and compare per-layer output + the cumulative next-token argmax + logit
       rel-err.  Returns (per_layer_stats, cumulative_stats) or raises."""
    global _LAYER_SCHEME
    built = []
    for i in layer_indices:
        layer, meta = _try_build_hf_layer(torch, transformers, i, weight_map, opener)
        built.append((i, layer, meta))
        print(f"[layer] built HF decoder layer {i} (patched {meta['patched']} fp8 Linears)")

    hidden0 = _synthetic_input(torch, seed).to(torch.float32).unsqueeze(0)  # [1,1,H]
    dev = hidden0.device
    seqlen = hidden0.shape[1]
    position_ids = torch.arange(seqlen, device=dev).unsqueeze(0)

    # Standalone HF decoder layers unpack `cos, sin = position_embeddings` inside
    # attention, so we MUST supply it.  The meta-built model's rotary_emb has meta
    # buffers, so re-instantiate the SAME rotary class on the real device (it is tiny
    # -- just inv_freq derived from config).
    pos_emb = None
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        _cfg = AutoConfig.from_pretrained(CKPT_DIR, trust_remote_code=True)
        with torch.device("meta"):
            _m = AutoModelForCausalLM.from_config(_cfg, trust_remote_code=True)
        _rot_cls = type(getattr(getattr(_m, "model", _m), "rotary_emb"))
        try:
            _rot = _rot_cls(config=_cfg).to(dev)
        except TypeError:
            _rot = _rot_cls(_cfg).to(dev)
        pos_emb = _rot(hidden0, position_ids)   # (cos, sin)
        print("[layer] rotary ready -> position_embeddings supplied")
    except Exception as _e:
        print(f"[layer] rotary unavailable ({type(_e).__name__}: {_e}); trying without")

    _cache_pos = position_ids[0]
    def _call_layer(layer, h):
        # arch/version-tolerant: try richest signature first, degrade to minimal.
        attempts = [
            dict(position_embeddings=pos_emb, position_ids=position_ids,
                 attention_mask=None, use_cache=False, cache_position=_cache_pos),
            dict(position_embeddings=pos_emb, position_ids=position_ids,
                 attention_mask=None, use_cache=False),
            dict(position_embeddings=pos_emb, position_ids=position_ids),
            dict(position_embeddings=pos_emb),
            dict(position_ids=position_ids),
            dict(),
        ]
        errs = []
        for kw in attempts:
            kw = {k: v for k, v in kw.items()
                  if not (k == "position_embeddings" and v is None)}
            try:
                return layer(h, **kw)
            except Exception as e:
                errs.append(f"[{'+'.join(kw.keys()) or 'bare'}] {type(e).__name__}: {e}")
        raise RuntimeError("all layer-call signatures failed:\n    "
                           + "\n    ".join(errs))

    def run_chain():
        h = hidden0
        outs = []
        for i, layer, _ in built:
            res = _call_layer(layer, h)
            h = res[0] if isinstance(res, (tuple, list)) else res
            outs.append(h)
        return h, outs

    _LAYER_SCHEME = "gold"
    hg, outs_g = run_chain()
    _LAYER_SCHEME = "ours"
    ho, outs_o = run_chain()

    per_layer = []
    for (i, _, _), og, gg in zip(built, outs_o, outs_g):
        st = error_stats(gg.flatten().tolist(), og.flatten().tolist())
        per_layer.append(dict(layer=i, max_abs=st["max_abs"], rms_abs=st["rms_abs"],
                              max_rel=st["max_rel"],
                              bf16_exact=f"{st['exact']}/{st['n']}"))

    # cumulative: final norm + lm_head -> logits -> argmax + rel-err
    norm_w, lm_head_w = tail
    cum = dict(logit_proxy=False)
    lo = _logit_proxy(torch, ho[0], norm_w, lm_head_w)
    lg = _logit_proxy(torch, hg[0], norm_w, lm_head_w)
    if lo is not None and lg is not None:
        st = error_stats(lg, lo)
        cum = dict(logit_proxy=True, argmax_ours=argmax(lo), argmax_gold=argmax(lg),
                   argmax_match=int(argmax(lo) == argmax(lg)),
                   logit_max_rel=st["max_rel"], logit_max_abs=st["max_abs"],
                   topk8_overlap=topk_overlap(lg, lo, 8))
    return per_layer, cum


# ============================================================================
# CORE compare driver (mode auto|ffn|layer) -- shared by the T4 / A100 functions.
# ============================================================================
def _compare_impl(n_layers, mode, seed=0):
    _ensure_on_path()
    import torch
    import glm_fp8_contract as contract

    layers = list(range(int(n_layers)))
    # ensure the shards are cached (idempotent)
    if not os.path.exists(_index_path()):
        _partial_download(layers, want_extras=True)
    weight_map = _load_weight_map()
    # any requested layer missing a shard? fetch just those.
    prefixes = _layer_prefixes(layers)
    missing = any(k.startswith(prefixes)
                  and not os.path.exists(os.path.join(CKPT_DIR, s))
                  for k, s in weight_map.items())
    if missing:
        _partial_download(layers, want_extras=True)
        weight_map = _load_weight_map()

    opener = _open_shard_cache()
    cfg = _cfg_get()
    tail = (_get_tail_tensor("model.norm.weight", weight_map, opener),
            _get_tail_tensor("lm_head.weight", weight_map, opener))

    used_mode = mode
    result = dict(mode_requested=mode, n_layers=n_layers)

    # ---- mode=layer / auto: try the real HF assembled layer chain ----
    if mode in ("layer", "auto"):
        try:
            import transformers
            per_layer, cum = _compare_layers_hf(
                torch, transformers, layers, weight_map, opener, tail, seed)
            result.update(mode_used="layer", per_layer=per_layer, cumulative=cum)
            _print_layer_result(result)
            return result
        except Exception as e:
            print(f"[compare] mode='layer' unavailable ({type(e).__name__}: {e})")
            if mode == "layer":
                result.update(mode_used="layer-FAILED", error=str(e))
                return result
            print("[compare] falling back to mode='ffn' (accumulator isolation on "
                  "the assembled real-weight FFN).")
            used_mode = "ffn"

    # ---- mode=ffn (solid) ----
    per_layer = []
    for i in layers:
        tens = _load_layer_tensors(i, weight_map, opener)
        if not tens:
            per_layer.append(dict(layer=i, error="no tensors cached for this layer"))
            continue
        per_layer.append(_compare_layer_ffn(torch, contract, i, tens, cfg, tail, seed + i))
    result.update(mode_used="ffn", per_layer=per_layer,
                  cumulative=_aggregate_ffn(per_layer))
    _print_ffn_result(result)
    return result


def _aggregate_ffn(per_layer):
    """FFN mode has no true token chain (no attention/residual here); the
       'cumulative' summary aggregates per-layer isolation across the N layers."""
    rows = [r for r in per_layer if "max_rel" in r]
    if not rows:
        return dict(note="no comparable layers")
    am = [r["argmax_match"] for r in rows if r.get("argmax_match") is not None]
    return dict(
        note="aggregate over N independently-evaluated FFN blocks (not a token chain)",
        layers=len(rows),
        worst_max_rel=max(r["max_rel"] for r in rows),
        worst_max_abs=max(r["max_abs"] for r in rows),
        mean_rms_abs=sum(r["rms_abs"] for r in rows) / len(rows),
        argmax_proxy_match=(None if not am else f"{sum(am)}/{len(am)}"),
        dense_moe_transition_covered=(any(r.get("kind") == "dense" for r in rows)
                                      and any(r.get("kind") == "moe" for r in rows)),
    )


def _print_ffn_result(result):
    print("=== PARTIAL-F1 (mode=ffn): OUR contract vs fp32-accumulate over the "
          "ASSEMBLED real-weight FFN ===")
    for r in result["per_layer"]:
        if "error" in r:
            print(f"  layer {r['layer']}: ERROR {r['error']}")
            continue
        am = "" if r["argmax_match"] is None else f" argmax_proxy={r['argmax_match']}"
        print(f"  L{r['layer']:<2d} [{r['kind']:<5s}] bf16_exact={r['bf16_exact']} "
              f"max_rel={r['max_rel']:.3e} max_abs={r['max_abs']:.3e}{am}")
    print(f"  cumulative: {result['cumulative']}")


def _print_layer_result(result):
    print("=== PARTIAL-F1 (mode=layer): OUR contract vs fp32-accumulate INSIDE the "
          "REAL HF GlmMoeDsa layer chain ===")
    for r in result["per_layer"]:
        print(f"  L{r['layer']:<2d} bf16_exact={r['bf16_exact']} "
              f"max_rel={r['max_rel']:.3e} max_abs={r['max_abs']:.3e}")
    print(f"  cumulative (final norm + lm_head): {result['cumulative']}")


# ============================================================================
# (2) SMOKE GATE -- run ONE layer end-to-end, cheap, BEFORE the full spend.
# ============================================================================
def _smoke_impl(layer_idx, mode):
    _ensure_on_path()
    print(f"=== SMOKE GATE: 1 layer (idx={layer_idx}), mode={mode} ===")
    # only this layer's shards need to be present
    if not os.path.exists(_index_path()):
        _partial_download([layer_idx], want_extras=True)
    weight_map = _load_weight_map()
    prefix = f"model.layers.{layer_idx}."
    if any(k.startswith(prefix) and not os.path.exists(os.path.join(CKPT_DIR, s))
           for k, s in weight_map.items()):
        _partial_download([layer_idx], want_extras=True)

    import torch
    import glm_fp8_contract as contract
    opener = _open_shard_cache()
    cfg = _cfg_get()
    tail = (_get_tail_tensor("model.norm.weight", weight_map, opener),
            _get_tail_tensor("lm_head.weight", weight_map, opener))
    tens = _load_layer_tensors(layer_idx, weight_map, opener)
    if not tens:
        print("SMOKE FAIL: no tensors cached for this layer (download issue).")
        return dict(ok=False, reason="no tensors")

    # try the requested mode for a single layer; fall back to ffn
    used = mode
    if mode in ("layer", "auto"):
        try:
            import transformers
            per_layer, cum = _compare_layers_hf(
                torch, transformers, [layer_idx], weight_map, opener, tail, 0)
            print(f"[smoke] HF layer per-layer: {per_layer}  cumulative: {cum}")
            used = "layer"
        except Exception as e:
            print(f"[smoke] mode='layer' unavailable ({type(e).__name__}: {e})")
            if mode == "layer":
                print("SMOKE FAIL: requested mode=layer could not build the HF graph.")
                return dict(ok=False, reason=f"layer build failed: {e}")
            used = "ffn"
    if used == "ffn":
        r = _compare_layer_ffn(torch, contract, layer_idx, tens, cfg, tail, 0)
        if "error" in r:
            print(f"SMOKE FAIL: {r['error']}")
            return dict(ok=False, reason=r["error"])
        print(f"[smoke] FFN per-layer: {r}")

    print("SMOKE OK: real shard load -> assembled forward (ours) -> fp32-acc ref "
          f"-> compare all succeeded (mode_used={used}).")
    print("[smoke] COST NOTE: the full N-layer compare is estimated at "
          "$0.10-0.25 on T4 (mode=ffn) or ~$0.6-1.5 on A100-80GB (mode=layer); "
          "download already cached.  Proceed with:  modal run "
          "tools/modal_partial_f1.py --layers 6 --mode auto")
    return dict(ok=True, mode_used=used, layer=layer_idx)


# ============================================================================
# @app.function wrappers (GPU tiers) -- thin shells over the impls above.
# ============================================================================
@app.function(image=image, gpu=GPU_SMOKE, volumes={WEIGHTS_DIR: volume},
              secrets=_SECRETS, timeout=TO_SMOKE)
def smoke_gate(layer_idx: int = 0, mode: str = "auto"):
    return _smoke_impl(int(layer_idx), mode)


@app.function(image=image, gpu=GPU_COMPARE_T4, volumes={WEIGHTS_DIR: volume},
              secrets=_SECRETS, timeout=TO_COMPARE)
def compare_t4(n_layers: int = DEFAULT_N_LAYERS, mode: str = "ffn"):
    return _compare_impl(int(n_layers), mode)


@app.function(image=image, gpu=GPU_COMPARE_A100, volumes={WEIGHTS_DIR: volume},
              secrets=_SECRETS, timeout=TO_COMPARE)
def compare_a100(n_layers: int = DEFAULT_N_LAYERS, mode: str = "auto"):
    return _compare_impl(int(n_layers), mode)


# ============================================================================
# LOCAL ENTRYPOINT -- prints the up-front cost estimate, then dispatches.
# ============================================================================
@app.local_entrypoint()
def main(layers: int = DEFAULT_N_LAYERS, mode: str = "ffn", gpu: str = "t4",
         smoke: int = 0, download_only: int = 0):
    """`modal run tools/modal_partial_f1.py [--layers N] [--mode ffn|layer|auto]
                                            [--gpu t4|a100] [--smoke 1]
                                            [--download-only 1]`

       --smoke 1        : run ONLY the 1-layer smoke gate (cheapest; do this first)
       --download-only 1: only cache the shards (CPU fn, no GPU)
       --mode ffn       : SOLID accumulator isolation over the assembled FFN (T4)
       --mode auto      : try the real HF layer chain (vs HF), fall back to ffn
       --gpu a100       : use A100-80GB (needed if mode=layer MoE won't fit T4)
    """
    est = {("t4", "ffn"): "~$0.10-0.25", ("t4", "auto"): "~$0.10-0.25 (T4; HF-layer "
           "MoE may not fit -> use --gpu a100)", ("t4", "layer"): "T4 may OOM on MoE "
           "layers; consider --gpu a100", ("a100", "auto"): "~$0.6-1.5",
           ("a100", "layer"): "~$0.6-1.5", ("a100", "ffn"): "~$0.15-0.4"}
    print("############################################################")
    print("# GLM-5.2-FP8 PARTIAL-F1 multi-layer fidelity gate")
    print(f"#   layers=0..{layers-1}  mode={mode}  gpu={gpu}")
    print(f"#   est. compare cost: {est.get((gpu, mode), 'see header cost table')}")
    print("#   (download on CPU fn ~ $0.03-0.20; smoke ~ $0.03-0.10)")
    print("#   HARD CAPS: download 2h, smoke 20m, compare 90m (worst-case ~$4.4)")
    print("############################################################")

    if download_only:
        r = download_layers.remote(int(layers))
        print(f"[download] {r}")
        return

    if smoke:
        r = smoke_gate.remote(0, mode)               # always the cheapest GPU (T4)
        print(f"[smoke] result: {r}")
        return

    if gpu == "a100":
        r = compare_a100.remote(int(layers), mode)
    else:
        r = compare_t4.remote(int(layers), mode)
    print(f"[compare] mode_used={r.get('mode_used')}  cumulative={r.get('cumulative')}")


# ============================================================================
# LOCAL SELF-CHECK (no modal / torch / GPU): exercise the pure, torch-free logic
# so `python3 tools/modal_partial_f1.py` is a quick smoke test in this repo.
# ============================================================================
def _selftest():
    # reused helpers resolve from modal_validate
    assert argmax([1, 5, 2]) == 1
    assert argmax_match_rate([1, 2], [1, 2]) == 1.0
    assert error_stats([1.0, 2.0], [1.0, 2.5])["exact"] == 1
    assert abs(topk_overlap([4, 3, 2, 1], [4, 3, 1, 0], 2) - 1.0) < 1e-12

    # shard-scoping logic is pure -- test with a fake weight_map
    global CKPT_DIR
    wm = {
        "model.layers.0.mlp.gate_proj.weight": "s0.safetensors",
        "model.layers.0.self_attn.q_a_proj.weight": "s0.safetensors",
        "model.layers.3.mlp.experts.5.up_proj.weight": "s1.safetensors",
        "model.layers.7.mlp.gate.weight": "s2.safetensors",
        "model.norm.weight": "s9.safetensors",
        "lm_head.weight": "s9.safetensors",
    }
    prefixes = _layer_prefixes([0, 3])
    got = set()
    for k, sh in wm.items():
        if k.startswith(prefixes):
            got.add(sh)
        elif k in _extras_keys():
            got.add(sh)
    assert got == {"s0.safetensors", "s1.safetensors", "s9.safetensors"}, got
    assert "s2.safetensors" not in got                # layer 7 NOT requested

    # _sib_scale resolution
    tens = {"mlp.up_proj.weight": 1, "mlp.up_proj.weight_scale_inv": 2}
    assert _sib_scale(tens, "mlp.up_proj.weight") == 2
    assert _sib_scale({"a.weight": 1}, "a.weight") is None

    # _aggregate_ffn shape + dense/moe transition flag
    agg = _aggregate_ffn([
        dict(layer=0, kind="dense", max_rel=1e-3, max_abs=2e-3, rms_abs=1e-4, argmax_match=1),
        dict(layer=3, kind="moe", max_rel=2e-3, max_abs=3e-3, rms_abs=2e-4, argmax_match=1),
    ])
    assert agg["dense_moe_transition_covered"] is True
    assert agg["worst_max_rel"] == 2e-3
    assert agg["argmax_proxy_match"] == "2/2"

    # config constants sanity (real dims)
    assert (HIDDEN, Q_LORA, KV_LORA, N_DENSE, VOCAB) == (6144, 2048, 512, 3, 154880)

    print(f"modal_partial_f1 self-check: PASS  "
          f"(modal {'present' if _HAS_MODAL else 'ABSENT (shim)'})")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
