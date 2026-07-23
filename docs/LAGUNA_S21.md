# Porting AIPU to Laguna-S-2.1 (UD-Q4_K_XL)

Target: **`unsloth/Laguna-S-2.1-GGUF : UD-Q4_K_XL`** — a 118B MoE (~8B active/token)
in the same GGUF k-quant ecosystem as the GLM-5.2 build that `main` targets.

This branch (`laguna-s-2.1`) adds a second model target **without disturbing the
verified GLM-5.2 datapath on `main`**. The rule stays: bit-exact or it doesn't ship —
every new block gets a golden gate + a must-fail injection pair, same discipline as
the GLM path.

> **Provenance of the numbers below.** Rows marked *(card)* are from the HF model
> card. Rows marked **[VERIFY]** must be read from the model's `config.json` /
> the GGUF metadata before any RTL is sized — do NOT hardcode a guessed value.

---

## 1. What the model is (card)

| Property | Laguna-S-2.1 | GLM-5.2 (`main`) |
|---|---|---|
| Total / active params | 118B / ~8B per token | 753B / ~40B |
| MoE | 256 routed, **top-10**, + 1 shared | 256 routed, top-8, + 1 shared |
| Attention | **GQA** — 8 KV heads, head_dim 128, **per-head softplus output gating** | MLA (latent) + DSA (sparse top-2048 indexer) |
| Layer layout | **12 global + 36 sliding-window (window 512)** = 48 layers | uniform MLA+DSA per layer, 3 dense-front FFN |
| Context | 1,048,576 | 1,048,576 |
| Arch id | `laguna` | `GlmMoeDsaForCausalLM` |
| Quant | UD-Q4_K_XL (Q4_K + Q6_K + Q8_0 + F16 mix) | UD-Q4_K_XL (same mix) |

**Consequence of "smaller":** at Q4 the whole checkpoint is **~70 GB** (vs 467 GB),
and only ~8B params stream per token (vs ~40B). On the same rung-③ 512 GB / ~1.1 TB/s
box this runs **several× faster than GLM-5.2 and fits with large headroom** [EST] —
the roofline denominator shrinks with the active-param footprint.

---

## 2. What carries over unchanged (the foundation — already verified on `main`)

These are format-level or fully parameterized, so they inherit directly:

- **Q4_K / Q6_K / Q8_0 dequant** — bit-exact, model-agnostic (the moat). This is
  exactly why "이거도 Q4_K_XL로" works: the dequant contract is inherited, not rebuilt.
  → re-run `tools/gguf_crosscheck.py` against the Laguna GGUF to re-seal it on the
  real bytes (Phase 1).
- **Q4_K GEMM core** (`glm_matmul_q4k`), **RMSNorm**, **softmax** — dimension-parameterized.
- **MoE router + top-k selector** (`moe_router_q4k`, `topk_select`) + **expert path**
  (`swiglu_expert_q4k`) — `TOPK` and `N_EXPERT` are parameters; **top-10 is a config
  change** (`TOPK=8 → 10`), 1 shared expert already modeled.
- **Memory system** — `ddr5_xbar`, `weight_loader_q4k`, `expert_cache_pf`,
  `kv_cache_pager`, boot loader, CDC. Medium-agnostic; re-sized by config.
- **The entire verification harness** — bit-exact gates, injection pairing, the
  68→79-gate manifest, PHY-closure loopback, cycle-stall emulation. New Laguna
  blocks plug into the same `make` gate discipline.

---

## 3. What is genuinely new (attention is a different machine)

The current attention orchestrator is MLA+DSA-specific. Laguna is standard GQA with
two extra elements the repo does **not** have today (confirmed by grep on `src/`):

1. **GQA attention orchestrator** — 8 KV heads, head_dim 128, grouped-query. No latent
   c_kv/k_rope compression, no DSA indexer. Reuses the inner QK^T / softmax / V blocks;
   the orchestrator around them is new. (Same core add as the GLM-4.7-Air discussion.)
2. **Sliding-window attention (window 512)** — NOT implemented today (`src/` "window"
   hits are all clock-gating, unrelated). Needs windowed causal masking in the KV gather
   **plus a per-layer selector** so 12 layers run global attention and 36 run SWA.
3. **Per-head softplus output gating** — NOT present today. A new datapath element on
   the attention output: `out = attn_out * softplus(gate)` per head. softplus is close
   to but not identical to the SiLU/sigmoid already in `glm_act.v` — verify/add the exact
   form against the reference.

Config also needs: 8 KV heads @ dim 128, 48-layer global/SWA layout, TOPK=10, the 118B
shape, and RoPE parameters.

---

## 4. Open items to resolve from config.json / GGUF metadata before sizing RTL  **[VERIFY]**

- Q attention head count + GQA group size (only **KV heads = 8** is on the card).
- hidden_dim, intermediate_dim (dense vs MoE), vocab_size, RMSNorm eps.
- Are there dense-front FFN layers (GLM has 3), or is every layer MoE?
- RoPE: theta, full vs partial rope, interleave.
- **Speculative decoding:** does Laguna ship an MTP / self-draft head? If not, the
  `glm_q4k_spec_system` composition and the measured A_eff/accept-rate inputs do **not**
  transfer — throughput would drop to the no-spec roofline until a drafter is chosen.
- Exact softplus output-gating formula + the global-vs-SWA per-layer schedule.
- Global attention layers: plain full-causal, or also gated/other?

---

## 5. Phased plan (each phase is bit-exact-gated before the next)

1. **Phase 1 — inherit + config.** Branch (done), read config.json/GGUF metadata into a
   Laguna config set, and **re-seal the dequant on the real Laguna GGUF bytes**
   (`tools/gguf_crosscheck.py`). Early, cheap, high-value proof the moat carries.
2. **Phase 2 — MoE + FFN at Laguna shape.** TOPK=10, 256+1 experts, the real dims;
   golden-check the router/expert path (reuses `main`'s gates, re-parameterized).
3. **Phase 3 — GQA orchestrator.** New attention orchestrator over the existing inner
   attention blocks; bit-exact vs a numpy/ggml GQA reference + injection pair.
4. **Phase 4 — sliding-window + global/SWA layout.** Windowed masking + per-layer selector;
   golden vs reference, injection pair (a wrong window must FAIL).
5. **Phase 5 — softplus output gating.** New datapath; bit-exact vs reference, injection pair.
6. **Phase 6 — assemble + spec.** Full forward bit-exact golden (embed → 48×(GQA/SWA+MoE)
   → norm → LM head → argmax); resolve the MTP question and compose spec-decode if a
   drafter exists.
7. **Phase 7 — gate + manifest.** Fold the Laguna gates into a `laguna-release-gate`,
   pin exact counts, add loopback/PHY-closure for the new families.

Foundation (dequant + MoE + memory + verification infra) carries ~70–80% of the work;
the new surface is the attention machine (Phases 3–5). Comparable to, and a bit more than,
the plain-GQA GLM-4.7-Air port because of SWA + output gating.
