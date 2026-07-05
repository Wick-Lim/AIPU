# Full-config elaboration study — glm_model_fp8 at the real 753B GLM-5.2-FP8 shape

**PRODUCT_ROADMAP P1.2.** Parametrize the committed RTL *slice* (MODEL_DIM=128, 6
layers, 8 experts, VOCAB=256) UP to the **real full GLM-5.2-FP8** config and check it
**ELABORATES cleanly** (no width-overflow / parametrization bug), catching full-scale
RTL issues the slice cannot. This is an **elaboration-only** study — no full-config
simulation (intractable, see §6).

Companion to [`P12_SCALEUP.md`](P12_SCALEUP.md) (task B4), which established the
structural contract with verilator/yosys at *reduced* MODEL_DIM/VOCAB (768/1024) and
N_EXPERT=16, Q_LORA=1536. **This study pushes to the TRUE full config**
(MODEL_DIM=6144, VOCAB=154880, N_EXPERT=256, Q_LORA=2048) and adds the independent
`iverilog -tnull` path.

## 1. HONESTY — what was and was NOT checked

- **Checked (this doc):** front-end **elaboration** = module/parameter resolution,
  `$clog2`/derived-width evaluation, port-width and part-select type/width checking,
  replication-count sign, and full-hierarchy connectivity — at the **real** dims.
- **NOT checked here:** behavior/functional correctness, timing, area, gate mapping.
  Elaboration ≠ synthesis ≠ simulation. Functional fidelity is proven **at the slice**
  by the committed `glm_model_fp8` TBs and, on **real checkpoint weights**, by
  [`REAL_CKPT_VALIDATION.md`](REAL_CKPT_VALIDATION.md); the FP8 datapaths are known
  un-synthesizable through yosys abc (no abc/full-synth attempted).

## 2. Parameter map (slice → full)

Source of truth: [`configs/full_glm52.vh`](../configs/full_glm52.vh) (every value cited to
`config.json` of `zai-org/GLM-5.2-FP8` / [`ACCEL_GLM52.md`](ACCEL_GLM52.md)).

| Module param | Slice | **Full** | Real-config source |
|---|---|---|---|
| MODEL_DIM | 128 | **6144** | `hidden_size` |
| L / N_DENSE | 6 / 3 | **78 / 3** | `num_hidden_layers` / `first_k_dense_replace` |
| VOCAB | 256 | **154880** | `vocab_size` |
| H_HEADS | 4 | **64** | `num_attention_heads` |
| NOPE / ROPE / V_DIM | 16/16/32 | **192 / 64 / 256** | `qk_nope`/`qk_rope`/`v_head_dim` |
| Q_LORA / KV_LORA | 64 / 32 | **2048 / 512** | `q_a_proj`/`kv_a_proj` safetensors (confirmed) |
| N_EXPERT / TOPK | 8 / 2 | **256 / 8** | `n_routed_experts` / `num_experts_per_tok` |
| INTER_MOE / INTER_DENSE | 64 / 256 | **2048 / 12288** ‡ | `moe_intermediate_size` / `intermediate_size` |
| TOPK_ATTN | 8 | **2048** | `index_topk` (DSA budget) |
| POSW | 20 | **20** | 2²⁰ = 1,048,576 ≥ 1M context |
| S_MAX | 8 | **8** † | latent-ring / KV scratch depth |
| PE_N / TN / LM_TN / BLK / PE_M | 4/4/4/128/1 | unchanged | microarch tiling (not model config) |

**† S_MAX = 8 (ASSUMPTION, FLAGGED).** `mla_attn_fp8` sizes its attention scratch
(`scores`/`probs`/`vstore`/union) by S_MAX; the real 1M context lives in the **POSW=20**
position field, *not* S_MAX. S_MAX is the KV latent-ring depth and is kept small for a
tractable elaboration. It sizes counters/scratch only; the datapath-width
parameterization under study is independent of it. (See §5 for a real consequence.)

**‡ INTER_DENSE = 12288 (found value, flagged).** The dense-front FFN (`intermediate_size`
for layers 0..N_DENSE-1), distinct from `moe_intermediate_size`=2048. Confirmed
INTER_DENSE ≥ INTER_MOE, so the `FF_GWD ≥ FF_GWM` config-validity constraint of
[`P12_SCALEUP.md`](P12_SCALEUP.md) §3 holds (no negative replication).

Wrapper (deliverable, does NOT touch `src/`): [`test/full_config_elab_wrap.v`](../test/full_config_elab_wrap.v)
— instantiates `glm_model_fp8` with all params overridden from `full_glm52.vh`, every
port except clk/rst left dangling (elaboration study, not a sim).

## 3. Methods

```
# path A — iverilog -tnull (type/width elaboration, then stop; no sim binary)
iverilog -g2012 -I src -I configs -tnull -pfileline=1 \
  test/full_config_elab_wrap.v  src/glm_model_fp8.v src/glm_decoder_block_fp8.v \
  src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v \
  src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
  src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v

# path B — verilator --lint-only (independent front-end elaboration)
verilator --lint-only -sv -I src -I configs --top-module full_config_elab_wrap <same files>

# path C — yosys hierarchy -check (elaborate + connectivity check; STOP before proc/opt)
yosys -p "read_verilog -sv -I src -I configs <same files>; hierarchy -top full_config_elab_wrap -check"
```

A throwaway command-line-overridable wrapper (`build/fce_sweep.v`, `ifndef`-guarded
dims) was used for the dimensional sweeps in §4/§5.

## 4. Results per tool

### Path B — verilator `--lint-only`: **PASS at the TRUE full config** ✅

The **true full config** (MODEL_DIM=6144, L=78, VOCAB=154880, H_HEADS=64,
NOPE=192/ROPE=64/V_DIM=256, Q_LORA=2048/KV_LORA=512, N_EXPERT=256/TOPK=8,
INTER_MOE=2048/INTER_DENSE=12288, TOPK_ATTN=2048) elaborated the **whole hierarchy**
(21 modules, full depth `…u_full.u_block.u_attn.u_dsa.u_topk`) with:

- **exit 0 — ZERO errors.** Walltime **24.2 s** (elab 5.7 s), peak **1.76 GB**.
- Warnings, all **non-error** and fully attributed (§5): 54 PINMISSING (dangling wrapper
  ports — expected), ~~4122 SELRANGE~~ + 7 WIDTHCONCAT (the S_MAX≪TOPK_ATTN choice + wide
  reset fills), 7 WIDTHEXPAND + 6 WIDTHTRUNC (config-independent style lints).

> **UPDATE (SELRANGE fixed, byte-identical):** the 4122 SELRANGE were subsequently cleared
> to **10** (99.76 %) — the swiglu/moe_router exponent-max-tree pad leaves now clamp their
> (discarded) out-of-range read index, and `mla_attn_fp8` uses `SWIN=min(S_MAX,TOPK)` (the
> tight bound; also shrinks the full-config scratch). Byte-identical: `glm_model_fp8_tb`
> `{4,31,20}`, swiglu 1024, moe_router 185, mla 7 — all unchanged. The residual 10 are the
> intricate mla union-slot indices (benign; verilator still exit 0). See the `fix(lint)` commit.

This is the authoritative full-config elaboration result: **the parameterization threads
cleanly at real 753B dims — no width overflow, no unresolved/negative-width parameter, no
$clog2 edge, no unknown module across the full hierarchy.**

### Path A — iverilog `-tnull`: PASS at every real dim; **hits a tool wall at MODEL_DIM=6144**

`iverilog -tnull` cleanly elaborated (exit 0, no warnings) at:

| config | time | verdict |
|---|---|---|
| SLICE (committed TB and via wrapper) | 0 s | clean |
| SLICE + each real dim **individually** to full: H_HEADS=64 / V_DIM=256 / N_EXPERT=256 / INTER_MOE=2048 / **VOCAB=154880** | ≤1 s each | clean |
| SLICE + INTER_DENSE=12288 | 6 s | clean |
| SLICE + MODEL_DIM = 512 / 1024 / 2048 / 3072 / 4096 | 0 / 2 / 3 / 11 / 12 s | clean |
| **Full real geometry** (true VOCAB/heads/experts/inter/topk_attn) + MODEL_DIM=256 / 512 | ~45 s | **clean** |
| Full real geometry + MODEL_DIM = 1024 / 2048 | > 45 s | did not finish |
| **TRUE full** (MODEL_DIM=6144) | **killed at 898 s (~15 min) CPU, 346 MB, no completion** | **tool wall** |

**Root cause of the wall (NOT an RTL bug):** iverilog's front-end (`ivl`) unrolls
constant-bound behavioral `for` loops and materializes wide part-selects. `glm_model_fp8`
and children have many `for (i=0;i<MODEL_DIM;…)` loops over MODEL_DIM-wide vectors, so
elaboration cost grows ~O(MODEL_DIM²) and multiplies with the other large dims
(INTER_DENSE=12288, HV=H_HEADS·V_DIM=16384, N_EXPERT=256). VOCAB=154880 alone is *fine*
(iverilog stores the `lbuf`/`logits` arrays compactly — 0 s). The wall is **MODEL_DIM=6144
combined with the other full dims**, a front-end scaling limit of iverilog, not a
parametrization defect: every real dim elaborates clean individually and the entire real
geometry elaborates clean with MODEL_DIM reduced to 256–512. verilator (path B) — which
does not unroll for lint — elaborates the same true-full config in 24 s.

### Path C — yosys `hierarchy -check`: blocked by a known yosys-0.66 quirk (tool, not RTL)

`hierarchy` fails immediately (exit 1, 0 s) with:

```
src/glm_decoder_block_fp8.v:486: ERROR: Static cast is only supported in SystemVerilog mode.
```

Line 486 is `if ((esc < ECW'(N_EXPERT)) && …)` — a **valid SystemVerilog** static cast.
`read_verilog -sv` reads the file fine; the error only appears when `hierarchy` **re-derives**
the module — yosys 0.66 loses the `-sv` flag on derived-module re-elaboration. It reproduces
**at SLICE defaults with no parameter override**, so it is independent of the full-config
values — a documented tool limitation ([`P12_SCALEUP.md`](P12_SCALEUP.md) §2), **not** an RTL
bug. yosys hierarchy is therefore not a usable elaboration path for this hierarchy; verilator
(B) is the authoritative check.

## 5. The one real finding — S_MAX ≪ TOPK_ATTN width-lint family (benign, tied to B7)

The DSA union scratch in `mla_attn_fp8` is sized by `SWIN = TOPK` (the module's TOPK is
`TOPK_ATTN`, wired at `glm_decoder_block_fp8.v:304` `.TOPK(TOPK_ATTN)`), so at full config
`SWIN = 2048` → `SWINW = $clog2(2048) = 11`. But the per-row union index `ksel` is
`reg [IDXW:0]` with `IDXW = $clog2(S_MAX=8) = 3` → **4 bits**. Expressions like
`ksel[SWINW-1:0]` (mla_attn_fp8.v:1306/1369/1440, …) thus slice **11 bits from a 4-bit
reg**, which verilator reports as `SELRANGE` (+ the `{K*SCORE_W{…}}`=65536 reset fills as
`WIDTHCONCAT`).

**Attribution (measured):** the dangling SLICE wrapper (TOPK_ATTN=8) yields **0 SELRANGE**;
flipping only `TOPK_ATTN 8→2048` (S_MAX still 8) adds **+26 SELRANGE + 2 WIDTHCONCAT**. So
these warnings are **not** dangling-port artifacts — they are caused **solely by the
S_MAX(8) ≪ TOPK_ATTN(2048) decoupling** (at true full config the count grows to 4122 as
MODEL_DIM/heads/experts multiply the flagged index expressions).

**Is it a bug?** No — it does **not** break elaboration (verilator exits 0) and is
**functionally benign**: only `u_cnt ≤ S_MAX` distinct keys are ever cached, so `ksel`
holds values 0..S_MAX-1 (fits in 4 bits) and the extra slice bits read as 0. But it **is** a
genuine, documentable width inconsistency, surfaced *only* because we set S_MAX=8 while the
real DSA budget TOPK_ATTN=2048 sizes SWIN. It is a direct manifestation of the flagged
**B7 caveat** (decouple the attention window from S_MAX). A real full-config integration
would resolve it by raising S_MAX to the DSA window or making SWIN independent of TOPK_ATTN
(B7); as a small cleanup, `ksel` / the union-slot registers could be sized by
`min(S_MAX, TOPK_ATTN)` so the slice widths agree.

The remaining warnings are config-independent (present identically at slice):
`WIDTHTRUNC`×6 (the `VSTORE_RAM` integer mode-flag used as a 1-bit condition) and
`WIDTHEXPAND`×7 (operand auto-extension in memory-address arithmetic). Neither is a
full-scale defect.

## 6. Verdict — per module

verilator elaborated the **entire hierarchy at the true full config with zero errors**, so
every module passes structurally; iverilog independently passes each at every real dim
(MODEL_DIM tractable). No module ERRORs at full config.

| Module | Full-config elaboration | Notes |
|---|---|---|
| `glm_model_fp8` (top) | **PASS** | true VOCAB=154880 LM-head/argmax/logits threads; wide reset fills only |
| `glm_decoder_block_fp8` | **PASS** | dense/MoE mode mux clean; INTER_DENSE≥INTER_MOE holds (no neg. replication) |
| `mla_attn_fp8` | **PASS** (with §5 lints) | true 64-head / NOPE192 / ROPE64 / V256 / KVL512 geometry threads; carries the benign S_MAX≪TOPK_ATTN width-lints |
| `moe_router_fp8` | **PASS** | 256-expert router + topk_select clean |
| `swiglu_expert_fp8` | **PASS** | INTER_MOE=2048 / INTER_DENSE=12288 gate/up/down clean |
| `glm_matmul_fp8` | **PASS** | ragged [128,128] block-scale K threads at real dims (also `glm_matmul_fp8_tb` TEST 6) |

**Overall: the full 753B GLM-5.2-FP8 config parametrization ELABORATES CLEANLY.** No
width-overflow, no unresolved/negative-width parameter, no $clog2 edge, no unknown module
— verified end-to-end at the *true* real dims (MODEL_DIM=6144, VOCAB=154880, N_EXPERT=256,
Q_LORA=2048, …) by verilator, and per-dimension by iverilog. Two **tool** limitations, not
RTL bugs, were characterized (iverilog's MODEL_DIM=6144 unroll wall; yosys-0.66 `-sv`-loss
on derived modules). The single genuine RTL observation is the **benign S_MAX≪TOPK_ATTN
width-lint family** in `mla_attn_fp8`, a documented consequence of the flagged S_MAX choice
(B7) — functionally harmless, worth a small width-clamp cleanup.

## 7. What is explicitly OUT of scope

- **Full-config functional simulation** — intractable: the LM-head GEMV alone streams
  MODEL_DIM(6144) × VOCAB(154880) ≈ 2.4×10⁸ K-beats **per token**; a 256-expert MoE layer is
  billions of cycles/token. P1.2 is a **structural/elaboration** contract, not "set params +
  run the TB." (Slice functional fidelity: committed TBs; real-weight fidelity:
  `REAL_CKPT_VALIDATION.md`.)
- **1M-context attention scratch (S_MAX)** — kept small; decoupling the window from S_MAX is
  task **B7**.
- **Synthesis / gate mapping / area / timing** — not attempted (FP8 datapaths are
  un-synthesizable through yosys abc by design).
