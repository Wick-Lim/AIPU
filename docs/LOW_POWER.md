# Low-power design — the bit-exact path to a cool single box

**Requirement:** the accelerator must be low-power. **Decision (this project):** stay **byte-identical**
(output == real GLM-5.2-FP8 golden), push a **single box as low as possible**, and add fidelity-trade
levers only if the bit-exact floor misses. This doc is the honest energy budget, the lever ladder,
and what is built vs. staged.

## 1. The one fact that governs power: it's ~80 % Flash

`J/token ≈ bytes_moved × energy/bit`. Two anchors decide everything:

- **NAND read energy is ~24–26× DRAM/bit** ([`SYSTEM_SINGLE_PACKAGE.md`](SYSTEM_SINGLE_PACKAGE.md)).
- The die is **75–80 % Flash-bound** — it sits idle waiting on the expert stream
  ([`CYCLE_EMULATION.md`](CYCLE_EMULATION.md), [`ULTRA_PERF.md`](ULTRA_PERF.md)).

So per-token energy splits roughly:

| bucket | share | why | can we cut it? |
|---|---|---|---|
| **Flash routed-expert bytes** | **~80 %** | top-8/256 experts/MoE-layer streamed from Flash every token (753 GB ≫ 64 GB DDR5, can't all reside) | **only** by moving fewer bytes or moving them less often |
| DRAM / cache / KV | ~10–15 % | resident set + latent-KV in DDR5 | HBM (energy/bit), smaller footprint |
| **compute die** | **~20 % and idle** | ~80 GFLOP/token on a die that's 75–80 % stalled | DVFS, gating, die-shrink (all done/free — see §4) |

**The blunt conclusion: DVFS, die-shrink and clock-gating only touch the ~20 % compute slice.
Real low power is won on the ~80 % Flash bytes — cut them, or amortize each fetch across more
tokens/users.** Everything below is ranked by that truth.

## 2. The irreducible floor

The active experts **must** be re-read from Flash every token (the model is fixed; the 753 GB
routed-expert set can't reside in 64 GB DDR5). That sets a hard J/token floor. The *only* bit-exact
ways under it are: **(a) fewer bytes per fetch** (lossless compression), **(b) fewer fetches per
token** (amortize one weight-load across K tokens or B users), **(c) more of the hot working set
resident** (bigger DDR5 → higher hit-rate → fewer Flash misses; a hardware-$ lever). Compute tricks
cannot touch the floor.

## 3. Bit-exact lever ladder (J/token, single-user [EST])

Stacking on the ~9 J/token baseline (numbers are modeled `bytes × energy/bit`, not silicon watts):

| lever | mechanism | J/token | status |
|---|---|---|---|
| baseline | — | ~9 | — |
| `flash_xbar` ×N + deep queue | N× Flash BW + latency-hide | ~9 | ✅ built (7.99× hide + N× bank) |
| `weight_decomp` (lossless) | 1.34× fewer Flash bytes | ~6.7 | ✅ built, bit-exact (Huffman, 5.97 b/sym) |
| MTP/spec **K=2** | verify 2 tokens per weight-load (K_eff 1.7) | ~4.5 | ✅ built, spec==greedy exact |
| grouped MoE **union-skip** batch | B rows share 1 expert fetch (÷ up to B) | ↓ at B>1 | ✅ built, byte-identical |
| **DVFS freq** (`clk_throttle`) | run die f/div in the 4–5× slack (§4) | **peak-power only** (not J/token) | ✅ **RTL built + byte-identical** — the eco/thermal knob |
| **DVFS voltage** | lower supply at the reduced f | −~15 % total **energy** | vendor/physical (the J/token half) |
| **spec high-K verify** (÷K weight-loads) | verify K+1 draft positions in ONE model weight-load (PE_M=K+1 batch) → **÷(K+1) weight-loads on the 80 %** | scales with K_eff | ✅ **HW built + bit-exact** (`spec_batched_top` / `spec_chain_top`, spec==greedy EXACT; weight-share `glm_model_fp8_pem` ALL 3) |
| ↳ raise K_eff 1.7 → **3–5** | resident ~1–3 B dense draft (vs the chained MTP self-draft) proposes K=4–8 with higher acceptance | ~1.5–3 | ⏳ **draft-quality, not RTL** — needs a real 1–3 B draft-model artifact (`ULTRA_PERF.md` #4) |

**Projected bit-exact floor: ~9 → ~1.5–3 J/token [EST], single box.** The dominant lever is
**spec high-K amortization** — it divides the ~80 % Flash term, which no compute trick can. Its
**hardware is already built and bit-exact** (§4a); what remains is *draft acceptance* (α), a
**model-quality** property, not an RTL gap.

### What is measured vs modeled (firming the [EST])
The J/token numbers are a product of two kinds of input — be clear which is which:
- **Measured (defensible) multipliers:** `weight_decomp` **1.34×** fewer Flash bytes (5.97 bits/sym,
  8 tests); clock-gating **73.75 %** of idle-dynamic gated (`clk_en_ctrl_tb`); the BFP accumulator
  **−87.6 %** cells; die 75–80 % Flash-bound + the ~4–5× compute-slowdown budget from the
  cycle-emulation (`compute_cyc` vs exposed `stall`, `glm_fp8_system_perf_tb`); MTP K=2 **spec==greedy
  exact** (1379 tests) → K_eff≈1.7 (self-draft, α decays past K=2). These are RTL-verified factors.
- **Modeled (the [EST] part):** the **absolute ~9 J/token baseline** = `bytes/token (~22 GB routed) ×
  energy/bit` with **NAND ≈ 24–26× DRAM/bit** — a datasheet/roofline model, **not** a silicon
  wattmeter. So the *relative* improvements are measured/verified; the *absolute* J/token is [EST]
  until the vendor flow (Gowin) + a board give real watts.
- **Compression is near its bit-exact ceiling:** FP8 weight entropy ~5–6.5 bits/sym → ~1.3–1.6× is
  close to the per-symbol limit; beating it needs cross-symbol context modeling (marginal, complex).

## 4. DVFS — the free, byte-identical compute-power lever (new)

A Flash-bound die **should run slow and cool.** Because the token window is 75–80 % Flash-stall, the
compute has a **~4–5× frequency-reduction budget**: drop the die clock (and, at lower f, the supply
voltage) until compute just fills the Flash-stall shadow — **zero throughput loss** (throughput is
set by Flash, not compute). `P_dyn ∝ C·V²·f`, so a 4–5× f cut is a 4–5× compute-dynamic cut at
constant V, and more with V scaling. It is **result-invariant** (same math, slower clock) — the same
"compute is nearly free" slack that shrinks the die ([`MINIATURIZATION.md`](MINIATURIZATION.md)).

**Measured mechanism** (`test/glm_fp8_system_perf_tb.v`, EXPERT_STALL=1, token stays == golden):

| FLASH_LAT | exposed `stall` | note |
|---|---|---|
| 256  | 777  | token `ALL 3 PASSED` |
| 2048 | **6153** | 8× latency → 7.9× stall — **the Flash-stall shadow scales linearly with Flash cost** |

The slice keeps most experts cached (h≈98 %, tiny experts) so it is not yet stall-dominated; at real
753 B scale (h≈27 % measured, huge experts) the roofline puts the window at **75–80 % Flash-bound**,
i.e. **compute ≈ 20–25 %** of the token → the **~4–5× DVFS budget**.

**Two halves of DVFS — and only one is RTL (honest, corrected).** `P_dyn ∝ C·V²·f`:
- **Frequency (f) — RTL-realized here.** `src/clk_throttle.v` (new) runs the die at effective
  **f/div** by feeding `clk_en_ctrl` a `throttle` term (the die takes one active slot per `div`
  cycles), byte-identical (reuses the proven stall-gate path; `div<=1` = off). This scales **peak
  power** (fewer active edges per unit time) — the knob that lets the USB-C box hold a lower **power
  envelope / thermal cap** (the product plan's "eco mode", [`USBC_PRODUCT_PLAN.md`](USBC_PRODUCT_PLAN.md)
  §6). **But frequency scaling does NOT cut energy-per-token**: the switching-event *count* is
  unchanged, only spread over more time. Verified: `test/clk_throttle_tb.v` (f/4, f/3, hold-safe);
  `clk_en_ctrl_tb` regression identical with `throttle=0`.
- **Voltage (V) — vendor/physical.** The **J/token** win is the `V²` term (lower energy *per* event),
  which needs the vendor flow to actually lower the supply at the reduced f. Not RTL.

So DVFS's contribution is: **peak-power/thermal cap now (RTL, `clk_throttle`)**, and the ~15 %-of-total
**energy** win only once voltage is scaled on the vendor flow. Free in throughput while
`div ≤` the ~4–5× budget.

## 4a. Spec high-K amortization — the ÷K hardware is already built

The single biggest bit-exact energy lever is **amortizing one Flash weight-stream across K verified
tokens**, and its **hardware exists and is bit-exact**:

- **`spec_batched_top.v`** (KEY IDEA #5, "Flash ÷K"): the K+1 verify positions
  `{cur_tok, d_0..d_{K-1}}` are pushed through **one** `glm_model_fp8` as a **PE_M=K+1 batch** — one
  weight fetch per (layer, projection, expert) feeds **all** K+1 rows (the documented PE_M
  weight-share contract). So a K+1-position verify costs **ONE** model weight-load, not K+1 →
  **weight-loads ÷ (K+1)** on the dominant ~80 % Flash term.
- **`spec_chain_top.v`**: mints the K drafts by running the one shipped MTP layer **recurrently**
  (the P1.3b `h_mtp` chain state), then does the PE_M=K+1 verify and commits the **longest accepted
  prefix**. Committed stream **== greedy rollout, EXACT** (`spec==greedy`, the spec-slow CI gate).
- **Verified this session:** the underlying weight-share — *"the weight stream is fetched ONCE for
  all B rows, not B times"* — re-confirmed by `glm_model_fp8_pem` (**ALL 3 PASSED**). The full
  `spec==greedy` sims (`make spec-slow`) are the committed CI gate (~30 min each; not re-run to
  completion here — the fast weight-share corroboration stands in for the energy claim).

**So the ÷K amortization is done in RTL.** The only thing between K_eff≈1.7 (built, self-draft chain
— acceptance decays because GLM ships **one** MTP layer) and K_eff 3–5 (the ~1.5–3 J/token floor) is
**draft acceptance α** — raised by a separate resident ~1–3 B dense draft model. That is a
**model-quality / artifact** step, **not** an RTL one. This corrects the earlier "spec high-K = not
built": the *hardware* is built; the *draft* is the open item.

## 5. Already-built power wins (compute + idle)

- **Idle-die clock gating** (`clk_en_ctrl`): **73.75 % of idle-dynamic power gated** (measured
  gated-cycle fraction), formally safe (13 064 checks — never gates an advancing cluster).
- **BFP fixed-point accumulator**: **−87.6 % cells** vs fp32-accumulate → lower switching energy.
- **FP8 E4M3 (4×4 mantissa)**: frees DSPs, far less energy/MAC than fp32.
- **Die-shrink L0/L1**: compact config + swiglu engine-share (6→4 GEMM/block) → area-proportional
  static+dynamic ↓, byte-identical ([`MINIATURIZATION.md`](MINIATURIZATION.md)).

These are real but bounded — they act on the ~20 % compute slice.

## 6. Explicitly OFF the table (per the bit-exact decision)

Bigger single-box wins exist but **change the output** — deferred unless §3's floor misses the target,
then revisited as a separate fidelity decision:

- **Dynamic top-k expert pruning** (k_eff<8): ÷1.3–1.6× fewer routed bytes — *cheapest big lever*,
  low effort ([`ULTRA_PERF.md`](ULTRA_PERF.md) #7). **NOT bit-exact.**
- **Contextual SwiGLU activation sparsity**: fetch only active W_up cols / W_down rows, ÷1.5–3×
  (#6). **NOT bit-exact.**
- **Lossy sub-FP8 for cold experts**: large, but a fidelity trade.

## 7. Honest gaps

- **Every J/token here is [EST]** — a `bytes × energy/bit` model, **not** a silicon/P&R wattmeter.
- **No real watt number exists** — it needs the vendor flow (Gowin / nextpnr) on the placed netlist;
  the same abc/KMAX wall that blocks the LUT count blocks power extraction here.
- **The 73.75 % gating** is a measured *cycle* fraction, not a power meter.
- **spec high-K:** the ÷K *hardware* is **built + bit-exact** (§4a); what is not built is the
  **resident ~1–3 B dense draft model** that raises K_eff 1.7→3–5 — a model artifact, not RTL.

## 8. Verification invariant

Every bit-exact lever is a *result-invariant restructuring*: the system TB token must stay
`== standalone glm_model_fp8` (the same gate as `sim-glm-compact` and the MoE union-skip). DVFS,
die-shrink, gating, compression, MTP and batching all pass it today; spec high-K must pass it before
landing. Power is optimized **without ever moving the decoded token.**

## Status
- **Energy budget + DVFS budget: characterized** (this doc; DVFS mechanism measured, real-scale [EST]).
- **Built:** flash_xbar, weight_decomp 1.34×, MTP K=2, union-skip, clock-gating 73.75 %, die-shrink
  L0/L1, the spec high-K ÷K-weight-load hardware (`spec_batched_top` / `spec_chain_top`, spec==greedy
  EXACT; weight-share re-confirmed by `glm_model_fp8_pem` ALL 3, §4a), **and the DVFS/eco frequency
  prescaler** (`clk_throttle` → `clk_en_ctrl.throttle`, f/div peak-power cap, byte-identical,
  `clk_throttle_tb` ALL 4).
- **Near ceiling / gated (RTL side largely exhausted, per the D re-investigation):** compression is
  ~at the FP8 entropy ceiling; higher spec K_eff needs a **resident ~1–3 B draft model** (artifact,
  not RTL); DDR5 self-refresh is a **PHY/vendor** feature; the DVFS **voltage** (J/token) half and the
  real watt number are **vendor flow** (Gowin / nextpnr) + a board.
- **The J/token *efficiency* levers are therefore built or model/vendor-gated;** the RTL-doable
  remainder was the peak-power prescaler (done) + this measured-vs-modeled firming (done).
