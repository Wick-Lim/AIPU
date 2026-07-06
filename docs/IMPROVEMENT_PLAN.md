# GLM-5.2-FP8 accelerator — performance / power improvement plan

The governing model (from `docs/SYSTEM_SINGLE_PACKAGE.md` §7):

```
  tokens/s  ≈   NVMe_BW   /  [ (1 − h) × footprint ]   ×   K
  J/token   ≈   bytes_moved × energy/bit   (NVMe read bytes dominate, ~24–26× DRAM/bit [EST])
```

So every lever is one of: **raise NVMe_BW**, **raise hit-rate h**, **shrink footprint**, **raise K
(speculative)**, or **cut bytes_moved** (which helps both tok/s and J/token). The compute die is
*not* the bottleneck (~20–25 % utilized) — die-side optimization (the −87.6 % accumulator, fmax,
formal) improved cost/thermals/correctness but does **not** move tok/s or J/token until the NVMe/storage
tier is unblocked. This plan targets the real bottleneck.

**The memory-stall mechanism this plan rests on is now measured on real RTL cycles.**
`glm_fp8_system` gained an `EXPERT_STALL` parameter (default 0 = byte-identical to the committed
system) that clock-gates the compute die for exactly the cycles a demand-miss is being serviced by
the NVMe/storage backend, so `cyc_per_tok` **grows with `FLASH_LAT`** as a direct measurement (flat 7947 → 8607 @
`FLASH_LAT=256`) while the token stays bit-exact — matching the exposed demand-stall
(`stall = 3·FLASH_LAT + 9`, slope = miss count). This upgrades the roofline's stall term from
*assumed* to *counted*; see [`CYCLE_EMULATION.md`](CYCLE_EMULATION.md).

Baseline (measured h, [EST] BW): NVMe ~50 GB/s [EST] (aggregate across many PCIe lanes/drives — a single
Gen4 x4 NVMe ~7 GB/s), h=27 %, K=1 → **~3 tok/s single-user**, ~8–10 J/token.

Legend: 🟢 RTL-doable in this repo · 🟡 system/architecture (design + vendor IP) · 🔴 out of RTL scope.

---

## P1 — NVMe/storage bandwidth (the linear lever; biggest single win)

| # | Item | Type | Plan | Est. impact |
|---|---|---|---|---|
| 1.1 | **`flash_xbar`** — N-channel banked storage-read fabric | 🟢 | Same pattern as `ddr5_xbar`: stripe expert fetches across N back-end channels → ~N× aggregate NVMe_BW. `flash_xbar` is a committed RTL identifier: it is the medium-agnostic storage-read fabric (address → weight bytes); in the product its NAND back-end is a labeled placeholder swapped for an **NVMe/PCIe host controller**, so the N channels map to **PCIe lanes / multiple NVMe drives**, not NAND dies. A 1–4 TB NVMe store reaches "10s of GB/s" [EST] by striping reads across lanes/drives (one Gen4 x4 NVMe ~7 GB/s; scale with lanes/drives). Build + BMC-verify like ddr5_xbar. **✅ DONE: deep per-channel outstanding queue (QDEPTH) hides storage read latency 7.99× (Little's law) + N_CH banking stacks (~57× combined @8ch×Q8); 2049 tests + BMC K=12.** | **~N× tok/s** (linear). 4ch ≈ 3→12 tok/s |
| 1.2 | **NVMe expert layout** — co-activated experts on different channels | 🟢 | Offline placement so a token's top-8 experts spread across channels/drives (avoid channel hotspots), mirroring the DDR5 stripe. **✅ DONE: `tools/flash_layout.py` (balanced greedy least-conflict packer). Measured N_CH=8: optimized 55% of 8× peak BW vs naive round-robin 39% (~+40%) — kills the 4/5/6-on-one-channel collision tail (99.5% of fetches ≤2/ch). Honest: 8× unreachable (top-8 ≥ 8ch pigeonhole + popularity skew); the win is removing hotspots, not reaching peak. 8ch is the top-8 sweet spot. The pigeonhole cap and the sub-expert-striping (RAID-0) option that removes it are consolidated in [`FLASH_STRIPING.md`](FLASH_STRIPING.md).** | **~+40 %** of flash_xbar's realizable BW (sustains 1.1) |
| 1.3 | **Deeper NVMe read pipeline** — more outstanding requests | 🟢 | Raise the fetch queue depth so the NVMe/storage tier stays saturated despite per-read latency (BW = outstanding / latency). Wire into `expert_cache_pf` + `flash_xbar`. | recovers the latency-bound gap to peak BW |
| 1.4 | Faster NVMe medium (PCIe5 NVMe / more drives / more lanes) | 🟡 | Vendor/board choice; the NVMe host controller / PCIe root is vendor IP. The bandwidth a board can actually feed is **rung-dependent** — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md): the near-term prove-it rung is a low-end FPGA dev board + DDR4 + one NVMe; the funded product is a custom board (mid FPGA + DDR5-multi-channel **or** HBM + multi-NVMe over more PCIe lanes) — not the bring-up dev board. DDR5 is the **rung-2** memory, not a hard-asserted single spec. Document the BW target the RTL fabric must feed. | linear, but $ + board |

## P2 — Cut bytes moved (raises tok/s AND lowers J/token — the dual win)

| # | Item | Type | Plan | Est. impact |
|---|---|---|---|---|
| 2.1 | **Expert decompressor** in the NVMe→DDR5 path | 🟢 | FP8 expert weights have entropy < 8 bits; store them losslessly-compressed on the NVMe SSD and decompress on-chip during fetch (e.g. a lightweight zero-RLE / dictionary / Huffman decoder). Cuts NVMe bytes/expert ~1.3–1.6× → effective NVMe_BW ↑ and NVMe read energy ↓ by the same factor. Build a `weight_decomp` unit + verify decompressed == original FP8. **✅ DONE: canonical-Huffman `weight_decomp`, bit-exact lossless, measured 1.34× on representative FP8 weights (5.97 bits/sym); 8 tests.** | **~1.34×** measured tok/s + NVMe read energy |
| 2.2 | **MTP K>1 / better draft** — verify more tokens per weight-load | 🟢 | Extend `spec_decode_seq` to a K>1 multi-token draft (longest-accepted-prefix). **✅ DONE + MEASURED: DRAFT_K>1 support (g_kn batch verifier), spec==greedy EXACT for K=1/2/3 (1379 tests), backward-compat (621 + formal intact), K=1 byte-identical. Eff tok/pass (chained-decay, the shipped 1-MTP-layer reality) at α=0.7: K=1→1.69, K=2→2.08, K=3→2.17. HONEST: the shipped model has 1 MTP layer so K>1 must chain the head autoregressively (acceptance decays) → K=2 is the sweet spot; K=3+ is marginal unless a deeper MTP stack lands. Realized in RTL by `spec_chain_top` (`make spec-slow`): the MTP head runs recurrently on chain hidden-state `h_mtp` to mint K drafts, then a PE_M=K+1 batched-verify in one weight-load commits the accepted prefix (spec==greedy; self-draft K_eff ~1.7–2.2).** | **K=2 ≈ +23 %** over K=1 (chained) |
| 2.3 | **Higher cache hit-rate h** — bigger cache + predictor-driven prefetch | 🟢 | Wire `expert_predictor` into `expert_cache_pf` prefetch. **❌ MEASURED NO-OP at GLM cache size (`expert_prefetch_top`):** at SLOTS=900 (>reuse distance) the predictor hints the most-popular expert, which LRU **already keeps resident** → 0 prefetches issued, hit-rate byte-identical to baseline; deep LOOKAHEAD 1/2/3 identical. Only helps in the narrow regime just *below* the reuse-distance knee (SLOTS=550: 0%→4.6%). **Conclusion: hit-rate is capped by fine-grained-routing entropy, not prefetch cleverness — this lever does not move the real config.** | **~0 %** at real cache size (honest) |
| 2.4 | **Union-skip grouped MoE** (batch axis) — fetch only the union of selected experts | 🟢 | The PE_M>1 grouped MoE in `glm_decoder_block_fp8` scans the expert axis (`T_ESCAN`) and fetches **only** the union of experts any of the B rows selected (combinational `any_has` membership), not all N_EXPERT. **✅ DONE: BYTE-IDENTICAL — `glm_decoder_block_fp8_union_tb` ALL 4 (*"PE_M=2 evaluated 3 experts, skipped 5 of 8, bit-exact"*), `glm_model_fp8_pem` ALL 3, decoder TB ALL 9; reference pattern `batched_moe.v`, now proven at full B-coverage (`make bcov`: B∈{1,2,3,5,8} × routing {same,distinct,random,overlap}, batched(PE_M=B) == B per-row runs BIT-EXACT, union fetched once).** The PE_M batch-widen enabler is **DONE 4/4** (swiglu/router/mla/mtp). Realizes ULTRA_PERF #1's aggregate footprint reduction **in the model**. On the single-user box the batch axis is exercised only by speculative decode's small `PE_M=K+1` self-verify batch (`spec_chain_top`, B=1 per user); the large-batch B≈256 case is the **non-target datacenter-serving** regime (kept as analysis, not this product). | up to **~32×** fewer NVMe expert fetches at small batch (the single-user spec-decode verify batch); ~0 at B≈256 (union≈all — the non-target datacenter-batch regime) |

## P3 — Hide latency / raise utilization

| # | Item | Type | Plan | Est. impact |
|---|---|---|---|---|
| 3.1 | **Predictor-driven deep prefetch loop** | 🟢 | Use `expert_predictor` confidence to prefetch ahead of the demand cache (L+2..) so the NVMe/storage tier stays saturated and the die stalls less; double/triple-buffer experts. | sustains P1/P2 (keeps NVMe busy) |
| 3.2 | **Idle-die clock gating** (`clk_en_ctrl`) | 🟢 | The die idles ~75 % waiting on NVMe/storage; gate the compute lanes during fetch stalls + boot. **✅ DONE: work-driven clock-ENABLE controller (synthesis infers the ICG cell), same-cycle wake + hysteresis, formally-safe (never gates an advancing cluster, 13 064 checks). Measured gated-cycle fraction 73.75 % at the 25 %-duty NVMe/storage-bound die** (≈ the idle fraction minus a 4-cycle wake margin). | **~74 %** of compute idle-dynamic power gated |

## P4 — Energy-specific (J/token)

> **Full low-power design + the bit-exact roadmap to ~1.5–3 J/token [EST] is in
> [`LOW_POWER.md`](LOW_POWER.md)** (energy is ~80 % NVMe/storage read bytes → amortize the fetch; DVFS on the
> 75–80 %-idle die is a free, byte-identical compute-power lever; spec high-K is the staged floor-setter).

The NVMe/storage read byte movement is ~80 % of per-token energy. P2 (decompress, MTP, hit-rate) is the main
energy lever — it directly cuts NVMe bytes. P3.2 (clock gating) trims idle. Beyond RTL:

| # | Item | Type | Note |
|---|---|---|---|
| 4.1 | HBM instead of DDR5 (if energy ≫ cost) | 🟡 | HBM is the lower-energy-per-bit fast tier; the DDR5 choice trades energy for cost. A build-time / **rung-2** board option (DDR5-multi-channel vs HBM — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)); HBM stacks are also what anchor the **rung-3** ASIC's on-package memory. |
| 4.2 | Computational storage / near-NVMe compute | 🔴 | Moves compute to the data to avoid moving bytes — out of RTL scope (RTL can't add IO pins/PHY or near-memory silicon). This **near-memory compute** is exactly a **rung-3 SoC/ASIC** capability (HBM stacks + many-channel PHY + near-memory FP8 at ~TB/s) that breaks the FPGA's IO/PHY ceiling once amortized over volume — the endgame, not a dead-end (see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) rung ③). |

---

## Projected combined effect (single-user, [EST])

Stacking the 🟢 RTL items on the baseline (~3 tok/s, ~8–10 J/token):

| Step | Lever | tok/s | J/token | status |
|---|---|---|---|---|
| baseline | — | ~3 | ~9 | — |
| + flash_xbar 4ch (1.1) | NVMe_BW ×4 | ~12 | ~9 | ✅ built (7.99× latency-hide + N× bank) |
| + expert decompress 1.34× (2.1) | bytes ÷1.34 | ~16 | ~6.7 | ✅ built (measured 1.34×) |
| + MTP K_eff 1.7 (2.2) | ÷ traffic | ~27 | ~4.5 | ⏳ involved (1 MTP layer) |
| ~~+ hit-rate (2.3)~~ | ~~(1−h)↓~~ | — | — | ❌ measured NO-OP at real cache |
| + clock gating (3.2) | idle ↓ | ~27 | ~3.8 | ⏳ enable-side RTL (ICG vendor) |

**Revised target (honest, post-measurement): ~3 → ~16 tok/s with the levers built + verified in RTL today** (flash_xbar ×4 +
decompress 1.34×), **~27 with MTP K>1**; **~9 → ~4–7 J/token**. The cache-hit-rate lever (2.3) was
*measured* to be a no-op at the real cache size — the popular experts the predictor names are
already resident — so the real gains come from **NVMe/storage bandwidth + fewer bytes (decompress, MTP)**,
not cache cleverness. All numbers [EST]; the built RTL items are verified here.

> **These are RTL-lever multipliers on the storage roofline — not a hardware headline.** The absolute
> tok/s each multiplier lands on is set by the board's **memory bandwidth**, which is **rung-dependent**
> (see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)). Staged to the ladder: **rung ① (near-term prove-it —
> low-end FPGA + DDR4 + 1 NVMe) ~5–8 tok/s [EST]**, the honest cheap demo; **rung ② (funded custom board
> — DDR5-multi-channel / HBM + multi-NVMe) ~15–40 tok/s [EST]**, where the ~16–27 lever-stacked figures
> here live; **rung ③ (volume SoC/ASIC — HBM + near-memory compute) ~40+ tok/s [EST]**. "Built today"
> means the RTL is built + verified — **not** that rung-② speed is reachable on rung-① cheap hardware.
> The same bit-exact FP8 RTL runs on every rung; only the bandwidth it is fed changes.

## Execution order (RTL, by impact-per-effort)

1. **`flash_xbar`** (P1.1) — biggest single win, proven pattern (clone ddr5_xbar), BMC-verifiable.
2. **`weight_decomp`** (P2.1) — dual tok/s+energy win, self-contained, verify decode==FP8.
3. **predictor-driven deep prefetch** (P3.1) + hit-rate (P2.3) — wire the built predictor in, measure.
4. **MTP K>1** (P2.2) — extend the speculative loop, verify spec==greedy still exact.
5. **idle clock-gating** (P3.2) — power, low-risk.
6. Document the 🟡 system items (NVMe layout, faster medium, HBM option) as build/board choices.
