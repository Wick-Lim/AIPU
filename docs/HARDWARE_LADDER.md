# Hardware performance ladder — prove cheap, scale on funding

*The honest hardware plan for the local single-user GLM-5.2 box (753B-param MoE, ~40B active/token).
It replaces the earlier flat "64 GB DDR5 / ~100 GB/s / 25–40 tok/s" assumption with a **staged ladder**:
the performance you get is set by **memory bandwidth** — specifically the **NVMe/PCIe bandwidth that
streams the routed experts** — memory bandwidth is set by the **FPGA/silicon's IO + PHY**, and that is
set by **how much money is in the build**. So the plan is: **prove it works cheap → raise → scale.***

> All tok/s here are **[EST]** — first-order projections from the bandwidth roofline
> (`tok/s ≈ sustained streaming BW / [(1−h)·routed footprint] · K`), **not** measured silicon. There is
> **no** routed-netlist Fmax, no PnR fit, and no running board yet; every figure below stays **[EST]**
> until a Vivado/Gowin fit + a bring-up board measures a real Fmax ÷ cyc_per_tok. Only rung ① is a
> near-term buildable proof; ②③ are funding-gated projections.

> **Local-device retarget (Q4_K).** `main` develops the **Q4_K local-inference track** — the target
> weight store is the published `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL` (**~467 GB**, ~38% smaller than the
> 753 GB FP8 checkpoint). At ~0.6 B/param avg vs FP8's ~1.0, Q4_K reads **~1.6× fewer bytes/token**, so on
> a bandwidth-bound box it is **~1.6× faster than FP8** at the same interface [EST]. "Bit-exact" on this
> track means **bit-exact to our ggml-Q4_K reference `tools/q4k_ref.py`** — the whole-file /
> mixed-type Q6_K·Q8_0·F16 / real published-GGUF checks are **OPEN** (see
> [`../README.md`](../README.md)). FP8 is preserved on branch **`fp8`** + tag **`fp8-verified-baseline`**
> ([`Q4K_RETARGET.md`](Q4K_RETARGET.md)).

---

## The one thing that sets performance: **memory bandwidth, not compute**

The workload is **NVMe/PCIe-bandwidth-bound**. The Q4_K compute die is small and sits largely idle
(~75–80% starved) behind the memory system (documented throughout — [`ULTRA_PERF.md`](ULTRA_PERF.md),
[`MINIATURIZATION.md`](MINIATURIZATION.md)). So **compute is cheap; the wall is reading weights** — every
token streams **~25 GB** of weights (~40B active params × ~0.6 B/param at Q4_K), split into two very
different byte pools:

- **~9 GB hot-set** (MLA projections all layers + shared expert + dense FFN + router + embed/LM-head).
  These are the **same** bytes every token, so they **cache in the DDR working set** — read once, reused.
- **~14 GB routed-expert bytes** (top-8 of 256 experts per layer). These **change every token** with the
  router's choice, so they **cannot be cached** at a useful hit rate — routing entropy caps it, and the
  predictor-prefetch lever is a **MEASURED no-op** on the trace harness. They must be **streamed fresh
  from NVMe/Flash every token**. **This ~14 GB is the wall.**

So the bit-exact roofline is `tok/s ≈ sustained NVMe/Flash streaming BW / ~14 GB`. The ~9 GB hot-set
lives in DDR and is free after the first token; **DDR bandwidth only starts to matter once routed experts
also hit in the DDR cache** — which the routing entropy above limits. **The streaming bandwidth is set by
the number of PCIe lanes / NVMe drives (and, on the upper rungs, DDR5 channels / HBM stacks), which is set
by the chip's IO pins + hard PHYs** — a physical property of the silicon you buy, *not* something RTL can
add (our `ddr5_xbar`/`flash_xbar` already parameterize the channel count; the ceiling is the chip's pins).
**More bandwidth = a bigger/newer chip and more drives = more money.** That single fact produces the ladder.

---

## The ladder

| Rung | Silicon | Streaming path | tok/s [EST] | Bit-exact? | ~Box BOM | Funding | When |
|---|---|---|---|---|---|---|---|
| **① Prove-it (cheap)** | low-end FPGA (Kintex US+ **KU3P** class) + DDR4 hot-set cache | 1–2 NVMe (~7–14 GB/s) … striped ~14 drives (~100 GB/s) | **~0.5–1 … ~5–8** | **yes** (bit-exact throughout) | ~$1–2 k (floor) → higher w/ drive array | self / minimal | **now (the demo)** |
| **② Custom board** | mid FPGA (Versal / Agilex / HBM-class US+) DDR5 multi-ch or HBM | DDR5 8–12 ch or HBM (~400 GB/s–1 TB/s) feeding the working set | **~15–40** | **contingent** — needs expert-cache hit rate *or* non-bit-exact pruning | ~$3–6 k | seed | post-raise |
| **③ SoC / ASIC** | custom silicon (HBM stacks + many-channel PHY + near-memory Q4_K compute) | HBM3 on-package (~TB/s) | **~120 aspirational** | tiered (467 GB ≠ fits HBM) | high NRE, low unit | Series B+ / volume | at scale |

*Per-rung parts, box BOM, and per-seat economics: [`BOM.md`](BOM.md) — all cost/economics figures are
**[EST]/[PENDING]** planning numbers, not quotes. Short version — the BOM is memory/storage/board-dominated,
the FPGA is a minority, and a rung-② ~$3–6 k box is order-of-magnitude cheaper than an 8×H100 self-host
for the offline-753B use case [EST].*

Each rung is **the same RTL** — whose **Q4_K GEMM core is bit-exact vs the ggml reference**
(`glm_matmul_q4k` 160/160), while the **assembled end-to-end model golden is still OPEN**
([`../README.md`](../README.md)). Only the memory interface it drives changes; nothing about the model
changes across rungs — **just the bandwidth the silicon can feed it.**

### ① Prove-it — the near-term goal
A **low-end Kintex UltraScale+ (KU3P-class)** dev board, DDR4 holding the **~9 GB hot-set cache**, and an
NVMe/Flash array streaming the **~14 GB routed experts**. Throughput is set by that storage array, and it
is **bit-exact at every point on the band**:

- **BOM floor — 1–2 NVMe (~7–14 GB/s) → ~0.5–1 tok/s [EST].** The honest cheap-box number. Slow, but real.
- **4 NVMe (~28 GB/s) → ~2 tok/s [EST].**
- **Striped ~14 drives (~100 GB/s) → ~5–8 tok/s [EST]** — the top of rung ①, still bit-exact; the striping
  strategy is [`FLASH_STRIPING.md`](FLASH_STRIPING.md).

**Slow, but real and bit-exact** — the point of this rung is not speed, it's proving *"the full GLM-5.2
runs on real FPGA silicon, offline, producing the reference model's tokens."* It is a **reduced-config**
demo first, since a dev board's small DDR/Flash can't hold 467 GB — see
[`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md). That working demo is what makes rung ② fundable.
**Prove cheap, then raise.**

### ② Custom board — post-seed
A **custom PCB** (artwork + assembly outsourced) carrying a **mid-tier FPGA with DDR5 multi-channel or
HBM** (Versal / Agilex / an HBM-class UltraScale+) + multiple NVMe over more PCIe lanes. DDR5 doubles
per-channel BW vs DDR4; HBM gives ~460 GB/s/stack — either reaches **~400 GB/s–1 TB/s to the working
set**. But note the **honesty knob**: that bandwidth feeds the DDR/HBM-resident **~9 GB hot-set** at full
rate; the **~14 GB routed experts still change every token**, so **~15–40 tok/s [EST] is *contingent***,
not free. You reach it only by either

- **(bit-exact) landing routed experts in the DDR/HBM cache** at a real hit rate — capped by routing
  entropy, and the predictor-prefetch path is a **MEASURED no-op** ([`ULTRA_PERF.md`](ULTRA_PERF.md)); or
- **(NOT bit-exact) activation-sparsity / expert pruning** — a separate model-quality decision that trades
  the bit-exact guarantee for fewer streamed experts.

Absent either, the box stays on the rung-① NVMe wall even with fast DDR. The two hardware routes
(DDR5-many-channels vs newer-FPGA-DDR5 vs HBM) **converge near the same ~$3–6 k build** — the cost is the
memory-bandwidth silicon, whichever way you buy it. This is the interactive product rung.

### ③ SoC / ASIC — at volume
**Reframe of the earlier "ASIC out of scope".** That call was made under *"compute-bound → ASIC's compute
edge is wasted"*. But the real bottleneck is **memory bandwidth (IO pins + PHY)**, and **an ASIC is
exactly what breaks the FPGA's IO/PHY ceiling** — it can integrate **HBM stacks + many-channel controllers
+ near-memory Q4_K compute** that no FPGA package offers, at **~TB/s**, with **lower per-unit cost and
power** once amortized over volume. An HBM3 ceiling (~3 TB/s) roofs at **~120 tok/s** — but **aspirational**:
the ~467 GB Q4_K checkpoint **does not fit** in an HBM budget (≤192 GB), so an ASIC still needs the tiered
NVMe→HBM streaming path, not an all-HBM resident model. So ASIC is **not off the table — it is the endgame**:
its multi-million NRE and months–years lead time only pay off **at manufacturing volume**, exactly where a
shipping product lives. **Sequence: FPGA (rungs ①②) to prove + reach product-market fit → ASIC (③) when
volume justifies the NRE and demands the lower $/seat + higher tok/s + lower power.** Not now (no volume,
no capital); real later (that's how bandwidth-bound silicon products scale).

---

## Why this ordering is the right bet

- **De-risk before spend.** A ~$1–2 k FPGA proves the RTL on real silicon before any ~$3–6 k custom board
  or multi-million ASIC. Skipping straight to expensive hardware bets money on unverified silicon behavior.
- **The demo, not the spec, raises the money.** "GLM-5.2 runs bit-exact (vs the ggml reference) on an FPGA,
  offline" (rung ①) is a stronger pitch than a datasheet promise of 40 tok/s. Investors discount `[EST]`;
  they fund a working box.
- **Each rung funds the next.** Prove-it → seed → custom board → PMF → Series B → ASIC. Standard
  bootstrapped-hardware sequencing; no rung asks for money the previous rung hasn't justified.
- **Same moat throughout.** Offline / air-gapped, full-frontier, Q4_K-native — unchanged on every rung.
  Only the tok/s the silicon can feed goes up. The [`ICP.md`](ICP.md) buyer (offline is *mandatory*) values
  *"it runs at all, provably local"* on rung ①, and pays more for rung ②'s speed.

## Honest caveats

- Every tok/s is **[EST]** — roofline projections, no routed netlist / Fmax / running board. Even rung ①
  is projected until the FPGA demo measures a real Fmax ÷ cyc_per_tok.
- **Bit-exact ≠ real GGUF.** All of the above is bit-exact only to our ggml reimpl `tools/q4k_ref.py`;
  the whole-file, mixed-type (Q6_K/Q8_0/F16 — RTL is **Q4_K-only**), and real published-GGUF / llama.cpp
  checks are **OPEN** ([`../README.md`](../README.md)).
- **Separate the bit-exact band from the knobs.** Rung ① (~0.5–8 tok/s) is bit-exact throughout. Rung ②'s
  ~15–40 needs *either* a DDR-cached-expert hit rate (routing-entropy-limited; prefetch is a measured
  no-op) *or* non-bit-exact pruning — state which one when quoting it.
- Rung ① is **reduced-config** (dev-board DDR/storage can't hold 467 GB); the full model needs the custom
  board (②). The demo proves the *mechanism on real silicon*, not the full box.
- Compute-side levers (e.g. the prior FP8 track's `weight_decomp` 1.34× lossless pack — **FP8-specific,
  branch `fp8`; no Q4_K re-run**) improve area/power/timing but **do not move an NVMe-bound roofline** —
  only more streaming bandwidth (drives / channels / HBM) moves tok/s.
- Chip/board prices are order-of-magnitude; exact FPGA quotes need a distributor, exact board cost needs
  the PCB-house quote. BOM is memory/storage-dominated, not FPGA-dominated. All economics **[EST]/[PENDING]**.
