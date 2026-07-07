# Hardware performance ladder — prove cheap, scale on funding

*The honest hardware plan for the local single-user 753B box. It replaces the earlier flat
"64 GB DDR5 / ~100 GB/s / 25–40 tok/s" assumption with a **staged ladder**: the performance you get is
set by **memory bandwidth**, memory bandwidth is set by the **FPGA/silicon's IO + PHY**, and that is set
by **how much money is in the build**. So the plan is: **prove it works cheap → raise → scale.***

> All tok/s here are **[EST]** — first-order projections from the bandwidth roofline
> (`tok/s ≈ storage/DDR BW / [(1−h)·footprint] · K`), **not** measured silicon. Only rung ① is a
> near-term buildable proof; ②③ are funding-gated projections.

> **Local-device retarget (Q4_K).** `main` now develops the **Q4_K local-inference track** — the target
> weight store is the published `unsloth/GLM-5.2-GGUF : UD-Q4_K_XL` (**467 GB**, ~38% smaller than the
> 753 GB FP8 checkpoint). Since the box is memory-bandwidth-bound, the smaller footprint lifts the roofline:
> the per-token hot / routed byte counts below scale down ~proportionally (fewer bytes/token → more tok/s
> at the same bandwidth), so the tok/s figures are conservative for Q4_K. FP8 is preserved on branch
> **`fp8`** + tag **`fp8-verified-baseline`** ([`Q4K_RETARGET.md`](Q4K_RETARGET.md)).

---

## The one thing that sets performance: **memory bandwidth, not compute**

The workload is memory-bandwidth-bound. The FP8 compute die is small and sits largely idle behind the
memory system (documented throughout — [`ULTRA_PERF.md`](ULTRA_PERF.md), [`MINIATURIZATION.md`](MINIATURIZATION.md)).
So **compute is cheap; the wall is reading weights** — every token reads:

- **~28 GB hot weights** (MLA projections all layers + shared expert + dense FFN + router + embed/LM-head),
  reduced to **~19 GB** with the bit-exact levers (`weight_decomp` 1.34× + MLA weight-absorption) —
  served from **fast DDR**.
- **~16 GB routed-expert bytes** (top-8/layer, cache-miss fraction) — streamed from **NVMe/PCIe**.

`tok/s` is then `DDR_BW / hot_footprint` (DDR-bound) capped by `NVMe_BW / routed_footprint`
(storage-bound). **Both bandwidths are set by the number of memory channels / PCIe lanes, which is set by
the chip's IO pins + hard PHYs** — a physical property of the silicon you buy, *not* something RTL can add
(our `ddr5_xbar` already parameterizes `DDR_NCH`; the ceiling is the chip's pins). **More bandwidth = a
bigger/newer chip = more money.** That single fact produces the ladder.

---

## The ladder

| Rung | Silicon | Memory | tok/s [EST] | ~Box BOM | Funding | When |
|---|---|---|---|---|---|---|
| **① Prove-it (cheap)** | low-end FPGA (Kintex US+ **KU3P** class) | DDR4 ~4 ch (~100 GB/s) + 1 NVMe | **~5–8** | ~$1–2 k | self / minimal | **now (the demo)** |
| **② Custom board** | mid FPGA (Versal / HBM-class) DDR5 multi-ch | DDR5 8–12 ch or HBM (~300–600 GB/s) + multi-NVMe | **~15–40** | ~$3–6 k | seed | post-raise |
| **③ SoC / ASIC** | custom silicon (HBM stacks + many-channel PHY) | HBM / on-package (~TB/s) | **~40+**, lower $/seat, lower power | high NRE, low unit | Series B+ / volume | at scale |

*Per-rung parts, box BOM, and per-seat economics: [`BOM.md`](BOM.md). Short version — the BOM is
memory/storage/board-dominated, the FPGA is a minority, and a rung-② ~$3–6 k box is ~50–100× cheaper than
an 8×H100 self-host for the offline-753B use case.*

Each rung is **the same verified RTL** (bit-exact FP8 datapath); only the memory interface it drives
changes. Nothing about the model or its correctness changes across rungs — **just the bandwidth the
silicon can feed it.**

### ① Prove-it — the near-term goal
A **low-end Kintex UltraScale+ (KU3P-class)** dev board + DDR4 (≈4 channels, ~100 GB/s) + one NVMe.
At the ~19 GB bit-exact hot footprint that is **~5 tok/s** DDR-bound (a few more with caching/spec-K).
**Slow, but real and bit-exact** — the point of this rung is not speed, it's proving *"the full 753B runs
on real FPGA silicon, offline, producing the real model's tokens"* (a **reduced-config** demo first, since
a dev board's small DDR/Flash can't hold 753 GB — see [`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md)). That
working demo is what makes rung ② fundable. **Prove cheap, then raise.**

### ② Custom board — post-seed
A **custom PCB** (PCB-artwork + assembly outsourced) carrying a **mid-tier FPGA with DDR5 multi-channel
or HBM** (Versal / Agilex / an HBM-class UltraScale+) + multiple NVMe over more PCIe lanes. DDR5 doubles
per-channel BW vs DDR4, HBM gives ~460 GB/s/stack — either reaches **~300–600 GB/s → ~15–40 tok/s [EST]**,
the interactive product. Note the two routes (DDR4-many-channels on a big FPGA vs DDR5-few-channels on a
newer FPGA vs HBM) **converge near the same ~$3–6 k build** — the cost is the memory-bandwidth silicon,
whichever way you buy it.

### ③ SoC / ASIC — at volume
**Reframe of the earlier "ASIC out of scope".** That call was made under *"compute-bound → ASIC's compute
edge is wasted"*. But the real bottleneck is **memory bandwidth (IO pins + PHY)**, and **an ASIC is
exactly what breaks the FPGA's IO/PHY ceiling** — it can integrate **HBM stacks + many-channel controllers
+ near-memory FP8 compute** that no FPGA package offers, at **~TB/s**, with **lower per-unit cost and
power** once amortized over volume. So ASIC is **not off the table — it is the endgame**: its multi-million
NRE and months–years lead time only pay off **at manufacturing volume**, exactly where a shipping product
lives. **Sequence: FPGA (rungs ①②) to prove + reach product-market fit → ASIC (③) when volume justifies
the NRE and demands the lower $/seat + higher tok/s + lower power.** Not now (no volume, no capital); real
later (that's how bandwidth-bound silicon products scale).

---

## Why this ordering is the right bet

- **De-risk before spend.** A ~$1–2 k FPGA proves the RTL on real silicon before any ~$3–6 k custom board
  or multi-million ASIC. Skipping straight to expensive hardware bets money on unverified silicon behavior.
- **The demo, not the spec, raises the money.** "753B runs bit-exact on an FPGA, offline" (rung ①) is a
  stronger pitch than a datasheet promise of 40 tok/s. Investors discount `[EST]`; they fund a working box.
- **Each rung funds the next.** Prove-it → seed → custom board → PMF → Series B → ASIC. Standard
  bootstrapped-hardware sequencing; no rung asks for money the previous rung hasn't justified.
- **Same moat throughout.** Offline / air-gapped, full-frontier, bit-exact — unchanged on every rung. Only
  the tok/s the silicon can feed goes up. The [`ICP.md`](ICP.md) buyer (offline is *mandatory*) values
  *"it runs at all, provably local"* on rung ①, and pays more for rung ②'s speed.

## Honest caveats

- Every tok/s is **[EST]**; even rung ① is projected until the FPGA demo measures a real Fmax ÷ cyc_per_tok.
- Rung ① is **reduced-config** (dev-board DDR/storage can't hold 753 GB); the full model needs the custom
  board (②). The demo proves the *mechanism on real silicon*, not the full box.
- The hot-set reduction is capped at **~1.5× bit-exact** ([`ULTRA_PERF.md`](ULTRA_PERF.md)); bigger cuts need
  activation sparsity (not bit-exact — a separate quality decision).
- Chip/board prices are order-of-magnitude; exact FPGA quotes need a distributor, exact board cost needs
  the PCB-house quote. BOM is memory/storage-dominated, not FPGA-dominated.
