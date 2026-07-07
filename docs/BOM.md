# BOM & per-seat economics — the box across the 3 rungs

*What the box actually costs to build, at each rung of [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md).
The point: the FPGA is **not** the dominant cost — **memory + storage + board** are. Every figure is an
**order-of-magnitude [EST]**; exact FPGA quotes need a distributor, exact board cost needs a PCB-house
quote. Prices are single-unit / low-volume unless noted.*

> **Why this exists.** The FPGA-fit track ([`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md)) sets the FPGA class;
> the FPGA class + memory + storage set the BOM; the BOM sets the per-seat price; the per-seat price is
> what makes the [`ICP.md`](ICP.md) economics real. This doc closes that chain with actual numbers.

---

## The cost shape (read this first)

The workload is memory-bandwidth-bound, so the box is **memory- and storage-dominated, not compute-dominated**:

- **Compute (FPGA)** — cheap relative to the rest, and cheaper still at ASIC volume. Even a mid FPGA is a
  minority of the BOM.
- **Fast DDR (bandwidth)** — the real cost driver, because *bandwidth* (channels/PHY) is what performance
  needs, and more channels = a bigger chip + more DRAM + a harder board.
- **NVMe (capacity)** — cheap per TB; the 753 GB model fits ~1 TB.
- **Board + power + enclosure** — rises steeply with signal speed (DDR5/PCIe Gen4/HBM = 8–12-layer
  controlled-impedance PCB, outsourced design + assembly).

So "make the box cheaper" ≈ "need less memory bandwidth" ≈ "accept lower tok/s" — the ladder, in cost form.

---

## Rung ① — prove-it box (now, ~$1–2 k)

Low-end **Kintex UltraScale+ (KU3P-class)** dev board + DDR4 + one NVMe. **Reduced-config demo** (a dev
board's DDR/storage can't hold 753 GB); goal is *"real 753B-family RTL runs on real FPGA silicon,
offline, bit-exact"*, at ~5–8 tok/s [EST].

| Line | Part (example) | ~Cost | Note |
|---|---|---|---|
| FPGA (on dev board) | KU3P-class board (e.g. a KCU-class or KU3P eval) | **~$1,000–2,500** | dev board = FPGA + power + clocks + JTAG bundled; buy the board, not the raw chip |
| DDR | on-board (dev board's own DDR4) | (incl.) | dev board ships with some DDR4 |
| NVMe | 1× M.2 NVMe 1 TB | ~$60–100 | holds a reduced-config weight image |
| Vivado | ML edition (KU3P may need paid) | ~$0–3,000/yr | WebPACK covers small parts; confirm KU3P tier |
| **Prove-it total** | | **~$1,000–2,500 + tool** | one unit, for the demo — not a product |

> This rung is **capex for the demo**, not a per-seat product cost. Its job is to convert `[EST]` → a
> measured Fmax ÷ cyc_per_tok and a real "it runs" video. Cheapest path to the fundable proof.

---

## Rung ② — custom product board (post-seed, ~$3–6 k/box)

Custom PCB (outsourced artwork + assembly) carrying a **mid FPGA with DDR5 multi-channel or HBM** + big
DDR + multi-NVMe. This is the actual **shippable single-user box** at ~15–40 tok/s [EST].

| Line | Part | ~Cost | Note |
|---|---|---|---|
| **FPGA** | Versal / Agilex / HBM-class US+ (DDR5 or HBM, multi-PCIe) | **~$1,500–5,000** | the bandwidth-capable chip; a minority of BOM |
| **Fast DDR** | 64 GB DDR5 (multi-channel) *or* HBM (on-package, 16–32 GB) | ~$300–700 (DDR5) / (HBM in chip) | the hot-set cache; bandwidth is the cost, not GB |
| **NVMe** | 1–4 TB (1–2 drives over PCIe) | ~$100–400 | full 753 GB model + KV overflow |
| **PCB** | 8–12-layer controlled-impedance, outsourced design | ~$300–800 (proto/unit; NRE separate) | DDR5/PCIe Gen4 signal integrity = many layers |
| **Assembly** | BGA reflow + PnP (turnkey) | ~$200–600/unit | vendor does it; BGA can't be hand-soldered |
| **Power / clock / connectors / enclosure / USB-C** | PMIC, oscillators, M.2/PCIe conn, case | ~$150–400 | |
| **Vivado/Quartus** | paid (amortized) | ~$3,000/yr / N units | tool, not per-box |
| **Rung-② box BOM** | | **~$2,500–6,000/unit** | + one-time NRE (PCB design ~$10–30 k, paid once) |

**One-time (NRE, not per-box):** custom PCB design/artwork outsourced **~$10,000–30,000+**, plus a few
board revisions. Amortized over units, negligible per-seat at any real volume.

---

## Rung ③ — SoC/ASIC (at volume, endgame)

Custom silicon (HBM + many-channel PHY + near-memory FP8). **~40+ tok/s, lower power, lower $/seat** — but
only after volume justifies the NRE.

| Line | ~Cost | Note |
|---|---|---|
| **ASIC NRE** (masks, tapeout, IP) | **~$1 M–10 M+** one-time | mature node (not bleeding-edge — bandwidth-bound, not compute-bound); dominant risk |
| **Per-unit silicon** (at volume) | ~$50–300/chip | far below FPGA once amortized |
| **HBM** (on-package) | ~$100–400 | the bandwidth source |
| **NVMe + board + assembly** | ~$300–800 | simpler board than FPGA (integration on-die) |
| **Rung-③ box BOM (at volume)** | **~$1,000–2,000/unit** + amortized NRE | the cost-down + perf + power win the user flagged |

ASIC only makes sense once **unit volume × (FPGA-cost − ASIC-cost) > NRE** — i.e. at product-market fit /
Series-B scale. Sequenced last, on purpose.

---

## Per-seat economics — does it sell?

The pitch is **not** "cheap tokens" — it's *"the only turnkey way to run a frontier model where the cloud
can't go, offline, at a seat price"* ([`ICP.md`](ICP.md)). So the comparison that matters:

| Option | Frontier 753B? | Offline / air-gapped? | ~Cost per seat |
|---|---|---|---|
| Cloud frontier API | ✅ | ❌ (disqualifies the ICP) | ~$20–200/mo — *but banned* |
| Mac/GPU + 70 B local | ❌ (quality gap) | ✅ | ~$3–6 k one-time |
| 8×H100 self-host 753 B | ✅ | ✅ | **~$250–400 k** (shared, + power + MLOps) |
| **This box — rung ②** | **✅** | **✅** | **~$3–6 k/box (one seat) + support** |
| **This box — rung ③ (volume)** | ✅ | ✅ | **~$1–2 k/box** |

**The number that sells:** a rung-② box at **~$3–6 k** vs **8×H100 at ~$250–400 k** = **~50–100× cheaper**
for the offline-753B use case. Not a "$500 desk accessory," but for a buyer whose alternative is a
$400 k datacenter build (or *nothing*, because the cloud is barred), a **$5 k provably-offline frontier
box** is a trivial line item — legal already pays $100–500/seat/mo for Westlaw-class tools; a per-seat
appliance fits.

## Honest limits

- All prices are **order-of-magnitude [EST]** — FPGA needs a distributor quote, board needs a PCB-house
  quote, ASIC NRE is a wide band. Treat as ranges, not commitments.
- Rung-② tok/s (~15–40) is the funded number; the **near-term demo (rung ①) is ~5–8 and reduced-config**.
- BOM is **memory/storage/board-dominated**; the FPGA is a minority. "Cheaper box" means "less bandwidth"
  means "lower tok/s" — the ladder, in money.
- Software / host / support / margin are **on top** of these hardware BOMs (a product sells above BOM).
