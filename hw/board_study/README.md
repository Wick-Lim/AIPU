# v3-proto board — routability study

Turns the [`R3_APPLIANCE_SPEC.md`](../../docs/R3_APPLIANCE_SPEC.md) §5c board claim
— *"1280-bit, 24 GB ×20 board-mounted, **12-layer HDI, ~130×110 mm**"* — from a
hand-calculation into a **checked, falsifiable** artifact. Two scripts, both `$0`
and reproducible:

| script | question it answers | verdict |
|---|---|---|
| `escape_analysis.py` | does the 1280-bit escape fit in **6 signal layers** (the 12-layer stack's budget)? | **PASS (with margin)** — ~4 signal layers needed vs 6 available; model ~calibrated (±1–2 layers) |
| `gen_gerbers.py` | do 20 DRAM + the SoC physically **fit in 130×110 mm** with clearance? | **DRC PASS** — 0 courtyard overlaps, ≥1 mm edge clearance, 43% area |

## What each proves — and its honest limit

**Escape analysis** (`escape_analysis.py`). Computes the escape-routing layer count
from first principles: via-in-pad HDI, the ring-cut channel-crossing bound (a signal
on ring *r* must cross every outer ring; each ring's channel capacity is
`perimeter × traces_per_channel × layers`). For the SoC BGA (~4,300 balls, 0.75 mm
pitch, **2,800 signals** to escape) it needs **~4 signal layers** — inside the 6 a
12-layer stack (6 SIG / 4 GND / 2 PWR) provides. A built-in **calibration case**
(phone-AP-class, ~680 signals in a 35×35 = 1,225-ball package) lands at **5 vs the
~6 real phones use** — so the model is **roughly calibrated (±1–2 layers, if
anything mildly optimistic), NOT a conservative upper bound**. Treat the SoC's 4 as
a central estimate; even optimistic by 1–2 layers (real ~4–6) it still fits within
6, so the 2-layer spare absorbs the uncertainty and the 12-layer claim is
**defensible at analytical fidelity — margin tighter than a routed board confirms**.
*Limit:* it counts channel capacity, not literal routed traces; it ignores the
shorter per-bank crossing a directional DRAM placement would give (that only helps).
A real place-and-route confirms the final margin.

**Gerbers + placement DRC** (`gen_gerbers.py`). Emits RS-274X gerbers + an Excellon
drill for the board **to true 1:1 scale** (open in any gerber viewer, e.g.
[tracespace.io](https://tracespace.io), or the committed `board_preview.svg`):
board outline, the SoC, and 20 LPDDR5X 24 GB 496-FBGA packages in a ring (6 top /
6 bottom rows + 4 left / 4 right columns). The **placement DRC** — the check that
actually answers *"does it fit"* — verifies **0 courtyard overlaps** and **≥1 mm
board-edge clearance**. It caught a real error first pass: the naive ring collided
at the corners because four 15 mm packages cannot sit along the SoC's 50 mm edge
(they span ~60 mm) — the L/R columns had to move to the outboard corner lanes and
the rows tighten. *Limit:* this is placement + outline + courtyard (fab/assembly-
footprint fidelity), **not routed copper**; routability is the escape analysis's job.

## What is NOT claimed

This is a **routability study**, not a fab-ready board. The SoC ball-out is the
§5c estimate (~4,300 balls) — a **dummy footprint**; a manufacturable board needs
the real SoC package (post-tapeout) and full signal-integrity work (impedance,
byte-lane length-match, microvia stackup sim). What the study *does* settle:
**12 layers and 130×110 mm are the right order of magnitude, defensible from
physics, not hand-waved** — and both claims are now falsifiable (change the inputs,
the verdicts change).

## Reproduce

```sh
python3 hw/board_study/escape_analysis.py   # layer-count verdict
python3 hw/board_study/gen_gerbers.py        # gerbers + drill + placement DRC + preview
```
Outputs: `gerbers/*.gbr`, `gerbers/board.drl`, `board_preview.svg`.
