# Part selection — the minimum info to lock the rung-① / ② board BOM

*How the physical parts (FPGA, DDR4, NVMe, power) get **confirmed** for the offline
single-user 753B box. This is the bridge between the FPGA-fit measurement and a real
PCB design/BOM.* Companions: [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) (why performance =
bandwidth = the silicon's IO/PHY), [`BOM.md`](BOM.md) (per-rung cost), [`MINIATURIZATION.md`](MINIATURIZATION.md)
(the compact config + the fit-measurement plan), [`FPGA_DEMO_PLAN.md`](FPGA_DEMO_PLAN.md).

> **Nothing here is a committed choice yet.** Every FPGA/DDR/NVMe number below is
> either a **[DERIVED]** target from the bandwidth roofline or **[PENDING]** on the one
> measurement that gates all of it — the vendor place-and-route fit. Silicon fit is not
> yet measured (Phase D / E1 in [`MINIATURIZATION.md`](MINIATURIZATION.md)).

---

## The single gating input: the FPGA-fit measurement

Part selection cannot start from a guess. **FPGA size = fit**, and the fit is the one
thing not yet measured. So the order is fixed:

```
1. Vendor flow measures fit for the compact config
     (PE_N / DDR_NCH / KV_RESIDENT / EFIFO_DEPTH / CACHE_SLOTS, all result-invariant)
       -> real LUT / DSP / BSRAM / URAM utilization           [PENDING]
2. Smallest FPGA that holds that utilization + >=30% headroom  -> FPGA device locked
3. That device's I/O banks + transceivers set DDR4 channel count + PCIe lanes
4. DDR4 (BW + capacity) and NVMe (capacity + PCIe gen) follow from steps 2-3
5. Power tree (rails x max current) follows mechanically from the three datasheets
```

**Until step 1 produces numbers, everything downstream is a target, not a choice.** The
compact config already exists and is byte-identical (`make synth-glm-compact` /
`sim-glm-compact`); what is missing is the routed resource count on real vendor silicon.

---

## 1. FPGA candidates

Repo-designated class: **low-end Kintex UltraScale+ (KU3P-class)**. Hard requirements: a
**multi-channel DDR4 interface**, a **PCIe hard block** (for NVMe), and enough fabric +
DSP + block RAM to hold the FP8 compute die at the compact config.

| Candidate | Role | DDR4 channels (approx) | PCIe | Note |
|---|---|---|---|---|
| **Gowin GW5AT-138** (Tang Mega 138K) | **fit-measurement only (①a)** | DDR3 only, 1 GB soldered | Gen3 x8 (1 core, ~8 GB/s) | **Not a part-selection target** — cheap board to *measure the fit* on; cannot hold 753 GB, no DDR4/NVMe, so it can't run the ①b streaming demo. |
| **Kintex US+ KU3P** (XCKU3P) | if fit is small | HP-bank limited, ~1–2 ch | Gen3/4 hard block | the repo-designated minimum class |
| **KU5P / KU11P / KU15P** | if fit is larger or needs 3–4 ch DDR4 | 2–4 ch | Gen3/4 | more banks/resources + headroom, higher cost |

**Decision rule:** the ①a fit number picks among KU3P / KU5P / KU11P / KU15P. Today only
"around the KU3P class" is confirmable; the exact device is [PENDING] on the fit.

> ①a (fit) vs ①b (demo): *any* FPGA — including the Gowin/Tang Mega — can measure how
> much the die fits (it's a synthesis/P&R resource count). But the ①b demo (offline,
> bit-exact, real tokens streamed from NVMe) needs DDR4 + NVMe, i.e. a KU3P-class board.
> See [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md).

## 2. DDR4 — bandwidth + capacity

- **Bandwidth target.** Every token reads the ~19–21 GB hot-weight set from DDR
  (`tok/s ≈ DDR_BW / hot_footprint`, [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)).
  - DDR4-3200 = **25.6 GB/s per 64-bit channel**.
  - 3 channels ≈ **~77 GB/s**  ·  4 channels ≈ **~102 GB/s** (the doc's ~100 GB/s rung-① target).
- **Capacity.** The ~21 GB hot set must be **resident** → **≥ 24 GB**; leave room for an
  expert cache → **32–48 GB**.
  - e.g. **16 GB × 3 ch = 48 GB** (~77 GB/s)  or  **8 GB × 4 ch = 32 GB** (~102 GB/s).
- **Form.** On a custom board, mount **DDR4 BGA devices directly** (better SI than SO-DIMM);
  a 64-bit channel = 4× x16 devices. Length-matched, impedance-controlled routing.

*(This is the tier — DDR4-many-channels — that rung ② may swap for DDR5-fewer-channels or
HBM to reach ~15–40 tok/s; same bit-exact RTL, only the bandwidth changes.)*

## 3. NVMe — the model store

- **Interface:** PCIe **Gen3 x4 (~3.5 GB/s)** or **Gen4 x4 (~7 GB/s)** M.2 connector.
- **Capacity:** rung-① reduced-config image → **1 TB** is plenty; the full 753 GB target → 1–2 TB.
- **Constraint:** one FPGA PCIe hard block = one NVMe. Multiple drives (rung ②) need
  bifurcation or a PCIe switch + more lanes.

## 4. Power tree — designed on the custom board

| Rail | For | Approx |
|---|---|---|
| VCCINT | FPGA core | 0.85–0.9 V, **high current** (scales with die size) |
| VCCAUX / VCCBRAM | FPGA aux | 1.8 V / 0.9 V |
| MGTAVCC / MGTAVTT | transceivers (PCIe) | 0.9 V / 1.2 V |
| VCCO (DDR banks) | DDR4 I/O | 1.2 V |
| DDR4 VDD / VPP / VTT | memory | 1.2 / 2.5 / 0.6 V (VDDQ/2 tracking) |
| NVMe | M.2 | 3.3 V |

→ a **multi-rail PMIC + several DC-DCs**. Power integrity (decoupling, planes) is, with
DDR routing, the core of the layout difficulty.

---

## Minimum-info checklist to LOCK each part

| Part | Minimum info required to confirm | Source |
|---|---|---|
| **FPGA** | LUT/DSP/BSRAM/URAM utilization + ≥30% headroom → smallest device; DDR4 channels + PCIe lanes = bank/transceiver budget | **①a fit measurement — [PENDING]** |
| **DDR4** | target tok/s → required BW → channels × speed; hot set ~21 GB → capacity ≥ 32 GB | roofline ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)) |
| **NVMe** | reduced-config image size → capacity; Gen3 vs Gen4 → bandwidth | FPGA PCIe generation |
| **Power** | after the three above: each rail's voltage × max current → PMIC selection | FPGA/DDR/NVMe datasheets |

## Fit-measurement status (what is measured vs still blocked)

**Measured — per compute unit (open flow, DSP-inferred)** — `glm_matmul_fp8` maps to
**~17.8 K LUT4-equiv + 20 DSP (MULT18X18/MULT9X9) + ~5.4 K DFF** (yosys 0.66 `synth_gowin`
with DSP inference; the `abc -lut4` path that "timed out" was a tool limitation, not a
design one — see [`../fpga/README.md`](../fpga/README.md)).

**Measured — compact whole-system coarse demand (open flow, pre-map)** — `glm_fp8_system_cdc`
at the compact config (PE_N=2, DDR_NCH=2, KV_RESIDENT=8, EFIFO_DEPTH=8, CACHE_SLOTS=2):

| Coarse cell | Count | Reads as |
|---|---|---|
| `$mul` (multipliers) | **346** | multiply ops = **DSP demand ceiling** (small FP8 mants pack; not 1 DSP each) |
| `$mem_v2` (memories) | **54** | BSRAM demand (count; sizes set the block total) |
| `$dff` (registers) | 4004 cells (multi-bit) | FF demand |
| `$add` / `$sub` / `$mux` | 7558 / 2063 / 112657 | LUT/carry logic demand |

**Which device to compare against — NOT GW5AST-138.** The Gowin **GW5AST-138** (Tang Mega,
138,240 LUT4 / 298 DSP) is **DDR3-only, no NVMe** → it **cannot be the product board** (the ①b
streaming demo needs DDR4 + NVMe). It was only ever a *fabric-fit curiosity*, and comparing 346
mul ops to its 298 DSP ("DSP tight") is measuring against a **disqualified device** — an artifact,
not a real constraint. The real target is a **KU3P-class part (DDR4-capable)** with **~1,368 DSP48E2**
(XCKU3P — confirm vs datasheet), several × the Gowin part → **346 mul ops is comfortable, DSP is
not the binding resource on the actual target.**

**BLOCKED — the authoritative routed fit against the real (KU3P-class) device** (LUT / DSP / BSRAM +
**Fmax**). Neither flow on this machine hits the right device:
- **Gowin open flow (oss-cad-suite)** → targets GW5A (the disqualified DDR3 part), *and* yosys 0.66
  has no `gw5a` DSP inference (`mul2dsp` is gw1n/gw2a-only), so the FP8 datapath explodes in
  `abc -lut4`. Wrong device **and** a tool gap.
- **KU3P-class fit** needs **AMD/Xilinx Vivado** (DDR4 IP + real DSP48 map + P&R) — **not installed
  on this Mac**. So the fit-that-matters is tooling-blocked here, separate from the Gowin issue.

| Field | Value | Status |
|---|---|---|
| Per-unit fit (`glm_matmul_fp8`, Gowin DSP-inferred) | ~17.8K LUT4-eq + 20 DSP + 5.4K DFF | **[MEASURED]** — indicative only (wrong-device fabric) |
| Whole-system multiply demand (device-agnostic) | 346 `$mul` ops (compact) | **[MEASURED, coarse]** |
| Whole-system memory demand (device-agnostic) | 54 arrays (compact) | **[MEASURED, coarse]** |
| Routed LUT / DSP / BSRAM on a **KU3P-class** part | — | [PENDING — needs **Vivado**] |
| Routed Fmax | — | [PENDING — needs Vivado] |
| → FPGA device chosen (KU3P / 5P / 11P class) | DSP not tight; sized by LUT/BRAM + DDR4 banks | [PENDING — needs Vivado fit] |
| → DDR4 channels × size | — | [PENDING] |
| → NVMe (gen × capacity) | — | [PENDING] |
| → power rails / PMIC | — | [PENDING] |

> **Correction of record:** an earlier draft treated GW5AST-138's 298 DSP as the fit budget and
> read "DSP tight." That device is DDR4-disqualified; the device-agnostic demand (346 mul / 54 mem)
> is the real measurement, and it is comfortable on the KU3P-class target. The authoritative routed
> fit needs Vivado on a KU3P-class part.

---

**Bottom line.** The only prerequisite to part selection is the **fit measurement (①a)**.
Once it lands: the FPGA device is chosen (KU3P/5P/11P class), the DDR4 channels/capacity
follow from the roofline, and NVMe + power fall out of those datasheets mechanically —
which is exactly the input a PCB schematic/layout needs to begin.
