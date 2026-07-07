# USB-C Personal AI Appliance — Productization Plan

**What this is.** The concrete plan to turn the verified accelerator into a **shippable USB-C
external device**: a small, self-powered, active-cooled box that runs the real
`zai-org/GLM-5.2-FP8` (753 GB) locally — **fully offline / air-gapped, no internet ever** — and
streams tokens to a host computer over a single USB-C cable.

**Relationship to the other roadmaps (read together):**
- [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) — the **performance ladder** (the anchor for every tok/s
  headline here). Throughput is set by **memory bandwidth**, which is set by the FPGA/silicon's IO +
  PHY, which is set by budget — so speed is **staged across rungs**: ① prove-it FPGA ~5–8 · ② funded
  custom board ~15–40 · ③ SoC/ASIC at volume ~40+ tok/s [EST]. Read every tok/s in this doc as
  **rung-dependent** per that ladder.
- [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) — the **RTL / silicon track** (P1–P4: real-model
  fidelity, full-scale, robustness, vendor-IP + FPGA physical). *That* makes the chip correct and
  real. **This** doc is the **device / appliance track** on top of it: form factor, power, thermal,
  host software, enclosure, manufacturing, go-to-market.
- [`SYSTEM_SINGLE_PACKAGE.md`](SYSTEM_SINGLE_PACKAGE.md) — the single-package memory/tiering + cost
  model (DDR5 fast tier + 1–4 TB NVMe bulk store; USB-C carries only token IDs).
- [`LOW_POWER.md`](LOW_POWER.md) — the energy budget + the bit-exact low-power levers (the ~80 %
  storage-read-byte reality — NVMe/PCIe reads dominate per-token energy; spec ÷K, compression, DVFS).
- [`OPERATION_FLOW.md`](OPERATION_FLOW.md) — how one token flows end-to-end.

> **All power/cost/size figures here are [EST]** — modeled, not measured on silicon or a board.
> The single biggest unknown is the **FPGA fit** (unmeasured; yosys abc/KMAX wall) — it sets the
> FPGA class, and therefore the size, thermal, BOM and power. **De-risking it is Phase D0.**

---

## 1. Product definition

**One-liner:** *frontier AI that works with the ethernet unplugged* — plug a USB-C box into the
computer you already own and run the full 753 B model **fully offline / air-gapped**. Nothing
leaves because there's no path out (the audit is literally *"does it still work with the ethernet
unplugged?"* — yes), and no vendor can rate-limit, deprecate, or cut you off — so private and
subscription-free come as the *result*.

| | |
|---|---|
| **Form factor** | small active-cooled external box (external-SSD → mini-PC sized), self-powered, **USB-C data link** to host |
| **What it runs** | the real `zai-org/GLM-5.2-FP8` (753 GB), **bit-exact** to the published model (not a quantized approximation) |
| **Throughput** | **rung-dependent** [EST] — ~5–8 tok/s on the near-term **prove-it** FPGA, ~15–40 on the **funded custom board** (the old flat ~25–40 was this rung-② number); staged by memory bandwidth per [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) |
| **Power** | ~80–110 W at interactive throughput (self-powered; ~30 W throttled) [EST] |
| **Interface** | USB-C (USB 3.2 Gen 2 is ample — only token IDs cross; heavy traffic stays internal) |
| **Host** | thin driver + a local OpenAI-compatible endpoint → existing chat UIs / editors point at it |
| **Target user** | air-gap / offline-mandated users (SCIF, defense-forward, isolated OT/critical-infra, field/edge), privacy-critical individuals, local-AI enthusiasts, cost-heavy power users |
| **What it is NOT** | not a bus-powered dongle, not a multi-user server, not a general computer, not multi-modal |

**Why the form factor fits the architecture.** The heavy traffic (NVMe↔DDR5↔die, ~22 GB/token)
is **entirely internal**; USB-C carries only the text token stream (a few bytes/token) + control.
So USB-C bandwidth is a **non-issue** and the box is genuinely self-contained. The production top
`glm_fp8_system_cdc` **already carries the 2-clock host/USB CDC** (31 tests, token identical across
async clocks) — the device interface was designed in from the start.

**Positioning.** Unlike a Mac Studio (replace your computer) or a consumer GPU (can't hold/stream
753 GB at all), this is an **accessory that adds frontier-model capability to the machine you
already have** — closer in spirit to an eGPU, but sipping desktop-PC power and running a far bigger
model. It creates a new category: an **external frontier-LLM accelerator** (vs. edge-AI USB sticks
like Coral/Hailo, which only run tiny models). And unlike **any** cloud option — including
"secured cloud" (in-VPC/tenant deployment, zero-retention APIs, confidential-computing/TEE
enclaves) — it runs **with the ethernet unplugged**; every cloud variant still needs connectivity
and fails that test, which is what unlocks the air-gap / offline environments others can't serve
(SCIFs, defense-forward ops, isolated OT/critical-infra, field/edge). The moat is the
**combination — offline + full frontier (753 B) + appliance price** — since offline *alone* is
table-stakes (a 70 B laptop model is offline too): the 70 B laptop fails frontier quality, 8×H100
fails price/form-factor, and secured cloud fails the unplugged test.

---

## 1a. Power-on behavior (plug → ready → tokens)

The user experience is close to *"plug in, use it,"* with **a short boot** — not instant-on, and
with **one one-time setup**:

- **One-time provisioning (factory/first setup):** the 753 GB model is written to the internal
  **NVMe SSD** (`ckpt_pack.py` / `flash_layout.py` — committed tool names). Done once; survives power cycles.
- **Every power-on (~1–2 s [EST]):** power → clocks/PLL lock → resets → DDR5 PHY training + NVMe
  init → **`boot_loader` streams the ~28 GB resident set NVMe→DDR5** → its `done` **releases
  inference** → USB device enumerates on the host. **Inference is gated by `boot_loader.done`, not
  by power-on.** (Full condition list + RTL detail: [`OPERATION_FLOW.md`](OPERATION_FLOW.md) §1.)
- **Per request:** the host sends token IDs + position over USB-C; the box streams the demand
  experts from the **NVMe SSD**, runs the model, returns next tokens. **Session KV lives in the box's DDR5** —
  the host only exchanges tokens.

**What crosses USB-C:** only `start` / `prompt_tok` / `start_pos` / `s_len` in and
`next_tok` / `tok_valid` / `busy` / `done` out — token IDs + control. The heavy weight/KV traffic
never leaves the box.

**Two power inputs, one data cable.** Because the box is self-powered (§7), the physical picture is:
a **DC/PD power input** + a **USB-C data cable** to the host. It is not a bus-powered stick; think
"powered external box that appears to the host as a local AI endpoint after a ~1–2 s boot."

**Device-readiness signaling (to build):** the host driver should surface the boot state (booting →
loading resident set → ready) so the app can show "warming up" instead of failing calls before
`done`. (Phase D2.)

---

## 2. Current state (honest baseline)

**Done (the core IP — see [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) "Keep"):**
- Full FP8 datapath (MLA + DSA + 256-expert MoE + MTP), **bit-exact** to the FP8 contract at a
  small-but-faithful slice; next-token argmax matches the golden.
- The memory system (ddr5_xbar, flash_xbar, kv_cache_pager, expert_cache_pf, weight_loader,
  boot_loader), BMC-proven; the batching stack (PE_M on all 4 wrappers, union-skip MoE);
  the **spec ÷K weight-load hardware** (spec_batched_top / spec_chain_top).
  (`flash_xbar` is the **medium-agnostic storage-read fabric** — address→weight-bytes with
  latency hiding — that in the product **fronts an NVMe/PCIe host controller**; the RTL name is
  a committed identifier, and only the NAND-specific backend is swapped — the abstraction, the
  compute die, `weight_loader`, `expert_cache_pf` and `kv_cache_pager` are unchanged.)
- Production top `glm_fp8_system_cdc` with **host/USB + memory + compute CDC** (async-clock token
  identity verified).
- Power/area levers: BFP accumulator (−87.6 % cells), clock-gating (73.75 % idle), die-shrink
  L0/L1, DVFS budget (4–5×, measured), compression 1.34×.
- Energy + BOM **models** ([`LOW_POWER.md`](LOW_POWER.md), [`SYSTEM_SINGLE_PACKAGE.md`](SYSTEM_SINGLE_PACKAGE.md)).
- **Host software scaffold** ([`host/`](../host/README.md)) — a local **OpenAI-compatible server**
  (`/v1/chat/completions`, streaming SSE), the exact RTL host protocol (`aipu_device.py`, mirrors
  `glm_fp8_system_cdc` + the boot-loader-done readiness gate), the **real GLM-5.2 BPE tokenizer** +
  a port of GLM's chat template, and host-side sampling — buildable/testable with **zero hardware**
  against a mock backend (the Phase D2 first deliverable).

**Not done (the gaps to a product):**
- **Real-model full-scale fidelity** (PRODUCT_ROADMAP P1 — *the* gate: the real 753 GB weights must
  produce the real model's tokens end-to-end). **Blocking for everything.**
- **FPGA fit / bitstream** — unmeasured (yosys wall); no placed-and-routed design, no real board.
- **Host software — the real backend + driver** — the API/protocol/tokenizer layer is scaffolded
  (above); what remains is a signed cross-OS **USB-C driver** and a **real backend** (simulator- then
  hardware-backed) + production runtime/scheduler behind the swappable mock.
- **Physical device** — no PCB, no power architecture, no thermal design, no enclosure, no
  certification.

---

## 3. Target product spec (the requirements to hit)

| Spec | Target | Notes |
|---|---|---|
| Model | GLM-5.2-FP8, bit-exact | the differentiator vs quantized boxes |
| Throughput | **rung-②** funded board ≥ 25 tok/s (goal 30–40) single-user; **rung-①** prove-it FPGA ~5–8 [EST] | comfortable interactive on the product rung; staged per [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md) |
| Power (peak) | ≤ 110 W self-powered; stretch ≤ 100 W USB-PD | see §7 power decision |
| Idle power | ≤ 10 W | clock-gating + DVFS |
| Size | ≤ ~1 L enclosure | external-SSD → small-mini-PC |
| Noise | quiet (≤ ~35 dBA) | desktop appliance |
| Interface | USB-C (USB 3.2 Gen 2 min) | + separate DC power in |
| Host OS | macOS + Windows + Linux | signed driver + local server |
| Retail price | ~$2.5–8 k [EST] | FPGA-class dependent (see §8) |
| Model update | field-updatable weights (rewrite the NVMe model store) | new GLM point releases |

---

## 4. The device-specific gap (beyond the RTL track)

The RTL track (PRODUCT_ROADMAP P1–P4) makes the accelerator correct & synthesizable. The **device**
adds these workstreams, which that roadmap only touches lightly:

1. **Power architecture** — ~80–110 W can't be bus-powered; needs a self-powered design or USB-PD
   EPR. (§7)
2. **Thermal / acoustics** — dissipate ~100 W quietly in ~1 L. (§7)
3. **Host software & UX** — driver + local OpenAI-compatible server + onboarding; the difference
   between "a board" and "a product people use." (Phase D2)
4. **Industrial design & enclosure** — the physical box, connectors, LEDs, mounting.
5. **Manufacturing & certification** — DFM, USB-IF, FCC/CE/EMC, safety, yield/binning.
6. **Support & lifecycle** — warranty, RMA, firmware/model updates, docs.

---

## 5. Phased plan (device track — "D" phases, gated)

Each phase has a **GATE**: a go/no-go you must pass before spending on the next.

### Phase D0 — De-risk & measure *(do FIRST; cheap, decisive)*
- **D0.1 Real-model fidelity** — execute PRODUCT_ROADMAP P1.1 (real 753 GB checkpoint → real
  tokens). **Hard prerequisite** — no product until this passes.
- **D0.2 FPGA fit** — take the design through the **vendor flow (Gowin / nextpnr / Vivado)** to get
  real LUT/DSP/BRAM/DDR-PHY utilization → pick the **FPGA class** (this unblocks size, thermal, BOM,
  power). This is the project's #1 unknown; closes the yosys wall.
- **D0.3 Power point** — decide the operating point (§7): ~100 W self-powered "fast" vs ~40 W
  throttled "quiet/small". Drives thermal + PSU + enclosure.
- **D0.4 Bring-up feasibility** — confirm the target FPGA dev-kit has the DDR5 + NVMe (M.2/PCIe) +
  USB-C IO the design needs. *Dev-board vs product split:* the **Tang Mega 138K Pro** (GW5AT-138)
  dev board is a **bring-up / reduced-demo board only** — it has ~32 MB onboard Flash + ~1 GB
  soldered DDR3 [EST], nowhere near the model store — so the **product needs a custom board**
  (GW5AT-138-class + big DDR5 + NVMe via M.2/PCIe). The dev board proves the pipeline at reduced
  scale; the NVMe/DDR5 model store lives on the custom board (Phase D3).
- **GATE D0:** real tokens match the real model **and** the design fits a costable FPGA at a viable
  power point. *If FPGA fit forces a >$8 k data-center card, revisit scope (throttle / smaller
  resident set / cost target).*

### Phase D1 — FPGA prototype bring-up
- D1.1 Port the full-scale RTL to the chosen FPGA; DDR5 + NVMe (M.2/PCIe) + USB-C PHY integration
  (vendor IP, PRODUCT_ROADMAP P3.1).
- D1.2 Boot-load the real model to the NVMe model store (productionize `ckpt_pack.py` /
  `flash_layout.py`); first end-to-end token over USB-C on real hardware.
- D1.3 Measure the **real** tok/s, watts, thermals — replace every [EST] with a number.
- **GATE D1:** the dev-kit emits the real model's tokens over USB-C at ≥ target tok/s within the
  power point. *This is the "it actually works as a device" proof.*

### Phase D2 — Host software stack *(parallel with D1 once tokens flow)*
- D2.1 USB-C device driver (macOS/Windows/Linux), signed; robust enumeration / hot-plug / recovery.
- D2.2 **Local OpenAI-compatible server** — so existing clients (chat UIs, VS Code, Claude Code,
  etc.) point at `localhost` and just work. Tokenizer + sampling params + streaming. *(API surface +
  protocol + real GLM BPE tokenizer + chat template + streaming already scaffolded in
  [`host/`](../host/README.md) against a mock backend; remaining is the real backend + driver.)*
- D2.3 Management app: model load/update, health, thermal/power telemetry, firmware update.
- **GATE D2:** a user installs the app, points their editor at the box, and chats — no CLI.

### Phase D3 — Custom board, power, thermal, enclosure
- D3.1 Multi-layer controlled-impedance PCB (FPGA + DDR5 + NVMe/M.2 + USB-C + power); PRODUCT_ROADMAP P4.1.
- D3.2 Power subsystem: PSU / USB-PD, power domains, DVFS hooks (PRODUCT_ROADMAP P2.4), protection.
- D3.3 Thermal: heatsink + quiet fan curve for the power point; acoustic target.
- D3.4 Industrial design: enclosure, connectors, status LEDs, EMI shielding.
- **GATE D3:** a custom unit hits target tok/s/W in the final thermal/acoustic envelope.

### Phase D4 — DFM, certification, pilot build
- D4.1 Design-for-manufacture, test fixtures, board bring-up automation, yield/binning.
- D4.2 Certification: USB-IF, FCC/CE, EMC, electrical safety.
- D4.3 Reliability qual: temp/voltage/aging, NVMe endurance (read-mostly → benign; writes only on model update), burn-in.
- D4.4 Small pilot production run.
- **GATE D4:** certified units pass reliability + a pilot cohort uses them without support fires.

### Phase D5 — Launch & lifecycle
- D5.1 Packaging, docs, onboarding, warranty/RMA.
- D5.2 Model-update pipeline (new GLM point releases → NVMe model-store rewrite tool).
- D5.3 Support + a channel for architecture-change models (new arch ⇒ RTL update ⇒ bitstream push).

---

## 6. Key decisions / forks (resolve early)

| Decision | Options | Recommendation |
|---|---|---|
| **Power point** | ~100 W fast (self-powered) · ~40 W quiet (throttled, slower) | ship **self-powered ~90 W** (interactive tok/s); offer a quiet/eco mode — **the RTL knob exists** (`clk_throttle` runs the die f/div, byte-identical, [`LOW_POWER.md`](LOW_POWER.md) §4) |
| **Power delivery** | own PSU/barrel · USB-PD EPR 240 W · bundled PD brick | **own DC input** (EPR host+cable support is still rare); USB-C = data |
| **FPGA class** | mid FPGA (~$0.5–2 k) · data-center card (~$3–8 k) | **decided by D0.2 measurement** — the pivotal cost driver |
| **Model updates** | NVMe model-store rewrite tool · sealed | **field-updatable** (GLM ships point releases) |
| **Openness** | closed appliance · open driver/API | **open local API** (drives adoption via existing clients) |
| **NVMe drive grade** | TLC (balanced) · SLC (low-power, $$) · QLC (cheap, slow) | **TLC** NVMe bulk; SLC only if power ≫ cost ([`LOW_POWER.md`](LOW_POWER.md)) |

---

## 7. Power & thermal (the hard constraint)

Estimated **~80–110 W** at interactive throughput ([`LOW_POWER.md`](LOW_POWER.md)) — **above USB
bus power**, so:

| USB-PD tier | budget | verdict |
|---|---|---|
| USB-C default | 15 W | ✗ |
| PD SPR | 60 / 100 W | △ only if throttled to ≤100 W, zero headroom |
| PD 3.1 EPR | 140–240 W | ✓ headroom, but needs a compatible host **and** cable (rare) |

Also, a laptop port cannot both source ~100 W **and** have the device consume it. **Conclusion: a
self-powered box (own DC/PD input) with USB-C as the data link** — the eGPU / NAS model. Thermal:
~100 W dissipated quietly in ~1 L needs a heatsink + a tuned fan; a **~40 W throttled mode** enables
smaller/quieter builds at lower tok/s. Power breakdown is **memory-dominated** (NVMe + DDR5
~60–70 %); the compute die is ~20–30 % and mostly gated.

---

## 8. BOM & pricing [EST]

*Canonical per-rung BOM + per-seat economics: [`BOM.md`](BOM.md). Summary below.*

| Component | Rung ① prove-it (budget FPGA) | Rung ② product (custom board) |
|---|---|---|
| FPGA (largest uncertainty) | ~$0.5–1.5 k (low/mid, e.g. KU3P / GW5AT-class) | ~$3–8 k (Versal / Agilex / HBM-class) |
| Fast DDR tier | ~$150–300 (**DDR4** ~4 ch, near-term) | ~$300 (**DDR5** multi-ch / HBM, 64–128 GB) |
| NVMe SSD ~1–2 TB (M.2 / PCIe Gen3–4 x4) | ~$100–250 (1 drive) | ~$300–500 (2–4 TB or multi-drive) |
| PCB / PSU / controllers / enclosure | ~$150–300 | ~$300 |
| **BOM total** | **~$1.0–2.4 k** | **~$4–9 k** |
| **Indicative retail (≈2× BOM)** | **~$2.5–5 k** | **~$8–16 k** |

**Rung mapping ([`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)):** the two paths are the ladder's first two
rungs — **rung ①** the ~$1–2 k prove-it demo box (budget FPGA + DDR4, the near-term build) and **rung ②**
the funded product (custom board, DDR5/HBM) — the ladder's ~$3–6 k box, up to ~$4–9 k for the
HBM / data-center-card variant. **Rung ③ — a SoC/ASIC at manufacturing volume** — is the cost endgame:
HBM stacks + many-channel PHY + near-memory FP8 compute at ~TB/s, with **lower $/seat and lower power**
once the multi-million NRE amortizes over volume. Not now (no volume, no capital); sequenced *after* the
FPGA rungs prove product-market fit — the same verified RTL on every rung.

**Cost insight:** power is memory-dominated but **cost is FPGA-dominated** — the FPGA class (set by
D0.2) is the pivotal BOM lever. Algorithmic levers (spec ÷K, compression, DVFS) improve
power/throughput **without** touching BOM.

**vs the alternative:** an 8×H200 cloud/cluster is ~$250–300 k capital + ~6–10 kW; this box is
~50–100× cheaper capital and ~50–70× lower power — the trade is single-user throughput. And for an
air-gap / offline buyer the cloud isn't on the menu at any price: even a "secured cloud" (VPC /
zero-retention / TEE) needs connectivity and fails the unplugged test.

---

## 9. Risks & mitigations (ranked)

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | **FPGA fit unknown** — may need an expensive data-center card, breaking the cost story | high | **D0.2 measure first**; die-shrink levers ([`MINIATURIZATION.md`](MINIATURIZATION.md)) + compact config; throttle/scope if needed |
| 2 | **Real-model fidelity not yet closed** — no product until real weights → real tokens | high | D0.1 (PRODUCT_ROADMAP P1.1) is the very first gate |
| 3 | **Power > USB-PD** | med | self-powered design (§7); accept it's not a bus stick |
| 4 | **Host software effort** — driver + server + cross-OS is real work | med | OpenAI-compatible API to reuse the existing client ecosystem |
| 5 | **Thermal/acoustics** in a small box | med | power-point choice; eco mode; proven eGPU-class cooling |
| 6 | **Model architecture churn** — a new model arch needs RTL changes | med | field weight-updates for point releases; RTL update path for arch changes |
| 7 | **Certification / manufacturing** cost & time | med | budget D4; pilot before scale |
| 8 | **Real watts/tok-s differ from [EST]** | low-med | D1.3 replaces every estimate with a measurement |

---

## 10. Rough timeline [EST] (small team; parallelizable)

| Phase | Scope | Rough duration |
|---|---|---|
| D0 | fidelity + FPGA fit + power point | 1–3 months |
| D1 | FPGA prototype, first real tokens over USB-C | 3–6 months |
| D2 | host software (parallel with D1/D3) | 3–6 months |
| D3 | custom board + power + thermal + enclosure | 6–9 months |
| D4 | DFM + certification + pilot | 6–12 months |
| D5 | launch + lifecycle | ongoing |

**Verified RTL → shipping consumer device: ~18–36 months**, dominated by board/cert/mfg, not the
RTL. D0 is cheap and decisive — **do it before committing to the rest.**

---

## 11. Immediate next steps (this quarter)

1. **D0.1** — run the real-checkpoint full-model fidelity check on a GPU host (PRODUCT_ROADMAP P1.1).
2. **D0.2** — get the design through a vendor FPGA flow for a real utilization number → FPGA class.
3. **D0.3** — pick the power point (~90 W self-powered recommended) and draft the thermal envelope.
4. Wire the **host-side local OpenAI-compatible server** ([`host/`](../host/README.md), already
   scaffolded against a mock backend) to a simulator-backed backend now (no hardware needed) so the
   software is ready when D1 tokens flow.

---

## Status
- **This plan: drafted** (device track, complements the RTL track in PRODUCT_ROADMAP.md).
- **Gating unknowns:** real-model fidelity (D0.1) + FPGA fit (D0.2) — both must close before board spend.
- **Everything downstream** (size, thermal, BOM, price, timeline) is bounded by the FPGA class D0.2 returns.
