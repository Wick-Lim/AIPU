# Quant / prop-trading outreach kit — the co-lead wedge

*The full BD motion for the **quant / prop-trading / hedge-fund** wedge — the co-lead tested in parallel
with legal ([`ICP.md`](ICP.md), [`ICP_OUTREACH_KIT.md`](ICP_OUTREACH_KIT.md)). Same goal: **Gate 3**,
one signed design-partner LOI. Quant is the **fastest** wedge to that first LOI — but it has its own
honest limits as a first customer. Both are below.*

> **Honest stance (same as the legal kit).** No shippable box yet (assembled-model fidelity vs the real
> GGUF + board bring-up/demo are open; the FPGA fit itself is measured — routed on XCKU3P at 46.5 MHz).
> This is a **design-partner / validation** motion, not a sale: we're building it, and we want a
> few funds that actually have the constraint to shape it and sign a **non-binding LOI to pilot when the
> demo lands.** Never imply a finished product — quant buyers are the most skeptical of hype in the
> market; overclaiming loses them instantly.

---

## 0. Why quant is the co-lead — and its honest limits as a first customer

**Why it's the fastest wedge:**
- **Small teams, fast decisions, deep pockets** — a box is a rounding error; they pay for any edge.
- **FPGA-native culture** — they already run FPGAs in the trading path, so a custom FPGA appliance is
  *familiar infrastructure*, not an exotic risk. **The founder's RTL/FPGA background is a genuine
  warm-intro edge** into this community (they speak the same language — see §2).
- **Extreme data secrecy is existential** — a leaked signal / position / strategy = lost alpha. They
  will *not* put proprietary research or data in any cloud. That's the exact constraint.
- **They value the engineering** — a formally-verified control plane, a Q4_K compute core bit-exact to
  the ggml reference, and an FPGA-native datapath land with their infra/security people in a way they
  won't in most segments.

**Honest limits (say these to yourself before you pitch):**
- **Big funds already run on-prem GPU clusters.** For a top-tier fund with an ML-infra team, "run a
  private LLM" is *solved* — the box competes with "run it on our own GPUs," not just "vs cloud." The
  box wins clearest for (a) **per-desk / air-gapped** deployment, (b) funds **without** a big GPU/MLOps
  team who want it **turnkey**, (c) footprint / power / **per-seat** economics. Be honest where their
  cluster already serves it.
- **It is NOT the trading hot path.** Interactive research speed — measured-proxy design points [EST]
  span ~0.5–1 tok/s (NVMe-only prove-it rung) to ~13–47 on the funded custom board, ~54–127 with a
  larger DRAM cache ([`H_MEASUREMENT.md`](H_MEASUREMENT.md); staged hardware, see
  [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)) — **not microseconds**. Never let them think it's a
  trading accelerator.
- **It is NOT "alpha in a box."** It doesn't generate signals. It lets researchers use a frontier model
  on data they currently can't. Sell **confidentiality-enabled research productivity**, not edge-in-a-box.
  Quant people respect precision and will end the call the moment you overclaim.

---

## 1. Target — the account and the three people

**Account fit (all three):**
- **Proprietary data/research they won't cloud** — signals, positions, strategies, code, internal
  research (every serious fund).
- **Wants frontier-LLM leverage on it** — parsing filings/earnings/news at scale, code on strategies,
  an internal research assistant / Q&A over proprietary docs.
- **No turnkey private-753B setup yet, OR wants per-desk air-gap** — i.e. the box adds something over
  "we already serve Llama-70B off our cluster."

**Best first targets:** mid-size hedge funds & **prop shops** (fastest-moving, flattest, FPGA-native),
and quant desks under information-barrier constraints. The mega-funds (Citadel/DE Shaw/Two Sigma class)
have the infra solved and slower procurement — pursue via the infra community, but expect the mid-tier
to sign first.

**The three personas:**

| Persona | Title patterns | Cares about | Role |
|---|---|---|---|
| **Champion** | Head of Research Engineering / Quant Dev / ML Infra / CTO; a research-engineer running an internal LLM POC | Shipping a private frontier-LLM capability without an MLOps project or a leak | **Entry point** |
| **Sponsor** | Portfolio Manager / desk head / Head of Research | Research throughput on data they can't currently touch with AI | **Edge owner** |
| **Approver** | CISO / Head of Infosec (already colo/air-gap/FPGA-savvy) | "Nothing leaves the network," auditable non-egress | **Unblocker — loves the air-gap + formally-verified control plane** |

**Where to find them:** quant/HFT **infra people are active + technical on LinkedIn/Twitter/X**; firms
hiring "LLM / ML-infra engineer" or "research engineer, NLP" (a tell they're building this); the
**FPGA / low-latency-infra community** (conferences, meetups, open-source HFT/FPGA circles) — this
overlaps directly with the founder's world, the **single best warm-intro channel** for this wedge.

---

## 2. The message (translated for quant — terse, technical, zero hype)

> **"Your researchers want frontier LLMs on your proprietary data — filings, news, positions,
> strategies, code — but that data can't touch a cloud, and standing up a private 753B stack is an
> MLOps project. We're building an appliance that runs a full frontier model **entirely inside your
> network** — nothing leaves, runs air-gapped, turnkey, per-desk. And it's built like your infra:
> **FPGA-based, a formally-verified control plane, and a Q4_K compute core bit-exact to the ggml
> reference** — so your infosec team can audit that it can't exfiltrate and verify the numeric core
> against a reference themselves."**

The infra/security persona is the one to win, and this wedge is the one that *rewards* leading with the
engineering: **FPGA datapath + formally-verified control plane + a bit-exact Q4_K compute core + the
unplugged-ethernet test** all speak their language. (In every other segment the tech is back-pocket; here it's a front-door credibility hook.)

---

## 3. Cold outreach (short, technical — quant people delete hype on sight)

### Email A — to the Champion (research eng / ML-infra lead)
> **Subject: full 753B, air-gapped, on your own network**
>
> [Name] — I'm building an appliance that runs a full frontier LLM (753B) **entirely inside your
> network** — nothing leaves, runs with the ethernet unplugged. It's for using frontier models on data
> that can't go to a cloud (research, filings, positions, code) **without** standing up your own 753B
> GPU stack + MLOps to do it. FPGA-based, formally-verified control plane, Q4_K compute core bit-exact
> to the ggml reference.
>
> Picking a few quant design partners to shape it. 15 min? I mostly want to learn how you run models on
> confidential data today.

### Email B — to a Sponsor PM / desk head
> **Subject: LLMs on your research — nothing leaves the building**
>
> [Name] — do your researchers hit a wall using frontier AI on anything confidential (signals,
> positions, filings, strategies) because the good models are all cloud? I'm building an on-prem
> appliance that runs a full frontier model air-gapped, per desk. Selecting a few quant design partners
> — worth 15 min to pressure-test whether it's real for your desk?

### LinkedIn / X DM (very short)
> Building an appliance that runs a full frontier LLM (753B) **inside** a fund's network — air-gapped,
> nothing leaves — for models on data that can't touch a cloud. FPGA-based, formally-verified control
> plane. Picking a few quant design partners. Open to 15 min?

### Warm intro (via the FPGA / quant-infra network — the strongest channel)
> Would you intro me to [Name] at [Fund]? I'm building on-prem frontier AI on an FPGA — runs a full 753B
> model air-gapped inside the fund's network — and I want a quant design partner who won't put research
> in a cloud. We're both FPGA people; one line from you would carry a lot. Blurb attached.

### Follow-ups (terse; two, then stop)
- **Nudge 1:** *"One line is enough: is 'frontier LLMs on data that can't leave your network' a real
  need, or already solved on your own cluster? Either answer helps me."*
- **Nudge 2:** *"Last note — [concrete proof: the Q4_K compute core is bit-exact to the ggml reference,
  and the whole-chip design is placed-and-routed on a real FPGA at a measured 46.5 MHz — compute and
  fit risk retired, board bring-up is the next milestone]. If it's not relevant I'll stop; if it is,
  15 min: [link]."*

---

## 4. Discovery call — script (15–25 min; quant calls are short)

**Frame (30s):** *"Not selling — there's no box yet. I'm building it and learning from funds that have
the constraint, to find a couple of design partners. Mostly want to ask how you run this today."*

**Qualify (ask, don't pitch):**
1. "Where do your researchers use LLMs — and where are they *not allowed* to, because of what data it is?"
2. "Do you run models **on-prem** today? On your own GPU cluster? Who stands it up and maintains it?"
   *(This decides whether the box adds anything — probe honestly.)*
3. "What would frontier-quality-on-confidential-data actually unlock — research throughput, code,
   filings/news at scale, internal Q&A over proprietary docs?"
4. "Any **isolated / air-gapped** research environments or desks? Would a per-desk box matter?"
5. "How do you buy infra — capex per box? Who signs — infra, infosec, the PM?"

**Green flags:** a hard secrecy mandate on research/signals · **no** turnkey private-753B today · a
per-desk air-gap need · an infra/infosec lead who lights up at the FPGA/verified-RTL angle. **Red
flags:** "we already serve Llama-70B off our cluster, it's fine" (capability solved — deprioritize
unless per-desk air-gap or turnkey-753B matters) · wants microsecond latency (wrong product) · expects
a **signal generator** ("does it give alpha?").

**The honest positioning in the call (say it plainly):** *"This is a confidentiality-enabled research
tool — a frontier model on data you can't cloud. It is not the trading hot path (measured-proxy design
points [EST]: ~0.5–1 tok/s NVMe-only on the prove-it rung, ~13–47 on the funded custom board —
wall-clock unmeasured until a board runs; not microseconds), and it does not generate signals. I won't
oversell it."* Precision **is** the sell here.

**The ask:** *"Would [Fund] be a design partner? A couple of technical sessions on your infra + security
requirements, and a short non-binding letter that you'd pilot it when the air-gapped demo is ready. No
cost — you get first access and shape it around how you actually run infra."*

### Objection handling (quant-specific)

| They say | You say (honest) |
|---|---|
| "We already run models on our own GPU cluster." | "Then you've solved the capability — so the box is only interesting for (a) **per-desk air-gapped** deployment, (b) skipping the 753B quantize/serve/MLOps for a **turnkey** full-frontier box, (c) footprint/power/cost per seat. If your cluster already serves 753B privately to every desk, you may not need us — I'd rather know that now." |
| "Is it fast enough?" | "For research/analysis the funded board is — **~13–47 tok/s at its design points, up to ~54–127 with a larger DRAM cache; the near-term NVMe-only prove-it rung is ~0.5–1** ([EST] roofline on measured-proxy h/U — [`H_MEASUREMENT.md`](H_MEASUREMENT.md); wall-clock unmeasured until a board runs; staged hardware — see [`HARDWARE_LADDER.md`](HARDWARE_LADDER.md)), interactive. It is **not** the trading hot path, not microseconds. Different tool entirely." |
| "Does it give us edge / alpha?" | "It doesn't generate signals. It lets your researchers use a frontier model on data they currently can't — a productivity + secrecy edge, not a signal. I won't pretend otherwise." |
| "Why not a zero-retention API or an on-prem vendor model?" | "Both still need a connection. The box runs **disconnected** — the audit is 'does it work with the cable unplugged?'. Zero-retention/VPC/TEE all fail that; if that bar doesn't matter to you, you don't need us." |
| "How do we trust it doesn't exfiltrate?" | "There's no path out — it's air-gapped — and that non-egress is exactly the part your infosec team can audit directly. The datapath is a **formally-verified control plane** (BMC + k-induction) with a **Q4_K compute core bit-exact to the ggml reference**, so they can verify the numeric core too; matching the published GGUF end-to-end is validation we're still finishing, and I'll say so." |
| "You're pre-product." | "Yes — that's why I want you now. Design partners shape it and get first access; the tech risk is unusually retired — formally-verified controllers, a Q4_K compute core bit-exact to the ggml reference, and a whole-chip design placed-and-routed on a KU3P-class FPGA at a measured 46.5 MHz. The bounded remaining work is board bring-up and end-to-end fidelity vs the real GGUF. I'm asking for your requirements + an LOI, not a purchase." |

---

## 5. The design-partner offer

**They give:** 2–3 technical sessions (infra + security requirements, use cases), a signed non-binding
LOI, a named infra/infosec contact to define the air-gap audit, willingness to pilot when the demo lands.
**They get:** direct product influence (built around how *they* run infra), first access / priority
allocation, design-partner pricing, a head start on private frontier AI. **No cost** at this stage.

## 6. LOI

Use the one-page non-binding template in [`ICP_OUTREACH_KIT.md`](ICP_OUTREACH_KIT.md) §6 — swap **"Firm"
→ "Fund"** and **"privileged legal work" → "proprietary research / data"**. Keep it non-binding to lower
the bar to signature.

## 7. Cadence & the one metric

- **Target list:** ~15–25 accounts (the quant universe is smaller than legal — quality + warm intros
  over volume). **Lead with the FPGA/quant-infra warm-intro network** — the founder's edge here.
- **Outreach:** terse, technical, ~5–8 touches/week; Champion (infra/eng) first.
- **Goal:** book ~4–6 discovery calls → **1 signed design-partner LOI.** Quant's job in the two-wedge
  test is to get there **fastest**; that first LOI + the measured FPGA demo is the pre-seed package
  ([`ICP.md`](ICP.md)). Run it **in parallel with the legal kit**; let the reply rate crown the beachhead.
- **Disqualify fast.** "Solved on our cluster / no per-desk air-gap need" is a clean no — log it, move on.
