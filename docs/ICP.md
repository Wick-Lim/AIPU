# ICP — who buys the local 753B box first

*The ideal customer profile for the product defined in [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) /
[`USBC_PRODUCT_PLAN.md`](USBC_PRODUCT_PLAN.md): a **local, single-user appliance** running a full
frontier open-weight model (GLM-5.2-FP8, 753B) on-premise — private, offline, no per-token fees,
~25–40 tok/s [EST] interactive, **one box / one seat** (B=1).*

---

## The intersection that defines the buyer

The box only wins for someone who needs **all three** at once:

1. **Frontier-scale quality** — a 7–70 B model that fits a laptop/Mac is *not good enough* for their task.
2. **Data cannot leave the premises** — a legal / regulatory / contractual / sovereignty bar on cloud LLM APIs.
3. **Appliance, not infrastructure** — they want to *buy/plug a box*, not build and run a GPU cluster + MLOps.

Miss **any one** and a cheaper option dominates: drop (1) → a Mac/GPU running Llama/Qwen 70 B; drop
(2) → the cloud API (cheaper, faster, zero capex); drop (3) → a private GPU build. **The ICP lives only
in the triple intersection.** Everything below is about finding the segment where that intersection is
*acute, reachable, and funded.*

---

## The ICP — defined by the **constraint**, not the vertical

> **A confidentiality-bound team that needs frontier-model quality on data it legally or contractually
> cannot send to a cloud LLM — in *any* industry.** That constraint, not a job title or a sector, is the
> ICP. The **product is horizontal**; what must be narrow is the **first go-to-market wedge**.

**Don't confuse the market with the wedge.** The need is horizontal — legal, quant/finance, IP-heavy
R&D, government, healthcare all share the exact triple intersection above. Constraining the *product* or
the *pitch* to one sector would be a mistake. But a pre-seed team can only build **one** message, learn
**one** buyer, and produce **one** reference at a time, so the *first sales motion* must be narrow —
otherwise you get five half-run motions and zero references. **Pick the first wedge empirically**, by
`speed-to-first-LOI × acute pain × ability to pay × reference value`, and **run the top two in parallel
for ~2 weeks** — let the reply rate crown the beachhead rather than betting blind.

**The two co-lead wedges to test first:**

| | **Legal** (law firms / corp legal) | **Quant / prop-trading / hedge funds** |
|---|---|---|
| Pain | Privilege + client confidentiality bar cloud AI | Won't leak signals/positions/research to any cloud |
| Frontier quality needed? | Yes — hard drafting/analysis | Yes — dense research/filings/code |
| Pays per-seat? | Yes — $100–500+/seat/mo tooling culture | Yes — deep pockets, pays for any edge |
| Speed to sign | Weeks–months (innovation/KM buyer) | **Fastest** — small, fast decisions; **FPGA-native** already |
| Market size | **Largest** | Smaller seat count |
| Reference value | High within legal | High within finance |
| The trade | biggest market, more conservative | fastest first LOI, narrower |

Both are strong; they're strong *differently* (legal = biggest, quant = fastest). Testing both de-risks
the beachhead bet. The rest of this doc uses **legal as the worked example** (and the
[`ICP_OUTREACH_KIT.md`](ICP_OUTREACH_KIT.md) instantiates it), but every section below is
constraint-driven and re-instantiates for whichever wedge pulls — swap only the buyer persona + vocabulary.

**The legal wedge in detail** (the worked example):

| Fit test | Legal |
|---|---|
| Frontier quality needed? | **Yes** — contract analysis, privilege review, drafting, case reasoning are hard; 70 B models visibly under-perform. |
| Data barred from cloud? | **Absolutely** — attorney–client privilege + client confidentiality mandates; many clients contractually forbid cloud AI on their matters. |
| Buys per-seat tools already? | **Yes** — firms pay **$100–$500+/seat/month** for Westlaw/Lexis/Harvey/etc. A per-seat AI appliance fits the existing budget line. |
| Work is text/document-heavy? | **Yes** — the exact modality an LLM is best at. |
| Procurement speed | **Weeks–months** via an innovation/KM partner — faster than gov/health. |
| Willingness to pay | **High** — legal bills at $300–$1000+/hr; even small time savings justify a box. |

**The legal buyer & champion:** the firm's **Director of Innovation / Knowledge Management / Practice
Technology** (the person already piloting "private AI"), sponsored by a **practice-group partner** who
feels the confidentiality pain, with **IT/Security** as the approver (they *love* "nothing leaves the
building").

## Why they can't solve it today

- **Cloud API (GPT/Claude/Gemini):** *banned or restricted* for privileged/client-confidential matters —
  data leaves the premises; many client engagement letters forbid it outright.
- **A smaller on-prem model (Llama/Qwen 70 B on a workstation GPU):** *quality gap* on hard legal
  reasoning, **and** it's still a GPU box someone has to stand up, quantize, patch, and babysit.
- **Self-host the real 753 B on GPUs:** **8×H100 ≈ $250–400 k capex** + kW-scale power + an MLOps team —
  absurd overkill for a handful of confidentiality-bound seats; it's a *datacenter build*, not a product.
- **Do nothing:** they leave frontier AI on the table for exactly their **highest-value, most sensitive**
  work — the work they most want leverage on.

**Our wedge (the one-liner):** *the only turnkey appliance that runs a **frontier-scale** model
**locally**, at a **per-seat price**, not a datacenter price.*

## Economic case (per seat)

| Option | Frontier quality? | Data stays local? | ~Cost to put on one confidential desk |
|---|---|---|---|
| Cloud frontier API | ✅ | ❌ (disqualifies) | ~$20–200/mo — *but not allowed* |
| Mac/GPU + 70 B local | ❌ (quality gap) | ✅ | ~$3–6 k one-time |
| 8×H100 private 753 B | ✅ | ✅ | ~$250–400 k + power + MLOps (shared, not per-seat) |
| **This box (target)** | **✅ (753 B)** | **✅** | **target BOM in the low-$k's + a per-seat SW/support subscription** |

The pitch is **not** "cheaper tokens" (the cloud API is cheaper if you're *allowed* to use it). It's
**"the only way to give this specific professional frontier-model leverage on data they cannot put in
the cloud, at a desk price."** The buyer is paying for **access under a constraint**, not for throughput.

> Honest gate: the box's BOM (data-center-class FPGA + 1 TB Flash + 64 GB DDR5 + board) must land at a
> *per-seat-defensible* price. That number is exactly what the **FPGA-fit measurement**
> ([`../fpga/`](../fpga/README.md), the parallel track) decides — it sets the FPGA class, which sets the
> BOM. **The ICP is only real once that BOM lands in the low-$k's per seat.**

## The pilot (first engagement — aim for a signed design partner in ~90 days)

1. **One firm, one practice group, 1–3 boxes** on their network (or a loaner), each a single attorney's
   confidential-AI desk. No multi-tenant serving (that's the non-target datacenter regime).
2. **Their documents, their matters** — the whole point is the data never leaves; run it on a real
   matter they'd otherwise never touch with cloud AI.
3. **Success metric = "would you pay per-seat for this?"** measured on *quality on confidential work +
   the confidentiality guarantee*, not on tok/s. (25–40 tok/s is comfortably interactive; speed is not
   the sale — *provable locality* is.)
4. **Deliverable that unlocks the pilot:** a working box (or even the compact FPGA config) that (a) runs
   real GLM-5.2 weights locally and (b) can be **audited to prove no network egress** — security's
   checkbox is the close.

## Wedge menu & expansion ladder

Quant is a **co-lead wedge tested up front** (see the two-wedge table above), not a fallback. The rest
are the **expansion ladder** — sequence them after the beachhead proves out, don't run them all at once.

| Segment | Why it's strong | When / caution |
|---|---|---|
| **Quant / prop-trading / hedge funds** | Fastest-moving buyers, **FPGA-native culture**, deep pockets, extreme data secrecy (won't leak signals/positions), love any edge | **Co-lead — test in parallel with legal from day one.** Fewer seats/account, but the fastest path to a first LOI. |
| **IP-sensitive R&D** (pharma, semiconductor, defense-adjacent, proprietary-code shops) | Genuinely un-cloudable secrets (molecules, chip designs, source); real budgets | Second wave — longer eval cycles; more heterogeneous use cases to support. |
| **Government / defense / intelligence** | Highest willingness to pay; air-gap is *mandatory*; sovereignty tailwind | **Slow procurement (12–24 mo)** + a **GLM = Chinese-origin model** problem for US federal (needs an RTL retarget to a Western model first). High-value, wrong *first* customer. |
| **Healthcare / clinical** (PHI/HIPAA) | Huge market, absolute privacy mandate | Very risk-averse, long procurement, heavy validation burden. |
| **Sovereign-AI / non-US-aligned orgs** | GLM's Chinese origin is an *advantage*; model-sovereignty tailwind | Fragmented, relationship-driven, geopolitically complex. |

## Disqualifiers (who is **NOT** the ICP — stay focused)

- **Anyone fine with the cloud API** — that's most of the market; the API is cheaper/faster. Don't chase.
- **Anyone happy with a 70 B local model** — a Mac/GPU serves them cheaper; we only win when frontier
  quality is required.
- **Multi-user / high-QPS serving** — that's the **non-target datacenter regime** (per-user ~0.14 tok/s
  batched; see [`ULTRA_PERF.md`](ULTRA_PERF.md) §4). The box is **one seat**. Selling it as a shared
  server breaks the whole value prop.
- **US federal / defense that cannot run a Chinese-origin model** — parked until the RTL is retargeted to
  a Western frontier open-weight model (a real, but later, RTL effort).

## The two things that make this ICP investable (both in flight)

1. **A measured FPGA fit → a real per-seat BOM** (the [`fpga/`](../fpga/README.md) D0.2 track). Without it,
   "low-$k's per seat" is a hope, not a price.
2. **One signed design partner from the primary segment** — a single law firm innovation team that says
   *"yes, we'd pay per seat for a provably-local frontier box."* One real LOI > any spec sheet. The
   copy-paste BD motion to land it — targeting, cold outreach, discovery script, the design-partner
   offer, and an LOI template — is in [`ICP_OUTREACH_KIT.md`](ICP_OUTREACH_KIT.md).
