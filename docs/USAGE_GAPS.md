# Usage-Gap Register

**The gap between a verified accelerator and a usable device.**

> One-line truth: the RTL is verified; the software a user actually touches
> (host RAG / GUI / visualization / tuning / persistence, the USB-C transport,
> multi-context routing, caching, and real provisioning) is largely **unbuilt**.
> Several lifecycle and safety decisions are architecture — cheap to decide now,
> expensive to retrofit after board/boot-loader/protocol freeze.

This register turns the findings of a structured usage review into a durable
planning artifact. The review ran **63 agents** across the seven usage
dimensions a real owner passes through — provisioning, host software,
interactive session quality, multi-context, power/thermal/physical, the
flagship air-gapped RAG/GUI/viz workflow, and reliability/failure modes — and
**confirmed 49 problems** (37 high, 11 medium, 1 low) with reproducible
evidence.

This is **product-stage reality, not a regression.** The datapath, KV pager,
boot DMA engine, ECC/reset/MBIST building blocks, and formal properties are
strong and stay strong. What the review found is that the *experience layer*
wrapped around that silicon — the part a buyer plugs in and touches — has not
been built yet, and that a handful of decisions underneath it need to be locked
before hardware and protocol freeze. The public investor page already frames
this honestly; this document is the engineering-facing companion to that framing.

Scope note: many findings are the *same underlying gap* seen from different
usage angles (the ~70 s cold boot and the absent RAG/GUI each surfaced in four
or five dimensions). Those are **stated once and cross-referenced**, so the 49
raw findings collapse into a smaller set of distinct gaps below.

---

## LOCK IN NOW — architecture decisions, expensive later

Three decisions are not software backlog. They shape the board, the
boot-loader, and the USB wire format. If they are deferred past freeze, fixing
them means a hardware or protocol respin. Decide them before that freeze.

> **Progress (2026-07): all three now have a verified RTL/tool foundation**
> (parameter-gated, default-off, the default netlist proven byte-identical by
> yosys sequential equivalence — so nothing verified was disturbed):
> **§A** — `boot_loader` gained an `INTEGRITY` mode (magic/version/CRC manifest
> gate; fail-closed on truncated/wrong-version/bad-CRC/bad-magic — `done` never
> releases a bad model): `make boot-integrity` (3712 tests + equivalence). A real
> streaming provisioner, `tools/provision_image.py`, now packs a real GGUF into a
> binary block image + signed manifest (per-tensor sha256 + resident-hot/expert
> segment list a boot-loader can consume), proven on real GGUFs: `make
> provision-selftest`. *(Still open: the A/B dual-slot + atomic activate-pointer
> policy is a board/firmware decision on top of this foundation.)*
> **§B** — `weight_loader_q4k` gained a `WEIGHT_ECC` SECDED mode (single-bit
> corrected, double-bit flagged, corrected-error counter for scrub): `make
> weight-ecc` (+ equivalence). *(Still open: wiring the scrub loop + check-bit
> storage into the physical LPDDR array.)*
> **§C** — `glm_q4k_system_cdc` gained a `PROTO_CTX` mode carrying a
> context/sequence id end-to-end through the CDC FIFOs + a telemetry-counter
> readback: `make cdc-protocol` (+ equivalence). *(Still open: full N-context
> scheduling in the core and the host-side multiplexer.)*

### A. Provisioning A/B dual-slot + boot-time integrity/version check — *brick prevention*

**Decide:** two model slots on NVMe (active / staged), an atomic
active-pointer flip, rollback to the last-good slot, and a boot-time
verify (hash/CRC + version + resident-set descriptor match) that gates
inference on "model present AND valid AND correct version."

**Why now:** today a model update is an undefined, all-or-nothing **467 GB
single-copy rewrite** — a power blip mid-write permanently bricks the box, with
no rollback and no version visibility (`docs/USBC_PRODUCT_PLAN.md:218,300,312`;
`docs/OPERATION_FLOW.md:298-299`). Worse, boot is a **raw block move with no
verification** — `src/boot_loader.v:14-18` copies NVMe→DRAM and
`:270-275` raises `done` purely on a word count, so a partial or wrong image
boots "ready" and returns confidently wrong output forever
(`docs/OPERATION_FLOW.md:89-90`; `host/aipu_device.py:9-14` mirrors only
start/busy/done/tok — no model-validity state). Slot layout, the spare NVMe
capacity for a second slot, and the boot-loader's verify/rollback FSM are all
**physical/boot-ROM commitments**. They cannot be added by a host update later.

### B. ECC + scrub on the resident ~467 GB weights

**Decide:** SECDED (or better) plus a background scrubber on the always-on
weight array, and the DRAM-retention / fast-resume story that goes with it.

**Why now:** the production top instantiates the KV pager with the **ECC param
omitted → ECC=0** (`src/glm_q4k_system.v:676-688`), and `grep 'ecc|secded|scrub'
src/ddr5_xbar.v` is empty. A 467 GB array that stays powered for the life of the
device has **zero bit-flip protection** — silent, undetectable weight rot that
degrades answers with no error and no logged event
(`docs/PRODUCT_ROADMAP.md:99`, P2.1 "Remains: DDR5/NVMe payload-byte ECC"). ECC
changes the memory width, the controller, and the scrub scheduling — it is a
**silicon/board decision**, not a firmware patch. The verified ECC RAM /
reset-sync / MBIST blocks already in-tree (commit `a3e0d3c`) are the raw
material; the decision is to *wire them into the resident-weight path*.

### C. USB/host protocol context/sequence-id + a power/telemetry channel

**Decide:** the on-wire frame carries a **context/sequence id** and a **sequence
number**, and reserves a **telemetry/control channel** (power/thermal readback,
eco-mode knob, model-update status, spec-chain accept rate).

**Why now:** the shipped host port carries only `{prompt_tok, start_pos, s_len}`
→ `{busy, tok_valid, next_tok}` (`src/glm_q4k_system_cdc.v:198-205,378`;
`host/aipu_device.py:12,154`). With no context id, the host **physically cannot
route tokens to N contexts** — multi-context is impossible end-to-end no matter
what the RTL can do, and the pager's `append_seq`/`gather_seq` ports
(`src/kv_cache_pager.v:112-117`) are never driven by the CDC top. With no
telemetry field, the promised **GUI power tuning has no wire** to `clk_throttle`
(the eco knob is RTL-built but unreachable — `docs/OPERATION_FLOW.md:105-108`,
`src/clk_throttle.v:38`), and throttle/thermal collapse is invisible to the
user. A wire format is frozen once host, firmware, and CDC RTL agree on it;
retrofitting fields after freeze breaks every deployed unit. **Add the fields
now, even if the endpoints that consume them ship later.**

---

## Findings by theme

Tags: **[LOCK-IN-NOW]** architecture, decide before freeze · **[SOFTWARE-TRACK]**
host/embedded software to build · **[DESIGN]** RTL/system design point to resolve ·
**[DOC-FIX]** inconsistency or under-communication to correct.

### 1 · Provisioning, boot, and updates

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **No real provisioning pipeline.** The packer has only ever processed synthetic data and emits a simulation `.hex` (one word/line), not an NVMe image (`tools/ckpt_pack_q4k.py:15-18,409-412`; real-GGUF end-to-end OPEN per `docs/OPERATION_FLOW.md:339-341`). | No delivered means to build a shippable unit; the "plug in and use it" first-run has no working step behind it. | high | [SOFTWARE-TRACK] |
| **Boot resident-set descriptor is never generated** — a gap between the packer's word-offset manifest and the physical segment table `boot_loader` consumes (`src/boot_loader.v:84-88,68-73`; `tools/ckpt_pack_q4k.py:391-397`; no descriptor generator in `tools/`). | The verified boot-loader cannot actually be driven for the real model. | high | [LOCK-IN-NOW] → §A |
| **Update = undefined all-or-nothing 467 GB rewrite, brick risk, no A/B/rollback/versioning.** | A routine update is a multi-hour single-copy rewrite that can permanently brick the box on a power blip. | high | [LOCK-IN-NOW] → §A |
| **No runtime model present/valid/version gate** — a partial or wrong image is DMA'd and inference released with no signal (`docs/OPERATION_FLOW.md:89-90`; `host/aipu_device.py:9-14`). | A mis-provisioned box appears "ready" and returns confidently wrong output. | high | [LOCK-IN-NOW] → §A |
| **Boot copies ~467 GB with no integrity check** — `done` on word count only, no CRC/hash/error path (`src/boot_loader.v:14-18,270-275,90-106`). | The box can boot "ready" on a corrupted model and produce wrong tokens indefinitely with no error. | high | [LOCK-IN-NOW] → §A |
| **`flash_layout.py` is miscredited as the provisioning tool** — it only prints channel-balance tables (`docs/USBC_PRODUCT_PLAN.md:116-117` vs `tools/flash_layout.py:1-29,265-272`). | Anyone provisioning by the docs hits a dead end. | med | [DOC-FIX] |
| **Air-gapped ingress path is undefined** — how the 467 GB file physically reaches the box on first setup or update is never specified (`docs/OPERATION_FLOW.md:104-108,298-299`; USB-C carries only tokens, `:126-128`). | The air-gapped customer has no supported way to load or refresh the model — the flagship use case has an undefined first step. | med | [DESIGN] |

> **Cross-ref — the ~70 s cold boot** appears here and in §2, §3, §5, §7. Stated
> once: the flagship residency box loads 467 GB from volatile LPDDR5X on **every**
> power-on in **~70 s** (`docs/R3_APPLIANCE_SPEC.md:18,114`), yet
> `docs/OPERATION_FLOW.md:96-99` advertises ~1–2 s (that is the 17 GB
> streaming-rung number), and the host models it as sub-second
> (`host/aipu_device.py:119,215`; `--boot-seconds` default 0.4). No in-flight
> readiness UX is built. **[DOC-FIX + SOFTWARE-TRACK]** — reconcile the number and
> build a progress signal; client timeouts otherwise fire before the box is ready.

### 2 · Host software the user touches

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **No USB-C transport or driver of any kind** — `USBBackend` is a to-build note (`host/README.md:154-156`); no `libusb`/`bulk_out` in repo. | Even the minimal "prompt in, token out over the cable" path cannot run; everything demonstrable is loopback-to-mock on one PC. | high | [SOFTWARE-TRACK] |
| **Multi-context host layer corrupts itself** — `ThreadingHTTPServer` over one shared mutable device, no lock/queue anywhere (`host/aipu_server.py:301-303`; `host/aipu_device.py:193,233-237`). | Two concurrent conversations interleave/garble or race; the "N agents" promise actively breaks under concurrency. | high | [SOFTWARE-TRACK] |
| **Prefix/prompt caching not implemented and actively defeated** — `generate()` calls `reset_session()` then re-prefills every token every turn (`host/aipu_device.py:174-203`); no session state server-side. | On a bandwidth-bound box every turn re-streams the whole history's weights; a 20-turn chat gets quadratically slower. | high | [SOFTWARE-TRACK] |
| **No telemetry/management endpoints** for the promised dashboards (accept rate, power/thermal, update) — only `/v1/models` and `/health` (`host/aipu_server.py:196-206`). | The "tune and observe via GUI" pillar has neither UI nor backend to feed one. | high | [SOFTWARE-TRACK] |
| **No conversation-history persistence** — server is fully stateless (`host/aipu_server.py`, no storage code). | None of the user's history is retained; the "appliance that remembers you" value is absent. | high | [SOFTWARE-TRACK] |
| **Only real-token backend is hardcoded to a deleted build** — `build/glm_model_fp8_sim` isn't tracked at HEAD; on main the software emits a canned string (`host/aipu_sim_backend.py:39`; `host/aipu_device.py:92-101`; honestly noted `README.md:331`). | Anyone evaluating the software on main sees only a mock echo. | med | [SOFTWARE-TRACK] |
| **Readiness model understates cold start ~35–175×** (0.4–2 s modeled vs ~70 s real) (`host/aipu_device.py:119,215`; `host/aipu_server.py:278`). | Any UX built on it misrepresents the wait; calls during boot fail or hang on a misleading ready signal. | med | [DOC-FIX] |
| **Sampling/tuning knobs have no observable effect** — `configure_sampling` records but MockDevice ignores; penalties "accepted and ignored" (`host/aipu_device.py:162-172`; `host/README.md:90-94`). | A tuning GUI would show sliders that change nothing. | low | [SOFTWARE-TRACK] |

### 3 · Interactive session quality

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **Long-prompt prefill is serial and pays full per-token weight streaming** — catastrophic first-token latency (`host/aipu_device.py:174-183`; `src/glm_q4k_system.v:507-574`; on-die prefill write-back not implemented per `src/glm_q4k_soc_ms.v:204-206`). | **Rate corrected (2026-07): prefill runs at ~43 tok/s, not the ~80 previously stated here.** ~80 is the DECODE rate, and R3 §2's decode constant divides by the spec chain's acceptance — an amortisation prefill cannot use, because the prompt's tokens are already known, so there is nothing to speculate. Decode is `14·U/A + 11/A + 0.5` at the measured K=1 point (A_eff=1.87, U(1)=1.00) = **13.87 GB/tok** → 1.1 TB/s ÷ 13.87 ≈ 79 ≈ the stated 80. Prefill is the same formula at A=1/U=1: the full `14 + 11 + 0.5 = 25.5 GB/tok` → 1.1 TB/s ÷ 25.5 ≈ **43 tok/s** (v3-proto's 1.54 TB/s ≈ 60) — **1.84× the decode cost per token**. *(This row first cited R3's `≈15.4 GB/tok` as the decode constant; 15.4 is the retired K=4 proxy vintage and reproduces none of the project's live headlines — see R3 §2's re-derivation. The prefill arithmetic was unaffected: 25.5 does not depend on it.)* So a 4 K doc ≈ **95 s** to first token (not ~50 s) and 16 K ≈ **6.3 min** (not 3+) — a **1.7× understatement**. **Cross-turn recurrence is FIXED** (prefix/KV cache, `host/aipu_device.py`; measured 5.8× total prefill work and 82.7% hit on the real GLM template path), so this now bites only the FIRST turn / a cold long paste. Batched prefill would be the remaining lever but is architecturally blocked — see the row below. | high | [DESIGN] |
| **Batched prefill is unreachable: the die has no KV egress path** — the computed latent is write-only and read nowhere (`src/mla_attn_q4k.v:381-386`, declared inside `lint_off UNUSEDSIGNAL`; `ckv_cur` is written at `:967`, reset at `:895`, **zero reads**), and the host FSM's KV append is low for the entire `mdl_start→mdl_done` window (`src/glm_q4k_system.v:517-519,553-564`). KV bytes are host-supplied via `kv_row_in` on every top. | A PE_M row at position p+i can never attend the key of row p+i-1 — that key is not produced anywhere in the design, so the amortisation that makes long prompts tolerable elsewhere is not available. Honestly: **there is no prefill datapath at all** — the die has never consumed a KV latent it produced. Note the ceiling is modest even if built: the repo's own union formula caps it at **~1.5–2× at feasible B** (the expert union barely shrinks below B≈32; the 11× needs B≈256 ≈ 770 TFLOP/s, infeasible). Serves one case the prefix cache cannot: a cold long paste. | med | [DESIGN] |
| ~~**Raising the context window would SILENTLY FREEZE attention to the first TOPK tokens**~~ — **THREADED (2026-07).** `DSA_REAL_IDX` was a parameter of `src/mla_attn_q4k.v:170` **and of no other file in the repo**: `glm_decoder_block_q4k` instantiated the attention passing `PER_ROW_POS/SLEN/SEQ` but **not** it, so the production hierarchy hard-wired it to 0 and **=1 was unreachable from any top**. Now threaded `glm_q4k_system(_cdc)` → `glm_model_q4k` → `glm_decoder_block_q4k` → `mla_attn_q4k`, default 0. **Verified so far:** the source diff is exactly the parameter declaration plus the instance connection at each level (nothing else changed), every level defaults to 0 and forwards it, so the attention receives 0 — **the identical value it defaulted to before threading**. **NOT yet verified:** the netlist equivalence gate `make dsa-thread-equiv` is written and registered in `release-gate`, but has **not completed a passing run** — `equiv_induct` over a full decoder with `mla_attn_q4k` elaborated as a REAL module (a blackbox would not exercise the parameter's journey, which is the whole point) exceeds this machine's practical budget. **Until that gate goes green, "byte-identical" is an argument, not a proof.** | **What this fixed: reachability, not the window.** At 0 the indexer is fed zero key-index vectors, so "every key scores 0 and top-K keeps keys 0..min(S,TOPK)-1 by lower-index tie-break — **Q-INDEPENDENT**" (`mla_attn_q4k.v:155-158`). At the committed S_MAX=8/TOPK_ATTN=8 that is a **no-op for any value** — the dense path never pulls keys at all (`:165-169`). Raise S_MAX past TOPK_ATTN with the old unthreaded default and every query at every position would attend to **the first 8 tokens, forever**: fluent output, frozen prefix, green tests (nothing asserted **which** keys were selected). Raising the window is now a **decision** (`DSA_REAL_IDX=1`, proven bit-exact at the leaf by `make mla-sparse` at PE_M=3) rather than an accident of defaults. **The context window itself is still NOT raised** — see the rows below; this removed the trap that made raising it unsafe. | ~~high~~ **closed** | [DESIGN] |
| **Committed context window is tiny** — S_MAX=8 attention window, KV_CTX=1024 (`src/glm_q4k_system.v:128,145-146`; `src/mla_attn.v:210-212`); 1 M context is elaboration-only (`docs/FULL_CONFIG_ELAB.md:56-61`) while the UI implies ~500 K (`docs/USBC_PRODUCT_PLAN.md:96`). | Near the committed config, anything beyond a few tokens is silently outside the window; scaled up, the long-context path has never run end-to-end. | high | [DESIGN] |
| **No context-overflow policy or length guard host-side** — messages forwarded with no bound; position aliases past KV_CTX (`host/aipu_server.py:208-268`; `src/kv_cache_pager.v:73-74,193`). | A long paste yields silently wrong output (aliased KV) instead of a graceful truncate or clear error — looks like hallucination to defense/finance/health users. | high | [DESIGN] |
| **Prefix-cache is a doc bullet only** — committed code throws away all KV every request (dedup of §2 caching gap, restated for the session lens; `host/aipu_device.py:146-147,193`). | The signature "interactive, cache-always-on" feel delivers nothing. | high | [SOFTWARE-TRACK] |
| *Cross-ref:* **~70 s cold start with no readiness UX** (see §1 box) — jarring instant-on failure for a phone-tethered user. | | high | [DOC-FIX] |
| *Cross-ref:* **Multi-context aggregate-throughput promise contradicts B=1 scope** (see §4). | | high | [DESIGN] |

### 4 · Multi-context

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **Shipped USB-C protocol has no context/sequence id** — host cannot route tokens to N contexts; pager's per-seq ports never driven (`src/glm_q4k_system_cdc.v:198-205,378`; `src/kv_cache_pager.v:112-117`). | Multi-context is impossible end-to-end regardless of RTL. | high | [LOCK-IN-NOW] → §C |
| **Production top instantiates the pager at NSEQ=1** — multi-context RTL lives only in the unshipped `glm_q4k_soc_ms` side module, Q4_K re-run PENDING (`src/glm_q4k_system.v:676-687`; `src/glm_q4k_system_cdc.v:6-7,33`; `src/glm_q4k_soc_ms.v:87`; `docs/OPERATION_FLOW.md:321`). | The device a user plugs in cannot hold more than one context's KV; the "verified" capability is unwired and not numerically re-validated at Q4_K. | high | [DESIGN] |
| **No multi-context software path** — host stack is hard single-session, keyed by a global cursor (`host/aipu_device.py:102-113,193,227-254`; `host/aipu_server.py:68-75,303`). | Two contexts produce cross-talk or a mid-generation reset; the box is one-conversation-at-a-time. | high | [SOFTWARE-TRACK] |
| **Batching is fixed lockstep, not elastic** — host FSM prefills all seqs then steps in lockstep, no per-context arrival or EOS (`src/glm_q4k_soc_ms.v:307-431,383,416`). | You can't join an in-progress batch; one long context stalls all others — "everyone waits for the slowest." | high | [DESIGN] |
| **No KV-sharing / prefix dedup across contexts** — NSEQ independent windows (`src/kv_cache_pager.v:23-32,90-93`); the owner's "better-than-linear via KV-sharing" premise isn't what the design does. | The mechanism the aggregate-throughput number rests on is absent, and the number is unmeasured and contradicted by the docs' own notes. | high | [DESIGN] |
| **Context count conflated with KV byte budget; no admission control** beyond a fixed lane count (`src/glm_q4k_soc_ms.v:72,87`; no scheduler in `host/`). | The (PE_M+1)-th context has undefined behavior — "run several agents" silently caps with no backpressure. | high | [DESIGN] |
| **Per-context KV isolation is addressing-only** — no bounds/permission enforcement, no adversarial cross-context test (`src/kv_cache_pager.v:189-198,182-185`; `src/glm_q4k_soc_ms.v:477-478`). | For the security-strict buyer, isolation is unenforced and untested; one index bug leaks one conversation's KV into another. | med | [DESIGN] |

### 5 · Power, thermal, physical

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **"Plug USB-C into my laptop and go" is physically impossible** — the box is dead until a second power source is attached; a laptop port sources only 7.5–15 W vs the box's ~40–80 W (v3-volume ~40–60, proto ~50–80), and what ships (own DC / USB-PD EPR / bundled brick) is unresolved (`docs/R3_APPLIANCE_SPEC.md:81-83`; `docs/USBC_PRODUCT_PLAN.md:310,328-330`; no "no-power" device state in `host/aipu_device.py`). | The single most-stated usage promise fails at the very first action; the user thinks the device is broken. | high | [DESIGN] |
| **Volatile 512 GB LPDDR5X: any power blip wipes the model AND all resident cache** — contradicting both "plug in, ready" and "unconditional caching" (`docs/R3_APPLIANCE_SPEC.md:18,114`; no retention/fast-resume in `src/`). | Every cold start is a ~70 s wait; a bumped cable means 70 s + total loss of the promised always-reused KV. | high | [LOCK-IN-NOW] → §B |
| **No power/thermal telemetry surfaced** — DeviceState is only BOOTING/READY/BUSY; no temp/power/throttle attribute (`host/aipu_device.py:96-99`; `docs/USBC_PRODUCT_PLAN.md:281`). | An 80→8 tok/s throttle drop with no on-screen reason reads as "broken"; the user can't tell it's thermal/power or act on it. | high | [LOCK-IN-NOW] → §C |
| **No closed-loop thermal management in RTL** — the throttle knob is a static input tied to 0, no sensor drives it (`src/clk_throttle.v:38`; `src/clk_gate_cluster.v:76`). | During the long sessions the box is pitched for, it can run hot and loud with no automatic protection; "≤35 dBA quiet" has no mechanism behind it. | med | [DESIGN] |
| **Eco/power knob is unreachable by the user** — the protocol carries only token IDs, so the planned GUI power tuning has no wire to `clk_throttle` (`docs/OPERATION_FLOW.md:105-108`; `host/aipu_device.py:12,154`). | The eco/quiet mode is marketed and RTL-built but is a dead control. | med | [LOCK-IN-NOW] → §C |
| **Power number is inconsistent across docs** — 40–60 W vs 80–110 W vs 30 W throttled (`docs/R3_APPLIANCE_SPEC.md:79,173-175`; `docs/USBC_PRODUCT_PLAN.md:50,211,320`). | The buyer can't size an adapter, predict desk heat, or set noise expectations; two docs give two pictures. | med | [DOC-FIX] |
| **Idle/standby power for the 512 GB box is unanalyzed** — the ≤10 W target is inherited from the smaller streaming box, yet the 70 s boot pushes 24/7-on (`docs/USBC_PRODUCT_PLAN.md:212`; `docs/LOW_POWER.md:325`). | The user pays continuous standby watts and 24/7 fan noise without being told the number; ≤10 W likely doesn't hold. | med | [DOC-FIX] |

### 6 · The flagship air-gapped RAG / GUI / visualization workflow

> This is the single most-cited reason to buy the box, and it is **0% built.**
> Stated once here; it also surfaced under host software (§2) and reliability (§7).

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **The entire RAG stack does not exist** — embedder residency, RAG store, graph view are labeled NEW work items; `grep embed/vector/retriev` over host+src = 0 hits (`docs/USBC_PRODUCT_PLAN.md:90-91,108-109`; `host/aipu_server.py:196-210`). | When a design partner says "show me the RAG," there is nothing to run — not a prototype, not a stub. | high | [SOFTWARE-TRACK] |
| **No embedded web UI / GUI** — the "plug in and a UI appears at aipu.local" experience is entirely absent; `find *.html/*.js/*.css` = 0 files (`docs/USBC_PRODUCT_PLAN.md:84-94`). | The "it just works, no software to install" demo cannot happen; best case is a technical user curling a Python endpoint. | high | [SOFTWARE-TRACK] |
| **Visualization graphs unbuilt with no data plumbing** — knowledge graph / timeline / spec-chain telemetry have no collection path; USB-C carries only token IDs + position (`docs/USBC_PRODUCT_PLAN.md:92-94`; `docs/OPERATION_FLOW.md:105-107`). | The "see visualization graphs" half of the flagship triad is nonexistent and can't be quickly prototyped. | high | [SOFTWARE-TRACK] |
| **GUI tuning surface has nothing to tune and no GUI** — sampling knobs are API-only and partly inert (`docs/USBC_PRODUCT_PLAN.md:93`; `host/README.md` sampling table). | The "tune it via GUI" third pillar is unbuilt and, where wired, non-functional against the only runnable backend. | high | [SOFTWARE-TRACK] |

### 7 · Reliability and failure modes

| Gap | User impact | Sev | Tag |
|---|---|---|---|
| **No ECC on the resident model weights** — 467 GB always-on DRAM with zero bit-flip protection or scrub (`src/glm_q4k_system.v:676-688`; empty `grep ecc\|secded\|scrub`; `docs/PRODUCT_ROADMAP.md:99`, P2.1). | Silent, undetectable weight rot; answers subtly degrade with no error — intolerable for a defense/finance/health buyer trusting local answers. | high | [LOCK-IN-NOW] → §B |
| **Power loss = full ~70 s cold reload + total session loss, no hold-up/resume** — `start` always restarts from `rseg=0` (`src/boot_loader.v:207-221`; `docs/R3_APPLIANCE_SPEC.md:16-21`). | Every power blip is a multi-minute total outage plus loss of the entire conversation/RAG working state — the opposite of "always-on personal infra." | high | [DESIGN] |
| **Boot has no integrity check** — a bad/partial provisioning or NVMe read error yields a silently wrong model (dedup of §1; `src/boot_loader.v:14-18,270-275`). | The box can run a corrupted model indefinitely with no error signalled. | high | [LOCK-IN-NOW] → §A |
| **Context/KV overflow wraps silently** — unbounded position counter aliases past KV_CTX, no "context full" signal (dedup of §3; `src/kv_cache_pager.v:177,306-309`; `src/glm_q4k_system.v:145`). | A long session quietly starts returning garbage attention instead of cleanly reporting "context full." | high | [DESIGN] |
| **Encryption-at-rest for NVMe KV/history is unbuilt** — promised to the security buyer; `grep encrypt\|aes\|cipher` over src+host = empty (`docs/USBC_PRODUCT_PLAN.md:99-100`). | For the buyer who chose the box because "data never leaves," a removed drive exposes plaintext history and KV — the security promise fails at physical-theft, the exact boundary it was written to cover. | high | [SOFTWARE-TRACK] |
| **Unconditional caching writes continuously to the single M.2 that also holds the 467 GB model** — no wear-leveling/quota (`docs/USBC_PRODUCT_PLAN.md:95-104`; `docs/R3_APPLIANCE_SPEC.md:115`; `docs/USBC_PRODUCT_PLAN.md:294`). | On a device meant to run for years, steady cache writes wear the drive that holds the model; when it degrades, the user loses history and model store at once. | med | [DESIGN] |
| *Cross-ref:* **RAG/GUI/viz 0% built** (§6) and **multi-context contradicts B=1 scope** (§4) also surfaced as reliability failures. | | high | [SOFTWARE-TRACK] / [DESIGN] |

---

## Severity summary

Counts are over the **49 confirmed findings** as reviewed (before the dedup
above, so the ~70 s boot and the RAG/GUI gap are each counted once per dimension
they broke in):

| Severity | Count |
|---|---:|
| High | 37 |
| Medium | 11 |
| Low | 1 |
| **Total** | **49** |

By classification (post-dedup, distinct gaps):

| Tag | Meaning | Distinct gaps |
|---|---|---:|
| **[LOCK-IN-NOW]** | Architecture — decide before board/boot-loader/protocol freeze | 3 decisions (§A ECC-adjacent boot/provisioning, §B ECC+scrub, §C protocol) covering ~8 findings |
| **[SOFTWARE-TRACK]** | Host/embedded software to build (transport, RAG, GUI, viz, persistence, caching, encryption) | ~17 |
| **[DESIGN]** | RTL/system design point to resolve (context window, prefill, multi-context batching, power delivery, thermal, wear) | ~15 |
| **[DOC-FIX]** | Inconsistency or under-communication to correct now (boot timing, power band, idle power) | ~5 |

---

## Closing — honest framing

**This is product-stage reality, not a regression.** Nothing in this register
says the accelerator is broken. The datapath is bit-exact against a numpy
golden, the KV pager and boot DMA engine are verified, the ECC / reset-sync /
MBIST / clock-gating building blocks exist and pass, and the formal properties
hold. That work is real and it is strong.

What the review makes unambiguous is that a **verified accelerator is not yet a
usable device.** The software a buyer touches — the transport, the RAG store,
the embedded UI, the visualizations, the tuning surface, history persistence,
multi-context routing, and real provisioning — is largely unbuilt, and a small
number of lifecycle/safety decisions (A/B provisioning + boot integrity, ECC +
scrub on resident weights, a context-id + telemetry protocol) need to be locked
before hardware and protocol freeze because they are cheap now and a respin
later.

The right read of this document is not "the project regressed." It is: **the
hard, de-riskable silicon problem is largely solved; the remaining work is the
experience and lifecycle layer, and three of those items must be decided before
freeze.** Sequencing the three LOCK-IN-NOW decisions ahead of the
software-track build is the single highest-leverage planning move available.
