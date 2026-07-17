# Spec-chain × memory-system composition (design, 2026-07) — KV_WRITEBACK Step 5

**The one step that raises tok/s: 60 → ~80/111. Multi-day; built in verified sub-steps.**

## Why this, why now

`glm_q4k_system` runs the model at `PE_M=1` = no speculation = 25.5 GB/token = **60 tok/s**
(measured). The 111 tok/s design point needs `A_eff=1.87`, which needs **K drafts verified
in ONE weight-load** — a `PE_M=K+1` batch (`spec_batched_top`). But `spec_batched_top` has
the batched verify and **no memory system** (its KV/weight pulls route to TB stubs), while
`glm_q4k_system` has the memory system and `PE_M=1`. Neither top has both. Composing them is
the `[설계필요]`.

The blocker that made the batch *incorrect* — "draft j cannot attend to draft j-1 because the
computed latent is write-only" — is now **removed**: `docs/KV_WRITEBACK_DESIGN.md` Steps 1-3
built + verified the die-internal KV write-back (per-(layer,position), full-logit bit-exact).
So a *position-accurate* batched verify is now buildable.

## What the pieces are (measured)

- `spec_batched_top` — one `glm_model_q4k` at `PE_M=K+1`; K drafts arrive on a TB port
  (`draft_in`/`n_draft`); it produces `truth_vec {m_1..m_{K+1}}` (the model's per-row argmax)
  and feeds `spec_decode_seq`. KV/weight pulls routed straight up to stubs.
- `spec_decode_seq` (399 lines, committed, `make spec-adapt` green) — the accept/reject
  bookkeeping: draft==truth → ACCEPT (commit 2), else REJECT (commit the verify token).
  Commits the LONGEST ACCEPTED PREFIX of the model's OWN argmax, so the committed stream is
  structurally a prefix of greedy decode for ANY K (spec==greedy invariant).
- `spec_decode_top` — mints drafts with an `mtp_head` (the draft SOURCE spec_batched_top
  lacks).
- `glm_q4k_system` — the memory system (weight_loader, expert_cache, ddr5_xbar,
  kv_cache_pager) + the KV write-back (`SELF_KV`), at `PE_M=1`.

## Build plan (each verified, byte-identical-when-off)

Gate behind the existing `PE_M` (default 1 = today's top, byte-identical) — no new master
flag needed; `PE_M=1` IS the no-speculation top.

**5a — Thread the params.** Add `PE_M`, `SWIN`, `PER_ROW_POS`, `PER_ROW_SLEN` to
`glm_q4k_system` and pass them to `u_model` (like `DSA_REAL_IDX`/`SELF_KV` were threaded).
At `PE_M=1` byte-identical (resident-equiv / self-kv-equiv method). SWIN needs the
`min(PE_M*TOPK, S_MAX)` bound (mla_attn already asserts it); at PE_M>1 SWIN must grow — thread
it so it CAN. Verify: `PE_M=1` byte-identical; the top ELABORATES at `PE_M=K+1` (e.g. K=1..4).

**5b — Position-accurate batched verify over the KV write-back.** With `SELF_KV=1` and
`PE_M=K+1`, the K+1 verify rows are K+1 logical positions; each row j appends its (layer,pos)
KV and row j+1 gathers it — `PER_ROW_POS=1` gives each row its own position. This is exactly
what the KV write-back now supports (Step 3 keyed per-(layer,position); extend to per-ROW
within one batched pass). Verify: a K+1-row batched pass is BIT-EXACT vs K+1 sequential
single-row decodes of the same tokens (the batch must equal the serial truth), full-logit.
INJECTION: break the per-row position (all rows share pos) → must FAIL.

**5c — Accept/reject + draft source.** Instantiate `spec_decode_seq` and a draft source
(`mtp_head`, as `spec_decode_top` does) in `glm_q4k_system`. Verify the **spec==greedy
invariant**: the composed top's committed token stream === a plain `PE_M=1` greedy decode of
the same prompt, for K=1..4 and varying draft accept rates. INJECTION: commit a raw
(rejected) draft → must FAIL (the stream would diverge from greedy).

**5d — Realise A_eff.** Instrument weight-loads per committed token; confirm it matches
`(K drafts)/(1 weight-load) × accept` → the measured `A_eff` (≈1.87 at K=1 with the
measured accept). Report the ACTUAL bytes/token the composed top produces and the tok/s it
implies — replacing "111 is an unrealised design point" with a measured number (or an
honest smaller one if accept is lower than the proxy). No overclaim: whatever the RTL
measures is the number.

## Verification discipline (no compromise)

Every sub-step: byte-identical at `PE_M=1`; bit-exact vs an independent reference at `PE_M>1`;
an injection that a real composition bug (wrong per-row position, committed raw draft, mixed
KV) FAILS. The spec==greedy invariant is the load-bearing correctness statement — a composed
top that commits anything other than a prefix of greedy is wrong, however fast.
