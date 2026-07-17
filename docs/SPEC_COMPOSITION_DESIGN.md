# Spec-chain × memory-system composition (design, 2026-07) — KV_WRITEBACK Step 5

**The one step that raises tok/s: 60 → ~80/111. Multi-day; built in verified sub-steps.**

**STATUS: DONE (2026-07). All sub-steps 5a–5d complete and verified in `glm_q4k_spec_system`
(`make spec-greedy`, `ALL 31 TESTS PASSED`). The composed top holds all three pieces at once
— memory system + `PE_M=K+1` batched verify + spec accept/reject loop — commits a bit-exact
prefix of greedy (spec==greedy invariant), and its hardware `weight_loads` counter MEASURES
the A_eff amortization (ALL-ACCEPT hits the K+1/load ceiling; see 5d). The only input the RTL
does not determine is the external accept rate `r`, a measured model property.**

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

**5b — Position-accurate batched verify. RE-SCOPED 2026-07 after the 5b investigation
found the original scope was WRONG (a plausible-but-wrong trap the KV write-back does NOT
close).**

The original plan said "extend the KV write-back to per-ROW within one batched pass" — as
if the batched verify only needed system wiring. It does not. A position-accurate batched
verify needs **INTRA-BATCH CAUSAL ATTENTION**: within ONE `PE_M=K+1` pass, row j (draft at
pos p+j) must attend keys at positions p..p+j-1, and positions p+1..p+j-1 are the
*current-token keys of rows 0..j-1 computed in that same pass*. The MLA core does NOT do
this — it does batched attention over a **SHARED, already-populated** KV cache
(`mla_attn_q4k.v:72-92` SHARED-CONTEXT invariant: rows share s_len and the key set, differ
only in query/causal-extent). Confirmed: `kv_lat_row` egress is `ckv_cur[0]` ONLY (`:1787`);
rows 1..K's latents are never produced above the leaf; `ckv_cur[r]` is never injected into
the union/DSA/score key set. The KV write-back (Steps 1-3) gives PASS-to-PASS sequential KV,
NOT intra-batch KV. So naive seam-wiring yields spec_batched_top's shared-KV semantics →
batch ≠ serial for K≥1. This is exactly the plausible-but-wrong verify to refuse.

So 5b splits:

  **5b-leaf — intra-batch causal MLA (a NEW leaf feature in `mla_attn_q4k`, the correctness-
  critical core).** Add, behind a param (default off = today's shared-context batch, byte-
  identical): PE_M-wide latent egress (`ckv_cur[i]`/`krope_cur[i]` for all rows), and
  injection of rows 0..PE_M-2's current keys as attendable union keys tagged with batch
  position, with per-row causal masking (row j attends intra-batch key i iff i<j), correct
  per-position RoPE (`krope_cur[i]` already roped at pos_i), RMSNorm+W_uk/W_uv on `ckv_cur[i]`,
  and DSA over the combined pager+intra-batch key set. VERIFY (leaf oracle): a batched pass
  over consecutive causal positions == the serial single-row chain of the same tokens, BIT-
  EXACT full-logit (mla-sparse style). INJECTION: drop the causal mask (row j sees key j) →
  must FAIL. This is the hard, must-be-right sub-step; multi-day; it is the real blocker
  A_eff was always gated on.

  **5b-sys — compose.** Once 5b-leaf is bit-exact: widen the 4 `u_model` seam ports
  (`token_id/logits/argmax/h_state` — `glm_q4k_system.v:552`) so PE_M rows flow, connect the
  per-row `pos_vec/s_len_vec/seq_vec`, thread the PE_M-wide latent egress. VERIFY: a K+1-row
  batched pass in the system is BIT-EXACT vs K+1 serial decodes; byte-identical at PE_M=1.

**5c — Accept/reject + draft source.** Instantiate `spec_decode_seq` and a draft source
(`mtp_head`, as `spec_decode_top` does) in `glm_q4k_system`. Verify the **spec==greedy
invariant**: the composed top's committed token stream === a plain `PE_M=1` greedy decode of
the same prompt, for K=1..4 and varying draft accept rates. INJECTION: commit a raw
(rejected) draft → must FAIL (the stream would diverge from greedy).

**5d — Realise A_eff. DONE (2026-07, `make spec-greedy`).** `glm_q4k_spec_system` carries a
hardware `weight_loads` counter that pulses once per `PE_M=K+1` outer pass (`:551`,
S_LAUNCH), and a `total_tokens` counter of committed tokens. The gate now READS BOTH from
the DUT and asserts, per config: (a) **faithfulness** `weight_loads === actual passes` — the
die does exactly ONE weight-load per K+1-row batch (if the batch secretly re-loaded weights
per row this fails); (b) **ceiling** under ALL-ACCEPT `total_tokens === passes*(K+1)` — every
pass commits exactly K+1 tokens for its one load. `A_eff = total_tokens/weight_loads` is
therefore a MEASURED number, and `bytes/token = 25.50 GB / A_eff`. Measured (all 9 configs,
`ALL 31 TESTS PASSED`):

| pattern | K=1 | K=2 | K=3 |
|---|---|---|---|
| **ACCEPT (ceiling = K+1/load)** | 2.00 → 12.75 GB | 3.00 → 8.50 GB | **4.00 → 6.37 GB** |
| REJECT (floor = no gain) | 1.00 → 25.50 GB | 1.00 → 25.50 GB | 1.00 → 25.50 GB |
| MIXED (p=1 accepted) | 1.50 → 17.00 GB | 2.00 → 12.75 GB | 2.00 → 12.75 GB |

**What this measures vs what stays external.** The RTL now MEASURES the amortization
mechanism — tokens committed per ONE weight-load — from a hardware counter, and it hits the
K+1 ceiling exactly. That is the fact `A_eff` was always gated on ("K drafts verified in one
weight-load"), and it is now realised in one composed top (memory system + `PE_M=K+1` + the
spec loop), not an unrealised design point. What the RTL does NOT (and cannot) invent is the
production ACCEPT RATE `r` — the model's MTP draft quality — which selects WHICH column the
box operates in. That is an external `[실측-입력 EST]` (`H_MEASUREMENT.md`: r₁=0.87 →
`A_eff=1.87` at K=1). So `tok/s = BW / (25.50/A_eff)`; at the measured `A_eff=1.87` →
13.87 GB/token → **111 tok/s under 1.54 TB/s** (or ~80 under the 1.1 TB/s resident box). No
overclaim: the amortization is measured RTL; the accept rate feeding it is a measured model
property; the tok/s is their product.

## Verification discipline (no compromise)

Every sub-step: byte-identical at `PE_M=1`; bit-exact vs an independent reference at `PE_M>1`;
an injection that a real composition bug (wrong per-row position, committed raw draft, mixed
KV) FAILS. The spec==greedy invariant is the load-bearing correctness statement — a composed
top that commits anything other than a prefix of greedy is wrong, however fast.
