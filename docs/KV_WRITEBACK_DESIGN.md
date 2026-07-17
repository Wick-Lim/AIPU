# KV latent write-back → self-attending die (design, 2026-07)

**Status: DESIGN + scoping. Multi-step RTL build, not a single-patch change. This file
is the honest, reviewable foundation before any RTL is touched — no compromise.**

## Why this is the tok/s critical path

The production top `glm_q4k_system` instantiates the model at `PE_M=1` and does NOT
compose the spec chain, so it runs with **no speculation** → 25.5 GB/token → ~60 tok/s,
NOT the 111 the design point assumes (§2 / USAGE_GAPS). The spec chain earns
`A_eff=1.87` (13.87 GB/token) by verifying K drafts in ONE weight-load. But
`spec_batched_top` itself records (`:24-33`) that its batched verify is NOT
position-accurate: *"there is no die-internal KV write-back path (the computed latent is
write-only), so draft j cannot attend to draft j-1."* So **the KV write-back is the
shared prerequisite** for a position-accurate spec verify, and therefore for realising
`A_eff` in one top. It also, independently, is what lets the die do prefill at all
(USAGE_GAPS #25: "the die has never consumed a KV latent it produced").

## What is actually there today (measured, `glm_q4k_system`)

- The model COMPUTES the latent per (layer, row): `mla_attn_q4k.v:384 ckv_cur[PE_M][KV_LORA]`
  (`x*W_dkv`) and `:386 krope_cur[PE_M][ROPE]` (`x*W_kr`, roped). These are internal
  regs — **not exposed as an output** of `mla_attn_q4k`, so nothing above can write them.
- The pager `kv_cache_pager` is real RTL (bounded RESIDENT ring + COLD spill). One row
  = `[c_kv | k_rope]`, `ROW_BITS = (KV_LORA+ROPE)*16 = 768`.
- Read is HALF-wired: `glm_q4k_system.v:692` `.gather_valid(kc_req), .gather_idx(pg_gather_idx)`
  — the model's read REQUEST drives the pager's gather. But the pager's `row_out` goes to
  the system OUTPUT `kv_row_out` (observability), and the model's `kc_ckv/kc_krope` are fed
  from a system INPUT (TB responder) — so the read DATA never comes from the pager.
- Write is a stub: `.append_row(kv_row_in)` ← a TB input; the die's computed latent is
  routed nowhere.

Net: the pager is instantiated and observed, but **it is not in the model's KV loop**;
the model's KV read/write are TB stubs.

## Protocol compatibility (verified — the wiring is structurally sound)

| model KV read (glm_model_q4k) | pager gather |
|---|---|
| `kc_req` (out) | `gather_valid` |
| `kc_idx` (out, IDXW) | `gather_idx` |
| `kc_ckv[KV_LORA*16]` / `kc_krope[ROPE*16]` (in) | `row_out[ROW_BITS]` split 512/256 |
| `kc_valid` (in) | `row_valid` |

Write: expose `{krope_cur[row], ckv_cur[row]}` packed to `ROW_BITS=768` → pager
`append_row`, `append_valid` pulsed when the row's latent is committed.

## The one hard problem: per-(layer,position) keying

KV is **per (layer, position)** (`kv_cache_pager.v:8` "one latent row per token per
layer"; model annotates every pull with `db_layer`). But the pager rings by
`(seq, slot=position)` only — **no layer dimension** — and is sized `KV_CTX=1024`
positions, `KV_RESIDENT=16`, with **no factor of L**. So closing the round-trip is NOT
just connecting ports; it needs one of:

- **(A) flat index** `pg_idx = db_layer*KV_CTX + pos`, pager sized `L*KV_CTX` rows. One
  pager, an adder + wider store. Verification-tractable at the slice (L=6). At full scale
  the store is the real per-layer KV = 87.8 KB/token × context — that lives in external
  LPDDR (stubbed; the RTL we verify is the indexing/protocol, not the DRAM array).
- (B) L pagers — rejected (L× the control, no upside over (A)).

Chosen: **(A) flat (layer,pos) index into one pager.**

## Build plan (each step verified + byte-identical-when-off, RESIDENT-pattern)

Gate the whole thing behind a new default-0 param `SELF_KV` so `SELF_KV=0` is
byte-identical to today's TB-stub top (prove with the resident-equiv-style structural
check).

1. **Expose the latent.** New output of `mla_attn_q4k`: `kv_lat_row[ROW_BITS]` (+ a
   `kv_lat_valid`) driven from `{krope_cur, ckv_cur}` of the committed row. Route up
   `mla_attn → glm_decoder_block_q4k → glm_model_q4k → glm_q4k_system`. Additive ports,
   driven from existing regs.
2. **L=1 round-trip first.** At `SELF_KV=1`, `L=1`: wire latent→append and
   pager `row_out`→model `kc_ckv/kc_krope`. No layer index yet. Verify the die attends to
   its OWN written KV, bit-exact vs a reference that models the same append/gather.
3. **Layer keying.** Add `pg_idx = db_layer*KV_CTX + pos`, size the pager `L*KV_CTX`,
   verify L>1 (slice L=6) bit-exact.
4. **Prefill.** With the round-trip closed, feed a K-token prompt as K appends before
   decode; verify prefill == per-token decode of the same prompt.
5. **Then** compose the spec chain (separate doc): thread `PE_M/SWIN/PER_ROW_POS/
   PER_ROW_SLEN` into `glm_q4k_system` and host the draft-verify — now position-accurate
   because (2)-(4) give draft j the real KV of draft j-1.

## Verification (no-compromise)

Every step: (a) `SELF_KV=0` byte-identical to base (structural, resident-equiv method);
(b) `SELF_KV=1` bit-exact vs an independent reference; (c) a soundness/injection test
that a corrupted latent (wrong pack, wrong index) FAILS the check. A round-trip that
can't tell a mis-indexed KV from a correct one is worthless.

## Honest scope

This is multi-day. It is NOT started as RTL yet — this file fixes the architecture so
the RTL is built right the first time. The protocol compatibility and the packing math
are verified above; the open engineering is the flat (layer,pos) store sizing at the
slice and the append/gather timing across the model's layer loop.
