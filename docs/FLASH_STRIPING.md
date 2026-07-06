# NVMe/PCIe striping & placement strategy — turning N NVMe drives / PCIe lanes into N× bandwidth

**The one fact that drives everything:** single-user decode throughput is **linear in
NVMe read bandwidth** (`tok/s ≈ NVMe_BW / [(1−h)·footprint] · K`), and a 1–4 TB on-board
NVMe SSD is **not one fast lane — realized bandwidth scales with parallel NVMe drives / PCIe
lanes**. Aggregate bandwidth is `N_CH × per-channel BW` (each channel = a striping endpoint
backed by an NVMe drive / PCIe lane), but that N× is only *realized* if every token's reads
spread evenly across the channels. **How weights are laid out across channels is therefore a
first-class performance lever, not an afterthought.** This doc consolidates the two mechanisms
that deliver it and lays out the striping design space.

The workload (real GLM-5.2): **75 MoE layers × 256 experts, top-8 fetched per layer per
token, ~37 MB/expert (FP8)**, plus the always-read shared expert, MLA/attention weights,
embeddings and the DSA index. The 753 GB of weights live on the NVMe SSD; DDR5 is the working
cache (hit rate `h`); the FP8 die computes.

---

## 1. Two layers of the solution (both already built + verified)

Realizing N× aggregate BW takes **a fabric** (issue reads to N channels in parallel and hide
storage latency) **and a placement** (make a token's reads actually land on different channels).

### 1a. The fabric — `src/flash_xbar.v` (P1.1, ✅ built, BMC-verified)

An `N_CH`-channel banked read fabric. `channel = req_addr[BANK_LSB +: log2(N_CH)]`, so
block-granular addresses spray across channels. (`flash_xbar` is a committed RTL name; in the
product it is the storage-read fabric that fronts the **NVMe/PCIe backend** — a labeled
placeholder host controller. The crossbar's read-request / latency-hiding abstraction is
medium-agnostic: address → weight bytes, with the NAND-specific backend swapped for an
NVMe/PCIe endpoint.) The hard part vs a DDR5 crossbar is **latency**: an NVMe read returns in
~10–100 µs = **thousands of cycles** (`FLASH_LAT`), so by Little's law

```
BW_channel ≈ min(1 read/cycle, QDEPTH / FLASH_LAT)
```

one outstanding read per channel delivers a catastrophic `1/FLASH_LAT`. The fabric therefore
gives **each channel a deep outstanding-request budget `QDEPTH`** (issue read N+1, N+2, … while
read N is still in flight to the drive — NVMe natively supports deep command queues).
`QDEPTH ~ FLASH_LAT` hides the entire storage latency → ~1 read/cycle/channel → **~N_CH
reads/cycle aggregate**.

- **Verified:** 2049 directed tests + **BMC K=12** — per-channel no-overflow
  (`cnt[c] ≤ QDEPTH`), `outstanding ≤ N_CH·QDEPTH`, `inflight ≤ outstanding`, no underflow
  (see [`FORMAL.md`](FORMAL.md)).
- **Measured effect:** latency-hide ~7.99× (Little's law) × N_CH banking → **~57× combined
  @ 8ch × Q8**; throughput is then **linear in N_CH** (4ch ≈ 3→12 tok/s in
  [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) P1.1).

### 1b. The placement — `tools/flash_layout.py` (P1.2, ✅ built, measured)

The fabric only reaches N× **if a token's top-8 experts sit on 8 different channels**. Which
8 experts fire is **data-dependent** (the router decides per token), so placement must make
*any* top-8 subset balanced. Key structural fact the packer exploits: the co-activation graph
is **block-diagonal by layer** — layer L's top-8 only ever co-activates with layer L's other
experts — so each layer's 256 experts are packed onto `N_CH` channels **independently**
(`layout[L·256 + e] → channel`). The packer is a **balanced greedy least-conflict** placer:
co-activated experts to different channels, channels kept load-balanced.

- **Measured (N_CH=8):** optimized layout sustains **55% of 8× peak** vs naive round-robin
  **39%** → **+40%**; 99.5% of fetches touch ≤2 experts/channel (the 4/5/6-on-one-channel
  collision tail is removed).

---

## 2. Why whole-expert placement caps at ~55% — the pigeonhole wall

`flash_layout.py` places **whole experts** on channels (strategy **A**). At `N_CH = 8` with
**top-8**, the ideal is 1 expert/channel = 8× = 100%, but two forces make that unreachable:

1. **Pigeonhole / birthday collisions.** 8 data-dependent experts over 8 channels: even with a
   perfect *static* layout, the *dynamic* top-8 set for a given token can put 2+ experts on one
   channel. Expected distinct channels hit by 8 uniform picks over 8 ≈ `8·(1−(7/8)^8) ≈ 5.25`
   → a hard ceiling around **~66%** before any skew, ~55% measured with real popularity.
2. **Popularity skew.** MoE routing isn't uniform (training's load-balance loss only softens
   it); hot experts over-load their channel.

`flash_layout.py` is honest about this: **8× is unreachable at 8ch**; the win is *removing
hotspots* (+40%), not reaching peak. **8 channels is the top-8 sweet spot** for strategy A —
below it you serialize, above it you pay channels that a single token can't all use (though
batching does — §4).

**The pigeonhole is a property of placing an atomic 37 MB expert on one channel.** Remove that
constraint and the cap disappears — which is strategy B.

---

## 3. The striping design space

| Strategy | Unit on a channel | Per-token balance | State / cost | Status |
|---|---|---|---|---|
| **A. Whole-expert placement** | one 37 MB expert → one channel | pigeonhole-capped (~55% @8ch) | offline map `layout[L·256+e]` (`flash_layout.py`) | ✅ built |
| **B. Sub-expert striping (RAID-0)** | each expert's bytes striped over **all** N_CH | **~100%, data-independent** | none (address arithmetic) | 📐 proposed |
| **C. Hybrid** | stripe within a channel *group*, place groups | tunable | group map + stripe | 📐 proposed |

### B — sub-expert striping (the pigeonhole-free option)

Stripe **every** expert's weight block across **all** `N_CH` channels (RAID-0), stripe unit
`S`. Then **every** expert fetch — whichever 8 the router picks — reads `37 MB / N_CH` from
each channel: all channels are always busy, **aggregate BW is fully used regardless of the
data-dependent selection.** The pigeonhole cap vanishes; there is no offline map to compute or
ship (the channel is just `addr[BANK_LSB +: log2(N_CH)]` at fine granularity — the fabric of
§1a already does this, only the *stripe unit* changes).

**The one constraint — keep the stripe ≥ the NVMe/SSD efficient read granularity.** An NVMe
read command carries fixed per-command overhead, and the SSD internally reads NAND pages
(~16 KB); stripes finer than that waste bandwidth (per-command overhead + a full internal
page-read per stripe). At `S ≥ 16 KB`: a 37 MB expert over `N_CH = 32` = ~1.15 MB/channel =
~72 pages/channel → efficient, no random-read penalty. So choose `page ≤ S ≤ 37 MB/N_CH`.

**Trade vs A:** B needs **more channels than top-k to *matter* (A) is false for B** — B is
balanced at *any* `N_CH`, including `N_CH > 8`, because it doesn't rely on 8 experts hitting 8
distinct dies. That is exactly what lets you push `N_CH` up (16, 32, …) for more BW without the
pigeonhole fighting you. Cost: the same fetch now issues `N_CH` smaller reads instead of 1 big
one — more outstanding requests (the QDEPTH budget of §1a absorbs this) and slightly more
request-fabric traffic.

### C — hybrid (stripe within groups, place groups)

Split `N_CH` into `G` groups of `N_CH/G`; stripe an expert across one group, place experts to
groups with `flash_layout.py`'s packer. Lets you bound per-fetch read count (group size) while
still spreading popularity across groups. Useful if fine striping's request-count is a fabric
concern.

---

## 4. What to stripe, and batching

**Stripe everything on the per-token critical path**, not just experts:

- **Routed experts** (top-8/layer) — the bulk; strategy A/B/C above.
- **Shared expert** — read *every* token → must be striped across all channels (never pin to
  one, or it's a guaranteed hotspot).
- **MLA/attention projection weights** (the 7 projections) — read every token/layer.
- **Embeddings + LM head, DSA index** — the LM head GEMV and index reads are large and constant.

**Batching (B rows) changes the channel arithmetic in strategy A's favor** — but note this is
the **non-target aggregate/datacenter regime**, not the product. The product is a **local,
single-user box running `B=1`**; large `B` only arises when batching many *different* users'
tokens together, which the personal box does not do. This paragraph is kept as analysis of what
the *same* silicon could do batched. With `B` tokens
processed per weight fetch (the **PE_M-batch path — now complete: DONE 4/4 across
swiglu/router/mla/mtp, and the union-fetch integrated in `glm_decoder_block_fp8`**, all verified
bit-exact), a layer's *union* of active experts approaches all 256 as `B` grows
(`E[distinct] = 256·(1−0.96875^B)`, where `0.96875 = 1−8/256`), so the
fetch set is large and naturally spreads across channels — the pigeonhole tail shrinks and
`N_CH > 8` becomes useful even under strategy A. The decoder block's PE_M>1 MoE loop already
fetches **only** that union (a `T_ESCAN` scan + combinational `any_has` skip of non-union experts),
so **the batch axis of this striping story is realized in the model, not just modeled** — up to
~32× fewer expert fetches at small batch, ~none at B≈256 (union≈all). See
[`ULTRA_PERF.md`](ULTRA_PERF.md) #1. **Striping (per-token, strategy A/B) and batching compose:**
striping keeps the single-user product (`B=1`) balanced — *that* is this box; batching keeps a
**non-target datacenter deployment** (`B`≈256, many *different* users) balanced on the same silicon.

---

## 5. Bandwidth → throughput (why this is the cheap lever)

Baseline (measured `h`, [EST] BW): **NVMe/storage 50 GB/s, h=27%, K=1 → ~3 tok/s, ~8–10 J/token**
([`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md)). Because tok/s is linear in realized NVMe BW:

| Realized NVMe/storage BW | approx single-user tok/s | how |
|---|---|---|
| 50 GB/s (baseline aggregate) | ~3 | baseline |
| ~4× (4 channels, strategy A) | ~12 | flash_xbar P1.1 (measured linear) |
| ~8× × 0.55 (8 channels, strategy A) | ~13–14 | + flash_layout +40% (pigeonhole-capped) |
| ~8× × ~1.0 (8 channels, strategy B) | ~24 | sub-expert striping removes the cap |
| ~16–32 channels (strategy B) | ~24–40+ | push N_CH; B stays balanced |

**Honest BW anchor [EST]:** the absolute figures above are roofline numbers carried over from a
NAND-multi-channel model, not per-device NVMe measurements — a single NVMe SSD is ~3.5 GB/s
(PCIe Gen3 x4), ~7 GB/s (Gen4 x4), ~14 GB/s (Gen5 x4). So the 50 GB/s baseline and the N× rows
are **aggregate across multiple NVMe drives / many PCIe lanes** — realizing them is the custom
board's job (several M.2/PCIe endpoints), not a single-drive claim, and the specific 50 GB/s
anchor does not cleanly re-derive to one NVMe. The tok/s stay [EST].

The **~25–40 tok/s sweet spot** — a usable interactive rate on a box that runs the **full
753 GB GLM-5.2 model fully offline / air-gapped** (nothing leaves because there is no path out;
the model is provisioned once, itself doable offline) at a fraction of cloud cost/power — is
reachable by **adding NVMe drives / PCIe lanes + striping** —
**not** by moving to HBM. NVMe scales
by `$/GB` commodity SSD; HBM charges a `$/GB/s` premium. Each channel adds an NVMe controller +
PCIe I/O power (a power/area/cost limit), so `N_CH` is bounded — but the scaling curve is
dramatically cheaper per GB/s than HBM.

---

## 6. Industry direction — HBF

**High Bandwidth Flash (HBF)** — SanDisk / SK hynix (announced 2025) — stacks NAND like HBM to
deliver HBM-class bandwidth from flash storage. The product today streams weights from an
**NVMe SSD** over PCIe; HBF is an emerging **device-level** storage backend that would slot in
behind the same medium-agnostic `flash_xbar` read / striping abstraction (*high-capacity
storage resident + bandwidth by parallelism*) for a future BW jump — the on-board striping /
placement strategy here rides the same curve.

---

## 7. Honest scope

- **A is built + measured; B/C are design (this doc).** Strategy A (`flash_xbar` fabric +
  `flash_layout` placement) is RTL + BMC-verified + measured (55% of 8× peak). Sub-expert
  striping (B) is an **address-granularity change to the same fabric** — no new datapath — but
  is **not yet implemented or measured**; the ~1.0 balance is the pigeonhole-free *expectation*,
  to be confirmed by extending `flash_layout.py`'s measurement harness to stripe mode.
- **All tok/s here are [EST]** — bandwidth-roofline model, not silicon (see
  [`ULTRA_PERF.md`](ULTRA_PERF.md), [`PHYSICAL_SKY130.md`](PHYSICAL_SKY130.md)). Real systems
  land below the roofline (achievable-vs-peak BW, cache-`h` optimism, second-order walls).
- **The compute is bit-exact** to real GLM-5.2-FP8 ([`BIT_ACCURACY.md`](BIT_ACCURACY.md));
  striping is a *placement* choice and does not touch numerics.

## 8. Files

- `src/flash_xbar.v` — the N_CH-channel banked read fabric (QDEPTH latency-hide); fronts the
  NVMe/PCIe storage backend (medium-agnostic; committed RTL name).
- `tools/flash_layout.py` — offline expert→channel packer + balance measurement (`--nch N`,
  `--dump-map`). Extending it with a `--stripe` mode measures strategy B.
- [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) P1.1/P1.2 — the levers this consolidates.
- [`FORMAL.md`](FORMAL.md) — `flash_xbar` BMC properties.
