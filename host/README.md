# AIPU host software (D2 scaffold)

The host-side software that turns the AIPU device into **a local, single-user
OpenAI-compatible endpoint** (one box, one user — binds `127.0.0.1` by default, so the
whole thing runs fully offline / air-gapped: no external network, no cloud ever, and
nothing leaves because there's no path out — the endpoint still answers with the
ethernet unplugged, which is also the audit. A personal appliance's front door, not a
multi-tenant service) — point any existing
client (a chat UI, a VS Code extension, the `openai` SDK with
`base_url=http://localhost:8000/v1`) at it and it drives the device through the exact
RTL host protocol. This is the **software track's first deliverable**
([`docs/USBC_PRODUCT_PLAN.md`](../docs/USBC_PRODUCT_PLAN.md) Phase D2), buildable and
testable **with zero hardware** so it is ready when D1 (first real tokens over USB-C)
lands.

## What this host stack IS and IS NOT (honest ledger)

**IT IS, today, on `main`:**
- An **OpenAI-compatible HTTP API shim** (`/v1/models`, `/v1/chat/completions`
  streaming + non-streaming, `/health`) — stdlib only, 0 deps.
- A **real tokenizer** (byte-level, plus the real GLM-5.2 BPE when `tokenizer.json` is
  present) and a **GLM-5.2 chat template** (text path).
- A **device-protocol driver** (`aipu_device.py`) that mirrors the `glm_q4k_system_cdc`
  host handshake exactly, with host-side `max_tokens` / `stop` / `finish_reason`.
- Two backends: a **mock** (canned, self-labelled reply — proves the plumbing, default)
  and an **on-main RTL co-sim** (`--backend sim`: real but slow, untrained-slice,
  fixed-vector `glm_model_q4k` argmax tokens — a datapath witness, **not** a chatbot).

**IT IS NOT (none of this is built here — it is the future software track, see
[`docs/USBC_PRODUCT_PLAN.md`](../docs/USBC_PRODUCT_PLAN.md)):**
- **No RAG** — no document ingestion, embedder, or vector store.
- **No GUI / web app** — no chat UI, no visualization (knowledge-graph / timeline), no
  tuning dashboard, and no telemetry endpoints to feed one.
- **No USB-C transport** — no libusb/pyusb/CDC-NCM driver; all traffic is loopback over
  `127.0.0.1` to an in-process backend (`USBBackend` is a D1 to-build).
- **No multi-context / sessions** — the server is single-user; concurrent requests
  share one device object with no session routing or locking.
- **No persistence** — fully stateless; no conversation history / KV reuse across
  requests, nothing written to disk.
- **No real model tokens for your prompt** — the mock echoes a canned string; the sim
  emits fixed-vector slice tokens. Real language needs the full-model / hardware
  backend (D1+).

## What's real vs. scaffold (honest)

| Piece | Status |
|---|---|
| Device protocol (`aipu_device.py`) | **real** — mirrors `glm_q4k_system_cdc`'s host interface exactly (`start`/`prompt_tok`/`start_pos`/`s_len` → `busy`/`done`/`next_tok`/`tok_valid`) + the boot-loader-done readiness gate |
| OpenAI API surface (`aipu_server.py`) | **real** — `/v1/models`, `/v1/chat/completions` (streaming SSE + non-streaming), `/health`; stdlib only, 0 deps |
| Generation loop | **real** — prefill → autoregressive decode → token streaming |
| Tokenizer | **both** — byte-level (stdlib, exact round-trip) **and the REAL GLM-5.2 BPE** (`tokenizer.json` via the `tokenizers` lib); `make_tokenizer()` picks GLM when available, else byte. Verified: round-trips English / Korean / code, streaming-safe across multi-byte chars (tokenizer vocab 154856 tokens, eos `<\|endoftext\|>`=154820; the RTL config / LM-head width pads this to **154880** = next multiple of 128, so RTL-side docs quote 154880) |
| Chat template (`aipu_chat_template.py`) | **real** — a faithful port of GLM-5.2's official `chat_template.jinja` (text path); applied when the GLM tokenizer is active (see below). Byte scaffold / `--raw` keep the naive flatten |
| Sampling params | **partly host-side** — `max_tokens` + `stop` sequences + `finish_reason` are enforced host-side (real); `temperature`/`top_p`/`top_k`/`seed` are plumbed to the device (honored device-side; the mock is greedy) — see the table below |
| Backend (`MockDevice`) | **scaffold** — replays a clearly-labelled canned reply (tokenizer-agnostic: proves the plumbing for BOTH vocabularies, **not** the model). Swap for a simulator-backed or real-USB-C backend without touching the server |

The point: the **protocol + API + streaming + tokenizer are done and swappable**;
only the *backend* (real device / full-model runtime) remains — the D1/hardware
dependency, not blocking this layer.

## Tokenizer

```sh
pip install tokenizers            # once
host/fetch_tokenizer.sh           # ~20 MB from the public repo -> host/tokenizer.json (gitignored)
python3 host/aipu_server.py       # now uses the GLM BPE tokenizer (auto-detected)
# or point at a path:  python3 host/aipu_server.py --tokenizer /path/to/tokenizer.json
```

Without `tokenizers` or `tokenizer.json`, the server falls back to the byte tokenizer
(the plumbing still works end-to-end). `make_tokenizer()` in `aipu_tokenizer.py` is
the single selection point; the GLM tokenizer is paired with a real GLM-vocab backend
(the byte MockDevice is fine for either, since it replays whatever ids the server
encodes).

## Chat template

`apply_chat_template(messages)` in **`aipu_chat_template.py`** formats OpenAI-style
`messages` into the single prompt string GLM-5.2 expects, using GLM's special tokens
(each a **single** id in the GLM BPE vocab): `[gMASK]`, `<sop>`, `<|system|>`,
`<|user|>`, `<|assistant|>`, `<think>`. A `user`+`system` chat renders as:

```
[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>{system}<|user|>{user}<|assistant|><think>
```

**Fidelity (honest):** this is a faithful Python port of the *common text path* of the
official template
[`zai-org/GLM-5.2-FP8/chat_template.jinja`](https://huggingface.co/zai-org/GLM-5.2-FP8/resolve/main/chat_template.jinja)
(downloaded + read verbatim — **high confidence** for plain system/user/assistant
turns). GLM-5.2 is a *thinking* model, so the template auto-injects a
`<|system|>Reasoning Effort: {High|Max}` turn and ends the prompt with `<think>` (both
toggle via `enable_thinking` / `reasoning_effort`). Note GLM-5.2 drops the `\n` after
each role tag that older GLM-4 templates used. **Not ported** (kept as a standalone
function so it's easy to extend): tool/function-calling (`<tool_call>`/`<|observation|>`)
and multi-modal image/video/audio parts (they fall back to visible text + the
template's media `<reminder>`).

The template applies **only when the GLM tokenizer is active**. The byte scaffold and
`--raw` use the naive `role: content` flatten (the mock just round-trips, so the exact
format doesn't matter there).

```sh
python3 host/aipu_server.py                 # GLM tokenizer -> GLM chat template
python3 host/aipu_server.py --raw           # force the naive flatten (debug)
```

## Sampling parameters

`/v1/chat/completions` accepts the standard OpenAI sampling fields
(`SamplingParams.from_request` in `aipu_device.py`). Honestly, some are enforced
host-side today and some require a logits-capable device backend:

| Param | Where | Status |
|---|---|---|
| `max_tokens` | **host** | **real** — caps the decode loop; `finish_reason: "length"` when it triggers |
| `stop` (str or list) | **host** | **real** — generation stops when a stop string appears in the decoded text; output truncated (exclusive), streaming-safe across token boundaries; `finish_reason: "stop"` |
| `seed` | **host→device** | threaded to the device (`configure_sampling`) and echoed as `system_fingerprint`; the RTL sampler seeds on-device |
| `temperature` | **device** | plumbed to the device; **`sampler.v` samples on-device from logits.** The MockDevice returns **argmax (greedy)** and **ignores** it — no host-side logits to sample a canned stream, and faking it would be dishonest |
| `top_p` | **device** | same as `temperature` (device-side; mock greedy) |
| `top_k` | **device** | same as `temperature` (device-side; mock greedy) |
| `presence_penalty` | — | **accepted and ignored** (no host-side logit bias in the scaffold) |
| `frequency_penalty` | — | **accepted and ignored** |

So `max_tokens`, `stop`, and `finish_reason` are *real and useful today*;
`temperature`/`top_p`/`top_k` become live the moment a logits-capable backend (real
device / full-model runtime) lands and overrides `configure_sampling()` — no server
changes needed. `finish_reason` is set correctly: `"stop"` for a stop sequence or EOS,
`"length"` for the `max_tokens` cap.

## Run

```sh
python3 host/aipu_server.py                     # http://127.0.0.1:8000/v1  (stdlib only)

curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"hi"}]}'
# streaming (SSE):
curl -sN localhost:8000/v1/chat/completions -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

From the `openai` Python SDK:

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(c.chat.completions.create(model="aipu-glm-5.2-q4k",
      messages=[{"role": "user", "content": "hi"}]).choices[0].message.content)
```

## Test

```sh
python3 host/test_aipu.py        # 18 tests: tokenizer round-trip, boot gate, generation,
                                 # max-tokens/no-truncation, server end-to-end, GLM chat
                                 # template (structure, history, multimodal, special-token
                                 # encoding), sampling-param parsing + plumbing, stop-sequence
                                 # truncation, and finish_reason (stop vs length)
# with the real GLM tokenizer (if host/tokenizer.json is present it's auto-detected):
AIPU_TOKENIZER_JSON=host/tokenizer.json python3 host/test_aipu.py
```

## Backends

Selectable with `--backend`; each is an `AIPUDevice` subclass — the server, generation
loop, streaming, tokenizer, and OpenAI surface are unchanged.

- **`MockDevice`** (`--backend mock`, default) — replays a canned reply through the
  protocol; zero deps, instant. Proves the plumbing for byte OR GLM vocab.
- **`SimulatorBackend`** (`--backend sim`, `aipu_sim_backend.py`) — **on-main co-sim**:
  runs the on-main product top **`glm_model_q4k`** via its `make model-q4k`
  iverilog/`vvp` build and returns the **REAL argmax next-tokens the RTL forward pass
  produces** (bit-exact vs the numpy golden), wired into the device protocol — the
  *server → real RTL → real token* path. Honest caveats, in bold because they matter:
  **SLOW** (each forward is the full assembled model in an event sim → minutes/run for
  the committed VOCAB=256 slice; cached per process; **not interactive**); **untrained
  SLICE** model (MODEL_DIM=128/L=6/VOCAB=256 → genuine datapath outputs, **not
  language**); **FIXED testbench vectors** — the streamed tokens are the TB's built-in
  golden cases (`build/mq4k/stim.hex`), so **they are NOT a response to your prompt**
  (arbitrary-prompt drive needs the model's full weight/embedding/KV pull-port ROM
  harness — a larger TB effort). This is a **datapath witness, not a chatbot**. Build it
  first with `make model-q4k` (produces `build/glm_model_q4k_full_sim` +
  `build/mq4k/*.hex`). Verified: the fast `SPEC_SLICE` (VOCAB=16, `make
  model-q4k-smoke`) emits `{13, 3, 13}`; the committed VOCAB=256 slice emits its own
  genuine argmax vectors. *(History: the prior fp8-era backend targeted the
  `glm_model_fp8` slice, which was removed from `main` and lives on branch `fp8`; this
  backend was retargeted to the on-main `glm_model_q4k`.)*
- **`USBBackend`** (to build at D1) — the real USB-C driver: **does not exist yet.** No
  libusb/pyusb/CDC-NCM transport is implemented; everything today is loopback over
  `127.0.0.1` to an in-process backend. The plan: enumerate the device, send the
  token/control words over the bulk endpoint, read back `next_tok` (the CDC host
  interface is already in the RTL). Pairs with the GLM tokenizer + a chat template.

```sh
make model-q4k                                   # build the RTL slice sim first (once)
python3 host/aipu_server.py --backend sim        # real RTL slice (SLOW, fixed-vector slice tokens)
```
