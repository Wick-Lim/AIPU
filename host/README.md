# AIPU host software (D2 scaffold)

The host-side software that turns the AIPU device into **a local OpenAI-compatible
endpoint** — point any existing client (a chat UI, a VS Code extension, the `openai`
SDK with `base_url=http://localhost:8000/v1`) at it and it drives the device through
the exact RTL host protocol. This is the **software track's first deliverable**
([`docs/USBC_PRODUCT_PLAN.md`](../docs/USBC_PRODUCT_PLAN.md) Phase D2), buildable and
testable **with zero hardware** so it is ready when D1 (first real tokens over USB-C)
lands.

## What's real vs. scaffold (honest)

| Piece | Status |
|---|---|
| Device protocol (`aipu_device.py`) | **real** — mirrors `glm_fp8_system_cdc`'s host interface exactly (`start`/`prompt_tok`/`start_pos`/`s_len` → `busy`/`done`/`next_tok`/`tok_valid`) + the boot-loader-done readiness gate |
| OpenAI API surface (`aipu_server.py`) | **real** — `/v1/models`, `/v1/chat/completions` (streaming SSE + non-streaming), `/health`; stdlib only, 0 deps |
| Generation loop | **real** — prefill → autoregressive decode → token streaming |
| Tokenizer | **both** — byte-level (stdlib, exact round-trip) **and the REAL GLM-5.2 BPE** (`tokenizer.json` via the `tokenizers` lib); `make_tokenizer()` picks GLM when available, else byte. Verified: round-trips English / Korean / code, streaming-safe across multi-byte chars (vocab 154856, eos `<\|endoftext\|>`=154820) |
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
print(c.chat.completions.create(model="aipu-glm-5.2-fp8",
      messages=[{"role": "user", "content": "hi"}]).choices[0].message.content)
```

## Test

```sh
python3 host/test_aipu.py        # 6 tests: tokenizer round-trip, boot gate,
                                 # generation, max-tokens, server end-to-end, no-truncation
```

## Backends

Selectable with `--backend`; each is an `AIPUDevice` subclass — the server, generation
loop, streaming, tokenizer, and OpenAI surface are unchanged.

- **`MockDevice`** (`--backend mock`, default) — replays a canned reply through the
  protocol; zero deps, instant. Proves the plumbing for byte OR GLM vocab.
- **`SimulatorBackend`** (`--backend sim`, `aipu_sim_backend.py`) — **implemented**:
  runs the committed `glm_model_fp8` slice via its iverilog/`vvp` build and returns the
  **REAL argmax tokens the RTL forward pass produces** (measured: `{4, 31, 20}`), wired
  into the device protocol — the *server → real RTL → real token* co-sim path. Honest
  caveats: **SLOW** (~12 min/run — measured 752 s, cached per process, not interactive);
  **slice** model (VOCAB=256, untrained → real datapath outputs, not language);
  **fixed** testbench vectors (arbitrary-prompt drive needs the model's full weight/KV
  pull-port ROM harness — a larger TB effort). Needs `build/glm_model_fp8_sim`
  (`make unittests`, ~8 min once).
- **`USBBackend`** (to build at D1) — the real USB-C driver: enumerate the device, send
  the token/control words over the bulk endpoint, read back `next_tok` (the CDC host
  interface is already in the RTL). Pairs with the GLM tokenizer + a chat template.

```sh
python3 host/aipu_server.py --backend sim        # real RTL slice (SLOW, slice tokens)
```
