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
| Tokenizer | **scaffold** — byte-level (exact round-trip); the real GLM-5.2 `tokenizer.json` plugs into the same `encode`/`decode` interface |
| Backend (`MockDevice`) | **scaffold** — returns a clearly-labelled canned reply (proves the plumbing, **not** the model). Swap for a simulator-backed or real-USB-C backend without touching the server |

The point: the **protocol + API + streaming are done and swappable**; only the
*backend* (real device / full-model runtime) and the *real tokenizer* remain, and
those are the D1/hardware dependency — not blocking this layer.

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

## Wiring a real backend

Implement `AIPUDevice` (`aipu_device.py`) for the target:

- **`SimulatorBackend`** — shell out to the iverilog/`vvp` build of `glm_fp8_system_cdc`,
  driving `start`/`prompt_tok`/`start_pos`/`s_len` and reading `next_tok`/`tok_valid`.
  Produces the *slice* model's tokens (small vocab) — useful for protocol/HW co-sim.
- **`USBBackend`** — the real USB-C driver: enumerate the device, send the token/control
  words over the bulk endpoint, read back `next_tok` (the CDC host interface is already
  in the RTL). Add the real GLM tokenizer + chat template.

The server, generation loop, streaming, and OpenAI surface are unchanged — only the
`AIPUDevice` subclass changes.
