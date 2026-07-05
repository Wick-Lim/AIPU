#!/usr/bin/env python3
"""
aipu_server.py -- a local OpenAI-compatible HTTP server backed by an AIPU device.

Point any OpenAI-compatible client (chat UIs, VS Code extensions, `openai` SDK with
base_url=http://localhost:8000/v1) at this server and it drives the AIPU device
through the exact RTL host protocol (aipu_device.py). This is the D2 software
scaffold from docs/USBC_PRODUCT_PLAN.md: the API surface, the device-protocol
plumbing, and token streaming are REAL and swappable to the hardware backend; the
default MockDevice returns a clearly-labelled canned response (proving the loop, not
the model). A byte-level tokenizer makes text round-trip exactly through token ids.

Run:   python3 host/aipu_server.py            # 0 external deps (stdlib only)
Test:  curl -s localhost:8000/v1/models
       curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
            -d '{"model":"aipu-glm-5.2-fp8","messages":[{"role":"user","content":"hi"}]}'
       # streaming:
       curl -sN localhost:8000/v1/chat/completions -H 'content-type: application/json' \
            -d '{"model":"aipu-glm-5.2-fp8","messages":[{"role":"user","content":"hi"}],"stream":true}'
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from aipu_device import AIPUDevice, MockDevice          # noqa: E402


# ---------------------------------------------------------------------------
# Byte-level tokenizer (scaffold): text <-> byte ids (0..255), exact round-trip.
# The real GLM-5.2 tokenizer (tokenizer.json) plugs in here at D2 -- same encode/
# decode interface, larger vocab.
# ---------------------------------------------------------------------------
class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    class _Stream:
        """Incremental UTF-8 decoder: feed byte ids, get decodable text deltas."""
        def __init__(self):
            self._buf = bytearray()

        def push(self, byte_id: int) -> str:
            if byte_id < 0 or byte_id > 255:
                return ""
            self._buf.append(byte_id)
            try:
                out = self._buf.decode("utf-8")
                self._buf.clear()
                return out
            except UnicodeDecodeError:
                return ""                       # wait for the rest of the char

        def flush(self) -> str:
            out = self._buf.decode("utf-8", errors="replace")
            self._buf.clear()
            return out

    def stream(self) -> "ByteTokenizer._Stream":
        return self._Stream()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class AIPUServer:
    def __init__(self, device: AIPUDevice, tokenizer: ByteTokenizer):
        self.device = device
        self.tok = tokenizer

    def _prompt_text(self, messages: list[dict]) -> str:
        """Flatten the chat messages into a single prompt (a real backend would apply
           the model's chat template; scaffold concatenates)."""
        parts = []
        for m in messages:
            parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
        return "\n".join(parts)

    def _prime_mock(self, prompt: str) -> None:
        if isinstance(self.device, MockDevice):
            self.device.set_reply(
                f"[AIPU mock device] protocol round-trip OK -- received "
                f"{len(prompt)} chars over the host interface. Real tokens need the "
                f"hardware / full-model backend (docs/USBC_PRODUCT_PLAN.md D1).")

    def generate_text(self, messages, max_tokens):
        """Non-streaming: full assistant text."""
        prompt = self._prompt_text(messages)
        self._prime_mock(prompt)
        prompt_ids = self.tok.encode(prompt)
        st = self.tok.stream()
        chunks = []
        for tok in self.device.generate(prompt_ids, max_tokens):
            chunks.append(st.push(tok))
        chunks.append(st.flush())
        return "".join(chunks), len(prompt_ids)

    def generate_stream(self, messages, max_tokens):
        """Streaming: yield text deltas as tokens decode."""
        prompt = self._prompt_text(messages)
        self._prime_mock(prompt)
        prompt_ids = self.tok.encode(prompt)
        st = self.tok.stream()
        for tok in self.device.generate(prompt_ids, max_tokens):
            piece = st.push(tok)
            if piece:
                yield piece
        tail = st.flush()
        if tail:
            yield tail


def _now() -> int:
    return int(time.time())


def make_handler(server: AIPUServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                 # quieter
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": server.device.model_id, "object": "model",
                     "created": _now(), "owned_by": "aipu"}]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                server.device.poll_ready()
                self._json(200, {"state": server.device.state,
                                 "model": server.device.model_id})
            else:
                self._json(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._json(404, {"error": {"message": f"no route {self.path}"}})
                return
            length = int(self.headers.get("content-length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except Exception as e:
                self._json(400, {"error": {"message": f"bad json: {e}"}})
                return
            messages = req.get("messages", [])
            max_tokens = int(req.get("max_tokens", 256))
            stream = bool(req.get("stream", False))
            cid = f"chatcmpl-aipu-{_now()}"
            model = server.device.model_id

            if stream:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "close")
                self.end_headers()

                def sse(obj):
                    self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                    self.wfile.flush()

                sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                     "model": model, "choices": [{"index": 0,
                     "delta": {"role": "assistant"}, "finish_reason": None}]})
                try:
                    for piece in server.generate_stream(messages, max_tokens):
                        sse({"id": cid, "object": "chat.completion.chunk",
                             "created": _now(), "model": model, "choices": [{"index": 0,
                             "delta": {"content": piece}, "finish_reason": None}]})
                    sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                         "model": model, "choices": [{"index": 0, "delta": {},
                         "finish_reason": "stop"}]})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            text, n_prompt = server.generate_text(messages, max_tokens)
            n_completion = len(server.tok.encode(text))
            self._json(200, {
                "id": cid, "object": "chat.completion", "created": _now(),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant",
                             "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_completion,
                          "total_tokens": n_prompt + n_completion}})

    return Handler


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="AIPU local OpenAI-compatible server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--boot-seconds", type=float, default=0.4,
                   help="modelled device boot (resident-set load) time")
    args = p.parse_args(argv)

    device = MockDevice(boot_seconds=args.boot_seconds)
    device.power_on()
    server = AIPUServer(device, ByteTokenizer())
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    print(f"AIPU server on http://{args.host}:{args.port}/v1  "
          f"(model={device.model_id}, backend=MockDevice)")
    print("  GET  /v1/models   GET /health   POST /v1/chat/completions [stream]")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
