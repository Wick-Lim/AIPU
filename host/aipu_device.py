"""
aipu_device.py -- host-side driver abstraction for the AIPU USB-C device.

This mirrors the RTL host interface of `glm_fp8_system_cdc` (the production 2-clock
top) EXACTLY, so the same driver code that talks to a MockDevice today talks to the
real USB-C device tomorrow -- only the backend changes.

RTL host interface (host_clk domain, src/glm_fp8_system_cdc.v):
    in :  start, prompt_tok[TOKW], start_pos[POSW], s_len[IDXW+1]
    out:  busy, done, next_tok[TOKW], tok_valid
Boot: inference is released by `boot_loader.done` (the ~28 GB resident set is DMA'd
Flash->DDR5 first), NOT by power-on -- see docs/OPERATION_FLOW.md sec 1.

What crosses USB-C is ONLY these token IDs + control; the heavy weight/KV traffic is
internal to the device. Session KV lives in the device's DDR5 (per session).

Scope (honest): this is the D2 software scaffold (docs/USBC_PRODUCT_PLAN.md). The
protocol + generation loop are real; MockDevice returns a clearly-labelled canned
response (the plumbing, not the model). The real backend is either the RTL simulator
(slice model) or the shipped device / full-model runtime.
"""

from __future__ import annotations

import abc
import time


class DeviceState:
    OFF = "off"
    BOOTING = "booting"          # boot_loader streaming resident set Flash->DDR5
    READY = "ready"              # boot_loader.done asserted -> inference released
    BUSY = "busy"                # a decode step in flight


class AIPUDevice(abc.ABC):
    """Abstract AIPU device. Concrete backends: MockDevice (here), a simulator-backed
       driver, or the real USB-C driver. The generation loop in aipu_server.py uses
       ONLY this interface."""

    #: vocabulary size the device decodes over (real GLM-5.2: 154880; scaffold: 256).
    vocab_size: int = 256
    #: model identifier surfaced to OpenAI clients.
    model_id: str = "aipu-glm-5.2-fp8"

    def __init__(self) -> None:
        self.state = DeviceState.OFF
        self._boot_t0 = None

    # ---- lifecycle (mirrors power-on -> boot_loader.done -> ready) ---------------
    @abc.abstractmethod
    def _boot_seconds(self) -> float:
        """Modelled boot duration (resident-set load). Real device ~1-2 s [EST]."""

    def power_on(self) -> None:
        """Begin the boot sequence (async). Ready is gated by boot_loader.done."""
        if self.state in (DeviceState.READY, DeviceState.BUSY):
            return
        self.state = DeviceState.BOOTING
        self._boot_t0 = time.monotonic()

    def poll_ready(self) -> bool:
        """True once boot_loader.done would be asserted (resident set in DDR5)."""
        if self.state == DeviceState.OFF:
            self.power_on()
        if self.state == DeviceState.BOOTING:
            if time.monotonic() - (self._boot_t0 or 0) >= self._boot_seconds():
                self.state = DeviceState.READY
        return self.state in (DeviceState.READY, DeviceState.BUSY)

    def wait_ready(self, timeout: float = 30.0) -> None:
        t0 = time.monotonic()
        while not self.poll_ready():
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("AIPU device did not reach READY (boot_loader.done)")
            time.sleep(0.02)

    # ---- session (KV lives in the device's DDR5, per session) --------------------
    @abc.abstractmethod
    def reset_session(self) -> None:
        """Clear the KV cache / conversation state for a fresh sequence."""

    # ---- one decode step (mirrors: assert start+inputs -> tok_valid+next_tok) ----
    @abc.abstractmethod
    def step(self, prompt_tok: int, start_pos: int, s_len: int) -> int:
        """Feed one token at `start_pos` (with running length `s_len`) and return the
           device's next-token id. This is EXACTLY the RTL host handshake:
             start=1, prompt_tok, start_pos, s_len  ->  (busy) -> tok_valid, next_tok."""

    #: the token id that ends generation (EOS). Real value comes from the tokenizer.
    eos_token: int = -1

    def prefill(self, prompt_ids, start_pos: int) -> int:
        """Feed the prompt tokens and return the FIRST next-token. Default = real
           device semantics: one `step` per prompt token, the last output is the
           first generated token (the model's KV is built along the way)."""
        last_out = None
        pos = start_pos
        for tok in prompt_ids:
            last_out = self.step(tok, pos, pos + 1)
            pos += 1
        return last_out

    def generate(self, prompt_ids, max_new_tokens: int, start_pos: int = 0):
        """Autoregressive loop: prefill `prompt_ids` -> first token, then feed each
           generated token back. Yields token ids as they decode (server streams).
           Stops at eos_token or max_new_tokens."""
        self.wait_ready()
        self.reset_session()
        cur = self.prefill(prompt_ids, start_pos)
        pos = start_pos + len(prompt_ids)
        produced = 0
        while produced < max_new_tokens and cur is not None and cur != self.eos_token:
            yield cur
            produced += 1
            cur = self.step(cur, pos, pos + 1)
            pos += 1


class MockDevice(AIPUDevice):
    """A backend that implements the full protocol but returns a clearly-labelled
       canned response byte-stream. Proves the end-to-end plumbing (HTTP -> tokenize
       -> device protocol -> detokenize -> HTTP streaming) WITHOUT pretending to be
       the model. Uses a byte-level vocab (0..255) so text round-trips exactly."""

    vocab_size = 256
    eos_token = 256                                 # out-of-band EOS (not a byte)

    def __init__(self, boot_seconds: float = 0.4, reply: str | None = None) -> None:
        super().__init__()
        self._boot = boot_seconds
        self._reply_bytes: list[int] = []
        self._cursor = 0
        self._default_reply = reply

    def _boot_seconds(self) -> float:
        return self._boot

    def reset_session(self) -> None:
        # New sequence: reset the decode cursor (analogous to clearing the device's
        # DDR5 KV). The per-turn reply set by the server via set_reply() survives --
        # set_reply is called for THIS turn just before generate().
        self._cursor = 0

    def set_reply(self, text: str) -> None:
        """Server sets the canned reply for this turn (derived from the prompt)."""
        self._reply_bytes = list(text.encode("utf-8")) + [self.eos_token]
        self._cursor = 0

    def _ensure_reply(self) -> None:
        if not self._reply_bytes:
            self.set_reply(self._default_reply
                           or "[AIPU mock device] protocol OK -- real tokens need the "
                              "hardware/full-model backend.")

    def prefill(self, prompt_ids, start_pos: int) -> int:
        # Mock: the prompt builds no real KV and does NOT consume the canned reply;
        # the first decode token is reply[0]. (A real device would stream prompt
        # tokens through glm_fp8_system_cdc to build its DDR5 KV.)
        self._ensure_reply()
        self._cursor = 0
        out = self._reply_bytes[0] if self._reply_bytes else self.eos_token
        self._cursor = 1
        return out

    def step(self, prompt_tok: int, start_pos: int, s_len: int) -> int:
        # Mock decode: emit the next byte of the canned reply.
        self.state = DeviceState.BUSY
        self._ensure_reply()
        out = (self._reply_bytes[self._cursor]
               if self._cursor < len(self._reply_bytes) else self.eos_token)
        self._cursor += 1
        self.state = DeviceState.READY
        return out
