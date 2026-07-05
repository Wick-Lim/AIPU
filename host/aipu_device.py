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
import dataclasses
import time


@dataclasses.dataclass
class SamplingParams:
    """OpenAI-style sampling parameters, threaded from the HTTP request to the device.

    HONEST host-vs-device split (see host/README.md):
      * HOST-SIDE (this scaffold enforces): `max_tokens` (decode cap) and `stop`
        (truncate when a stop string appears in the decoded text) -- both real and
        applied in aipu_server.py. `seed` is threaded through to the device.
      * DEVICE-SIDE (`sampler.v` samples ON-DEVICE from logits): `temperature`,
        `top_p`, `top_k`, `seed`. The MockDevice returns the ARGMAX token (greedy) and
        IGNORES temperature/top_p/top_k -- true sampling needs a logits-capable
        backend, which programs these via `configure_sampling()`. Faking sampling on a
        canned/argmax stream would be dishonest, so we don't.
      * `presence_penalty`/`frequency_penalty` are accepted and ignored (no host-side
        logit bias in the scaffold).
    """

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0                                   # 0 = disabled
    stop: list = dataclasses.field(default_factory=list)
    seed: "int | None" = None
    presence_penalty: float = 0.0                    # accepted, ignored (device-side)
    frequency_penalty: float = 0.0                   # accepted, ignored (device-side)

    @classmethod
    def from_request(cls, req: dict) -> "SamplingParams":
        """Parse an OpenAI `/v1/chat/completions` request dict into SamplingParams,
           tolerating missing/malformed fields (falls back to defaults)."""
        def _num(key, default, cast):
            v = req.get(key, default)
            if v is None:
                return default
            try:
                return cast(v)
            except (TypeError, ValueError):
                return default

        stop = req.get("stop")
        if isinstance(stop, str):
            stops = [stop] if stop else []
        elif isinstance(stop, (list, tuple)):
            stops = [s for s in stop if isinstance(s, str) and s]
        else:
            stops = []

        seed = req.get("seed")
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed = None

        return cls(
            max_tokens=max(1, _num("max_tokens", 256, int)),
            temperature=_num("temperature", 1.0, float),
            top_p=_num("top_p", 1.0, float),
            top_k=_num("top_k", 0, int),
            stop=stops[:4],                          # OpenAI allows up to 4 stops
            seed=seed,
            presence_penalty=_num("presence_penalty", 0.0, float),
            frequency_penalty=_num("frequency_penalty", 0.0, float),
        )


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

    #: sampling programmed for the current generate() (None => greedy/argmax).
    _sampling: "SamplingParams | None" = None

    def configure_sampling(self, sampling: "SamplingParams") -> None:
        """Program device-side sampling for the next generate() call.

        The RTL `sampler.v` samples ON-DEVICE from the model's logits
        (temperature/top_p/top_k with `seed`). The MockDevice (and the slice
        SimulatorBackend) return the ARGMAX token -- i.e. GREEDY -- so they IGNORE
        temperature/top_p/top_k here: there are no host-side logits to sample from a
        canned/argmax stream, and faking it would be dishonest. A logits-capable
        backend OVERRIDES this to write the sampler's config registers; the default
        just records the params so a real backend / inspector can read them."""
        self._sampling = sampling

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

    def generate(self, prompt_ids, max_new_tokens: int, start_pos: int = 0,
                 sampling: "SamplingParams | None" = None):
        """Autoregressive loop: prefill `prompt_ids` -> first token, then feed each
           generated token back. Yields token ids as they decode (server streams).
           Stops at eos_token or max_new_tokens. `sampling` (optional) is programmed
           into the device via configure_sampling() -- honored device-side by a
           logits-capable backend; the mock is greedy and ignores it."""
        self.wait_ready()
        self.reset_session()
        if sampling is not None:
            self.configure_sampling(sampling)
        cur = self.prefill(prompt_ids, start_pos)
        pos = start_pos + len(prompt_ids)
        produced = 0
        while produced < max_new_tokens and cur is not None and cur != self.eos_token:
            yield cur
            produced += 1
            cur = self.step(cur, pos, pos + 1)
            pos += 1


class MockDevice(AIPUDevice):
    """A backend that implements the full protocol but REPLAYS a fixed list of token
       ids (set by the server for this turn). Tokenizer-agnostic -- the server encodes
       a clearly-labelled canned reply with whatever tokenizer is active (byte OR the
       real GLM BPE) and the mock replays those ids -- so it proves the end-to-end
       plumbing (HTTP -> tokenize -> device protocol -> detokenize -> HTTP streaming)
       for BOTH vocabularies WITHOUT pretending to be the model. A real device would
       run glm_fp8_system_cdc and emit real next-token ids here."""

    def __init__(self, boot_seconds: float = 0.4, eos_token: int = 256,
                 vocab_size: int = 256) -> None:
        super().__init__()
        self._boot = boot_seconds
        self.eos_token = eos_token                  # server sets this to tok.eos_id
        self.vocab_size = vocab_size
        self._reply_ids: list[int] = []
        self._cursor = 0

    def _boot_seconds(self) -> float:
        return self._boot

    def reset_session(self) -> None:
        # New sequence: reset the decode cursor (analogous to clearing the device's
        # DDR5 KV). The per-turn reply ids set by the server survive (set just before
        # generate()).
        self._cursor = 0

    def set_reply_ids(self, ids) -> None:
        """Server sets this turn's canned reply as token ids (in the active tokenizer's
           space), terminated by eos."""
        self._reply_ids = list(ids) + [self.eos_token]
        self._cursor = 0

    def prefill(self, prompt_ids, start_pos: int) -> int:
        # Mock: the prompt builds no real KV and does NOT consume the reply; the first
        # decode token is reply[0]. (A real device streams the prompt through
        # glm_fp8_system_cdc to build its DDR5 KV.)
        self._cursor = 0
        out = self._reply_ids[0] if self._reply_ids else self.eos_token
        self._cursor = 1
        return out

    def step(self, prompt_tok: int, start_pos: int, s_len: int) -> int:
        # Mock decode: emit the next reply id.
        self.state = DeviceState.BUSY
        out = (self._reply_ids[self._cursor]
               if self._cursor < len(self._reply_ids) else self.eos_token)
        self._cursor += 1
        self.state = DeviceState.READY
        return out
