#!/usr/bin/env python3
"""Self-tests for the AIPU host scaffold (no HTTP, no network): device protocol,
   tokenizers (byte + real GLM BPE if available), boot gate, and the server
   generation loop. Run:  python3 host/test_aipu.py   ->  exit 0 on PASS."""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from aipu_device import (MockDevice, AIPUDevice, DeviceState,   # noqa: E402
                         SamplingParams)
from aipu_tokenizer import ByteTokenizer, make_tokenizer, GLMTokenizer  # noqa: E402
from aipu_server import AIPUServer                           # noqa: E402
from aipu_chat_template import apply_chat_template            # noqa: E402


def _byte_dev(**kw):
    return MockDevice(eos_token=ByteTokenizer.eos_id, vocab_size=256, **kw)


class _FixedDevice(AIPUDevice):
    """Test device that yields a FIXED list of token ids -- bypasses MockDevice's
       canned-reply priming so tests can drive arbitrary decoded output (stop
       sequences, finish_reason). Records the SamplingParams it was configured with."""

    def __init__(self, ids, eos=ByteTokenizer.eos_id):
        super().__init__()
        self.eos_token = eos
        self._ids = list(ids)
        self._c = 0
        self.state = DeviceState.READY
        self.seen_sampling = None

    def _boot_seconds(self):
        return 0.0

    def reset_session(self):
        self._c = 0

    def configure_sampling(self, sampling):
        self.seen_sampling = sampling

    def prefill(self, prompt_ids, start_pos):
        self._c = 0
        out = self._ids[0] if self._ids else self.eos_token
        self._c = 1
        return out

    def step(self, prompt_tok, start_pos, s_len):
        out = self._ids[self._c] if self._c < len(self._ids) else self.eos_token
        self._c += 1
        return out


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    for s in ["hello", "AIPU 디바이스", "emoji 🚀 test", ""]:
        ids = tok.encode(s)
        st = tok.stream()
        out = "".join(st.push(i) for i in ids) + st.flush()
        assert out == s, (s, out)


def test_boot_gate():
    d = _byte_dev(boot_seconds=0.3)
    d.power_on()
    assert d.state == DeviceState.BOOTING
    assert not d.poll_ready()                    # not ready before boot_loader.done
    d.wait_ready(timeout=5)
    assert d.state in (DeviceState.READY, DeviceState.BUSY)


def test_generate_streams_full_reply():
    d = _byte_dev(boot_seconds=0.0)
    d.set_reply_ids(list(b"ABCDE"))
    out = list(d.generate(prompt_ids=[104, 105], max_new_tokens=100))   # "hi"
    assert out == [65, 66, 67, 68, 69], out      # eos not yielded


def test_generate_respects_max_tokens():
    d = _byte_dev(boot_seconds=0.0)
    d.set_reply_ids(list(b"A very long canned reply that exceeds the cap"))
    out = list(d.generate(prompt_ids=[104], max_new_tokens=5))
    assert len(out) == 5, out


def test_prefill_does_not_truncate():
    # regression: prefill steps must NOT consume the reply (the first-cut bug).
    d = _byte_dev(boot_seconds=0.0)
    d.set_reply_ids(list(b"HELLO"))
    long_prompt = list(range(50))                # 50 prompt tokens
    out = bytes(d.generate(long_prompt, max_new_tokens=100)).decode()
    assert out == "HELLO", out


def test_server_end_to_end_byte():
    d = _byte_dev(boot_seconds=0.0)
    srv = AIPUServer(d, ByteTokenizer())
    msgs = [{"role": "user", "content": "hello aipu"}]
    text, n_prompt = srv.generate_text(msgs, max_tokens=500)
    assert "protocol round-trip OK" in text, text
    assert "byte tokenizer" in text, text
    assert n_prompt > 0
    d.reset_session()
    streamed = "".join(srv.generate_stream(msgs, max_tokens=500))
    assert streamed == text, (streamed, text)


def test_glm_tokenizer_if_available():
    """Real GLM BPE round-trip + server end-to-end -- SKIPPED if `tokenizers` or
       tokenizer.json aren't present (byte fallback covers the plumbing)."""
    tok = make_tokenizer()
    if not isinstance(tok, GLMTokenizer):
        print("  SKIP test_glm_tokenizer_if_available (no tokenizers/tokenizer.json)")
        return
    for s in ["hello world", "AIPU 디바이스 test", "def f(x):\n    return x+1"]:
        ids = tok.encode(s)
        st = tok.stream()
        out = "".join(st.push(i) for i in ids) + st.flush()
        assert out == s, (s, out)
    # server end-to-end through the REAL GLM vocab
    d = MockDevice(boot_seconds=0.0, eos_token=tok.eos_id, vocab_size=tok.vocab_size)
    srv = AIPUServer(d, tok)
    msgs = [{"role": "user", "content": "hi"}]
    text, _ = srv.generate_text(msgs, max_tokens=500)
    assert "protocol round-trip OK" in text and "glm tokenizer" in text, text
    print(f"  (GLM vocab_size={tok.vocab_size}, eos_id={tok.eos_id})")


def test_simulator_backend_parse():
    """SimulatorBackend parse + protocol via a fake `vvp` (echo the per-case lines the
       real glm_model_q4k slice sim prints). The full vvp run is separately validated
       (`make model-q4k`) -- minutes-long, too slow for CI -- so we stub the subprocess
       with the exact TB output format and check the parse + device protocol. The stub
       tokens {13,3,13} are the real SPEC_SLICE argmax vectors (build/mq4k_s)."""
    import os
    import tempfile
    from aipu_sim_backend import SimulatorBackend
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("case 0: token=3 pos=1 s_len=2 -> argmax=13 (golden 13)    MATCH\n"
                "case 1: token=7 pos=3 s_len=2 -> argmax=3 (golden 3)    MATCH\n"
                "case 2: token=11 pos=0 s_len=1 -> argmax=13 (golden 13)    MATCH\n"
                "ALL 99 TESTS PASSED\n")
        path = f.name
    try:
        dev = SimulatorBackend(vvp="cat", vvp_binary=path, cwd="/tmp")
        dev.power_on()
        assert list(dev.generate([1, 2, 3], max_new_tokens=16)) == [13, 3, 13]
        dev.reset_session()
        assert list(dev.generate([9], max_new_tokens=16)) == [13, 3, 13]   # cached
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# TASK 1 -- GLM chat template
# ---------------------------------------------------------------------------
def test_chat_template_glm_structure():
    """GLM-5.2 template: leading [gMASK]<sop>, per-role special tokens in order, and
       the trailing generation prompt (thinking on -> <think>, off -> <think></think>)."""
    msgs = [{"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi there"}]
    p = apply_chat_template(msgs)
    assert p.startswith("[gMASK]<sop>"), p
    assert "<|system|>Reasoning Effort: Max" in p          # GLM-5.2 thinking-effort turn
    assert "<|system|>be nice" in p
    assert "<|user|>hi there" in p
    assert p.endswith("<|assistant|><think>"), p           # generation prompt (thinking)
    assert p.index("<|user|>hi there") < p.rindex("<|assistant|>")   # ordering
    # thinking disabled: no reasoning-effort turn, closed think block
    p2 = apply_chat_template(msgs, enable_thinking=False)
    assert "Reasoning Effort" not in p2
    assert p2.endswith("<|assistant|><think></think>"), p2
    # reasoning_effort=high -> "High"
    assert "Reasoning Effort: High" in apply_chat_template(msgs, reasoning_effort="high")
    # no generation prompt
    assert not apply_chat_template(msgs, add_generation_prompt=False).endswith("<think>")


def test_chat_template_clears_assistant_history():
    """Historical assistant turns render as <|assistant|><think></think>{answer} with
       prior <think> reasoning stripped (template default)."""
    p = apply_chat_template([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "<think>secret reasoning</think>the answer"},
        {"role": "user", "content": "b"}])
    assert "secret reasoning" not in p, p
    assert "<|assistant|><think></think>the answer" in p, p


def test_chat_template_multimodal_reminder():
    """Non-text content parts fall back to the template's media <reminder>."""
    p = apply_chat_template([{"role": "user", "content": [
        {"type": "text", "text": "look:"},
        {"type": "image_url", "image_url": {"url": "x"}}]}])
    assert "<|user|>look:" in p
    assert "unable to process this image" in p, p


def test_server_prompt_dispatch():
    """Server applies the GLM template when the tokenizer name is 'glm', else the naive
       flatten; --raw forces the flatten even for GLM."""
    d = _byte_dev(boot_seconds=0.0)
    msgs = [{"role": "user", "content": "hi"}]
    assert AIPUServer(d, ByteTokenizer()).prompt_text(msgs) == "user: hi"   # byte->flatten

    class _FakeGLM(ByteTokenizer):
        name = "glm"
    glm_prompt = AIPUServer(d, _FakeGLM()).prompt_text(msgs)
    assert glm_prompt.startswith("[gMASK]<sop>") and "<|user|>hi" in glm_prompt
    assert AIPUServer(d, _FakeGLM(), raw=True).prompt_text(msgs) == "user: hi"  # --raw


def test_chat_template_encodes_special_tokens_if_glm():
    """Through the REAL GLM BPE, template special tokens encode as SINGLE ids -- SKIPPED
       without tokenizers/tokenizer.json."""
    tok = make_tokenizer()
    if not isinstance(tok, GLMTokenizer):
        print("  SKIP test_chat_template_encodes_special_tokens_if_glm (no GLM tok)")
        return
    ids = tok.encode(apply_chat_template([{"role": "user", "content": "hi"}]))
    for sid in (154822, 154824, 154826, 154827, 154828):   # gMASK/sop/system/user/asst
        assert sid in ids, (sid, ids)


# ---------------------------------------------------------------------------
# TASK 2 -- OpenAI sampling parameters
# ---------------------------------------------------------------------------
def test_sampling_params_from_request():
    sp = SamplingParams.from_request({
        "temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_tokens": 32,
        "stop": ["\n\n", "END"], "seed": 123,
        "presence_penalty": 0.5, "frequency_penalty": 0.2})
    assert (sp.temperature, sp.top_p, sp.top_k) == (0.7, 0.9, 40)
    assert sp.max_tokens == 32 and sp.seed == 123
    assert sp.stop == ["\n\n", "END"]
    assert sp.presence_penalty == 0.5 and sp.frequency_penalty == 0.2
    assert SamplingParams.from_request({"stop": "STOP"}).stop == ["STOP"]   # str->list
    assert SamplingParams.from_request({}).stop == []
    bad = SamplingParams.from_request({"temperature": "hot", "max_tokens": None})
    assert bad.temperature == 1.0 and bad.max_tokens == 256                 # defaults


def test_stop_sequence_truncates():
    """A stop string in the decoded output truncates it (exclusive) and yields
       finish_reason 'stop' -- streaming-safe across a multi-token stop."""
    d = _FixedDevice(list(b"hello STOP world"))
    srv = AIPUServer(d, ByteTokenizer())
    sp = SamplingParams(max_tokens=100, stop=["STOP"])
    text, _, finish = srv.complete([{"role": "user", "content": "x"}], sp)
    assert text == "hello ", repr(text)
    assert finish == "stop", finish
    # streaming truncates identically (and never emits a partial stop)
    d.reset_session()
    streamed = "".join(srv.stream([{"role": "user", "content": "x"}], sp))
    assert streamed == "hello ", repr(streamed)


def test_finish_reason_length_vs_stop():
    """finish_reason: 'length' when max_tokens caps generation, 'stop' at natural eos."""
    d = _FixedDevice(list(b"ABCDEFGHIJ"))
    srv = AIPUServer(d, ByteTokenizer())
    text, _, finish = srv.complete([{"role": "user", "content": "x"}],
                                   SamplingParams(max_tokens=4))
    assert text == "ABCD" and finish == "length", (text, finish)
    d2 = _FixedDevice(list(b"AB"))
    text2, _, finish2 = AIPUServer(d2, ByteTokenizer()).complete(
        [{"role": "user", "content": "x"}], SamplingParams(max_tokens=100))
    assert text2 == "AB" and finish2 == "stop", (text2, finish2)


def test_sampling_params_reach_device():
    """temperature/top_p/top_k/seed are plumbed to the device (configure_sampling)."""
    d = _FixedDevice(list(b"hi"))
    sp = SamplingParams(max_tokens=8, temperature=0.3, top_k=5, seed=7)
    AIPUServer(d, ByteTokenizer()).complete([{"role": "user", "content": "x"}], sp)
    assert d.seen_sampling is sp, "sampling not plumbed to device"
    assert d.seen_sampling.temperature == 0.3 and d.seen_sampling.seed == 7


def test_mockdevice_accepts_and_ignores_sampling():
    """MockDevice accepts SamplingParams (records them) but IGNORES them: greedy replay
       is unchanged (honest -- true sampling is device-side, not faked on canned ids)."""
    d = _byte_dev(boot_seconds=0.0)
    d.set_reply_ids(list(b"XYZ"))
    sp = SamplingParams(max_tokens=100, temperature=0.0, top_k=1, seed=99)
    out = bytes(d.generate([1, 2, 3], max_new_tokens=100, sampling=sp)).decode()
    assert out == "XYZ", out                       # output unaffected by sampling
    assert d._sampling is sp                        # accepted/recorded


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"AIPU host scaffold: ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
