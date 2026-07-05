#!/usr/bin/env python3
"""Self-tests for the AIPU host scaffold (no HTTP, no network): device protocol,
   tokenizers (byte + real GLM BPE if available), boot gate, and the server
   generation loop. Run:  python3 host/test_aipu.py   ->  exit 0 on PASS."""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from aipu_device import MockDevice, DeviceState              # noqa: E402
from aipu_tokenizer import ByteTokenizer, make_tokenizer, GLMTokenizer  # noqa: E402
from aipu_server import AIPUServer                           # noqa: E402


def _byte_dev(**kw):
    return MockDevice(eos_token=ByteTokenizer.eos_id, vocab_size=256, **kw)


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
    """SimulatorBackend parse + protocol via a fake `vvp` (echo the argmax lines the
       real slice sim prints). The real vvp run is separately validated to emit
       exactly these tokens {4,31,20} in ~752 s -- too slow for CI, so we stub the
       subprocess and check the parse + device protocol."""
    import os
    import tempfile
    from aipu_sim_backend import SimulatorBackend
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("PASS[t] worst_rel=0 argmax dut=4 ref=4\n"
                "PASS argmax dut=31 ref=31\nPASS argmax dut=20 ref=20\n"
                "ALL 3 TESTS PASSED\n")
        path = f.name
    try:
        dev = SimulatorBackend(vvp="cat", vvp_binary=path, cwd="/tmp")
        dev.power_on()
        assert list(dev.generate([1, 2, 3], max_new_tokens=16)) == [4, 31, 20]
        dev.reset_session()
        assert list(dev.generate([9], max_new_tokens=16)) == [4, 31, 20]   # cached
    finally:
        os.unlink(path)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"AIPU host scaffold: ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
