#!/usr/bin/env python3
"""Self-tests for the AIPU host scaffold (no HTTP, no network): device protocol,
   byte tokenizer round-trip, boot gate, and the server generation loop.
   Run:  python3 host/test_aipu.py   ->  exit 0 on PASS."""
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from aipu_device import MockDevice, DeviceState              # noqa: E402
from aipu_server import ByteTokenizer, AIPUServer            # noqa: E402


def test_tokenizer_roundtrip():
    tok = ByteTokenizer()
    for s in ["hello", "AIPU 디바이스", "emoji 🚀 test", ""]:
        ids = tok.encode(s)
        st = tok.stream()
        out = "".join(st.push(i) for i in ids) + st.flush()
        assert out == s, (s, out)


def test_boot_gate():
    d = MockDevice(boot_seconds=0.3)
    d.power_on()
    assert d.state == DeviceState.BOOTING
    assert not d.poll_ready()                    # not ready before boot_loader.done
    d.wait_ready(timeout=5)
    assert d.state in (DeviceState.READY, DeviceState.BUSY)


def test_generate_streams_full_reply():
    d = MockDevice(boot_seconds=0.0)
    d.set_reply("ABCDE")
    out = list(d.generate(prompt_ids=[104, 105], max_new_tokens=100))   # "hi"
    # bytes of "ABCDE" then EOS (eos not yielded)
    assert out == [65, 66, 67, 68, 69], out


def test_generate_respects_max_tokens():
    d = MockDevice(boot_seconds=0.0)
    d.set_reply("A very long canned reply that exceeds the cap")
    out = list(d.generate(prompt_ids=[104], max_new_tokens=5))
    assert len(out) == 5, out


def test_server_end_to_end():
    d = MockDevice(boot_seconds=0.0)
    srv = AIPUServer(d, ByteTokenizer())
    msgs = [{"role": "user", "content": "hello aipu"}]
    text, n_prompt = srv.generate_text(msgs, max_tokens=500)
    assert "protocol round-trip OK" in text or "protocol OK" in text, text
    assert n_prompt > 0
    # streaming yields the same text
    d.reset_session()
    streamed = "".join(srv.generate_stream(msgs, max_tokens=500))
    assert streamed == text, (streamed, text)


def test_prefill_does_not_truncate():
    # regression: prefill steps must NOT consume the canned reply (the first-cut bug).
    d = MockDevice(boot_seconds=0.0)
    d.set_reply("HELLO")
    long_prompt = list(range(50))                # 50 prompt tokens
    out = bytes(d.generate(long_prompt, max_new_tokens=100)).decode()
    assert out == "HELLO", out


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"AIPU host scaffold: ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
