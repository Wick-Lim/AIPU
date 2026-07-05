"""
aipu_sim_backend.py -- an AIPUDevice backed by the RTL SIMULATOR (iverilog/vvp).

Drives the committed `glm_model_fp8` slice (its `vvp` build) and returns the REAL
argmax next-token ids the RTL forward pass produces, wired into the device protocol.
This is the "server -> real RTL -> real token" co-simulation path.

HONEST SCOPE (all real, all caveated):
  * SLOW -- the slice `vvp` run is ~12 min (measured: 752 s). Fine for co-sim /
    bring-up, NOT for interactive use. The result is cached per process.
  * SLICE model -- VOCAB=256, small/untrained: the tokens are genuine RTL outputs
    (e.g. {4, 31, 20}) but NOT language. This validates the datapath + protocol,
    not fluency.
  * FIXED vectors -- runs the testbench's built-in prompts, not an arbitrary user
    prompt. Driving glm_model_fp8 from a live prompt needs the full weight/embedding/
    KV pull-port ROM harness (a larger TB effort); the committed TB already carries
    that harness for its own vectors, which we reuse here.

Build the binary first (~8 min, once):
    make unittests            # or: iverilog ... -o build/glm_model_fp8_sim (see Makefile)
"""

from __future__ import annotations

import os
import re
import subprocess

from aipu_device import AIPUDevice, DeviceState

_ARGMAX_RE = re.compile(r"argmax dut=(\d+)")
_REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))


class SimulatorBackend(AIPUDevice):
    vocab_size = 256
    eos_token = 256                                  # out-of-band sentinel

    def __init__(self, vvp_binary: str = "build/glm_model_fp8_sim",
                 vvp: str = "vvp", timeout: float = 1800.0,
                 cwd: str | None = None) -> None:
        super().__init__()
        self.cwd = cwd or _REPO
        self.vvp_binary = vvp_binary
        self.vvp = vvp
        self.timeout = timeout
        self._tokens: list[int] | None = None
        self._cursor = 0

    def _boot_seconds(self) -> float:
        return 0.0                                   # the binary is already built

    def reset_session(self) -> None:
        self._cursor = 0                             # keep the cached RTL tokens

    def _binary_path(self) -> str:
        p = self.vvp_binary
        return p if os.path.isabs(p) else os.path.join(self.cwd, p)

    def _run_rtl(self) -> list[int]:
        """Run the vvp slice sim ONCE and parse the RTL argmax tokens (~12 min)."""
        binp = self._binary_path()
        if not os.path.exists(binp):
            raise FileNotFoundError(
                f"vvp binary {binp} not built. Build it first (~8 min): "
                f"`make unittests` (or the glm_model_fp8_sim iverilog line in the Makefile).")
        proc = subprocess.run([self.vvp, binp], cwd=self.cwd,
                              capture_output=True, text=True, timeout=self.timeout)
        toks = [int(m) for m in _ARGMAX_RE.findall(proc.stdout)]
        if not toks:
            raise RuntimeError(
                f"no 'argmax dut=<id>' lines in vvp output (rc={proc.returncode}); "
                f"tail: {proc.stdout[-400:]!r}")
        return toks

    def _ensure(self) -> None:
        if self._tokens is None:
            self.state = DeviceState.BUSY
            self._tokens = self._run_rtl()
            self.state = DeviceState.READY

    def prefill(self, prompt_ids, start_pos: int) -> int:
        self._ensure()
        self._cursor = 0
        out = self._tokens[0] if self._tokens else self.eos_token
        self._cursor = 1
        return out

    def step(self, prompt_tok: int, start_pos: int, s_len: int) -> int:
        self._ensure()
        out = (self._tokens[self._cursor]
               if self._cursor < len(self._tokens) else self.eos_token)
        self._cursor += 1
        return out


if __name__ == "__main__":
    # Manual smoke (SLOW ~12 min): python3 host/aipu_sim_backend.py
    dev = SimulatorBackend()
    dev.power_on()
    print("running the RTL slice sim (~12 min)...")
    toks = list(dev.generate(prompt_ids=[1, 2, 3], max_new_tokens=16))
    print(f"RTL argmax tokens: {toks}")
