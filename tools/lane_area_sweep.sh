#!/usr/bin/env bash
# lane_area_sweep.sh -- measure the MARGINAL gate cost of ONE MAC "lane".
#
# WHY: R3_APPLIANCE_SPEC.md §3 sizes the array in "lanes" and priced one at
# "dequant+4MAC ~= 2-3만 게이트", then concluded 3K lanes ~= "다이 수 mm²". Those two
# statements are inconsistent, and nothing in the repo measured either. This does.
#
# WHAT A LANE IS: glm_matmul_q4k is a PE_M x PE_N array; one PE_N COLUMN is one Q4_K
# dequant + PE_M MACs -- exactly §3's "lane". So the MARGINAL cost of +1 PE_N is the
# per-lane cost. We sweep PE_N and take the slope, which cancels the fixed control/FSM
# overhead that a single absolute number would wrongly fold in.
#
# HONEST SCOPE -- read before quoting the number:
#   * generic yosys gate mapping, NOT a PDK. GE weights below are OUR convention.
#   * glm_matmul_q4k self-describes as "the CORRECT-FIRST reference core" with a
#     COMBINATIONAL fp32 dequant+MAC per beat (src/glm_matmul_q4k.v:32). A real ASIC
#     lane would be repipelined: possibly smaller (sharing) or larger (retiming regs).
#   * therefore this bounds the REFERENCE core, and is [EST] for silicon. It is enough
#     to settle which of §3's two statements is wrong; it is NOT a tapeout number.
#
# Usage: tools/lane_area_sweep.sh          (needs yosys on PATH)
set -euo pipefail
cd "$(dirname "$0")/.."
command -v yosys >/dev/null || { echo "yosys not on PATH"; exit 1; }
OUT=$(mktemp -d)

for N in 1 2 3; do
  cat > "$OUT/s$N.ys" <<EOF
read_verilog -I src -DPE_N_SWEEP src/glm_matmul_q4k.v
chparam -set PE_M 4 -set PE_N $N -set KMAX 256 glm_matmul_q4k
hierarchy -top glm_matmul_q4k
synth -top glm_matmul_q4k -flatten
opt -full
stat
EOF
  yosys -q -s "$OUT/s$N.ys" -l "$OUT/r$N.log" 2>/dev/null || { echo "yosys failed at PE_N=$N"; cat "$OUT/r$N.log" | tail -20; exit 1; }
done

python3 - "$OUT" <<'PY'
import re, sys, pathlib
# NAND2-equivalent weights (OUR convention -- stated so the number is auditable).
W = {"NOT":0.67,"BUF":1.0,"AND":1.33,"NAND":1.0,"OR":1.33,"NOR":1.0,
     "XOR":2.67,"XNOR":2.67,"ANDNOT":1.33,"ORNOT":1.33,"MUX":2.33,
     "NMUX":2.33,"AOI3":1.67,"OAI3":1.67,"AOI4":2.0,"OAI4":2.0,"DFF":5.5}
def ge(log):
    tot = 0.0
    for line in pathlib.Path(log).read_text().splitlines():
        m = re.match(r"\s+\$_(\w+?)_\s+(\d+)", line)
        if m:
            cell, n = m.group(1), int(m.group(2))
            for k, w in W.items():
                if cell.startswith(k):
                    tot += n * w
                    break
    return tot
d = sys.argv[1]
g = {n: ge(f"{d}/r{n}.log") for n in (1, 2, 3)}
print("PE_N (lanes) |  NAND2-equivalents")
for n in (1, 2, 3):
    print(f"     {n}       |  {g[n]:>12,.0f}")
s1, s2 = g[2] - g[1], g[3] - g[2]
slope = (g[3] - g[1]) / 2
print(f"\nmarginal per lane: 1->2 = {s1:,.0f} | 2->3 = {s2:,.0f} | slope = {slope:,.0f} GE/lane")
print(f"fixed overhead (intercept) ~= {g[1] - slope:,.0f} GE")
print(f"\n§3 claims 2-3만 GE/lane -> measured ~{slope/1e4:.1f}만  ({slope/25000:.2f}x its midpoint)")
for lanes, label in ((3072, "3,072 (retired §3 table)"), (7263, "7,263 (re-derived, 1.54TB/s)")):
    tot = lanes * slope
    lo, hi = tot * 0.08e-6, tot * 0.11e-6          # mm^2 @ 0.08-0.11 um^2/GE [EST, external]
    print(f"  {label:<30} {tot/1e6:>6.0f}M GE -> {lo:>4.1f}-{hi:>4.1f} mm2 raw"
          f" -> {lo/0.7:>4.1f}-{hi/0.7:>4.1f} mm2 @70% util ({lo/0.7/480*100:.0f}-{hi/0.7/480*100:.0f}% of a 480mm2 die)")
print("\n[EST] the 0.08-0.11 um2/GE density is an OUTSIDE figure, not repo-sourced.")
PY
rm -rf "$OUT"
