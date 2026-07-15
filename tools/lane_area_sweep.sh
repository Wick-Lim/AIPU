#!/usr/bin/env bash
# lane_area_sweep.sh -- measure the MARGINAL gate cost of ONE MAC "lane".
#
# WHY: R3_APPLIANCE_SPEC.md §3 sizes the array in "lanes" and priced one at
# "dequant+4MAC ~= 2-3만 게이트", then concluded 3K lanes ~= "다이 수 mm²". Those two
# statements are inconsistent, and nothing in the repo ever measured either. This does.
#
# WHAT A LANE IS: glm_matmul_q4k is a PE_M x PE_N array; one PE_N COLUMN is one Q4_K
# dequant + PE_M MACs -- exactly §3's "lane". So the MARGINAL cost of +1 PE_N is the
# per-lane cost. We sweep PE_N and take the SLOPE, which cancels the fixed control/FSM
# overhead that a single absolute number would wrongly fold in.
#
# HONEST SCOPE -- read before quoting the number:
#   * generic yosys gate mapping (abc -g simple), NOT a PDK. The GE weights below are
#     OUR convention, stated here so the number is auditable rather than magic.
#   * glm_matmul_q4k self-describes as "the CORRECT-FIRST reference core" with a
#     COMBINATIONAL fp32 dequant+MAC per beat (src/glm_matmul_q4k.v:32). A real ASIC
#     lane would be repipelined: possibly smaller (resource sharing) or larger
#     (retiming registers). This therefore bounds the REFERENCE core, and is [EST] for
#     silicon. It is enough to settle WHICH of §3's two statements is wrong; it is not
#     a tapeout number, and it must not be quoted as "measured area".
#   * no sky130/OpenLane on this machine, and sky130 is 130nm anyway (wrong node) --
#     a real PDK number remains [측정필요].
#
# KMAX=32, not 256: KMAX sets the k-counter/accumulator width, which is FIXED overhead,
# not per-lane -- verified, PE_N=1 measures 44,747 GE at KMAX=256 vs 41,563 at KMAX=32, a
# constant 3,184 that the SLOPE cancels. KMAX=256 makes PE_N=2/3 synthesis blow up (it did,
# twice) for a number the slope does not depend on.
#
# Result (yosys 0.66, PE_M=4): PE_N=1/2/3 = 41,563 / 80,787 / 121,550 GE
#   -> slope = 39,994 GE per lane, intercept 1,570; deltas 39,225 and 40,763 = 3.8% spread,
#      i.e. genuinely linear. §3's '2-3만/lane' is the right order, low by ~1.3-1.6x.
#      §3's '3K lanes ~= 다이 수 mm²' is NOT: 3,072 lanes = 122.9M GE = 14.0-19.3 mm².
#
# Runtime: ~10 min.
# Usage: tools/lane_area_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
command -v yosys >/dev/null || { echo "yosys not on PATH"; exit 1; }
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

for N in 1 2 3; do
  echo "  synthesizing PE_N=$N ..." >&2
  cat > "$OUT/s$N.ys" <<EOF
read_verilog -I src src/glm_matmul_q4k.v
chparam -set PE_M 4 -set PE_N $N -set KMAX 32 glm_matmul_q4k
hierarchy -top glm_matmul_q4k
synth -top glm_matmul_q4k -flatten
abc -g simple
opt -full
stat
EOF
  yosys -s "$OUT/s$N.ys" -l "$OUT/r$N.log" >/dev/null 2>&1 \
    || { echo "yosys failed at PE_N=$N"; tail -20 "$OUT/r$N.log"; exit 1; }
done

python3 - "$OUT" <<'PY'
import re, sys, pathlib

# NAND2-equivalent weights -- OUR convention, stated so the result is auditable.
# Ordered LONGEST-KEY-FIRST: "$_SDFFE_PP0P_" must not fall through to a "DFF" rule,
# and "ANDNOT" must not be counted as "AND". A silently-unmatched cell would
# under-count the lane, which is exactly the kind of quiet error this file exists to
# stop -- so unmatched cells are reported, not ignored.
W = [("SDFFE", 5.5), ("SDFFCE", 5.5), ("SDFF", 5.5), ("DFFE", 5.5), ("DFFSR", 6.0),
     ("DFF", 5.5), ("ALDFF", 6.0), ("DLATCH", 4.0),
     ("ANDNOT", 1.33), ("ORNOT", 1.33), ("NAND", 1.0), ("NOR", 1.0),
     ("XNOR", 2.67), ("XOR", 2.67), ("AND", 1.33), ("OR", 1.33),
     ("NMUX", 2.33), ("MUX", 2.33), ("NOT", 0.67), ("BUF", 1.0),
     ("AOI3", 1.67), ("OAI3", 1.67), ("AOI4", 2.0), ("OAI4", 2.0)]

def ge(log):
    txt = pathlib.Path(log).read_text()
    start = txt.rfind("Printing statistics")           # the FINAL stat block
    total, unknown = 0.0, []
    for line in txt[start:].splitlines():
        m = re.match(r"\s+(\d+)\s+\$_(\w+?)_\s*$", line)   # yosys prints COUNT then NAME
        if not m:
            continue
        n, cell = int(m.group(1)), m.group(2)
        for k, w in W:
            if cell.startswith(k):
                total += n * w
                break
        else:
            unknown.append((cell, n))
    if unknown:
        print(f"  !! UNMATCHED cells (not counted): {unknown}")
    return total

d = sys.argv[1]
g = {n: ge(f"{d}/r{n}.log") for n in (1, 2, 3)}
print("\nPE_N (lanes) |  NAND2-equivalents")
for n in (1, 2, 3):
    print(f"     {n}       |  {g[n]:>12,.0f}")
s1, s2 = g[2] - g[1], g[3] - g[2]
slope = (g[3] - g[1]) / 2
print(f"\nmarginal per lane: 1->2 = {s1:,.0f} | 2->3 = {s2:,.0f} | slope = {slope:,.0f} GE/lane")
print(f"linearity check: {abs(s1-s2)/slope*100:.1f}% spread  (large spread => not a clean per-lane cost)")
print(f"fixed overhead (intercept) ~= {g[1] - slope:,.0f} GE")
print(f"\n§3 claims 2-3만 GE/lane  ->  measured ~{slope/1e4:.1f}만  ({slope/25000:.2f}x its midpoint)")
for lanes, label in ((3072, "3,072 (retired §3 table)"), (7263, "7,263 (re-derived @1.54TB/s)")):
    tot = lanes * slope
    lo, hi = tot * 0.08e-6, tot * 0.11e-6     # mm^2 @ 0.08-0.11 um^2/GE [EST, EXTERNAL]
    print(f"  {label:<30} {tot/1e6:>6.0f}M GE -> {lo/0.7:>5.1f}-{hi/0.7:>5.1f} mm2 @70% util"
          f"  ({lo/0.7/480*100:>4.1f}-{hi/0.7/480*100:>4.1f}% of a 480mm2 die)")
print("\n[EST] 0.08-0.11 um2/GE is an OUTSIDE density figure, NOT repo-sourced -- the")
print("      dominant uncertainty here. See header for the reference-core caveat.")
PY
