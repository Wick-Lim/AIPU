#!/usr/bin/env bash
#=============================================================================
# tools/cov_run.sh -- Verilator structural code-coverage driver for the
#                     GLM-5.2-FP8 accelerator (invoked by `make coverage`).
#
# For each (module | primary-src | full source list) in the WORKING SET below:
#   1. verilate the SystemVerilog testbench + its RTL sources into a runnable
#      --binary sim with --coverage-line --coverage-toggle,
#   2. run the sim (from the repo root, so file-reading TBs resolve their
#      inputs), directing the per-run coverage database to build/cov/<mod>/
#      via the +verilator+coverage+file+ plusarg,
#   3. filter that database to the module's OWN primary source file and read
#      the line / toggle / branch coverage with verilator_coverage,
#   4. merge every per-run database into build/cov/merged.dat.
#
# This measures STRUCTURAL (line + toggle + branch) coverage of the committed
# "slice" configuration under the existing behavioral TBs. It is NOT a
# functional / full-config coverage claim -- the bit-fidelity proof is the
# byte-identical iverilog suite (`make unittests`). See docs/COVERAGE.md.
#
# Only modules whose TB verilates AND runs cleanly under Verilator 5.x are
# listed. TBs that use process::self()/event constructs that trip Verilator's
# --timing codegen, or whose bit-exact FP self-check depends on $exp/$sqrt
# real-number semantics that differ from the reference iverilog sim, are
# documented as out-of-scope in docs/COVERAGE.md and NOT listed here.
#=============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERILATOR="${VERILATOR:-verilator}"
VCOV="${VERILATOR_COVERAGE:-verilator_coverage}"
COVDIR="build/cov"
SUMMARY="$COVDIR/summary.txt"
MERGED="$COVDIR/merged.dat"
mkdir -p "$COVDIR"

# ---- WORKING SET: "module | primary_src | tb + srcs" ----------------------
# (source lists mirror the per-module combos in the Makefile `unittests` target)
WORKING=(
"fp8_e4m3|src/fp8_e4m3.vh|test/fp8_e4m3_tb.v"
"glm_matmul_fp8|src/glm_matmul_fp8.v|test/glm_matmul_fp8_tb.v src/glm_matmul_fp8.v src/glm_fp_pipe.v"
"glm_act|src/glm_act.v|test/glm_act_tb.v src/glm_act.v"
"glm_softmax|src/glm_softmax.v|test/glm_softmax_tb.v src/glm_softmax.v src/glm_fp_pipe.v"
"rope_interleave_unit|src/rope_interleave_unit.v|test/rope_interleave_unit_tb.v src/rope_interleave_unit.v"
"sampler|src/sampler.v|test/sampler_tb.v src/sampler.v src/topk_select.v src/glm_softmax.v src/glm_fp_pipe.v"
"weight_decomp|src/weight_decomp.v|test/weight_decomp_tb.v src/weight_decomp.v"
"clk_en_ctrl|src/clk_en_ctrl.v|test/clk_en_ctrl_tb.v src/clk_en_ctrl.v"
"clk_throttle|src/clk_throttle.v|test/clk_throttle_tb.v src/clk_throttle.v src/clk_en_ctrl.v"
"icg_cell|src/icg_cell.v|test/icg_cell_tb.v src/icg_cell.v"
"ecc_secded|src/ecc_secded.v|test/ecc_secded_tb.v src/ecc_secded.v"
"reset_sync|src/reset_sync.v|test/reset_sync_tb.v src/reset_sync.v"
"mbist_ctrl|src/mbist_ctrl.v|test/mbist_ctrl_tb.v src/mbist_ctrl.v"
"ecc_mem_wrap|src/ecc_mem_wrap.v|test/ecc_mem_wrap_tb.v src/ecc_mem_wrap.v src/ecc_secded.v"
"kv_ecc_ring|src/kv_ecc_ring.v|test/kv_ecc_ring_tb.v src/kv_ecc_ring.v src/ecc_secded.v"
)

# weight_decomp_tb reads a python-generated FP8 vector (relative to repo root).
if [ ! -f scratchpad/wd_vec.txt ]; then
  mkdir -p scratchpad
  python3 tools/fp8_gen.py gen scratchpad/wd_vec.txt >/dev/null 2>&1 || true
fi

# pull "NN.N%" and "(a/b)" out of a verilator_coverage summary line, e.g.
#   "  line      : 82.1% (   23/   28)"  ->  "82.1|23|28"
# (verilator right-pads small counts with spaces, so strip ()%/ first, portably)
metric() { # $1=covfile  $2=line|toggle|branch  -> "pct|num|den"
  "$VCOV" "$1" 2>/dev/null | grep -E "^  $2 " \
    | sed -E 's/[()%]//g; s#/# #g' \
    | awk '{print $3"|"$4"|"$5; exit}'
}

: > "$SUMMARY"
MERGE_INPUTS=()
NWORK=0; NFAIL=0

printf '%-22s %-8s %-10s %-8s %-10s %-8s %-10s  %s\n' \
  MODULE LINE% "(cov/tot)" TOGGLE% "(cov/tot)" BRANCH% "(cov/tot)" STATUS | tee -a "$SUMMARY"
printf '%.0s-' {1..104} | tee -a "$SUMMARY"; echo | tee -a "$SUMMARY"

for entry in "${WORKING[@]}"; do
  IFS='|' read -r mod prim srcs <<< "$entry"
  tb=$(echo "$srcs" | awk '{print $1}')
  top=$(basename "$tb" .v)
  d="$COVDIR/$mod"
  rm -rf "$d"; mkdir -p "$d"

  # 1. build
  if ! "$VERILATOR" --binary --coverage-line --coverage-toggle -Isrc \
         --Mdir "$d" -Wno-fatal --timing --top-module "$top" \
         $srcs > "$d/build.log" 2>&1 || [ ! -x "$d/V$top" ]; then
    printf '%-22s %s\n' "$mod" "BUILD-FAIL (see $d/build.log)" | tee -a "$SUMMARY"
    NFAIL=$((NFAIL+1)); continue
  fi

  # 2. run from repo root; direct coverage db into the module dir
  if ! "./$d/V$top" +verilator+coverage+file+"$d/coverage.dat" > "$d/run.log" 2>&1 \
       || ! grep -qE 'ALL [0-9]+ TESTS PASSED' "$d/run.log" \
       || [ ! -f "$d/coverage.dat" ]; then
    printf '%-22s %s\n' "$mod" "RUN-FAIL (see $d/run.log)" | tee -a "$SUMMARY"
    NFAIL=$((NFAIL+1)); continue
  fi

  # 3. filter to this module's own primary source, read metrics
  grep -F "$prim" "$d/coverage.dat" > "$d/cov_dut.dat" 2>/dev/null
  IFS='|' read -r lp ln ld <<< "$(metric "$d/cov_dut.dat" line)"
  IFS='|' read -r tp tn td <<< "$(metric "$d/cov_dut.dat" toggle)"
  IFS='|' read -r bp bn bd <<< "$(metric "$d/cov_dut.dat" branch)"
  [ -z "${td:-}" ] && td=0
  tshow="${tp:-0.0}"; [ "${td:-0}" = "0" ] && tshow="n/a"
  bshow="${bp:-0.0}"; [ "${bd:-0}" = "0" ] && bshow="n/a"
  printf '%-22s %-8s %-10s %-8s %-10s %-8s %-10s  %s\n' \
    "$mod" "${lp:-0.0}" "(${ln:-0}/${ld:-0})" \
    "$tshow" "(${tn:-0}/${td:-0})" \
    "$bshow" "(${bn:-0}/${bd:-0})" "PASS" | tee -a "$SUMMARY"

  MERGE_INPUTS+=("$d/coverage.dat")
  NWORK=$((NWORK+1))
done

# 4. merge every per-run database
if [ ${#MERGE_INPUTS[@]} -gt 0 ]; then
  "$VCOV" --write "$MERGED" "${MERGE_INPUTS[@]}" >/dev/null 2>&1
fi

echo | tee -a "$SUMMARY"
echo "== Merged design-source coverage (all src/ points, TB excluded) ==" | tee -a "$SUMMARY"
if [ -f "$MERGED" ]; then
  grep -a $'\002src/' "$MERGED" > "$COVDIR/merged_src.dat" 2>/dev/null
  "$VCOV" "$COVDIR/merged_src.dat" 2>/dev/null | grep -E 'line|toggle|branch' | tee -a "$SUMMARY"
fi
echo | tee -a "$SUMMARY"
echo "coverage: $NWORK module(s) measured, $NFAIL skipped. DB: $MERGED  Summary: $SUMMARY" | tee -a "$SUMMARY"
[ "$NFAIL" -eq 0 ]
