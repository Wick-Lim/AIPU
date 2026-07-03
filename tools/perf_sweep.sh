#!/usr/bin/env bash
#=============================================================================
# tools/perf_sweep.sh -- cycle-accurate throughput sweep for glm_fp8_system.
#
#   Compiles test/glm_fp8_system_perf_tb.v ONCE per config (iverilog -P
#   overrides the TB's FLASH_LAT_CFG / DDR_NCH_CFG / CACHE_SLOTS_CFG params),
#   runs it, and greps the machine-readable "PERF ..." line into a table.
#
#   Sweeps (see the GLM roofline [EST] assumptions this measures on real RTL):
#     A) FLASH_LAT in {8,64,256,1024}  (DDR_NCH=4, CACHE_SLOTS=4)  -> stall growth
#     B) DDR_NCH   in {1,2,4}          (FLASH_LAT=256, CACHE_SLOTS=4) -> channel scaling
#     C) CACHE_SLOTS in {2,4}          (FLASH_LAT=256, DDR_NCH=4)   -> cache-size effect
#
#   NOTE: uses bash (NOT zsh) so unquoted $SRC word-splits into file args.
#   Run from the repo root:  bash tools/perf_sweep.sh
#=============================================================================
set -u

IVERILOG="${IVERILOG:-iverilog}"
VVP="${VVP:-vvp}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="build/perf_sweep"
mkdir -p "$BUILD"

TB=test/glm_fp8_system_perf_tb.v
TBMOD=glm_fp8_system_perf_tb

# ---- design source list (mirrors the Makefile glm_fp8_system_sim target) ----
SRC="src/glm_fp8_system.v src/weight_decomp.v src/glm_model_fp8.v \
src/ddr5_xbar.v src/weight_loader.v src/expert_cache_pf.v src/expert_cache_ctrl.v \
src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v \
src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v \
src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v \
src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v"

RESULTS=()   # one "sweep|flash|nch|slots|status|perfline" row per run

# run_cfg <sweep-tag> <FLASH_LAT> <DDR_NCH> <CACHE_SLOTS>
run_cfg() {
    local tag="$1" fl="$2" nch="$3" slots="$4"
    local key="fl${fl}_nch${nch}_cs${slots}"
    local bin="$BUILD/perf_${key}"
    local blog="$BUILD/build_${key}.log"
    local rlog="$BUILD/run_${key}.log"

    printf '>> [%-6s] FLASH_LAT=%-4s DDR_NCH=%-2s CACHE_SLOTS=%-2s ... ' \
        "$tag" "$fl" "$nch" "$slots"

    if ! $IVERILOG -g2012 -I src \
            -P ${TBMOD}.FLASH_LAT_CFG=${fl} \
            -P ${TBMOD}.DDR_NCH_CFG=${nch} \
            -P ${TBMOD}.CACHE_SLOTS_CFG=${slots} \
            -o "$bin" $TB $SRC >"$blog" 2>&1; then
        echo "BUILD-FAIL (see $blog)"
        RESULTS+=("$tag|$fl|$nch|$slots|BUILD-FAIL|")
        return
    fi

    $VVP "$bin" >"$rlog" 2>&1
    local perf pass
    perf="$(grep -E '^PERF flash_lat=' "$rlog" | head -1)"
    pass="$(grep -Ec 'ALL [0-9]+ TESTS PASSED' "$rlog")"

    if [ "$pass" -ge 1 ] && [ -n "$perf" ]; then
        echo "PASS"
        RESULTS+=("$tag|$fl|$nch|$slots|PASS|$perf")
    else
        echo "CHECK-FAIL (functional check did not pass; number invalid)"
        RESULTS+=("$tag|$fl|$nch|$slots|CHECK-FAIL|")
    fi
}

echo "=================================================================="
echo " glm_fp8_system cycle-accurate throughput sweep"
echo "=================================================================="

# ---- A) FLASH_LAT sweep (latency sensitivity / stall growth) ----
for FL in 8 64 256 1024; do run_cfg FLASH "$FL" 4 4; done
# ---- B) DDR_NCH sweep (channel scaling; powers of two) ----
for NCH in 1 2 4; do run_cfg DDRNCH 256 "$NCH" 4; done
# ---- C) CACHE_SLOTS sweep (cache-size effect) ----
for CS in 2 4; do run_cfg SLOTS 256 4 "$CS"; done

# ---- helper: pull "field=" value out of a PERF line ----
field() { echo "$1" | grep -oE "$2=[0-9]+" | head -1 | cut -d= -f2; }

echo
echo "=================================================================="
echo " COLLECTED TABLE  (cyc_per_tok = cold-token start->tok_valid;"
echo "                   stall/hit/miss = cumulative ec_* over 3 tokens)"
echo "=================================================================="
printf '%-7s %-9s %-7s %-11s %-9s %-8s %-7s %-8s %-5s %-5s %-5s\n' \
    SWEEP FLASH_LAT DDR_NCH CACHE_SLOTS STATUS CYC/TOK STALL EFF_CYC MEM% HIT MISS
printf -- '------- --------- ------- ----------- --------- -------- ------- -------- ----- ----- -----\n'
for row in "${RESULTS[@]}"; do
    IFS='|' read -r tag fl nch slots status perf <<<"$row"
    if [ "$status" = "PASS" ]; then
        cyc="$(field "$perf" cyc_per_tok)"
        stall="$(field "$perf" stall)"
        hit="$(field "$perf" hit)"
        miss="$(field "$perf" miss)"
        # INTEGRATED (die must wait for missed weights): effective = compute + exposed demand-stall.
        eff=$(( cyc + stall ))
        p10=$(( stall * 1000 / (cyc + stall) ))   # tenths of a percent (truncated)
        mempct="$(( p10 / 10 )).$(( p10 % 10 ))"
    else
        cyc="-"; stall="-"; hit="-"; miss="-"; eff="-"; mempct="-"
    fi
    printf '%-7s %-9s %-7s %-11s %-9s %-8s %-7s %-8s %-5s %-5s %-5s\n' \
        "$tag" "$fl" "$nch" "$slots" "$status" "$cyc" "$stall" "$eff" "$mempct" "$hit" "$miss"
done

echo
echo "Raw PERF lines:"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r tag fl nch slots status perf <<<"$row"
    [ -n "$perf" ] && echo "  $perf"
done
