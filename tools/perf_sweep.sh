#!/usr/bin/env bash
#=============================================================================
# tools/perf_sweep.sh -- FAITHFUL cycle-accurate throughput sweep for
#                        glm_fp8_system (TRACK P: make the die PAY the Flash stall).
#
#   Compiles test/glm_fp8_system_perf_tb.v ONCE per config (iverilog -P overrides
#   the TB knobs), runs it, and greps the machine-readable PERF/PERF_DETAIL/
#   PERF_INTEG lines into an INTEGRATED table.
#
#   FAITHFUL METRIC (EXPERT_STALL_CFG=1, the TB default):
#     The compute die is clock-gated for exactly the cycles expert_cache_pf holds
#     ec_busy (every cycle a DEMAND MISS is serviced by Flash), so the MEASURED
#     start->tok_valid latency ACTUALLY PAYS the memory stall -- cyc_per_tok GROWS
#     with FLASH_LAT -- while the committed token stays byte-identical to the
#     free-running run (clock-gating a synchronous die is transparent).  The
#     per-token demand-stall that lands INSIDE each measured window is reported on
#     the PERF_INTEG line, so mem% below is a MEASUREMENT, not a post-hoc add-on.
#
#   Sweeps:
#     A) SLICE  : FLASH_LAT in {8,64,256,1024}   (L=4,  N_EXPERT=4, CACHE_SLOTS=4,
#                 DDR_NCH=4) -- the committed small slice; a few COLD misses.
#     D) SCALE  : FLASH_LAT in {8,64,256,1024}   (L=6,  N_EXPERT=8, CACHE_SLOTS=2,
#                 DDR_NCH=4) -- larger model: N_EXPERT>CACHE_SLOTS so the cache
#                 THRASHES and every token keeps missing -> the memory fraction
#                 grows toward the Flash-bound regime AS A MEASUREMENT.
#     Z) BASELINE (EXPERT_STALL=0, decoupled/observer-only) at SLICE FL in {8,1024}
#                 -- shows the OLD cyc_per_tok is ~FLAT vs FLASH_LAT (the die never
#                 paid); contrast with the faithful A rows at the same FL.
#
#   NOTE: uses bash (NOT zsh) so unquoted $SRC word-splits into file args.
#   Run from the repo root:  bash tools/perf_sweep.sh
#   Env: SWEEP=core (default: A+D+Z) | full (also B/C channel+cache sweeps).
#=============================================================================
set -u

IVERILOG="${IVERILOG:-iverilog}"
VVP="${VVP:-vvp}"
SWEEP="${SWEEP:-core}"
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

RESULTS=()   # one "tag|fl|nch|slots|L|NE|stall|status|perf|detail|integ" row per run

# run_cfg <tag> <FLASH_LAT> <DDR_NCH> <CACHE_SLOTS> <L> <N_EXPERT> <EXPERT_STALL>
run_cfg() {
    local tag="$1" fl="$2" nch="$3" slots="$4" lyr="$5" ne="$6" est="$7"
    local key="fl${fl}_nch${nch}_cs${slots}_L${lyr}_NE${ne}_es${est}"
    local bin="$BUILD/perf_${key}"
    local blog="$BUILD/build_${key}.log"
    local rlog="$BUILD/run_${key}.log"

    printf '>> [%-8s] FL=%-4s L=%-2s NE=%-2s CS=%-2s NCH=%-2s stall=%s ... ' \
        "$tag" "$fl" "$lyr" "$ne" "$slots" "$nch" "$est"

    if ! $IVERILOG -g2012 -I src \
            -P ${TBMOD}.FLASH_LAT_CFG=${fl} \
            -P ${TBMOD}.DDR_NCH_CFG=${nch} \
            -P ${TBMOD}.CACHE_SLOTS_CFG=${slots} \
            -P ${TBMOD}.L_CFG=${lyr} \
            -P ${TBMOD}.N_EXPERT_CFG=${ne} \
            -P ${TBMOD}.EXPERT_STALL_CFG=${est} \
            -o "$bin" $TB $SRC >"$blog" 2>&1; then
        echo "BUILD-FAIL (see $blog)"
        RESULTS+=("$tag|$fl|$nch|$slots|$lyr|$ne|$est|BUILD-FAIL|||")
        return
    fi

    $VVP "$bin" >"$rlog" 2>&1
    local perf detail integ pass
    perf="$(grep -E '^PERF flash_lat=' "$rlog" | head -1)"
    detail="$(grep -E '^PERF_DETAIL ' "$rlog" | head -1)"
    integ="$(grep -E '^PERF_INTEG ' "$rlog" | head -1)"
    pass="$(grep -Ec 'ALL [0-9]+ TESTS PASSED' "$rlog")"

    if [ "$pass" -ge 1 ] && [ -n "$perf" ]; then
        echo "PASS"
        RESULTS+=("$tag|$fl|$nch|$slots|$lyr|$ne|$est|PASS|$perf|$detail|$integ")
    else
        echo "CHECK-FAIL (functional binding did not pass; number invalid; see $rlog)"
        RESULTS+=("$tag|$fl|$nch|$slots|$lyr|$ne|$est|CHECK-FAIL|||")
    fi
}

echo "=================================================================="
echo " glm_fp8_system FAITHFUL cycle-accurate throughput sweep"
echo "   (die clock-gated on expert-cache demand-miss => latency PAYS Flash)"
echo "=================================================================="

# ---- A) SLICE FLASH_LAT sweep (faithful) ----
for FL in 8 64 256 1024; do run_cfg SLICE "$FL" 4 4 4 4 1; done
# ---- D) SCALE FLASH_LAT sweep (faithful; thrashing cache -> higher miss count) ----
for FL in 8 64 256 1024; do run_cfg SCALE "$FL" 4 2 6 8 1; done
# ---- Z) BASELINE (EXPERT_STALL=0): decoupled cyc_per_tok is ~flat vs FLASH_LAT ----
for FL in 8 1024; do run_cfg BASE-OFF "$FL" 4 4 4 4 0; done

if [ "$SWEEP" = "full" ]; then
    for NCH in 1 2 4; do run_cfg DDRNCH 256 "$NCH" 4 4 4 1; done
    for CS in 2 4;    do run_cfg SLOTS  256 4 "$CS" 4 4 1; done
fi

# ---- helper: pull "field=" value out of a line ----
field() { echo "$1" | grep -oE "$2=[0-9]+" | head -1 | cut -d= -f2; }

echo
echo "=================================================================="
echo " INTEGRATED TABLE"
echo "   cyc_cold  = token-0 (cold-cache) start->tok_valid  [FAITHFUL: pays stall]"
echo "   cyc_sum   = sum of the 3 measured tokens' latencies"
echo "   stall_sum = demand-stall cycles that landed INSIDE those windows"
echo "               (== cumulative ec_demand_stall when no leakage; MEASURED)"
echo "   mem%      = 100 * stall_sum / cyc_sum  (fraction of decode time die was"
echo "               frozen waiting on Flash -- the measured memory fraction)"
echo "=================================================================="
printf '%-8s %-4s %-9s %-4s %-3s %-4s %-9s %-9s %-9s %-7s %-5s\n' \
    SWEEP STALL FLASH_LAT L NE CS CYC_COLD CYC_SUM STALL_SUM MEM% MISS
printf -- '-------- ---- --------- ---- --- ---- --------- --------- --------- ------- -----\n'
for row in "${RESULTS[@]}"; do
    IFS='|' read -r tag fl nch slots lyr ne est status perf detail integ <<<"$row"
    if [ "$status" = "PASS" ]; then
        ccold="$(field "$detail" cyc_cold)"
        cw2="$(field "$detail" cyc_warm2)"
        cw3="$(field "$detail" cyc_warm3)"
        miss="$(field "$perf" miss)"
        iws="$(field "$integ" in_window_stall)"
        [ -z "$iws" ] && iws=0
        csum=$(( ccold + cw2 + cw3 ))
        if [ "$csum" -gt 0 ]; then
            p10=$(( iws * 1000 / csum ))
            mempct="$(( p10 / 10 )).$(( p10 % 10 ))"
        else
            mempct="-"
        fi
    else
        ccold="-"; csum="-"; iws="-"; mempct="$status"; miss="-"
    fi
    printf '%-8s %-4s %-9s %-4s %-3s %-4s %-9s %-9s %-9s %-7s %-5s\n' \
        "$tag" "$est" "$fl" "$lyr" "$ne" "$slots" "$ccold" "$csum" "$iws" "$mempct" "$miss"
done

echo
echo "Raw PERF / PERF_INTEG lines:"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r tag fl nch slots lyr ne est status perf detail integ <<<"$row"
    [ -n "$perf" ]  && echo "  $perf"
    [ -n "$integ" ] && echo "    $integ"
done
