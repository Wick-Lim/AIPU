#!/usr/bin/env bash
#=============================================================================
# tools/perf_sweep.sh -- FAITHFUL cycle-accurate throughput sweep for
#                        glm_q4k_system (the Q4_K port of the fp8 track's
#                        TRACK-P harness; audit #15 / docs/CYCLE_EMULATION.md).
#
#   Compiles test/glm_q4k_system_perf_tb.v ONCE per config (iverilog -P
#   overrides the TB knobs), runs it, and greps the machine-readable
#   PERF/PERF_DETAIL/PERF_INTEG lines into an INTEGRATED table.
#
#   FAITHFUL METRIC (EXPERT_STALL_CFG=1, the TB default):
#     The compute die is clock-gated for exactly the cycles expert_cache_pf
#     holds ec_busy (every cycle a DEMAND MISS is being refilled), so the
#     MEASURED start->tok_valid latency ACTUALLY PAYS the weight-fetch stall --
#     cycles/token GROWS with FLASH_LAT when RESIDENT=0 -- while the committed
#     token stays byte-identical to the free-running run (clock-gating a
#     synchronous die is transparent; the TB's binding check against a
#     standalone glm_model_q4k asserts this every token).  stall/token is the
#     demand-stall that landed INSIDE each measured window, so the memory
#     fraction below is a MEASUREMENT, not a post-hoc add-on.
#
#   Sweeps (each run = a 4-token decode sequence: cold token 0 + 3 warmer):
#     H) HIT    : FLASH_LAT in {8,1024}     (NE=4, CS=4, RESIDENT=0) -- cache-
#                 hit-heavy: all experts fit, only a few COLD misses; cycles/
#                 token is ~flat vs FLASH_LAT after warm-up.
#     T) THRASH : FLASH_LAT in {8,256,1024} (NE=8, CS=2, RESIDENT=0) -- the
#                 cache THRASHES (N_EXPERT > CACHE_SLOTS): every token keeps
#                 missing, so the memory fraction grows with FLASH_LAT AS A
#                 MEASUREMENT.
#     R) RESID  : FLASH_LAT in {8,1024}     (NE=8, CS=2, RESIDENT=1) -- the
#                 rung-3 full-residency SKU: the SAME thrashing miss stream is
#                 refilled by ddr5_xbar (TAG_EFILL, ~DDR_ROW_LAT round-trip),
#                 so cycles/token is ~flat vs FLASH_LAT and the Flash channel
#                 never fires for experts.
#     Z) BASE-OFF: EXPERT_STALL=0 at THRASH FL in {8,1024} -- the decoupled
#                 (observer-only) baseline: the die never pays the stall, so
#                 cycles/token is ~FLAT vs FLASH_LAT and matches the compute-
#                 only split of the faithful rows.  (At high FLASH_LAT the
#                 free-running die can outpace the observer-only cache; FIFO
#                 drops are reported on the PERF line, not failed -- see the
#                 TB's check (d).)
#
#   NOTE: uses bash (NOT zsh) so unquoted $SRC word-splits into file args.
#   Run from the repo root:  bash tools/perf_sweep.sh
#   Env: SWEEP=core (default) | full (also sweeps DDR_NCH and CACHE_SLOTS).
#   Exit status: non-zero if ANY run fails to build, self-check, or emit PERF.
#=============================================================================
set -u

IVERILOG="${IVERILOG:-iverilog}"
VVP="${VVP:-vvp}"
SWEEP="${SWEEP:-core}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="build/perf_sweep"
mkdir -p "$BUILD"

TB=test/glm_q4k_system_perf_tb.v
TBMOD=glm_q4k_system_perf_tb

# ---- design source list (mirrors the Makefile GLM_Q4K_SYS_SRCS) ----
SRC="src/glm_q4k_system.v src/glm_model_q4k.v src/ddr5_xbar.v \
src/weight_loader_q4k.v src/expert_cache_pf.v src/expert_cache_ctrl.v \
src/kv_cache_pager.v src/glm_decoder_block_q4k.v src/mla_attn_q4k.v \
src/swiglu_expert_q4k.v src/moe_router_q4k.v src/glm_matmul_q4k.v \
src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v \
src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v \
src/sampler.v src/glm_fp_pipe.v src/weight_decomp.v src/ecc_secded.v"

RESULTS=()   # one "tag|fl|nch|slots|ne|res|stall|status|perf|detail|integ" per run
NFAIL=0

# run_cfg <tag> <FLASH_LAT> <DDR_NCH> <CACHE_SLOTS> <N_EXPERT> <RESIDENT> <EXPERT_STALL>
run_cfg() {
    local tag="$1" fl="$2" nch="$3" slots="$4" ne="$5" res="$6" est="$7"
    local key="fl${fl}_nch${nch}_cs${slots}_ne${ne}_r${res}_es${est}"
    local bin="$BUILD/perf_${key}"
    local blog="$BUILD/build_${key}.log"
    local rlog="$BUILD/run_${key}.log"

    printf '>> [%-8s] FL=%-4s NE=%-2s CS=%-2s NCH=%-2s res=%s stall=%s ... ' \
        "$tag" "$fl" "$ne" "$slots" "$nch" "$res" "$est"

    if ! $IVERILOG -g2012 -I src \
            -P ${TBMOD}.FLASH_LAT_CFG=${fl} \
            -P ${TBMOD}.DDR_NCH_CFG=${nch} \
            -P ${TBMOD}.CACHE_SLOTS_CFG=${slots} \
            -P ${TBMOD}.N_EXPERT_CFG=${ne} \
            -P ${TBMOD}.RESIDENT_CFG=${res} \
            -P ${TBMOD}.EXPERT_STALL_CFG=${est} \
            -o "$bin" $TB $SRC >"$blog" 2>&1; then
        echo "BUILD-FAIL (see $blog)"
        RESULTS+=("$tag|$fl|$nch|$slots|$ne|$res|$est|BUILD-FAIL|||")
        NFAIL=$((NFAIL+1))
        return
    fi

    $VVP "$bin" >"$rlog" 2>&1
    local perf detail integ pass
    perf="$(grep -E '^PERF q4k ' "$rlog" | head -1)"
    detail="$(grep -E '^PERF_DETAIL ' "$rlog" | head -1)"
    integ="$(grep -E '^PERF_INTEG ' "$rlog" | head -1)"
    pass="$(grep -Ec 'ALL [0-9]+ TESTS PASSED' "$rlog")"

    if [ "$pass" -ge 1 ] && [ -n "$perf" ]; then
        echo "PASS"
        RESULTS+=("$tag|$fl|$nch|$slots|$ne|$res|$est|PASS|$perf|$detail|$integ")
    else
        echo "CHECK-FAIL (functional binding did not pass; number invalid; see $rlog)"
        RESULTS+=("$tag|$fl|$nch|$slots|$ne|$res|$est|CHECK-FAIL|||")
        NFAIL=$((NFAIL+1))
    fi
}

echo "=================================================================="
echo " glm_q4k_system FAITHFUL cycle-accurate throughput sweep (Q4_K)"
echo "   (die clock-gated on expert-cache demand-miss => latency PAYS the"
echo "    weight-fetch wait; binding vs standalone glm_model_q4k per token)"
echo "=================================================================="

# ---- H) cache-hit-heavy (all experts resident after cold fills) ----
for FL in 8 1024;     do run_cfg HIT    "$FL" 4 4 4 0 1; done
# ---- T) thrashing cache (NE > CS): mem fraction grows with FLASH_LAT ----
for FL in 8 256 1024; do run_cfg THRASH "$FL" 4 2 8 0 1; done
# ---- R) RESIDENT=1 (rung-3): same misses, DDR-tier refill, ~flat vs FL ----
for FL in 8 1024;     do run_cfg RESID  "$FL" 4 2 8 1 1; done
# ---- Z) EXPERT_STALL=0 decoupled baseline: the die never pays ----
for FL in 8 1024;     do run_cfg BASE-OFF "$FL" 4 2 8 0 0; done

if [ "$SWEEP" = "full" ]; then
    # NCH=1 is degenerate (ddr5_xbar requires >=2 channels; see CYCLE_EMULATION.md)
    for NCH in 2; do run_cfg DDRNCH 256 "$NCH" 2 8 0 1; done
    for CS in 4 8; do run_cfg SLOTS  256 4 "$CS" 8 0 1; done
fi

# ---- helper: pull "field=" value out of a line ----
field() { echo "$1" | grep -oE "$2=[0-9]+" | head -1 | cut -d= -f2; }

echo
echo "=================================================================="
echo " INTEGRATED TABLE (4-token decode sequence; cold token 0 + 3 warm)"
echo "   cyc/tok   = mean start->tok_valid cycles  [FAITHFUL when stall=1]"
echo "   stall/tok = mean demand-stall cycles INSIDE the measured windows"
echo "               (the weight-fetch stall split; MEASURED)"
echo "   comp/tok  = cyc/tok - stall/tok  (the compute split)"
echo "   mem%      = 100 * stall_sum / cyc_sum  (fraction of decode time the"
echo "               die was frozen waiting on the weight refill)"
echo "=================================================================="
printf '%-8s %-5s %-3s %-9s %-3s %-3s %-9s %-9s %-9s %-7s %-5s\n' \
    SWEEP STALL RES FLASH_LAT NE CS CYC/TOK STALL/TOK COMP/TOK MEM% MISS
printf -- '-------- ----- --- --------- --- --- --------- --------- --------- ------- -----\n'
for row in "${RESULTS[@]}"; do
    IFS='|' read -r tag fl nch slots ne res est status perf detail integ <<<"$row"
    if [ "$status" = "PASS" ]; then
        cpt="$(field "$perf" 'cycles/token')"
        spt="$(field "$perf" 'stall/token')"
        kpt="$(field "$perf" 'compute/token')"
        csum="$(field "$perf" cyc_sum)"
        ssum="$(field "$perf" stall_sum)"
        miss="$(field "$perf" miss)"
        if [ -n "$csum" ] && [ "$csum" -gt 0 ]; then
            p10=$(( ssum * 1000 / csum ))
            mempct="$(( p10 / 10 )).$(( p10 % 10 ))"
        else
            mempct="-"
        fi
    else
        cpt="-"; spt="-"; kpt="-"; mempct="$status"; miss="-"
    fi
    printf '%-8s %-5s %-3s %-9s %-3s %-3s %-9s %-9s %-9s %-7s %-5s\n' \
        "$tag" "$est" "$res" "$fl" "$ne" "$slots" "$cpt" "$spt" "$kpt" "$mempct" "$miss"
done

echo
echo "Raw PERF / PERF_INTEG lines:"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r tag fl nch slots ne res est status perf detail integ <<<"$row"
    [ -n "$perf" ]  && echo "  $perf"
    [ -n "$integ" ] && echo "    $integ"
done

if [ "$NFAIL" -ne 0 ]; then
    echo
    echo "FAILED: $NFAIL sweep run(s) did not pass"
    exit 1
fi
