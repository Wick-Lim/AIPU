#!/usr/bin/env bash
# check_test_counts.sh -- enforce the EXACT per-gate test count against a manifest.
#
# WHY: every gate recipe greps `ALL [0-9]+ TESTS PASSED` -- a wildcard that passes even
#   if the testbench silently ran FEWER tests than intended (a loop bound reduced, an
#   `ifdef` block dropped, a build flag like -DSPEC_FULL missing).  The tb still prints
#   "ALL <fewer> PASSED" and the wildcard is satisfied.  This checker closes that: it
#   compares each gate's ACTUAL printed count(s) against test/expected_test_counts.txt and
#   fails on (a) a count mismatch (silent under/over-testing) or (b) a manifest gate whose
#   set of runs changed (a run dropped entirely, or an extra run appeared).
#
# MULTISET: some banner names print more than once per release-gate (the same unit run
#   under two configs).  The manifest pins the FULL multiset of counts per name; this
#   compares them as histograms (order-independent), so dropping/adding any single run --
#   or changing any one run's count -- is caught.
#
# USAGE: tools/check_test_counts.sh <release-gate-log>   (log = full stdout of `make release-gate`)
#   Manifest: test/expected_test_counts.txt, one "NAME c1 [c2 ...]" per line (NAME = the
#   exact string inside the `[...]` banner prefix, no spaces).  '#'/blank lines ignored.
#
# Implemented in awk (POSIX) so it runs on the stock macOS bash 3.2 (no `declare -A`).
set -u

LOG="${1:?usage: check_test_counts.sh <release-gate-log>}"
MANIFEST="$(dirname "$0")/../test/expected_test_counts.txt"
[ -f "$LOG" ]      || { echo "check_test_counts: log not found: $LOG"; exit 2; }
[ -f "$MANIFEST" ] || { echo "check_test_counts: manifest not found: $MANIFEST"; exit 2; }

awk '
    # ---- pass 1: manifest (first file) ----
    FNR==NR {
        line=$0; sub(/#.*/,"",line)
        nf=split(line,f," ")
        if (nf==0) next
        name=f[1]
        if (nf<2) { print "MANIFEST-BAD: [" name "] has no expected count"; bad=1; next }
        mseen[name]=1
        for (i=2;i<=nf;i++) { want[name SUBSEP f[i]]++; expn[name]++; vals[name SUBSEP f[i]]=f[i] }
        next
    }
    # ---- pass 2: release-gate log (second file) ----
    /\] ALL [0-9]+ TESTS PASSED/ {
        s=$0
        b=index(s,"["); if(!b) next
        rest=substr(s,b+1)
        e=index(rest,"]"); if(!e) next
        name=substr(rest,1,e-1)
        a=index(s,"] ALL "); if(!a) next
        tail=substr(s,a+6)                     # after "] ALL "
        n=tail+0
        if (tail !~ /^[0-9]+ TESTS PASSED/) next
        act[name SUBSEP n]++; actn[name]++; vals[name SUBSEP n]=n
        seen[name]=1
    }
    END {
        fail=bad
        checked=0
        for (nm in mseen) {
            checked++
            if (!(nm in seen)) {
                el=""; for (k in want) { split(k,p,SUBSEP); if(p[1]==nm) el=el " " p[2] }
                print "MISSING : [" nm "] never printed ALL N TESTS PASSED (gate did not run?)  expected:" el
                fail=1; continue
            }
            # histogram compare over the union of values seen for this name
            for (k in vals) {
                split(k,p,SUBSEP); if (p[1]!=nm) continue
                v=p[2]; ec=want[nm SUBSEP v]+0; ac=act[nm SUBSEP v]+0
                if (ec!=ac) {
                    print "MISMATCH: [" nm "] count " v " ran " ac " time(s), manifest pins " ec " (silent under/over-test, dropped/added run, or update the manifest)"
                    fail=1
                }
            }
        }
        if (fail) { print "FAILED: test-count manifest check (" checked " gates checked)"; exit 1 }
        print "ALL " checked " GATE COUNTS MATCH  (exact per-gate test-count multisets pinned vs test/expected_test_counts.txt)"
    }
' "$MANIFEST" "$LOG"
