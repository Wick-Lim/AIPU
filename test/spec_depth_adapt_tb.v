`timescale 1ns/1ps
//============================================================================
// spec_depth_adapt_tb.v -- ADAPTIVE DRAFT DEPTH (runtime-variable K):
//   spec_decode_seq(ADAPT=1).k_cur + the spec_depth_adapt policy, checked
//   against INDEPENDENT software goldens, across DRAFT_K = 2,3,4,6,8.
//----------------------------------------------------------------------------
// THE OVERRIDING PROPERTY (docs/R3_APPLIANCE_SPEC.md sec 9 + repo principle):
//   Adaptivity may only change HOW MANY tokens are drafted per pass.  The
//   accept/reject rule and the committed stream must stay identical to greedy
//   decode for ANY depth schedule -- any policy is output-invariant by
//   construction.  This TB proves it three ways, per K:
//
//   (a) FORCED depth schedules on spec_decode_seq(ADAPT=1).k_cur: constant
//       full depth, the k=1 edge, mid-stream depth changes (ramp up/down,
//       1<->K alternation), clamp edges (k_cur=0 -> depth 1, k_cur>K -> K),
//       n_draft interplay (nd_eff = min(nd, k), incl. nd=0 and nd>K), and
//       random (nd, k, fm) fuzzing.  The committed stream must equal the same
//       position-accurate greedy reference G that spec_decode_seq_k_tb uses;
//       all four counters exact; every beat X-checked.
//   (b) CLOSED LOOP with spec_depth_adapt driving k_cur: the DUT policy state
//       is asserted EVERY pass (before AND after the update) against a 1:1
//       software model of the same streak/THRESH rule; depth provably RISES
//       1 -> K under an all-accept regime, FALLS back to 1 under an
//       all-reject regime, and HOLDS across empty (n_draft=0) batches; the
//       committed stream STILL equals greedy G (output-invariance under the
//       policy, incl. random mixed regimes).
//   (c) ADAPT=0 (default-off) engines: the same stimulus with k_cur WIGGLING
//       (0, 1, >K, random) must behave as the FIXED-depth verifier -- k_cur
//       ignored, nd_eff = min(nd, K) -- the drop-in guarantee for every
//       existing consumer.
//
//   The new pass_done/pass_acc/pass_dep observation taps are themselves
//   asserted every pass: exactly ONE pulse per pass, values == golden p /
//   nd_eff, no X.
//
// Emits "ALL <N> TESTS PASSED"; $fatal on ANY spec!=greedy / counter /
// policy-model mismatch.
//============================================================================

//----------------------------------------------------------------------------
// One self-checking engine per (DRAFT_K, ADAPT_EN, THRESH) configuration.
// Engines self-serialize on `en` (clean interleave-free logs); each owns its
// own DUT pair, greedy reference G, capture buffers, and software goldens.
//----------------------------------------------------------------------------
module sda_eng #(
    parameter integer DRAFT_K  = 2,
    parameter integer ADAPT_EN = 1,
    parameter integer THRESH   = 2,
    parameter integer TOKW     = 16
)(
    input  wire        clk,
    input  wire        en,
    output reg  [31:0] tests_out,
    output reg         finished
);
    localparam integer DKW   = (DRAFT_K <= 1) ? 1 : $clog2(DRAFT_K + 1);
    localparam integer KCMAX = (1 << DKW) - 1;   // max value a DKW-bit port holds
    localparam integer CAP   = 16384;
    localparam [TOKW-1:0] MISMASK = 16'h5A3C;    // nonzero -> guaranteed mismatch

    // position-accurate greedy reference (nonzero so X is detectable)
    reg [TOKW-1:0] G [0:CAP-1];

    //------------------------------------------------------------------------
    // DUTs: verifier (adaptive-depth port) + depth policy observing its taps
    //------------------------------------------------------------------------
    reg                          rst, st, pv;
    reg  [DRAFT_K*TOKW-1:0]      dv;
    reg  [(DRAFT_K+1)*TOKW-1:0]  tv;
    reg  [DKW-1:0]               nd;
    reg  [DKW-1:0]               kc;       // forced depth (parts a/c)
    reg                          use_pol;  // 1 => policy drives k_cur (part b)
    wire [DKW-1:0]               k_pol;
    wire [DKW-1:0] kc_mux = (ADAPT_EN != 0 && use_pol) ? k_pol : kc;

    wire            cv, pd;
    wire [TOKW-1:0] ct;
    wire [31:0]     tt, mp, ac, rj;
    wire [DKW-1:0]  pa, pw;

    spec_decode_seq #(.TOKW(TOKW), .DRAFT_K(DRAFT_K), .ADAPT(ADAPT_EN)) dut (
        .clk(clk), .rst(rst), .start(st),
        .pass_valid(pv), .verified_tok({TOKW{1'b0}}), .draft_tok({TOKW{1'b0}}),
        .draft_present(1'b0),
        .commit_valid(cv), .commit_tok(ct), .accepted(),
        .total_tokens(tt), .main_passes(mp), .accepts(ac), .rejects(rj),
        .draft_vec(dv), .truth_vec(tv), .n_draft(nd),
        .k_cur(kc_mux), .pass_done(pd), .pass_acc(pa), .pass_dep(pw)
    );

    spec_depth_adapt #(.DRAFT_K(DRAFT_K), .THRESH(THRESH)) pol (
        .clk(clk), .rst(rst),
        .pass_done(pd), .pass_acc(pa), .pass_dep(pw),
        .k_cur(k_pol)
    );

    //------------------------------------------------------------------------
    // captures: commit stream (X-aware) + pass_* observation-tap monitor
    //------------------------------------------------------------------------
    reg [TOKW-1:0] got [0:CAP-1];
    integer got_n = 0;
    reg cap = 1'b0;
    always @(negedge clk) if (cap && cv) begin
        if (^ct === 1'bx) begin
            $display("FAIL[K%0d A%0d]: X on commit beat %0d", DRAFT_K, ADAPT_EN, got_n);
            $fatal(1, "X");
        end
        got[got_n] = ct; got_n = got_n + 1;
    end

    integer pd_seen = 0;
    reg [DKW-1:0] last_pa, last_pw;
    always @(negedge clk) if (cap && pd) begin
        if (^pa === 1'bx || ^pw === 1'bx) begin
            $display("FAIL[K%0d]: X on pass taps (pass %0d)", DRAFT_K, pd_seen);
            $fatal(1, "X");
        end
        last_pa = pa; last_pw = pw; pd_seen = pd_seen + 1;
    end

    //------------------------------------------------------------------------
    // software goldens: greedy cursor/counters + 1:1 model of spec_depth_adapt
    //------------------------------------------------------------------------
    integer c, gpas, gacc, grej;   // greedy cursor + pass/accept/reject counts
    integer g_k, g_streak;         // reference model of the policy state
    integer seed;

    task arm; begin
        rst = 1; st = 0; pv = 0; dv = 0; tv = 0; nd = 0; kc = 0; use_pol = 0; cap = 0;
        @(negedge clk); @(negedge clk); rst = 0;
        got_n = 0; pd_seen = 0; c = 0; gpas = 0; gacc = 0; grej = 0;
        g_k = 1; g_streak = 0;     // == spec_depth_adapt reset state
        cap = 1;
        @(negedge clk); st = 1; @(negedge clk); st = 0;
    end endtask

    // One verify pass.  nd_i -> n_draft port, kc_i -> k_cur port (forced mode;
    // ignored under use_pol), first mismatching draft at index fm (fm>=K =>
    // all match).  Golden depth the DUT must SCAN:
    //   ndc  = min(nd_i masked to DKW bits, K)            (the DUT's nd_w clamp)
    //   keff = K                        ADAPT_EN=0  (k_cur MUST be ignored)
    //        = clamp(kc_i, 1..K)        forced      (the DUT's kc_w clamp)
    //        = g_k (software policy)    closed loop
    //   dep  = min(ndc, keff) = nd_eff ; p = min(fm, dep)
    // Commits = G[c .. c+p]; accepts += p; rejects += dep - p; c += p+1.
    task do_pass(input integer nd_i, input integer kc_i, input integer fm);
        integer j, p, ndc, keff, dep;
        begin
            if (c + DRAFT_K >= CAP) begin
                $display("FAIL[K%0d]: greedy reference overrun c=%0d", DRAFT_K, c);
                $fatal(1, "cap");
            end
            if (use_pol) begin
                // depth source = policy: must equal the software model NOW
                if (k_pol !== g_k) begin
                    $display("FAIL[K%0d]: k_pol=%0d != model %0d before pass %0d",
                             DRAFT_K, k_pol, g_k, gpas);
                    $fatal(1, "policy!=model");
                end
                tests_out = tests_out + 1;
                keff = g_k;
            end else if (ADAPT_EN != 0) begin
                keff = kc_i & KCMAX;                 // as driven on the DKW port
                if (keff > DRAFT_K) keff = DRAFT_K;  // DUT clamps high
                if (keff == 0)      keff = 1;        // DUT clamps low
            end else begin
                keff = DRAFT_K;                      // ADAPT=0: k_cur ignored
            end
            ndc = nd_i & KCMAX;
            if (ndc > DRAFT_K) ndc = DRAFT_K;        // DUT nd_w clamp
            dep = (ndc < keff) ? ndc : keff;         // nd_eff
            p   = (fm < dep) ? fm : dep;

            dv = 0; tv = 0;
            for (j = 0; j <= DRAFT_K; j = j + 1) tv[j*TOKW +: TOKW] = G[c+j];
            for (j = 0; j < DRAFT_K; j = j + 1)
                dv[j*TOKW +: TOKW] = (j < fm) ? G[c+j] : (G[c+j] ^ MISMASK);
            nd = nd_i[DKW-1:0];
            kc = kc_i[DKW-1:0];

            gpas = gpas + 1; gacc = gacc + p; grej = grej + (dep - p); c = c + p + 1;

            @(negedge clk); pv = 1;
            @(negedge clk); pv = 0; dv = 0; tv = 0; nd = 0;
            for (j = 0; j < DRAFT_K + 2; j = j + 1) @(negedge clk);   // drain

            // observation taps: exactly ONE pulse per pass, golden values
            if (pd_seen !== gpas) begin
                $display("FAIL[K%0d]: pass_done pulses %0d exp %0d", DRAFT_K, pd_seen, gpas);
                $fatal(1, "pass_done");
            end
            if (last_pa !== p || last_pw !== dep) begin
                $display("FAIL[K%0d]: taps p=%0d dep=%0d exp p=%0d dep=%0d",
                         DRAFT_K, last_pa, last_pw, p, dep);
                $fatal(1, "taps");
            end
            tests_out = tests_out + 2;

            if (use_pol) begin
                // software model of the policy update (same rule as the RTL),
                // then the DUT policy must have landed on the same state
                if (dep != 0 && p == dep) begin
                    if (g_streak == THRESH - 1) begin
                        g_streak = 0;
                        if (g_k < DRAFT_K) g_k = g_k + 1;
                    end else g_streak = g_streak + 1;
                end else if (p < dep) begin
                    g_streak = 0;
                    if (g_k > 1) g_k = g_k - 1;
                end
                if (k_pol !== g_k) begin
                    $display("FAIL[K%0d]: k_pol=%0d != model %0d after pass %0d",
                             DRAFT_K, k_pol, g_k, gpas);
                    $fatal(1, "policy!=model");
                end
                tests_out = tests_out + 1;
            end
        end
    endtask

    // committed stream == greedy G positionally + all 4 counters exact
    task check(input [255:0] name);
        integer k;
        begin
            if (got_n !== c) begin
                $display("FAIL[K%0d A%0d %0s]: len got %0d exp %0d",
                         DRAFT_K, ADAPT_EN, name, got_n, c);
                $fatal(1, "len");
            end
            for (k = 0; k < c; k = k + 1) begin
                if (got[k] !== G[k]) begin
                    $display("FAIL[K%0d A%0d %0s]: beat %0d got %0d exp greedy %0d",
                             DRAFT_K, ADAPT_EN, name, k, got[k], G[k]);
                    $fatal(1, "spec!=greedy");
                end
                tests_out = tests_out + 1;
            end
            if (tt !== c)    begin $display("FAIL[K%0d %0s] total %0d exp %0d", DRAFT_K, name, tt, c);   $fatal(1,"t"); end
            if (mp !== gpas) begin $display("FAIL[K%0d %0s] pass %0d exp %0d",  DRAFT_K, name, mp, gpas); $fatal(1,"p"); end
            if (ac !== gacc) begin $display("FAIL[K%0d %0s] acc %0d exp %0d",   DRAFT_K, name, ac, gacc); $fatal(1,"a"); end
            if (rj !== grej) begin $display("FAIL[K%0d %0s] rej %0d exp %0d",   DRAFT_K, name, rj, grej); $fatal(1,"r"); end
            tests_out = tests_out + 4;
            $display("PASS[K%0d A%0d %0s] passes=%0d total=%0d acc=%0d rej=%0d k_pol=%0d",
                     DRAFT_K, ADAPT_EN, name, mp, tt, ac, rj, k_pol);
        end
    endtask

    //------------------------------------------------------------------------
    // scenario program
    //------------------------------------------------------------------------
    integer i, r1, r2, r3;
    initial begin
        tests_out = 0; finished = 0;
        rst = 1; st = 0; pv = 0; dv = 0; tv = 0; nd = 0; kc = 0; use_pol = 0; cap = 0;
        seed = 32'h5da0 + DRAFT_K * 17 + ADAPT_EN * 3 + THRESH;
        for (i = 0; i < CAP; i = i + 1) begin
            G[i] = $random(seed);
            if (G[i] == {TOKW{1'b0}}) G[i] = 16'h0d1e;   // keep nonzero (X-detect)
        end
        wait (en === 1'b1);
        @(negedge clk);

        //---------------- (a)/(c): FORCED depth schedules ----------------
        arm();
        // full-depth baseline (== fixed-K behavior)
        for (i = 0; i < 3; i = i + 1) do_pass(DRAFT_K, DRAFT_K, DRAFT_K);
        for (i = 0; i < 3; i = i + 1) do_pass(DRAFT_K, DRAFT_K, 0);
        for (i = 0; i <= DRAFT_K; i = i + 1) do_pass(DRAFT_K, DRAFT_K, i);
        check("full-depth");

        // k=1 edge: runtime depth 1 inside a K>1 elaboration
        for (i = 0; i <= DRAFT_K; i = i + 1) do_pass(DRAFT_K, 1, i);
        check("k1-edge");

        // clamp edges: k_cur=0 -> depth 1; k_cur=KCMAX (>K when representable) -> K
        do_pass(DRAFT_K, 0, DRAFT_K); do_pass(DRAFT_K, 0, 0);
        do_pass(DRAFT_K, KCMAX, DRAFT_K); do_pass(DRAFT_K, KCMAX, 0);
        check("clamp");

        // mid-stream depth changes: ramp 1..K then K..1 (all-accept), then
        // 1<->K alternation with random miss positions
        for (i = 1; i <= DRAFT_K; i = i + 1) do_pass(DRAFT_K, i, DRAFT_K);
        for (i = DRAFT_K; i >= 1; i = i - 1) do_pass(DRAFT_K, i, DRAFT_K);
        for (i = 0; i < 10; i = i + 1) begin
            r1 = $random(seed); if (r1 < 0) r1 = -r1;
            do_pass(DRAFT_K, (i % 2 == 0) ? 1 : DRAFT_K, r1 % (DRAFT_K + 1));
        end
        check("mid-stream");

        // n_draft interplay: nd walks 0..K at k=K; k walks 1..K at nd=K;
        // nd>K on the DKW-bit port (the nd_w clamp, when representable)
        for (i = 0; i <= DRAFT_K; i = i + 1) do_pass(i, DRAFT_K, DRAFT_K);
        for (i = 1; i <= DRAFT_K; i = i + 1) do_pass(DRAFT_K, i, DRAFT_K);
        do_pass(KCMAX, DRAFT_K, DRAFT_K);
        check("nd-interplay");

        // random fuzz: (nd, k, fm) all random incl. clamp values 0 / >K
        for (i = 0; i < 300; i = i + 1) begin
            r1 = $random(seed); if (r1 < 0) r1 = -r1;
            r2 = $random(seed); if (r2 < 0) r2 = -r2;
            r3 = $random(seed); if (r3 < 0) r3 = -r3;
            do_pass(r1 % (KCMAX + 1), r2 % (KCMAX + 1), r3 % (DRAFT_K + 1));
        end
        check("random");

        //---------------- (b): CLOSED LOOP with the policy ----------------
        if (ADAPT_EN != 0) begin
            arm(); use_pol = 1;
            // reset state: depth starts at 1
            if (k_pol !== 1) begin
                $display("FAIL[K%0d]: k_pol after reset = %0d exp 1", DRAFT_K, k_pol);
                $fatal(1, "reset");
            end
            tests_out = tests_out + 1;

            // RISE: all-accept regime must ramp depth 1 -> K (THRESH per step)
            for (i = 0; i < 2 * THRESH * DRAFT_K; i = i + 1)
                do_pass(DRAFT_K, 0, DRAFT_K);
            if (k_pol !== DRAFT_K) begin
                $display("FAIL[K%0d]: high-accept regime k_pol=%0d exp %0d",
                         DRAFT_K, k_pol, DRAFT_K);
                $fatal(1, "rise");
            end
            tests_out = tests_out + 1;
            check("pol-rise");

            // FALL: all-reject regime must back off to depth 1
            for (i = 0; i < DRAFT_K + 2; i = i + 1)
                do_pass(DRAFT_K, 0, 0);
            if (k_pol !== 1) begin
                $display("FAIL[K%0d]: low-accept regime k_pol=%0d exp 1", DRAFT_K, k_pol);
                $fatal(1, "fall");
            end
            tests_out = tests_out + 1;
            check("pol-fall");

            // HOLD: empty batches (n_draft=0 -> dep=0) are no evidence
            for (i = 0; i < THRESH; i = i + 1) do_pass(DRAFT_K, 0, DRAFT_K);
            r1 = g_k;   // model depth before the empty batches
            for (i = 0; i < 3; i = i + 1) do_pass(0, 0, DRAFT_K);
            if (k_pol !== r1) begin
                $display("FAIL[K%0d]: empty-batch hold k_pol=%0d exp %0d",
                         DRAFT_K, k_pol, r1);
                $fatal(1, "hold");
            end
            tests_out = tests_out + 1;
            check("pol-hold");

            // MIXED: high-accept regime then low-accept regime, random misses;
            // model equality is asserted inside EVERY do_pass, and the stream
            // must STILL equal greedy (output-invariance under the policy).
            for (i = 0; i < 150; i = i + 1) begin
                r1 = $random(seed); if (r1 < 0) r1 = -r1;
                do_pass(DRAFT_K, 0, (r1 % 10 < 9) ? DRAFT_K : (r1 % DRAFT_K));
            end
            for (i = 0; i < 150; i = i + 1) begin
                r1 = $random(seed); if (r1 < 0) r1 = -r1;
                do_pass(DRAFT_K, 0, (r1 % 10 < 2) ? DRAFT_K : (r1 % 2));
            end
            check("pol-mixed");
        end

        finished = 1;
    end
endmodule

//----------------------------------------------------------------------------
// top: serialize the engines (ADAPT=1 across all supported K, then the
// default-off ADAPT=0 engines), aggregate, print the gate line.
//----------------------------------------------------------------------------
module spec_depth_adapt_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg go = 1'b0;
    wire [31:0] t2, t3, t4, t6, t8, t2f, t4f, t8f;
    wire        f2, f3, f4, f6, f8, f2f, f4f, f8f;

    // (a)+(b): ADAPT=1 engines across all supported K (batch path is K>=2)
    sda_eng #(.DRAFT_K(2), .ADAPT_EN(1), .THRESH(2)) e2  (.clk(clk), .en(go),  .tests_out(t2),  .finished(f2));
    sda_eng #(.DRAFT_K(3), .ADAPT_EN(1), .THRESH(2)) e3  (.clk(clk), .en(f2),  .tests_out(t3),  .finished(f3));
    sda_eng #(.DRAFT_K(4), .ADAPT_EN(1), .THRESH(2)) e4  (.clk(clk), .en(f3),  .tests_out(t4),  .finished(f4));
    sda_eng #(.DRAFT_K(6), .ADAPT_EN(1), .THRESH(3)) e6  (.clk(clk), .en(f4),  .tests_out(t6),  .finished(f6));
    sda_eng #(.DRAFT_K(8), .ADAPT_EN(1), .THRESH(2)) e8  (.clk(clk), .en(f6),  .tests_out(t8),  .finished(f8));
    // (c): ADAPT=0 default -- k_cur wiggles but MUST be ignored (fixed depth)
    sda_eng #(.DRAFT_K(2), .ADAPT_EN(0), .THRESH(2)) e2f (.clk(clk), .en(f8),  .tests_out(t2f), .finished(f2f));
    sda_eng #(.DRAFT_K(4), .ADAPT_EN(0), .THRESH(2)) e4f (.clk(clk), .en(f2f), .tests_out(t4f), .finished(f4f));
    sda_eng #(.DRAFT_K(8), .ADAPT_EN(0), .THRESH(2)) e8f (.clk(clk), .en(f4f), .tests_out(t8f), .finished(f8f));

    initial begin
        @(negedge clk); go = 1'b1;
        wait (f8f === 1'b1);
        @(negedge clk);
        $display("ALL %0d TESTS PASSED",
                 t2 + t3 + t4 + t6 + t8 + t2f + t4f + t8f);
        $finish;
    end

    initial begin
        #100_000_000;   // global watchdog (whole run measures well under 1 ms)
        $display("FAIL: global timeout");
        $fatal(1, "timeout");
    end
endmodule
