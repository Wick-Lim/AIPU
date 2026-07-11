`timescale 1ns/1ps
// moe_router_q4k_tb.v -- functional TB for the Q4_K MoE router.
// Q4_K weight-response model + the renorm INVARIANT: the TOPK routed weights are
// w_j=(gate_j/s)*SCALE, so Sum(w_j)=SCALE (2.5) regardless of the sigmoid poly
// approx -- a robust end-to-end check that GEMV->sigmoid->topk->renorm wire up.
// The bit-exact gate is glm_matmul_q4k (proven 480/480).
// Real-dims sweep overrides (docs/SCALE_FUNCTIONAL.md item 2, `make scale-ops`):
//   -DTB_N_EXPERT=256 -DTB_TOPK=8 [-DTB_HIDDEN=..] runs the SAME renorm-invariant
//   contract at the real GLM-5.2 expert count / top-K.  Defaults reproduce the
//   committed slice run (HIDDEN=8, N_EXPERT=8, TOPK=2) byte-identically.
`ifndef TB_HIDDEN
    `define TB_HIDDEN 8
`endif
`ifndef TB_N_EXPERT
    `define TB_N_EXPERT 8
`endif
`ifndef TB_TOPK
    `define TB_TOPK 2
`endif
`ifndef TB_NTEST
    `define TB_NTEST 40
`endif
`ifndef TB_TIMEOUT_NS
    `define TB_TIMEOUT_NS 5000000
`endif
module moe_router_q4k_tb;
    localparam integer HIDDEN   = `TB_HIDDEN;
    localparam integer N_EXPERT = `TB_N_EXPERT;
    localparam integer TOPK     = `TB_TOPK;
    localparam integer PE_M     = 1;
    localparam integer KMAX     = 256;   // NSB=1
    localparam integer IDXW     = $clog2(N_EXPERT);
    localparam [31:0]  SCALE    = 32'h40200000;  // 2.5

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*HIDDEN*PE_M-1:0] x_vec = 0;
    wire busy, done;
    wire [4*N_EXPERT-1:0]  w_q;      // driven by model (reg)
    reg  [4*N_EXPERT-1:0]  w_q_r;
    reg  [16*N_EXPERT-1:0] w_d, w_dmin;
    reg  [96*N_EXPERT-1:0] w_scales;
    wire                   w_req;
    wire [$clog2(KMAX+1)-1:0] w_k;
    wire [TOPK*IDXW*PE_M-1:0] sel_idx;
    wire [TOPK*16*PE_M-1:0]   sel_weight;
    assign w_q = w_q_r;

    moe_router_q4k #(.HIDDEN(HIDDEN), .N_EXPERT(N_EXPERT), .TOPK(TOPK), .SCALE(SCALE),
                     .KMAX(KMAX), .PE_M(PE_M)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done), .x_vec(x_vec),
        .w_req(w_req), .w_k(w_k), .w_q(w_q), .w_d(w_d), .w_dmin(w_dmin), .w_scales(w_scales),
        .sel_idx(sel_idx), .sel_weight(sel_weight)
    );

    // stored Q4_K W_g (N_EXPERT columns, each K=HIDDEN codes + super-block)
    reg [3:0]  Wq [0:N_EXPERT-1][0:HIDDEN-1];
    reg [15:0] Wd [0:N_EXPERT-1];
    reg [15:0] Wm [0:N_EXPERT-1];
    reg [95:0] Ws [0:N_EXPERT-1];

    // combinational weight-response for column w_k
    integer e2;
    always @* begin
        w_q_r = 0; w_d = 0; w_dmin = 0; w_scales = 0;
        for (e2 = 0; e2 < N_EXPERT; e2 = e2 + 1) begin
            w_q_r   [4*e2  +: 4]  = Wq[e2][w_k];
            w_d     [16*e2 +: 16] = Wd[e2];
            w_dmin  [16*e2 +: 16] = Wm[e2];
            w_scales[96*e2 +: 96] = Ws[e2];
        end
    end

    function real f32r(input [31:0] f);
        real m; integer e, i; reg s;
        begin
            s = f[31]; e = f[30:23]; m = (e==0)?0.0:1.0;
            for (i=0;i<23;i=i+1) if (f[i]) m = m + 2.0**(i-23);
            m = (e==0) ? m*(2.0**-126) : m*(2.0**(e-127));
            f32r = s ? -m : m;
        end
    endfunction

    integer t, e, k, errors, ntest, ta, tb;
    integer idxs [0:TOPK-1];
    real wsum, wk_r;
    initial begin
        errors = 0; ntest = `TB_NTEST;
        @(negedge clk); rst = 0;
        for (t = 0; t < ntest; t = t + 1) begin
            // random token + Q4_K W_g
            for (k = 0; k < HIDDEN; k = k + 1) x_vec[16*k +: 16] = {$random} & 16'h7FFF; // small bf16-ish
            for (e = 0; e < N_EXPERT; e = e + 1) begin
                Wd[e] = 16'h211F;                    // ~0.01 fp16
                Wm[e] = {$random} & 16'h1000;         // small dmin
                Ws[e] = {$random, $random, $random} & 96'hFFFFFFFFFFFFFFFFFFFFFFFF;
                for (k = 0; k < HIDDEN; k = k + 1) Wq[e][k] = ($random) & 4'hF;
            end
            @(negedge clk); start = 1; @(negedge clk); start = 0;
            k = 0; while (!done && k < 20000) begin @(posedge clk); k = k + 1; end
            if (!done) begin $display("FAIL test %0d: no done", t); errors = errors + 1; end
            #1;
            // indices valid + pairwise distinct (all TOPK of them)
            for (ta = 0; ta < TOPK; ta = ta + 1) begin
                idxs[ta] = sel_idx[ta*IDXW +: IDXW];
                if (idxs[ta] >= N_EXPERT) begin
                    $display("FAIL test %0d: bad idx[%0d]=%0d", t, ta, idxs[ta]); errors = errors + 1; end
                for (tb = 0; tb < ta; tb = tb + 1)
                    if (idxs[ta] == idxs[tb]) begin
                        $display("FAIL test %0d: dup idx[%0d]==idx[%0d]==%0d", t, ta, tb, idxs[ta]); errors = errors + 1; end
            end
            // renorm invariant: Sum over the TOPK routed weights ~ SCALE (2.5)
            wsum = 0.0;
            for (ta = 0; ta < TOPK; ta = ta + 1) begin
                wk_r = f32r({sel_weight[ta*16 +: 16], 16'd0});
                wsum = wsum + wk_r;
            end
            if (^sel_weight === 1'bx) begin $display("FAIL test %0d: X in weight", t); errors = errors + 1; end
            else if (wsum < 2.40 || wsum > 2.60) begin
                $display("FAIL test %0d: renorm Sum(w)=%f != 2.5", t, wsum); errors = errors + 1; end
            @(negedge clk);
        end
        if (errors == 0)
            $display("[moe_router_q4k] ALL %0d TESTS PASSED (renorm Sum=SCALE, valid idx, GEMV->sigmoid->topk->renorm on Q4_K; HIDDEN=%0d N_EXPERT=%0d TOPK=%0d)", ntest, HIDDEN, N_EXPERT, TOPK);
        else $display("[moe_router_q4k] %0d FAILURES", errors);
        $finish;
    end
    initial begin #`TB_TIMEOUT_NS; $display("[moe_router_q4k] TIMEOUT"); $finish; end
endmodule
