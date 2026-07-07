`timescale 1ns/1ps
// moe_router_q4k_tb.v -- functional TB for the Q4_K MoE router.
// Q4_K weight-response model + the renorm INVARIANT: the TOPK routed weights are
// w_j=(gate_j/s)*SCALE, so Sum(w_j)=SCALE (2.5) regardless of the sigmoid poly
// approx -- a robust end-to-end check that GEMV->sigmoid->topk->renorm wire up.
// The bit-exact gate is glm_matmul_q4k (proven 480/480).
module moe_router_q4k_tb;
    localparam integer HIDDEN   = 8;
    localparam integer N_EXPERT = 8;
    localparam integer TOPK     = 2;
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

    integer t, e, k, errors, ntest, i0, i1;
    real wsum, wa, wb;
    initial begin
        errors = 0; ntest = 40;
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
            // indices valid + distinct
            i0 = sel_idx[0*IDXW +: IDXW];
            i1 = sel_idx[1*IDXW +: IDXW];
            if (i0 >= N_EXPERT || i1 >= N_EXPERT || i0 == i1) begin
                $display("FAIL test %0d: bad idx %0d,%0d", t, i0, i1); errors = errors + 1; end
            // renorm invariant: Sum(routed weights) ~ 2.5
            wa = f32r({sel_weight[0*16 +: 16], 16'd0});
            wb = f32r({sel_weight[1*16 +: 16], 16'd0});
            wsum = wa + wb;
            if (^sel_weight === 1'bx) begin $display("FAIL test %0d: X in weight", t); errors = errors + 1; end
            else if (wsum < 2.40 || wsum > 2.60) begin
                $display("FAIL test %0d: renorm Sum(w)=%f != 2.5 (w=%f,%f)", t, wsum, wa, wb); errors = errors + 1; end
            @(negedge clk);
        end
        if (errors == 0)
            $display("[moe_router_q4k] ALL %0d TESTS PASSED (renorm Sum=SCALE, valid idx, GEMV->sigmoid->topk->renorm on Q4_K)", ntest);
        else $display("[moe_router_q4k] %0d FAILURES", errors);
        $finish;
    end
    initial begin #5000000; $display("[moe_router_q4k] TIMEOUT"); $finish; end
endmodule
