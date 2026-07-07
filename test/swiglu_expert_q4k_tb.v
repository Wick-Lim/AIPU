`timescale 1ns/1ps
// swiglu_expert_q4k_tb.v -- functional TB for the Q4_K SwiGLU expert.
// Q4_K weight-response model + real-valued golden (tolerance), proving the operator
// composes gate/up/down (glm_matmul_q4k) + silu (glm_act) + merge end-to-end. The
// bit-exact gate is glm_matmul_q4k itself (proven 480/480); this checks the wiring.
module swiglu_expert_q4k_tb;
    localparam integer HIDDEN = 8;
    localparam integer INTER  = 8;
    localparam integer TN     = 4;
    localparam integer PE_M   = 1;
    localparam integer KMAX   = 256;    // NSB=1
    localparam integer NGgu   = INTER/TN;   // gate/up groups
    localparam integer NGd    = HIDDEN/TN;  // down groups

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg  [16*HIDDEN*PE_M-1:0] x_vec = 0;
    wire busy, done;
    wire [16*HIDDEN*PE_M-1:0] y_out;

    // DUT weight-request wires
    wire                     w_req;
    wire [1:0]               w_sel;
    wire [$clog2((INTER>HIDDEN?INTER:HIDDEN)/TN+1)-1:0] w_grp;
    wire [$clog2(KMAX+1)-1:0] w_k;
    // weight-response (driven combinationally by this TB's model)
    reg  [4*TN-1:0]  w_q, w_q_up;
    reg  [16*TN-1:0] w_d, w_dmin, w_d_up, w_dmin_up;
    reg  [96*TN-1:0] w_scales, w_scales_up;

    swiglu_expert_q4k #(.HIDDEN(HIDDEN), .INTER(INTER), .TN(TN), .KMAX(KMAX), .PE_M(PE_M)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done), .x_vec(x_vec),
        .w_req(w_req), .w_sel(w_sel), .w_grp(w_grp), .w_k(w_k),
        .w_q(w_q), .w_q_up(w_q_up),
        .w_d(w_d), .w_dmin(w_dmin), .w_scales(w_scales),
        .w_d_up(w_d_up), .w_dmin_up(w_dmin_up), .w_scales_up(w_scales_up),
        .y_out(y_out)
    );

    // ---- stored weights (per test) ----
    localparam [1:0] SEL_GATE = 2'd0, SEL_DOWN = 2'd2;
    reg [3:0]  Gq [0:INTER-1][0:HIDDEN-1];  reg [15:0] Gd[0:INTER-1]; reg [15:0] Gm[0:INTER-1]; reg [95:0] Gs[0:INTER-1];
    reg [3:0]  Uq [0:INTER-1][0:HIDDEN-1];  reg [15:0] Ud[0:INTER-1]; reg [15:0] Um[0:INTER-1]; reg [95:0] Us[0:INTER-1];
    reg [3:0]  Dq [0:HIDDEN-1][0:INTER-1];  reg [15:0] Dd[0:HIDDEN-1]; reg [15:0] Dm[0:HIDDEN-1]; reg [95:0] Ds[0:HIDDEN-1];

    integer t2, col;
    // combinational weight-response model
    always @* begin
        w_q = 0; w_q_up = 0; w_d = 0; w_dmin = 0; w_scales = 0;
        w_d_up = 0; w_dmin_up = 0; w_scales_up = 0;
        for (t2 = 0; t2 < TN; t2 = t2 + 1) begin
            col = w_grp*TN + t2;
            if (w_sel == SEL_GATE) begin
                if (col < INTER) begin
                    w_q      [4*t2  +: 4]  = Gq[col][w_k];
                    w_q_up   [4*t2  +: 4]  = Uq[col][w_k];
                    w_d      [16*t2 +: 16] = Gd[col]; w_dmin   [16*t2 +: 16] = Gm[col]; w_scales   [96*t2 +: 96] = Gs[col];
                    w_d_up   [16*t2 +: 16] = Ud[col]; w_dmin_up[16*t2 +: 16] = Um[col]; w_scales_up[96*t2 +: 96] = Us[col];
                end
            end else begin // DOWN
                if (col < HIDDEN) begin
                    w_q [4*t2  +: 4]  = Dq[col][w_k];
                    w_d [16*t2 +: 16] = Dd[col]; w_dmin[16*t2 +: 16] = Dm[col]; w_scales[96*t2 +: 96] = Ds[col];
                end
            end
        end
    end

    // ---- golden compare helpers ----
    function real bf16r(input [15:0] b);
        reg [31:0] f; begin f = {b, 16'd0}; bf16r = $bitstoreal({{32{1'b0}}, f}); end
    endfunction
    // (iverilog: build real from fp32 bits)
    function real f32r(input [31:0] f);
        real m; integer e, i; reg s;
        begin
            s = f[31]; e = f[30:23];
            if (e == 0) begin m = 0.0; for (i=0;i<23;i=i+1) if (f[i]) m = m + 2.0**(i-23); m = m * (2.0**-126); end
            else begin m = 1.0; for (i=0;i<23;i=i+1) if (f[i]) m = m + 2.0**(i-23); m = m * (2.0**(e-127)); end
            f32r = s ? -m : m;
        end
    endfunction

    integer fd, ntest, hh, ii, tn_f, t, n, k, o, code, errors, checks, r;
    reg [15:0] tmp; reg [95:0] stmp; reg [15:0] expy [0:HIDDEN*PE_M-1]; real tol [0:HIDDEN*PE_M-1];
    real gotv, expv, tt;

    task load_w4(input integer N, input integer K, input integer which);
        integer nn, kk; reg [15:0] dt, mt; reg [95:0] st;
        begin
            for (nn = 0; nn < N; nn = nn + 1) begin
                code=$fscanf(fd,"%h",dt); code=$fscanf(fd,"%h",mt); code=$fscanf(fd,"%h",st);
                for (kk = 0; kk < K; kk = kk + 1) begin
                    code=$fscanf(fd,"%h",tmp);
                    if (which==0) Gq[nn][kk]=tmp[3:0];
                    else if (which==1) Uq[nn][kk]=tmp[3:0];
                    else Dq[nn][kk]=tmp[3:0];
                end
                if (which==0) begin Gd[nn]=dt; Gm[nn]=mt; Gs[nn]=st; end
                else if (which==1) begin Ud[nn]=dt; Um[nn]=mt; Us[nn]=st; end
                else begin Dd[nn]=dt; Dm[nn]=mt; Ds[nn]=st; end
            end
        end
    endtask

    initial begin
        errors=0; checks=0;
        fd = $fopen("build/swiglu_q4k_vec.txt","r");
        if (fd==0) begin $display("[swiglu_expert_q4k] FAIL: no vec file"); $finish; end
        code=$fscanf(fd,"%d %d %d %d", ntest, hh, ii, tn_f);
        @(negedge clk); rst=0;
        for (t=0; t<ntest; t=t+1) begin
            for (k=0; k<HIDDEN*PE_M; k=k+1) begin code=$fscanf(fd,"%h",tmp); x_vec[16*k +: 16]=tmp; end
            load_w4(INTER, HIDDEN, 0);   // gate
            load_w4(INTER, HIDDEN, 1);   // up
            load_w4(HIDDEN, INTER, 2);   // down
            for (k=0; k<HIDDEN*PE_M; k=k+1) begin code=$fscanf(fd,"%h",expy[k]); end
            for (k=0; k<HIDDEN*PE_M; k=k+1) begin code=$fscanf(fd,"%f",tol[k]); end

            @(negedge clk); start=1; @(negedge clk); start=0;
            k=0; while (!done && k<20000) begin @(posedge clk); k=k+1; end
            if (!done) begin $display("[swiglu_expert_q4k] FAIL test %0d: no done", t); errors=errors+1; end
            #1;
            for (o=0; o<HIDDEN*PE_M; o=o+1) begin
                gotv = f32r({y_out[16*o +: 16], 16'd0});
                expv = f32r({expy[o], 16'd0});
                tt   = tol[o];
                checks = checks + 1;
                if (^y_out[16*o +: 16] === 1'bx) begin $display("FAIL test %0d o%0d: X", t, o); errors=errors+1; end
                else if (((gotv-expv) > tt) || ((expv-gotv) > tt)) begin
                    $display("FAIL test %0d o%0d: got %f exp %f tol %f", t, o, gotv, expv, tt); errors=errors+1;
                end
            end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors==0) $display("[swiglu_expert_q4k] ALL %0d TESTS PASSED (%0d experts, functional vs Q4_K golden)", checks, ntest);
        else $display("[swiglu_expert_q4k] %0d/%0d FAILURES", errors, checks);
        $finish;
    end
    initial begin #5000000; $display("[swiglu_expert_q4k] TIMEOUT"); $finish; end
endmodule
