`timescale 1ns/1ps
//============================================================================
// weight_loader_q4k_tb.v  --  BIT-EXACT check for weight_loader_q4k.
//
//   The Q4_K sibling of test/weight_loader_tb.v.  weight_loader_q4k reads a
//   memory image in the storage layout it expects, and DRIVES glm_matmul_q4k's
//   Q4_K weight pull (mm_start / mm_k_len / mm_w_q / mm_w_d / mm_w_dmin /
//   mm_w_scales / mm_in_valid).  The TB owns the activation side (a_col),
//   tracking the loader's beat stream.  The streamed GEMM output is compared
//   BIT-EXACT against the ggml Q4_K golden C from tools/q4k_ref.py, materialised
//   by tools/q4k_matmul_gen.py into build/wlq4k_vec.txt.
//
//   Because the golden C is `bf16( SUM_k fp32(a_k)*w_deq_k )` computed by
//   q4k_ref.matmul_q4k_col (bit-exact to ggml dequantize_row_q4_K), a green run
//   proves the loader reconstructs the Q4_K weight pull -- header (d/dmin/scales)
//   super-block packing + 4-bit code stream, in the exact order/timing the GEMM
//   consumes -- with NO re-quantization.  X-AWARE: any X bit is a hard failure.
//   Covers single- and multi-super-block tiles (n_sblk = ceil(K/256) = 1..3) and
//   partial K.  Emits "ALL <N> TESTS PASSED" + $fatal on mismatch.
//
//   Vector file (build/wlq4k_vec.txt), format identical to build/q4k_vec.txt:
//     line 0        : NTEST PE_M PE_N
//     per tile      : "K NSB"
//                     d[pj][sb]      (PE_N*NSB x 4hex fp16, pj-outer sb-inner)
//                     dmin[pj][sb]   (PE_N*NSB x 4hex fp16)
//                     scales[pj][sb] (PE_N*NSB x 24hex, 96b packed)
//                     for k: a[pi] (PE_M x 4hex bf16)  q[pj] (PE_N x 1hex 4b)
//                     c[pi*PE_N+pj]  (PE_M*PE_N x 4hex bf16 golden)
//============================================================================
module weight_loader_q4k_tb;
    localparam integer PE_M   = 2;                 // must match the vector file
    localparam integer PE_N   = 2;
    localparam integer KMAX   = 1024;              // one..four Q4_K super-blocks along K
    localparam integer NSB    = (KMAX + 255) / 256; // 4
    localparam integer ADDR_W = 16;
    localparam integer DATA_W = 256;               // Q4_K super-block header / code beat
    localparam integer KW     = $clog2(KMAX+1);
    localparam integer SBW    = $clog2(NSB+1);

    integer errors = 0;
    integer checks = 0;

    // ---------------- shared clock / reset ----------------
    reg clk = 1'b0;
    reg rst;
    always #5 clk = ~clk;

    // ---------------- tile storage (declared before use) ----------------
    reg [15:0] a_a  [0:PE_M-1][0:KMAX-1];   // bf16 activations   A[pi][k]
    reg [ 3:0] q_a  [0:PE_N-1][0:KMAX-1];   // 4-bit Q4_K codes   W[k][pj]
    reg [15:0] d_a  [0:PE_N-1][0:NSB-1];    // fp16 super-block d    per (col, sb)
    reg [15:0] dm_a [0:PE_N-1][0:NSB-1];    // fp16 super-block dmin per (col, sb)
    reg [95:0] sc_a [0:PE_N-1][0:NSB-1];    // 96b packed scales     per (col, sb)
    reg [15:0] exp_c[0:PE_M*PE_N-1];        // golden bf16 output

    // ====================================================================
    // DUT PATH : weight_loader_q4k reads a memory image and DRIVES glm_matmul_q4k.
    // ====================================================================
    reg                       load = 1'b0;
    reg  [ADDR_W-1:0]         desc_base;
    reg  [KW-1:0]             desc_klen;
    reg  [SBW-1:0]            desc_nsblk;

    wire                      mem_en;
    wire [ADDR_W-1:0]         mem_addr;
    reg  [DATA_W-1:0]         mem_data;

    wire                      mm_start;
    wire [KW-1:0]             mm_k_len;
    wire [ 4*PE_N-1:0]        mm_w_q;
    wire [16*PE_N*NSB-1:0]    mm_w_d;
    wire [16*PE_N*NSB-1:0]    mm_w_dmin;
    wire [96*PE_N*NSB-1:0]    mm_w_scales;
    wire                      mm_in_valid;
    wire                      ld_busy;
    wire                      ld_done;

    weight_loader_q4k #(
        .PE_N(PE_N), .KMAX(KMAX), .ADDR_W(ADDR_W), .DATA_W(DATA_W)
    ) u_ld (
        .clk(clk), .rst(rst),
        .load(load), .desc_base(desc_base), .desc_klen(desc_klen), .desc_nsblk(desc_nsblk),
        .mem_en(mem_en), .mem_addr(mem_addr), .mem_data(mem_data),
        .mm_start(mm_start), .mm_k_len(mm_k_len), .mm_w_q(mm_w_q),
        .mm_w_d(mm_w_d), .mm_w_dmin(mm_w_dmin), .mm_w_scales(mm_w_scales),
        .mm_in_valid(mm_in_valid),
        .busy(ld_busy), .done(ld_done)
    );

    // activation side for the DUT: the TB owns it (the loader is the beat
    // master).  a_col tracks the loader's beat stream.
    reg  [KW:0]              dut_beat;
    integer                  dut_K;
    reg  [16*PE_M-1:0]       acol_d;
    wire                      dut_busy;
    wire                      dut_ov;
    wire [16*PE_M*PE_N-1:0]   dut_cout;

    glm_matmul_q4k #(.PE_M(PE_M), .PE_N(PE_N), .KMAX(KMAX)) u_dut (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_k_len),
        .w_d(mm_w_d), .w_dmin(mm_w_dmin), .w_scales(mm_w_scales),
        .in_valid(mm_in_valid), .a_col(acol_d), .w_q(mm_w_q),
        .busy(dut_busy), .out_valid(dut_ov), .c_out(dut_cout)
    );

    // beat counter: reset at mm_start, ++ on each streamed weight beat, so on
    // the cycle mm_in_valid is high for beat k, dut_beat == k.
    always @(posedge clk) begin
        if (rst)            dut_beat <= {(KW+1){1'b0}};
        else if (mm_start)  dut_beat <= {(KW+1){1'b0}};
        else if (mm_in_valid) dut_beat <= dut_beat + 1'b1;
    end

    // combinational activation feed -> A[*][dut_beat] on the in_valid cycle.
    integer ai;
    always @* begin
        acol_d = {(16*PE_M){1'b0}};
        for (ai = 0; ai < PE_M; ai = ai + 1)
            acol_d[16*ai +: 16] = (dut_beat < dut_K) ? a_a[ai][dut_beat] : 16'h0000;
    end

    // ---------------- latency-1 read memory (DDR5/Flash stub) ----------------
    localparam integer MEM_WORDS = 4096;
    reg [DATA_W-1:0] mem [0:MEM_WORDS-1];
    always @(posedge clk) begin
        if (mem_en) mem_data <= mem[mem_addr];
    end

    // ====================================================================
    // Build the memory image in the storage layout the loader expects.
    //   HEADER region : base + (pj*nsblk + sb), one packed word each
    //                     word[15:0]=d, word[31:16]=dmin, word[127:32]=scales.
    //   CODE   region : base + nsblk*PE_N + k, word[4*pj+:4] = q_a[pj][k].
    // ====================================================================
    task build_mem(input [ADDR_W-1:0] base, input integer K, input integer nsblk);
        integer i, k, pj, sb, li;
        reg [DATA_W-1:0] word;
        begin
            for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = {DATA_W{1'b0}};
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < nsblk; sb = sb + 1) begin
                    li   = pj*nsblk + sb;
                    word = {DATA_W{1'b0}};
                    word[15:0]   = d_a [pj][sb];
                    word[31:16]  = dm_a[pj][sb];
                    word[127:32] = sc_a[pj][sb];
                    mem[base + li] = word;
                end
            for (k = 0; k < K; k = k + 1) begin
                word = {DATA_W{1'b0}};
                for (pj = 0; pj < PE_N; pj = pj + 1) word[4*pj +: 4] = q_a[pj][k];
                mem[base + nsblk*PE_N + k] = word;
            end
        end
    endtask

    // drive one tile: lay the weights into the image, let the loader pull.
    task run_dut(input [ADDR_W-1:0] base, input integer K, input integer nsblk);
        begin
            build_mem(base, K, nsblk);
            dut_K      = K;
            desc_base  = base;
            desc_klen  = K[KW-1:0];
            desc_nsblk = nsblk[SBW-1:0];
            @(negedge clk);
            load = 1'b1;
            @(negedge clk);
            load = 1'b0;
            // wait for the streamed GEMM to complete.
            do @(negedge clk); while (dut_ov !== 1'b1);
        end
    endtask

    // ====================================================================
    // bit-exact compare (X-aware) of the loader-streamed GEMM vs the golden.
    // ====================================================================
    task compare(input integer t, input integer K, input integer nsblk);
        integer pi, pj, e;
        reg [15:0] g, d;
        begin
            for (pi = 0; pi < PE_M; pi = pi + 1)
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    e = pi*PE_N + pj;
                    checks = checks + 1;
                    d = dut_cout[16*e +: 16];
                    g = exp_c[e];
                    if (^d === 1'bx) begin
                        errors = errors + 1;
                        $display("  FAIL test %0d [%0d,%0d]: loader/GEMM has X (%b)", t, pi, pj, d);
                    end else if (d !== g) begin
                        errors = errors + 1;
                        $display("  FAIL test %0d [%0d,%0d] (K=%0d nsblk=%0d): loader=%h golden=%h",
                                 t, pi, pj, K, nsblk, d, g);
                    end
                end
        end
    endtask

    // ====================================================================
    // Stimulus : read build/wlq4k_vec.txt, run each tile through the loader.
    // ====================================================================
    integer fd, code, ntest, pm, pn, t, K, nsblk, pi, pj, sb, k;
    reg [15:0] tmp16;
    reg [95:0] tmp96;
    reg [3:0]  tmpq;
    reg [ADDR_W-1:0] base;

    initial begin
        fd = $fopen("build/wlq4k_vec.txt", "r");
        if (fd == 0) begin
            $display("[weight_loader_q4k] FAIL: cannot open build/wlq4k_vec.txt (run: python3 tools/q4k_matmul_gen.py 40 2 2 build/wlq4k_vec.txt)");
            $fatal(1, "missing vectors");
        end
        code = $fscanf(fd, "%d %d %d", ntest, pm, pn);
        if (pm != PE_M || pn != PE_N) begin
            $display("[weight_loader_q4k] FAIL: vector PE_M/PE_N (%0d,%0d) != TB (%0d,%0d)",
                     pm, pn, PE_M, PE_N);
            $fatal(1, "dim mismatch");
        end

        rst = 1'b1;
        repeat (5) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            code = $fscanf(fd, "%d %d", K, nsblk);
            // header params  (pj-outer, sb-inner)
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < nsblk; sb = sb + 1) begin
                    code = $fscanf(fd, "%h", tmp16); d_a[pj][sb] = tmp16; end
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < nsblk; sb = sb + 1) begin
                    code = $fscanf(fd, "%h", tmp16); dm_a[pj][sb] = tmp16; end
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < nsblk; sb = sb + 1) begin
                    code = $fscanf(fd, "%h", tmp96); sc_a[pj][sb] = tmp96; end
            // per-beat activations + codes
            for (k = 0; k < K; k = k + 1) begin
                for (pi = 0; pi < PE_M; pi = pi + 1) begin
                    code = $fscanf(fd, "%h", tmp16); a_a[pi][k] = tmp16; end
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    code = $fscanf(fd, "%h", tmpq);  q_a[pj][k] = tmpq; end
            end
            // golden outputs
            for (pi = 0; pi < PE_M*PE_N; pi = pi + 1) begin
                code = $fscanf(fd, "%h", tmp16); exp_c[pi] = tmp16; end

            // vary the descriptor base to exercise a non-trivial base too.
            base = (t & 1) ? 16'd1600 : 16'd0;
            run_dut(base, K, nsblk);
            compare(t, K, nsblk);
            repeat (2) @(negedge clk);
        end
        $fclose(fd);

        if (errors == 0)
            $display("[weight_loader_q4k] ALL %0d TESTS PASSED (%0d tiles, loader->GEMM bit-exact vs ggml Q4_K golden)",
                     checks, ntest);
        else begin
            $display("[weight_loader_q4k] %0d/%0d CHECKS FAILED", errors, checks);
            $fatal(1, "weight_loader_q4k binding check FAILED");
        end
        $finish;
    end

    // safety timeout
    initial begin
        #5000000;
        $display("[weight_loader_q4k] TIMEOUT");
        $fatal(1, "weight_loader_q4k binding check TIMEOUT");
    end
endmodule
