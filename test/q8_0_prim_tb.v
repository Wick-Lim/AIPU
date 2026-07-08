`timescale 1ns/1ps
`include "q4k_mixed.vh"
// q8_0_prim_tb.v -- BIT-EXACT directed check of the Q8_0 per-weight dequant
//   primitive q8_0_deq (q4k_mixed.vh): w = d * f32(int8 qs), d fp16, qs signed
//   int8, no offset -- vs the ggml golden (tools/q4k_ref.dequantize_block_q8_0)
//   emitted by tools/q4k_mixed_gen.py.  Full 32-bit (uint32) compare.
//
//   Sources (from tools/q4k_mixed_gen.py):
//     build/q8_0_prim_vec.txt  330 raw Q8_0 blocks (edge + random), golden y[32]
//                              edges: qs=-128/-1/0/127, d subnormal/+-0/large
//     build/q4k_deq_vec.txt    independent Q8_0 per-weight code golden
module q8_0_prim_tb;
    integer errors  = 0;
    integer n_q8raw = 0;
    integer n_q8cod = 0;

    reg [15:0] d16;
    reg [7:0]  qs [0:31];
    reg [31:0] yg [0:31];
    integer fd, r, nblk, b, i;

    integer ntest, t, ty, npos, p, j;
    reg [15:0] dtmp;
    reg [7:0]  sctmp;
    reg [15:0] codetmp;
    reg [31:0] wexp;

    initial begin
        // ---------- Q8_0 raw blocks (edge + random) ----------
        fd = $fopen("build/q8_0_prim_vec.txt", "r");
        if (fd == 0) begin $display("FAIL cannot open build/q8_0_prim_vec.txt"); errors = errors + 1; end
        r = $fscanf(fd, "%d", nblk);
        for (b = 0; b < nblk; b = b + 1) begin
            r = $fscanf(fd, "%h", d16);
            for (i = 0; i < 32; i = i + 1) r = $fscanf(fd, "%h", qs[i]);
            for (i = 0; i < 32; i = i + 1) r = $fscanf(fd, "%h", yg[i]);
            for (i = 0; i < 32; i = i + 1) begin
                n_q8raw = n_q8raw + 1;
                if (q8_0_deq(d16, qs[i]) !== yg[i]) begin
                    errors = errors + 1;
                    if (errors <= 20) $display("FAIL q8 b=%0d i=%0d d=%h qs=%h -> %h exp %h",
                                               b, i, d16, qs[i], q8_0_deq(d16, qs[i]), yg[i]);
                end
            end
        end
        $fclose(fd);

        // ---------- independent Q8_0 code golden from q4k_deq_vec.txt ----------
        fd = $fopen("build/q4k_deq_vec.txt", "r");
        if (fd == 0) begin $display("FAIL cannot open build/q4k_deq_vec.txt"); errors = errors + 1; end
        r = $fscanf(fd, "%d", ntest);
        for (t = 0; t < ntest; t = t + 1) begin
            r = $fscanf(fd, "%d %d", ty, npos);
            if (ty == 1) begin                              // Q6_K -- consume header + rows
                r = $fscanf(fd, "%h", dtmp);
                for (j = 0; j < 16; j = j + 1) r = $fscanf(fd, "%h", sctmp);
                for (p = 0; p < npos; p = p + 1) r = $fscanf(fd, "%h %h", codetmp, wexp);
            end else if (ty == 2) begin                     // Q8_0
                r = $fscanf(fd, "%h", dtmp);
                for (p = 0; p < npos; p = p + 1) begin
                    r = $fscanf(fd, "%h %h", codetmp, wexp);
                    n_q8cod = n_q8cod + 1;
                    if (q8_0_deq(dtmp, codetmp[7:0]) !== wexp) begin
                        errors = errors + 1;
                        if (errors <= 20) $display("FAIL deqvec q8 t=%0d p=%0d -> %h exp %h",
                                                   t, p, q8_0_deq(dtmp, codetmp[7:0]), wexp);
                    end
                end
            end else begin                                  // F16 -- consume rows
                for (p = 0; p < npos; p = p + 1) r = $fscanf(fd, "%h %h", codetmp, wexp);
            end
        end
        $fclose(fd);

        if (errors == 0)
            $display("[q8_0_prim] ALL %0d TESTS PASSED (%0d Q8_0 raw + %0d Q8_0 code)",
                     n_q8raw + n_q8cod, n_q8raw, n_q8cod);
        else
            $display("[q8_0_prim] %0d FAILURES", errors);
        $finish;
    end
endmodule
