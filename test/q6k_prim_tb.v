`timescale 1ns/1ps
`include "q4k_mixed.vh"
// q6k_prim_tb.v -- BIT-EXACT directed check of the mixed-type dequant primitives
//   s8_to_fp32 (q4k.vh) + q6k_assemble / q6k_deq / q6k_deq_raw (q4k_mixed.vh),
//   plus F16 passthrough (f16_deq), vs the ggml goldens emitted by
//   tools/q4k_mixed_gen.py.  Every comparison is a full 32-bit (uint32) compare.
//
//   Sources (all from tools/q4k_mixed_gen.py):
//     build/s8_fp32_vec.txt   exhaustive int8->fp32 (all 256 bytes)
//     build/q6k_prim_vec.txt  173 raw Q6_K blocks (edge + random), golden y[256]
//     build/q4k_deq_vec.txt   independent Q6_K/F16 per-weight code goldens
//     build/f16_prim_vec.txt  212 F16 values (edge + random)
//   The Q6_K blocks are walked in exact ggml y-position order (half/l/is_ +0/2/4/6)
//   so bit-extraction, the p>>4 scale index, the -32 signed offset, and the
//   (d*sc)*q multiply order are ALL exercised against the golden.
module q6k_prim_tb;
    integer errors  = 0;
    integer n_s8    = 0;
    integer n_q6raw = 0;
    integer n_q6cod = 0;
    integer n_f16   = 0;

    reg [31:0] s8mem [0:255];        // exhaustive s8->fp32 golden
    reg [15:0] d16;                  // one Q6_K block
    reg [7:0]  ql  [0:127];
    reg [7:0]  qh  [0:63];
    reg [7:0]  sc  [0:15];
    reg [31:0] yg  [0:255];

    integer fd, r, nblk, b, half, l, is_c, i;
    integer qlb, qhb, scb, yb;

    // q4k_deq_vec.txt parse
    integer ntest, t, ty, npos, p, j;
    reg [15:0] dtmp;
    reg [7:0]  sctmp [0:15];
    reg [15:0] codetmp;
    reg [31:0] wexp;
    integer nval, v;
    reg [15:0] raw16;

    // one Q6_K weight: check BOTH the raw-assemble path and the pre-assembled
    // code path against the same golden `exp`.
    task check_q6(input [15:0] d, input [7:0] scv,
                  input [7:0] qlv, input [7:0] qhv, input [1:0] sel, input [31:0] exp);
        reg [7:0]  qsig8;
        reg [5:0]  code6;
        reg [31:0] gr, gc;
        begin
            gr = q6k_deq_raw(d, scv, qlv, qhv, sel);      // raw ql/qh -> wdeq
            n_q6raw = n_q6raw + 1;
            if (gr !== exp) begin
                errors = errors + 1;
                if (errors <= 20) $display("FAIL q6raw d=%h sc=%h ql=%h qh=%h sel=%0d -> %h exp %h",
                                           d, scv, qlv, qhv, sel, gr, exp);
            end
            qsig8 = q6k_assemble(qlv, qhv, sel);           // int8(code-32)
            code6 = qsig8 + 8'd32;                          // recover 0..63
            gc    = q6k_deq(d, scv, code6);                 // pre-assembled path
            n_q6cod = n_q6cod + 1;
            if (gc !== exp) begin
                errors = errors + 1;
                if (errors <= 20) $display("FAIL q6code d=%h sc=%h code=%0d -> %h exp %h",
                                           d, scv, code6, gc, exp);
            end
        end
    endtask

    initial begin
        // ---------- s8_to_fp32 exhaustive (signed int8: min/max/neg all covered) ----
        $readmemh("build/s8_fp32_vec.txt", s8mem);
        for (i = 0; i < 256; i = i + 1) begin
            n_s8 = n_s8 + 1;
            if (s8_to_fp32(i[7:0]) !== s8mem[i]) begin
                errors = errors + 1;
                if (errors <= 20) $display("FAIL s8 %0d -> %h exp %h", i, s8_to_fp32(i[7:0]), s8mem[i]);
            end
        end

        // ---------- Q6_K raw blocks (edge + random), walked in ggml order ----------
        fd = $fopen("build/q6k_prim_vec.txt", "r");
        if (fd == 0) begin $display("FAIL cannot open build/q6k_prim_vec.txt"); errors = errors + 1; end
        r = $fscanf(fd, "%d", nblk);
        for (b = 0; b < nblk; b = b + 1) begin
            r = $fscanf(fd, "%h", d16);
            for (i = 0; i < 128; i = i + 1) r = $fscanf(fd, "%h", ql[i]);
            for (i = 0; i < 64;  i = i + 1) r = $fscanf(fd, "%h", qh[i]);
            for (i = 0; i < 16;  i = i + 1) r = $fscanf(fd, "%h", sc[i]);
            for (i = 0; i < 256; i = i + 1) r = $fscanf(fd, "%h", yg[i]);
            for (half = 0; half < 2; half = half + 1) begin
                qlb = half*64; qhb = half*32; scb = half*8; yb = half*128;
                for (l = 0; l < 32; l = l + 1) begin
                    is_c = l / 16;                          // 0 or 1
                    // sel 0/1/2/3 -> y+0/+32/+64/+96 ; sc[scb+is_+0/2/4/6]
                    check_q6(d16, sc[scb+is_c+0], ql[qlb+l],    qh[qhb+l], 2'd0, yg[yb+l+ 0]);
                    check_q6(d16, sc[scb+is_c+2], ql[qlb+l+32], qh[qhb+l], 2'd1, yg[yb+l+32]);
                    check_q6(d16, sc[scb+is_c+4], ql[qlb+l],    qh[qhb+l], 2'd2, yg[yb+l+64]);
                    check_q6(d16, sc[scb+is_c+6], ql[qlb+l+32], qh[qhb+l], 2'd3, yg[yb+l+96]);
                end
            end
        end
        $fclose(fd);

        // ---------- independent Q6_K code golden + F16 from q4k_deq_vec.txt ----------
        fd = $fopen("build/q4k_deq_vec.txt", "r");
        if (fd == 0) begin $display("FAIL cannot open build/q4k_deq_vec.txt"); errors = errors + 1; end
        r = $fscanf(fd, "%d", ntest);
        for (t = 0; t < ntest; t = t + 1) begin
            r = $fscanf(fd, "%d %d", ty, npos);
            if (ty == 1) begin                              // Q6_K
                r = $fscanf(fd, "%h", dtmp);
                for (j = 0; j < 16; j = j + 1) r = $fscanf(fd, "%h", sctmp[j]);
                for (p = 0; p < npos; p = p + 1) begin
                    r = $fscanf(fd, "%h %h", codetmp, wexp);
                    n_q6cod = n_q6cod + 1;
                    if (q6k_deq(dtmp, sctmp[p >> 4], codetmp[5:0]) !== wexp) begin
                        errors = errors + 1;
                        if (errors <= 20) $display("FAIL deqvec q6 t=%0d p=%0d -> %h exp %h",
                                                   t, p, q6k_deq(dtmp, sctmp[p>>4], codetmp[5:0]), wexp);
                    end
                end
            end else if (ty == 2) begin                     // Q8_0 -- consume (checked in q8_0_prim_tb)
                r = $fscanf(fd, "%h", dtmp);
                for (p = 0; p < npos; p = p + 1) r = $fscanf(fd, "%h %h", codetmp, wexp);
            end else begin                                  // F16
                for (p = 0; p < npos; p = p + 1) begin
                    r = $fscanf(fd, "%h %h", codetmp, wexp);
                    n_f16 = n_f16 + 1;
                    if (f16_deq(codetmp) !== wexp) begin
                        errors = errors + 1;
                        if (errors <= 20) $display("FAIL deqvec f16 t=%0d p=%0d -> %h exp %h",
                                                   t, p, f16_deq(codetmp), wexp);
                    end
                end
            end
        end
        $fclose(fd);

        // ---------- F16 passthrough (edge: +/-0, subnormal, min/max normal, inf) ----
        fd = $fopen("build/f16_prim_vec.txt", "r");
        if (fd == 0) begin $display("FAIL cannot open build/f16_prim_vec.txt"); errors = errors + 1; end
        r = $fscanf(fd, "%d", nval);
        for (v = 0; v < nval; v = v + 1) begin
            r = $fscanf(fd, "%h %h", raw16, wexp);
            n_f16 = n_f16 + 1;
            if (f16_deq(raw16) !== wexp) begin
                errors = errors + 1;
                if (errors <= 20) $display("FAIL f16prim v=%0d raw=%h -> %h exp %h",
                                           v, raw16, f16_deq(raw16), wexp);
            end
        end
        $fclose(fd);

        if (errors == 0)
            $display("[q6k_prim] ALL %0d TESTS PASSED (%0d s8 + %0d Q6_K raw + %0d Q6_K code + %0d F16)",
                     n_s8 + n_q6raw + n_q6cod + n_f16, n_s8, n_q6raw, n_q6cod, n_f16);
        else
            $display("[q6k_prim] %0d FAILURES", errors);
        $finish;
    end
endmodule
