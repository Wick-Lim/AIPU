`timescale 1ns/1ps
//============================================================================
// weight_decomp_tb.v -- BIT-EXACT round-trip golden for src/weight_decomp.v,
// the on-chip streaming canonical-Huffman weight decompressor (glm_q4k_system
// under DECOMP=1).  Closes the functional-verification gap: the decompressor
// had NO testbench and NO gate ever set DECOMP=1, so a latent bug would ship
// WRONG WEIGHTS into every matmul the moment DECOMP is enabled.
//
// CONTRACT PROVEN (bit-exact, lossless):
//   the FP8 byte that comes out === the FP8 byte that went into the OFFLINE
//   encoder.  tools/fp8_huff.py (the matching length-limited package-merge
//   canonical Huffman encoder named in weight_decomp.v's own header) emits, per
//   block, the canonical tables + the compressed byte stream + the ORIGINAL
//   decoded bytes into build/weight_decomp_vec.txt.  This TB loads the tables,
//   streams the compressed bytes through the REAL weight_decomp, and asserts
//   EVERY decoded byte === the expected byte, that EOB fires, and that exactly
//   the expected byte-count is produced -- across 7 blocks spanning code lengths
//   1..15, single-symbol / near-uniform / skewed distributions, and output
//   back-pressure (out_ready toggling).  The generator itself self-checks
//   encode==independent-decode before emitting, so this is a THREE-way agreement
//   (Python encode / Python decode-model / RTL decode).
//
// Vec file layout (whitespace, $fscanf):
//   line 0:            NTEST MAXLEN EOB_SYM
//   per test, 5 lines: NCODES NCOMP NDEC MODE
//                      count[1..MAXLEN]           (decimal)
//                      sym[0..NCODES-1]           (decimal canonical order)
//                      comp[0..NCOMP-1]           (hex bytes, compressed stream)
//                      dec[0..NDEC-1]             (hex bytes, expected output)
//============================================================================
`ifndef TB_VEC
    `define TB_VEC "build/weight_decomp_vec.txt"
`endif
`ifndef TB_TIMEOUT_NS
    `define TB_TIMEOUT_NS 800000000
`endif
module weight_decomp_tb;
    localparam integer MAXLEN  = 15;
    localparam integer SYMW    = 9;
    localparam integer COUNTW  = 10;
    localparam integer AW      = 9;
    localparam integer BUFW    = 32;
    localparam integer EOB_SYM = 256;

    // storage bounds (max over the corpus, sized generously)
    localparam integer MAXCOMP = 16384;
    localparam integer MAXDEC  = 65536;
    localparam integer MAXSYM  = 512;

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1;

    // ---- DUT ports ----
    reg               tbl_we = 0, tbl_sel = 0;
    reg  [AW-1:0]     tbl_addr = 0;
    reg  [COUNTW-1:0] tbl_wdata = 0;
    reg               start = 0;
    reg  [7:0]        in_byte = 0;
    reg               in_valid = 0;
    wire              in_ready;
    wire [7:0]        out_byte;
    wire              out_valid;
    reg               out_ready = 1;
    wire              eob;

    weight_decomp #(
        .MAXLEN(MAXLEN), .SYMW(SYMW), .COUNTW(COUNTW),
        .AW(AW), .BUFW(BUFW), .EOB_SYM(EOB_SYM)
    ) dut (
        .clk(clk), .rst(rst),
        .tbl_we(tbl_we), .tbl_sel(tbl_sel),
        .tbl_addr(tbl_addr), .tbl_wdata(tbl_wdata),
        .start(start),
        .in_byte(in_byte), .in_valid(in_valid), .in_ready(in_ready),
        .out_byte(out_byte), .out_valid(out_valid),
        .out_ready(out_ready), .eob(eob)
    );

    integer fd, code, ntest, f_maxlen, f_eob;
    integer t, l, i, ncodes, ncomp, ndec, mode;
    integer ci, di, guard, guard_max, errors, checks;
    reg [COUNTW-1:0] cnt_arr [0:MAXLEN];
    reg [SYMW:0]     sym_arr [0:MAXSYM-1];
    reg [7:0]        comp_arr [0:MAXCOMP-1];
    reg [7:0]        dec_arr  [0:MAXDEC-1];
    reg [15:0]       tmp;

    initial begin
        errors = 0; checks = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin
            $display("[weight_decomp] FAIL: cannot open %s (run: python3 tools/fp8_huff.py)", `TB_VEC);
            $fatal;
        end
        code = $fscanf(fd, "%d %d %d", ntest, f_maxlen, f_eob);
        if (f_maxlen != MAXLEN || f_eob != EOB_SYM) begin
            $display("[weight_decomp] FAIL: vec MAXLEN/EOB (%0d,%0d) != TB (%0d,%0d)",
                     f_maxlen, f_eob, MAXLEN, EOB_SYM);
            $fatal;
        end

        // release reset
        @(negedge clk); rst = 0; @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            code = $fscanf(fd, "%d %d %d %d", ncodes, ncomp, ndec, mode);
            if (ncomp > MAXCOMP || ndec > MAXDEC || ncodes > MAXSYM) begin
                $display("[weight_decomp] FAIL test %0d: block exceeds TB storage (ncodes=%0d ncomp=%0d ndec=%0d)", t, ncodes, ncomp, ndec);
                $fatal;
            end
            for (l = 1; l <= MAXLEN; l = l + 1) begin
                code = $fscanf(fd, "%d", cnt_arr[l]);
            end
            for (i = 0; i < ncodes; i = i + 1) begin
                code = $fscanf(fd, "%d", tmp); sym_arr[i] = tmp[SYMW:0];
            end
            for (i = 0; i < ncomp; i = i + 1) begin
                code = $fscanf(fd, "%h", comp_arr[i]);
            end
            for (i = 0; i < ndec; i = i + 1) begin
                code = $fscanf(fd, "%h", dec_arr[i]);
            end

            // ---- load canonical tables (count_table addr 1..MAXLEN; symbol_table 0..ncodes-1) ----
            for (l = 1; l <= MAXLEN; l = l + 1) begin
                @(negedge clk);
                tbl_we = 1; tbl_sel = 0; tbl_addr = l[AW-1:0]; tbl_wdata = cnt_arr[l];
            end
            for (i = 0; i < ncodes; i = i + 1) begin
                @(negedge clk);
                tbl_we = 1; tbl_sel = 1; tbl_addr = i[AW-1:0];
                tbl_wdata = {{(COUNTW-SYMW-1){1'b0}}, sym_arr[i]};
            end
            @(negedge clk); tbl_we = 0;

            // ---- begin the block (clears bit/codeword state + eob) ----
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;

            // ---- stream compressed bytes in, capture + check decoded bytes out ----
            ci = 0; di = 0; guard = 0;
            in_valid = 0; in_byte = 0; out_ready = 1'b1;
            guard_max = 16*ncomp + 4*ndec + 2000;
            while (eob !== 1'b1 && guard < guard_max) begin
                // drive input for the upcoming posedge
                if (ci < ncomp) begin in_valid = 1'b1; in_byte = comp_arr[ci]; end
                else            begin in_valid = 1'b0;                          end
                // output back-pressure: mode 1 toggles out_ready to stall the decoder
                out_ready = (mode == 0) ? 1'b1 : guard[0];

                // handshakes that WILL fire at the next posedge (all sampled regs stable now)
                if (in_valid && in_ready) ci = ci + 1;
                if (out_valid && out_ready) begin
                    checks = checks + 1;
                    if (^out_byte === 1'bx) begin
                        $display("FAIL test %0d: X in decoded byte %0d", t, di);
                        errors = errors + 1;
                    end else if (di >= ndec) begin
                        $display("FAIL test %0d: decoder emitted MORE than %0d bytes", t, ndec);
                        errors = errors + 1;
                    end else if (out_byte !== dec_arr[di]) begin
                        $display("FAIL test %0d: decoded byte %0d = %02h, expected %02h",
                                 t, di, out_byte, dec_arr[di]);
                        errors = errors + 1;
                    end
                    di = di + 1;
                end
                @(negedge clk);
                guard = guard + 1;
            end
            in_valid = 0;

            // ---- end-of-block invariants ----
            checks = checks + 1;
            if (eob !== 1'b1) begin
                $display("FAIL test %0d: eob never asserted (got %0d of %0d bytes)", t, di, ndec);
                errors = errors + 1;
            end
            if (di !== ndec) begin
                $display("FAIL test %0d: produced %0d bytes, expected %0d", t, di, ndec);
                errors = errors + 1;
            end
        end
        $fclose(fd);

        if (errors == 0)
            $display("[weight_decomp] ALL %0d TESTS PASSED (%0d blocks, bit-exact canonical-Huffman round-trip vs tools/fp8_huff.py)", checks, ntest);
        else begin
            $display("[weight_decomp] %0d/%0d FAILURES", errors, checks);
            $fatal;
        end
        $finish;
    end

    initial begin #`TB_TIMEOUT_NS; $display("[weight_decomp] TIMEOUT"); $fatal; end
endmodule
