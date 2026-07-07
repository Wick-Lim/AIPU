`timescale 1ns/1ps
`include "q4k.vh"
// q4k_prim_tb.v -- directed check of q4k.vh primitives (fp16 decode + scale unpack)
// vs the ggml golden (tools/q4k_ref.py). Bit-exact assertions.
module q4k_prim_tb;
    integer errors = 0;
    integer nf = 0, ns = 0;

    task check_f16(input [15:0] h, input [31:0] exp);
        reg [31:0] got;
        begin
            got = fp16_to_fp32(h);
            nf = nf + 1;
            if (got !== exp) begin
                $display("FAIL fp16 %h -> %h  exp %h", h, got, exp);
                errors = errors + 1;
            end
        end
    endtask

    task check_sm(input [3:0] j, input [95:0] sc, input [11:0] exp);
        reg [11:0] got;
        begin
            got = q4k_scale_min(j, sc);
            ns = ns + 1;
            if (got !== exp) begin
                $display("FAIL scale_min j=%0d -> %h  exp %h", j, got, exp);
                errors = errors + 1;
            end
        end
    endtask

    localparam [95:0] SC = 96'h6B2A36C53CC261873FE85103;

    initial begin
        // ---- fp16 decode (incl. signed zero, subnormals, max normal) ----
        check_f16(16'h0000, 32'h00000000);
        check_f16(16'h8000, 32'h80000000);
        check_f16(16'h3C00, 32'h3F800000);
        check_f16(16'h211F, 32'h3C23E000);
        check_f16(16'h1D1F, 32'h3BA3E000);
        check_f16(16'hB400, 32'hBE800000);
        check_f16(16'h5F00, 32'h43E00000);
        check_f16(16'h03FF, 32'h387FC000);   // largest subnormal
        check_f16(16'h0001, 32'h33800000);   // smallest subnormal
        check_f16(16'h7BFF, 32'h477FE000);   // max normal 65504

        // ---- get_scale_min_k4 for all 8 sub-blocks ----
        check_sm(4'd0, SC, 12'h1C3);
        check_sm(4'd1, SC, 12'h851);
        check_sm(4'd2, SC, 12'h0A8);
        check_sm(4'd3, SC, 12'hF3F);
        check_sm(4'd4, SC, 12'hB05);
        check_sm(4'd5, SC, 12'h4D6);
        check_sm(4'd6, SC, 12'hCBA);
        check_sm(4'd7, SC, 12'h18B);

        if (errors == 0)
            $display("[q4k_prim] ALL %0d TESTS PASSED (%0d fp16 decode, %0d scale unpack)", nf+ns, nf, ns);
        else
            $display("[q4k_prim] %0d FAILURES", errors);
        $finish;
    end
endmodule
