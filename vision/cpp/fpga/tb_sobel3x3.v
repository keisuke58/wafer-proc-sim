// tb_sobel3x3.v — self-checking testbench for the streaming Sobel core.
//
// Builds a WxH test image, zero-pads it to (W+2)x(H+2), streams the padded
// frame through the DUT, and compares every emitted (gx, gy) against an
// independent reference Sobel computed here from the same padded array.
// Prints "TB PASS"/"TB FAIL" and exits non-zero on any mismatch.
`default_nettype none
`timescale 1ns/1ps

module tb_sobel3x3;
    localparam integer W  = 6;
    localparam integer H  = 8;
    localparam integer PW = W + 2;
    localparam integer PH = H + 2;
    localparam integer DW = 8;
    localparam integer TOTAL = PW * PH;     // padded pixels streamed
    localparam integer NOUT  = W * H;       // results expected

    reg clk = 0, rst_n = 0, in_valid = 0;
    reg [DW-1:0] in_pixel = 0;
    wire out_valid;
    wire signed [DW+3:0] out_gx, out_gy;

    sobel3x3 #(.PW(PW), .DW(DW)) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_pixel(in_pixel),
        .out_valid(out_valid), .out_gx(out_gx), .out_gy(out_gy));

    always #5 clk = ~clk;

    // Padded image: 1-pixel zero border around a deterministic pattern.
    reg [DW-1:0] P [0:PH-1][0:PW-1];
    integer pr, pc;
    initial begin
        for (pr = 0; pr < PH; pr = pr + 1)
            for (pc = 0; pc < PW; pc = pc + 1)
                if (pr == 0 || pr == PH-1 || pc == 0 || pc == PW-1)
                    P[pr][pc] = 0;                               // zero border
                else
                    P[pr][pc] = ((pc*37) + (pr*91) + (pc*pc)) % 256;  // pattern
    end

    // Independent reference Sobel at padded centre (cy, cx).
    function signed [DW+3:0] ref_gx(input integer cy, input integer cx);
        ref_gx = (P[cy-1][cx+1] + 2*P[cy][cx+1] + P[cy+1][cx+1])
               - (P[cy-1][cx-1] + 2*P[cy][cx-1] + P[cy+1][cx-1]);
    endfunction
    function signed [DW+3:0] ref_gy(input integer cy, input integer cx);
        ref_gy = (P[cy+1][cx-1] + 2*P[cy+1][cx] + P[cy+1][cx+1])
               - (P[cy-1][cx-1] + 2*P[cy-1][cx] + P[cy-1][cx+1]);
    endfunction

    integer fed = 0, cap = 0, fails = 0;
    integer oy, ox, cy, cx;
    reg signed [DW+3:0] egx, egy;

    // Drive one padded pixel per clock; capture registered outputs.
    always @(posedge clk) begin
        if (rst_n) begin
            if (fed < TOTAL) begin
                in_valid <= 1'b1;
                in_pixel <= P[fed / PW][fed % PW];
                fed <= fed + 1;
            end else begin
                in_valid <= 1'b0;
            end

            if (out_valid) begin
                oy = cap / W;  ox = cap % W;      // original coordinate
                cy = oy + 1;   cx = ox + 1;       // padded centre
                egx = ref_gx(cy, cx);
                egy = ref_gy(cy, cx);
                if (out_gx !== egx || out_gy !== egy) begin
                    fails = fails + 1;
                    if (fails <= 5)
                        $display("MISMATCH @ (%0d,%0d): dut=(%0d,%0d) ref=(%0d,%0d)",
                                 oy, ox, out_gx, out_gy, egx, egy);
                end
                cap = cap + 1;
            end
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        // Run long enough to stream all pixels plus pipeline drain.
        repeat (TOTAL + 20) @(posedge clk);

        $display("streamed=%0d captured=%0d expected=%0d fails=%0d",
                 fed, cap, NOUT, fails);
        if (cap != NOUT) begin
            $display("TB FAIL: output count %0d != expected %0d", cap, NOUT);
            $fatal(1);
        end
        if (fails != 0) begin
            $display("TB FAIL: %0d mismatches", fails);
            $fatal(1);
        end
        $display("TB PASS: %0d/%0d outputs bit-exact vs reference", cap, NOUT);
        $finish;
    end
endmodule

`default_nettype wire
