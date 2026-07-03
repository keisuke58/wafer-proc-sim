// sobel3x3.v — synthesizable streaming 3x3 Sobel operator (line-buffer design).
//
// One pixel per clock, raster order. Two line buffers hold the previous two
// rows; a 3x3 shift window feeds the gradient arithmetic. Borders are handled
// upstream by feeding a zero-padded stream: a WxH image is padded to
// (W+2)x(H+2) with a 1-pixel zero border, so every emitted window centre is
// interior to the padded frame and the datapath needs no border logic. For a
// padded width PW and height PH the module emits (PH-2)*(PW-2) = H*W results
// in raster order — one per original pixel.
//
// out_gx / out_gy are signed; the 3x3 Sobel of 8-bit pixels fits in 11 bits
// (max magnitude 4*255 = 1020). The ports are declared [DW+3:0] (= DW+4 bits,
// 12 bits for DW=8), giving one spare bit of headroom.
//
// Flow control: fixed-rate, always-accept. The core consumes one pixel per
// clock whenever in_valid is high and never stalls, so a downstream consumer
// must always be ready (there is no ready/backpressure handshake).
`default_nettype none
`timescale 1ns/1ps

module sobel3x3 #(
    parameter integer PW = 10,   // padded frame width (image width + 2)
    parameter integer DW = 8     // pixel bit width
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire [DW-1:0]        in_pixel,
    output reg                  out_valid,
    output reg  signed [DW+3:0] out_gx,
    output reg  signed [DW+3:0] out_gy
);
    localparam integer CW = (PW < 2) ? 1 : $clog2(PW);

    // Line buffers: lb0 = previous row, lb1 = the row before that.
    reg [DW-1:0] lb0 [0:PW-1];
    reg [DW-1:0] lb1 [0:PW-1];

    reg [CW-1:0] col;
    reg [31:0]   row;

    // 3x3 window taps. Suffix _l/_c/_r = left/centre/right column (col-2/-1/0).
    // Row prefix t/m/b = top/mid/bottom (row-2 / row-1 / row-0).
    reg [DW-1:0] t_l, t_c, t_r;
    reg [DW-1:0] m_l, m_c, m_r;
    reg [DW-1:0] b_l, b_c, b_r;

    // Blocking-updated window this cycle; sign-extend for the arithmetic.
    function signed [DW+3:0] sx(input [DW-1:0] v);
        sx = $signed({4'b0, v});
    endfunction

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col <= 0; row <= 0; out_valid <= 1'b0; out_gx <= 0; out_gy <= 0;
            t_l<=0; t_c<=0; t_r<=0; m_l<=0; m_c<=0; m_r<=0; b_l<=0; b_c<=0; b_r<=0;
        end else if (in_valid) begin
            // Shift the window left and bring in this column of each row.
            // Blocking so the gradient below sees the completed 3x3 window
            // whose bottom-right corner is the incoming pixel at (row, col).
            t_l = t_c; t_c = t_r; t_r = lb1[col];   // (row-2, col)
            m_l = m_c; m_c = m_r; m_r = lb0[col];   // (row-1, col)
            b_l = b_c; b_c = b_r; b_r = in_pixel;   // (row-0, col)

            // Sobel of the window centred at (row-1, col-1).
            out_gx <= (sx(t_r) + (sx(m_r) <<< 1) + sx(b_r))
                    - (sx(t_l) + (sx(m_l) <<< 1) + sx(b_l));
            out_gy <= (sx(b_l) + (sx(b_c) <<< 1) + sx(b_r))
                    - (sx(t_l) + (sx(t_c) <<< 1) + sx(t_r));
            out_valid <= (row >= 2) && (col >= 2);

            // Advance line buffers: old row-1 slides to row-2, this pixel
            // becomes row-1 for the next pass over this column.
            lb1[col] <= lb0[col];
            lb0[col] <= in_pixel;

            if (col == PW-1) begin col <= 0; row <= row + 1; end
            else                  col <= col + 1;
        end else begin
            out_valid <= 1'b0;
        end
    end
endmodule

`default_nettype wire
