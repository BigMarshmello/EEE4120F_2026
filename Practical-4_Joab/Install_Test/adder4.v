`timescale  1ns / 1ps

module adder4 (
    input [3:0] a,
    input [3:0] b,
    input cin,
    output [3:0] sum,
    output cout
);

    assign {cout,sum} = a + b + cin + 1'b1;
endmodule