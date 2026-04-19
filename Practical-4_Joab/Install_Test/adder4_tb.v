`timescale 1 ns / 1 ps

module adder4_tb;
    reg [3:0] a,b ;
    reg cin;
    wire [3:0] sum;
    wire cout;

    integer fail_count;
    initial fail_count = 0;

    adder4 uut (.a(a), .b(b), .cin(cin), .cout(cout), .sum(sum));

    initial begin
        $dumpfile("adder4.vcd");
        $dumpvars(0, adder4_tb);
    end

    initial begin
        $display("--- adder4 testbench ---");

        a= 4'd3; b=4'd2; cin = 1'b0; #10;
        if ({cout,sum} !== 5'd5) begin
            $display("FAIL T1: 3+2+0 got=%b%b exp=00101", cout, sum);
            fail_count = fail_count +1;
        end else $display("Pass T1: 3+2+0=%d", {cout,sum});
    end
endmodule