// =============================================================================
// EEE4120F HPES Project -- VSS (Vectorized Signal Star)
// File        : ALU_tb.v
// Description : Testbench for the modified ALU module.
//               Section 1 preserves the original Practical 4 scalar tests
//               (with vector_mode = 0) to confirm the baseline behaviour is
//               unchanged. Section 2 adds new tests for vector mode
//               (vector_mode = 1) which exercise the SIMD-lite 2 x 8-bit
//               lane-split operations and verify that no carry / borrow
//               propagates across the lane boundary (between bit 7 and bit 8).
//
// Run:
//   iverilog -Wall -I ../src -o ../build/alu_sim ../src/ALU.v ALU_tb.v
//   cd ../test && ../build/alu_sim
//   gtkwave ../waves/alu_tb.vcd &
// =============================================================================
 
`timescale 1ns / 1ps
`include "../src/Parameter.v"
 
module ALU_tb;
 
    // -------------------------------------------------------------------------
    // DUT port connections
    // Inputs to the DUT are declared as reg (so the testbench can drive them).
    // Outputs from the DUT are declared as wire (driven by the DUT).
    // -------------------------------------------------------------------------
    reg  [15:0] a;
    reg  [15:0] b;
    reg  [ 2:0] alu_control;
    reg         vector_mode;     // NEW: 0 = scalar, 1 = SIMD-lite 2 x 8-bit
    wire [15:0] result;
    wire        zero;
 
    // -------------------------------------------------------------------------
    // DUT instantiation -- named port connections
    // -------------------------------------------------------------------------
    ALU uut (
        .a           (a),
        .b           (b),
        .alu_control (alu_control),
        .vector_mode (vector_mode),
        .result      (result),
        .zero        (zero)
    );
 
    // -------------------------------------------------------------------------
    // Waveform dump -- always include this block
    // -------------------------------------------------------------------------
    initial begin
        $dumpfile("../waves/alu_tb.vcd");
        $dumpvars(0, ALU_tb);
    end
 
    // -------------------------------------------------------------------------
    // Failure counter
    // -------------------------------------------------------------------------
    integer fail_count;
    integer test_id;
 
    initial begin
        fail_count = 0;
        test_id    = 1;
    end
 
    // -------------------------------------------------------------------------
    // Reusable check task
    // Compares 'got' against 'expected' and prints PASS or FAIL.
    // Increments fail_count on mismatch.
    // -------------------------------------------------------------------------
    task check_result;
        input [15:0] got;
        input [15:0] expected;
        input [63:0] id;        // test number for display
        begin
            if (got !== expected) begin
                $display("FAIL [T%0d]: result = %0d (0x%h), expected = %0d (0x%h)",
                         id, got, got, expected, expected);
                fail_count = fail_count + 1;
            end else begin
                $display("PASS [T%0d]: result = %0d (0x%h)", id, got, got);
            end
        end
    endtask
 
    task check_zero;
        input got;
        input expected;
        input [63:0] id;
        begin
            if (got !== expected) begin
                $display("FAIL [T%0d] zero flag: got = %b, expected = %b", id, got, expected);
                fail_count = fail_count + 1;
            end else begin
                $display("PASS [T%0d] zero flag = %b", id, got);
            end
        end
    endtask
 
    // =========================================================================
    // STIMULUS AND CHECKING
    // =========================================================================
    initial begin
        $display("=== ALU Testbench (VSS Extension) ===");
 
        // ---------------------------------------------------------------------
        // SECTION 1 -- Scalar regression (Practical 4 baseline behaviour)
        // vector_mode = 0 forces the ALU into its original scalar 16-bit mode.
        // ---------------------------------------------------------------------
        $display("");
        $display("######## SECTION 1: SCALAR REGRESSION (vector_mode = 0) ########");
        vector_mode = 1'b0;
 
        $display("--- ADD (alu_control = 3'b000) ---");
 
        a = 16'd10; b = 16'd5; alu_control = 3'b000; #10;
        check_result(result, 16'd15, test_id); test_id = test_id +1;
 
        a = 16'hFFFF; b = 16'd1; alu_control = 3'b000; #10;
        check_result(result, 16'd0, test_id); test_id = test_id +1;
 
        a = 16'd0; b = 16'd0; alu_control = 3'b000; #10;
        check_result(result, 16'd0, test_id); test_id = test_id +1;
 
 
        $display("--- SUB (alu_control = 3'b001) ---");
 
        a = 16'd10; b = 16'd5; alu_control = 3'b001; #10;
        check_result(result, 16'd5, test_id);
        check_zero(zero,1'b0,test_id);test_id = test_id +1;
 
        a = 16'd7; b = 16'd7; alu_control = 3'b001; #10;
        check_result(result, 16'd0, test_id);
        check_zero(zero,1'b1,test_id);test_id = test_id +1;
 
        a = 16'd5; b = 16'd10; alu_control = 3'b001; #10;
        check_result(result, 16'd65531, test_id); test_id = test_id +1;
 
 
        $display("--- INV / NOT (alu_control = 3'b010) ---");
 
        a = 16'h0000; alu_control = 3'b010; #10;
        check_result(result,16'hFFFF,test_id); test_id = test_id +1;
 
        a = 16'hA5A5; alu_control = 3'b010; #10;
        check_result(result,16'h5A5A,test_id); test_id = test_id +1;
 
 
        $display("--- SHL (alu_control = 3'b011) ---");
 
        a = 16'h0001; b = 16'd4; alu_control = 3'b011; #10;
        check_result(result, 16'h0010, test_id); test_id = test_id +1;
 
        a = 16'h0003; b = 16'd2; alu_control = 3'b011; #10;
        check_result(result, 16'h000c, test_id); test_id = test_id +1;
 
 
        $display("--- SHR (alu_control = 3'b100) ---");
 
        a = 16'h0080; b = 16'd4; alu_control = 3'b100; #10;
        check_result(result, 16'h0008, test_id); test_id = test_id +1;
 
        a = 16'h0001; b = 16'd1; alu_control = 3'b100; #10;
        check_result(result, 16'h0000, test_id); test_id = test_id +1;
 
 
        $display("--- AND (alu_control = 3'b101) ---");
 
        a = 16'hFFFF; b = 16'h0F0F; alu_control = 3'b101; #10;
        check_result(result, 16'h0F0F, test_id); test_id = test_id + 1;
 
        a = 16'hAAAA; b = 16'h5555; alu_control = 3'b101; #10;
        check_result(result, 16'h0000, test_id); test_id = test_id + 1;
 
 
        $display("--- OR (alu_control = 3'b110) ---");
 
        a = 16'hF0F0; b = 16'h0F0F; alu_control = 3'b110; #10;
        check_result(result, 16'hFFFF, test_id); test_id = test_id + 1;
 
        a = 16'hAAAA; b = 16'h5555; alu_control = 3'b110; #10;
        check_result(result, 16'hFFFF, test_id); test_id = test_id + 1;
 
 
        $display("--- SLT (alu_control = 3'b111) ---");
 
        a = 16'd5; b = 16'd10; alu_control = 3'b111; #10;
        check_result(result, 16'd1, test_id); test_id = test_id + 1;
 
        a = 16'd10; b = 16'd10; alu_control = 3'b111; #10;
        check_result(result, 16'd0, test_id); test_id = test_id + 1;
 
        a = 16'd15; b = 16'd3; alu_control = 3'b111; #10;
        check_result(result, 16'd0, test_id); test_id = test_id + 1;
 
 
        $display("--- Zero flag edge cases ---");
 
        a = 16'd5; b = 16'd5; alu_control = 3'b001; #10;
        check_zero(zero, 16'd1, test_id); test_id = test_id + 1;
 
        a = 16'd10; b = 16'd5; alu_control = 3'b001; #10;
        check_zero(zero, 16'd0, test_id); test_id = test_id + 1;
 
        a = 16'hFFFF; alu_control = 3'b010; #10;
        check_zero(zero, 16'd1, test_id); test_id = test_id + 1;
 
 
        // ---------------------------------------------------------------------
        // SECTION 2 -- Vector mode (SIMD-lite 2 x 8-bit lanes)
        // vector_mode = 1: operands interpreted as packed [hi: a[15:8] | lo: a[7:0]],
        // operations applied per-lane with NO carry/borrow across bit 7 -> 8.
        // ---------------------------------------------------------------------
        $display("");
        $display("######## SECTION 2: VECTOR MODE (vector_mode = 1) ########");
        vector_mode = 1'b1;
 
        $display("--- VADD (alu_control = 3'b000) ---");
 
        // Basic VADD: hi 0x01+0x03 = 0x04, lo 0x02+0x04 = 0x06
        a = 16'h0102; b = 16'h0304; alu_control = 3'b000; #10;
        check_result(result, 16'h0406, test_id); test_id = test_id + 1;
 
        // KEY PROOF -- carry break at bit 8.
        // Same inputs in scalar would give 0x0100 (carry crosses lane boundary).
        // Vector: hi 0x00+0x00 = 0x00, lo 0xFF+0x01 = 0x00 (carry discarded).
        a = 16'h00FF; b = 16'h0001; alu_control = 3'b000; #10;
        check_result(result, 16'h0000, test_id);
        check_zero(zero, 1'b1, test_id); test_id = test_id + 1;
 
        // Both lanes saturate-and-wrap independently.
        a = 16'hFF01; b = 16'h0102; alu_control = 3'b000; #10;
        check_result(result, 16'h0003, test_id); test_id = test_id + 1;
 
 
        $display("--- VSUB (alu_control = 3'b001) ---");
 
        // Basic VSUB: hi 0x20-0x01=0x1F, lo 0x30-0x02=0x2E
        a = 16'h2030; b = 16'h0102; alu_control = 3'b001; #10;
        check_result(result, 16'h1F2E, test_id); test_id = test_id + 1;
 
        // KEY PROOF -- borrow break.
        // Low lane underflows (0x00 - 0x01 = 0xFF) but the high lane is untouched.
        a = 16'h0500; b = 16'h0001; alu_control = 3'b001; #10;
        check_result(result, 16'h05FF, test_id); test_id = test_id + 1;
 
 
        $display("--- VINV (alu_control = 3'b010) ---");
 
        // Bitwise -- same behaviour as scalar; included for ISA completeness.
        a = 16'h0F0F; alu_control = 3'b010; #10;
        check_result(result, 16'hF0F0, test_id); test_id = test_id + 1;
 
 
        $display("--- VSHL (alu_control = 3'b011) ---");
 
        // Per-lane logical left shift; shift count for each lane is the low
        // nibble of the corresponding lane of b.
        // hi: 0x01 << b[11:8]=3 = 0x08;  lo: 0x02 << b[3:0]=1 = 0x04
        a = 16'h0102; b = 16'h0301; alu_control = 3'b011; #10;
        check_result(result, 16'h0804, test_id); test_id = test_id + 1;
 
 
        $display("--- VSHR (alu_control = 3'b100) ---");
 
        // hi: 0x80 >> 1 = 0x40;  lo: 0x40 >> 2 = 0x10
        a = 16'h8040; b = 16'h0102; alu_control = 3'b100; #10;
        check_result(result, 16'h4010, test_id); test_id = test_id + 1;
 
 
        $display("--- VAND (alu_control = 3'b101) ---");
 
        // Bitwise -- same behaviour as scalar; included for ISA completeness.
        a = 16'hF0F0; b = 16'h0F0F; alu_control = 3'b101; #10;
        check_result(result, 16'h0000, test_id);
        check_zero(zero, 1'b1, test_id); test_id = test_id + 1;
 
 
        $display("--- VOR (alu_control = 3'b110) ---");
 
        // Bitwise -- same behaviour as scalar; included for ISA completeness.
        a = 16'hF0F0; b = 16'h0F0F; alu_control = 3'b110; #10;
        check_result(result, 16'hFFFF, test_id); test_id = test_id + 1;
 
 
        $display("--- VSLT (alu_control = 3'b111) ---");
 
        // Per-lane unsigned compare; each lane outputs 0x01 or 0x00.
        // Both lanes a < b -> 0x0101
        a = 16'h0203; b = 16'h0304; alu_control = 3'b111; #10;
        check_result(result, 16'h0101, test_id); test_id = test_id + 1;
 
        // Neither lane a < b -> 0x0000
        a = 16'h0305; b = 16'h0204; alu_control = 3'b111; #10;
        check_result(result, 16'h0000, test_id); test_id = test_id + 1;
 
        // Only high lane a < b -> 0x0100  (proves lanes are independent)
        a = 16'h0205; b = 16'h0304; alu_control = 3'b111; #10;
        check_result(result, 16'h0100, test_id); test_id = test_id + 1;
 
 
        // -----------------------------------------------------------------------
        // Summary
        // -----------------------------------------------------------------------
        $display("");
        if (fail_count == 0)
            $display("=== ALL %0d TESTS PASSED ===", test_id - 1);
        else
            $display("=== %0d / %0d TESTS FAILED ===", fail_count, test_id - 1);
 
        $finish;
    end
 
endmodule
 