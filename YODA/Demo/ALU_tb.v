// =========================================================================
// EEE4120F HPES Project: VSS (Vectorized Signal Star)
// Group 7 — Kloppers (KLPJOA002), Hillman (HLLALE010)
// =========================================================================
//
// File        : ALU_tb.v
// Description : Testbench for the dual-mode ALU. Drives the same operand
//               patterns through scalar (16-bit) and vector (2x8-bit lane)
//               modes and verifies the lane split correctness against a
//               software golden model expressed inline as `expect_*` tasks.
//
//               This testbench produces:
//                 1. A console PASS/FAIL log used as the gold-standard output.
//                 2. A VCD waveform file (ALU_tb.vcd) for GTKWave inspection,
//                    where the broken carry between bits 7 and 8 is visible.
//
// Run with:  iverilog -o alu_tb.vvp ALU_tb.v ../ALU.v
//            vvp alu_tb.vvp
//            gtkwave ALU_tb.vcd
// =========================================================================

`timescale 1ns / 1ps
`include "Parameter.v"

module ALU_tb;

    // DUT ports
    reg          vector_mode;
    reg  [15:0]  a, b;
    reg  [2:0]   alu_control;
    wire [15:0]  result;
    wire         zero;

    // Counters for the gold-standard summary
    integer pass_count = 0;
    integer fail_count = 0;
    integer test_id    = 0;

    // -------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------
    ALU dut (
        .vector_mode (vector_mode),
        .a           (a),
        .b           (b),
        .alu_control (alu_control),
        .result      (result),
        .zero        (zero)
    );

    // -------------------------------------------------------------------
    // Golden-measure check task
    // -------------------------------------------------------------------
    task check;
        input [127:0] label;        // up to 16 chars label
        input [15:0]  expected;
        begin
            test_id = test_id + 1;
            #1; // allow combinational settling for prints
            if (result === expected) begin
                $display("  [%0d] PASS  %-12s  mode=%b op=%b  a=%h b=%h  result=%h  (expected %h)",
                          test_id, label, vector_mode, alu_control, a, b, result, expected);
                pass_count = pass_count + 1;
            end else begin
                $display("  [%0d] FAIL  %-12s  mode=%b op=%b  a=%h b=%h  result=%h  EXPECTED %h",
                          test_id, label, vector_mode, alu_control, a, b, result, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // -------------------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------------------
    initial begin
        $dumpfile("ALU_tb.vcd");
        $dumpvars(0, ALU_tb);

        $display("=================================================================");
        $display(" VSS ALU Golden-Measure Test          Group 7 (KLPJOA002, HLLALE010)");
        $display("=================================================================");

        // ============== SCALAR MODE (P4 baseline behaviour) ==============
        vector_mode = 1'b0;
        $display("\n--- Scalar mode (16-bit, baseline) ---");

        // ADD: 0x00FF + 0x0001 = 0x0100  (carry from bit 7 into bit 8 must propagate)
        a = 16'h00FF; b = 16'h0001; alu_control = 3'b000; #2;
        check("ADD-carry", 16'h0100);

        // SUB: 0x0100 - 0x0001 = 0x00FF  (borrow propagates the same way)
        a = 16'h0100; b = 16'h0001; alu_control = 3'b001; #2;
        check("SUB-borrow", 16'h00FF);

        // SHL by 4: 0x0FFF << 4 = 0xFFF0
        a = 16'h0FFF; b = 16'h0004; alu_control = 3'b011; #2;
        check("SHL4", 16'hFFF0);

        // SLT: 5 < 10  => 1
        a = 16'h0005; b = 16'h000A; alu_control = 3'b111; #2;
        check("SLT-true", 16'h0001);

        // ============== VECTOR MODE (SIMD-lite 2x8-bit lanes) =============
        vector_mode = 1'b1;
        $display("\n--- Vector mode (2 x 8-bit lanes, VSS) ---");

        // VADD lane-independence: hi=0x0A+0x20=0x2A, lo=0x05+0x07=0x0C
        a = 16'h0A05; b = 16'h2007; alu_control = 3'b000; #2;
        check("VADD-basic", 16'h2A0C);

        // VADD broken carry: low lane overflow (0xFF+0x01=0x00 mod 256)
        //                    MUST NOT corrupt the high lane (0x10 stays 0x10)
        // This is the key VSS demonstration on the waveform.
        a = 16'h10FF; b = 16'h0001; alu_control = 3'b000; #2;
        check("VADD-nocarry", 16'h1000);

        // VSUB lane-independence: hi=0x80-0x40=0x40, lo=0x10-0x02=0x0E
        a = 16'h8010; b = 16'h4002; alu_control = 3'b001; #2;
        check("VSUB-basic", 16'h400E);

        // VSUB low-lane wrap: lo=0x00-0x01=0xFF, hi=0x20-0x10=0x10
        // Demonstrates borrow does NOT leak from low lane into high lane.
        a = 16'h2000; b = 16'h1001; alu_control = 3'b001; #2;
        check("VSUB-noborrow", 16'h10FF);

        // VAND lane-independence
        a = 16'hF0F0; b = 16'h0FFF; alu_control = 3'b101; #2;
        check("VAND", 16'h00F0);

        // VOR
        a = 16'hF000; b = 16'h00F0; alu_control = 3'b110; #2;
        check("VOR", 16'hF0F0);

        // VSLT both lanes: hi 0x05<0x10 => 0x01, lo 0xFF<0x01 => 0x00
        a = 16'h05FF; b = 16'h1001; alu_control = 3'b111; #2;
        check("VSLT-mix", 16'h0100);

        // VSHL per-lane: hi=0x01<<2=0x04, lo=0x03<<1=0x06
        // Note: ALU uses b[11:8] for hi-lane shift, b[3:0] for lo-lane shift
        a = 16'h0103; b = 16'h0201; alu_control = 3'b011; #2;
        check("VSHL", 16'h0406);

        // VSHR per-lane: hi=0x80>>2=0x20, lo=0x40>>1=0x20
        a = 16'h8040; b = 16'h0201; alu_control = 3'b100; #2;
        check("VSHR", 16'h2020);

        // VINV (operates on full 16-bit a; b ignored)
        a = 16'hAA55; b = 16'h0000; alu_control = 3'b010; #2;
        check("VINV", 16'h55AA);

        // ============== Summary ==============
        $display("\n=================================================================");
        $display(" Results: %0d PASS, %0d FAIL  (out of %0d tests)",
                 pass_count, fail_count, test_id);
        if (fail_count == 0)
            $display(" *** GOLDEN MEASURE MATCH — VSS ALU verified ***");
        else
            $display(" *** %0d MISMATCH(ES) — see PASS/FAIL log above ***", fail_count);
        $display("=================================================================");

        $finish;
    end

endmodule
