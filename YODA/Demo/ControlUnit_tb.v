// =========================================================================
// EEE4120F HPES Project: VSS (Vectorized Signal Star)
// Group 7 — Kloppers (KLPJOA002), Hillman (HLLALE010)
// =========================================================================
//
// File        : ControlUnit_tb.v
// Description : Testbench for the extended Main Control Unit. Verifies:
//                 - All P4 opcodes produce their original control signals
//                   (vector_mode must be 0 for these — no regression)
//                 - The new OP_VECTOR (4'b1110) asserts vector_mode=1,
//                   ALUOp=11, reg_dst=1, reg_write=1, and leaves memory /
//                   branch / jump quiet.
// =========================================================================

`timescale 1ns / 1ps
`include "Parameter.v"

module ControlUnit_tb;

    reg  [3:0] opcode;
    wire [1:0] alu_op;
    wire       jump, beq, bne, mem_read, mem_write;
    wire       alu_src, reg_dst, mem_to_reg, reg_write;
    wire       vector_mode;

    integer pass = 0, fail = 0, tid = 0;

    ControlUnit dut (
        .opcode      (opcode),
        .alu_op      (alu_op),
        .jump        (jump),
        .beq         (beq),
        .bne         (bne),
        .mem_read    (mem_read),
        .mem_write   (mem_write),
        .alu_src     (alu_src),
        .reg_dst     (reg_dst),
        .mem_to_reg  (mem_to_reg),
        .reg_write   (reg_write),
        .vector_mode (vector_mode)
    );

    // Bundle: {reg_dst, alu_src, mem_to_reg, reg_write, mem_read, mem_write,
    //          beq, bne, alu_op[1:0], jump, vector_mode}  -> 12 bits
    task expect_bundle;
        input [127:0] label;
        input [11:0]  exp;
        reg   [11:0]  got;
        begin
            tid = tid + 1;
            #1;
            got = {reg_dst, alu_src, mem_to_reg, reg_write, mem_read, mem_write,
                   beq, bne, alu_op, jump, vector_mode};
            if (got === exp) begin
                $display("  [%0d] PASS  %-10s  op=%b  ctrl=%b", tid, label, opcode, got);
                pass = pass + 1;
            end else begin
                $display("  [%0d] FAIL  %-10s  op=%b  ctrl=%b  expected=%b",
                          tid, label, opcode, got, exp);
                fail = fail + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("ControlUnit_tb.vcd");
        $dumpvars(0, ControlUnit_tb);

        $display("=================================================================");
        $display(" VSS ControlUnit Golden Measure       Group 7");
        $display("=================================================================");
        $display("\n  Bundle bits = {reg_dst, alu_src, mem_to_reg, reg_write,");
        $display("                 mem_read, mem_write, beq, bne, alu_op[1:0],");
        $display("                 jump, vector_mode}\n");

        // Baseline opcodes
        opcode = 4'b0000; expect_bundle("LD",   12'b011110_00_10_00);
        opcode = 4'b0001; expect_bundle("ST",   12'b010001_00_10_00);
        opcode = 4'b0010; expect_bundle("ADD",  12'b100100_00_00_00);
        opcode = 4'b1001; expect_bundle("SLT",  12'b100100_00_00_00);
        opcode = 4'b1010; expect_bundle("RSVD", 12'b000000_00_00_00);
        opcode = 4'b1011; expect_bundle("BEQ",  12'b000000_10_01_00);
        opcode = 4'b1100; expect_bundle("BNE",  12'b000000_01_01_00);
        opcode = 4'b1101; expect_bundle("JMP",  12'b000000_00_00_10);

        // VSS extension — vector_mode must be 1 for ONLY this opcode
        opcode = 4'b1110; expect_bundle("VECTOR", 12'b100100_00_11_01);

        $display("\n=================================================================");
        $display(" Results: %0d PASS, %0d FAIL  (out of %0d tests)", pass, fail, tid);
        $display("=================================================================");
        $finish;
    end

endmodule
