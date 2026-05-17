// =========================================================================
// EEE4120F HPES Project: VSS (Vectorized Signal Star)
// Group 7 — Kloppers (KLPJOA002), Hillman (HLLALE010)
// =========================================================================
//
// File        : ALU_Control_tb.v
// Description : Testbench for the extended ALU_Control. Verifies that:
//                 - Baseline ALUOp 00/01/10 still decode the P4 opcode set
//                 - New ALUOp 11 (ALUOP_VECTOR) routes funct[2:0] straight
//                   through to ALU_Cnt for every vector op
// =========================================================================

`timescale 1ns / 1ps
`include "Parameter.v"

module ALU_Control_tb;

    reg  [1:0]  ALUOp;
    reg  [3:0]  Opcode;
    reg  [2:0]  funct;
    wire [2:0]  ALU_Cnt;

    integer pass = 0, fail = 0, tid = 0;

    ALU_Control dut (
        .ALUOp   (ALUOp),
        .Opcode  (Opcode),
        .funct   (funct),
        .ALU_Cnt (ALU_Cnt)
    );

    task expect;
        input [2:0] e;
        input [127:0] label;
        begin
            tid = tid + 1;
            #1;
            if (ALU_Cnt === e) begin
                $display("  [%0d] PASS  %-14s  ALUOp=%b Op=%b funct=%b -> %b",
                          tid, label, ALUOp, Opcode, funct, ALU_Cnt);
                pass = pass + 1;
            end else begin
                $display("  [%0d] FAIL  %-14s  ALUOp=%b Op=%b funct=%b -> %b (expected %b)",
                          tid, label, ALUOp, Opcode, funct, ALU_Cnt, e);
                fail = fail + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("ALU_Control_tb.vcd");
        $dumpvars(0, ALU_Control_tb);

        $display("=================================================================");
        $display(" VSS ALU_Control Golden Measure       Group 7");
        $display("=================================================================");
        funct = 3'b000;

        // --- Baseline (P4) cases ---
        $display("\n--- Baseline ALUOp decoding (unchanged from P4) ---");
        ALUOp = `ALUOP_MEM;    Opcode = 4'b0000; expect(3'b000, "LD-addr");
        ALUOp = `ALUOP_MEM;    Opcode = 4'b0001; expect(3'b000, "ST-addr");
        ALUOp = `ALUOP_BRANCH; Opcode = 4'b1011; expect(3'b001, "BEQ-cmp");
        ALUOp = `ALUOP_BRANCH; Opcode = 4'b1100; expect(3'b001, "BNE-cmp");
        ALUOp = `ALUOP_RTYPE;  Opcode = 4'b0010; expect(3'b000, "R-ADD");
        ALUOp = `ALUOP_RTYPE;  Opcode = 4'b0011; expect(3'b001, "R-SUB");
        ALUOp = `ALUOP_RTYPE;  Opcode = 4'b0100; expect(3'b010, "R-INV");
        ALUOp = `ALUOP_RTYPE;  Opcode = 4'b1001; expect(3'b111, "R-SLT");

        // --- VSS extension: funct passes through directly ---
        $display("\n--- VSS extension (ALUOp=11, funct -> ALU_Cnt) ---");
        ALUOp = `ALUOP_VECTOR; Opcode = `OP_VECTOR;
        funct = 3'b000; expect(3'b000, "VADD-funct");
        funct = 3'b001; expect(3'b001, "VSUB-funct");
        funct = 3'b010; expect(3'b010, "VINV-funct");
        funct = 3'b011; expect(3'b011, "VSHL-funct");
        funct = 3'b100; expect(3'b100, "VSHR-funct");
        funct = 3'b101; expect(3'b101, "VAND-funct");
        funct = 3'b110; expect(3'b110, "VOR-funct");
        funct = 3'b111; expect(3'b111, "VSLT-funct");

        $display("\n=================================================================");
        $display(" Results: %0d PASS, %0d FAIL  (out of %0d tests)", pass, fail, tid);
        $display("=================================================================");
        $finish;
    end

endmodule
