// =========================================================================
// EEE4120F HPES Project: VSS (Vectorized Signal Star)
// =========================================================================
//
// GROUP NUMBER: 7
//
// MEMBERS:
//   - Member 1 Joab Gray Kloppers, KLPJOA002
//   - Member 2 Alex Hillman, HLLALE010

// File        : ALU_Control.v
// Description : ALU Control Unit, extended for VSS.
//
//               Maps ALUOp (from the Main Control Unit), the 4-bit opcode,
//               and (for vector instructions) the instruction funct field
//               to the 3-bit ALU_Cnt that drives the ALU's operation select.
//
//               ALUOp encodings:
//                 2'b10 (ALUOP_MEM)    -> always ADD (LD/ST address)
//                 2'b01 (ALUOP_BRANCH) -> always SUB (branch compare)
//                 2'b00 (ALUOP_RTYPE)  -> decoded from opcode
//                 2'b11 (ALUOP_VECTOR) -> NEW (VSS): pass funct[2:0] through
//
//               This is a purely combinational module.
//
// Task 5 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module ALU_Control (
    input  [1:0] ALUOp,         // From ControlUnit:
                                //   2'b10 = memory access (always ADD for address)
                                //   2'b01 = branch      (always SUB for comparison)
                                //   2'b00 = R-type      (decode from opcode)
    input  [3:0] Opcode,        // Instruction opcode field [15:12]
    input  [2:0] funct,         // NEW: Instruction funct field [2:0]
    output reg [2:0] ALU_Cnt    // To ALU alu_control input
);

    // -------------------------------------------------------------------------
    // TODO: Concatenate ALUOp and Opcode into a single 6-bit control word so
    //       you can use a casex statement with don't-care bits.
    //
    //       wire [5:0] control_in;
    //       assign control_in = {ALUOp, Opcode};
    //
    //       The casex truth table (from Section 3.4 of the manual):
    //
    //       control_in | ALU_Cnt | Operation   | Instruction
    //       -----------+---------+-------------+------------------
    //       6'b10xxxx  |  3'b000 | ADD         | LD, ST
    //       6'b01xxxx  |  3'b001 | SUB         | BEQ, BNE
    //       6'b000010  |  3'b000 | ADD         | ADD
    //       6'b000011  |  3'b001 | SUB         | SUB
    //       6'b000100  |  3'b010 | INV (NOT)   | INV
    //       6'b000101  |  3'b011 | SHL         | SHL
    //       6'b000110  |  3'b100 | SHR         | SHR
    //       6'b000111  |  3'b101 | AND         | AND
    //       6'b001000  |  3'b110 | OR          | OR
    //       6'b001001  |  3'b111 | SLT         | SLT
    //       default    |  3'b000 | ADD (safe)  | reserved / undefined
    //
    //       Implement using:
    //           always @(*) begin
    //               casex (control_in)
    //                   6'b10xxxx : ALU_Cnt = 3'b000;
    //                   ...
    //                   default   : ALU_Cnt = 3'b000;
    //               endcase
    //           end
    //
    //       IMPORTANT: The 'x' in casex patterns matches any logic value
    //       (0, 1, X, or Z). This correctly encodes don't-care bits for the
    //       opcode field when ALUOp selects memory or branch mode.
    // -------------------------------------------------------------------------
    wire [5:0] control_in;
    assign control_in = {ALUOp,Opcode};

    always @(*) begin
        casex (control_in)

            // VSS extension: vector mode passes the instruction's [2:0] funct
            // field straight through. Because VFUNC reuses the ALU's 3-bit
            // operation encoding, the ALU sees the right alu_control value
            // (combined with vector_mode = 1 from the ControlUnit) directly.
            6'b11xxxx : ALU_Cnt = funct;
            
            // Practical 4 unchanged code
            6'b10xxxx : ALU_Cnt = 3'b000;
            6'b01xxxx : ALU_Cnt = 3'b001;
            6'b000010 : ALU_Cnt = 3'b000;
            6'b000011 : ALU_Cnt = 3'b001;
            6'b000100 : ALU_Cnt = 3'b010;
            6'b000101 : ALU_Cnt = 3'b011;
            6'b000110 : ALU_Cnt = 3'b100;
            6'b000111 : ALU_Cnt = 3'b101;
            6'b001000 : ALU_Cnt = 3'b110;
            6'b001001 : ALU_Cnt = 3'b111;
            default : ALU_Cnt = 3'b000;
        endcase
    end


endmodule
