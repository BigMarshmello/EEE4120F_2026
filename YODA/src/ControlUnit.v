// =========================================================================
// EEE4120F HPES Project: VSS (Vectorized Signal Star)
// =========================================================================
//
// GROUP NUMBER: 7
//
// MEMBERS:
//   - Member 1 Joab Gray Kloppers, KLPJOA002
//   - Member 2 Alex Hillman, HLLALE010

// File        : ControlUnit.v
// Description : Main Control Unit, extended for VSS.
//               Decodes the 4-bit opcode and asserts the full set of
//               control signals for the Datapath. Adds a new opcode case
//               for OP_VECTOR (4'b1110) and a new output vector_mode that
//               drives the ALU into SIMD-lite lane-split operation.
// Task 6 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module ControlUnit (
    input  [3:0] opcode,        // Instruction opcode [15:12] from Datapath

    // ALU control
    output reg [1:0] alu_op,    // Passed to ALU_Control: 10=mem, 01=branch, 00=R-type

    // PC control
    output reg       jump,      // Assert to select the jump target PC
    output reg       beq,       // Assert to enable branch-on-equal
    output reg       bne,       // Assert to enable branch-on-not-equal

    // Memory control
    output reg       mem_read,  // Assert to enable data memory read output
    output reg       mem_write, // Assert to write data memory on posedge clk

    // Datapath multiplexer selects
    output reg       alu_src,   // 0 = RS2 register value; 1 = sign-extended immediate
    output reg       reg_dst,   // 0 = instr[8:6] (I-type WS); 1 = instr[5:3] (R-type WS)
    output reg       mem_to_reg,// 0 = ALU result; 1 = data memory read data (for LD)
    output reg       reg_write,  // Assert to write the register file on posedge clk

    // VSS extension
    output reg       vector_mode // NEW: 1 = ALU operates in 2x8-bit SIMD-lite mode
);

    // -------------------------------------------------------------------------
    // Control signal truth table (extended for VSS):
    //
    // Opcode | Instr     | RegDst ALUSrc MemToReg RegWrite MemRd MemWr Branch ALUOp Jump VecMode
    // -------+-----------+-----------------------------------------------------------------------
    // 0000   | LD        |   0      1       1        1       1     0     0      10    0     0
    // 0001   | ST        |   0      1       0        0       0     1     0      10    0     0
    // 0010-  | R-type    |   1      0       0        1       0     0     0      00    0     0
    // 1001   | (ADD-SLT) |
    // 1010   | Reserved  |   0      0       0        0       0     0     0      00    0     0
    // 1011   | BEQ       |   0      0       0        0       0     0     1      01    0     0
    // 1100   | BNE       |   0      0       0        0       0     0     1      01    0     0
    // 1101   | JMP       |   0      0       0        0       0     0     0      00    1     0
    // 1110   | VECTOR    |   1      0       0        1       0     0     0      11    0     1   last col is new
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // TODO: Implement the control unit using always @(*) and a case statement.
    //
    //       STEP 1 — Assign safe defaults to ALL outputs at the top of the
    //       always block BEFORE the case statement. This prevents accidental
    //       latches when an opcode branch does not assign every signal:
    //
    //           always @(*) begin
    //               // Safe defaults: no writes, no branches, no jumps
    //               reg_dst   = 1'b0;
    //               alu_src   = 1'b0;
    //               mem_to_reg= 1'b0;
    //               reg_write = 1'b0;
    //               mem_read  = 1'b0;
    //               mem_write = 1'b0;
    //               beq       = 1'b0;
    //               bne       = 1'b0;
    //               alu_op    = 2'b00;
    //               jump      = 1'b0;
    //
    //               case (opcode)
    //                   4'b0000: begin  // LD
    //                       reg_dst   = 1'b0;
    //                       alu_src   = 1'b1;
    //                       mem_to_reg= 1'b1;
    //                       reg_write = 1'b1;
    //                       mem_read  = 1'b1;
    //                       alu_op    = 2'b10;
    //                   end
    //
    //                   4'b0001: begin  // ST
    //                       ...
    //                   end
    //
    //                   // R-type instructions share identical control signals.
    //                   // List each opcode individually OR use a Verilog 2001
    //                   // comma-separated case item:
    //                   // 4'b0010, 4'b0011, 4'b0100, 4'b0101,
    //                   // 4'b0110, 4'b0111, 4'b1000, 4'b1001: begin ...
    //
    //                   4'b1010: begin  // Reserved — must be a no-operation
    //                       // All outputs remain at safe defaults.
    //                       // No register or memory side-effects.
    //                   end
    //
    //                   4'b1011: begin  // BEQ
    //                       beq    = 1'b1;
    //                       alu_op = 2'b01;
    //                   end
    //
    //                   4'b1100: begin  // BNE
    //                       ...
    //                   end
    //
    //                   4'b1101: begin  // JMP
    //                       ...
    //                   end
    //
    //                   default: begin
    //                       // Safe defaults already set above.
    //                   end
    //               endcase
    //           end
    // -------------------------------------------------------------------------

    always @(*) begin
        // Safe defaults: no writes, no branches, no jumps
        reg_dst   = 1'b0;
        alu_src   = 1'b0;
        mem_to_reg= 1'b0;
        reg_write = 1'b0;
        mem_read  = 1'b0;
        mem_write = 1'b0;
        beq       = 1'b0;
        bne       = 1'b0;
        alu_op    = 2'b00;
        jump      = 1'b0;
        vector_mode = 1'b0; // NEW

        case (opcode)
            4'b0000: begin  // LD
                reg_dst   = 1'b0;
                alu_src   = 1'b1;
                mem_to_reg= 1'b1;
                reg_write = 1'b1;
                mem_read  = 1'b1;
                alu_op    = 2'b10;
            end

            4'b0001: begin  // ST
                alu_op = 2'b10;
                mem_write = 1'b1;
                alu_src = 1'b1;
            end

            // R-type instructions share identical control signals.
            // List each opcode individually OR use a Verilog 2001
            // comma-separated case item:
            // 4'b0010, 4'b0011, 4'b0100, 4'b0101,
            // 4'b0110, 4'b0111, 4'b1000, 4'b1001: begin ...

            4'b0010, 4'b0011, 4'b0100, 4'b0101, 4'b0110, 4'b0111, 4'b1000, 4'b1001: begin
                reg_dst = 1'b1;
                reg_write = 1'b1;
            end

            4'b1010: begin  // Reserved — must be a no-operation
                // All outputs remain at safe defaults.
                // No register or memory side-effects.
            end

            4'b1011: begin  // BEQ
                beq    = 1'b1;
                alu_op = 2'b01;
            end

            4'b1100: begin  // BNE
                alu_op = 2'b01;
                bne = 1'b1;
            end

            4'b1101: begin  // JMP
                jump = 1'b1;
            end

            // VSS extension: VECTOR instruction (opcode 1110).
            // Behaves like an R-type for register addressing (WS in [5:3],
            // operands from RS1/RS2, no memory access, writes back ALU result),
            // but sets ALUOp = 2'b11 so ALU_Control passes the funct field
            // through, and asserts vector_mode so the ALU operates in
            // SIMD-lite 2 x 8-bit lane-split mode.
            4'b1110: begin  // VECTOR
                reg_dst     = 1'b1;       // R-type WS encoding
                alu_src     = 1'b0;       // ALU operand B = RS2 register value
                mem_to_reg  = 1'b0;       // Write back ALU result
                reg_write   = 1'b1;       // Enable register write
                alu_op      = 2'b11;      // ALUOP_VECTOR -> funct -> ALU_Cnt
                vector_mode = 1'b1;       // SIMD-lite ALU mode
            end
 
            default: begin
                // Safe defaults already set above.
            end
        endcase
    end


endmodule
