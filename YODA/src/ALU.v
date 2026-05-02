// =========================================================================
// Practical 4: StarCore-1 — Single-Cycle Processor in Verilog
// =========================================================================
//
// GROUP NUMBER:7
//
// MEMBERS:
//   - Member 1 Joab Gray Kloppers, KLPJOA002
//   - Member 2 Alex Hillman, HLLALE010

// File        : ALU.v
// Description : 16-bit Arithmetic and Logic Unit (ALU).
//               Implements all arithmetic and logic operations required by
//               the StarCore ISA. This is a purely combinational module —
//               it has no clock input and no internal state.
//
// Task 1 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module ALU (
    input vector_mode,          // NEW: 0 = scalar 16-bit, 1 = SIMD lane split 2x8-bit

    input  [15:0] a,            // Operand A  — connected to GPR read data 1
    input  [15:0] b,            // Operand B  — connected to ALUSrc mux output
    input  [ 2:0] alu_control,  // Operation select — driven by ALU_Control unit
    output reg [15:0] result,   // Computed result  — fed to DataMemory and write-back mux
    output         zero         // Zero flag: asserted (1) when result == 16'd0
);

    // -------------------------------------------------------------------------
    // TODO: Implement the zero flag using a continuous assignment.
    //       The zero output must be a wire driven by a single assign statement.
    //       It should be 1 when result equals 16'd0, and 0 otherwise.
    //
    //       assign zero = ...;
    // -------------------------------------------------------------------------
    assign zero = result == 16'd0;

    // -------------------------------------------------------------------------
    // TODO: Implement the ALU operations using a combinational always block.
    //
    //       Use: always @(*) begin
    //                case (alu_control)
    //                    ...
    //                endcase
    //            end
    //
    //       ALUcnt | Operation | Expression
    //       -------+-----------+------------------------------
    //       3'b000 | ADD       | result = a + b
    //       3'b001 | SUB       | result = a - b
    //       3'b010 | INV       | result = ~a   (bitwise NOT; b is ignored)
    //       3'b011 | SHL       | result = a << b[3:0]
    //       3'b100 | SHR       | result = a >> b[3:0]
    //       3'b101 | AND       | result = a & b
    //       3'b110 | OR        | result = a | b
    //       3'b111 | SLT       | result = (a < b) ? 16'd1 : 16'd0  (unsigned)
    //       default| ADD       | result = a + b   (safe fallback)
    //
    //       Hint: The SLT comparison uses unsigned arithmetic because Verilog
    //       treats reg/wire values as unsigned by default. This is correct
    //       for the StarCore ISA.
    // -------------------------------------------------------------------------
    always @(*) begin
    if (vector_mode) begin
    // SIMD-lite: 2 x 8 bit lanes
    // Note: the carry chain is broken between bit 7 and bit 8 because each slice
    // expression evaluates as its own independent 8-bit op. 
        case (alu_control)
            3'b000: begin // VADD
                result[15:8] = a[15:8] + b[15:8];
                result[7:0] = a[7:0] + b[7:0];
            end
            3'b001: begin // VSUB
                result[15:8] = a[15:8] - b[15:8];
                result[7:0] = a[7:0] - b[7:0];
            end
            3'b010: begin // VINV 
                result = ~a;
            end
            3'b011: begin // VSHL logical shift
                result[15:8] = a[15:8] << b[11:8];
                result[7:0] = a[7:0] << b[3:0];
            end
            3'b100: begin //VSHR
                result[15:8] = a[15:8] >> b[11:8];
                result[7:0] = a[7:0] >> b[3:0];
            end
            3'b101: begin // VAND
                result = a&b;
            end
            3'b110: begin // VOR
                result = a | b;
            end
            3'b111: begin //VSLT
                result[15:8] = (a[15:8] < b[15:8]) ? 8'h01 : 8'h00;
                result[7:0] = (a[7:0] < b[7:0]) ? 8'h01 : 8'h00;
            end
            default: begin // safe fallback -> VADD
                result[15:8] = a[15:8] + b[15:8];
                result[7:0] = a[7:0] + b[7:0];
            end
        endcase

    end else begin
    // 16-bit scalar from Practical 4
        case (alu_control)
            3'b000: result = a+b;
            3'b001: result = a-b;
            3'b010: result = ~a;
            3'b011: result = a << b[3:0];
            3'b100: result = a >> b[3:0];
            3'b101: result = a&b;
            3'b110: result = a | b;
            3'b111: result = (a < b) ? 16'd1 : 16'd0;
            default: result = a+b;
        endcase
    end
end



endmodule
