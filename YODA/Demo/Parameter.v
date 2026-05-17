// =========================================================================
// Parameter.v - Shared constants for VSS / StarCore-1
// =========================================================================
// This is a lightweight stub. If your project already has a richer
// Parameter.v, replace this file with that one.

`ifndef _PARAMETER_V_
`define _PARAMETER_V_

// ALUOp encodings used by ControlUnit and ALU_Control
`define ALUOP_RTYPE  2'b00
`define ALUOP_BRANCH 2'b01
`define ALUOP_MEM    2'b10
`define ALUOP_VECTOR 2'b11   // NEW for VSS

// Opcodes (instr[15:12])
`define OP_LD     4'b0000
`define OP_ST     4'b0001
`define OP_ADD    4'b0010
`define OP_SUB    4'b0011
`define OP_INV    4'b0100
`define OP_SHL    4'b0101
`define OP_SHR    4'b0110
`define OP_AND    4'b0111
`define OP_OR     4'b1000
`define OP_SLT    4'b1001
`define OP_RSVD   4'b1010
`define OP_BEQ    4'b1011
`define OP_BNE    4'b1100
`define OP_JMP    4'b1101
`define OP_VECTOR 4'b1110   // NEW for VSS

`endif
