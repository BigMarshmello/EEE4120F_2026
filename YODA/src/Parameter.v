// =========================================================================
// EEE4120F HPES Project: VSS (Vectorized Signal Star)
// =========================================================================
// GROUP NUMBER:7
//
// MEMBERS:
//   - Member 1 Joab Gray Kloppers, KLPJOA002
//   - Member 2 Alex Hillman, HLLALE010

// File        : Parameter.v
// Description : Shared compile-time parameters used across all modules.
//               Extends the Practical 4 baseline with named constants for
//               the StarCore-1 ISA opcodes, ALUOp encodings, ALU operation
//               encodings, and the VSS vector funct field.
//               Include this file at the top of every .v file:
//                   `include "../src/Parameter.v"
// =============================================================================

`ifndef PARAMETER_H_
`define PARAMETER_H_

// ---------------------------------------------------------------------------
// Memory dimensions
// ---------------------------------------------------------------------------
`define COL     16          // Data/instruction word width (bits)
`define ROW_I   16          // Instruction memory depth (words, 16 x 16-bit)
`define ROW_D    8          // Data memory depth (words,  8 x 16-bit)

// ---------------------------------------------------------------------------
// Simulation control
// Increase SIM_TIME if your test program needs more clock cycles to complete.
// At 10 ns per clock (100 MHz) each #10 is one half-period; 320 ns = 16 cycles.
// ---------------------------------------------------------------------------
`define SIM_TIME  #130      // Total simulation time for integration testbench

// ---------------------------------------------------------------------------
// Output file for data-memory dump (used in DataMemory.v $fmonitor)
// ---------------------------------------------------------------------------
`define DMEM_LOG  "./waves/dmem_log.txt"

// ===========================================================================
// StarCore-1 ISA opcode encodings (instruction bits [15:12])
// ---------------------------------------------------------------------------
// Practical 4 baseline:
`define OP_LD       4'b0000  // Load:      LD  WS, offset(RS1)
`define OP_ST       4'b0001  // Store:     ST  RS2, offset(RS1)
`define OP_ADD      4'b0010  // R-type:    ADD WS, RS1, RS2
`define OP_SUB      4'b0011  // R-type:    SUB WS, RS1, RS2
`define OP_INV      4'b0100  // R-type:    INV WS, RS1        (b ignored)
`define OP_SHL      4'b0101  // R-type:    SHL WS, RS1, RS2
`define OP_SHR      4'b0110  // R-type:    SHR WS, RS1, RS2
`define OP_AND      4'b0111  // R-type:    AND WS, RS1, RS2
`define OP_OR       4'b1000  // R-type:    OR  WS, RS1, RS2
`define OP_SLT      4'b1001  // R-type:    SLT WS, RS1, RS2
`define OP_RSVD     4'b1010  // Reserved   (co-processor; no-op in baseline)
`define OP_BEQ      4'b1011  // Branch:    BEQ RS1, RS2, offset
`define OP_BNE      4'b1100  // Branch:    BNE RS1, RS2, offset
`define OP_JMP      4'b1101  // Jump:      JMP offset
 
// VSS extension (Specialized Path):
`define OP_VECTOR   4'b1110  // SIMD-lite vector instructions; sub-op in [2:0]
// 4'b1111 currently unallocated.

// ===========================================================================
// ALUOp encoding (2-bit signal from ControlUnit -> ALU_Control)
// ---------------------------------------------------------------------------
`define ALUOP_RTYPE   2'b00  // R-type: decode operation from opcode
`define ALUOP_BRANCH  2'b01  // Always SUB (for branch comparison)
`define ALUOP_MEM     2'b10  // Always ADD (for LD/ST address calculation)
`define ALUOP_VECTOR  2'b11  // VSS: pass instruction funct[2:0] as ALU_Cnt
 
// ===========================================================================
// ALU operation encoding (3-bit alu_control input to the ALU)
// Used for both scalar and vector modes; vector_mode selects which mode.
// ---------------------------------------------------------------------------
`define ALU_ADD  3'b000
`define ALU_SUB  3'b001
`define ALU_INV  3'b010
`define ALU_SHL  3'b011
`define ALU_SHR  3'b100
`define ALU_AND  3'b101
`define ALU_OR   3'b110
`define ALU_SLT  3'b111
 
 // ===========================================================================
// VFUNC encoding (instruction[2:0] when opcode == OP_VECTOR)
// Reuses the ALU's existing 3-bit operation encoding directly so the
// ALU_Control unit can pass it through unchanged when ALUOp = ALUOP_VECTOR.
// ---------------------------------------------------------------------------
`define VF_VADD  3'b000
`define VF_VSUB  3'b001
`define VF_VINV  3'b010
`define VF_VSHL  3'b011
`define VF_VSHR  3'b100
`define VF_VAND  3'b101
`define VF_VOR   3'b110
`define VF_VSLT  3'b111
 
`endif  // PARAMETER_H_