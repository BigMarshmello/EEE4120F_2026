// =========================================================================
// EEE4120F HPES Project: VSS (Vectorized Signal Star)
// =========================================================================
//
// GROUP NUMBER: 7
//
// MEMBERS:
//   - Member 1 Joab Gray Kloppers, KLPJOA002
//   - Member 2 Alex Hillman, HLLALE010

// File        : StarCore1.v
// Description : Top-level StarCore-1 processor module, extended for VSS.
//               Connects the Datapath and ControlUnit together.
//               Adds a single new internal wire (vector_mode) routed from
//               the ControlUnit to the Datapath, enabling SIMD-lite ALU
//               operation when the VECTOR opcode is decoded.
//
// Task 8 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module StarCore1 (
    input clk       // System clock — drives both the Datapath and GPR/DataMemory
    input  wire       rst_n,      // Add if you are using reset
    input  wire [15:0] gpio_in,    // NEW: Board pins
    output wire [15:0] gpio_out    // NEW: Board pins
);

    // =========================================================================
    // INTERNAL CONTROL WIRES
    // These signals connect the ControlUnit outputs to the Datapath inputs,
    // and the Datapath opcode output back to the ControlUnit input.
    // =========================================================================

    // TODO: Declare all internal control wires here.
    //
    //       wire        jump;
    //       wire        beq;
    //       wire        bne;
    //       wire        mem_read;
    //       wire        mem_write;
    //       wire        alu_src;
    //       wire        reg_dst;
    //       wire        mem_to_reg;
    //       wire        reg_write;
    //       wire [1:0]  alu_op;
    //       wire [3:0]  opcode;

    wire        jump;
    wire        beq;
    wire        bne;
    wire        mem_read;
    wire        mem_write;
    wire        alu_src;
    wire        reg_dst;
    wire        mem_to_reg;
    wire        reg_write;
    wire [1:0]  alu_op;
    wire [3:0]  opcode;
    wire        vector_mode;   // NEW: VSS — routed from ControlUnit to Datapath


    // =========================================================================
    // DATAPATH INSTANTIATION
    // =========================================================================

    // TODO: Instantiate the Datapath module using named port connections.
    //       All control inputs come from the ControlUnit wires declared above.
    //       The opcode output goes to the ControlUnit input.
    //
    //       Datapath DU (
    //           .clk        (clk),
    //           .jump       (jump),
    //           .beq        (beq),
    //           .bne        (bne),
    //           .mem_read   (mem_read),
    //           .mem_write  (mem_write),
    //           .alu_src    (alu_src),
    //           .reg_dst    (reg_dst),
    //           .mem_to_reg (mem_to_reg),
    //           .reg_write  (reg_write),
    //           .alu_op     (alu_op),
    //           .opcode     (opcode)
    //       );

    Datapath DU (
        .clk        (clk),
        .rst_n      (rst_n),
        .gpio_in    (gpio_in),   // Route down to Datapath
        .gpio_out   (gpio_out),  // Route up from Datapath
        .jump       (jump),
        .beq        (beq),
        .bne        (bne),
        .mem_read   (mem_read),
        .mem_write  (mem_write),
        .alu_src    (alu_src),
        .reg_dst    (reg_dst),
        .mem_to_reg (mem_to_reg),
        .reg_write  (reg_write),
        .alu_op     (alu_op),
        .vector_mode (vector_mode),  // NEW
        .opcode     (opcode)
    );


    // =========================================================================
    // CONTROL UNIT INSTANTIATION
    // =========================================================================

    // TODO: Instantiate the ControlUnit module.
    //       Its single input is the opcode from the Datapath.
    //       Its outputs drive all the Datapath control inputs.
    //
    //       ControlUnit CU (
    //           .opcode     (opcode),
    //           .alu_op     (alu_op),
    //           .jump       (jump),
    //           .beq        (beq),
    //           .bne        (bne),
    //           .mem_read   (mem_read),
    //           .mem_write  (mem_write),
    //           .alu_src    (alu_src),
    //           .reg_dst    (reg_dst),
    //           .mem_to_reg (mem_to_reg),
    //           .reg_write  (reg_write)
    //       );

    ControlUnit CU (
        .opcode     (opcode),
        .alu_op     (alu_op),
        .jump       (jump),
        .beq        (beq),
        .bne        (bne),
        .mem_read   (mem_read),
        .mem_write  (mem_write),
        .alu_src    (alu_src),
        .reg_dst    (reg_dst),
        .mem_to_reg (mem_to_reg),
        .reg_write  (reg_write),
        .vector_mode (vector_mode)   // NEW
    );


endmodule
