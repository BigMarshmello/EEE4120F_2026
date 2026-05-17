// =========================================================================
// Practical 4: StarCore-1 — Single-Cycle Processor in Verilog
// =========================================================================
//
// GROUP NUMBER: 7
//
// MEMBERS:
//   - Member 1 Joab Gray Kloppers, KLPJOA001
//   - Member 2 Name, Student Number
//   - Member 3 Name, Student Number

// File        : IOR.v
// Description : Input Output Memory File.
//               2 registers, each 16 bits wide (IO0 IO1).
//               One asynchronous (combinational) read port.
//               One synchronous (clocked, positive-edge) write port.
//
// Task 2 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module IOMEM (
    input  wire clk,
    input  wire rst_n,          
    
    input  wire        write_en,
    input  wire [15:0] write_data,
    output wire [15:0] read_data,

    // NEW: Upgraded to 16 bits!
    output wire [15:0] gpio_out, 
    input  wire [15:0] gpio_in   
);

    // Internal registers upgraded to 16 bits
    reg [15:0] gpio_out_reg;
    reg [15:0] gpio_in_sync1; 
    reg [15:0] gpio_in_sync2;

    // Direct routing (no address decoding needed inside this module 
    // because the "traffic cop" in Datapath.v only enables this 
    // module when the address is exactly 0x1000).
    assign gpio_out = gpio_out_reg;
    assign read_data = gpio_in_sync2;

    // Write Logic (16-bit)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gpio_out_reg <= 16'b0;
        end else if (write_en) begin
            gpio_out_reg <= write_data; // Captures all 16 bits at once
        end
    end

    // Read Synchronization Logic (16-bit)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gpio_in_sync1 <= 16'b0;
            gpio_in_sync2 <= 16'b0;
        end else begin
            gpio_in_sync1 <= gpio_in;       
            gpio_in_sync2 <= gpio_in_sync1; 
        end
    end
endmodule
