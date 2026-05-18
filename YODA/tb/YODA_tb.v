// =============================================================================
// EEE4120F Project — StarCore-1 Processor with GPIO and 16-bit Instructions
// File        : YODA_tb.v
// Description : Testbench for the GPIO and 16-bit instruction modules
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module StarCore1_tb;

    // -------------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------------
    reg clk;
    reg rst_n;
    reg [15:0] gpio_in;
    wire [15:0] gpio_out;

    initial clk = 1'b0;
    always  #5 clk = ~clk;     // 10 ns period — 100 MHz

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    StarCore1 uut (
        .clk(clk),
        .rst_n(rst_n),
        .gpio_in(gpio_in),
        .gpio_out(gpio_out)
    );

    // -------------------------------------------------------------------------
    // Waveform dump — captures ALL signals in the design hierarchy
    // -------------------------------------------------------------------------
    initial begin
        $dumpfile("../waves/star.vcd");
        $dumpvars(0, StarCore1_tb);
    end

    // -------------------------------------------------------------------------
    // Failure counter
    // -------------------------------------------------------------------------
    integer fail_count;
    integer test_id;

    initial begin
        fail_count = 0;
        test_id    = 1;
    end

    // -------------------------------------------------------------------------
    // Check tasks — compare 16-bit values observed via hierarchical reference
    // -------------------------------------------------------------------------
    task check16;
        input [15:0] got;
        input [15:0] expected;
        input [63:0] id;
        begin
            if (got !== expected) begin
                $display("FAIL [T%0d]: got = 0x%h (%0d), expected = 0x%h (%0d)",
                         id, got, got, expected, expected);
                fail_count = fail_count + 1;
            end else
                $display("PASS [T%0d]: value = 0x%h (%0d)", id, got, got);
        end
    endtask

    // -------------------------------------------------------------------------
    // Cycle-by-cycle execution trace
    // This always block fires on every rising clock edge and prints the current
    // processor state. It is your primary debugging tool.
    //
    // TODO: Uncomment this block once Datapath.v is fully implemented.
    //       Until then, it will cause "Unable to bind" errors because the
    //       internal signals (pc_current, instr, etc.) do not yet exist.
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        $display("%0t ns | PC=0x%h | instr=%b | R0=%3d R1=%3d R2=%3d R3=%3d | alu=%0d z=%b",
            $time,
            uut.DU.pc_current,
            uut.DU.instr,
            uut.DU.reg_file.reg_array[0],
            uut.DU.reg_file.reg_array[1],
            uut.DU.reg_file.reg_array[2],
            uut.DU.reg_file.reg_array[3],
            uut.DU.alu_result,
            uut.DU.zero_flag
        );
    end

    // =========================================================================
    // MAIN STIMULUS BLOCK
    // =========================================================================
    initial begin
        $display("=== StarCore-1 Integration Testbench ===");
        $display("=== Program loaded from ./test/test.prog ===");
        $display("=== Data memory loaded from ./test/test.data ===");
        $display("");
        rst_n  = 1'b1;
        gpio_in = 16'h0004; // Initialize GPIO input to 0

        // -----------------------------------------------------------------------
        // Wait for the simulation to run long enough for your program to
        // complete at least one full pass. Adjust SIM_TIME in Parameter.v
        // if your program needs more cycles.
        // -----------------------------------------------------------------------
        `SIM_TIME;

        // -----------------------------------------------------------------------
        // POST-SIMULATION VERIFICATION
        //
        // TODO: After implementing Datapath.v and StarCore1.v, uncomment the
        //       check16() calls below and fill in the expected values for your
        //       specific test program.
        //
        //       All hierarchical references below (uut.DU.*, uut.DU.reg_file.*,
        //       uut.DU.dm.*) are commented out because they reference signals
        //       that do not exist until the Datapath is implemented.
        //       Uncomment them one section at a time as you complete each task.
        // -----------------------------------------------------------------------

        $display("");
        $display("--- Post-Simulation Verification (implement Datapath first) ---");

        // LD mem[0] R1 = 1
        $display("Checking R1 after LD mem[0] -> R1:");
        check16(uut.DU.reg_file.reg_array[1], 16'h0001, test_id);
        test_id = test_id + 1;

        // ADD R3, R1, R1
        $display("Checking R3 after ADD (should be 0x0001 + 0x0001 = 0x0002):");
        check16(uut.DU.reg_file.reg_array[3], 16'h0002, test_id);
        test_id = test_id + 1;

        
        //$display("Checking R1 after LD GPIO_in -> R1:");
        //check16(uut.DU.dm.memory[2], 16'h0003, test_id);
        //test_id = test_id + 1;

        // LD GPIO_In R2
        $display("Checking R2 after LD GPIO_in -> R2:");
        check16(uut.DU.reg_file.reg_array[2], 16'h0004, test_id);
        test_id = test_id + 1;

        // ADD R4, R1, R2
        $display("Checking R3 after ADD (should be 0x0004 + 0x0001 = 0x0005):");
        check16(uut.DU.reg_file.reg_array[3], 16'h0005, test_id);
        test_id = test_id + 1;

        // -----------------------------------------------------------------------
        // Print register and memory state (safe to uncomment after Task 7)
        // -----------------------------------------------------------------------
        $display("");
        $display("--- Final Register File State ---");
        $display("R0=0x%h  R1=0x%h  R2=0x%h  R3=0x%h",
            uut.DU.reg_file.reg_array[0], uut.DU.reg_file.reg_array[1],
            uut.DU.reg_file.reg_array[2], uut.DU.reg_file.reg_array[3]);
        $display("R4=0x%h  R5=0x%h  R6=0x%h  R7=0x%h",
            uut.DU.reg_file.reg_array[4], uut.DU.reg_file.reg_array[5],
            uut.DU.reg_file.reg_array[6], uut.DU.reg_file.reg_array[7]);
        
        $display("");
        $display("--- Final Data Memory State ---");
        $display("Mem[0]=0x%h  Mem[1]=0x%h  Mem[2]=0x%h  Mem[3]=0x%h",
            uut.DU.dm.memory[0], uut.DU.dm.memory[1],
            uut.DU.dm.memory[2], uut.DU.dm.memory[3]);
        $display("Mem[4]=0x%h  Mem[5]=0x%h  Mem[6]=0x%h  Mem[7]=0x%h",
            uut.DU.dm.memory[4], uut.DU.dm.memory[5],
            uut.DU.dm.memory[6], uut.DU.dm.memory[7]);

        // -----------------------------------------------------------------------
        // Summary
        // -----------------------------------------------------------------------
        $display("");
        if (fail_count == 0)
            $display("=== ALL %0d INTEGRATION TESTS PASSED ===", test_id - 1);
        else
            $display("=== %0d / %0d INTEGRATION TESTS FAILED ===", fail_count, test_id - 1);

        $finish;
    end

endmodule