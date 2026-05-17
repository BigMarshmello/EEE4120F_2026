#!/usr/bin/env python3
# =========================================================================
# EEE4120F HPES Project: VSS (Vectorized Signal Star)
# Group 7 — Kloppers (KLPJOA002), Hillman (HLLALE010)
# =========================================================================
#
# File         : vss_golden.py
# Description  : Software golden-measure model of the VSS ALU.
#                Independent reference implementation against which the
#                Verilog DUT is checked. If both agree, the hardware is
#                provably equivalent to the specification.
#
# Usage:        python3 vss_golden.py
#               # OR import as a module:
#               from vss_golden import alu_scalar, alu_vector
# =========================================================================

MASK16 = 0xFFFF
MASK8  = 0xFF

# ALU operation codes (match the Verilog ALU)
ADD, SUB, INV, SHL, SHR, AND, OR, SLT = range(8)
OP_NAMES = ["ADD", "SUB", "INV", "SHL", "SHR", "AND", "OR", "SLT"]


def alu_scalar(a: int, b: int, op: int) -> int:
    """Reference for the 16-bit scalar ALU (StarCore-1 baseline)."""
    a &= MASK16
    b &= MASK16
    if   op == ADD: r = (a + b) & MASK16
    elif op == SUB: r = (a - b) & MASK16
    elif op == INV: r = (~a)    & MASK16
    elif op == SHL: r = (a << (b & 0xF)) & MASK16
    elif op == SHR: r = (a >> (b & 0xF)) & MASK16
    elif op == AND: r = a & b
    elif op == OR:  r = a | b
    elif op == SLT: r = 1 if a < b else 0
    else:           r = (a + b) & MASK16
    return r


def alu_vector(a: int, b: int, op: int) -> int:
    """Reference for the VSS SIMD-lite ALU: two independent 8-bit lanes.

    The carry chain between bit 7 and bit 8 is BROKEN, so low-lane
    overflow can never leak into the high lane.
    """
    a &= MASK16
    b &= MASK16
    a_hi, a_lo = (a >> 8) & MASK8, a & MASK8
    b_hi, b_lo = (b >> 8) & MASK8, b & MASK8

    if op == ADD:
        hi = (a_hi + b_hi) & MASK8
        lo = (a_lo + b_lo) & MASK8
    elif op == SUB:
        hi = (a_hi - b_hi) & MASK8
        lo = (a_lo - b_lo) & MASK8
    elif op == INV:
        # Inverts the full 16-bit word (matches the Verilog ALU)
        return (~a) & MASK16
    elif op == SHL:
        # Per-lane shift amount: hi uses b[11:8], lo uses b[3:0]
        hi = (a_hi << ((b >> 8) & 0xF)) & MASK8
        lo = (a_lo << (b & 0xF)) & MASK8
    elif op == SHR:
        hi = (a_hi >> ((b >> 8) & 0xF)) & MASK8
        lo = (a_lo >> (b & 0xF)) & MASK8
    elif op == AND:
        return (a & b) & MASK16
    elif op == OR:
        return (a | b) & MASK16
    elif op == SLT:
        hi = 1 if a_hi < b_hi else 0
        lo = 1 if a_lo < b_lo else 0
    else:
        hi = (a_hi + b_hi) & MASK8
        lo = (a_lo + b_lo) & MASK8

    return ((hi << 8) | lo) & MASK16


# -------------------------------------------------------------------------
# Self-check: print the expected outputs for every vector in ALU_tb.v.
# These values are what the Verilog DUT must produce to PASS.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("=================================================================")
    print(" VSS Golden Model — expected ALU outputs")
    print(" Group 7 (KLPJOA002, HLLALE010)")
    print("=================================================================\n")

    tests = [
        # (mode, label, a, b, op)
        ("scalar", "ADD-carry",    0x00FF, 0x0001, ADD),
        ("scalar", "SUB-borrow",   0x0100, 0x0001, SUB),
        ("scalar", "SHL4",         0x0FFF, 0x0004, SHL),
        ("scalar", "SLT-true",     0x0005, 0x000A, SLT),
        ("vector", "VADD-basic",   0x0A05, 0x2007, ADD),
        ("vector", "VADD-nocarry", 0x10FF, 0x0001, ADD),  # key VSS demo
        ("vector", "VSUB-basic",   0x8010, 0x4002, SUB),
        ("vector", "VSUB-noborrow",0x2000, 0x1001, SUB),  # key VSS demo
        ("vector", "VAND",         0xF0F0, 0x0FFF, AND),
        ("vector", "VOR",          0xF000, 0x00F0, OR),
        ("vector", "VSLT-mix",     0x05FF, 0x1001, SLT),
        ("vector", "VSHL",         0x0103, 0x0201, SHL),
        ("vector", "VSHR",         0x8040, 0x0201, SHR),
        ("vector", "VINV",         0xAA55, 0x0000, INV),
    ]

    print(f"  {'mode':<7} {'label':<14} {'op':<4}  a       b       expected")
    print(f"  {'-'*7} {'-'*14} {'-'*4}  {'-'*7} {'-'*7} {'-'*7}")
    for mode, label, a, b, op in tests:
        ref = alu_scalar(a, b, op) if mode == "scalar" else alu_vector(a, b, op)
        print(f"  {mode:<7} {label:<14} {OP_NAMES[op]:<4}  0x{a:04X}  0x{b:04X}  0x{ref:04X}")

    print("\nThese are the values the Verilog ALU must produce.")
    print("Run ALU_tb.v under iverilog and compare against this list.")
