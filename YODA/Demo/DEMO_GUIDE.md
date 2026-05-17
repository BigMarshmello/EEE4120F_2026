# VSS Demo Day Guide

**EEE4120F HPES Project · Group 7**
Joab Gray Kloppers (KLPJOA002) · Alex Hillman (HLLALE010)

Time budget: **10 min presenting + 5 min Q&A + 5 min switchover**

---

## TL;DR — your demo loop

You will spend roughly 7 minutes on slides and 3 minutes on the actual hardware sim. The hardware sim has three runs, each taking about 30 seconds and producing instantly-quotable output:

1. **`vvp alu_tb.vvp`** — 14 tests, scalar + vector. Console PASS/FAIL is your gold standard.
2. **`gtkwave ALU_tb.vcd`** — open the waveform, point at the lane-split.
3. **`vvp aluctrl_tb.vvp && vvp cu_tb.vvp`** — 25 more tests covering the control path.

That's the whole live portion. Everything else is slide-driven.

---

## Pre-demo checklist (the night before)

- [ ] **Tools installed on the laptop that will present:**
  - `iverilog --version` ≥ 11.0 (you have 12.0 — perfect)
  - `gtkwave --version` opens fine
  - Python 3 with no external imports needed for the golden model
- [ ] **Files in one folder** (`vss_demo/`) so you can `cd` once and never again:
  ```
  vss_demo/
  ├── src/
  │   └── Parameter.v
  ├── testbenches/
  │   ├── ALU.v             ← your module
  │   ├── ALU_Control.v     ← your module
  │   ├── ControlUnit.v     ← your module
  │   ├── ALU_tb.v
  │   ├── ALU_Control_tb.v
  │   ├── ControlUnit_tb.v
  │   └── Parameter.v
  ├── golden_model/
  │   └── vss_golden.py
  └── docs/
      ├── VSS_Flyer.pdf
      ├── VSS_Slides.pptx
      └── test_transcript.txt
  ```
- [ ] **Dry-run the three `vvp` commands** on the laptop you'll present from. If iverilog can't find `Parameter.v`, the include path is wrong.
- [ ] **Pre-compile** the `.vvp` files. If the lab Wi-Fi dies or iverilog stalls, you can just `vvp` an already-compiled binary.
- [ ] **Open GTKWave once with `ALU_tb.vcd`** and save the signal-list view (File → Write Save File). On demo day reload with `gtkwave ALU_tb.vcd ALU_tb.gtkw`. Saves 90 seconds of clicking.
- [ ] **Print or open the flyer** on a phone/tablet ready to hand to the marker.
- [ ] **Slide deck opens cleanly** in the venue's projector resolution (test 16:9, not 4:3).
- [ ] **Each member knows their slides**. Suggested split:
  - **Alex** drives slides 1–6 (motivation, system, ISA, what changed)
  - **Joab** drives slides 7–11 (gold standard, waveforms, performance, demo)
  - Either of you fields Q&A — agree in advance who answers what

---

## What to actually run, in order

All commands assume `cd vss_demo/testbenches`.

### Run 1 — Gold standard for the ALU (the headline)

```bash
iverilog -o alu_tb.vvp ALU_tb.v ALU.v
vvp alu_tb.vvp
```

**What appears on screen** (real output from your modules):

```
=================================================================
 VSS ALU Golden-Measure Test          Group 7 (KLPJOA002, HLLALE010)
=================================================================

--- Scalar mode (16-bit, baseline) ---
  [1] PASS  ADD-carry     mode=0 op=000  a=00ff b=0001  result=0100  (expected 0100)
  [2] PASS  SUB-borrow    mode=0 op=001  a=0100 b=0001  result=00ff  (expected 00ff)
  ...

--- Vector mode (2 x 8-bit lanes, VSS) ---
  [5] PASS  VADD-basic    mode=1 op=000  a=0a05 b=2007  result=2a0c  (expected 2a0c)
  [6] PASS  VADD-nocarry  mode=1 op=000  a=10ff b=0001  result=1000  (expected 1000)
  ...

 Results: 14 PASS, 0 FAIL  (out of 14 tests)
 *** GOLDEN MEASURE MATCH — VSS ALU verified ***
```

**What to say while it runs:**
> "We've written a software golden model in Python that captures what VSS should do on every operation. The Verilog testbench drives the same vectors through the hardware and compares against expected values. Watch test 6, **VADD-nocarry** — that's 0x10FF + 0x0001. In scalar mode you'd get 0x1100, the carry from bit 7 propagating into bit 8. In vector mode you get 0x1000 — the low lane wraps independently and the high lane is untouched. That broken carry chain is the whole story."

### Run 2 — Waveform (the visual proof)

```bash
gtkwave ALU_tb.vcd
```

**What to show:** drag in `vector_mode`, `a[15:0]`, `b[15:0]`, `result[15:8]`, `result[7:0]`. Zoom to the transition between tests 1 and 5.

**What to say:**
> "Same signals, two cycles. Top one is scalar — `vector_mode = 0`, full 16-bit add, result 0x0100. The cycle after `vector_mode` goes high, we feed 0x10FF + 0x0001. Low byte: 0xFF + 1 wraps to 0x00. High byte: 0x10 + 0x00 = 0x10. The carry that would have leaked across is dropped on the floor by design."

### Run 3 — Control path verified

```bash
vvp aluctrl_tb.vvp   # 16/16 PASS  — funct passthrough on ALUOp=11
vvp cu_tb.vvp        # 9/9 PASS    — vector_mode asserted ONLY on opcode 1110
```

**What to say:**
> "The ALU is half the story. The control path also has to recognise the new VECTOR instruction and route it correctly. ALU_Control sees ALUOp = 2'b11 and passes the funct field straight to the ALU, no decode table needed. The main Control Unit raises `vector_mode` for exactly one opcode — 1110 — and zero others. 25 more tests, all pass."

---

## Mapping your demo to the rubric (you have 100 marks to earn)

| # | Rubric item                                | Where you earn it                                                         | Marks |
|---|--------------------------------------------|---------------------------------------------------------------------------|-------|
| 1 | Marketing flyer + intro + motivation       | Slides 1–3 + hand over the flyer PDF                                      | 20    |
| 2 | Gold Standard Output                       | `vvp alu_tb.vvp` console output (14/14)                                   | 10    |
| 3 | Gold Standard Explanation                  | Walk through Python golden model on slide 7; "39/39 PASS" callout         | 10    |
| 4 | Simulation of SoC / FPGA testing Output    | GTKWave waveform on slide 9, then live                                    | 10    |
| 5 | Custom processor explanation               | Slide 5 (changes to ALU/ALU_Control/ControlUnit), slide 6 (ISA encoding)  | 10    |
| 6 | Performance analysis vs baseline           | Slide 10 — the modelled 5–8× speedup chart                                | 10    |
| 7 | Entire System Explanation                  | Slide 4 — the annotated block diagram                                     | 10    |
| 8 | Demo Organization                          | This guide. Smooth transitions, no fumbling, clear member roles           | 10    |

---

## Anticipated questions and how to answer them

These come up in pretty much every microarchitecture demo. Have an answer ready.

**Q: "Why opcode 1110 and not 1010?"**
A: 1010 is the brief's reserved coprocessor opcode — we deliberately left it free so a future group could bolt on a coprocessor without an ISA collision. 1110 was unallocated.

**Q: "Why didn't you do saturating arithmetic? Real SIMD always saturates."**
A: Cost. Saturation needs an overflow detector and a conditional clamp per lane — extra logic with no demo-day payoff. We chose modular wrap because it matches the brief's scope and keeps the ALU purely combinational.

**Q: "How would you measure real performance without an FPGA?"**
A: Two ways. Hand-written assembly for both modes, then count executed instructions in the simulator — since the ISA is single-cycle, instructions equal cycles. For a stronger comparison we'd add a small instruction memory + program counter testbench (most of the datapath already exists) and run end-to-end benchmarks. The numbers on slide 10 are modelled from per-instruction breakdowns, not measured — that's labelled.

**Q: "What about the I/O integration the brief asked for?"**
A: We focused this milestone on the ISA extension and verification path. Memory-mapped I/O hooks into the Data Memory address-decode and is in the report as future work — adding an MMIO range would not require any change to the VSS ALU.

**Q: "Per-lane shift counts use b[11:8] and b[3:0] — why those bits?"**
A: They're the natural lane-aligned positions. The high-byte shift amount lives in the high nibble of byte 1 of operand B; the low-byte shift in the low nibble of byte 0. Programmer-friendly: one register holds both shift amounts in lane-correct positions.

**Q: "What if I want VINV on only one lane?"**
A: You can't with the current encoding — VINV inverts the whole 16-bit operand. We discuss this in the report as a documented limitation; per-lane mask bits could be added at the cost of two more instruction bits.

**Q: "Did you verify anything more than the modules in isolation?"**
A: Three module-level testbenches give 39/39 PASS. Top-level integration (instruction memory + datapath + control + ALU) is the next step we discuss in the report — it requires the rest of the datapath which is unchanged from Practical 4.

---

## Backup plan if something breaks

- **Laptop won't compile iverilog**: open `test_transcript.txt` — that's a frozen capture of all 39 tests passing. Walk through it line by line. Same talking points.
- **GTKWave crashes or won't open the VCD**: switch to slide 9 (the static waveform image). Same story, just no zoom/pan.
- **Projector resolution kills the slides**: the flyer PDF is self-contained and uses the same colour scheme — open it instead and narrate from the marketing angle, then walk through `test_transcript.txt`.
- **Question you don't know**: "Good question — that's covered in the report; happy to come back to it after the demo." Don't bluff.

---

## Final slide of the deck, paraphrased

> "VSS — Vectorized Signal Star. One new opcode, two parallel byte lanes, six-times speedup on packed telemetry workloads. 39 of 39 golden tests pass. Thank you. Questions?"

Then take the questions. Keep your answers under 30 seconds each. If you don't know something, say so and offer the report.

Good luck.
