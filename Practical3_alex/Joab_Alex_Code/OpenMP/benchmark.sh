#!/bin/bash

# ============================================================
# benchmark.sh - Run a command N times and collect timing data
# ============================================================
# Usage: ./benchmark.sh [options]
#   -n  Number of runs per processor count (default: 1000)
#   -m  Max number of processors, iterates NP=1..MAX (default: 1)
#   -s  Matrix size (default: 10)
#   -f  Input/output file base name (default: energy)
# ============================================================

# --- Defaults ---
RUNS=100
MAX_NP=1
SIZE=10
FILEBASE="energy"

# --- Parse arguments ---
while getopts "n:m:s:f:" opt; do
  case $opt in
    n) RUNS="$OPTARG" ;;
    m) MAX_NP="$OPTARG" ;;
    s) SIZE="$OPTARG" ;;
    f) FILEBASE="$OPTARG" ;;
    *) echo "Unknown option: -$OPTARG"; exit 1 ;;
  esac
done

# --- Derive file paths and CSV name from size ---
INPUT="input/${FILEBASE}${SIZE}"
OUTPUT="output/${FILEBASE}${SIZE}.txt"
CSV_FILE="benchmark_${FILEBASE}${SIZE}.csv"

TOTAL=$((RUNS * MAX_NP))

# --- Setup ---
echo "============================================"
echo "  Benchmark Configuration"
echo "  Matrix size:   $SIZE"
echo "  Runs per NP:   $RUNS"
echo "  NP range:      1 to $MAX_NP"
echo "  Total runs:    $TOTAL"
echo "  Input:         $INPUT"
echo "  Output:        $OUTPUT"
echo "  CSV output:    $CSV_FILE"
echo "============================================"
echo ""

# Write CSV header
echo "np,run,Tinit_seconds,Tcomp_seconds" > "$CSV_FILE"

# --- Outer loop: processor counts ---
for ((np=1; np<=MAX_NP; np++)); do

  echo ">>> Starting NP=$np ($RUNS runs)..."

  # --- Inner loop: repeated runs ---
  for ((i=1; i<=RUNS; i++)); do

    # Run the command and capture stdout
    RAW_OUTPUT=$(make run ARGS="-p $np -i $INPUT -o $OUTPUT" 2>&1)

    # Parse Tinit and Tcomp from output
    TINIT=$(echo "$RAW_OUTPUT" | grep -oP 'Tinit:\s+\K[0-9]+\.[0-9]+')
    TCOMP=$(echo "$RAW_OUTPUT" | grep -oP 'Tcomp:\s+\K[0-9]+\.[0-9]+')

    # Validate that we got values
    if [[ -z "$TINIT" || -z "$TCOMP" ]]; then
      echo "  WARNING: NP=$np Run $i failed to parse output. Skipping."
      echo "  Raw output was: $RAW_OUTPUT"
      continue
    fi

    # Append to CSV
    echo "$np,$i,$TINIT,$TCOMP" >> "$CSV_FILE"

    # Progress indicator every 50 runs
    if (( i % 50 == 0 )); then
      echo "  Progress: $i / $RUNS runs complete..."
    fi

  done

  # Per-NP summary using awk
  echo ""
  awk -F',' -v np="$np" 'NR>1 && $1==np {
    tinit_sum += $3; tcomp_sum += $4;
    tinit_min = (count==0 || $3 < tinit_min) ? $3 : tinit_min;
    tcomp_min = (count==0 || $4 < tcomp_min) ? $4 : tcomp_min;
    tinit_max = ($3 > tinit_max) ? $3 : tinit_max;
    tcomp_max = ($4 > tcomp_max) ? $4 : tcomp_max;
    count++
  } END {
    printf "  NP=%d Summary:\n", np;
    printf "    Tinit  -> avg: %.6f  min: %.6f  max: %.6f\n", tinit_sum/count, tinit_min, tinit_max;
    printf "    Tcomp  -> avg: %.6f  min: %.6f  max: %.6f\n", tcomp_sum/count, tcomp_min, tcomp_max;
  }' "$CSV_FILE"
  echo ""

done

# --- Final summary ---
echo "============================================"
echo "All done! Results saved to $CSV_FILE"
echo "============================================"
