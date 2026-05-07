#!/bin/bash

# Script to analyze constraint tuple satisfiability across tasks

SATISFIED_PAIRS_FILE="create_tuples/judge_results/all_satisfied_pairs.jsonl"
OUTPUT_DIR="create_tuples/tuple_satisfiability_analysis"
EXPECTED_TASKS_PER_TUPLE=100
BIN_SIZE=5

echo "Running constraint tuple satisfiability analysis..."
echo "Input: $SATISFIED_PAIRS_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

python pipeline/create_tuples/plot_tuple_satisfiability.py \
    --satisfied-pairs-file "$SATISFIED_PAIRS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --expected-tasks-per-tuple "$EXPECTED_TASKS_PER_TUPLE" \
    --bin-size "$BIN_SIZE"

echo ""
echo "Analysis complete! Results saved to $OUTPUT_DIR"
