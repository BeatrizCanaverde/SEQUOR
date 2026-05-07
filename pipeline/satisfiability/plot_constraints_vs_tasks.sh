#!/bin/bash

# Script to run constraint vs tasks analysis
# This will generate histogram plots showing the distribution of constraints by their task satisfaction percentage

# Configuration
TASKS_FILE="satisfiability/data/tasks.jsonl"
OUTPUT_DIR="satisfiability/constraints_vs_tasks_analysis"
NUM_TASKS=100
BIN_SIZE=5  # Size of percentage bins (e.g., 5 creates bins: 0-5%, 5-10%, ..., 95-100%)

# Find all model output files
MODEL_FILES=(satisfiability/outputs_judge/*.jsonl)

# Check if we have any model files
if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    echo "Error: No model output files found in satisfiability/outputs_judge_2_new/"
    exit 1
fi

echo "Found ${#MODEL_FILES[@]} model output files:"
for file in "${MODEL_FILES[@]}"; do
    echo "  - $file"
done

echo ""
echo "Using bin size: ${BIN_SIZE}%"
echo "  (Histogram bins: 0-${BIN_SIZE}%, ${BIN_SIZE}-$((BIN_SIZE*2))%, ..., $((100-BIN_SIZE))-100%)"
echo ""

# Run the analysis
python pipeline/satisfiability/plot_constraints_vs_tasks.py \
    --tasks-file "$TASKS_FILE" \
    --model-files "${MODEL_FILES[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --num-tasks "$NUM_TASKS" \
    --bin-size "$BIN_SIZE"

echo ""
echo "Analysis complete! Check the plots in: $OUTPUT_DIR"
echo "  - Histograms: $OUTPUT_DIR/histograms/"
echo "  - Cumulative plots: $OUTPUT_DIR/plots/"
echo "  - Data by percentage: $OUTPUT_DIR/data/"
echo "  - All constraints with percentages: $OUTPUT_DIR/all_constraints_percentages.jsonl"
echo "  - Analysis log: $OUTPUT_DIR/analysis_log.txt"
