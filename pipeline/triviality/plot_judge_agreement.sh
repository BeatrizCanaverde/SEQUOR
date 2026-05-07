#!/bin/bash

# Script to analyze (No, No, No) verdicts for trivial constraints analysis
# This will generate plots showing the distribution of constraints by their (No, No, No) verdict percentage

# Configuration
JUDGE_FOLDER="triviality/outputs_judge"  # Folder containing judge subfolders
OUTPUT_DIR="triviality/all_no_analysis"
BIN_SIZE=5  # Size of percentage bins (e.g., 5 creates bins: 0-5%, 5-10%, ..., 95-100%)

# Check if judge folder exists
if [ ! -d "$JUDGE_FOLDER" ]; then
    echo "Error: Judge folder not found: $JUDGE_FOLDER"
    echo "Please run the judge evaluation first (see trivial_constraints/llm_judge.sh)"
    exit 1
fi

# Count judge subfolders (excluding 'logs')
num_judges=$(find "$JUDGE_FOLDER" -mindepth 1 -maxdepth 1 -type d ! -name "logs" | wc -l)

if [ "$num_judges" -eq 0 ]; then
    echo "Error: No judge subfolders found in $JUDGE_FOLDER"
    echo "Expected structure: $JUDGE_FOLDER/judge_name/*.jsonl"
    exit 1
fi

echo "Trivial Constraints Analysis - (No, No, No) Verdicts"
echo "======================================================"
echo "Judge folder: $JUDGE_FOLDER"
echo "Found $num_judges judge subfolder(s)"
echo ""
echo "Judge subfolders:"
find "$JUDGE_FOLDER" -mindepth 1 -maxdepth 1 -type d ! -name "logs" -exec basename {} \;
echo ""
echo "Using bin size: ${BIN_SIZE}%"
echo "  (Histogram bins: 0-${BIN_SIZE}%, ${BIN_SIZE}-$((BIN_SIZE*2))%, ..., $((100-BIN_SIZE))-100%)"
echo ""

# Run the analysis
python pipeline/triviality/plot_judge_agreement.py \
    --judge-folder "$JUDGE_FOLDER" \
    --output-dir "$OUTPUT_DIR" \
    --bin-size "$BIN_SIZE"

echo ""
echo "Analysis complete! Check the plots in: $OUTPUT_DIR"
echo "  - Histograms: $OUTPUT_DIR/histograms/"
echo "  - Cumulative plots: $OUTPUT_DIR/plots/"
echo "  - Data by percentage: $OUTPUT_DIR/data/"
echo "  - All constraints with percentages: $OUTPUT_DIR/all_constraints_percentages.jsonl"
echo "  - Analysis log: $OUTPUT_DIR/analysis_log.txt"
