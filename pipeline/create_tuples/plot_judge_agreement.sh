#!/bin/bash

# Script to run judge acceptance analysis for create_tuples experiments
# This will generate plots showing the distribution of constraint tuples by judge acceptance
# Methodology: Counts tuples where all judges have >= threshold% Yes votes

# Configuration
JUDGE_FOLDER="create_tuples/tuples"  # Folder containing judge subfolders
OUTPUT_DIR="create_tuples/judge_agreement_analysis"
BIN_SIZE=5  # Size of percentage bins (e.g., 5 creates bins: 0-5%, 5-10%, ..., 95-100%)

# Check if judge folder exists
if [ ! -d "$JUDGE_FOLDER" ]; then
    echo "Error: Judge folder not found: $JUDGE_FOLDER"
    echo "Please run the judge evaluation first"
    exit 1
fi

# Count judge subfolders (excluding 'logs')
num_judges=$(find "$JUDGE_FOLDER" -mindepth 1 -maxdepth 1 -type d ! -name "logs" | wc -l)

if [ "$num_judges" -eq 0 ]; then
    echo "Error: No judge subfolders found in $JUDGE_FOLDER"
    echo "Expected structure: $JUDGE_FOLDER/judge_name/*.jsonl"
    exit 1
fi

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
python pipeline/create_tuples/plot_judge_agreement.py \
    --judge-folder "$JUDGE_FOLDER" \
    --output-dir "$OUTPUT_DIR" \
    --bin-size "$BIN_SIZE"

echo ""
echo "Analysis complete! Check the plots in: $OUTPUT_DIR"
echo "  - Histograms: $OUTPUT_DIR/histograms/"
echo "  - Cumulative plots: $OUTPUT_DIR/plots/"
echo "  - Data by percentage: $OUTPUT_DIR/data/"
echo "  - All tuples with percentages: $OUTPUT_DIR/all_tuples_percentages.jsonl"
echo "  - Analysis log: $OUTPUT_DIR/analysis_log.txt"
