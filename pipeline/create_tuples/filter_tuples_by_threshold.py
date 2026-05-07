#!/usr/bin/env python3
"""
Filter constraint tuples based on a satisfiability threshold.
Takes tuples from an input file where the satisfiability score >= threshold.
"""

import argparse
import json
from pathlib import Path


def filter_tuples_by_threshold(input_file: Path, output_dir: Path, threshold: float) -> None:
    """
    Filter tuples from input file based on satisfiability threshold.
    Preserves original lines exactly as-is (copy and paste).
    
    Args:
        input_file: Path to input JSONL file containing tuples with satisfiability scores
        output_dir: Directory to save filtered output
        threshold: Minimum satisfiability score (tuples with score >= threshold are kept)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input file and keep original lines
    lines_to_keep = []
    total_tuples = 0
    
    print(f"Reading tuples from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            total_tuples += 1
            
            # Parse only to check the score, but keep the original line
            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON: {stripped_line[:100]}...")
                continue
            
            # Check for satisfiability score in different possible field names
            score = None
            if 'percentage' in record:
                score = record['percentage']
            elif 'min_yes_votes' in record:
                score = record['min_yes_votes']
            elif 'min_acceptance' in record:
                score = record['min_acceptance']
            elif 'satisfiability' in record:
                score = record['satisfiability']
            else:
                print(f"Warning: No satisfiability score found in record: {record}")
                continue
            
            # Filter based on threshold - keep the original line
            if score >= threshold:
                lines_to_keep.append(line)  # Keep original line with original formatting
    
    # Generate output filename based on input filename and threshold
    input_stem = input_file.stem  # filename without extension
    output_file = output_dir / f"{input_stem}_threshold_{threshold}.jsonl"
    
    # Write filtered lines to output file (exact copy)
    print(f"\nFiltering tuples with satisfiability >= {threshold}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines_to_keep:
            f.write(line)  # Write original line as-is
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total tuples in input: {total_tuples}")
    print(f"  Tuples with score >= {threshold}: {len(lines_to_keep)} ({len(lines_to_keep)/total_tuples*100:.2f}%)")
    print(f"  Filtered tuples removed: {total_tuples - len(lines_to_keep)}")
    print(f"\nOutput saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Filter constraint tuples based on satisfiability threshold.")
    parser.add_argument('--input-file', type=str, required=True, help='Path to input JSONL file containing tuples with satisfiability scores')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save filtered output')
    parser.add_argument('--threshold', type=float, required=True, help='Minimum satisfiability score (tuples with score >= threshold are kept)')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    # Validate input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Run filtering
    filter_tuples_by_threshold(input_file, output_dir, args.threshold)


if __name__ == "__main__":
    main()
