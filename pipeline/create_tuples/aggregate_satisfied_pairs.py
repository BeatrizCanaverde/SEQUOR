#!/usr/bin/env python3
"""
Script to aggregate satisfied_pairs.jsonl files from individual model analysis directories
into a single cumulative_satisfied_pairs.jsonl file.
"""

import argparse
from pathlib import Path

def aggregate_satisfied_pairs(input_files, output_file):
    output_path = Path(output_file)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    files_processed = 0

    print(f"Aggregating results from {len(input_files)} files...")
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        for file_path in input_files:
            pairs_file = Path(file_path)
            
            if pairs_file.exists():
                print(f"  Reading {pairs_file}...")
                with open(pairs_file, "r", encoding="utf-8") as infile:
                    count = 0
                    for line in infile:
                        if line.strip():
                            outfile.write(line)
                            count += 1
                    total_lines += count
                    files_processed += 1
            else:
                print(f"  Warning: {pairs_file} not found, skipping.")

    print(f"\nSuccess!")
    print(f"Processed {files_processed} files.")
    print(f"Total satisfied pairs written to {output_file}: {total_lines}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate satisfied_pairs.jsonl files into a single cumulative file.")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of paths to satisfied_pairs.jsonl files")
    parser.add_argument("--output", required=True, help="Output file path for cumulative results")
    args = parser.parse_args()
    
    aggregate_satisfied_pairs(args.inputs, args.output)
