#!/usr/bin/env python3
"""
Script to filter and intersect constraints from three analysis phases.

This script:
1. Takes 3 input JSONL files with constraints and percentages
2. Takes 3 thresholds (one for each file)
3. Filters each file by its respective threshold (percentage >= threshold)
4. Intersects the remaining constraints from all three files
5. Outputs a JSONL file with the intersected constraints
6. Outputs a txt file with statistics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set


def load_constraints(file_path: str) -> Dict[str, float]:
    """
    Load constraints from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Dictionary mapping constraint text to percentage
    """
    constraints = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            constraints[data['constraint']] = data['percentage']
    return constraints


def filter_constraints(constraints: Dict[str, float], threshold: float) -> Dict[str, float]:
    """
    Filter constraints by threshold.
    
    Args:
        constraints: Dictionary of constraints with percentages
        threshold: Minimum percentage to keep
        
    Returns:
        Filtered dictionary of constraints
    """
    return {c: p for c, p in constraints.items() if p >= threshold}


def intersect_constraints(
    constraints1: Dict[str, float],
    constraints2: Dict[str, float],
    constraints3: Dict[str, float]
) -> Set[str]:
    """
    Find the intersection of constraints across three dictionaries.
    
    Args:
        constraints1: First dictionary of constraints
        constraints2: Second dictionary of constraints
        constraints3: Third dictionary of constraints
        
    Returns:
        Set of constraint texts present in all three dictionaries
    """
    set1 = set(constraints1.keys())
    set2 = set(constraints2.keys())
    set3 = set(constraints3.keys())
    return set1 & set2 & set3


def main():
    parser = argparse.ArgumentParser(description='Filter and intersect constraints from three analysis phases')
    parser.add_argument('--satisfiability', type=str, required=True, help='Path to satisfiability analysis JSONL file')
    parser.add_argument('--subjectivity', type=str, required=True, help='Path to subjectivity analysis JSONL file')
    parser.add_argument('--triviality', type=str, required=True, help='Path to triviality analysis JSONL file')
    parser.add_argument('--satisfiability-threshold', type=float, required=True, help='Threshold for satisfiability (percentage >= this value)')
    parser.add_argument('--subjectivity-threshold', type=float, required=True, help='Threshold for subjectivity (percentage >= this value)')
    parser.add_argument('--triviality-threshold', type=float, required=True, help='Threshold for triviality (percentage >= this value)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')

    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all constraints
    print("Loading constraints...")
    satisfiability_all = load_constraints(args.satisfiability)
    subjectivity_all = load_constraints(args.subjectivity)
    triviality_all = load_constraints(args.triviality)
    
    print(f"Loaded {len(satisfiability_all)} constraints from satisfiability file")
    print(f"Loaded {len(subjectivity_all)} constraints from subjectivity file")
    print(f"Loaded {len(triviality_all)} constraints from triviality file")
    
    # Filter by thresholds
    print("\nFiltering constraints by thresholds...")
    satisfiability_filtered = filter_constraints(satisfiability_all, args.satisfiability_threshold)
    subjectivity_filtered = filter_constraints(subjectivity_all, args.subjectivity_threshold)
    triviality_filtered = filter_constraints(triviality_all, args.triviality_threshold)
    
    satisfiability_removed = len(satisfiability_all) - len(satisfiability_filtered)
    subjectivity_removed = len(subjectivity_all) - len(subjectivity_filtered)
    triviality_removed = len(triviality_all) - len(triviality_filtered)
    
    print(f"Satisfiability: {len(satisfiability_filtered)} remaining, {satisfiability_removed} removed (threshold: {args.satisfiability_threshold})")
    print(f"Subjectivity: {len(subjectivity_filtered)} remaining, {subjectivity_removed} removed (threshold: {args.subjectivity_threshold})")
    print(f"Triviality: {len(triviality_filtered)} remaining, {triviality_removed} removed (threshold: {args.triviality_threshold})")
    
    # Intersect constraints
    print("\nIntersecting filtered constraints...")
    intersected_constraints = intersect_constraints(
        satisfiability_filtered,
        subjectivity_filtered,
        triviality_filtered
    )
    
    print(f"Intersection result: {len(intersected_constraints)} constraints")
    
    # Prepare output data
    output_data = []
    for constraint in sorted(intersected_constraints):
        output_data.append({
            'constraint': constraint,
            'satisfiability': satisfiability_filtered[constraint],
            'subjectivity': subjectivity_filtered[constraint],
            'triviality': triviality_filtered[constraint]
        })
    
    # Write intersected constraints to JSONL
    output_jsonl = output_dir / 'intersected_constraints.jsonl'
    print(f"\nWriting intersected constraints to {output_jsonl}...")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Write statistics report
    output_stats = output_dir / 'intersection_statistics.txt'
    print(f"Writing statistics to {output_stats}...")
    with open(output_stats, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CONSTRAINT INTERSECTION STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("INPUT FILES:\n")
        f.write(f"  Satisfiability: {args.satisfiability}\n")
        f.write(f"  Subjectivity:   {args.subjectivity}\n")
        f.write(f"  Triviality:     {args.triviality}\n\n")
        
        f.write("THRESHOLDS:\n")
        f.write(f"  Satisfiability: {args.satisfiability_threshold}%\n")
        f.write(f"  Subjectivity:   {args.subjectivity_threshold}%\n")
        f.write(f"  Triviality:     {args.triviality_threshold}%\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("FILTERING RESULTS:\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Satisfiability Analysis:\n")
        f.write(f"  Original count:  {len(satisfiability_all):>6}\n")
        f.write(f"  Removed:         {satisfiability_removed:>6} ({satisfiability_removed/len(satisfiability_all)*100:.2f}%)\n")
        f.write(f"  Remaining:       {len(satisfiability_filtered):>6} ({len(satisfiability_filtered)/len(satisfiability_all)*100:.2f}%)\n\n")
        
        f.write(f"Subjectivity Analysis:\n")
        f.write(f"  Original count:  {len(subjectivity_all):>6}\n")
        f.write(f"  Removed:         {subjectivity_removed:>6} ({subjectivity_removed/len(subjectivity_all)*100:.2f}%)\n")
        f.write(f"  Remaining:       {len(subjectivity_filtered):>6} ({len(subjectivity_filtered)/len(subjectivity_all)*100:.2f}%)\n\n")
        
        f.write(f"Triviality Analysis:\n")
        f.write(f"  Original count:  {len(triviality_all):>6}\n")
        f.write(f"  Removed:         {triviality_removed:>6} ({triviality_removed/len(triviality_all)*100:.2f}%)\n")
        f.write(f"  Remaining:       {len(triviality_filtered):>6} ({len(triviality_filtered)/len(triviality_all)*100:.2f}%)\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("INTERSECTION RESULTS:\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Constraints in all three analyses:  {len(intersected_constraints):>6}\n\n")
        
        # Additional analysis
        only_in_12 = set(satisfiability_filtered.keys()) & set(subjectivity_filtered.keys()) - set(triviality_filtered.keys())
        only_in_13 = set(satisfiability_filtered.keys()) & set(triviality_filtered.keys()) - set(subjectivity_filtered.keys())
        only_in_23 = set(subjectivity_filtered.keys()) & set(triviality_filtered.keys()) - set(satisfiability_filtered.keys())
        
        f.write(f"Constraints in Satisfiability & Subjectivity only:  {len(only_in_12):>6}\n")
        f.write(f"Constraints in Satisfiability & Triviality only:    {len(only_in_13):>6}\n")
        f.write(f"Constraints in Subjectivity & Triviality only:      {len(only_in_23):>6}\n\n")
        
        only_sat = set(satisfiability_filtered.keys()) - set(subjectivity_filtered.keys()) - set(triviality_filtered.keys())
        only_sub = set(subjectivity_filtered.keys()) - set(satisfiability_filtered.keys()) - set(triviality_filtered.keys())
        only_tri = set(triviality_filtered.keys()) - set(satisfiability_filtered.keys()) - set(subjectivity_filtered.keys())
        
        f.write(f"Constraints only in Satisfiability:  {len(only_sat):>6}\n")
        f.write(f"Constraints only in Subjectivity:    {len(only_sub):>6}\n")
        f.write(f"Constraints only in Triviality:      {len(only_tri):>6}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"OUTPUT FILES:\n")
        f.write(f"  Intersected constraints: {output_jsonl}\n")
        f.write(f"  Statistics report:       {output_stats}\n")
        f.write("=" * 80 + "\n")
    
    print("\nDone!")
    print(f"\nResults saved to:")
    print(f"  - {output_jsonl}")
    print(f"  - {output_stats}")


if __name__ == '__main__':
    main()
