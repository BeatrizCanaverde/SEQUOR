import json
from pathlib import Path
import argparse


def load_constraints_with_scores(constraints_file):
    """Load constraints from data/constraints.jsonl and create a mapping."""
    constraint_scores = {}
    
    with open(constraints_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                constraint = data['constraint']
                num_yes = data['num_yes']
                constraint_scores[constraint] = num_yes
    
    return constraint_scores


def filter_constraints_in_file(input_file, output_file, constraint_scores, target_scores={9, 10}):
    """Filter constraints in a file based on num_yes scores."""
    filtered_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if line.strip():
                total_count += 1
                data = json.loads(line)
                constraint = data['constraint']
                
                # Check if this constraint has a score of 9 or 10
                if constraint in constraint_scores and constraint_scores[constraint] in target_scores:
                    outfile.write(line)
                    filtered_count += 1
    
    return filtered_count, total_count


def main(args):
    # Define paths
    constraints_file = Path(args.constraints_file)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading constraints from {constraints_file}...")
    constraint_scores = load_constraints_with_scores(constraints_file)
    print(f"Loaded {len(constraint_scores)} constraints with scores")
    
    # Count constraints with num_yes 9 or 10
    high_score_count = sum(1 for score in constraint_scores.values() if score in {9, 10})
    print(f"Found {high_score_count} constraints with num_yes = 9 or 10")
    
    # Process each file in constraints_analysis
    print(f"\nProcessing files from {input_dir}...")
    
    total_files = 0
    total_original = 0
    total_filtered = 0
    
    for input_file in sorted(input_dir.glob("*.jsonl")):
        output_file = output_dir / input_file.name
        
        filtered_count, original_count = filter_constraints_in_file(
            input_file, output_file, constraint_scores
        )
        
        total_files += 1
        total_original += original_count
        total_filtered += filtered_count
        
        print(f"  {input_file.name}: {original_count} -> {filtered_count} constraints")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files processed: {total_files}")
    print(f"  Total original constraints: {total_original}")
    print(f"  Total filtered constraints: {total_filtered}")
    print(f"  Retention rate: {total_filtered/total_original*100:.1f}%")
    print(f"\nFiltered constraints saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter constraints based on num_yes scores.")
    parser.add_argument('--constraints-file', type=str, required=True, help='Path to JSONL file with constraints.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory with JSONL files of constraints to filter.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output results.')
    args = parser.parse_args()

    main(args)
