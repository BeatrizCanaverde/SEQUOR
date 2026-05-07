import json
import os
from collections import defaultdict
import re
import argparse
from pathlib import Path


def load_jsonl(file_path):
    """Load all lines from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_final_answer(judge_text):
    """
    Extract the final verdict (Yes or No) from the judge text.
    Returns 'yes' or 'no' (lowercase) or None if not found.
    """
    # Look for pattern "Final Verdict: [[Yes]]" or "Final Verdict: [[No]]"
    matches = re.findall(r'\*?\*?Final Verdict\*?\*?:?\s*\[\[(?:Yes|No)\]\]', judge_text, re.IGNORECASE)
    if matches:
        # Take the last occurrence
        answer = matches[-1]
        if 'Yes' in answer or 'yes' in answer:
            return 'yes'
        elif 'No' in answer or 'no' in answer:
            return 'no'
    return None


def process_judge_outputs(data):
    """
    Process judge outputs to count no answers per constraint.
    Returns tuple of (constraint_no_counts, all_constraints).
    """
    # Dictionary to count no answers per constraint
    constraint_no_counts = defaultdict(int)
    
    # Dictionary to store all constraints (to avoid duplicates)
    all_constraints = set()
    
    # Counter for missing answers
    missing_answers = 0
    
    # Read the input file and process each line
    print("Processing judge outputs...")
    for line_num, item in enumerate(data, 1):
                
        constraint = item['constraint']
        judge = item['judge']
        
        # Add constraint to the set
        all_constraints.add(constraint)
        
        # Extract final verdict
        final_verdict = extract_final_answer(judge)
        
        if final_verdict == 'no':
            constraint_no_counts[constraint] += 1
        elif final_verdict is None:
            missing_answers += 1
            print(f"Warning: Could not extract final verdict from line {line_num}")
    
    print(f"Total unique constraints found: {len(all_constraints)}")
    print(f"Could not extract final verdict from {missing_answers} lines")

    # Group constraints by their no count (0 to 10)
    constraints_by_count = defaultdict(list)
    
    for constraint in all_constraints:
        no_count = constraint_no_counts[constraint]
        constraints_by_count[no_count].append(constraint)
    
    return constraints_by_count


def main(args):

    # Load data
    data = load_jsonl(args.input_file)
    
    # Output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process judge outputs to count no answers per constraint
    # Group constraints by their no count (0 to 10)
    constraints_by_count = process_judge_outputs(data)
    
    # Create output files for each count (0 to 10)
    print("\nCreating output files...")
    for count in range(11):  # 0 to 10
        output_file = output_dir / f"{count}.jsonl"
        constraints_with_count = constraints_by_count[count]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for constraint in sorted(constraints_with_count):  # Sort for consistency
                json.dump({'constraint': constraint}, f)
                f.write('\n')
        
        print(f"Created {output_file} with {len(constraints_with_count)} constraints")
    
    # Print summary statistics
    print("\n=== Summary ===")
    for count in range(11):
        num_constraints = len(constraints_by_count[count])
        print(f"Constraints with {count} no answers: {num_constraints}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze judge outputs to count no answers per constraint.")
    parser.add_argument('--input-file', type=str, required=True, help='Path to JSONL file with judge outputs.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output results.')
    args = parser.parse_args()

    main(args)
