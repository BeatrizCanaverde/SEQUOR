"""
Analyze judge outputs to find tuple-task pairs where all judges agree 
that all constraints are satisfied.
"""

import argparse
import json
import re
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


def extract_verdict(judge_output):
    """
    Extract verdict from judge output.
    Returns True if verdict is Yes, False if No, None if unclear.
    Take the LAST match if multiple verdicts exist.
    """
    # Look for "Final Verdict: [[Yes]]" or "Final Verdict: [[No]]"
    # Take the last match (use findall instead of search)
    matches = re.findall(r'Final Verdict:\s*\[\[(Yes|No)\]\]', judge_output, re.IGNORECASE)
    if matches:
        # Return the last match
        return matches[-1].lower() == 'yes'
    
    # If no match, try more lenient patterns (missing : or brackets)
    # Pattern without colon: "Final Verdict [[Yes]]"
    matches = re.findall(r'Final Verdict\s*\[\[(Yes|No)\]\]', judge_output, re.IGNORECASE)
    if matches:
        return matches[-1].lower() == 'yes'
    
    # Pattern without double brackets: "Final Verdict: [Yes]" or "Final Verdict Yes"
    matches = re.findall(r'Final Verdict:?\s*\[?(Yes|No)\]?', judge_output, re.IGNORECASE)
    if matches:
        return matches[-1].lower() == 'yes'
    
    return None


def create_tuple_task_key(constraints, task):
    """Create a unique key for a tuple-task pair."""
    # Sort constraints to ensure consistent keys
    constraints_str = '|||'.join(sorted(constraints))
    return f"{constraints_str}:::{task}"


def main(args):
    # Load judge outputs from all three judges
    judge_files = [args.judge1_file, args.judge2_file, args.judge3_file]
    judge_names = ['judge1', 'judge2', 'judge3']
    
    print(f"Loading judge outputs from {len(judge_files)} files...")
    
    judge_data_list = []
    for judge_file, judge_name in zip(judge_files, judge_names):
        if not Path(judge_file).exists():
            print(f"ERROR: Judge file not found: {judge_file}")
            return
        data = load_jsonl(judge_file)
        print(f"  {judge_name}: Loaded {len(data)} entries from {judge_file}")
        judge_data_list.append(data)
    
    # Ensure all judges have the same number of entries
    num_entries = len(judge_data_list[0])
    if not all(len(data) == num_entries for data in judge_data_list):
        print("ERROR: Judge files have different numbers of entries!")
        for i, data in enumerate(judge_data_list):
            print(f"  Judge {i+1}: {len(data)} entries")
        return
    
    print(f"\nAnalyzing {num_entries} entries...")
    
    # Track satisfied pairs and verdict statistics
    satisfied_pairs = []
    unsatisfied_pairs = []
    verdict_stats = {
        'total_constraints': 0,
        'verdicts_found': 0,
        'verdicts_missing': 0,
        'all_yes': 0,
        'some_no': 0
    }
    
    # Analyze each entry
    for idx in range(num_entries):
        # Get data from all judges (should be identical except for judge_outputs)
        entry_j1 = judge_data_list[0][idx]
        entry_j2 = judge_data_list[1][idx]
        entry_j3 = judge_data_list[2][idx]
        
        # Verify that task and constraints match across judges
        task = entry_j1['task']
        constraints = entry_j1['constraints']
        
        if (entry_j2['task'] != task or entry_j3['task'] != task or
            entry_j2['constraints'] != constraints or entry_j3['constraints'] != constraints):
            print(f"WARNING: Entry {idx} has mismatched tasks/constraints across judges!")
            continue
        
        # Check if all judges agree that all constraints are satisfied
        all_constraints_satisfied = True
        constraint_verdict_details = []
        
        for constraint_idx in range(len(constraints)):
            verdict_stats['total_constraints'] += 1
            
            # Get verdicts from all three judges for this constraint
            verdict_j1 = extract_verdict(entry_j1['judge_outputs'][constraint_idx])
            verdict_j2 = extract_verdict(entry_j2['judge_outputs'][constraint_idx])
            verdict_j3 = extract_verdict(entry_j3['judge_outputs'][constraint_idx])
            
            # Track verdict extraction success
            if verdict_j1 is not None and verdict_j2 is not None and verdict_j3 is not None:
                verdict_stats['verdicts_found'] += 1
            if verdict_j1 is None or verdict_j2 is None or verdict_j3 is None:
                verdict_stats['verdicts_missing'] += 1
                if idx < 5:  # Log first few cases for debugging
                    print(f"WARNING: Entry {idx}, constraint {constraint_idx}: Missing verdict")
                    print(f"  J1: {verdict_j1}, J2: {verdict_j2}, J3: {verdict_j3}")
            
            # Store verdict details for potential debugging
            constraint_verdict_details.append({
                'constraint': constraints[constraint_idx],
                'judge1': verdict_j1,
                'judge2': verdict_j2,
                'judge3': verdict_j3
            })
            
            # Check if all judges agree it's Yes
            if not (verdict_j1 is True and verdict_j2 is True and verdict_j3 is True):
                all_constraints_satisfied = False
        
        # Create tuple-task pair key
        pair_key = create_tuple_task_key(constraints, task)
        
        if all_constraints_satisfied:
            verdict_stats['all_yes'] += 1
            satisfied_pairs.append({
                'task': task,
                'constraints': constraints,
                'answer': entry_j1['answer'],
                'key': pair_key
            })
        else:
            verdict_stats['some_no'] += 1
            unsatisfied_pairs.append({
                'task': task,
                'constraints': constraints,
                'key': pair_key
            })
    
    print(f"\n=== Results ===")
    print(f"Total entries analyzed: {num_entries}")
    print(f"Satisfied pairs (all judges agree on all constraints): {len(satisfied_pairs)}")
    print(f"Unsatisfied pairs: {len(unsatisfied_pairs)}")
    print(f"\n=== Verdict Statistics ===")
    print(f"Total constraints evaluated: {verdict_stats['total_constraints']}")
    print(f"Verdicts successfully extracted: {verdict_stats['verdicts_found']} ({100*verdict_stats['verdicts_found']/max(1, verdict_stats['total_constraints']):.1f}%)")
    print(f"Verdicts missing/unclear: {verdict_stats['verdicts_missing']} ({100*verdict_stats['verdicts_missing']/max(1, verdict_stats['total_constraints']):.1f}%)")
    print(f"Entries with all constraints satisfied: {verdict_stats['all_yes']}")
    print(f"Entries with some constraints unsatisfied: {verdict_stats['some_no']}")
    
    # Save satisfied pairs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    satisfied_path = output_dir / 'satisfied_pairs.jsonl'
    with open(satisfied_path, 'w', encoding='utf-8') as f:
        for pair in satisfied_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"\nSaved satisfied pairs to: {satisfied_path}")
    
    # Save unsatisfied pairs (for reference)
    unsatisfied_path = output_dir / 'unsatisfied_pairs.jsonl'
    with open(unsatisfied_path, 'w', encoding='utf-8') as f:
        for pair in unsatisfied_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"Saved unsatisfied pairs to: {unsatisfied_path}")
    
    # Save just the keys for quick lookup (for filtering in next iteration)
    satisfied_keys_path = output_dir / 'satisfied_keys.txt'
    with open(satisfied_keys_path, 'w', encoding='utf-8') as f:
        for pair in satisfied_pairs:
            f.write(pair['key'] + '\n')
    
    print(f"Saved satisfied keys to: {satisfied_keys_path}")
    
    # Save verdict statistics summary
    stats_path = output_dir / 'verdict_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(verdict_stats, f, indent=2, ensure_ascii=False)
    
    print(f"Saved verdict statistics to: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze judge outputs to find satisfied tuple-task pairs.")
    parser.add_argument('--judge1-file', type=str, required=True, help='Path to first judge output JSONL file.')
    parser.add_argument('--judge2-file', type=str, required=True, help='Path to second judge output JSONL file.')
    parser.add_argument('--judge3-file', type=str, required=True, help='Path to third judge output JSONL file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save analysis results.')
    args = parser.parse_args()
    main(args)
