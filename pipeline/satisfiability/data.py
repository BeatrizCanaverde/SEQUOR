import json
import random
from pathlib import Path
import argparse


# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def load_jsonl(file_path):
    """Load all lines from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def sample_constraints(constraints_file, num_constraints, output_file):
    """Sample random constraints from badwords_filtered.jsonl."""
    print(f"Loading constraints from {constraints_file}...")
    all_constraints = load_jsonl(constraints_file)
    print(f"Total constraints available: {len(all_constraints)}")
    
    if num_constraints > 0:
        # Sample random constraints
        sampled_constraints = random.sample(all_constraints, num_constraints)
    else:
        # If num_constraints is 0 or less, take all constraints
        sampled_constraints = all_constraints
    
    # Save to output file
    save_jsonl(sampled_constraints, output_file)
    
    return sampled_constraints


def sample_tasks(tasks_dir, num_tasks_per_file, output_file):
    """Sample tasks from each file in the tasks directory."""
    print(f"\nLoading tasks from {tasks_dir}...")
    
    # Get all JSONL files in the directory
    jsonl_files = sorted(tasks_dir.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    sampled_prompts = []
    
    for file_path in jsonl_files:
        # Load all lines from the file
        file_data = load_jsonl(file_path)
        
        if file_data:
            # Sample prompts from this file
            if num_tasks_per_file <= 0:
                # Sample all tasks from the file
                num_tasks = len(file_data)
            else:
                # Sample specified number of tasks
                num_tasks = min(num_tasks_per_file, len(file_data))
            
            sampled_lines = random.sample(file_data, num_tasks)
            sampled_prompts.extend(sampled_lines)
            print(f"  Sampled {num_tasks} tasks from {file_path.name}")
        
    # Save to output file
    save_jsonl(sampled_prompts, output_file)
    
    return sampled_prompts


def main(args):
    """Main execution function."""
    # Convert string paths to Path objects
    constraints_file = Path(args.constraints_file)
    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not constraints_file.exists():
        print(f"Error: Constraints file not found: {constraints_file}")
        return
    
    if not tasks_dir.exists():
        print(f"Error: Tasks directory not found: {tasks_dir}")
        return
    
    if not tasks_dir.is_dir():
        print(f"Error: Tasks path is not a directory: {tasks_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Sampling Constraints and Tasks for Evaluation")
    print("=" * 60)
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Sample constraints
    constraints_output = output_dir / "constraints.jsonl"
    constraints = sample_constraints(constraints_file, args.num_constraints, constraints_output)
    print(f"Total constraints sampled: {len(constraints)}")
    
    # Sample tasks
    tasks_output = output_dir / "tasks.jsonl"
    tasks = sample_tasks(tasks_dir, args.num_tasks_per_file, tasks_output)
    print(f"Total tasks sampled: {len(tasks)}")
    
    print()
    print("=" * 60)
    print("Sampling Complete!")
    print("=" * 60)
    print(f"Constraints saved to: {constraints_output}")
    print(f"Tasks saved to: {tasks_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample random constraints and tasks for evaluation.")
    parser.add_argument("--constraints-file", type=str, required=True, help="Path to the input constraints JSONL file.")
    parser.add_argument("--tasks-dir", type=str, required=True, help="Path to the directory containing task JSONL files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the directory where output files will be saved.")
    parser.add_argument("--num-constraints", type=int, default=200, help="Number of constraints to sample (0 = all constraints).")
    parser.add_argument("--num-tasks-per-file", type=int, default=1, help="Number of tasks to sample from each file (0 = all tasks).")
    args = parser.parse_args()
    main(args)