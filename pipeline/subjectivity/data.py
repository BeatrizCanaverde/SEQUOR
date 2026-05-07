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


def save_constraints(constraints_file, output_file):
    """Save all constraints to output file."""
    print(f"Loading constraints from {constraints_file}...")
    all_constraints = load_jsonl(constraints_file)
    print(f"Total constraints available: {len(all_constraints)}")
    
    # Save all constraints to output file
    save_jsonl(all_constraints, output_file)
    print(f"Saved {len(all_constraints)} constraints to {output_file}")
    
    return all_constraints


def sample_and_split_tasks(tasks_dir, num_task_files, tasks_per_file, num_tasks_per_input_file, output_dir):
    """Sample tasks and split them into multiple files."""
    print(f"\nLoading tasks from {tasks_dir}...")
    
    # Get all JSONL files in the directory
    jsonl_files = sorted(tasks_dir.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Load the first N tasks from each file
    all_tasks = []
    for file_path in jsonl_files:
        file_data = load_jsonl(file_path)
        # Take only the first N tasks from this file
        tasks_to_take = file_data[:num_tasks_per_input_file]
        all_tasks.extend(tasks_to_take)
        print(f"  Loaded first {len(tasks_to_take)} tasks from {file_path.name}")
    
    print(f"Total tasks loaded: {len(all_tasks)}")
    
    # Calculate total tasks needed
    total_tasks_needed = num_task_files * tasks_per_file
    
    if len(all_tasks) < total_tasks_needed:
        print(f"Warning: Not enough tasks available. Need {total_tasks_needed}, have {len(all_tasks)}")
        print(f"Will use all available tasks and split evenly.")
        total_tasks_needed = len(all_tasks)
    
    # Sample unique tasks to ensure no duplicates
    sampled_tasks = random.sample(all_tasks, total_tasks_needed)
    
    # Split into separate files
    for i in range(num_task_files):
        start_idx = i * tasks_per_file
        end_idx = min((i + 1) * tasks_per_file, len(sampled_tasks))
        tasks_for_file = sampled_tasks[start_idx:end_idx]
        
        output_file = output_dir / f"tasks_{i+1}.jsonl"
        save_jsonl(tasks_for_file, output_file)
        print(f"Saved {len(tasks_for_file)} tasks to {output_file}")
    
    return sampled_tasks


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
    print("Preparing Constraints and Tasks for Subjectivity Evaluation")
    print("=" * 60)
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Save all constraints
    constraints_output = output_dir / "constraints.jsonl"
    constraints = save_constraints(constraints_file, constraints_output)
    
    # Sample and split tasks
    tasks = sample_and_split_tasks(tasks_dir, args.num_task_files, args.tasks_per_file, args.num_tasks_per_input_file, output_dir)
    
    print()
    print("=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"Total constraints: {len(constraints)}")
    print(f"Total tasks sampled: {len(tasks)}")
    print(f"Tasks split into {args.num_task_files} files with {args.tasks_per_file} tasks each")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare constraints and tasks for subjectivity evaluation.")
    parser.add_argument("--constraints-file", type=str, required=True, help="Path to the input constraints JSONL file.")
    parser.add_argument("--tasks-dir", type=str, required=True, help="Path to the directory containing task JSONL files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the directory where output files will be saved.")
    parser.add_argument("--num-task-files", type=int, default=4, help="Number of task files to create.")
    parser.add_argument("--tasks-per-file", type=int, default=25, help="Number of tasks per file.")
    parser.add_argument("--num-tasks-per-input-file", type=int, required=True, help="Number of tasks to extract from each input tasks file (takes the first N tasks).")
    args = parser.parse_args()
    main(args)
