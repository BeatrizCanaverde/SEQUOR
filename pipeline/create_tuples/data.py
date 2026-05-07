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
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_tasks_from_dir(tasks_dir: Path, num_tasks_per_file: int = None) -> list:
    """Load tasks from all JSONL files in a directory, including source filename."""
    print(f"Loading tasks from {tasks_dir}...")
    
    # Get all JSONL files in the directory
    jsonl_files = sorted(tasks_dir.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Load tasks from each file
    all_tasks = []
    for file_path in jsonl_files:
        file_data = load_jsonl(file_path)
        
        # Take only the first N tasks from this file if specified
        if num_tasks_per_file:
            tasks_to_take = file_data[:num_tasks_per_file]
        else:
            tasks_to_take = file_data
        
        # Add source filename to each task
        for task in tasks_to_take:
            if isinstance(task, dict):
                task['source_file'] = file_path.name
            else:
                # If task is not a dict, wrap it in one
                task = {'data': task, 'source_file': file_path.name}
            all_tasks.append(task)
        
        print(f"  Loaded {len(tasks_to_take)} tasks from {file_path.name}")
    
    print(f"Total tasks loaded: {len(all_tasks)}")
    return all_tasks


def sample_and_save_tasks(all_tasks: list, num_tasks: int, output_file: Path):
    """Sample N tasks randomly and save to output file."""
    
    if len(all_tasks) < num_tasks:
        print(f"Warning: Not enough tasks available. Need {num_tasks}, have {len(all_tasks)}")
        print(f"Will use all available tasks.")
        num_tasks = len(all_tasks)
    
    # Sample unique tasks to ensure no duplicates
    sampled_tasks = random.sample(all_tasks, num_tasks)
    
    # Save to output file
    save_jsonl(sampled_tasks, output_file)
    print(f"Saved {len(sampled_tasks)} tasks to {output_file}")
    
    return sampled_tasks


def main(args):
    """Main execution function."""
    # Convert string paths to Path objects
    tasks_dir = Path(args.tasks_dir)
    output_file = Path(args.output_file)
    
    # Validate inputs
    if not tasks_dir.exists():
        print(f"Error: Tasks directory not found: {tasks_dir}")
        return
    
    if not tasks_dir.is_dir():
        print(f"Error: Tasks path is not a directory: {tasks_dir}")
        return
    
    print("=" * 60)
    print("Sampling Tasks for Evaluation")
    print("=" * 60)
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Load all tasks
    all_tasks = load_tasks_from_dir(tasks_dir, args.num_tasks_per_input_file)
    
    # Sample and save tasks
    sampled_tasks = sample_and_save_tasks(all_tasks, args.num_tasks, output_file)
    
    print()
    print("=" * 60)
    print("Task Sampling Complete!")
    print("=" * 60)
    print(f"Total tasks sampled: {len(sampled_tasks)}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample tasks randomly from a directory for evaluation.")
    parser.add_argument("--tasks-dir", type=str, required=True, help="Path to the directory containing task JSONL files.")
    parser.add_argument("--num-tasks", type=int, required=True, help="Number of tasks to sample.")
    parser.add_argument("--num-tasks-per-input-file", type=int, default=None, help="Number of tasks to extract from each input file (takes the first N). If not specified, takes all tasks from each file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output file where sampled tasks will be saved.")
    args = parser.parse_args()
    main(args)
