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


def sample_tasks(tasks_dir, num_tasks_per_file, output_dir):
    """Sample tasks from each file in the tasks directory and distribute across multiple output files."""
    print(f"\nLoading tasks from {tasks_dir}...")
    
    # Get all JSONL files in the directory
    jsonl_files = sorted(tasks_dir.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Create lists to hold tasks for each output file
    tasks_by_index = [[] for _ in range(num_tasks_per_file)]
    
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
            
            # Randomly sample tasks and shuffle them
            # sampled_lines = random.sample(file_data, num_tasks)
            # random.shuffle(sampled_lines)
            
            # Distribute sampled tasks randomly across output files
            for i, task in enumerate(file_data[:num_tasks]):
                tasks_by_index[i].append(task)
            
            print(f"  Sampled {num_tasks} tasks from {file_path.name}")
    
    # Save each set of tasks to its own file
    all_tasks = []
    for i, tasks in enumerate(tasks_by_index, start=1):
        output_file = output_dir / f"tasks_{i}.jsonl"
        save_jsonl(tasks, output_file)
        print(f"  Saved {len(tasks)} tasks to {output_file.name}")
        all_tasks.extend(tasks)
    
    return all_tasks


def main(args):
    """Main execution function."""
    constraints_file = Path(args.constraints_file)
    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Join constraints
    all_data = load_jsonl(constraints_file)

    output_file = output_dir / "constraints.jsonl"
    save_jsonl(all_data, output_file)
    
    print(f"\nTotal constraints processed: {len(all_data)}")
    print(f"Constraints saved to: {output_file}")

    # Sample tasks
    tasks = sample_tasks(tasks_dir, args.num_tasks_per_file, output_dir)
    
    print(f"\nTotal tasks sampled: {len(tasks)}")
    print(f"Tasks saved to: {output_dir}/tasks_*.jsonl")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect constraints and sample random tasks for evaluation.")
    parser.add_argument("--constraints-file", type=str, required=True, help="Path to the constraints JSONL file.")
    parser.add_argument("--tasks-dir", type=str, required=True, help="Path to the directory containing task JSONL files.")
    parser.add_argument("--num-tasks-per-file", type=int, default=2, help="Number of tasks to sample from each file (0 = all tasks).")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the directory where output files will be saved.")
    args = parser.parse_args()
    main(args)