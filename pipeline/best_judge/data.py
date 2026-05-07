#!/usr/bin/env python3
"""
Script to generate task-constraint pairs for judge evaluation.

This script:
1. Randomly selects N task files from a tasks directory
2. Takes the first task from each selected file
3. Randomly selects N constraints from a constraints file
4. Randomly pairs tasks with constraints (each used once)
5. Generates prompts using two different templates
6. Outputs two JSONL files, each with the same pairs but different prompt formats
   - Template 1: Task followed by "Constraint: <constraint>"
   - Template 2: "Please follow this constraint: <constraint>" followed by task
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load all lines from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: Path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_first_tasks_from_files(tasks_dir: Path, num_prompts: int) -> List[Tuple[Dict, str]]:
    """
    Randomly select task files and extract the first task from each.
    
    Args:
        tasks_dir: Directory containing task JSONL files
        num_prompts: Number of task files to select
        
    Returns:
        List of tuples (task_dict, filename)
    """
    # Get all JSONL files in the directory
    jsonl_files = sorted(tasks_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {tasks_dir}")
    
    print(f"Found {len(jsonl_files)} JSONL files in {tasks_dir}")
    
    # Check if we have enough files
    if len(jsonl_files) < num_prompts:
        raise ValueError(
            f"Not enough task files. Need {num_prompts}, but only found {len(jsonl_files)}"
        )
    
    # Randomly select task files
    selected_files = random.sample(jsonl_files, num_prompts)
    print(f"Randomly selected {num_prompts} task files")
    
    # Extract the first task from each file
    tasks_with_filenames = []
    for file_path in selected_files:
        file_data = load_jsonl(file_path)
        if not file_data:
            print(f"Warning: File {file_path.name} is empty, skipping")
            continue
        
        first_task = file_data[0]
        tasks_with_filenames.append((first_task, file_path.name))
        print(f"  Loaded first task from {file_path.name}")
    
    if len(tasks_with_filenames) < num_prompts:
        raise ValueError(
            f"Could not load enough tasks. Expected {num_prompts}, got {len(tasks_with_filenames)}"
        )
    
    return tasks_with_filenames


def load_random_constraints(constraints_file: Path, num_constraints: int) -> List[str]:
    """
    Randomly select constraints from the constraints file.
    
    Args:
        constraints_file: Path to the constraints JSONL file
        num_constraints: Number of constraints to select
        
    Returns:
        List of constraint strings
    """
    all_constraints = load_jsonl(constraints_file)
    
    if not all_constraints:
        raise ValueError(f"No constraints found in {constraints_file}")
    
    print(f"\nLoaded {len(all_constraints)} constraints from {constraints_file}")
    
    # Check if we have enough constraints
    if len(all_constraints) < num_constraints:
        raise ValueError(
            f"Not enough constraints. Need {num_constraints}, but only found {len(all_constraints)}"
        )
    
    # Randomly select constraints
    selected_constraints_dicts = random.sample(all_constraints, num_constraints)
    
    # Extract just the constraint text (handle both 'constraint' and 'text' keys)
    constraints = []
    for c in selected_constraints_dicts:
        if 'constraint' in c:
            constraints.append(c['constraint'])
        elif 'text' in c:
            constraints.append(c['text'])
        else:
            raise ValueError(f"Constraint dictionary missing 'constraint' or 'text' key: {c}")
    
    print(f"Randomly selected {num_constraints} constraints")
    
    return constraints


def create_positive_template(task: Dict, constraint: str) -> str:
    """
    Create a prompt using positive template: Constraint followed.
    
    Args:
        task: Task dictionary containing 'prompt' field
        constraint: Constraint string
        
    Returns:
        Combined prompt string
    """
    # Get the task prompt
    task_prompt = task.get('prompt', '')
    
    positive_template = "Address the following task while adhering to the given constraint.\n\nConstraint:\n{constraint}\n\nTask:\n{task}"
    combined_prompt = positive_template.format(task=task_prompt, constraint=constraint)
    
    return combined_prompt


def create_negative_template(task: Dict, constraint: str) -> str:
    """
    Create a prompt using negative template: Constraint NOT followed.
    
    Args:
        task: Task dictionary containing 'prompt' field
        constraint: Constraint string
        
    Returns:
        Combined prompt string
    """
    # Get the task prompt
    task_prompt = task.get('prompt', '')
    
    negative_template = "Address the following task WITHOUT adhering to the given constraint. I repeat: your answer must NOT follow the constraint.\n\nConstraint:\n{constraint}\n\nTask:\n{task}"
    combined_prompt = negative_template.format(task=task_prompt, constraint=constraint)
    
    return combined_prompt


def generate_task_constraint_pairs(
    tasks_with_filenames: List[Tuple[Dict, str]],
    constraints: List[str],
    template_func
) -> List[Dict]:
    """
    Randomly pair tasks with constraints and generate prompts using a template function.
    
    Args:
        tasks_with_filenames: List of tuples (task_dict, filename)
        constraints: List of constraint strings
        template_func: Function to use for creating prompts
        
    Returns:
        List of dictionaries with prompt, task, filename, and constraint
    """
    # Shuffle both lists to create random pairings
    shuffled_tasks = tasks_with_filenames.copy()
    shuffled_constraints = constraints.copy()
    
    random.shuffle(shuffled_tasks)
    random.shuffle(shuffled_constraints)
    
    # Create pairs
    pairs = []
    for (task, filename), constraint in zip(shuffled_tasks, shuffled_constraints):
        prompt = template_func(task, constraint)
        
        pair = {
            'prompt': prompt,
            'task': task.get('prompt', ''),
            'filename': filename,
            'constraint': constraint
        }
        pairs.append(pair)
    
    return pairs


def main(args):
    
    # Convert to Path objects
    tasks_dir = Path(args.tasks_dir)
    constraints_file = Path(args.constraints_file)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not tasks_dir.exists():
        print(f"Error: Tasks directory not found: {tasks_dir}")
        return
    
    if not tasks_dir.is_dir():
        print(f"Error: Tasks path is not a directory: {tasks_dir}")
        return
    
    if not constraints_file.exists():
        print(f"Error: Constraints file not found: {constraints_file}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Generating Task-Constraint Pairs for Judge Evaluation")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Number of prompts: {args.num_prompts}")
    print()
    
    # Load tasks from random files
    print("Loading tasks...")
    tasks_with_filenames = load_first_tasks_from_files(tasks_dir, args.num_prompts)
    
    # Load random constraints
    print("\nLoading constraints...")
    constraints = load_random_constraints(constraints_file, args.num_prompts)
    
    # Create the shuffled pairings once (to ensure both templates use the same pairs)
    print("\nCreating random task-constraint pairings...")
    shuffled_tasks = tasks_with_filenames.copy()
    shuffled_constraints = constraints.copy()
    random.shuffle(shuffled_tasks)
    random.shuffle(shuffled_constraints)
    
    # Generate pairs with template 1
    print("Generating prompts with template 1...")
    positive_template = []
    for (task, filename), constraint in zip(shuffled_tasks, shuffled_constraints):
        prompt = create_positive_template(task, constraint)
        pair = {
            'prompt': prompt,
            'task': task.get('prompt', ''),
            'filename': filename,
            'constraint': constraint
        }
        positive_template.append(pair)
    print(f"Generated {len(positive_template)} pairs with template 1")
    
    # Generate pairs with template 2 (using same pairings, different prompt format)
    print("Generating prompts with template 2...")
    negative_template = []
    for (task, filename), constraint in zip(shuffled_tasks, shuffled_constraints):
        prompt = create_negative_template(task, constraint)
        pair = {
            'prompt': prompt,
            'task': task.get('prompt', ''),
            'filename': filename,
            'constraint': constraint
        }
        negative_template.append(pair)
    print(f"Generated {len(negative_template)} pairs with template 2")
    
    # Save to output files with default names
    output_file_positive = output_dir / 'positive.jsonl'
    output_file_negative = output_dir / 'negative.jsonl'
    
    save_jsonl(positive_template, output_file_positive)
    save_jsonl(negative_template, output_file_negative)
    
    print()
    print("=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print(f"Output file 1 (Template 1 - Task then Constraint): {output_file_positive}")
    print(f"Output file 2 (Template 2 - Constraint then Task): {output_file_negative}")
    print(f"Total pairs generated per file: {len(positive_template)}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate task-constraint pairs for judge evaluation')
    parser.add_argument('--tasks-dir', type=str, required=True, help='Path to the directory containing task JSONL files')
    parser.add_argument('--constraints-file', type=str, required=True, help='Path to the constraints JSONL file')
    parser.add_argument('--num-prompts', type=int, required=True, help='Number of prompts to create')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for the generated prompts')
    args = parser.parse_args()
    main(args)
