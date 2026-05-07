import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import sys

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Palatino LT STD", "Book Antiqua", "Georgia", "serif"],
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 15,
    "figure.titlesize": 13
})


PLOT_COLOR = "#F7A8D9"


class DualLogger:
    """Logger that writes to both stdout and a file."""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, message):
        """Write message to both stdout and log file."""
        self.stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def close(self):
        """Close the log file."""
        self.log_file.close()


def load_jsonl(file_path):
    """Load all lines from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_judge_answer(judge_text):
    """
    Parse the judge answer in JSON format.
    Returns dict with keys 'question 1' through 'question 4' mapped to boolean values,
    or None if parsing fails.
    """
    try:
        judge_text = judge_text.strip()
        
        # First, try to find the JSON object in the text
        # Look for the first '{' and the last '}'
        start_idx = judge_text.find('{')
        end_idx = judge_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return None
        
        # Extract the JSON portion
        json_str = judge_text[start_idx:end_idx+1]
        
        # Strip markdown code fences if present
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        data = json.loads(json_str)
        
        # Extract answers for each question
        result = {}
        for i in range(1, 5):
            question_key = f"question {i}"
            answer_raw = data.get(question_key, "")
            
            # Convert to string and normalize
            if isinstance(answer_raw, str):
                answer = answer_raw.strip()
            else:
                answer = str(answer_raw).strip()
            
            # Remove brackets if present: "[[Yes]]" -> "Yes"
            answer = answer.replace('[', '').replace(']', '').strip().lower()
            
            if answer in ['yes', 'no']:
                result[question_key] = (answer == 'yes')
            else:
                return None
        
        return result
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        return None


def check_constraint_satisfies_all_questions(answers):
    """
    Check if a constraint satisfies all 4 questions:
    - question 1: Yes
    - question 2: No
    - question 3: Yes
    - question 4: Yes
    
    Returns True if all conditions are met, False otherwise.
    """
    if answers is None:
        return False
    
    return (
        answers.get('question 1', False) == True and
        answers.get('question 2', False) == False and
        answers.get('question 3', False) == True and
        answers.get('question 4', False) == True
    )


def load_canonical_tasks(tasks_file):
    """
    Load the canonical task order from tasks.jsonl.
    Returns a list of task prompts in their canonical order.
    """
    tasks = []
    with open(tasks_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                tasks.append(data['prompt'])
    return tasks


def load_model_data(file_path, canonical_tasks, num_tasks=100):
    """
    Load model outputs and organize by task and constraint.
    Uses canonical_tasks to determine task indices.
    Returns a dict: constraint -> task_idx -> satisfied (bool)
    """
    print(f"Loading {file_path}...")
    data = load_jsonl(file_path)
    
    # Build mapping from task text to canonical index
    task_to_idx = {task: idx for idx, task in enumerate(canonical_tasks[:num_tasks])}
    
    # Organize data by constraint and task
    constraint_task_map = defaultdict(dict)
    
    failed_parses = 0
    processed = 0
    skipped_tasks = 0
    
    for item in data:
        constraint = item['constraint']
        task = item['task']
        judge = item['judge']
        
        # Get task index from canonical order
        if task not in task_to_idx:
            skipped_tasks += 1
            continue  # Skip tasks not in the first num_tasks
        
        task_idx = task_to_idx[task]
        
        # Parse judge answer
        answers = parse_judge_answer(judge)
        if answers is None:
            failed_parses += 1
            continue
        
        # Check if constraint satisfies all questions for this task
        satisfied = check_constraint_satisfies_all_questions(answers)
        constraint_task_map[constraint][task_idx] = satisfied
        processed += 1
    
    print(f"  Processed {processed} entries ({failed_parses} failed to parse, {skipped_tasks} skipped)")
    print(f"  Using {num_tasks} tasks from canonical order")
    print(f"  Found {len(constraint_task_map)} unique constraints")
    
    return constraint_task_map


def compute_constraint_percentages(constraint_task_map, num_tasks):
    """
    For each constraint, compute the percentage of tasks for which it is satisfied.
    
    Args:
        constraint_task_map: Dict mapping constraint -> task_idx -> satisfied (bool)
        num_tasks: Total number of tasks to consider
    
    Returns:
        Dict mapping constraint -> percentage (0-100)
    """
    constraint_percentages = {}
    
    for constraint, task_results in constraint_task_map.items():
        # Count how many of the tasks are satisfied
        satisfied_tasks = 0
        for task_idx in range(num_tasks):
            if task_idx in task_results and task_results[task_idx]:
                satisfied_tasks += 1
        
        # Calculate percentage
        percentage = (satisfied_tasks / num_tasks) * 100.0
        constraint_percentages[constraint] = percentage
    
    return constraint_percentages


def compute_constraint_percentages_multiple_models(model_maps, num_tasks):
    """
    For constraints present in ALL models, compute the percentage of tasks satisfied in ALL models.
    A constraint-task pair is considered satisfied only if it's satisfied in ALL models.
    
    Args:
        model_maps: List of dicts, each mapping constraint -> task_idx -> satisfied (bool)
        num_tasks: Total number of tasks to consider
    
    Returns:
        Dict mapping constraint -> percentage (0-100)
    """
    if not model_maps:
        return {}
    
    # Get intersection of constraints across all models
    all_constraints = set(model_maps[0].keys())
    for model_map in model_maps[1:]:
        all_constraints &= set(model_map.keys())
    
    constraint_percentages = {}
    
    for constraint in all_constraints:
        # Count tasks where constraint is satisfied in ALL models
        satisfied_tasks = 0
        for task_idx in range(num_tasks):
            satisfied_in_all = True
            for model_map in model_maps:
                if task_idx not in model_map[constraint] or not model_map[constraint][task_idx]:
                    satisfied_in_all = False
                    break
            
            if satisfied_in_all:
                satisfied_tasks += 1
        
        # Calculate percentage
        percentage = (satisfied_tasks / num_tasks) * 100.0
        constraint_percentages[constraint] = percentage
    
    return constraint_percentages


def create_percentage_histogram(constraint_percentages, bin_size=10):
    """
    Create a histogram of constraints by their satisfaction percentage.
    
    Args:
        constraint_percentages: Dict mapping constraint -> percentage (0-100)
        bin_size: Size of each percentage bin (default: 10)
    
    Returns:
        Two lists: percentage_bins and counts
    """
    # Create bins: 0-10, 10-20, ..., 90-100
    bins = list(range(0, 101, bin_size))
    counts = [0] * len(bins)
    
    for constraint, percentage in constraint_percentages.items():
        # Find which bin this percentage belongs to
        bin_idx = min(int(percentage / bin_size), len(bins) - 1)
        counts[bin_idx] += 1
    
    return bins, counts


def compute_cumulative_counts(constraint_percentages):
    """Calculate cumulative counts: for each percentage, count constraints with percentage >= that value."""
    percentage_points = list(range(0, 101))  # All values from 0 to 100
    cumulative_counts = []
    
    for threshold in percentage_points:
        count = sum(1 for pct in constraint_percentages.values() if pct >= threshold)
        cumulative_counts.append(count)
    
    return percentage_points, cumulative_counts


def plot_single_model(model_data, model_name, output_dir, max_tasks=100, bin_size=5):
    """Plot histogram of constraints by satisfaction percentage for a single model."""
    print(f"\nCalculating percentage distribution for {model_name}...")
    
    # Compute percentage for each constraint
    constraint_percentages = compute_constraint_percentages(model_data, max_tasks)
    print(f"  Total constraints: {len(constraint_percentages)}")
    
    # Create histogram
    percentage_bins, counts = create_percentage_histogram(constraint_percentages, bin_size)
    
    # Print distribution
    for i, (pct, count) in enumerate(zip(percentage_bins, counts)):
        pct_end = pct + bin_size if i < len(percentage_bins) - 1 else 100
        print(f"  {pct}% - {pct_end}%: {count} constraints")
    
    # Create histogram plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        percentage_bins,
        counts,
        width=bin_size * 0.8,
        align='edge',
        edgecolor='black',
        alpha=0.7,
        color=PLOT_COLOR,
    )
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    str(count), ha='center', va='bottom', fontsize=13)
    
    plt.xlabel('Percentage of Tasks (%)', fontsize=13)
    plt.ylabel('Number of Constraints', fontsize=13)
    # plt.title(f'Distribution of Constraints by Task Satisfaction Percentage\n{model_name}', fontsize=13)
    plt.xticks(percentage_bins + [100], [f'{x}' for x in percentage_bins] + ['100'])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    # Save histogram to histograms subfolder
    histogram_dir = output_dir / 'histograms'
    histogram_dir.mkdir(parents=True, exist_ok=True)
    output_file = histogram_dir / f'constraints_percentage_histogram_{model_name.replace("/", "_")}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {output_file}")
    
    # Create cumulative plot
    percentage_points, cumulative_counts = compute_cumulative_counts(constraint_percentages)
    
    plt.figure(figsize=(12, 6))
    plt.plot(
        percentage_points,
        cumulative_counts,
        marker='o',
        linewidth=2,
        markersize=4,
        color=PLOT_COLOR,
    )
    
    # Add value labels at key points (every 10%)
    for x, y in zip(percentage_points, cumulative_counts):
        if x % 10 == 0:  # Label every 10%
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=13)
    
    plt.xlabel('Minimum Percentage of Tasks (%)', fontsize=13)
    plt.ylabel('Number of Constraints', fontsize=13)
    # plt.title(f'Cumulative: Constraints Satisfying ≥ X% of Tasks\n{model_name}', fontsize=13)
    plt.xticks(list(range(0, 101, 10)), [f'{x}' for x in range(0, 101, 10)])
    plt.grid(True, alpha=0.3)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    # Save cumulative plot to plots subfolder
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_file = plots_dir / f'constraints_cumulative_{model_name.replace("/", "_")}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative plot to {output_file}")
    
    return percentage_bins, counts


def plot_model_pair(model_maps, model_names, output_dir, max_tasks=100, bin_size=5):
    """Plot histogram of constraints by satisfaction percentage for a pair of models (intersection)."""
    pair_name = f"{model_names[0]} & {model_names[1]}"
    print(f"\nCalculating percentage distribution for pair: {pair_name}...")
    
    # Compute percentage for each constraint (must be satisfied in both models)
    constraint_percentages = compute_constraint_percentages_multiple_models(model_maps, max_tasks)
    print(f"  Total constraints (in both models): {len(constraint_percentages)}")
    
    # Create histogram
    percentage_bins, counts = create_percentage_histogram(constraint_percentages, bin_size)
    
    # Print distribution
    for i, (pct, count) in enumerate(zip(percentage_bins, counts)):
        pct_end = pct + bin_size if i < len(percentage_bins) - 1 else 100
        print(f"  {pct}% - {pct_end}%: {count} constraints")
    
    # Create histogram plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        percentage_bins,
        counts,
        width=bin_size * 0.8,
        align='edge',
        edgecolor='black',
        alpha=0.7,
        color=PLOT_COLOR,
    )
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    str(count), ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Percentage of Tasks (%)', fontsize=12)
    plt.ylabel('Number of Constraints', fontsize=12)
    plt.title(f'Distribution of Constraints by Task Satisfaction Percentage (Both Models)\n{pair_name}', fontsize=13)
    plt.xticks(percentage_bins + [100], [f'{x}' for x in percentage_bins] + ['100'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    safe_name = pair_name.replace("/", "_").replace(" ", "_").replace("&", "and")
    # Save histogram to histograms subfolder
    histogram_dir = output_dir / 'histograms'
    histogram_dir.mkdir(parents=True, exist_ok=True)
    output_file = histogram_dir / f'constraints_percentage_histogram_{safe_name}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {output_file}")
    
    # Create cumulative plot
    percentage_points, cumulative_counts = compute_cumulative_counts(constraint_percentages)
    
    plt.figure(figsize=(12, 6))
    plt.plot(
        percentage_points,
        cumulative_counts,
        marker='o',
        linewidth=2,
        markersize=4,
        color=PLOT_COLOR,
    )
    
    # Add value labels at key points (every 10%)
    for x, y in zip(percentage_points, cumulative_counts):
        if x % 10 == 0:  # Label every 10%
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Minimum Percentage of Tasks (%)', fontsize=12)
    plt.ylabel('Number of Constraints', fontsize=12)
    plt.title(f'Cumulative: Constraints Satisfying ≥ X% of Tasks (Both Models)\n{pair_name}', fontsize=13)
    plt.xticks(list(range(0, 101, 10)), [f'{x}' for x in range(0, 101, 10)])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save cumulative plot to plots subfolder
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_file = plots_dir / f'constraints_cumulative_{safe_name}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative plot to {output_file}")
    
    return percentage_bins, counts


def plot_all_models(model_maps, model_names, output_dir, max_tasks=100, bin_size=5):
    """Plot histogram of constraints by satisfaction percentage for all models (intersection)."""
    print(f"\nCalculating percentage distribution for all {len(model_names)} models...")
    
    # Compute percentage for each constraint (must be satisfied in all models)
    constraint_percentages = compute_constraint_percentages_multiple_models(model_maps, max_tasks)
    print(f"  Total constraints (in all models): {len(constraint_percentages)}")
    
    # Create histogram
    percentage_bins, counts = create_percentage_histogram(constraint_percentages, bin_size)
    
    # Print distribution
    for i, (pct, count) in enumerate(zip(percentage_bins, counts)):
        pct_end = pct + bin_size if i < len(percentage_bins) - 1 else 100
        print(f"  {pct}% - {pct_end}%: {count} constraints")
    
    # Create histogram plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(percentage_bins, counts, width=bin_size * 0.8, align='edge', 
                   edgecolor='black', alpha=0.7, color=PLOT_COLOR)
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    str(count), ha='center', va='bottom', fontsize=13)
    
    plt.xlabel('Percentage of Tasks (%)', fontsize=13)
    plt.ylabel('Number of Constraints', fontsize=13)
    # plt.title(f'Distribution of Constraints by Task Satisfaction Percentage (Both Models)\n{pair_name}', fontsize=13)
    plt.xticks(percentage_bins + [100], [f'{x}' for x in percentage_bins] + ['100'])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    # Save histogram to histograms subfolder
    histogram_dir = output_dir / 'histograms'
    histogram_dir.mkdir(parents=True, exist_ok=True)
    output_file = histogram_dir / f'constraints_percentage_histogram_all_models.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {output_file}")
    
    # Create cumulative plot
    percentage_points, cumulative_counts = compute_cumulative_counts(constraint_percentages)
    
    plt.figure(figsize=(12, 6))
    plt.plot(
        percentage_points,
        cumulative_counts,
        marker='o',
        linewidth=2,
        markersize=4,
        color=PLOT_COLOR,
    )
    
    # Add value labels at key points (every 10%)
    for x, y in zip(percentage_points, cumulative_counts):
        if x % 10 == 0:  # Label every 10%
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=13)
    
    plt.xlabel('Minimum Percentage of Tasks (%)', fontsize=13)
    plt.ylabel('Number of Constraints', fontsize=13)
    # plt.title(f'Cumulative: Constraints Satisfying ≥ X% of Tasks (Both Models)\n{pair_name}', fontsize=13)
    plt.xticks(list(range(0, 101, 10)), [f'{x}' for x in range(0, 101, 10)])
    plt.grid(True, alpha=0.3)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    # Save cumulative plot to plots subfolder
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_file = plots_dir / f'constraints_cumulative_all_models.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative plot to {output_file}")
    
    # Save constraints by percentage interval
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Group constraints by percentage bins
    bins_data = defaultdict(list)
    for constraint, percentage in constraint_percentages.items():
        bin_idx = min(int(percentage / bin_size), int(100 / bin_size) - 1)
        bin_start = bin_idx * bin_size
        bins_data[bin_start].append({
            'constraint': constraint,
            'percentage': percentage
        })
    
    # Save each bin to a separate file
    print(f"\nSaving constraints by {bin_size}% intervals...")
    for bin_start in sorted(bins_data.keys()):
        bin_end = bin_start + bin_size
        constraints_in_bin = bins_data[bin_start]
        
        output_file = data_dir / f'constraints_{bin_start:02d}_{bin_end:02d}_percent.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in constraints_in_bin:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  {bin_start}%-{bin_end}%: {len(constraints_in_bin)} constraints saved to {output_file.name}")
    
    print(f"\nAll constraint data saved to {data_dir}")
    
    # Save complete list of all constraints with percentages
    all_constraints_file = output_dir / 'all_constraints_percentages.jsonl'
    print(f"\nSaving complete list of all constraints with percentages...")
    
    # Sort by percentage (descending) for easier analysis
    sorted_constraints = sorted(constraint_percentages.items(), key=lambda x: x[1], reverse=True)
    
    with open(all_constraints_file, 'w', encoding='utf-8') as f:
        for constraint, percentage in sorted_constraints:
            f.write(json.dumps({
                'constraint': constraint,
                'percentage': percentage
            }, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(sorted_constraints)} constraints to {all_constraints_file}")
    
    return percentage_bins, counts


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file - redirect all stdout
    log_file_path = output_dir / 'analysis_log.txt'
    logger = DualLogger(log_file_path)
    
    # Redirect stdout to logger
    old_stdout = sys.stdout
    sys.stdout = logger
    
    print(f"Analysis started")
    print(f"Output directory: {output_dir}")
    print(f"Bin size: {args.bin_size}%")
    print(f"Number of tasks: {args.num_tasks}")
    print("="*60)
    
    # Load canonical task order
    print(f"\nLoading canonical task order from {args.tasks_file}...")
    canonical_tasks = load_canonical_tasks(args.tasks_file)
    print(f"Loaded {len(canonical_tasks)} tasks from canonical order")
    
    # Load all model files
    model_files = args.model_files
    model_names = [Path(f).stem for f in model_files]
    
    print(f"\nProcessing {len(model_files)} model files...")
    print(f"Model names: {model_names}")
    
    # Load data for each model
    model_data_list = []
    for file_path in model_files:
        model_data = load_model_data(file_path, canonical_tasks, num_tasks=args.num_tasks)
        model_data_list.append(model_data)
    
    # Plot individual models
    print("\n" + "="*60)
    print("PLOTTING INDIVIDUAL MODELS")
    print("="*60)
    for i, (model_data, model_name) in enumerate(zip(model_data_list, model_names)):
        plot_single_model(model_data, model_name, output_dir, max_tasks=args.num_tasks, bin_size=args.bin_size)
    
    # Plot pairs of models
    if len(model_files) >= 2:
        print("\n" + "="*60)
        print("PLOTTING MODEL PAIRS")
        print("="*60)
        for i, j in combinations(range(len(model_files)), 2):
            plot_model_pair(
                [model_data_list[i], model_data_list[j]],
                [model_names[i], model_names[j]],
                output_dir,
                max_tasks=args.num_tasks,
                bin_size=args.bin_size
            )
    
    # Plot all models together
    if len(model_files) >= 3:
        print("\n" + "="*60)
        print("PLOTTING ALL MODELS TOGETHER")
        print("="*60)
        plot_all_models(model_data_list, model_names, output_dir, max_tasks=args.num_tasks, bin_size=args.bin_size)
    
    print("\n" + "="*60)
    print("DONE!")
    print(f"Log saved to: {log_file_path}")
    print("="*60)
    
    # Restore stdout and close the logger
    sys.stdout = old_stdout
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histogram of constraints by their task satisfaction percentage for different models.")
    parser.add_argument('--model-files', type=str, nargs='+', required=True, help='Paths to model output JSONL files')
    parser.add_argument('--tasks-file', type=str, required=True, help='Path to the tasks.jsonl file with canonical task order')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output plots')
    parser.add_argument('--num-tasks', type=int, default=100, help='Number of tasks to analyze (default: 100)')
    parser.add_argument('--bin-size', type=int, default=5, help='Size of percentage bins for histogram (default: 5, giving bins of 0-5%%, 5-10%%, etc.)')
    args = parser.parse_args()
    main(args)
