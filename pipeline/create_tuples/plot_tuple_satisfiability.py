import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import sys

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Palatino LT STD", "Book Antiqua", "Georgia", "serif"],
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "figure.titlesize": 13
})

PLOT_COLOR = "#4ECDC4"


class DualLogger:
    """Logger that writes to both stdout and a log file."""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, message):
        """Write message to both stdout and log file."""
        self.stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        """Flush the log file."""
        self.log_file.flush()
        self.stdout.flush()
    
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


def extract_constraint_tuple_key(constraints):
    """
    Create a unique key for a constraint tuple.
    Constraints are sorted to ensure consistent keys.
    """
    return '|||'.join(sorted(constraints))


def analyze_tuple_satisfiability(satisfied_pairs):
    """
    Analyze how many tasks per constraint tuple have at least one satisfying answer.
    
    Args:
        satisfied_pairs: List of dicts with 'task', 'constraints', 'answer', 'key'
    
    Returns:
        tuple_task_counts: Dict mapping tuple_key -> set of unique tasks
    """
    tuple_task_map = defaultdict(set)
    
    for item in satisfied_pairs:
        task = item['task']
        constraints = item['constraints']
        
        # Create tuple key
        tuple_key = extract_constraint_tuple_key(constraints)
        
        # Add this task to the set for this tuple
        tuple_task_map[tuple_key].add(task)
    
    return tuple_task_map


def compute_tuple_percentages(tuple_task_map, expected_tasks_per_tuple=100):
    """
    For each constraint tuple, compute the percentage of tasks with at least one satisfying answer.
    
    Args:
        tuple_task_map: Dict mapping tuple_key -> set of unique tasks
        expected_tasks_per_tuple: Expected number of tasks per tuple (default: 100)
    
    Returns:
        Dict mapping tuple_key -> percentage (0-100)
    """
    tuple_percentages = {}
    
    for tuple_key, tasks in tuple_task_map.items():
        num_satisfied_tasks = len(tasks)
        percentage = (num_satisfied_tasks / expected_tasks_per_tuple) * 100.0
        tuple_percentages[tuple_key] = percentage
    
    return tuple_percentages


def create_histogram(tuple_percentages, bin_size=10):
    """
    Create histogram data for tuples by their task satisfaction percentage.
    
    Args:
        tuple_percentages: Dict mapping tuple_key -> percentage (0-100)
        bin_size: Size of each percentage bin (default: 10)
    
    Returns:
        Two lists: percentage_bins and counts
    """
    # Create bins: 0-10, 10-20, ..., 90-100
    bins = list(range(0, 101, bin_size))
    counts = [0] * len(bins)
    
    for tuple_key, percentage in tuple_percentages.items():
        # Find which bin this percentage belongs to
        bin_idx = min(int(percentage / bin_size), len(bins) - 1)
        counts[bin_idx] += 1
    
    return bins, counts


def compute_cumulative_counts(tuple_percentages):
    """
    Calculate cumulative counts: for each percentage threshold, 
    count tuples with percentage >= that threshold.
    """
    percentage_points = list(range(0, 101))  # All values from 0 to 100
    cumulative_counts = []
    
    for threshold in percentage_points:
        count = sum(1 for pct in tuple_percentages.values() if pct >= threshold)
        cumulative_counts.append(count)
    
    return percentage_points, cumulative_counts


def plot_tuple_satisfiability(tuple_percentages, output_dir, bin_size=5):
    """
    Create plots showing the distribution of constraint tuples by task satisfaction percentage.
    """
    if not tuple_percentages:
        print("No data to plot!")
        return
    
    print(f"\nCreating plots...")
    print(f"Total constraint tuples: {len(tuple_percentages)}")
    
    # Create histogram
    percentage_bins, counts = create_histogram(tuple_percentages, bin_size)
    
    # Print distribution
    print("\nDistribution of constraint tuples by task satisfaction percentage:")
    for i, (pct, count) in enumerate(zip(percentage_bins, counts)):
        pct_end = pct + bin_size if i < len(percentage_bins) - 1 else 100
        print(f"  {pct}% - {pct_end}%: {count} tuples")
    
    # Create histogram plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(percentage_bins, counts, width=bin_size * 0.8, align='edge', 
                   edgecolor='black', alpha=0.7, color=PLOT_COLOR)
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    str(count), ha='center', va='bottom', fontsize=11)
    
    plt.xlabel('Percentage of Tasks (%)')
    plt.ylabel('Number of Tuples')
    plt.xticks(percentage_bins + [100], [f'{x}' for x in percentage_bins] + ['100'])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save histogram
    histogram_dir = output_dir / 'histograms'
    histogram_dir.mkdir(parents=True, exist_ok=True)
    output_file = histogram_dir / 'tuple_satisfiability_histogram.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {output_file}")
    
    # Create cumulative plot
    percentage_points, cumulative_counts = compute_cumulative_counts(tuple_percentages)
    
    plt.figure(figsize=(12, 6))
    plt.plot(percentage_points, cumulative_counts, marker='o', linewidth=2, 
             markersize=4, color=PLOT_COLOR)
    
    # Add value labels at key points (every 10%)
    for x, y in zip(percentage_points, cumulative_counts):
        if x % 10 == 0:  # Label every 10%
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=11)
    
    plt.xlabel('Minimum Percentage of Tasks (%)')
    plt.ylabel('Number of Tuples')
    plt.xticks(list(range(0, 101, 10)), [f'{x}' for x in range(0, 101, 10)])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save cumulative plot
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_file = plots_dir / 'tuple_satisfiability_cumulative.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative plot to {output_file}")
    
    # Save tuple data by percentage intervals
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Group tuples by percentage bins
    bins_data = defaultdict(list)
    for tuple_key, percentage in tuple_percentages.items():
        bin_idx = min(int(percentage / bin_size), int(100 / bin_size) - 1)
        bin_start = bin_idx * bin_size
        # Split tuple key back into constraints
        constraints = tuple_key.split('|||')
        bins_data[bin_start].append({
            'constraints': constraints,
            'percentage': percentage
        })
    
    # Save each bin to a separate file
    print(f"\nSaving constraint tuples by {bin_size}% intervals...")
    for bin_start in sorted(bins_data.keys()):
        bin_end = bin_start + bin_size
        tuples_in_bin = bins_data[bin_start]
        
        output_file = data_dir / f'tuples_{bin_start:02d}_{bin_end:02d}_percent.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tuples_in_bin:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  {bin_start}%-{bin_end}%: {len(tuples_in_bin)} tuples saved to {output_file.name}")
    
    print(f"\nAll tuple data saved to {data_dir}")
    
    # Save complete list of all tuples with percentages
    all_tuples_file = output_dir / 'all_tuples_percentages.jsonl'
    print(f"\nSaving complete list of all tuples with percentages...")
    
    # Sort by percentage (descending) for easier analysis
    sorted_tuples = sorted(tuple_percentages.items(), key=lambda x: x[1], reverse=True)
    
    with open(all_tuples_file, 'w', encoding='utf-8') as f:
        for tuple_key, percentage in sorted_tuples:
            constraints = tuple_key.split('|||')
            f.write(json.dumps({
                'constraints': constraints,
                'percentage': percentage
            }, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(sorted_tuples)} tuples to {all_tuples_file}")


def main(args):
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up dual logger
    log_file = output_dir / 'analysis_log.txt'
    logger = DualLogger(log_file)
    sys.stdout = logger
    
    print("="*80)
    print("Constraint Tuple Satisfiability Analysis")
    print("="*80)
    print(f"Input file: {args.satisfied_pairs_file}")
    print(f"Output directory: {output_dir}")
    print(f"Expected tasks per tuple: {args.expected_tasks_per_tuple}")
    print(f"Histogram bin size: {args.bin_size}%")
    print()
    
    # Load satisfied pairs data
    print("Loading satisfied pairs data...")
    satisfied_pairs = load_jsonl(args.satisfied_pairs_file)
    print(f"Loaded {len(satisfied_pairs)} satisfied pairs")
    
    # Analyze tuple satisfiability
    print("\nAnalyzing tuple satisfiability...")
    tuple_task_map = analyze_tuple_satisfiability(satisfied_pairs)
    print(f"Found {len(tuple_task_map)} unique constraint tuples")
    
    # Compute percentages
    print("\nComputing satisfaction percentages...")
    tuple_percentages = compute_tuple_percentages(tuple_task_map, args.expected_tasks_per_tuple)
    
    # Print some statistics
    print(f"\nStatistics:")
    percentages_list = list(tuple_percentages.values())
    print(f"  Min percentage: {min(percentages_list):.2f}%")
    print(f"  Max percentage: {max(percentages_list):.2f}%")
    print(f"  Mean percentage: {sum(percentages_list)/len(percentages_list):.2f}%")
    print(f"  Median percentage: {sorted(percentages_list)[len(percentages_list)//2]:.2f}%")
    
    # Count tuples at key thresholds
    print(f"\nTuples with at least X% tasks satisfied:")
    for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        count = sum(1 for pct in percentages_list if pct >= threshold)
        print(f"  ≥{threshold}%: {count} tuples ({count/len(percentages_list)*100:.1f}%)")
    
    # Create plots
    plot_tuple_satisfiability(tuple_percentages, output_dir, bin_size=args.bin_size)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    # Close logger
    sys.stdout = logger.stdout
    logger.close()
    print(f"Log saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze constraint tuple satisfiability across tasks.")
    parser.add_argument('--satisfied-pairs-file', type=str, required=True, help='Path to cumulative_satisfied_pairs.jsonl file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output plots and data')
    parser.add_argument('--expected-tasks-per-tuple', type=int, default=100, help='Expected number of tasks per constraint tuple (default: 100)')
    parser.add_argument('--bin-size', type=int, default=5, help='Bin size for histogram in percentage points (default: 5)')   
    args = parser.parse_args()
    main(args)
