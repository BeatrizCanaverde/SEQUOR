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
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 15,
    "figure.titlesize": 13
})


PLOT_COLOR = "#FF667A"


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
    
    def flush(self):
        """Flush the log file."""
        self.log_file.flush()
        self.stdout.flush()
    
    def close(self):
        """Close the log file."""
        self.log_file.close()


def load_jsonl(file_path):
    """Load all lines from a JSONL file, skipping malformed JSON."""
    data = []
    failed_json_parses = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    failed_json_parses += 1
                    # Print warning for first few failures
                    if failed_json_parses <= 3:
                        print(f"  WARNING: Skipping malformed JSON at line {line_num}: {e}")
    if failed_json_parses > 0:
        print(f"  Skipped {failed_json_parses} malformed JSON line(s)")
    return data


def parse_judge_verdict(judge_text):
    """
    Parse the judge verdict from text.
    Looks for patterns like "Final Verdict: [[Yes]]" or "Final Verdict: [[No]]"
    Picks the LAST occurrence if there are multiple.
    Returns True for Yes, False for No, None if parsing fails.
    """
    try:
        judge_text_lower = judge_text.strip().lower()
        
        # Pattern 1: Look for "final verdict: [[yes]]" or "final verdict: [[no]]"
        # Use rfind to get the LAST occurrence
        last_final_verdict_idx = judge_text_lower.rfind("final verdict:")
        if last_final_verdict_idx != -1:
            # Extract the section after "final verdict:"
            verdict_section = judge_text_lower[last_final_verdict_idx:]
            if "[[yes]]" in verdict_section:
                return True
            elif "[[no]]" in verdict_section:
                return False
        
        # Pattern 2: Just look for "[[yes]]" or "[[no]]" (last occurrence)
        if "[[yes]]" in judge_text_lower or "[[no]]" in judge_text_lower:
            # Find the last occurrence of each
            last_yes_idx = judge_text_lower.rfind("[[yes]]")
            last_no_idx = judge_text_lower.rfind("[[no]]")
            
            # Return the verdict that appears last in the text
            if last_no_idx > last_yes_idx:
                return False
            elif last_yes_idx > last_no_idx:
                return True
        
        # If no pattern matches, return None
        return None
    except Exception as e:
        return None


def load_judge_data_from_folder(judge_folder):
    """
    Load all judge outputs from a judge's folder and organize by constraint and task.
    A judge may have multiple model output files.
    Returns a dict: constraint -> task -> verdict (bool or None)
    """
    judge_folder = Path(judge_folder)
    judge_name = judge_folder.name
    print(f"\nLoading data for judge: {judge_name}")
    
    # Find all .jsonl files in this judge's folder
    jsonl_files = list(judge_folder.glob("*.jsonl"))
    print(f"  Found {len(jsonl_files)} JSONL files")
    
    # Organize data by constraint and task
    constraint_task_map = defaultdict(dict)
    
    failed_parses = 0
    processed = 0
    
    for jsonl_file in jsonl_files:
        print(f"  Loading {jsonl_file.name}...")
        data = load_jsonl(jsonl_file)
        
        for item in data:
            constraint = item['constraint']
            task = item['task']
            judge_text = item['judge']
            
            # Parse judge verdict
            verdict = parse_judge_verdict(judge_text)
            if verdict is None:
                failed_parses += 1
                continue
            
            # Store verdict (if same constraint-task pair appears multiple times,
            # the last one will be kept)
            constraint_task_map[constraint][task] = verdict
            processed += 1
    
    print(f"  Total processed: {processed} entries ({failed_parses} failed to parse)")
    print(f"  Found {len(constraint_task_map)} unique constraints for {judge_name}")
    
    return constraint_task_map


def compute_unanimous_percentages(judge_data_list):
    """
    For each constraint, compute the percentage of tasks where all judges agree.
    Agreement means either all "Yes" or all "No".
    
    Args:
        judge_data_list: List of dicts from each judge, mapping constraint -> task -> verdict (bool)
    
    Returns:
        Dict mapping constraint -> percentage (0-100)
    """
    if not judge_data_list:
        return {}
    
    num_judges = len(judge_data_list)
    print(f"\nCalculating unanimous agreement percentages across {num_judges} judges...")
    
    # Get intersection of constraints across all judges
    all_constraints = set(judge_data_list[0].keys())
    for judge_data in judge_data_list[1:]:
        all_constraints &= set(judge_data.keys())
    
    print(f"Found {len(all_constraints)} constraints present in all {num_judges} judges")
    
    constraint_percentages = {}
    
    for constraint in all_constraints:
        # Get all unique tasks for this constraint across all judges
        all_tasks = set()
        for judge_data in judge_data_list:
            all_tasks.update(judge_data[constraint].keys())
        
        # Count tasks where all judges agree
        unanimous_tasks = 0
        total_tasks = 0
        
        for task in all_tasks:
            # Get verdicts from all judges for this task
            verdicts = []
            all_judges_have_task = True
            
            for judge_data in judge_data_list:
                if task in judge_data[constraint]:
                    verdicts.append(judge_data[constraint][task])
                else:
                    all_judges_have_task = False
                    break
            
            # Only count if all judges evaluated this task
            if all_judges_have_task:
                total_tasks += 1
                # Check if all verdicts are the same (unanimous)
                if len(set(verdicts)) == 1:  # All verdicts are identical
                    unanimous_tasks += 1
        
        # Calculate percentage
        if total_tasks > 0:
            percentage = (unanimous_tasks / total_tasks) * 100.0
            constraint_percentages[constraint] = percentage
        else:
            # Skip constraints with no valid tasks
            continue
    
    print(f"Calculated percentages for {len(constraint_percentages)} constraints")
    
    return constraint_percentages


def compute_cumulative_counts(constraint_percentages):
    """
    Calculate cumulative counts: for each percentage threshold, 
    count constraints with percentage >= that threshold.
    """
    percentage_points = list(range(0, 101))  # All values from 0 to 100
    cumulative_counts = []
    
    for threshold in percentage_points:
        count = sum(1 for pct in constraint_percentages.values() if pct >= threshold)
        cumulative_counts.append(count)
    
    return percentage_points, cumulative_counts


def create_histogram(constraint_percentages, bin_size=10):
    """
    Create histogram data for constraints by their unanimous agreement percentage.
    
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


def plot_unanimous_agreement(constraint_percentages, output_dir, bin_size=5):
    """
    Create plots showing the distribution of constraints by unanimous agreement percentage.
    """
    if not constraint_percentages:
        print("No data to plot!")
        return
    
    print(f"\nCreating plots...")
    print(f"Total constraints: {len(constraint_percentages)}")
    
    # Create histogram
    percentage_bins, counts = create_histogram(constraint_percentages, bin_size)
    
    # Print distribution
    print("\nDistribution of constraints by unanimous agreement percentage:")
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
    # plt.title('Distribution of Constraints by Judge Agreement Percentage', fontsize=13)
    plt.xticks(percentage_bins + [100], [f'{x}' for x in percentage_bins] + ['100'])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    # Save histogram
    histogram_dir = output_dir / 'histograms'
    histogram_dir.mkdir(parents=True, exist_ok=True)
    output_file = histogram_dir / 'judge_agreement_histogram.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {output_file}")
    
    # Create cumulative plot
    percentage_points, cumulative_counts = compute_cumulative_counts(constraint_percentages)
    
    plt.figure(figsize=(12, 6))
    plt.plot(percentage_points, cumulative_counts, marker='o', linewidth=2, 
             markersize=4, color=PLOT_COLOR)
    
    # Add value labels at key points (every 10%)
    for x, y in zip(percentage_points, cumulative_counts):
        if x % 10 == 0:  # Label every 10%
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=13)
    
    plt.xlabel('Minimum Percentage of Tasks (%)', fontsize=13)
    plt.ylabel('Number of Constraints', fontsize=13)
    # plt.title('Cumulative: Constraints with ≥ X% Unanimous Judge Agreement', fontsize=13)
    plt.xticks(list(range(0, 101, 10)), [f'{x}' for x in range(0, 101, 10)])
    plt.grid(True, alpha=0.3)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    # Save cumulative plot
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_file = plots_dir / 'judge_agreement_cumulative.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative plot to {output_file}")
    
    # Save constraint data by percentage intervals
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


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file - redirect all stdout
    log_file_path = output_dir / 'analysis_log.txt'
    logger = DualLogger(log_file_path)
    
    # Redirect stdout to logger
    old_stdout = sys.stdout
    sys.stdout = logger
    
    print("="*60)
    print("JUDGE AGREEMENT ANALYSIS")
    print("="*60)
    print(f"Judge folder: {args.judge_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Bin size: {args.bin_size}%")
    print("="*60)
    
    # Find all judge subfolders (ignore 'logs' folder)
    judge_folder = Path(args.judge_folder)
    if not judge_folder.exists():
        print(f"Error: Judge folder not found: {judge_folder}")
        sys.stdout = old_stdout
        logger.close()
        return
    
    # Get all subdirectories except 'logs'
    judge_subfolders = [d for d in judge_folder.iterdir() 
                        if d.is_dir() and d.name.lower() != 'logs']
    judge_subfolders.sort()  # Sort for consistent ordering
    
    print(f"\nFound {len(judge_subfolders)} judge folders:")
    for folder in judge_subfolders:
        print(f"  - {folder.name}")
    
    if len(judge_subfolders) == 0:
        print(f"Error: No judge folders found in {judge_folder}")
        sys.stdout = old_stdout
        logger.close()
        return
    
    # Load all judge data
    judge_data_list = []
    for judge_subfolder in judge_subfolders:
        judge_data = load_judge_data_from_folder(judge_subfolder)
        judge_data_list.append(judge_data)
    
    # Compute unanimous agreement percentages
    constraint_percentages = compute_unanimous_percentages(judge_data_list)
    
    # Create plots
    plot_unanimous_agreement(constraint_percentages, output_dir, bin_size=args.bin_size)
    
    print("\n" + "="*60)
    print("DONE!")
    print(f"Log saved to: {log_file_path}")
    print("="*60)
    
    # Restore stdout and close the logger
    sys.stdout = old_stdout
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze judge agreement on constraint-task pairs in subjectivity analysis.")
    parser.add_argument('--judge-folder', type=str, required=True, help='Path to folder containing judge subfolders (each subfolder represents a judge with verdict files)')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output plots and data')
    parser.add_argument('--bin-size', type=int, default=5, help='Size of percentage bins for histogram (default: 5, giving bins of 0-5%%, 5-10%%, etc.)')
    args = parser.parse_args()
    main(args)
