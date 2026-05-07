import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import sys


PLOT_COLOR = "#4DE722"


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
    Load all judge outputs from a judge's folder and organize by constraint tuple.
    Each constraint tuple has a list of verdicts (one per task, 100 tasks total).
    Returns a dict: tuple_of_constraints -> list of verdicts (True/False/None for each task)
    """
    judge_folder = Path(judge_folder)
    judge_name = judge_folder.name
    print(f"\nLoading data for judge: {judge_name}")
    
    # Find all _processed.jsonl files in this judge's folder
    jsonl_files = list(judge_folder.glob("*_processed.jsonl"))
    print(f"  Found {len(jsonl_files)} processed JSONL files")
    
    # Organize data by constraint tuple
    tuple_verdicts_map = {}
    
    failed_parses = 0
    processed = 0
    
    for jsonl_file in jsonl_files:
        print(f"  Loading {jsonl_file.name}...")
        data = load_jsonl(jsonl_file)
        
        for item in data:
            # constraints is a list of strings (the tuple)
            constraints = item['constraints']
            # model_outputs is a list of "Yes"/"No"/null (one per task)
            model_outputs = item.get('model_outputs', [])
            
            # Convert model outputs to boolean verdicts
            verdicts = []
            for output in model_outputs:
                if output is None:
                    verdicts.append(None)
                    failed_parses += 1
                elif isinstance(output, str):
                    if output.lower().startswith("yes"):
                        verdicts.append(True)
                        processed += 1
                    elif output.lower().startswith("no"):
                        verdicts.append(False)
                        processed += 1
                    else:
                        verdicts.append(None)
                        failed_parses += 1
                else:
                    verdicts.append(None)
                    failed_parses += 1
            
            # Use tuple as key (hashable)
            constraint_tuple = tuple(constraints)
            
            # Store verdict list
            tuple_verdicts_map[constraint_tuple] = verdicts
    
    print(f"  Total processed: {processed} verdicts ({failed_parses} None or failed to parse)")
    print(f"  Found {len(tuple_verdicts_map)} unique constraint tuples for {judge_name}")
    
    return tuple_verdicts_map


def compute_judge_yes_counts(judge_data_list):
    """
    For each constraint tuple, compute the absolute count of "Yes" votes for each judge.
    This matches the methodology used in compare_3_processed_outputs.py.
    
    Args:
        judge_data_list: List of dicts from each judge, mapping constraint_tuple -> list of verdicts
    
    Returns:
        Dict mapping constraint_tuple -> dict of {judge_idx: yes_count}
    """
    if not judge_data_list:
        return {}
    
    num_judges = len(judge_data_list)
    print(f"\nCalculating Yes vote counts for {num_judges} judges...")
    
    # Get intersection of constraint tuples across all judges
    all_tuples = set(judge_data_list[0].keys())
    for judge_data in judge_data_list[1:]:
        all_tuples &= set(judge_data.keys())
    
    print(f"Found {len(all_tuples)} constraint tuples present in all {num_judges} judges")
    
    tuple_judge_yes_counts = {}
    
    for constraint_tuple in all_tuples:
        judge_yes_counts = {}
        
        for judge_idx, judge_data in enumerate(judge_data_list):
            verdicts = judge_data[constraint_tuple]
            
            # Count Yes votes (matching compare_3_processed_outputs.py methodology)
            # This counts absolute Yes votes, treating None as not-Yes
            yes_count = sum(1 for v in verdicts if v is True)
            judge_yes_counts[judge_idx] = yes_count
        
        tuple_judge_yes_counts[constraint_tuple] = judge_yes_counts
    
    print(f"Calculated Yes vote counts for {len(tuple_judge_yes_counts)} constraint tuples")
    
    return tuple_judge_yes_counts


def compute_minimum_yes_counts(tuple_judge_yes_counts):
    """
    For each constraint tuple, compute the minimum Yes vote count across all judges.
    This represents the "bottleneck" judge with the lowest Yes vote count.
    
    Args:
        tuple_judge_yes_counts: Dict mapping constraint_tuple -> dict of {judge_idx: yes_count}
    
    Returns:
        Dict mapping constraint_tuple -> minimum_yes_count
    """
    tuple_min_yes_counts = {}
    
    for constraint_tuple, judge_yes_counts in tuple_judge_yes_counts.items():
        min_yes_count = min(judge_yes_counts.values())
        tuple_min_yes_counts[constraint_tuple] = min_yes_count
    
    return tuple_min_yes_counts


def compute_cumulative_counts(tuple_judge_yes_counts):
    """
    Calculate cumulative counts: for each Yes vote threshold, 
    count constraint tuples where ALL judges have yes_count >= that threshold.
    
    Args:
        tuple_judge_yes_counts: Dict mapping constraint_tuple -> dict of {judge_idx: yes_count}
    
    Returns:
        threshold_points (list), cumulative_counts (list)
    """
    threshold_points = list(range(0, 101))  # Thresholds from 0 to 100 Yes votes
    cumulative_counts = []
    
    for threshold in threshold_points:
        # Count tuples where ALL judges have yes_count >= threshold
        count = sum(
            1 for judge_yes_counts in tuple_judge_yes_counts.values()
            if all(cnt >= threshold for cnt in judge_yes_counts.values())
        )
        cumulative_counts.append(count)
    
    return threshold_points, cumulative_counts


def create_histogram(tuple_min_yes_counts, bin_size=10):
    """
    Create histogram data for constraint tuples by their minimum Yes vote count.
    
    Args:
        tuple_min_yes_counts: Dict mapping constraint_tuple -> min_yes_count
        bin_size: Size of each bin (default: 10)
    
    Returns:
        Two lists: bins and counts
    """
    # Create bins: 0-10, 10-20, ..., 90-100
    bins = list(range(0, 101, bin_size))
    counts = [0] * len(bins)
    
    for constraint_tuple, yes_count in tuple_min_yes_counts.items():
        # Find which bin this yes_count belongs to
        bin_idx = min(int(yes_count / bin_size), len(bins) - 1)
        counts[bin_idx] += 1
    
    return bins, counts


def plot_judge_acceptance(tuple_judge_yes_counts, tuple_min_yes_counts, output_dir, bin_size=5):
    """
    Create plots showing the distribution of constraint tuples by judge acceptance.
    Uses the methodology: count tuples where all judges have >= threshold Yes votes (absolute count).
    """
    if not tuple_min_yes_counts:
        print("No data to plot!")
        return
    
    print(f"\nCreating plots...")
    print(f"Total constraint tuples: {len(tuple_min_yes_counts)}")
    
    # Create histogram based on minimum Yes vote count
    bins, counts = create_histogram(tuple_min_yes_counts, bin_size)
    
    # Print distribution
    print("\nDistribution of constraint tuples by minimum judge Yes vote count:")
    for i, (bin_start, count) in enumerate(zip(bins, counts)):
        bin_end = bin_start + bin_size if i < len(bins) - 1 else 100
        print(f"  {bin_start}-{bin_end} Yes votes: {count} tuples")
    
    # Create histogram plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(bins, counts, width=bin_size * 0.8, align='edge', 
                   edgecolor='black', alpha=0.7, color=PLOT_COLOR)
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    str(count), ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Minimum Judge Yes Vote Count (Absolute)', fontsize=12)
    plt.ylabel('Number of Constraint Tuples', fontsize=12)
    plt.title('Distribution of Constraint Tuples by Minimum Judge Yes Vote Count', fontsize=14)
    plt.xticks(bins + [100], [str(x) for x in bins] + ['100'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save histogram
    histogram_dir = output_dir / 'histograms'
    histogram_dir.mkdir(parents=True, exist_ok=True)
    output_file = histogram_dir / 'judge_agreement_histogram.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {output_file}")
    
    # Create cumulative plot
    threshold_points, cumulative_counts = compute_cumulative_counts(tuple_judge_yes_counts)
    
    plt.figure(figsize=(12, 6))
    plt.plot(threshold_points, cumulative_counts, marker='o', linewidth=2, 
             markersize=4, color=PLOT_COLOR)
    
    # Add value labels at key points (every 10)
    for x, y in zip(threshold_points, cumulative_counts):
        if x % 10 == 0:  # Label every 10 Yes votes
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Acceptance Threshold (Number of Yes Votes)', fontsize=12)
    plt.ylabel('Number of Constraint Tuples Accepted by All Judges', fontsize=12)
    plt.title('Cumulative: Constraint Tuples Where All Judges Have ≥ X Yes Votes', fontsize=14)
    plt.xticks(list(range(0, 101, 10)), [str(x) for x in range(0, 101, 10)])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save cumulative plot
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_file = plots_dir / 'judge_agreement_cumulative.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative plot to {output_file}")
    
    # Save constraint tuple data by Yes vote count intervals
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Group constraint tuples by bins (using minimum Yes vote count)
    bins_data = defaultdict(list)
    for constraint_tuple, yes_count in tuple_min_yes_counts.items():
        bin_idx = min(int(yes_count / bin_size), int(100 / bin_size) - 1)
        bin_start = bin_idx * bin_size
        bins_data[bin_start].append({
            'constraints': list(constraint_tuple),
            'min_yes_votes': yes_count
        })
    
    # Save each bin to a separate file
    print(f"\nSaving constraint tuples by {bin_size} Yes vote intervals...")
    for bin_start in sorted(bins_data.keys()):
        bin_end = bin_start + bin_size
        tuples_in_bin = bins_data[bin_start]
        
        output_file = data_dir / f'tuples_{bin_start:02d}_{bin_end:02d}_yes_votes.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tuples_in_bin:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  {bin_start}-{bin_end} Yes votes: {len(tuples_in_bin)} tuples saved to {output_file.name}")
    
    print(f"\nAll constraint tuple data saved to {data_dir}")
    
    # Save complete list of all constraint tuples with Yes vote counts
    all_tuples_file = output_dir / 'all_tuples_yes_counts.jsonl'
    print(f"\nSaving complete list of all constraint tuples with Yes vote counts...")
    
    # Sort by minimum Yes count (descending) for easier analysis
    sorted_tuples = sorted(tuple_min_yes_counts.items(), key=lambda x: x[1], reverse=True)
    
    with open(all_tuples_file, 'w', encoding='utf-8') as f:
        for constraint_tuple, yes_count in sorted_tuples:
            f.write(json.dumps({
                'constraints': list(constraint_tuple),
                'min_yes_votes': yes_count
            }, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(sorted_tuples)} constraint tuples to {all_tuples_file}")


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
    print("JUDGE ACCEPTANCE ANALYSIS FOR CONSTRAINT TUPLES")
    print("="*60)
    print(f"Judge folder: {args.judge_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Bin size: {args.bin_size}")
    print(f"Methodology: Count tuples where all judges have >= threshold Yes votes (absolute count)")
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
    
    # Compute judge Yes vote counts (absolute counts for each judge)
    tuple_judge_yes_counts = compute_judge_yes_counts(judge_data_list)
    
    # Compute minimum Yes vote count across judges for each tuple
    tuple_min_yes_counts = compute_minimum_yes_counts(tuple_judge_yes_counts)
    
    # Create plots
    plot_judge_acceptance(tuple_judge_yes_counts, tuple_min_yes_counts, output_dir, bin_size=args.bin_size)
    
    print("\n" + "="*60)
    print("DONE!")
    print(f"Log saved to: {log_file_path}")
    print("="*60)
    
    # Restore stdout and close the logger
    sys.stdout = old_stdout
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze judge acceptance on constraint tuples. Counts tuples where all judges have >= threshold Yes votes (absolute count).")
    parser.add_argument('--judge-folder', type=str, required=True, help='Path to folder containing judge subfolders (each subfolder represents a judge with verdict files)')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output plots and data')
    parser.add_argument('--bin-size', type=int, default=5, help='Size of bins for histogram (default: 5, giving bins of 0-5, 5-10, etc. Yes votes)')
    args = parser.parse_args()
    main(args)
