from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import pandas as pd
import numpy as np
import re
from evaluation import EvaluationReport

# Custom pink/blue colormap
CUSTOM_RD_YL_GN = LinearSegmentedColormap.from_list(
    'CustomPinkBlue', ['#E91E63', '#FFE7D2', '#2196F3']
)

# Set Palatino font (or fallback)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'Palatino LT STD', 'Book Antiqua', 'Georgia', 'DejaVu Serif']
plt.rcParams['font.size'] = 14


def _get_model_order_score(model_name: str) -> int:
    """Return a score for sorting models according to a predefined order."""
    order = [
        r"gemini",
        r"qwen.*235b",
        r"gpt.*oss.*120b",
        r"llama.*3.3.*70b",
        r"qwen.*3.*30b",
        r"glm",
        r"gemma.*3.*27b",
        r"gpt.*oss.*20b",        
        r"gemma.*3.*12b",
        r"qwen.*3.*4b",
        r"gemma.*3.*4b",
    ]
    name_lower = model_name.lower()
    for i, pattern in enumerate(order):
        if re.search(pattern, name_lower):
            return i
    return 999


def _clean_model_name(model_name: str) -> str:
    """Remove common suffixes and clean up model names for display."""
    suffixes = ["_everything", "_single", "_tuples", "_add", "_replace", "_eval_scores"]
    cleaned = model_name
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
    
    # Optional: prettify names
    if "gemma-3-4b" in cleaned.lower(): return "Gemma3-4B"
    if "gemma-3-12b" in cleaned.lower(): return "Gemma3-12B"
    if "gemma-3-27b" in cleaned.lower(): return "Gemma3-27B"
    if "qwen3-4b" in cleaned.lower(): return "Qwen3-4B"
    if "qwen3-30b" in cleaned.lower(): return "Qwen3-30B-A3B"
    if "qwen3-235b" in cleaned.lower(): return "Qwen3-235B-A22B"
    if "glm" in cleaned.lower(): return "GLM-4.7-Flash"
    if "llama-3.3-70b" in cleaned.lower(): return "Llama-3.3-70B"
    if "gpt-oss-20b" in cleaned.lower(): return "GPT-oss-20B"
    if "gpt-oss-120b" in cleaned.lower(): return "GPT-oss-120B"
    if "gemini-3.1-flash-lite" in cleaned.lower(): return "Gemini-3.1"
    
    return cleaned


def save_overall_constraint_rate(
    metrics_results: Dict[str, Any], 
    output_dir: Path
):
    """Save overall constraint-following rate to a text file."""

    models = list(metrics_results.keys())
    file_path = output_dir / "overall_constraint_rate.txt"
    with open(file_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OVERALL CONSTRAINT FOLLOWING RATE\n")
        f.write("="*80 + "\n\n")
        f.write("This metric shows the overall success rate across all turns and conversations.\n\n")
        model_rates = []
        for model in models:
            data = metrics_results[model].get('overall_constraint_rate', {})
            if 'rate' in data:
                model_rates.append((model, data))
        
        model_rates.sort(key=lambda x: x[1]['rate'], reverse=True)
        f.write(f"{'Model':<32} {'Success Rate':<15} {'Successful Turns':<22} {'Total Turns':<16}\n")
        f.write("-"*70 + "\n")
        for model, data in model_rates:
            successful = data.get('successful_turns', data.get('successful_constraints', 0))
            total = data.get('total_turns', data.get('total_constraints', 0))
            f.write(f"{model:<32} {data['rate']:<15.4f} {successful:<22} {total:<16}\n")


def save_avg_turns_without_error(
    metrics_results: Dict[str, Any], 
    output_dir: Path
):
    """Save average turns without error to a text file."""

    models = list(metrics_results.keys())
    # Build a list of (model, metrics) and sort by avg_turns desc
    model_turns = [(model, metrics_results[model].get('avg_turns_without_error', {})) for model in models]
    model_turns.sort(key=lambda x: x[1].get('avg_turns', 0.0), reverse=True)

    file_path = output_dir / f"avg_turns_without_error.txt"
    with open(file_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"AVERAGE TURNS WITHOUT ERROR\n")
        f.write("="*80 + "\n\n")
        f.write("This metric shows the average number of consecutive successful turns from the start.\n\n")

        f.write(f"{'Model':<32} {'Avg Turns':<12} {'Total Conversations':<22}\n")
        f.write("-"*72 + "\n")
        for model, data in model_turns:
            avg = data.get('avg_turns', 0.0)
            total_conv = data.get('total_conversations', data.get('total_conversations', 0))
            f.write(f"{model:<32} {avg:<12.2f} {total_conv:<22}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED BREAKDOWN BY CONVERSATION\n")
        f.write("="*80 + "\n\n")

        for model, data in model_turns:
            f.write(f"\n{model}:\n")
            # compute_avg_turns_without_error returns keys: avg_turns, list_of_consecutive_successes, total_conversations
            raw_per_conv = data.get('list_of_consecutive_successes', data.get('per_conversation', []))
            # Handle relative scores where some ratios might be None (baseline 0 but eval > 0)
            per_conv = [x for x in raw_per_conv if x is not None]
            
            f.write(f"  Turns without error per conversation: {raw_per_conv}\n")
            if per_conv:
                f.write(
                    f"  Min: {min(per_conv)}, Max: {max(per_conv)}, "
                    f"Median: {sorted(per_conv)[len(per_conv)//2]}\n"
                )
            else:
                f.write("  Min: 0, Max: 0, Median: 0\n")


def save_per_turn_constraint_rate(
    metrics_results: Dict[str, Any], 
    output_dir: Path
):
    """Save per-turn constraint-following rate to a text file."""

    models = list(metrics_results.keys())
    all_turns = set()
    # metrics_results maps model -> scores where per_turn_constraint_rate is under each model
    for model in models:
        model_rates = metrics_results[model].get('per_turn_constraint_rate', {})
        for k in model_rates.keys():
            try:
                all_turns.add(int(k))
            except Exception:
                all_turns.add(k)
    sorted_turns = sorted(all_turns)
    file_path = output_dir / "per_turn_constraint_rate.txt"
    with open(file_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PER-TURN CONSTRAINT FOLLOWING RATE\n")
        f.write("="*80 + "\n\n")
        f.write("This metric shows the success rate for each turn position across all conversations.\n\n")
        header = f"{'Turn':<6}"
        for model in models:
            header += f"{model:<12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        def _num_rate(v):
            # v can be a dict like {'rate': 0.5, ...} or a number
            if isinstance(v, dict):
                return float(v.get('rate', 0.0))
            try:
                return float(v)
            except Exception:
                return 0.0

        for turn in sorted_turns:
            row = f"{turn:<6}"
            for model in models:
                model_rates = metrics_results[model].get('per_turn_constraint_rate', {})
                # try int key first (JSON keys may be strings)
                val = model_rates.get(turn)
                if val is None:
                    val = model_rates.get(str(turn), 0.0)
                rate = _num_rate(val)
                row += f"{rate:<12.4f}"
            f.write(row + "\n")


def save_cumulative_constraint_rate_per_turn(
    metrics_results: Dict[str, Any], 
    output_dir: Path
):
    """Save cumulative constraint-following rate per turn to a text file."""

    models = list(metrics_results.keys())
    all_turns = set()
    for model in models:
        model_rates = metrics_results[model].get('cumulative_constraint_rate_per_turn', {})
        for k in model_rates.keys():
            try:
                all_turns.add(int(k))
            except Exception:
                all_turns.add(k)
    sorted_turns = sorted(all_turns)
    file_path = output_dir / "cumulative_constraint_rate_per_turn.txt"
    with open(file_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CUMULATIVE CONSTRAINT FOLLOWING RATE PER TURN\n")
        f.write("="*80 + "\n\n")
        f.write("This metric shows the fraction of conversations where ALL constraints\n")
        f.write("from turn 1 up to the given turn were followed successfully.\n\n")
        header = f"{'Turn':<6}"
        for model in models:
            header += f"{model:<12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        def _num_rate(v):
            if isinstance(v, dict):
                return float(v.get('rate', 0.0))
            try:
                return float(v)
            except Exception:
                return 0.0

        for turn in sorted_turns:
            row = f"{turn:<6}"
            for model in models:
                model_rates = metrics_results[model].get('cumulative_constraint_rate_per_turn', {})
                val = model_rates.get(turn)
                if val is None:
                    val = model_rates.get(str(turn), 0.0)
                rate = _num_rate(val)
                row += f"{rate:<12.4f}"
            f.write(row + "\n")


def create_overall_constraint_rate_chart(
    metrics_results: Dict[str, Any], 
    output_dir: Path
):
    """Create a bar chart for overall constraint-following rate."""
    # Sort models based on predefined order
    models = sorted(metrics_results.keys(), key=_get_model_order_score)
    valid_models = []
    rates = []

    for model in models:
        data = metrics_results[model].get('overall_constraint_rate', {})
        if 'rate' in data:
            rates.append(data['rate'])
            valid_models.append(model)
    
    if not valid_models:
        return

    # Clean model names for display
    cleaned_labels = [_clean_model_name(m) for m in valid_models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(cleaned_labels, rates, color='#2196F3', alpha=0.7)
    plt.title('Overall Constraint Following Rate')
    plt.ylabel('Success Rate')
    plt.xlabel(None)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{rate:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "overall_constraint_rate.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def create_avg_turns_without_error_chart(
    metrics_results: Dict[str, Any], 
    output_dir: Path
):
    """Create a bar chart for average turns without error."""
    # Sort models based on predefined order
    models = sorted(metrics_results.keys(), key=_get_model_order_score)
    valid_models = []
    avg_turns = []

    for model in models:
        data = metrics_results[model].get('avg_turns_without_error', {})
        if 'avg_turns' in data:
            avg_turns.append(data['avg_turns'])
            valid_models.append(model)
            
    if not valid_models:
        return

    # Clean model names for display
    cleaned_labels = [_clean_model_name(m) for m in valid_models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(cleaned_labels, avg_turns, color='#E91E63', alpha=0.7)
    plt.title('Average Turns Without Error')
    plt.ylabel('Average Turns')
    plt.xlabel(None)
    plt.xticks(rotation=45)
    for bar, turns in zip(bars, avg_turns):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{turns:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "avg_turns_without_error.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def create_per_turn_constraint_rate_plot(
    metrics_results: Dict[str, Any], 
    output_dir: Path,
    experiment_name: str = None
):
    """Create a line plot for per-turn constraint-following rate."""

    # Sort models based on predefined order
    models = sorted(metrics_results.keys(), key=_get_model_order_score)
    plt.figure(figsize=(12, 8))
    def _should_keep(turn, exp_context):
        if not isinstance(turn, int): return True
        exp_lower = exp_context.lower()
        
        # Determine step size if present
        step_10 = "step_10" in exp_lower or "_10" in exp_lower
        step_5 = "step_5" in exp_lower or "_5" in exp_lower

        if "add" in exp_lower or "replace" in exp_lower:
            if step_10:
                # 1, 5, 10, 11, 15, 20, 21, 25, 30, ...
                if "add" in exp_lower and turn > 30: return False
                return turn == 1 or turn % 5 == 0 or (turn - 1) % 10 == 0
            elif step_5:
                # 1, 3, 5, 6, 8, 10, 11, 13, 15
                if "add" in exp_lower:
                    if turn > 15: return False
                    return turn in [1, 3, 5, 6, 8, 10, 11, 13, 15]
                # 1, 5, 6, 10, 11, 15, 16, 20, 21, ...
                return turn == 1 or turn % 5 == 0 or (turn - 1) % 5 == 0
            
        if any(kw in exp_lower for kw in ["everything", "single", "tuples"]):
            # 1, 5, 10, 15, 20, ...
            return turn == 1 or turn % 5 == 0
            
        return True

    for model in models:
        # Clean model name for label
        display_name = _clean_model_name(model)
        model_rates = metrics_results[model].get('per_turn_constraint_rate', {})
        items = []
        for k, v in model_rates.items():
            # normalize key
            try:
                key = int(k)
                key_label = str(key)
            except Exception:
                key = None
                key_label = str(k)

            # extract numeric rate
            if isinstance(v, dict):
                rate = float(v.get('rate', 0.0))
            else:
                try:
                    rate = float(v)
                except Exception:
                    rate = 0.0

            items.append((key, key_label, rate))

        if not items:
            continue

        # Filter turns based on experiment rules
        if experiment_name:
            items = [it for it in items if _should_keep(it[0], experiment_name)]
        else:
            # Default fallback if no experiment name provided
            items = [it for it in items if it[0] is not None and it[0] % 2 != 0]
        
        if not items:
            continue

        # If all keys are ints, use them as x; otherwise use sequential indices
        if all(it[0] is not None for it in items):
            items_sorted = sorted(items, key=lambda x: x[0])
            x = [it[0] for it in items_sorted]
            y = [it[2] for it in items_sorted]
        else:
            items_sorted = sorted(items, key=lambda x: x[1])
            x = list(range(1, len(items_sorted) + 1))
            y = [it[2] for it in items_sorted]

        plt.plot(x, y, marker='o', label=display_name, linewidth=2, markersize=4)
    plt.title('Per-Turn Constraint Following Rate')
    plt.xlabel('Turn Number')
    plt.ylabel('Success Rate')
    plt.legend(title=None)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "per_turn_constraint_rate.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def create_per_turn_constraint_rate_heatmap(
    metrics_results: Dict[str, Any], 
    output_dir: Path,
    experiment_name: str = None
):
    """Heatmap view of per-turn success rates (models x turns)."""

    # Sort models based on predefined order
    models = sorted(metrics_results.keys(), key=_get_model_order_score)
    all_turns: list[int | str] = []

    def _should_keep(turn, exp_context):
        if not isinstance(turn, int): return True
        exp_lower = exp_context.lower()
        
        # Determine step size if present
        step_10 = "step_10" in exp_lower or "_10" in exp_lower
        step_5 = "step_5" in exp_lower or "_5" in exp_lower

        if "add" in exp_lower or "replace" in exp_lower:
            if step_10:
                # 1, 5, 10, 11, 15, 20, 21, 25, 30, ...
                if "add" in exp_lower and turn > 30: return False
                return turn == 1 or turn % 5 == 0 or (turn - 1) % 10 == 0
            elif step_5:
                # 1, 3, 5, 6, 8, 10, 11, 13, 15
                if "add" in exp_lower:
                    if turn > 15: return False
                    return turn in [1, 3, 5, 6, 8, 10, 11, 13, 15]
                # 1, 5, 6, 10, 11, 15, 16, 20, 21, ...
                return turn == 1 or turn % 5 == 0 or (turn - 1) % 5 == 0
            
        if any(kw in exp_lower for kw in ["everything", "single", "tuples"]):
            # 1, 5, 10, 15, 20, ...
            return turn == 1 or turn % 5 == 0
            
        return True

    def _collect_turns():
        seen = set()
        for model in models:
            model_rates = metrics_results[model].get('per_turn_constraint_rate', {})
            for k in model_rates.keys():
                try:
                    k_int = int(k)
                    seen.add(k_int)
                except Exception:
                    seen.add(k)
        
        # Filter turns based on experiment rules
        if experiment_name:
            filtered = [t for t in seen if _should_keep(t, experiment_name)]
        else:
            filtered = [t for t in seen if isinstance(t, int) and t % 2 != 0]
        return sorted(filtered, key=lambda x: (isinstance(x, str), x))

    all_turns = _collect_turns()
    if not all_turns:
        return

    def _num_rate(v):
        if isinstance(v, dict):
            return float(v.get('rate', 0.0))
        try:
            return float(v)
        except Exception:
            return 0.0

    data = []
    for model in models:
        row = []
        model_rates = metrics_results[model].get('per_turn_constraint_rate', {})
        for turn in all_turns:
            val = model_rates.get(turn)
            if val is None:
                val = model_rates.get(str(turn), np.nan)
            row.append(_num_rate(val))
        data.append(row)

    # Clean model names for display
    cleaned_models = [_clean_model_name(m) for m in models]
    df = pd.DataFrame(data, index=cleaned_models, columns=[str(t) for t in all_turns])

    fig_w = max(6, len(all_turns) * 0.6)
    fig_h = max(4, len(models) * 0.4)
    plt.figure(figsize=(fig_w, fig_h))
    
    # Calculate aspect ratio for colorbar height to match row height
    # Row height in figure inches is roughly (fig_h / len(models))
    # Colorbar 'aspect' is (length / thickness). 
    # To make thickness = (fig_h / len(models)) * 0.5 (half height per user request), we set aspect = length / (row_height * 0.5).
    # Using a tighter pad for wider heatmaps to bring the colorbar closer
    is_wide = len(all_turns) > 14
    cbar_pad = 0.02 if is_wide else 0.05

    sns.heatmap(
        df, 
        annot=True, 
        fmt=".2f", 
        cmap=CUSTOM_RD_YL_GN, 
        vmin=0.0, 
        vmax=1.0, 
        annot_kws={"size": 10},
        cbar_kws={
            'label': 'Accuracy', 
            'orientation': 'vertical', 
            'pad': cbar_pad, 
            'shrink': 0.5,
        }
    )
    
    # Force axis tick label size
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Get colorbar and set its label and tick size
    ax = plt.gca()
    if hasattr(ax, 'collections') and len(ax.collections) > 0:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('Accuracy', size=11)

    plt.xlabel('Turn', fontsize=11)
    plt.ylabel(None)
    plt.title(None)
    plt.tight_layout()
    plt.savefig(output_dir / "per_turn_constraint_rate_heatmap.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def create_cumulative_constraint_rate_per_turn_plot(
    metrics_results: Dict[str, Any], 
    output_dir: Path
):
    """Create a line plot for cumulative constraint-following rate per turn."""
    
    # Sort models based on predefined order
    models = sorted(metrics_results.keys(), key=_get_model_order_score)
    plt.figure(figsize=(12, 8))
    for model in models:
        # Clean model name for label
        display_name = _clean_model_name(model)
        model_rates = metrics_results[model].get('cumulative_constraint_rate_per_turn', {})
        items = []
        for k, v in model_rates.items():
            try:
                key = int(k)
                key_label = str(key)
            except Exception:
                key = None
                key_label = str(k)

            if isinstance(v, dict):
                rate = float(v.get('rate', 0.0))
            else:
                try:
                    rate = float(v)
                except Exception:
                    rate = 0.0

            items.append((key, key_label, rate))

        if not items:
            continue

        if all(it[0] is not None for it in items):
            items_sorted = sorted(items, key=lambda x: x[0])
            x = [it[0] for it in items_sorted]
            y = [it[2] for it in items_sorted]
        else:
            items_sorted = sorted(items, key=lambda x: x[1])
            x = list(range(1, len(items_sorted) + 1))
            y = [it[2] for it in items_sorted]

        plt.plot(x, y, marker='s', label=display_name, linewidth=2, markersize=4)
    plt.title('Cumulative Constraint Following Rate per Turn')
    plt.xlabel('Turn')
    plt.ylabel('Cumulative Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_constraint_rate_per_turn.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def create_heatmap(
    data: pd.DataFrame, 
    output_path: Path
):
    """Create and save instruction-based heatmap visualization for a specific analysis mode."""

    n_instructions = len(data.index)
    n_turns = len(data.columns)
    fig_width = max(12, n_turns * 0.5)
    fig_height = max(8, n_instructions * 0.3)

    plt.figure(figsize=(fig_width, fig_height))
    mask = data.isna()
    ytick_fontsize = min(16, max(10, 250 / n_instructions))
    xtick_fontsize = min(10, max(8, 150 / n_turns))

    # Calculate colorbar aspect to match instruction row height (half height)
    cbar_thickness = (fig_height / max(1, n_instructions)) * 0.5
    cbar_length = fig_width * 0.5
    cbar_aspect = cbar_length / cbar_thickness if cbar_thickness > 0 else 40

    sns.heatmap(
        data,
        annot=False,
        cmap=CUSTOM_RD_YL_GN,
        vmin=0.0,
        vmax=1.0,
        mask=mask,
        cbar_kws={
            'label': "Individual Turn Success Rate", 
            'orientation': 'horizontal', 
            'pad': 0.1, 
            'shrink': 0.5,
            'aspect': cbar_aspect
        },
        linewidths=0.5,
        yticklabels=True,
        xticklabels=True
    )

    #plt.title(f'{model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Turn', fontsize=12)
    plt.ylabel('Constraint Type', fontsize=12)
    plt.xticks(rotation=45, fontsize=xtick_fontsize)
    plt.yticks(rotation=0, fontsize=ytick_fontsize)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.1)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def prepare_heatmap_data(
    eval_reports: dict[str, EvaluationReport], 
    turns: list[int], 
) -> pd.DataFrame:
    """
    Prepare data for heatmap visualization.
    Each row is a report (conversation), each column is a turn.
    Each cell is 1.0 if all judged constraints in that turn were
    followed, 0.0 if any judged constraint failed, and NaN if the
    turn had no parsed judgments.
    """
    report_names = sorted(eval_reports.keys())
    data = []

    for report_name in report_names:
        eval_report = eval_reports[report_name]
        row = []
        for turn in turns:
            # `turn` is 1-based (turn 1 -> eval_report[0]). Convert to 0-based index.
            idx = turn - 1
            if 0 <= idx < len(eval_report):
                prompt_eval = eval_report[idx]
                constraint_evals = prompt_eval.get("constraint_evaluations", [])
                judged_flags = [ce.get("followed") for ce in constraint_evals if ce.get("followed") is not None]
                if judged_flags:
                    rate = 1.0 if all(judged_flags) else 0.0
                else:
                    rate = np.nan
            else:
                rate = np.nan
            row.append(rate)
        data.append(row)

    df = pd.DataFrame(data, index=report_names, columns=[f"Turn {t}" for t in turns])
    return df
