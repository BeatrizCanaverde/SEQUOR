import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'Palatino LT STD', 'Book Antiqua', 'Georgia', 'DejaVu Serif']
plt.rcParams['font.size'] = 40
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40
plt.rcParams['legend.fontsize'] = 40

MODELS_MAP = {
    "gemma-3-4b-it": r"google__gemma-3-4b-it",
    "gemma-3-12b-it": r"google__gemma-3-12b-it",
    "gemma-3-27b-it": r"google__gemma-3-27b-it",
    "gemini-3.1-flash-lite-preview": r"google__gemini-3.1-flash-lite-preview",
    "Qwen3-4B-Inst-2507": r"Qwen__Qwen3-4B-Instruct-2507",
    "Qwen3-30B-A3B-Inst-FP8": r"Qwen__Qwen3-30B-A3B-Instruct-2507-FP8",
    "Qwen3-235B-A22B-Inst-FP8": r"Qwen3-235B-A22B-Instruct-2507-FP8",
    "GLM-4.7-Flash": r"zai-org__GLM-4.7-Flash",
    "Llama-3.3-70B-Inst": r"meta-llama__Llama-3.3-70B-Instruct",
    "gpt-oss-20b": r"openai__gpt-oss-20b",
    "gpt-oss-120b": r"openai__gpt-oss-120b",
}

LINE_EXPERIMENTS = {
    "Single": ("_single", "single_eval_scores.json"),
    "Tuples": ("_tuples", "tuple_size_3_eval_scores.json"),
    "Replace 5": ("_replace", "step_size_5_eval_scores.json"),
    "Replace 10": ("_replace", "step_size_10_eval_scores.json"),
    "Add 5": ("_add", "step_size_5_eval_scores.json"),
    "Add 10": ("_add", "step_size_10_eval_scores.json"),
    "Everything": ("_everything", "single_eval_scores.json"),
}


def get_per_turn_rates(model_id, suffix, filename, SCORES_DIR):
    folder_name = f"{model_id}{suffix}"
    path = SCORES_DIR / folder_name / filename
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            per_turn = data.get("per_turn_constraint_rate", {})
            rates = {}
            for turn, v in per_turn.items():
                if isinstance(v, dict):
                    rate = v.get("rate")
                else:
                    rate = v
                if isinstance(rate, (int, float)):
                    rates[int(turn)] = rate * 100
            return rates
    except Exception:
        return None


def bootstrap_ci_values(data, n_bootstrap=1000, ci=95):
    if not data or len(data) < 2:
        val = np.mean(data) if data else np.nan
        return val, val, val
    boot_means = []
    data = np.array(data)
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(resample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper


def get_raw_successes(processed_root, model_id, suffix, step_size_filter=None):
    folder_name = f"{model_id}{suffix}"
    model_dir = processed_root / folder_name
    if not model_dir.exists():
        return None

    turn_successes = {}

    file_pattern = "*.jsonl"
    if step_size_filter:
        file_pattern = f"step_size_{step_size_filter}_*.jsonl"

    for file in model_dir.glob(file_pattern):
        try:
            with open(file, 'r') as f:
                for line_idx, line in enumerate(f):
                    turn = line_idx + 1
                    try:
                        data = json.loads(line)
                        evals = data.get("constraint_evaluations", [])
                        if not evals:
                            continue
                        followed = all(e.get("followed", False) for e in evals)
                        status = 1 if followed else 0

                        if turn not in turn_successes: turn_successes[turn] = []
                        turn_successes[turn].append(status)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    return turn_successes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--processed_evals", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    print("Computing global best model using Borda count...")
    model_borda_scores = {m: 0 for m in MODELS_MAP.values()}

    for regime_name, (suffix, filename) in LINE_EXPERIMENTS.items():
        regime_averages = {}
        for model_id in MODELS_MAP.values():
            rates = get_per_turn_rates(model_id, suffix, filename, args.input)
            if rates:
                regime_averages[model_id] = np.mean(list(rates.values()))
            else:
                regime_averages[model_id] = 0

        ranked_models = sorted(regime_averages.keys(), key=lambda m: regime_averages[m], reverse=True)

        num_models = len(ranked_models)
        for rank, model_id in enumerate(ranked_models):
            borda_points = num_models - 1 - rank
            model_borda_scores[model_id] += borda_points

    best_model_id = max(model_borda_scores, key=model_borda_scores.get)
    best_model_display = [name for name, val in MODELS_MAP.items() if val == best_model_id][0]
    print(f"Best model globally based on Borda count: {best_model_id} ({best_model_display}) with {model_borda_scores[best_model_id]} points")

    regime_plots = {}
    best_model_bootstrap = {}

    for regime_name, (suffix, filename) in LINE_EXPERIMENTS.items():
        print(f"Collecting data for {regime_name}...")
        pool_per_turn = {}
        best_model_per_turn = {}

        step_size_filter = None
        if "step_size_" in filename:
            try:
                step_size_filter = filename.split("step_size_")[1].split("_")[0]
            except Exception:
                pass

        rates = get_per_turn_rates(best_model_id, suffix, filename, args.input)
        if not rates:
            for m in MODELS_MAP.values():
                rates = get_per_turn_rates(m, suffix, filename, args.input)
                if rates: break

        if not rates:
            print(f"No score data found for any model in {regime_name}")
            continue

        expected_turns = sorted(rates.keys())

        for model_id in MODELS_MAP.values():
            raw_successes = get_raw_successes(args.processed_evals, model_id, suffix, step_size_filter=step_size_filter)
            if raw_successes:
                for t in expected_turns:
                    if t in raw_successes:
                        if t not in pool_per_turn:
                            pool_per_turn[t] = {}
                        if model_id not in pool_per_turn[t]:
                            pool_per_turn[t][model_id] = []
                        pool_per_turn[t][model_id].extend(raw_successes[t])

                        if model_id == best_model_id:
                            if t not in best_model_per_turn:
                                best_model_per_turn[t] = []
                            best_model_per_turn[t].extend(raw_successes[t])

        if not pool_per_turn:
            print(f"No raw data found for {regime_name}")
            continue

        found_turns = sorted(pool_per_turn.keys())
        print(f"Sampling for {regime_name} over turns: {len(found_turns)} turns...")

        avg_stats = []
        for t in found_turns:
            model_averages = []
            for model_id, successes in pool_per_turn[t].items():
                if successes:
                    model_avg = np.mean(successes) * 100
                    model_averages.append(model_avg)

            if model_averages:
                mean, lower, upper = bootstrap_ci_values(model_averages, n_bootstrap=100)
                avg_stats.append({"turn": t, "mean": mean, "lower": lower, "upper": upper})

        regime_plots[regime_name] = pd.DataFrame(avg_stats)

        best_stats = []
        for t in found_turns:
            if t in best_model_per_turn:
                scores_percent = [s * 100 for s in best_model_per_turn[t]]
                mean, lower, upper = bootstrap_ci_values(scores_percent, n_bootstrap=100)
                best_stats.append({"turn": t, "mean": mean, "lower": lower, "upper": upper})

        if best_stats:
            best_model_bootstrap[regime_name] = pd.DataFrame(best_stats)

    fig = plt.figure(figsize=(28, 18))
    gs = fig.add_gridspec(2, 8)

    axes = []
    for j in range(4):
        axes.append(fig.add_subplot(gs[0, j*2:(j+1)*2]))

    for j in range(3):
        axes.append(fig.add_subplot(gs[1, (j*2)+1:(j+1)*2+1]))

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'

    color_best = "#1f77b4"
    color_avg = "#ff7f0e"

    for i, (ax, regime_name) in enumerate(zip(axes, LINE_EXPERIMENTS.keys())):
        if regime_name not in regime_plots:
            ax.set_title(regime_name)
            continue

        df = regime_plots[regime_name]

        ax.plot(df["turn"], df["mean"], label="Average across models", color=color_avg, linewidth=2.5, alpha=0.9)
        ax.fill_between(df["turn"], df["lower"], df["upper"], color=color_avg, alpha=0.2)

        if regime_name in best_model_bootstrap:
            best_label = best_model_display
            if "gemini-3.1-flash-lite" in best_model_display.lower():
                best_label = "Gemini 3.1 Flash Lite"

            bdf = best_model_bootstrap[regime_name]
            ax.plot(bdf["turn"], bdf["mean"], label=best_label,
                    color=color_best, linewidth=2.5, alpha=0.9)
            ax.fill_between(bdf["turn"], bdf["lower"], bdf["upper"], color=color_best, alpha=0.2)

        ax.set_title(regime_name, fontsize=40, fontweight='bold', pad=20)
        ax.set_xlabel("Turn", fontsize=40, labelpad=15)
        if i == 0 or i == 4:
            ax.set_ylabel("Accuracy (%)", fontsize=40, labelpad=20)
        ax.set_ylim(0, 105)
        ax.set_xlim(0, 50)
        ax.set_xticks(range(10, 51, 10))
        ax.set_yticks(range(25, 105, 25))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(labelsize=34)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=40, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.6)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Bootstrap CI plots saved to {output_path}")


if __name__ == "__main__":
    main()
