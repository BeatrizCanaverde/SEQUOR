import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
import argparse

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'Palatino LT STD', 'Book Antiqua', 'Georgia', 'DejaVu Serif']
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22

EXPERIMENTS = ["single", "tuples", "replace", "add", "everything"]

MODELS = [
    "google__gemma-3-4b-it",
    "google__gemma-3-12b-it",
    "google__gemma-3-27b-it",
    "Qwen__Qwen3-4B-Instruct-2507",
    "Qwen__Qwen3-30B-A3B-Instruct-2507-FP8",
    "Qwen3-235B-A22B-Instruct-2507-FP8",
    "zai-org__GLM-4.7-Flash",
    "meta-llama__Llama-3.3-70B-Instruct",
    "openai__gpt-oss-20b",
    "openai__gpt-oss-120b",
]

EXPERIMENT_DISPLAY_NAMES = {
    "single": "Single",
    "tuples": "Tuples",
    "replace": "Replace 10",
    "add": "Add 10",
    "everything": "Everything"
}

EXPERIMENT_JSON_FILES = {
    "single": "single_eval_scores.json",
    "tuples": "tuple_size_3_eval_scores.json",
    "replace": "step_size_10_eval_scores.json",
    "add": "step_size_10_eval_scores.json",
    "everything": "single_eval_scores.json"
}


def get_turn_data(model_base, experiment, scores_dir):
    folder_name = f"{model_base}_{experiment}"
    json_name = EXPERIMENT_JSON_FILES.get(experiment, "single_eval_scores.json")
    path = scores_dir / folder_name / json_name

    if not path.exists():
        return None

    with open(path, 'r') as f:
        data = json.load(f)
        per_turn = data.get("per_turn_constraint_rate", {})

        rates = []
        for turn_key in per_turn:
            v = per_turn[turn_key]
            rate = v.get("rate") if isinstance(v, dict) else v
            if isinstance(rate, (int, float)):
                rates.append((int(turn_key), rate))

        if not rates:
            return None

        rates.sort()

        turn_1_val = None
        turn_last_val = None
        target_last = 50

        for turn, val in rates:
            if turn == 1:
                turn_1_val = val * 100
            if turn == target_last:
                turn_last_val = val * 100

        if turn_1_val is None and rates:
            turn_1_val = rates[0][1] * 100

        if turn_last_val is None and rates:
            turn_last_val = rates[-1][1] * 100

        return (turn_1_val, turn_last_val)


def create_multi_slope_plot(args):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    plt.subplots_adjust(wspace=0.3)

    SCORES_DIR = args.scores_dir
    OUTPUT_DIR = args.output_dir

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, exp in enumerate(EXPERIMENTS):
        ax = axes[i]
        all_model_data = []

        for model in MODELS:
            data = get_turn_data(model, exp, SCORES_DIR)
            if data:
                ax.plot([0, 1], data, color='dimgray', linewidth=1, alpha=0.5, marker='o', markersize=4)
                all_model_data.append(data)

        if all_model_data:
            all_model_data = np.array(all_model_data)
            avg_data = np.mean(all_model_data, axis=0)

            ax.plot([0, 1], avg_data, color=colors[i], linewidth=3, marker='o', markersize=6, zorder=10)

            ax.set_ylim(0, 100)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['1', '50'])
            ax.set_title(EXPERIMENT_DISPLAY_NAMES.get(exp, exp), fontweight='bold')

            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.tick_params(labelleft=True)

            if i == 0:
                ax.set_ylabel('Accuracy (%)')
        else:
            ax.set_title(f"{EXPERIMENT_DISPLAY_NAMES.get(exp, exp)} (No Data)")
            print(f"Warning: No data found for experiment '{exp}'")

    fig.text(0.5, -0.05, 'Turn', ha='center', fontsize=22)
    output_path = OUTPUT_DIR / "multi_experiment_slope_plots.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_dir", type=Path, required=True, help="Path to the directory containing the evaluation score JSON files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to the directory where the generated plot will be saved")
    args = parser.parse_args()

    create_multi_slope_plot(args)
