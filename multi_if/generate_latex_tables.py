import json
import argparse
from pathlib import Path

MODELS_MAP = {
    "Gemini-3.1-Flash-Lite": r"google__gemini-3.1-flash-lite-preview",
    "Qwen3-235B-A22B-Inst": r"Qwen3-235B-A22B-Instruct-2507-FP8",
    "GPT-oss-120B": r"openai__gpt-oss-120b",
    "Llama-3.3-70B-Inst": r"meta-llama__Llama-3.3-70B-Instruct",
    "Qwen3-30B-A3B-Inst": r"Qwen__Qwen3-30B-A3B-Instruct-2507-FP8",
    "GLM-4.7-Flash": r"zai-org__GLM-4.7-Flash",
    "Gemma3-27B": r"google__gemma-3-27b-it",
    "GPT-oss-20B": r"openai__gpt-oss-20b",
    "Gemma3-12B": r"google__gemma-3-12b-it",
    "Qwen3-4B-Inst": r"Qwen__Qwen3-4B-Instruct-2507",
    "Gemma3-4B": r"google__gemma-3-4b-it",
}

MODELS_ORDER = [
    "Gemini-3.1-Flash-Lite",
    "Qwen3-235B-A22B-Inst",
    "GPT-oss-120B",
    "Llama-3.3-70B-Inst",
    "Qwen3-30B-A3B-Inst",
    "GLM-4.7-Flash",
    "Gemma3-27B",
    "GPT-oss-20B",
    "Gemma3-12B",
    "Qwen3-4B-Inst",
    "Gemma3-4B",
]

EXPERIMENTS = {
    "Single": ("_single", "single_eval_scores.json"),
    "Tuples": ("_tuples", "tuple_size_3_eval_scores.json"),
    "Replace 5": ("_replace", "step_size_5_eval_scores.json"),
    "Replace 10": ("_replace", "step_size_10_eval_scores.json"),
    "Add 5": ("_add", "step_size_5_eval_scores.json"),
    "Add 10": ("_add", "step_size_10_eval_scores.json"),
    "Everything": ("_everything", "single_eval_scores.json"),
}


def get_turn_diff_50_1(model_id, suffix, filename, SCORES_DIR):
    folder_name = f"{model_id}{suffix}"
    path = SCORES_DIR / folder_name / filename
    if not path.exists():
        return None

    try:
        with open(path, 'r') as f:
            data = json.load(f)
            per_turn = data.get("per_turn_constraint_rate", {})

            def extract_rate(turn_key):
                v = per_turn.get(str(turn_key))
                if isinstance(v, dict):
                    return v.get("rate")
                return v

            rate_1 = extract_rate(1)
            available_turns = [int(t) for t in per_turn.keys() if t.isdigit()]
            if not available_turns:
                return None

            max_turn = max(available_turns)
            rate_last = extract_rate(max_turn)

            if rate_1 is not None and rate_last is not None:
                return (rate_last - rate_1) * 100
            return None
    except Exception:
        return None


def get_lowest_highest_diff(model_id, suffix, filename, SCORES_DIR):
    folder_name = f"{model_id}{suffix}"
    path = SCORES_DIR / folder_name / filename
    if not path.exists():
        return None

    try:
        with open(path, 'r') as f:
            data = json.load(f)
            per_turn = data.get("per_turn_constraint_rate", {})

            rates = []
            for v in per_turn.values():
                if isinstance(v, dict):
                    rate = v.get("rate")
                else:
                    rate = v
                if isinstance(rate, (int, float)):
                    rates.append(rate)

            if not rates:
                return None

            return (min(rates) - max(rates)) * 100
    except Exception:
        return None


def generate_table(caption, metric_func, label, SCORES_DIR):
    exp_keys = ["Single", "Tuples", "Replace 5", "Replace 10", "Add 5", "Add 10", "Everything"]
    num_cols = 7

    table = []
    table.append(r"\begin{table}[ht]")
    table.append(r"\centering")
    table.append(r"%\footnotesize")
    table.append(f"\\caption{{{caption}}}")
    table.append(r"\begin{tabular}{lccccccc}")
    table.append(r"\toprule")
    table.append(r" & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{Single}}} & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{Tuples}}} & \multicolumn{2}{c}{\textbf{Replace}} & \multicolumn{2}{c}{\textbf{Add}} & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{Everything}}} \\ \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    table.append(r"\textbf{Models} & \multicolumn{1}{c}{} & \multicolumn{1}{c}{} & \textbf{5} & \textbf{10} & \textbf{5} & \textbf{10} & \multicolumn{1}{c}{} \\")
    table.append(r"\midrule")

    column_values = [[] for _ in range(num_cols)]

    for display_name in MODELS_ORDER:
        model_id = MODELS_MAP[display_name]
        row = [display_name]
        for col_idx, exp_key in enumerate(exp_keys):
            suffix, filename = EXPERIMENTS[exp_key]
            val = metric_func(model_id, suffix, filename, SCORES_DIR)
            row.append(f"{val:.2f}" if val is not None else "-")
            if val is not None:
                column_values[col_idx].append(val)

        table.append("  & ".join(row) + r"  \\")

    table.append(r"\midrule")
    avg_row = [r"\textbf{Average}"]
    for vals in column_values:
        if vals:
            avg = sum(vals) / len(vals)
            avg_row.append(f"\\textbf{{{avg:.2f}}}")
        else:
            avg_row.append("-")
    table.append("  & ".join(avg_row) + r"  \\")

    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(f"\\label{{{label}}}")
    table.append(r"\end{table}")
    return "\n".join(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for evaluation results.")
    parser.add_argument("--input", type=Path, help="Directory containing the evaluation score JSON files.")
    parser.add_argument("--output", type=Path, help="File path where to save the generated tables.")
    args = parser.parse_args()

    table3 = generate_table(
        "Difference in success rate between the last turn and the 1st turn (%).",
        get_turn_diff_50_1,
        "tab:diff-50-1",
        args.input
    )

    table4 = generate_table(
        "Difference between lowest and highest per-turn success rates (Lowest - Highest, %).",
        get_lowest_highest_diff,
        "tab:lowest-highest-diff",
        args.input
    )

    with open(args.output, "w") as f:
        f.write(table3 + "\n\n")
        f.write(table4)

    print(f"Tables saved to {args.output}")
