import json
import argparse
from pathlib import Path
import statistics
import time
from transformers import AutoTokenizer
import logging
import warnings

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*longer than the specified maximum.*")

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


MODELS_MAP = {
    "Gemini-3.1-Flash-Lite": "google__gemini-3.1-flash-lite-preview",
    "Qwen3-235B-A22B-Inst": "Qwen3-235B-A22B-Instruct-2507-FP8",
    "GPT-oss-120B": "openai__gpt-oss-120b",
    "Llama-3.3-70B-Inst": "meta-llama__Llama-3.3-70B-Instruct",
    "Qwen3-30B-A3B-Inst": "Qwen__Qwen3-30B-A3B-Instruct-2507-FP8",
    "GLM-4.7-Flash": "zai-org__GLM-4.7-Flash",
    "Gemma3-27B": "google__gemma-3-27b-it",
    "GPT-oss-20B": "openai__gpt-oss-20b",
    "Gemma3-12B": "google__gemma-3-12b-it",
    "Qwen3-4B-Inst": "Qwen__Qwen3-4B-Instruct-2507",
    "Gemma3-4B": "google__gemma-3-4b-it",
}

TOKENIZER_PATHS = {
    "google__gemma-3-4b-it": "google/gemma-3-4b-it",
    "google__gemma-3-12b-it": "google/gemma-3-12b-it",
    "google__gemma-3-27b-it": "google/gemma-3-27b-it",
    "Qwen__Qwen3-4B-Instruct-2507": "Qwen__Qwen3-4B-Instruct-2507",
    "Qwen__Qwen3-30B-A3B-Instruct-2507-FP8": "Qwen__Qwen3-30B-A3B-Instruct-2507-FP8",
    "Qwen3-235B-A22B-Instruct-2507-FP8": "Qwen3-235B-A22B-Instruct-2507-FP8",
    "zai-org__GLM-4.7-Flash": "zai-org/GLM-4.7-Flash",
    "meta-llama__Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "openai__gpt-oss-20b": "openai/gpt-oss-20b",
    "openai__gpt-oss-120b": "openai/gpt-oss-120b",
    "google__gemini-3.1-flash-lite-preview": "google/gemini-3.1-flash-lite-preview",
}

EXPERIMENTS = [
    ("Single", "_single", None),
    ("Tuples", "_tuples", None),
    ("Replace 5", "_replace", "step_size_5"),
    ("Replace 10", "_replace", "step_size_10"),
    ("Add 5", "_add", "step_size_5"),
    ("Add 10", "_add", "step_size_10"),
    ("Everything", "_everything", None)
]


def load_tokenizer(model_dir):
    try:
        fix_mistral = "GLM" in model_dir or "mistral" in model_dir.lower()
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, fix_mistral_regex=fix_mistral)
    except Exception as e:
        logging.warning(f"Failed to load tokenizer from {model_dir}: {e}")
        return None


def compute_token_stats(model_key, exp_suffix, prefix_filter, tokenizer, RUNS_DIR):
    if tokenizer is None:
        return None

    runs_folder = f"{MODELS_MAP[model_key]}{exp_suffix}"

    if "Llama-3.3-70B" in runs_folder:
        runs_folder = runs_folder.replace("llama-3.3-70b-instruct", "Llama-3.3-70B-Instruct")
        chat_dir = RUNS_DIR / runs_folder / "chat"
        if not chat_dir.exists():
            runs_folder = runs_folder.replace("Llama-3.3-70B-Instruct", "llama-3.3-70b-instruct")
            chat_dir = RUNS_DIR / runs_folder / "chat"
    else:
        chat_dir = RUNS_DIR / runs_folder / "chat"

    if not chat_dir.exists():
        return None

    all_token_counts = []

    for file in chat_dir.glob("*.jsonl"):
        if prefix_filter and not file.name.startswith(prefix_filter):
            continue

        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content: continue
                if content.startswith('['):
                    messages = json.loads(content)
                else:
                    lines = content.split('\n')
                    messages = []
                    for line in lines:
                        if not line.strip(): continue
                        messages.append(json.loads(line))

                truncated_messages = []
                user_count = 0
                assistant_count = 0
                for m in messages:
                    if isinstance(m, dict):
                        role = m.get("role", "")
                        if role == "user":
                            if user_count < 50:
                                truncated_messages.append(m)
                                user_count += 1
                        elif role == "assistant":
                            if assistant_count < 50:
                                truncated_messages.append(m)
                                assistant_count += 1
                        if user_count >= 50 and assistant_count >= 50:
                            break
                messages = truncated_messages

                try:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                    tokens = tokenizer.encode(prompt, add_special_tokens=False)
                except Exception as e:
                    if "longer than the specified maximum" in str(e):
                        max_len = getattr(tokenizer, 'model_max_length', 128000)
                        try:
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                            tokens = tokenizer.encode(prompt, add_special_tokens=False)[:max_len]
                        except Exception:
                            text = " ".join([m.get("content", "") for m in messages if isinstance(m, dict)])
                            tokens = tokenizer.encode(text, add_special_tokens=False)[:max_len]
                    else:
                        text = " ".join([m.get("content", "") for m in messages if isinstance(m, dict)])
                        tokens = tokenizer.encode(text, add_special_tokens=False)

                all_token_counts.append(len(tokens))
        except Exception:
            continue

    if not all_token_counts:
        return None

    return {
        "mean": statistics.mean(all_token_counts),
        "median": statistics.median(all_token_counts),
        "std": statistics.stdev(all_token_counts) if len(all_token_counts) > 1 else 0.0,
        "raw": all_token_counts
    }


def generate_table(RUNS_DIR, CACHE_FILE, recompute=False):
    results = {}
    if not recompute and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                raw_results = json.load(f)

            name_mapping = {
                "Qwen3-235B-A22B-Inst-FP8": "Qwen3-235B-A22B-Inst",
                "Qwen3-30B-A3B-Inst-FP8": "Qwen3-30B-A3B-Inst",
                "Qwen3-4B-Instruct": "Qwen3-4B-Inst"
            }
            for old_name, new_name in name_mapping.items():
                if old_name in raw_results and new_name not in raw_results:
                    raw_results[new_name] = raw_results.pop(old_name)
                    logging.info(f"Mapped cached data from '{old_name}' to '{new_name}'")

            results = raw_results
            logging.info(f"Loaded results from cache: {CACHE_FILE}")
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")

    tokenizers_cache = {}
    needs_save = False

    for model_display, model_map_key in MODELS_MAP.items():
        if model_display not in results:
            results[model_display] = {}

        tokenizer_loaded = False
        tokenizer = None

        for exp_name, exp_suffix, prefix_filter in EXPERIMENTS:
            if not recompute and exp_name in results[model_display] and results[model_display][exp_name] is not None:
                continue

            if not tokenizer_loaded:
                tokenizer_path = TOKENIZER_PATHS.get(model_map_key)
                logging.info(f"Loading tokenizer for {model_display} from {tokenizer_path}")
                if tokenizer_path not in tokenizers_cache:
                    tokenizers_cache[tokenizer_path] = load_tokenizer(tokenizer_path)
                tokenizer = tokenizers_cache[tokenizer_path]
                tokenizer_loaded = True

            stats = compute_token_stats(model_display, exp_suffix, prefix_filter, tokenizer, RUNS_DIR)
            results[model_display][exp_name] = stats
            needs_save = True
            if stats:
                logging.info(f"{model_display} - {exp_name}: mean={stats['mean']:.1f}, std={stats['std']:.1f}")

    if needs_save:
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, 'w') as f:
                json.dump(results, f)
            logging.info(f"Saved results to cache: {CACHE_FILE}")
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

    def format_latex_table(stat_key, title_suffix, show_std=True):
        table = []
        table.append(r"\begin{table*}[h]")
        table.append(r"\centering")
        table.append(r"\small")

        col_def = "l" + "c" * len(EXPERIMENTS)
        table.append(rf"\begin{{tabular}}{{{col_def}}}")
        table.append(r"\toprule")

        header_row = [r"& \multicolumn{1}{c}{\multirow{2}{*}{\textbf{Single}}} & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{Tuples}}} & \multicolumn{2}{c}{\textbf{Replace}} & \multicolumn{2}{c}{\textbf{Add}} & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{Everything}}} \\"]
        table.append(" ".join(header_row))

        table.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}")

        sub_headers = ["Model", "", "", "5", "10", "5", "10", ""]
        table.append(" & ".join(sub_headers) + r" \\")
        table.append(r"\midrule")

        col_all_raw = {exp[0]: [] for exp in EXPERIMENTS}

        for model_display in MODELS_MAP.keys():
            row = [model_display]
            for exp_name, _, _ in EXPERIMENTS:
                stats = results[model_display][exp_name]
                if stats:
                    val = stats[stat_key] / 1000.0
                    std_val = stats['std'] / 1000.0
                    if stat_key == "mean" and show_std:
                        row.append(f"{val:.0f} \\small{{$\\pm$ {std_val:.0f}}}")
                    else:
                        row.append(f"{val:.0f}")
                    col_all_raw[exp_name].extend(stats["raw"])
                else:
                    row.append("-")

            table.append(" & ".join(row) + r" \\")

        table.append(r"\midrule")

        avg_row = ["Average"]
        for exp_name, _, _ in EXPERIMENTS:
            raw_vals = col_all_raw[exp_name]
            if raw_vals:
                v_mean = statistics.mean(raw_vals) / 1000.0
                v_std = (statistics.stdev(raw_vals) if len(raw_vals) > 1 else 0.0) / 1000.0
                if show_std:
                    avg_row.append(f"{v_mean:.0f} \\small{{$\\pm$ {v_std:.0f}}}")
                else:
                    avg_row.append(f"{v_mean:.0f}")
            else:
                avg_row.append("-")

        table.append(" & ".join(avg_row) + r" \\")
        table.append(r"\bottomrule")
        table.append(r"\end{tabular}")
        table.append(rf"\caption{{Average token count per conversation ({title_suffix}).}}")
        table.append(rf"\label{{tab:token_counts_{stat_key}}}")
        table.append(r"\end{table*}")
        flat_overall = [item for sublist in col_all_raw.values() for item in sublist]
        return "\n".join(table), flat_overall

    mean_table_std, overall_raw = format_latex_table("mean", "first 50 user + 50 assistant messages + chat template, with standard deviation", show_std=True)
    mean_table_simple, _ = format_latex_table("mean", "first 50 user + 50 assistant messages + chat template, simplified", show_std=False)

    if overall_raw:
        overall_avg = statistics.mean(overall_raw)
        overall_std = statistics.stdev(overall_raw) if len(overall_raw) > 1 else 0.0
        overall_median = statistics.median(overall_raw)
        print(f"\nOverall Stats across all models and regimes:")
        print(f"Mean: {overall_avg:.1f} +/- {overall_std:.1f}")
        print(f"Median: {overall_median:.1f}\n")

    return mean_table_simple + "\n\n" + mean_table_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=Path, required=True, help="Directory containg the model run folders with 'chat' subfolders")
    parser.add_argument("--cache_file", type=Path, default="local/plots/tokens_cache.json", help="Path to cache token statistics")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the generated LaTeX tables")
    parser.add_argument("--recompute", action="store_true", help="Recompute tokens even if cached values exist")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    latex_str = generate_table(RUNS_DIR=args.runs_dir, CACHE_FILE=args.cache_file, recompute=args.recompute)
    with open(args.output, "w", encoding='utf-8') as f:
        f.write(latex_str)

    print(f"Table successfully written to {args.output}")
    