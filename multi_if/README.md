# Multi-Turn Constraint Following Evaluation

This folder contains the code for evaluating LLMs on multi-turn constraint following. It supports five regimes that differ in how constraints are introduced and evolve across turns: *single*, *tuples*, *replace*, *add*, and *everything*. It also provides utilities for scoring, plotting, and generating LaTeX tables.

## Folder Structure

`multi_if/`:
- `cli.py`: Main CLI interface for testset generation, response generation, evaluation, scoring, and plotting.
- `constraints.py`: Utilities for loading constraints and constraint groups.
- `conversation.py`: Conversation management and async API calling.
- `evaluation.py`: LLM-as-a-judge evaluation logic.
- `test_prompts.py`: Testset and prompt generation for each experiment type.
- `visualization.py`: Plotting and visualization utilities.
- `process_evals.py`: Post-processing of raw judge outputs into standardized JSONL format.
- `extract_tasks.py`: Task extraction utilities.
- `submit_generate_responses.py`: Generate model responses to the testsets.
- `submit_generate_evals.py`: Run judgments on model responses to evaluate constraint adherence.
- `submit_generate_task_evals.py`: Run judgments on model responses to evaluate whether tasks are addressed.
- `generate_latex_tables.py`: Generate LaTeX tables for per-turn accuracy difference between last and first and best and worst turns.
- `generate_token_length_table.py`: Generate LaTeX tables reporting average token counts per conversation.
- `create_multi_slope_plots.py`: Generate slope plots comparing turn-1 vs turn-50 accuracy per regime.
- `plot_bootstrap_ci.py`: Generate bootstrap confidence interval plots for each regime.
- `constraint_plotting.py`: Generate circle distribution plots for constraint categories.


## Create Testsets

There are five regimes, each with a dedicated testset generation command. All commands expect a directory of task files and a constraints file.

### Single constraint

One constraint is given in the first turn and must be followed thereafter.

```bash
python multi_if/cli.py generate_testset_single \
    --testset_dir local/testsets/single \
    --tasks_path data/tasks \
    --constraints_path data/constraints/single.jsonl \
    --num_prompts 50 \
    --num_constraints 500 \
    --repetitions 1 \
    --replace False
```

### Tuples

A tuple of three constraints are given in the first turn and must be followed thereafter.

```bash
python multi_if/cli.py generate_testset_tuples \
    --testset_dir local/testsets/tuples \
    --tasks_path data/tasks \
    --constraints_path data/constraints/3.jsonl \
    --num_prompts 50 \
    --num_conversations 500 \
    --repetitions 1 \
    --replace false
```

### Replace

A constraint is given in the first turn and replaced every `step_size` turns. Each constraint must be followed until it is replaced. We consider two `step_size` values: 5 and 10.

```bash
python multi_if/cli.py generate_testset_replace \
    --testset_dir local/testsets/replace \
    --tasks_path data/tasks \
    --constraints_path data/constraints/single.jsonl \
    --step_sizes "[5,10]" \
    --num_conversations 500 \
    --num_prompts 50 \
    --repetitions 1 \
    --replace false
```

### Add

A constraint is given in the first turn, and additional ones are added every `step_size` turns, up to a maximum of three. Constraints accumulate; once introduced, they must be followed thereafter. We consider two `step_size` values: 5 and 10.

```bash
python multi_if/cli.py generate_testset_add \
    --testset_dir local/testsets/add \
    --tasks_path data/tasks \
    --constraints_path data/constraints/3.jsonl \
    --step_sizes "[5,10]" \
    --num_conversations 500 \
    --num_prompts 50 \
    --repetitions 1 \
    --replace false
```

### Everything

A mixture of the previous regimes. After a random number of turns (between `min_step_size` and `max_step_size`), up to three constraints are given, randomly accumulating with or replacing earlier ones.

```bash
python multi_if/cli.py generate_testset_everything \
    --testset_dir local/testsets/everything \
    --tasks_path data/tasks \
    --constraints_path data/constraints \
    --num_conversations 500 \
    --min_step_size 1 \
    --max_step_size 5 \
    --min_constraints 1 \
    --max_constraints 3 \
    --num_prompts 50 \
    --repetitions 1 \
    --replace false
```

## Generate Model Responses

For each regime and model, submit a response generation job using `submit_generate_responses.py`. Below are example calls for a local model (served via vLLM) and an API model (via OpenRouter).

**Local model:**
```bash
python multi_if/submit_generate_responses.py \
    --job.run_dir local/runs/Qwen__Qwen3-4B-Instruct-2507_add \
    --job.testset_dir local/testsets/add \
    --job.model_path Qwen/Qwen3-4B-Instruct-2507 \
    --job.max_turns 50 \
    --job.context_length 256000 \
    --job.tp_size 1 \
    --job.dp_size 1
```

**API model:**
```bash
python multi_if/submit_generate_responses.py \
    --job.run_dir local/runs/google__gemini-3.1-flash-lite-preview_replace \
    --job.testset_dir local/testsets/replace \
    --job.model_path google/gemini-3.1-flash-lite-preview \
    --job.api_url https://openrouter.ai/api/v1 \
    --job.max_turns 50 \
    --job.batch_size 50 \
    --job.max_concurrent 20
```

## Evaluate Model Responses

### Constraint Adherence

For each run, submit an evaluation job with `submit_generate_evals.py`. The judge model (e.g., `openai__gpt-oss-120b`) evaluates constraint adherence turn by turn, considering one constraint at a time.

```bash
python multi_if/submit_generate_evals.py \
    --job.run_dir local/runs/Qwen__Qwen3-4B-Instruct-2507_add \
    --job.testset_dir local/testsets/add \
    --job.output_dir local/evals/Qwen__Qwen3-4B-Instruct-2507_add \
    --job.model_path openai/gpt-oss-120b \
    --job.max_turns 50 \
    --job.tp_size 2 \
    --job.dp_size 1
```

### Task Completion

For each run, submit a task evaluation job with `submit_generate_task_evals.py`. The judge model evaluates, turn by turn, whether each model response meaningfully addresses the given task without seeing the associated constraints.

```bash
python multi_if/submit_generate_task_evals.py \
    --job.run_dir local/runs/Qwen__Qwen3-4B-Instruct-2507_add \
    --job.testset_dir local/testsets/add \
    --job.output_dir local/task_evals/Qwen__Qwen3-4B-Instruct-2507_add \
    --job.model_path openai/gpt-oss-120b \
    --job.max_turns 50 \
    --job.tp_size 2 \
    --job.dp_size 1
```

## Process Judge Answers

Process raw judge outputs by extracting the verdicts:

```bash
python multi_if/process_evals.py \
    --src-root local/evals \
    --dst-root local/processed_evals
```


## Metrics and Plots

### Per-run evaluation scores

Compute per-run evaluation scores from all processed runs:

```bash
python multi_if/cli.py compute_eval_scores_tree \
    --root_dir local/processed_evals \
    --output_dir local/eval_scores
```

### Compare models across regimes

Generate all per-regime comparison plots and summary files. Expects one subfolder per model under `local/eval_scores`, named `<model>_<experiment>` (e.g., `Qwen__Qwen3-4B-Instruct-2507_add`).

```bash
python multi_if/cli.py compare_models_scores_by_experiment \
    --scores_dir local/eval_scores \
    --output_dir local/plots
```

### Per-turn accuracy comparison

Generate aggregated and best-model per-turn accuracy line plots and heatmaps across all regimes:

```bash
python multi_if/cli.py plot_per_turn_accuracy_comparison \
    --scores_dir local/eval_scores \
    --output_path local/plots/per_turn_accuracy_comparison
```

### Slope plots (turn 1 vs turn 50)

Generate slope plots comparing turn-1 vs turn-50 accuracy across all regimes:

```bash
python multi_if/create_multi_slope_plots.py \
    --scores_dir local/eval_scores \
    --output_dir local/plots
```

### Bootstrap confidence intervals

Generate bootstrap CI plots for the average across models and the best model, for each regime:

```bash
python multi_if/plot_bootstrap_ci.py \
    --input local/eval_scores \
    --processed_evals local/processed_evals \
    --output local/plots/bootstrap_ci_multi.pdf
```

### LaTeX tables

Generate LaTeX tables for two per-turn statistics across all models and regimes:
- Difference in per-turn accuracy between the last and first turns (%).
- Difference in per-turn accuracy between the best and worst turns (%).

```bash
python multi_if/generate_latex_tables.py \
    --input local/eval_scores \
    --output local/plots/latex_tables.txt
```

Generate LaTeX tables reporting average token counts per conversation (simplified and with standard deviation):

```bash
python multi_if/generate_token_length_table.py \
    --runs_dir local/runs \
    --cache_file local/plots/tokens_cache.json \
    --output local/plots/tokens_table.txt
```
