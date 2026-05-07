# Sequor - Constraints Collection, Task Generation, Evaluation with LLM-as-a-Judge

This folder contains the code for extracting constraints from real-world conversational data, filtering them, and combining different evaluation phases for LLM constraint analysis. It additionally contains the code for synthetically generating tasks and running evaluations with LLMs-as-Judges in order to verify that models can reliably judge constraint adherence.

## Folder Structure

`pipeline/`:
  - `generate_tasks/`: Synthetic task generation based on personas.
  - `extract_constraints/`: Constraint extraction logic via vLLM.
  - `filter_constraints/`: Filtering constraints (language deduplication, minhash deduplication, badwords filtering).
  - `satisfiability/`: LLM judge scripts and analysis for constraint satisfiability.
  - `triviality/`: Generation and judge scripts for identifying trivial constraints.
  - `subjectivity/`: Generation and judge scripts for evaluating constraint subjectivity.
  - `combine_phases/`: Code to intersect filtering phases (satisfiability, triviality, and subjectivity) by thresholding.
  - `best_judge/`: Evaluating the performance of various judge models.
  - `create_tuples/`: Generating sets of N-tuples for multi-constraint tasks.


## Synthetic Task Generation

We first explain how to synthetically generate tasks, since they are required for the subsequent phases of constraint filtering. This phase takes personas to generate realistic agendas, creates specific activities and diverse questions, and then uses them to create multi-turn conversations.

```bash
# Generate a synthetic agenda based on personas
bash pipeline/generate_tasks/generate_agenda.sh

# Generate specific activities from the generated agenda
python pipeline/generate_tasks/extract_activities.py \
    --input_file tasks/vllm_agenda.jsonl \
    --output_file tasks/extracted_activities.jsonl

# Generate tasks/questions for each activity
bash pipeline/generate_tasks/generate_tasks.sh

# Expand the tasks/questions into full conversational turns
python pipeline/generate_tasks/create_conversations.py \
    --input_file tasks/vllm_questions.jsonl \
    --output_dir tasks/conversations
```

## Running the Constraint Collection Pipeline

The full execution flow follows the steps below.

1. **Extract constraints from conversations**
```bash
# Use vLLM to extract constraints based on conversational data
bash pipeline/extract_constraints/run_vllm_synthesize.sh

# Parse and format the raw extracted constraints into standard JSONL format
python pipeline/extract_constraints/parse_constraints.py \
    --constraints-file output_file_from_previous_bash.jsonl \
    --output-dir constraints_extracted
```

2. **Filtering constraints**
Run the scripts in `pipeline/filter_constraints/` (Language filter -> Minhash deduplication -> Badwords filter).

```bash
# Convert raw JSON constraints into JSONL format for downstream processing
python pipeline/filter_constraints/json_to_jsonl.py \
    --input_json_path constraints_extracted/parsed_constraints_path.json \
    --output_jsonl_path filter_data/constraints.jsonl

# Filter constraints to ensure they are in the target language (English)
python pipeline/filter_constraints/language_filter.py \
    --input_jsonl_path filter_data/constraints.jsonl \
    --output_dir filter_data

# Deduplicate similar constraints using MinHash to maintain diversity
python pipeline/filter_constraints/minhash_deduplication.py \
    --input_jsonl_path filter_data/language_filtered.jsonl \
    --output_dir filter_data

# Remove constraints containing inappropriate or bad words
python pipeline/filter_constraints/badwords_filter.py \
    --input_jsonl_path filter_data/minhash_deduplicated.jsonl \
    --output_dir filter_data
```

3. **Satisfiability phase**

This phase pairs the filtered constraints with the previously generated tasks, uses LLM judges to determine whether each constraint can be satisfied for the given task, and plots the satisfiability rates. This process retains constraints that are satisfiable across multiple contexts.

```bash
# Pair constraints with tasks to create a dataset for satisfiability analysis
python pipeline/satisfiability/data.py \
    --constraints-file filter_data/badwords_filtered.jsonl \
    --tasks-dir data/tasks \
    --output-dir satisfiability/data \
    --num-constraints 0

# Run the LLM judges to determine if each constraint can be satisfied in its paired task
bash pipeline/satisfiability/llm_judge.sh

# Generate plots showing the satisfiability rates across the constraints and tasks
bash pipeline/satisfiability/plot_constraints_vs_tasks.sh
```

4. **Triviality phase**

This phase evaluates if the constraints are trivially followed without active effort. We pair constraints with tasks, collect model responses to the tasks without specifying any constraint, and use LLM judges to determine whether the constraint is nevertheless satisfied. Afterwards, we plot their agreement.

```bash
# Prepare pairings of tasks and constraints specifically for the triviality analysis
python pipeline/triviality/data.py \
    --constraints-file filter_data/badwords_filtered.jsonl \
    --tasks-dir data/tasks \
    --num-tasks-per-file 1 \
    --output-dir triviality/data

# Sample model responses
bash pipeline/triviality/run_eval.sh

# Run the LLM judges to check if constraints are naturally fulfilled without explicit prompting
bash pipeline/triviality/llm_judge.sh

# Plot the agreement of the judge models on constraint triviality
bash pipeline/triviality/plot_judge_agreement.sh
```

5. **Subjectivity phase**

This phase verifies the objectivity of constraints. It employs multiple LLM judges to review model responses and calculates their level of agreement on whether constraints are followed or not. High agreement implies the constraint is objective and unambiguously verifiable, whereas low agreement implies a subjective constraint.

```bash
# Prepare data for subjectivity analysis by pairing tasks and constraints
python pipeline/subjectivity/data.py \
    --constraints-file filter_data/badwords_filtered.jsonl \
    --tasks-dir data/tasks \
    --output-dir subjectivity/data \
    --num-tasks-per-input-file 1 \
    --num-task-files 4 \
    --tasks-per-file 25

# Sample model responses
bash pipeline/subjectivity/run_eval.sh

# Use multiple LLM judges to independently assess constraint satisfaction on the responses
bash pipeline/subjectivity/llm_judge.sh

# Plot the judge agreement to measure the objectivity of the constraints
bash pipeline/subjectivity/plot_judge_agreement.sh
```

6. **Combine phases**

This step merges the results of the satisfiability, triviality, and subjectivity phases. By defining customizable thresholds (e.g., passing >= 70% rating for each component), we produce the final robust collection of constraints that are satisfiable, challenging, and objective.

```bash
# Intersect results from satisfiability, triviality, and subjectivity phases based on score thresholds
python pipeline/combine_phases/intersect_constraints.py \
    --satisfiability satisfiability/constraints_vs_tasks_analysis/all_constraints_percentages.jsonl \
    --subjectivity subjectivity/judge_agreement_analysis/all_constraints_percentages.jsonl \
    --triviality triviality/all_no_analysis/all_constraints_percentages.jsonl \
    --satisfiability-threshold 70.0 \
    --subjectivity-threshold 70.0 \
    --triviality-threshold 70.0 \
    --output-dir combine_phases/results
```


## Create Tuples of Constraints

This phase groups individual constraints into $N$-tuples; in our work, we consider only tuples of 3 constraints. It first samples tasks and maps them to combinations of constraints, and checks judge agreement to filter out tuples with contradictory constraints. Then, for the retained tuples, it samples model responses for each tuple-task pair, aggregates the performance measured by different LLM judges, and performs a final threshold-based filter to obtain robust constraint tuples.

```bash
# Select a subset of tasks to be used for tuple constraint creation
python pipeline/create_tuples/data.py \
    --tasks-dir data/tasks \
    --num-tasks 100 \
    --num-tasks-per-input-file 1 \
    --output-file create_tuples/data/tasks.jsonl

# Sample tuples of constraints from the final pool, pair them with tasks, and run the LLM judges to determine if each tuple can be satisfied in its paired task
bash pipeline/create_tuples/task_tuples.sh

# Process the outputs from previous command
python pipeline/create_tuples/process_model_outputs.py \
    --input-file create_tuples/tuples/openai__gpt-oss-120b/3.jsonl \
    --output-dir create_tuples/tuples/openai__gpt-oss-120b

# Plot the initial judge agreement on whether the tuples contain compatible constraints
bash pipeline/create_tuples/plot_judge_agreement.sh

# Filter out tuples that do not meet the minimum consistency threshold among judges
python pipeline/create_tuples/filter_tuples_by_threshold.py \
    --input-file create_tuples/judge_agreement_analysis/all_tuples_yes_counts.jsonl \
    --output-dir create_tuples/filtered_tuples \
    --threshold 70

# Sample model responses for each tuple-task pair
bash pipeline/create_tuples/run_eval.sh

# Run LLM judges on the model responses to check tuple constraint adherence
bash pipeline/create_tuples/llm_judge.sh

# For each model used to collect responses, retain all tuple-task pairs for which all judges agree the answer satisfies all associated constraints
python pipeline/create_tuples/analyze_judge_outputs.py \
    --judge1-file create_tuples/outputs_judge/Qwen__Qwen3-235B-A22B-Instruct-2507-FP8/google__gemma-3-27b-it.jsonl \
    --judge2-file create_tuples/outputs_judge/openai__gpt-oss-120b/google__gemma-3-27b-it.jsonl \
    --judge3-file create_tuples/outputs_judge/zai-org__GLM-4.7-FP8/google__gemma-3-27b-it.jsonl \
    --output-dir create_tuples/judge_results/google__gemma-3-27b-it

# Aggregate the previously retained tuple-task pairs for the different models used to collect responses
python pipeline/create_tuples/aggregate_satisfied_pairs.py \
    --inputs \
        create_tuples/judge_results/google__gemma-3-27b-it/satisfied_pairs.jsonl \
        create_tuples/judge_results/allenai__Olmo-3.1-32B-Instruct/satisfied_pairs.jsonl \
        create_tuples/judge_results/meta-llama__Llama-3.1-8B-Instruct/satisfied_pairs.jsonl \
        create_tuples/judge_results/openai__gpt-oss-20b/satisfied_pairs.jsonl \
    --output create_tuples/judge_results/all_satisfied_pairs.jsonl

# Generate visualizations showing the satisfaction rates for the retained tuples
bash pipeline/create_tuples/plot_tuple_satisfiability.sh

# Apply a final threshold filter to keep only the most robust and satisfiable constraint tuples
python pipeline/create_tuples/filter_tuples_by_threshold.py \
    --input-file create_tuples/tuple_satisfiability_analysis/all_tuples_percentages.jsonl \
    --output-dir create_tuples/filtered_tuples_final \
    --threshold 70
```


## Evaluation with LLM-as-a-Judge

This phase evaluates the performance of different LLMs when acting as judges to score constraint adherence. It prepares a test set, gathers gold responses from strong proprietary models, runs various judge models over these responses, and then compares their grading accuracy to determine the most reliable judge.

```bash
# Sample a subset of tasks and constraints to construct the judge evaluation dataset
python pipeline/best_judge/data.py \
    --tasks-dir data/tasks \
    --constraints-file data/constraints/single.jsonl \
    --num-prompts 500 \
    --output-dir best_judge/data

# Generate gold responses using strong proprietary models
bash pipeline/best_judge/run_eval_api.sh

# Use different LLM judges to score the generated responses against the target constraints
bash pipeline/best_judge/llm_judge.sh

# Analyze the judge outputs to evaluate and compare the accuracy of the different judges
python pipeline/best_judge/evaluate_judges.py \
    --input-dir best_judge/outputs_judge_gpt \
    --output-dir best_judge/evaluation_results_gpt
```

