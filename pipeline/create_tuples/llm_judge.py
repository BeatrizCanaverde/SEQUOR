import argparse
import json
from pathlib import Path
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_jsonl(file_path):
    """Load all lines from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_satisfied_keys(file_path):
    """Load satisfied keys from a text file (one key per line)."""
    if file_path is None or not Path(file_path).exists():
        return set()
    
    keys = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                keys.add(line)
    return keys


def create_tuple_task_key(constraints, task):
    """Create a unique key for a tuple-task pair."""
    # Sort constraints to ensure consistent keys
    constraints_str = '|||'.join(sorted(constraints))
    return f"{constraints_str}:::{task}"


def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text


# Template for judging individual constraints
JUDGE_TEMPLATE = """An assistant has been asked to perform a task. Your job is to assess whether the provided answer satisfies a given constraint. You may first reason about both the constraint and the answer. At the end, present your final verdict as either "Final Verdict: [[Yes]]" if the answer satisfies the constraint, or "Final Verdict: [[No]]" if it does not.

Does the following answer satisfy the constraint?

Answer: 
"{answer}"

Constraint:
"{constraint}"
"""


def main(args):

    random.seed(args.seed)

    # Load satisfied keys to exclude (if provided)
    satisfied_keys = load_satisfied_keys(args.exclude_satisfied_pairs)
    if satisfied_keys:
        print(f"Loaded {len(satisfied_keys)} satisfied tuple-task pairs to exclude")
    
    # Load data
    data = load_jsonl(args.data_file)
    print(f"Loaded {len(data)} data instances")
    
    # Filter out already satisfied pairs
    if satisfied_keys:
        original_count = len(data)
        data = [item for item in data 
                if create_tuple_task_key(item['constraints'], item['task']) not in satisfied_keys]
        print(f"After filtering: {len(data)} data instances remain ({original_count - len(data)} excluded)")

    # Check for existing output
    output_path = Path(args.output_dir) / f"{Path(args.data_file).stem}.jsonl"
    if output_path.exists() and not args.overwrite:
        print(f"Output file already exists: {output_path}")
        print("Use --overwrite to regenerate, or delete the file manually.")
        return

    # Load the model and tokenizer
    max_len = 80000  # Keep large context window for long prompts
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path, 
        seed=args.seed, 
        tensor_parallel_size=args.tensor_parallel_size, 
        pipeline_parallel_size=args.pipeline_parallel_size, 
        max_model_len=max_len,
        gpu_memory_utilization=0.85,
        max_num_seqs=256,
        enable_prefix_caching=True  # Enable for better performance
    )

    # Build ALL prompts at once (much faster than batching)
    print(f"\nBuilding prompts for {len(data)} instances with {len(data[0]['constraints'])} constraints each...")
    prompts = []
    prompts_meta = []  # (data_index, constraint_index, constraint_text)
    
    for data_idx, item in enumerate(data):
        answer = item['answer']
        constraints = item['constraints']
        task = item['task']
        
        # Create a prompt for each constraint
        for constraint_idx, constraint in enumerate(constraints):
            user_prompt = JUDGE_TEMPLATE.format(answer=answer, constraint=constraint)
            prompt = request_input_format(user_prompt, tokenizer)
            prompts.append(prompt)
            prompts_meta.append((data_idx, constraint_idx, constraint))
    
    print(f"Generated {len(prompts)} total prompts to evaluate")
    print(f"Sample prompt 0:\n{prompts[0][:500]}...\n")
    
    # Process ALL prompts in ONE call (much faster than small batches!)
    print(f"{'='*80}")
    print(f"Running inference on ALL {len(prompts)} prompts at once...")
    print(f"This processes everything in parallel - MUCH faster than batching!")
    print(f"{'='*80}\n")
    
    sampling_params = SamplingParams(max_tokens=8192)  # Judge outputs with reasoning
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"\n{'='*80}")
    print(f"Inference complete! Organizing and saving results...")
    print(f"{'='*80}\n")
    
    # Organize results
    results = []
    for i, item in enumerate(data):
        result = {
            'prompt': item['prompt'],
            'answer': item['answer'],
            'task': item['task'],
            'constraints': item['constraints'],
            'judge_outputs': [None] * len(item['constraints'])
        }
        results.append(result)
    
    # Fill in judge outputs
    for (data_idx, constraint_idx, constraint_text), output in zip(prompts_meta, outputs):
        results[data_idx]['judge_outputs'][constraint_idx] = output.outputs[0].text
    
    # Write results in chunks to save memory and provide progress updates
    print(f"Writing results to: {output_path}")
    chunk_size = 10000
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for i in range(0, len(results), chunk_size):
            chunk = results[i:i+chunk_size]
            for result in chunk:
                output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            output_file.flush()
            print(f"  Saved {min(i+chunk_size, len(results))}/{len(results)} results...")

    print(f"\n{'='*80}")
    print(f"All processing complete!")
    print(f"Processed {len(data)} instances ({len(prompts)} constraint checks)")
    print(f"Output saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers against multiple constraints using LLM judge (fast single-pass mode).")
    parser.add_argument('--data-file', type=str, required=True, help='Path to JSONL file with data to judge.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output results.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size for model.')
    parser.add_argument('--pipeline-parallel-size', type=int, default=1, help='Pipeline parallel size for model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file if it exists.')
    parser.add_argument('--exclude-satisfied-pairs', type=str, default=None, help='Path to file containing satisfied pair keys to exclude (one per line).')
    # Legacy args kept for backward compatibility but ignored
    parser.add_argument('--batch-size', type=int, default=1000, help='(Ignored - kept for compatibility)')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='(Ignored - kept for compatibility)')
    args = parser.parse_args()

    main(args)
