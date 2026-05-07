#!/usr/bin/env python3
"""
Script to run LLM judge evaluation on model responses.

This script:
1. Loads all JSONL files from the input directory (ignoring logs folder)
2. Extracts responses and constraints from each file
3. Sends them to the judge model for evaluation
4. Saves judgments to corresponding output files with the same names
"""

import argparse
import json
import os
from pathlib import Path
import random

# Set VLLM_USE_V1 before importing vllm
# os.environ["VLLM_USE_V1"] = "1"

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


def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_existing_output(file_path):
    """Load an existing output JSONL file, tolerating a truncated last line."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"  Warning: stopping at malformed line {line_num} in existing output "
                    f"{Path(file_path).name}"
                )
                break
    return data


def get_item_signature(item):
    """Return the fields used to verify an output line matches an input line."""
    return (
        item.get('response', item.get('answer', '')),
        item.get('constraint'),
    )


def get_resume_state(data, output_file):
    """Return the valid existing prefix for an output file and the resume index."""
    if not output_file.exists():
        return [], 0

    existing_output = load_existing_output(output_file)
    valid_prefix = []

    for index, existing_item in enumerate(existing_output):
        if index >= len(data):
            raise ValueError(
                f"Existing output has more lines than input for {output_file.name}. "
                "Remove the output file or use a different output directory."
            )

        if 'judge' not in existing_item:
            print(
                f"  Found incomplete output at line {index + 1} in {output_file.name}; "
                "resuming from there"
            )
            break

        if get_item_signature(existing_item) != get_item_signature(data[index]):
            raise ValueError(
                f"Existing output does not match input at line {index + 1} for {output_file.name}. "
                "Remove the output file or use a different output directory."
            )

        valid_prefix.append(existing_item)

    return valid_prefix, len(valid_prefix)


def request_input_format(user_prompt, tokenizer):
    """Format the prompt using the model's chat template."""
    system_prompt = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        #enable_thinking=False
    )
    return text


# Template for judging - can be easily modified
JUDGE_TEMPLATE = """An assistant has been asked to perform a task. Your job is to assess whether the provided answer satisfies a given constraint. You may first reason about both the constraint and the answer. At the end, present your final verdict as either “Final Verdict: [[Yes]]” if the answer satisfies the constraint, or “Final Verdict: [[No]]” if it does not.

Does the following answer satisfy the constraint?

Answer: 
"{answer}"

Constraint:
"{constraint}"
"""


def process_file(file_path, llm, tokenizer, sampling_params, output_dir):
    """
    Process a single input file and generate judge verdicts.
    
    Args:
        file_path: Path to the input JSONL file
        llm: The LLM judge model instance
        tokenizer: The tokenizer instance
        sampling_params: Sampling parameters for generation
        output_dir: Directory to save output file
    
    Returns:
        Path to the output file
    """
    print(f"\nProcessing file: {file_path.name}")
    
    # Load data
    data = load_jsonl(file_path)
    print(f"  Loaded {len(data)} entries")

    output_file = output_dir / file_path.name
    existing_output, resume_index = get_resume_state(data, output_file)

    if existing_output:
        print(f"  Found {len(existing_output)} existing judgments in: {output_file}")

    if resume_index == len(data):
        print(f"  Output already complete, skipping: {output_file}")
        return output_file

    remaining_data = data[resume_index:]
    print(f"  Generating judgments for {len(remaining_data)} missing entries")
    
    # Format prompts for judging
    prompts = []
    for item in remaining_data:
        answer = item.get('response', item.get('answer', ''))
        constraint = item['constraint']
        
        user_prompt = JUDGE_TEMPLATE.format(answer=answer, constraint=constraint)
        formatted_prompt = request_input_format(user_prompt, tokenizer)
        prompts.append(formatted_prompt)
    
    print(f"  Generating judgments...")
    
    # Generate judgments
    outputs = llm.generate(prompts, sampling_params)
    
    # Prepare output data
    output_data = list(existing_output)
    for i, output in enumerate(outputs):
        judgment_text = output.outputs[0].text
        
        # Combine original data with judgment
        output_item = remaining_data[i].copy()
        output_item['judge'] = judgment_text
        output_item['judge_prompt'] = output.prompt
        
        output_data.append(output_item)
    
    # Save to output file with same name
    save_jsonl(output_data, output_file)
    
    print(f"  Saved {len(output_data)} judgments to: {output_file}")
    
    return output_file


def main(args):
    random.seed(args.seed)
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSONL files in input directory (excluding logs folder)
    input_files = []
    for file_path in sorted(input_dir.glob("*.jsonl")):
        # Skip files in logs subdirectory
        if 'logs' in file_path.parts:
            continue
        input_files.append(file_path)
    
    if not input_files:
        raise ValueError(f"No JSONL files found in {input_dir}")
    
    print("=" * 80)
    print("LLM Judge Evaluation - Task-Constraint Pairs")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Judge model:      {args.model_path}")
    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f.name}")
    print()
    
    # Load the model and tokenizer
    max_len = 50000  # 8192  # Minimal context for V1 engine with 397B model
    print(f"Loading judge model with max_model_len={max_len}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_len,
        # gpu_memory_utilization=0.75,  # Very conservative for 397B model
        # dtype="auto",
        # max_num_seqs=16  # Minimal for V1 engine warmup
    )
    print("Judge model loaded successfully!")
    
    # Setup sampling parameters
    sampling_params = SamplingParams(max_tokens=max_len)
    
    # Process each file
    output_files = []
    for input_file in input_files:
        output_file = process_file(
            input_file,
            llm,
            tokenizer,
            sampling_params,
            output_dir
        )
        output_files.append(output_file)
    
    print()
    print("=" * 80)
    print("Judge Evaluation Complete!")
    print("=" * 80)
    print(f"Processed {len(input_files)} files")
    print(f"Output files:")
    for f in output_files:
        print(f"  - {f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM judge evaluation on model responses"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing input JSONL files with model responses'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save judgment results'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the judge model'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Tensor parallel size for model (default: 1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    main(args)
