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


def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text


# Template for judging - can be easily modified
JUDGE_TEMPLATE = """An assistant has been asked to perform a task. Your job is to assess whether the provided answer satisfies a given constraint. You may first reason about both the constraint and the answer. At the end, present your final verdict as either “Final Verdict: [[Yes]]” if the answer satisfies the constraint, or “Final Verdict: [[No]]” if it does not.

Does the following answer satisfy the constraint?

Answer: 
"{answer}"

Constraint:
"{constraint}"
"""


def main(args):

    random.seed(args.seed)

    # Load data
    data = load_jsonl(args.data_file)
    
    print(f"Loaded {len(data)} data instances")

    # Load the model and tokenizer
    max_len = 80000
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path, 
        seed=args.seed, 
        tensor_parallel_size=args.tensor_parallel_size, 
        pipeline_parallel_size=args.pipeline_parallel_size, 
        max_model_len=max_len,
        gpu_memory_utilization=0.85,  # Reduce from default 0.9 to leave more memory headroom
        max_num_seqs=256  # Reduce from default 1024 to use less memory during warmup
    )

    prompts = []
    for item in data:
        answer = item['answer']
        constraint = item['constraint']
        task = item['task']

        user_prompt = JUDGE_TEMPLATE.format(answer=answer, constraint=constraint)
        prompt = request_input_format(user_prompt, tokenizer)
        prompts.append(prompt)

    print(f"Loaded {len(prompts)} entries to process...\n\n")
    print(f"Sample 0: {prompts[0]}")

    sampling_params = SamplingParams(max_tokens=max_len)  # temperature=0.9, top_p=0.95, 
    outputs = llm.generate(prompts, sampling_params)

    output_path = Path(args.output_dir) / f"{Path(args.data_file).stem}.jsonl"
    with open(output_path, 'w') as out:
        for i, output in enumerate(outputs):
            out_txt = output.outputs[0].text
            out_data = {'prompt': output.prompt,
                        'judge': out_txt,
                        'constraint': data[i]['constraint'],
                        'task': data[i]['task'],
                        'answer': data[i]['answer']}
            out.write(json.dumps(out_data, ensure_ascii=False) + '\n')

    print(f"Outputted the results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers against constraints using LLM judge for subjectivity.")
    parser.add_argument('--data-file', type=str, required=True, help='Path to JSONL file with data to judge.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output results.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size for model.')
    parser.add_argument('--pipeline-parallel-size', type=int, default=1, help='Pipeline parallel size for model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()
    main(args)
