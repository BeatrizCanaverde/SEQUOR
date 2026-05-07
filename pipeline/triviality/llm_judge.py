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


#  in the context of the task
judge_template = """An assistant has been asked to perform a task. Your job is to assess whether the provided answer satisfies a given constraint. You may first reason about both the constraint and the answer. At the end, present your final verdict as either “Final Verdict: [[Yes]]” if the answer satisfies the constraint, or “Final Verdict: [[No]]” if it does not.

Does the following answer satisfy the constraint?

Answer: 
"{answer}"

Constraint:
"{constraint}"
"""


def main(args):

    random.seed(args.seed)

    # Load constraints and data
    constraints = load_jsonl(args.constraints_file)
    data = load_jsonl(args.data_file)
    
    print(f"Loaded {len(constraints)} constraints")
    print(f"Loaded {len(data)} data instances")

    # Load the model and tokenizer
    max_len = 100000    # 4096    # 8192    # 2048
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, seed=args.seed, tensor_parallel_size=args.tensor_parallel_size, max_model_len=max_len)  # , max_num_seqs=2, gpu_memory_utilization=0.8)

    prompts = []
    constraints_tasks = []
    for item in data:
        answer = item['answer']
        task = item['task']
        for line in constraints:
            constraint = line['text']
            user_prompt = judge_template.format(answer=answer, constraint=constraint)
            prompt = request_input_format(user_prompt, tokenizer)
            prompts.append(prompt)
            constraints_tasks.append((constraint, task))

    print(f"Loaded {len(prompts)} entries to process...\n\n")
    print(f"Sample 0: {prompts[0]}")

    sampling_params = SamplingParams(max_tokens=max_len) #, stop=["<|eot_id|>"])  temperature=0.9, top_p=0.95, 
    outputs = llm.generate(prompts, sampling_params)

    data_name = Path(args.data_file).name
    output_path = Path(args.output_dir) / data_name  # f"{model_path.replace('/', '__')}.jsonl"
    with open(output_path, 'w') as out:
        for i, output in enumerate(outputs):
            out_txt = output.outputs[0].text
            out_data = {'prompt': output.prompt,
                        'judge': out_txt,
                        'constraint': constraints_tasks[i][0],
                        'task': constraints_tasks[i][1]}
            out.write(json.dumps(out_data, ensure_ascii=False) + '\n')

    print(f"Outputted the results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers against constraints using LLM judge.")
    parser.add_argument('--constraints-file', type=str, default='triviality/data/constraints.jsonl', help='Path to JSONL file with constraints.')
    parser.add_argument('--data-file', type=str, required=True, help='Path to JSONL file with data to judge.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output results.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size for model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()

    main(args)
