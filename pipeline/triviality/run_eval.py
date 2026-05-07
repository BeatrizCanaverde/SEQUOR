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


def load_tasks(file_path):
    data = load_jsonl(file_path)
    tasks = []
    for item in data:
        task_prompt = item['prompt']
        tasks.append(task_prompt)
    return tasks


def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text



def main(args):

    random.seed(args.seed)

    # Load constraints and tasks
    tasks = load_tasks(args.tasks_file)
    
    print(f"Loaded {len(tasks)} tasks")

    # Load the model and tokenizer
    max_len = 8192    # 4096   # 8192    # 2048
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, seed=args.seed, tensor_parallel_size=args.tensor_parallel_size, max_model_len=max_len)

    prompts = []
    for task_prompt in tasks:
        prompt = request_input_format(task_prompt, tokenizer)
        prompts.append(prompt)

    print(f"Loaded {len(prompts)} entries to process...\n\n")
    print(f"Sample 0: {prompts[0]}")

    sampling_params = SamplingParams(max_tokens=max_len) #, stop=["<|eot_id|>"])  temperature=0.9, top_p=0.95, 
    outputs = llm.generate(prompts, sampling_params)

    output_path = Path(args.output_dir) / f"{model_path.replace('/', '__')}.jsonl"
    with open(output_path, 'w') as out:
        for i, output in enumerate(outputs):
            out_txt = output.outputs[0].text
            data = {'prompt': output.prompt}
            data['answer'] = out_txt
            data['task'] = tasks[i]
            out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Outputted the results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with constraints and tasks")
    parser.add_argument('--tasks-file', type=str, default='triviality/data/tasks.jsonl', help='Path to tasks JSONL file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output results.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size for model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()
    main(args)
