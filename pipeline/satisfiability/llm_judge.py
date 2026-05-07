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


def load_constraints(file_path):
    data = load_jsonl(file_path)
    constraints = []
    for item in data:
        constraint_text = item['text']
        constraints.append(constraint_text)
    return constraints


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


judge_template = """Identify whether a constraint is relevant to a task.

A task is a directive that specifies an action or goal. It tells a model to provide information or do something, such as "What is the capital of France?", "Summarize this text", "Translate this sentence", "Who are you?", "Generate a list of ideas", or "Why is the sky blue?".

A constraint is a restriction or condition that limits how the model should generate its output, rather than what task it performs. It guides the form, style, or structure of the response — ensuring it adheres to specific requirements or rules. 

We classify constraints into four main categories: 
1) Linguistic Guidelines: These dictate the use of particular language structures and terms, including grammatical styles, syntax, and specific dialects, like "Victorian English" or "technical jargon"; 
2) Style Rules: These direct the overall tone and audience of the text, varying from formal to persuasive or sophisticated, as in writing with a "respectful tone" or for "a young audience"; 
3) Format Specifications: These instruct the LLM on the structural presentation of its response, such as "write your answer as a sonnet" or "list ideas bullet-wise"; 
4) Number Limitations: These involve numeric-related instructions, like producing "a 500-word essay" or presenting "three arguments for your answer".

Below, you are given a task and a constraint. To determine whether the constraint is relevant to the task, answer seperately each of these questions with either [[Yes]] or [[No]]:
1) Is the constraint actually a restriction or condition that limits how the model should generate its output to the task?
2) Does the constraint target a different question, topic, or domain than the task itself?
3) Is the constraint applicable to the type of output the task requires?
4) Does the constraint fall within one of the four defined categories above?

You can first reason about the task and the constraint. Output only a valid JSON with this structure:
{{
    "reasoning": "write your reasoning here",
    "question 1": "[[Yes/No]]",
    "question 2": "[[Yes/No]]",
    "question 3": "[[Yes/No]]",
    "question 4": "[[Yes/No]]"
}}


Task:
{task}

Constraint:
{constraint}
"""


def main(args):

    random.seed(args.seed)

    # Load constraints and tasks
    constraints = load_constraints(args.constraints_file)
    tasks = load_tasks(args.tasks_file)
    
    print(f"Loaded {len(constraints)} constraints")
    print(f"Loaded {len(tasks)} tasks")
    
    # Load the model and tokenizer
    max_len = 100000    # 4096    # 8192    # 2048
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, seed=args.seed, tensor_parallel_size=args.tensor_parallel_size, max_model_len=max_len)  # , max_num_seqs=2, gpu_memory_utilization=0.8)

    prompts = []
    data = []
    for task in tasks:
        for constraint in constraints:
            user_prompt = judge_template.format(task=task, constraint=constraint)
            prompt = request_input_format(user_prompt, tokenizer)
            prompts.append(prompt)
            data.append({'task': task, 'constraint': constraint})

    print(f"Loaded {len(prompts)} entries to process...\n\n")
    print(f"Sample 0: {prompts[0]}")

    sampling_params = SamplingParams(max_tokens=max_len) # temperature=0.9, top_p=0.95, max_tokens=max_len) #, stop=["<|eot_id|>"])
    outputs = llm.generate(prompts, sampling_params)

    output_path = Path(args.output_dir) / f"{model_path.replace('/', '__')}.jsonl"
    with open(output_path, 'w') as out:
        for i, output in enumerate(outputs):
            out_txt = output.outputs[0].text
            out_data = {'prompt': output.prompt,
                        'judge': out_txt,
                        'constraint': data[i]['constraint'],
                        'task': data[i]['task']}
            out.write(json.dumps(out_data, ensure_ascii=False) + '\n')

    print(f"Outputted the results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers against constraints using LLM judge.")
    parser.add_argument('--constraints-file', type=str, required=True, help='Path to JSONL file with constraints.')
    parser.add_argument('--tasks-file', type=str, required=True, help='Path to JSONL file with tasks.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output results.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--tensor-parallel-size', type=int, default=4, help='Tensor parallel size for the model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()

    main(args)
