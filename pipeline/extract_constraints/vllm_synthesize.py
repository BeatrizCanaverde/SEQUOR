import argparse
import json
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset

from pipeline.extract_constraints.prompt_templates import one_turn_template, all_turns_template, all_turns_revised_template


def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text


def load_instruction_dataset(args):

    if args.dataset == "lmsys-chat-1m":
        instruction_dataset = load_dataset("lmsys/lmsys-chat-1m")['train']
    elif args.dataset == "WildChat-4.8M":
        instruction_dataset = load_dataset("allenai/WildChat-1M")['train']

    instruction_dataset = instruction_dataset.filter(lambda x: x["language"] == "English")
    if args.sample_size > 0:
        total = len(instruction_dataset)
        if args.sample_size < total:
           instruction_dataset = instruction_dataset.select(range(args.sample_size))

    user_turns_list = []
    for conversation in instruction_dataset['conversation']:
        if args.template == "one_turn":
            # only consider the first user turn
            user_turns = [turn["content"] for turn in conversation if turn["role"] == "user"]
            user_turns_list.append(user_turns[0].strip())
        elif args.template == "all_turns" or args.template == "all_turns_revised":
            # consider all user turns
            user_turns = [turn["content"] for turn in conversation if turn["role"] == "user"]
            user_turns_combined = ""
            for idx, user_turn in enumerate(user_turns):
                user_turn = user_turn.strip()
                user_turns_combined += f"Turn {idx+1}:\n{user_turn}\n\n"
            user_turns_combined = user_turns_combined.strip()
            user_turns_list.append(user_turns_combined)

    return user_turns_list


def main(args):

    random.seed(args.seed)

    # Load the appropriate dataset
    if args.dataset == "lmsys-chat-1m" or args.dataset == "WildChat-4.8M":
        dataset = load_instruction_dataset(args)
    else:
        raise ValueError("Invalid dataset.")

    # Load the appropriate template
    if args.template == "one_turn":
        template = one_turn_template
    elif args.template == "all_turns":
        template = all_turns_template
    elif args.template == "all_turns_revised":
        template = all_turns_revised_template
    elif args.template == "number_limitations":
        template = number_limitations_template
    else:
        print(args.template)
        raise ValueError("Invalid template type.")

    # Load the model and tokenizer
    max_len = 50000   # 8192    # 2048
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, seed=args.seed, tensor_parallel_size=2, max_model_len=max_len, max_num_seqs=2, gpu_memory_utilization=0.8)

    prompts = []
    for user_turn in dataset:
        user_prompt = template.format(user_turn=user_turn)
        prompt = request_input_format(user_prompt, tokenizer)
        prompts.append(prompt)

    print(f"Loaded {len(prompts)} entries to process...\n\n")
    print(f"Sample 0: {prompts[0]}")

    sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=max_len, stop=["<|eot_id|>"])
    outputs = llm.generate(prompts, sampling_params)

    with open(args.output_path, 'w') as out:
        for i, output in enumerate(outputs):
            out_txt = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            data = {'prompt': output.prompt, "input user_turn": dataset[i], "finish_reason": finish_reason}
            data['output'] = out_txt
            out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Outputted the results to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a dataset using a specified model and template.")
    parser.add_argument('--dataset', type=str, required=True, choices=['lmsys-chat-1m', 'WildChat-4.8M'], help='Dataset to be analyzed.')
    parser.add_argument('--sample_size', type=int, default=0, help='Number of samples to process from the dataset; Set it to 0 if you want to use the full set of 200k personas.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file.')
    parser.add_argument('--template', type=str, required=True, choices=['one_turn', 'all_turns', 'all_turns_revised'], help=("Prompt templates. You can also add more customized templates in code/templates.py"))
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()
    main(args)
