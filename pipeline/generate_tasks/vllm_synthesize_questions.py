import argparse
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from generate_tasks.prompt_templates import activities_template


def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def extract_questions(response_text):
    # Split by \n### (at least one newline, at least three #, optional whitespace)
    import re
    parts = re.split(r"\n#{3,}\s*", response_text)
    questions = [p.strip() for p in parts[1:] if p.strip()]
    return questions


def main(args):
    # Load extracted activities data
    with open(args.input_path, "r", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=args.tensor_parallel_size) # set based on available GPUs

    prompts = []
    persona_activity_pairs = []

    for entry in data:
        persona = entry["input persona"]
        activities = entry["activities"]
        for activity in activities:
            user_prompt = activities_template.format(persona=persona, activity=activity)
            prompt = request_input_format(user_prompt, tokenizer)
            prompts.append(prompt)
            persona_activity_pairs.append((persona, activity, user_prompt))

    print(f"Loaded {len(prompts)} persona-activity pairs to process...\n\n")
    print(f"Sample prompt:\n{prompts[0]}")

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192, stop=["<|eot_id|>"])
    outputs = llm.generate(prompts, sampling_params)

    with open(args.output_path, 'w', encoding='utf-8') as out:
        for i, output in enumerate(outputs):
            out_txt = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            persona, activity, user_prompt = persona_activity_pairs[i]
            #questions = extract_questions(out_txt)
            result = {
                "prompt": user_prompt,
                "input persona": persona,
                "finish_reason": finish_reason,
                "questions": out_txt.strip(),
            }
            out.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Outputted the results to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate activity-based questions for personas using a specified model.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the extracted activities jsonl file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPU tensor parallel partitions.')
    args = parser.parse_args()
    main(args)