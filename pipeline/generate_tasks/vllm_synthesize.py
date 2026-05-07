import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from generate_tasks.prompt_templates import agenda_template


def normalize_persona(raw_persona):
    """Coerce persona field to a clean string."""
    if raw_persona is None:
        return None
    if isinstance(raw_persona, list):
        return " ".join(str(p).strip() for p in raw_persona if p is not None).strip()
    return str(raw_persona).strip()


def load_personas(args):
    """Load personas with a fallback to streaming to handle schema glitches."""
    try:
        persona_dataset = load_dataset("proj-persona/PersonaHub", "elite_persona")["train"]
        if args.sample_size > 0:
            take = min(args.sample_size, len(persona_dataset))
            persona_dataset = persona_dataset.select(range(take))
        personas = [normalize_persona(p) for p in persona_dataset["persona"]]
        personas = [p for p in personas if p]
        return personas
    except Exception as exc:
        print(f"Standard load failed ({exc}); falling back to streaming to skip malformed rows.")

    return personas


def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def main(args):
    # Load the appropriate template
    if args.template == "agenda":
        template = agenda_template
    else:
        raise ValueError("Invalid template type.")

    personas = load_personas(args)
    print(f"Total number of input personas: {len(personas)}")

    # Load the model and tokenizer
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=args.tensor_parallel_size) # set based on available GPUs

    prompts = []
    max_len = 8192    # 2048

    for persona in personas:
        user_prompt = template.format(persona=persona)
        prompt = request_input_format(user_prompt, tokenizer)
        prompts.append(prompt)

    print(f"Loaded {len(prompts)} entries to process...\n\n")
    print(f"Sample 0: {prompts[0]}")

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_len, stop=["<|eot_id|>"])
    outputs = llm.generate(prompts, sampling_params)

    with open(args.output_path, 'w') as out:
        for i, output in enumerate(outputs):
            out_txt = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            data = {'prompt': output.prompt, "input persona": personas[i], "finish_reason": finish_reason}
            data['synthesized text'] = out_txt
            out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Outputted the results to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize text using a specified model and template.")
    parser.add_argument('--sample_size', type=int, default=0, help='Number of samples to process from the dataset; Set it to 0 if you want to use the full set of 200k personas.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPU tensor parallel partitions.')
    parser.add_argument('--template', type=str, required=True, choices=['ifeval', 'agenda', 'instruction', 'knowledge', 'npc', 'math'], help=("Prompt templates. You can add customized templates in pipeline/generate_tasks/prompt_templates.py"))
    args = parser.parse_args()
    main(args)
