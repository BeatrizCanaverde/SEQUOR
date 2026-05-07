import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence, Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MAX_MODEL_LEN = 50000  # 100000
DEFAULT_MAX_TOKENS = MAX_MODEL_LEN


def load_constraints(file_path: str) -> List[str]:
    """Load constraints from a JSONL file. Accepts lines that are strings or dicts with a 'constraint' field."""
    constraints: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, str):
                constraints.append(obj)
            elif isinstance(obj, dict) and "constraint" in obj:
                constraints.append(obj["constraint"])
            else:
                raise ValueError(f"Unsupported line format at {line_num}: {obj}")
    return constraints


def load_tasks_from_file(tasks_file: str) -> List[str]:
    """Load tasks from a JSONL file. Accepts raw strings or dicts with 'prompt'."""
    tasks: List[str] = []
    file_path = Path(tasks_file)
    
    if not file_path.exists():
        raise ValueError(f"Task file not found: {tasks_file}")
    
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            obj: Any = json.loads(raw)
            if isinstance(obj, str):
                tasks.append(obj)
            elif isinstance(obj, dict) and "prompt" in obj:
                tasks.append(obj["prompt"])
            else:
                raise ValueError(f"Unsupported task format in {file_path} line {line_num}: {obj}")
    return tasks


def request_input_format(user_prompt: str, tokenizer) -> str:
    system_prompt = "You are a helpfull assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


JUDGE_TEMPLATE = """You will receive a task and a list of constraints. Analyze whether the constraints can all be followed simultaneously by a single response to the task without contradiction. You can first reason about the task, the constraints, and their possible contradictions. At the end, reply with "Final Verdict: [[Yes]]" if they are jointly compatible, otherwise reply with "Final Verdict: [[No]]".

Task:
"{task}"

Constraints:
"{constraints}"
"""


def build_judge_prompt(task: str, constraints: Sequence[str]) -> str:
    formatted = "\n".join(f"- {c}" for c in constraints)
    return JUDGE_TEMPLATE.format(task=task, constraints=formatted)


def normalize_tuple(constraints: Sequence[str]) -> tuple:
    """Normalize a tuple of constraints for deduplication (order-insensitive)."""
    return tuple(sorted(constraints))


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    constraints = load_constraints(args.constraints_file)
    tasks = load_tasks_from_file(args.tasks_file)
    if len(constraints) < args.tuple_size:
        raise ValueError("Tuple size cannot exceed the number of available constraints.")
    if not tasks:
        raise ValueError("No tasks loaded from tasks file.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.tuple_size}.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=MAX_MODEL_LEN,
        # gpu_memory_utilization=0.85,
        # dtype="auto",
        max_num_seqs=256
    )

    sampling_params = SamplingParams(
        max_tokens=DEFAULT_MAX_TOKENS,
    )

    sampled_tuples: List[List[str]] = []
    skipped_duplicates = 0
    seen_tuples = set()

    while len(sampled_tuples) < args.num_tuples:
        candidate = random.sample(constraints, args.tuple_size)
        norm = normalize_tuple(candidate)
        if norm in seen_tuples:
            skipped_duplicates += 1
            continue
        seen_tuples.add(norm)
        sampled_tuples.append(candidate)

    all_prompts: List[str] = []
    prompt_meta: List[tuple[int, str]] = []  # (tuple_index, task_text)
    for idx, tuple_item in enumerate(sampled_tuples):
        for task in tasks:
            all_prompts.append(build_judge_prompt(task, tuple_item))
            prompt_meta.append((idx, task))

    print(f"Prepared {len(sampled_tuples)} tuples and {len(all_prompts)} prompts. Generating...", flush=True)

    formatted_prompts = [request_input_format(p, tokenizer) for p in all_prompts]
    outputs = llm.generate(formatted_prompts, sampling_params)

    tuple_outputs: List[List[str]] = [[] for _ in sampled_tuples]
    for (tuple_idx, _task_text), output in zip(prompt_meta, outputs):
        tuple_outputs[tuple_idx].append(output.outputs[0].text)

    with open(output_path, "w", encoding="utf-8") as out_file:
        for tuple_item, model_texts in zip(sampled_tuples, tuple_outputs):
            json.dump(
                {
                    "constraints": tuple_item,
                    "tasks": tasks,
                    "model_outputs": model_texts,
                },
                out_file,
                ensure_ascii=False,
            )
            out_file.write("\n")

    print(f"Finished. Saved tuples: {len(sampled_tuples)}")
    print(f"Skipped duplicate tuples during sampling: {skipped_duplicates}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample constraint tuples and record raw model analyses.")
    parser.add_argument("--constraints-file", required=True, help="Path to the constraints JSONL pool.")
    parser.add_argument("--tasks-file", required=True, help="Path to the tasks JSONL file.")
    parser.add_argument("--tuple-size", type=int, required=True, help="Number of constraints per tuple.")
    parser.add_argument("--num-tuples", type=int, default=10000, help="Number of tuples to generate (default: 10000).")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output JSONL.")
    parser.add_argument("--model-path", required=True, help="Model identifier or path for vLLM.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    main(args)
