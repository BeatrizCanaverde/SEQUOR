import argparse
import json
from pathlib import Path
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Import Harmony for GPT OSS models
try:
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    print("Warning: openai_harmony not available. GPT OSS models will not work.")


def load_jsonl(file_path):
    """Load all lines from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_tuples(file_path):
    """Load tuples (constraint sets) from a JSONL file."""
    data = load_jsonl(file_path)
    tuples = []
    for item in data:
        # Each item has a 'constraints' field which is a list of constraint strings
        constraints = item['constraints']
        tuples.append(constraints)
    return tuples


def load_tasks(file_path):
    """Load tasks from a JSONL file."""
    data = load_jsonl(file_path)
    tasks = []
    for item in data:
        # Assuming tasks have a 'prompt' field
        task_prompt = item.get('prompt', item.get('text', str(item)))
        tasks.append(task_prompt)
    return tasks


def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text


# Template for combining multiple constraints and task
PROMPT_TEMPLATE = """Address the following task while adhering to all the given constraints.

Constraints:
{constraints}

Task:
{task}
"""


def format_constraints(constraints):
    """Format a list of constraints as a numbered list."""
    return "\n".join(f"{i+1}. {c}" for i, c in enumerate(constraints))


def is_gpt_oss_model(model_path):
    """Check if the model path indicates a GPT OSS model."""
    model_path_lower = model_path.lower()
    return 'gpt' in model_path_lower and 'oss' in model_path_lower


def create_harmony_prompt_ids(user_prompt, encoding):
    """Create prompt token IDs using Harmony encoding for GPT OSS models."""
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions("You are a helpful assistant."),
            ),
            Message.from_role_and_content(Role.USER, user_prompt),
        ]
    )
    
    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    return prefill_ids


def parse_harmony_output(output_tokens, encoding):
    """Parse Harmony output tokens back into structured messages and extract text."""
    entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
    
    # Extract text from all messages
    texts = []
    for message in entries:
        msg_dict = message.to_dict()
        # Extract text content from various message types
        if 'content' in msg_dict:
            content = msg_dict['content']
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                    elif isinstance(item, str):
                        texts.append(item)
    
    return '\n'.join(texts) if texts else ''


def load_existing_outputs(output_path):
    """Load existing outputs and return a set of processed (constraints, task) tuples."""
    if not output_path.exists():
        return set(), 0
    
    processed = set()
    count = 0
    print(f"Loading existing outputs from {output_path}...")
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    constraints = tuple(sorted(data['constraints']))  # Use sorted tuple as hashable key
                    task = data['task']
                    processed.add((constraints, task))
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping malformed line: {e}")
                    continue
    
    print(f"Found {count} existing outputs")
    return processed, count


def main(args):

    random.seed(args.seed)

    # Load tuples and tasks
    tuples = load_tuples(args.tuples_file)
    tasks = load_tasks(args.tasks_file)
    
    print(f"Loaded {len(tuples)} tuples")
    print(f"Loaded {len(tasks)} tasks")

    # Load the model
    max_len = 32768
    max_tokens = 8192
    model_path = args.model_path
    
    # Check if this is a GPT OSS model
    use_harmony = is_gpt_oss_model(model_path)
    
    if use_harmony:
        if not HARMONY_AVAILABLE:
            raise ImportError("openai_harmony is required for GPT OSS models. Install it with: pip install openai-harmony")
        print(f"Detected GPT OSS model - using Harmony encoding")
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        tokenizer = None  # Not needed for Harmony
    else:
        print(f"Using standard tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        encoding = None
        stop_token_ids = None
    
    # Load LLM
    llm = LLM(
        model=model_path, 
        seed=args.seed, 
        tensor_parallel_size=args.tensor_parallel_size, 
        pipeline_parallel_size=args.pipeline_parallel_size, 
        max_model_len=max_len,
        trust_remote_code=True,
    )
    
    # Determine output path
    output_path = Path(args.output_dir) / f"{model_path.replace('/', '__')}.jsonl"
    
    # Load existing outputs if any
    processed_set, existing_count = load_existing_outputs(output_path)
    
    # Prepare all prompt data, skipping already processed entries
    all_prompts = []
    all_tuples_tasks = []
    skipped = 0
    
    for constraints in tuples:
        for task_prompt in tasks:
            # Check if this combination was already processed
            constraints_key = tuple(sorted(constraints))
            if (constraints_key, task_prompt) in processed_set:
                skipped += 1
                continue
            
            # Format the user prompt
            formatted_constraints = format_constraints(constraints)
            user_prompt = PROMPT_TEMPLATE.format(constraints=formatted_constraints, task=task_prompt)
            
            # Create prompt based on model type
            if use_harmony:
                # For GPT OSS, we'll create token IDs later in batch
                all_prompts.append(user_prompt)  # Store raw user prompt
            else:
                # For standard models, use chat template
                prompt = request_input_format(user_prompt, tokenizer)
                all_prompts.append(prompt)
            
            all_tuples_tasks.append((constraints, task_prompt))

    total_prompts = len(all_tuples_tasks)
    print(f"\nTotal entries: {len(tuples) * len(tasks)}")
    print(f"Already processed: {existing_count}")
    print(f"Skipped (duplicates): {skipped}")
    print(f"Remaining to process: {total_prompts}")
    
    if total_prompts == 0:
        print("\nAll entries have been processed! Nothing to do.")
        return
    
    print(f"\nSample prompt (first 500 chars): {str(all_prompts[0])[:500]}...\n")

    # Process in batches to avoid memory overflow
    batch_size = args.batch_size
    checkpoint_interval = args.checkpoint_interval
    print(f"Processing in batches of {batch_size}...")
    print(f"Checkpointing every {checkpoint_interval} outputs...")
    
    # Create sampling params based on model type
    if use_harmony and stop_token_ids:
        sampling_params = SamplingParams(max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    else:
        sampling_params = SamplingParams(max_tokens=max_tokens)
    
    # Open file in append mode to preserve existing outputs
    mode = 'a' if existing_count > 0 else 'w'
    processed_in_session = 0
    
    with open(output_path, mode, encoding='utf-8') as out:
        for batch_start in range(0, total_prompts, batch_size):
            batch_end = min(batch_start + batch_size, total_prompts)
            batch_prompts_raw = all_prompts[batch_start:batch_end]
            batch_tuples_tasks = all_tuples_tasks[batch_start:batch_end]
            
            batch_num = batch_start // batch_size + 1
            total_batches = (total_prompts + batch_size - 1) // batch_size
            print(f"Processing batch {batch_num}/{total_batches} (items {batch_start}-{batch_end-1})...")
            
            # Prepare batch for vLLM based on model type
            if use_harmony:
                # Create token IDs for each prompt in the batch
                batch_prompts = []
                for user_prompt in batch_prompts_raw:
                    prompt_ids = create_harmony_prompt_ids(user_prompt, encoding)
                    batch_prompts.append({"prompt_token_ids": prompt_ids})
            else:
                # Standard text prompts
                batch_prompts = batch_prompts_raw
            
            # Generate outputs
            outputs = llm.generate(batch_prompts, sampling_params)
            
            # Process outputs
            for i, output in enumerate(outputs):
                if use_harmony:
                    # Parse Harmony output
                    output_tokens = output.outputs[0].token_ids
                    out_txt = parse_harmony_output(output_tokens, encoding)
                    # For recording the original prompt, reconstruct it
                    original_prompt = batch_prompts_raw[i]
                else:
                    # Standard text output
                    out_txt = output.outputs[0].text
                    original_prompt = output.prompt
                
                data = {
                    'prompt': original_prompt,
                    'answer': out_txt,
                    'constraints': batch_tuples_tasks[i][0],
                    'task': batch_tuples_tasks[i][1]
                }
                out.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_in_session += 1
                
                # Checkpoint: flush to disk periodically
                if processed_in_session % checkpoint_interval == 0:
                    out.flush()
                    print(f"  ✓ Checkpoint: {processed_in_session}/{total_prompts} outputs saved ({existing_count + processed_in_session} total)")
            
            # Also flush after each batch
            out.flush()

    print(f"\n✓ Processing complete!")
    print(f"  Processed in this session: {processed_in_session}")
    print(f"  Total outputs in file: {existing_count + processed_in_session}")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with tuples (multiple constraints) and tasks")
    parser.add_argument('--tuples-file', type=str, required=True, help='Path to tuples JSONL file (constraint sets).')
    parser.add_argument('--tasks-file', type=str, required=True, help='Path to tasks JSONL file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output results.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size for model.')
    parser.add_argument('--pipeline-parallel-size', type=int, default=1, help='Pipeline parallel size for model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing prompts to avoid memory overflow.')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='Save outputs to disk every N items for crash recovery.')
    args = parser.parse_args()
    main(args)
