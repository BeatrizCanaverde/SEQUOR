#!/usr/bin/env python3
"""
Script to run LLM evaluation on task-constraint pairs using an API (e.g., OpenRouter).

This script:
1. Loads all JSONL files from the input directory
2. Extracts prompts from each file
3. Sends prompts to the specified API model
4. Saves responses to corresponding output files with the same names
"""

import argparse
import json
import os
from pathlib import Path
import random
import time
from openai import OpenAI
from tqdm import tqdm


def load_jsonl(file_path):
    """Load all lines from a JSONL file."""
    data = []
    if not os.path.exists(file_path):
        return []
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
            if item is None:
                continue
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_api_response(
    client,
    model,
    user_prompt,
    system_prompt="You are a helpful assistant.",
    max_tokens=4096,
    reasoning_enabled=False,
    extra_headers=None,
):
    """Get response from the API.

    Notes for OpenRouter:
    - To enable Gemini/other reasoning-capable models, pass `reasoning={"enabled": True}`.
      With the OpenAI Python client, this is supplied via `extra_body`.
    - When present, `reasoning_details` is returned on the assistant message.
    """
    try:
        extra_body = None
        if reasoning_enabled:
            extra_body = {"reasoning": {"enabled": True}}

        completion = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        msg = completion.choices[0].message

        # Be defensive: openai-python may expose additional fields as attrs or only in dict form.
        content = getattr(msg, "content", None)
        reasoning_details = getattr(msg, "reasoning_details", None)

        usage = None
        try:
            usage = completion.usage
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
        except Exception:
            usage = None

        # Fallback to dict extraction if needed
        if content is None or (reasoning_enabled and reasoning_details is None):
            try:
                completion_dict = completion.model_dump()
                msg_dict = completion_dict["choices"][0]["message"]
                content = content if content is not None else msg_dict.get("content")
                if reasoning_enabled and reasoning_details is None:
                    reasoning_details = msg_dict.get("reasoning_details")
                if usage is None:
                    usage = completion_dict.get("usage")
            except Exception:
                pass

        return {
            "content": content,
            "reasoning_details": reasoning_details,
            "usage": usage,
        }
    except Exception as e:
        print(f"Error calling API: {e}")
        return None


def process_file(file_path, client, args, output_dir):
    """
    Process a single input file and generate responses.
    """
    print(f"\nProcessing file: {file_path.name}")
    
    # Load data
    data = load_jsonl(file_path)
    print(f"  Loaded {len(data)} entries from input file")
    
    # Check for existing output file
    output_file = output_dir / file_path.name
    existing_data = {}
    
    if output_file.exists():
        print(f"  Found existing output file: {output_file.name}")
        existing_output = load_jsonl(output_file)
        
        # Create a mapping of prompts to responses for existing data
        for item in existing_output:
            if item is not None and 'prompt' in item and 'response' in item:
                existing_data[item['prompt']] = item
        
        print(f"  Loaded {len(existing_data)} existing responses")
    
    # Determine which entries need processing
    entries_to_process = []
    indices_to_process = []
    output_data = []
    
    for i, item in enumerate(data):
        # Stop if we've reached the max_entries limit
        if args.max_entries is not None and len(output_data) >= args.max_entries:
            break
        
        prompt = item['prompt']
        
        # Check if response already exists
        if prompt in existing_data:
            # Use existing response
            output_data.append(existing_data[prompt])
        else:
            # Need to generate response
            entries_to_process.append(item)
            indices_to_process.append(len(output_data))
            # Placeholder that will be filled
            output_data.append(None)
    
    # Report status
    already_done = len(output_data) - len(entries_to_process)
    print(f"  Status: {already_done} already completed, {len(entries_to_process)} need processing")
    
    if args.max_entries is not None:
        print(f"  Limiting to {args.max_entries} total entries (requested max)")
    
    # Generate responses for missing entries
    if entries_to_process:
        print(f"  Generating {len(entries_to_process)} new responses...")
        
        for i, item in enumerate(tqdm(entries_to_process, desc=f"Generating responses for {file_path.name}")):
            user_prompt = item['prompt']
            
            # Retry logic
            max_retries = 3
            response_obj = None
            for attempt in range(max_retries):
                response_obj = get_api_response(
                    client, 
                    args.model, 
                    user_prompt, 
                    max_tokens=args.max_tokens,
                    reasoning_enabled=args.reasoning_enabled,
                    extra_headers=args.extra_headers,
                )
                if response_obj and response_obj.get("content"):
                    break
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            if response_obj and response_obj.get("content"):
                output_item = item.copy()
                output_item['response'] = response_obj.get("content")
                if args.reasoning_enabled:
                    # Persist as-is so it can be passed back unmodified if you add follow-ups later.
                    output_item['reasoning_details'] = response_obj.get("reasoning_details")
                if response_obj.get("usage") is not None:
                    output_item['usage'] = response_obj.get("usage")
                output_item['model'] = args.model
                output_item['messages'] = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt},
                ]
                
                # Place at correct index
                output_data[indices_to_process[i]] = output_item
                
                # Save periodically or after each successful call to avoid data loss
                save_jsonl(output_data, output_file)
            else:
                print(f"\nFailed to get response for prompt at index {indices_to_process[i]} after {max_retries} retries")
    else:
        print(f"  No new responses needed - all entries already processed!")
    
    # Final save
    save_jsonl(output_data, output_file)
    
    print(f"  Saved {len(output_data)} total responses to: {output_file}")
    
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
    
    # Find all JSONL files in input directory
    if args.input_file:
        input_files = [input_dir / args.input_file]
    else:
        input_files = sorted(input_dir.glob("*.jsonl"))
    
    if not input_files:
        raise ValueError(f"No JSONL files found in {input_dir}")
    
    print("=" * 80)
    print("LLM Evaluation - Task-Constraint Pairs (API Mode)")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model:            {args.model}")
    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f.name}")
    print()
    
    # Initialize API client
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )

    # Prepare optional OpenRouter leaderboard headers
    extra_headers = {}
    if args.openrouter_referer:
        extra_headers["HTTP-Referer"] = args.openrouter_referer
    if args.openrouter_title:
        extra_headers["X-Title"] = args.openrouter_title
    args.extra_headers = extra_headers or None
    
    # Process each file
    output_files = []
    for input_file in input_files:
        output_file = process_file(
            input_file,
            client,
            args,
            output_dir
        )
        output_files.append(output_file)
    
    print()
    print("=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"Processed {len(input_files)} files")
    print(f"Output files:")
    for f in output_files:
        print(f"  - {f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluation on task-constraint pairs using an API")
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input JSONL files')
    parser.add_argument('--input-file', type=str, default=None, help='Specific input file to process (optional)')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output results')
    parser.add_argument('--model', type=str, required=True, help='API Model name (e.g., openai/gpt-4o, anthropic/claude-3-opus-20240229)')
    parser.add_argument('--base-url', type=str, default="https://openrouter.ai/api/v1", help='API base URL (default: OpenRouter)')
    parser.add_argument('--api-key', type=str, default=None, help='API key (will also check OPENROUTER_API_KEY or OPENAI_API_KEY env vars)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Max tokens to generate (default: 4096)')
    parser.add_argument('--reasoning-enabled', action='store_true', help='Enable OpenRouter reasoning (saves reasoning_details when returned)')
    parser.add_argument('--openrouter-referer', type=str, default=None, help='Optional OpenRouter header HTTP-Referer (for leaderboards)')
    parser.add_argument('--openrouter-title', type=str, default=None, help='Optional OpenRouter header X-Title (for leaderboards)')
    parser.add_argument('--max-entries', type=int, default=None, help='Maximum number of entries to process per file (default: None for all entries)')
    args = parser.parse_args()
    main(args)
