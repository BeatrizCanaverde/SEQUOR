import asyncio
import json
import time
import re

from pathlib import Path
from openai import AsyncOpenAI, Timeout
from openai.types.chat import ChatCompletionMessageParam

from typing import TypedDict, cast, Optional


class Message(TypedDict, total=False):
    role: str
    content: str
    thinking: str  # Optional field for reasoning models


Conversation = list[Message]


# Model-specific parsing strategies
MODEL_PARSING_STRATEGIES = {
    # OpenAI GPT-OSS models use channel-based format
    "openai__gpt-oss-20b": "channel",
    "openai__gpt-oss-120b": "channel",
    
    # Qwen Thinking models use <think></think> tags
    "Qwen3-30B-A3B-Thinking": "qwen_thinking",
    "Qwen3-235B-A22B-Thinking": "qwen_thinking",
    "Qwen3-Next-80B-A3B-Thinking": "qwen_thinking",
    "Qwen3.5": "qwen_thinking",  # Catch Qwen 3.5 models
}


def _get_parsing_strategy(model_path: str) -> str:
    """
    Determine the parsing strategy based on model path.
    
    Returns: "channel", "qwen_thinking", or "default"
    """
    model_path_lower = model_path.lower()
    
    for model_pattern, strategy in MODEL_PARSING_STRATEGIES.items():
        if model_pattern.lower() in model_path_lower:
            return strategy
    
    return "default"


def _parse_channel_response(response_text: str) -> tuple[Optional[str], str]:
    """
    Parse a response with channel tags and return (thinking, content).
    
    For OpenAI GPT-OSS models that use channel tags like:
    <|channel|>analysis<|message|>...thinking...<|end|><|start|>assistant<|channel|>final<|message|>...content...
    
    Returns (thinking_text, final_content)
    """
    if not response_text:
        return None, ""
    
    # Check if response contains channel tags
    if "<|channel|>" not in response_text:
        return None, response_text
    
    thinking = None
    content = response_text
    
    # Extract analysis/thinking channel
    analysis_match = re.search(
        r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
        response_text,
        re.DOTALL
    )
    if analysis_match:
        thinking = analysis_match.group(1).strip()
    
    # Extract final channel content
    final_match = re.search(
        r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)',
        response_text,
        re.DOTALL
    )
    if final_match:
        content = final_match.group(1).strip()
    else:
        # If no final channel found, use the whole response as content
        # but strip the analysis part if present
        if thinking:
            content = re.sub(
                r'<\|channel\|>analysis<\|message\|>.*?<\|end\|>(<\|start\|>assistant)?',
                '',
                response_text,
                flags=re.DOTALL
            ).strip()
    
    return thinking, content


def _parse_qwen_thinking_response(response_text: str) -> tuple[Optional[str], str]:
    """
    Parse a response with <think></think> tags and return (thinking, content).
    
    For Qwen Thinking models that output:
    <think>...thinking content...</think>\n\nActual response
    OR
    ...thinking content...\n</think>\n\nActual response  (without opening tag)
    
    Returns (thinking_text, final_content)
    """
    if not response_text:
        return None, ""
    
    # Check if response contains </think> tag
    if "</think>" not in response_text:
        return None, response_text
    
    # Try to extract thinking content between <think> and </think>
    think_match = re.search(
        r'<think>(.*?)</think>',
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    
    if think_match:
        thinking = think_match.group(1).strip()
        # Remove the entire <think>...</think> block from content
        content = re.sub(
            r'<think>.*?</think>\s*',
            '',
            response_text,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()
        return thinking, content
    
    # If no opening tag, assume everything before </think> is thinking
    # (Some Qwen models output thinking without the opening tag)
    think_end = response_text.find('</think>')
    if think_end != -1:
        thinking = response_text[:think_end].strip()
        # Content is everything after </think>
        content = response_text[think_end + len('</think>'):].strip()
        return thinking, content
    
    return None, response_text


def _parse_response(response_text: str, model_path: str = "") -> tuple[Optional[str], str]:
    """
    Parse a response based on the model's format.
    
    Returns (thinking_text, final_content)
    """
    strategy = _get_parsing_strategy(model_path)
    
    if strategy == "channel":
        return _parse_channel_response(response_text)
    elif strategy == "qwen_thinking":
        return _parse_qwen_thinking_response(response_text)
    else:
        # Default: no special parsing, return full text as content
        return None, response_text


def _prepare_messages_for_api(conversation: Conversation) -> list[dict]:
    """
    Prepare conversation messages for the API.
    
    IMPORTANT: Thinking traces from previous turns are NEVER sent back to the model.
    They are only stored in the saved conversation file for analysis.
    """
    prepared = []
    for msg in conversation:
        # Only send role and content - never include thinking field
        prepared.append({"role": msg["role"], "content": msg.get("content", "")})
    return prepared


class RunConversationRequest(TypedDict):
    user_msgs: list[Message]
    save_path: Path


async def run_conversations(
    *,
    requests: list[RunConversationRequest],
    api_url: str,
    api_model: str,
    api_key: str,
    max_tokens: int,
    max_concurrent: int = 16,
    max_retries: int = 20,
    initial_retry_delay: float = 1.0,
    request_timeout: float = 600.0,
) -> None:
    
    client = AsyncOpenAI(
        base_url=api_url,
        api_key=api_key,
        timeout=Timeout(request_timeout, connect=10.0),
    )
    semaphore = asyncio.Semaphore(max_concurrent)
    start = time.time()

    try:
        # Launch 1 asyncio Task per conversation
        _ = await asyncio.gather(
            *(
                run_conversation(
                    conv_id=i,
                    client=client,
                    user_msgs=req["user_msgs"],
                    model=api_model,
                    max_tokens=max_tokens,
                    save_path=req["save_path"],
                    log_every=10,
                    save_every=10,
                    semaphore=semaphore,
                    max_retries=max_retries,
                    initial_retry_delay=initial_retry_delay,
                )
                for i, req in enumerate(requests, start=1)
            )
        )
        elapsed = time.time() - start
        print(f"⌛ total wall-clock time: {_fmt_time(elapsed)}")
    finally:
        await client.close()


async def run_conversation(
    conv_id: int,
    client: AsyncOpenAI,
    user_msgs: list[Message],
    model: str,
    *,
    max_tokens: int,
    log_every: int,
    save_path: Path,
    save_every: int,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    initial_retry_delay: float,
) -> Conversation:
    """
    Drive a single conversation until an exception aborts it.
    Returns the (possibly partial) transcript.
    """
    conversation: Conversation = []
    start_turn = 1

    if save_path.exists():
        with open(save_path, "r", encoding="utf-8") as fh:
            conversation = json.load(fh)
        
        # Validate and fix any invalid message sequences
        fixed_conversation, num_removed = _validate_and_fix_conversation(conversation)
        if num_removed > 0:
            print(f"[conv {conv_id}] ⚠️  Detected invalid message sequence, removed {num_removed} messages", flush=True)
            conversation = fixed_conversation
            # Save the fixed conversation immediately
            _save_conversation(conversation, save_path)
        
        # If conversation ends with a user message (incomplete turn), remove it so we can retry
        if conversation and conversation[-1]["role"] == "user":
            print(f"[conv {conv_id}] ⚠️  Removing incomplete turn (user message without assistant response)", flush=True)
            conversation.pop()
            _save_conversation(conversation, save_path)
        
        # Count completed turns (user-assistant pairs), not just user messages
        completed_turns = sum(1 for msg in conversation if msg["role"] == "assistant")
        user_msgs = user_msgs[completed_turns:]
        start_turn = completed_turns + 1
        print(f"[conv {conv_id}] resuming from {completed_turns} completed turns", flush=True)
    else:
        print(f"[conv {conv_id}] starting new conversation: {save_path.name}", flush=True)

    start = time.time()
    for idx, user_turn in enumerate(user_msgs, start=start_turn):
        # Add delay between turns if using external API to avoid rate limits
        # (e.g., OpenRouter free tier has a strict limits)
        if idx > 1:
            is_external_api = "localhost" not in client.base_url.host and "127.0.0.1" not in client.base_url.host
            if is_external_api:
                # print(f"[conv {conv_id}] Waiting 60s before turn {idx} to avoid rate limits...", flush=True)
                await asyncio.sleep(20)

        # Append the user message
        conversation.append(user_turn)

        # Prepare messages with proper thinking/content separation
        prepared_messages = _prepare_messages_for_api(conversation)
        
        generation, finished = await _run_completion(
            client=client,
            messages=cast(list[ChatCompletionMessageParam], prepared_messages),
            model=model,
            max_tokens=max_tokens,
            semaphore=semaphore,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
        )

        # If there was a fatal error but we got no content, use a fallback response
        # This ensures we always complete exactly max_turns
        if finished and not generation:
            generation = "[Error: Unable to generate response]"
            print(f"[conv {conv_id}] ⚠️  Using fallback response for turn {idx} due to error", flush=True)

        # Parse the response to extract thinking and final content
        thinking, content = _parse_response(generation, model)
        
        # Create assistant message with proper fields
        assistant_msg: Message = {"role": "assistant", "content": content}
        if thinking:
            assistant_msg["thinking"] = thinking
        
        conversation.append(assistant_msg)

        if idx % log_every == 0:
            elapsed = time.time() - start
            print(f"[conv {conv_id}] {idx} - {_fmt_time(elapsed)}", flush=True)

        if idx % save_every == 0:
            _save_conversation(conversation, save_path)
            print(f"[conv {conv_id}] saved after {idx} messages", flush=True)

    # Final save
    _save_conversation(conversation, save_path)
    num_responses = len([msg for msg in conversation if msg["role"] == "assistant"])
    print(f"[conv {conv_id}] finished after {num_responses} responses - {_fmt_time(time.time() - start)}", flush=True)
    return conversation


Finished = bool


def _is_context_length_error(error: Exception) -> bool:
    """
    Check if the error is a context length exceeded error.
    """
    error_str = str(error).lower()
    return any(phrase in error_str for phrase in [
        "context length",
        "maximum context length",
        "token count exceeds",
        "requested token count",
    ])


async def _run_completion(
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    initial_retry_delay: float,
) -> tuple[str, Finished]:
    """
    Try to complete a chat request with exponential backoff retry logic.
    Implements a sliding window for context length errors by progressively
    removing the earliest conversation turns (2 messages at a time: user + assistant).
    
    Returns (generated_text, finished_flag).
    If finished_flag is True, the conversation should stop.
    """
    # Start with full message history
    current_messages = messages
    turns_removed = 0
    
    # Sliding window loop - continues until success or cannot remove more messages
    while True:
        # Attempt the API call with retries for transient errors
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=current_messages,
                    )
                return response.choices[0].message.content or "", False
            
            except asyncio.TimeoutError as e:
                delay = initial_retry_delay * (2 ** attempt)
                if attempt < max_retries - 1:
                    print(f"Timeout on attempt {attempt + 1}/{max_retries}, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Error occurred after {max_retries} attempts: Request timed out.")
                    return "", True
            
            except Exception as e:
                # Check if this is a context length error
                if _is_context_length_error(e):
                    # Break out of retry loop to handle sliding window
                    break
                
                # Check for Rate Limit (HTTP 429) to use custom delays
                is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
                
                # For other errors, use exponential backoff
                if is_rate_limit:
                    # Specific delay for API rate limits: 30s, 60s, 120s...
                    delay = 30 * (2 ** attempt)
                else:
                    delay = initial_retry_delay * (2 ** attempt)
                
                if attempt < max_retries - 1:
                    print(f"{'Rate limit' if is_rate_limit else 'Error'} on attempt {attempt + 1}/{max_retries}: {e}, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Error occurred after {max_retries} attempts: {e}")
                    return "", True
        else:
            # Completed all retries without hitting context length error
            # Should not reach here normally
            return "", True
        
        # Handle context length error with sliding window
        if len(current_messages) > 2:
            current_messages = current_messages[2:]
            turns_removed += 1
            print(f"Context length exceeded. Removing earliest turn (removed {turns_removed} turns total, {len(current_messages)} messages remaining). Retrying...")
            # Continue to next sliding window iteration
        else:
            print(f"Context length exceeded but only {len(current_messages)} messages remain. Cannot reduce further.")
            return "", True


def _validate_and_fix_conversation(conversation: Conversation) -> tuple[Conversation, int]:
    """
    Validate that conversation roles alternate properly (user/assistant/user/assistant...).
    If invalid sequences are found (two or more consecutive messages with the same role),
    truncate to the last valid alternating message.
    
    Returns:
        (fixed_conversation, num_removed): The fixed conversation and number of messages removed
    """
    if not conversation:
        return conversation, 0
    
    # Find the last position where the conversation is valid
    valid_up_to = 0
    prev_role = None
    
    for i, msg in enumerate(conversation):
        current_role = msg.get("role")
        
        # Check if this role is the same as previous (invalid)
        if prev_role == current_role:
            # Found invalid sequence - truncate here
            break
        
        # This message is valid
        valid_up_to = i + 1
        prev_role = current_role
    
    # If we need to truncate
    num_removed = len(conversation) - valid_up_to
    if num_removed > 0:
        return conversation[:valid_up_to], num_removed
    
    return conversation, 0


def _save_conversation(conversation: Conversation, save_path: Path) -> None:
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(conversation, fh, ensure_ascii=False, indent=2)


def load_conversation(save_path: Path) -> Conversation:
    with open(save_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt_time(time_in_seconds: float) -> str:
    """Convert time in seconds to a formatted string."""
    minutes = int(time_in_seconds // 60)
    seconds = int(time_in_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"






#### For single-turn runs ####

async def run_file(
    file: Path,
    prompts: list[dict],
    save_path: Path,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    api_model: str,
    max_tokens: int,
    run_completion_fn,
    save_conversation_fn,
    max_retries: int = 5,
    initial_retry_delay: float = 1.0,
):
    # Identifiers for which prompts need generation
    # indices_to_generate will store the 0-based index of the prompts list
    indices_to_generate = []
    conversation: list[Message] = []
    
    # If file exists, we check for empty responses to replace
    if save_path.exists():
        try:
            with open(save_path, "r", encoding="utf-8") as fh:
                conversation = json.load(fh)
            
            # Identify prompts that have empty/error assistant responses
            # Baseline structure: [User, Assistant, User, Assistant, ...]
            for i in range(0, len(conversation) - 1, 2):
                user_msg = conversation[i]
                assistant_msg = conversation[i+1]
                
                content = assistant_msg.get("content", "")
                is_empty = (not isinstance(content, str)) or (not content.strip()) or ("[Error:" in content)
                
                if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant" and is_empty:
                    turn_idx = i // 2
                    print(f"[baseline] ⚠️ Found empty response in {save_path.name} at turn {turn_idx + 1}. Marking for replacement.", flush=True)
                    indices_to_generate.append(turn_idx)
            
            # If the conversation is incomplete (ends with user msg), mark the last prompt for generation
            if conversation and conversation[-1].get("role") == "user":
                turn_idx = len(conversation) // 2
                if turn_idx not in indices_to_generate:
                    indices_to_generate.append(turn_idx)
                print(f"[baseline] ⚠️ Found trailing user message in {save_path.name}. Marking for generation.", flush=True)
                
            # Also need to run any prompts that aren't in the file yet
            completed_turns = len([m for m in conversation if m.get("role") == "assistant"])
            for turn_idx in range(completed_turns, len(prompts)):
                if turn_idx not in indices_to_generate:
                    indices_to_generate.append(turn_idx)

        except (json.JSONDecodeError, ValueError):
            print(f"[baseline] ⚠️ Error reading {save_path.name}, starting fresh.", flush=True)
            conversation = []
            indices_to_generate = list(range(len(prompts)))
    else:
        conversation = []
        indices_to_generate = list(range(len(prompts)))

    if not indices_to_generate:
        print(f"[baseline] {file.name} already complete.", flush=True)
        return

    print(f"[baseline] Processing {file.name} ({len(indices_to_generate)} responses to generate/replace)", flush=True)

    for turn_idx in indices_to_generate:
        prompt_data = prompts[turn_idx]
        user_msg: Message = {"role": "user", "content": prompt_data["prompt"]}
        
        # Determine where to put this turn in the conversation list
        # If it's a replacement, the turn starts at index turn_idx * 2
        insertion_point = turn_idx * 2
        
        # Ensure the list is long enough (for new prompts)
        while len(conversation) < insertion_point:
            dummy_user = {"role": "user", "content": "..."}
            dummy_assistant = {"role": "assistant", "content": "[Missing]"}
            conversation.extend([dummy_user, dummy_assistant])
            
        # Update or set the user message
        if insertion_point < len(conversation):
            conversation[insertion_point] = user_msg
        else:
            conversation.append(user_msg)

        print(f"[baseline] {file.name}: generating turn {turn_idx + 1}/{len(prompts)}", flush=True)

        generation, finished = await run_completion_fn(
            client=client,
            messages=[user_msg],
            model=api_model,
            max_tokens=max_tokens,
            semaphore=semaphore,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
        )

        gen = generation if not finished else ""

        # Parse the response to extract thinking and final content
        thinking, content = _parse_response(gen, api_model)
        
        # Create assistant message
        assistant_msg: Message = {"role": "assistant", "content": content}
        if thinking:
            assistant_msg["thinking"] = thinking
        
        # Insert assistant message at insertion_point + 1
        if (insertion_point + 1) < len(conversation):
            conversation[insertion_point + 1] = assistant_msg
        else:
            conversation.append(assistant_msg)

        print(f"[baseline] {file.name}: updated turn {turn_idx + 1} (len={len(content)})", flush=True)

        # save after each turn to allow resume
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_conversation_fn(conversation, save_path)
        

async def _run_all(tasks):
    await asyncio.gather(*tasks)