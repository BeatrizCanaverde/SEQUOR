from __future__ import annotations

import json
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypedDict

from openai import AsyncOpenAI, Timeout
from multi_if.constraints import ConstraintData
from multi_if.conversation import Conversation, load_conversation, _parse_response
from multi_if.test_prompts import TestPrompts
from multi_if.extract_tasks import process_prompt

sys.path.append(str(Path(__file__).parent))



class ConstraintEvaluation(TypedDict):
    constraint_id: str
    constraint_text: str
    judge_prompt: str
    # Keep the raw judge message for later parsing; no final classification extracted here.
    judge_response_raw: str
    # Separated thinking and content from judge response (if model produces thinking)
    judge_thinking: str | None
    judge_content: str
    # Classification is intentionally deferred; left as None until a dedicated parser is added.
    followed: bool | None
    reason: str | None


class PromptEvaluation(TypedDict):
    constraint_evaluations: list[ConstraintEvaluation]
    prompt: str
    response: str


EvaluationReport = list[PromptEvaluation]


# Prompts for judging
SYSTEM_PROMPT_CONSTRAINT = """An assistant has been asked to perform a task. Your job is to assess whether the provided answer satisfies a given constraint. You may first reason about both the constraint and the answer. At the end, present your final verdict as either “Final Verdict: [[Yes]]” if the answer satisfies the constraint, or “Final Verdict: [[No]]” if it does not.

Does the following answer satisfy the constraint?

Answer: 
"{answer}"

Constraint:
"{constraint}"
"""

SYSTEM_PROMPT_TASK = """Evaluate whether an assistant's answer appropriately addresses a given task. The task may be subjective or open-ended, and does not necessarily have a single correct answer. Focus on whether the response:
- Demonstrates a meaningful attempt to address the task.
- Provides relevant, coherent, and useful content.

You may first reason about both the task and the answer. At the end, present your final verdict as either “Final Verdict: [[Yes]]” if the answer addresses the task, or “Final Verdict: [[No]]” if it does not.

Does the following answer address the task?

Task:
"{constraint}"

Answer:
"{answer}"
"""


@dataclass(frozen=True)
class JudgeConfig:
    api_url: str
    api_model: str
    api_key: str
    # temperature: float = 0.0
    max_tokens: int = 8192


async def judge_async(*,
    judge: JudgeConfig,
    client: AsyncOpenAI,
    constraint: ConstraintData,
    user_prompt: str,
    assistant_response: str,
    semaphore: asyncio.Semaphore,
    evaluate_task_correctness: bool = False,
) -> ConstraintEvaluation:
    if evaluate_task_correctness:
        system_content = SYSTEM_PROMPT_TASK.format(
            answer=assistant_response,
            constraint=constraint["text"],
        )
    else:
        system_content = SYSTEM_PROMPT_CONSTRAINT.format(
            answer=assistant_response,
            constraint=constraint["text"],
        )

    # Use user role instead of system role - many models only respond to user messages
    messages = [{"role": "user", "content": system_content}]

    raw_content = ""
    reason = ""

    max_retries = 5
    base_backoff = 2.0  # Start with 2 second delay

    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                completion = await client.chat.completions.create(
                    model=judge.api_model,
                    messages=messages,
                    max_tokens=judge.max_tokens,
                    temperature=0.0,
                )

            raw_content = completion.choices[0].message.content or ""

            if raw_content:
                try:
                    payload = json.loads(raw_content) if raw_content else {}
                    reason = str(payload.get("reason", "")).strip()
                except json.JSONDecodeError:
                    reason = ""
                break

            # Empty response: wait longer before retry (exponential backoff)
            if attempt < max_retries:
                delay = base_backoff * (2 ** (attempt - 1))  # 2s, 4s, 8s, 16s
                print(f"[judge] Empty response (attempt {attempt}/{max_retries}), retrying in {delay:.1f}s...", flush=True)
                await asyncio.sleep(delay)
                continue
            print(f"[judge] Empty response after {max_retries} attempts for constraint {constraint['id']}", flush=True)
            reason = f"Empty response after {max_retries} attempts"

        except Exception as exc:
            reason = f"Judge error: {exc}"
            if attempt < max_retries:
                delay = base_backoff * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
                continue

    reason = reason.strip() or "No reason provided"
    
    # Parse judge response to separate thinking from final content
    # Use the judge model path to determine parsing strategy
    judge_thinking, judge_content = _parse_response(raw_content, judge.api_model)

    return ConstraintEvaluation(
        constraint_id=constraint["id"],
        constraint_text=constraint["text"],
        judge_prompt=system_content,
        judge_response_raw=raw_content,
        judge_thinking=judge_thinking,
        judge_content=judge_content,
        followed=None,
        reason=reason,
    )

        
async def _evaluate_one(
    relative_path: Path,
    prompts: list[dict],
    *,
    judge: JudgeConfig,
    chat_dir: Path,
    evaluation_judge_dir: Path,
    run_name: str,
    max_lines: int,
    evaluate_task_correctness: bool = False,
):
    # Check if evaluation already exists (resume functionality)
    clean_name = str(relative_path.with_suffix(""))
    flat_name = clean_name.replace("/", "_").replace("\\", "_")
    save_path = evaluation_judge_dir / f"{flat_name}.jsonl"
    
    conversation_file = _resolve_conversation_file(chat_dir, relative_path)
    conversation = load_conversation(conversation_file)
    
    # Check if we need to re-evaluate
    existing_report = None
    if save_path.exists():
        try:
            existing_report = load_evaluation_report(save_path)
            
            # Check if any turn in the existing report matches an empty/error response in the source conversation
            # For baselines, turn_idx i in existing_report corresponds to conversation[2*i+1]
            needs_regeneration = False
            for turn_idx, eval_turn in enumerate(existing_report):
                if turn_idx * 2 + 1 < len(conversation):
                    assistant_msg = conversation[turn_idx * 2 + 1]
                    content = assistant_msg.get("content", "")
                    # If the source response is NOT empty anymore but the judge evaluation was based on an empty one
                    eval_response = eval_turn.get("response", "")
                    if not isinstance(eval_response, str) or not eval_response.strip() or "[Error:" in eval_response:
                        if isinstance(content, str) and content.strip() and "[Error:" not in content:
                            needs_regeneration = True
                            print(f"[judge] [{run_name}] Turn {turn_idx+1} in evaluation was empty, but source now has content. Marking for re-evaluation.", flush=True)
            
            existing_turns = len(existing_report)
            current_turns = len(prompts)
            
            # Only skip if we have evaluated all current turns AND don't need any specific re-generations
            if not needs_regeneration and existing_turns >= current_turns:
                print(
                    f"[judge] [{run_name}] skipping {relative_path.name} (already evaluated {existing_turns} turns)",
                    flush=True,
                )
                return
        except Exception as e:
            print(
                f"[judge] [{run_name}] WARNING: could not load existing evaluation for {relative_path.name}: {e}. Re-evaluating.",
                flush=True,
            )

    # Consider only the first `max_lines` user-assistant pairs.
    max_msgs = max_lines * 2
    if len(conversation) > max_msgs:
        conversation = conversation[:max_msgs]

    total_constraints = sum(len(tp["constraints_data"]) for tp in prompts)
    print(
        f"[judge] [{run_name}] evaluating {relative_path.name} "
        f"({len(prompts)} turns, {total_constraints} constraints)",
        flush=True,
    )

    try:
        evaluation_report = await evaluate_conversation_async(
            judge=judge,
            test_prompts=prompts,
            conversation=conversation,
            save_path=save_path,
            existing_report=existing_report,
            evaluate_task_correctness=evaluate_task_correctness,
        )
    except Exception as e:
        print(
            f"[judge] [{run_name}] ERROR evaluating {relative_path.name}: {e}",
            flush=True,
        )
        return

    # Results have already been saved incrementally during evaluate_conversation_async
    # No need to call save_evaluation_report again
    
    print(
        f"[judge] [{run_name}] completed: {relative_path.name}",
        flush=True,
    )


def _resolve_conversation_file(chat_dir: Path, relative_path: Path) -> Path:
    """Resolve conversation path, handling nested or flattened chat filenames.

    Primary: chat_dir/<relative_path>.jsonl (preserves subfolders).
    Fallback: chat_dir/<relative_path_with_separators_replaced_by_underscores>.jsonl
    """

    clean = str(relative_path.with_suffix(""))
    primary = chat_dir / f"{clean}.jsonl"
    if primary.exists():
        return primary

    flat = clean.replace("/", "_").replace("\\", "_")
    alt = chat_dir / f"{flat}.jsonl"
    if alt.exists():
        return alt

    raise FileNotFoundError(
        f"Conversation file not found. Tried '{primary}' and '{alt}'."
    )


async def evaluate_conversation_async(
    *,
    judge: JudgeConfig,
    test_prompts: TestPrompts,
    conversation: Conversation,
    postprocess_function: Callable[[str], str] = lambda x: x,
    save_path: Optional[Path] = None,
    existing_report: Optional[EvaluationReport] = None,
    evaluate_task_correctness: bool = False,
):
    # Extract valid alternating user-assistant pairs
    # Stop when we encounter non-alternating messages or reach the end
    user_msgs = []
    assistant_msgs = []
    
    for i in range(0, len(conversation) - 1, 2):
        # Check if we have a valid user-assistant pair
        if conversation[i]["role"] == "user" and conversation[i + 1]["role"] == "assistant":
            user_msgs.append(conversation[i])
            assistant_msgs.append(conversation[i + 1])
        else:
            # Stop at first non-alternating pattern
            break
    
    # Log if we had to truncate incomplete conversation
    if len(conversation) % 2 != 0:
        print(f"[judge] Incomplete conversation: evaluating {len(user_msgs)} complete pairs, ignoring trailing user message", flush=True)
    elif len(user_msgs) < len(conversation) // 2:
        print(f"[judge] Non-alternating messages detected: evaluating only first {len(user_msgs)} valid pairs", flush=True)
    
    # If no valid pairs found, return empty results
    if not user_msgs:
        return []
    
    # Truncate test_prompts to match the number of valid pairs
    test_prompts = test_prompts[: len(user_msgs)]

    prompt_inputs: list[tuple[str, str, list[ConstraintData]]] = []

    for i, (test_prompt, user_msg, assistant_msg) in enumerate(
        zip(test_prompts, user_msgs, assistant_msgs)
    ):
        user_prompt = user_msg["content"]
        assert user_prompt == test_prompt["prompt"], f"User prompt does not match test prompt at index {i}."
        assistant_response = postprocess_function(assistant_msg["content"])
        
        if evaluate_task_correctness:
            # Extract task using process_prompt
            task_text, _ = process_prompt(user_prompt)
            
            # Mock constraint data representing the task
            task_mock_constraint: ConstraintData = {
                "id": "task",
                "name": "Task Completion",
                "text": task_text.strip(),
                "description": "Satisfy the task.",
                "instruction": "",
                "dependencies": [],
                "kwargs": {}
            }
            prompt_inputs.append((user_prompt, assistant_response, [task_mock_constraint]))
        else:
            prompt_inputs.append((user_prompt, assistant_response, test_prompt["constraints_data"]))

    # Create a single shared client for all judge calls
    client = AsyncOpenAI(
        base_url=judge.api_url,
        api_key=judge.api_key,
        timeout=Timeout(600.0, connect=10.0),
    )
    
    # Use a very conservative semaphore to avoid overwhelming the model server
    # Lower = more reliable, higher = faster but more failures
    semaphore = asyncio.Semaphore(6)
    
    # Open output file for incremental writing if save_path is provided
    output_file = None
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(save_path, "w")
    
    try:
        grouped: list[list[ConstraintEvaluation]] = []
        
        for turn_idx, (user_prompt, assistant_response, constraints) in enumerate(prompt_inputs):
            # Check if we can reuse the existing judgment for this turn
            can_reuse = False
            if existing_report and turn_idx < len(existing_report):
                eval_turn = existing_report[turn_idx]
                eval_response = eval_turn.get("response", "")
                
                # Check if the evaluation response matches the current response exactly
                # AND it's not one of our "empty/error" markers
                is_eval_healthy = isinstance(eval_response, str) and eval_response.strip() and "[Error:" not in eval_response
                if is_eval_healthy and eval_response == assistant_response:
                    can_reuse = True
            
            if can_reuse:
                print(f"[judge] Reuse valid judgment for turn {turn_idx + 1}", flush=True)
                turn_results = existing_report[turn_idx]["constraint_evaluations"]
            else:
                # Create tasks only for this turn's constraints
                turn_tasks = [
                    judge_async(
                        judge=judge,
                        client=client,
                        constraint=constraint,
                        user_prompt=user_prompt,
                        assistant_response=assistant_response,
                        semaphore=semaphore,
                        evaluate_task_correctness=evaluate_task_correctness,
                    )
                    for constraint in constraints
                ]
                
                # Wait for this turn's evaluations to complete before moving to next turn
                turn_results = await asyncio.gather(*turn_tasks)
            
            grouped.append(turn_results)
            
            total_constraints = sum(len(p[2]) for p in prompt_inputs)
            completed_constraints = sum(len(g) for g in grouped)
            print(f"[judge] progress {completed_constraints}/{total_constraints} constraints", flush=True)
            
            # Save this turn's results incrementally
            if output_file:
                prompt_eval = PromptEvaluation(
                    constraint_evaluations=turn_results,
                    prompt=user_prompt,
                    response=assistant_response,
                )
                _save_prompt_evaluation(output_file, prompt_eval)
            
            # Add a small delay between turns to give the model server breathing room
            if not can_reuse and turn_idx < len(prompt_inputs) - 1:
                await asyncio.sleep(0.5)
    finally:
        if output_file:
            output_file.close()
        # Ensure the client is properly closed
        await client.close()

    # Return results for compatibility (they've already been saved if save_path provided)
    evaluation_results: EvaluationReport = []
    for (user_prompt, assistant_response, _), constraint_evals in zip(prompt_inputs, grouped):
        evaluation_results.append(
            PromptEvaluation(
                constraint_evaluations=constraint_evals,
                prompt=user_prompt,
                response=assistant_response,
            )
        )

    return evaluation_results


def save_evaluation_report(path: Path, report: EvaluationReport) -> None:
    """Persist evaluation results while stripping transient judge-only fields."""

    def _clean_constraint_eval(constraint_eval: ConstraintEvaluation) -> dict:
        # Drop fields we don't want to store in the judge outputs.
        kept = {
            k: v
            for k, v in constraint_eval.items()
            if k not in {"followed", "reason"}
        }
        return kept

    def _clean_prompt_eval(prompt_eval: PromptEvaluation) -> dict:
        return {
            "constraint_evaluations": [
                _clean_constraint_eval(ce) for ce in prompt_eval["constraint_evaluations"]
            ],
            "prompt": prompt_eval["prompt"],
            "response": prompt_eval["response"],
        }

    with open(path, "w") as f:
        for item in report:
            f.write(json.dumps(_clean_prompt_eval(item)) + "\n")


def _save_prompt_evaluation(file_handle, prompt_eval: PromptEvaluation) -> None:
    """Save a single prompt evaluation to an open file handle (incremental saving)."""
    
    def _clean_constraint_eval(constraint_eval: ConstraintEvaluation) -> dict:
        # Drop fields we don't want to store in the judge outputs.
        kept = {
            k: v
            for k, v in constraint_eval.items()
            if k not in {"followed", "reason"}
        }
        return kept

    cleaned = {
        "constraint_evaluations": [
            _clean_constraint_eval(ce) for ce in prompt_eval["constraint_evaluations"]
        ],
        "prompt": prompt_eval["prompt"],
        "response": prompt_eval["response"],
    }
    
    file_handle.write(json.dumps(cleaned) + "\n")
    file_handle.flush()  # Ensure data is written immediately


def load_evaluation_report(path: Path) -> EvaluationReport:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


class ConstraintRateOverall(TypedDict):
    rate: float
    successful_turns: int
    total_turns: int


class TurnStats(TypedDict):
    rate: float
    successful_turns: int
    total_turns: int


ConstraintRateTurns = dict[int, TurnStats]


class AvgTurnsNoError(TypedDict):
    avg_turns: float
    list_of_consecutive_successes: list[int]
    total_conversations: int


class ConstraintRateTurnsCumulative(TypedDict):
    rate: float
    successful_turns: int
    total_turns: int


def _turn_success(prompt_eval: PromptEvaluation) -> tuple[bool, bool]:
    """Return (has_judgment, turn_success) for a single turn.

    Turn is successful only if every judged constraint is followed and at least
    one constraint was judged. A turn with no parsed judgments is ignored by
    callers via the has_judgment flag.

    Turns with empty responses are ignored (has_judgment = False).
    """

    # Ignore turns with empty responses (likely inference failures)
    response = prompt_eval.get("response", "")
    if not isinstance(response, str) or not response.strip():
        return False, False

    judged_flags = [
        constraint_eval.get("followed")
        for constraint_eval in prompt_eval["constraint_evaluations"]
        if constraint_eval.get("followed") is not None
    ]

    if not judged_flags:
        return False, False

    turn_success = all(judged_flags)
    return True, turn_success


def compute_overall_constraint_rate(
    eval_reports: list[EvaluationReport],
) -> ConstraintRateOverall:
    total_turns = 0
    successful_turns = 0

    for eval_report in eval_reports:
        for prompt_eval in eval_report:
            has_judgment, turn_success = _turn_success(prompt_eval)
            if not has_judgment:
                continue

            total_turns += 1
            if turn_success:
                successful_turns += 1

    overall_rate = (successful_turns / total_turns) if total_turns else 0.0

    return {
        "rate": overall_rate,
        "successful_turns": successful_turns,
        "total_turns": total_turns,
    }


def compute_per_turn_constraint_rate(
    eval_reports: list[EvaluationReport],
) -> ConstraintRateTurns:
    turn_success: dict[int, dict[str, int]] = {}

    for eval_report in eval_reports:
        for turn, prompt_eval in enumerate(eval_report, start=1):
            has_judgment, turn_success_flag = _turn_success(prompt_eval)
            if not has_judgment:
                continue

            if turn not in turn_success:
                turn_success[turn] = {"successful_turns": 0, "total_turns": 0}

            turn_success[turn]["total_turns"] += 1
            if turn_success_flag:
                turn_success[turn]["successful_turns"] += 1

    turn_averages: ConstraintRateTurns = {}
    for turn, stats in turn_success.items():
        total = stats["total_turns"]
        success = stats["successful_turns"]
        turn_averages[turn] = TurnStats(
            rate=(success / total) if total else 0.0,
            successful_turns=success,
            total_turns=total,
        )

    return turn_averages


def compute_avg_turns_without_error(
    eval_reports: list[EvaluationReport],
) -> AvgTurnsNoError:
    turns_without_error: list[int] = []

    for eval_report in eval_reports:
        consecutive_success = 0
        for prompt_eval in eval_report:
            has_judgment, turn_success = _turn_success(prompt_eval)

            # Without a parsed classification we cannot count this turn as successful.
            if not has_judgment:
                turn_success = False

            if turn_success:
                consecutive_success += 1
            else:
                break
        turns_without_error.append(consecutive_success)

    avg_turns = (sum(turns_without_error) / len(turns_without_error)) if turns_without_error else 0.0

    return {
        "avg_turns": avg_turns,
        "list_of_consecutive_successes": turns_without_error,
        "total_conversations": len(turns_without_error),
    }


def compute_cumulative_constraint_rate_per_turn(
    eval_reports: list[EvaluationReport],
) -> dict[int, ConstraintRateTurnsCumulative]:
    turn_stats: dict[int, dict[str, int]] = {}

    for eval_report in eval_reports:
        cumulative_success = True
        for turn, prompt_eval in enumerate(eval_report, start=1):
            if turn not in turn_stats:
                turn_stats[turn] = {"successful_turns": 0, "total_turns": 0}

            has_judgment, turn_success = _turn_success(prompt_eval)

            if not has_judgment:
                # Skip turns with no parsed classification data.
                continue

            if cumulative_success:
                if turn_success:
                    turn_stats[turn]["successful_turns"] += 1
                else:
                    cumulative_success = False

            turn_stats[turn]["total_turns"] += 1

    turn_averages: dict[int, ConstraintRateTurnsCumulative] = {}
    for turn, stats in sorted(turn_stats.items()):
        total = stats["total_turns"]
        success = stats["successful_turns"]
        turn_averages[turn] = ConstraintRateTurnsCumulative(
            rate=(success / total) if total else 0.0,
            successful_turns=success,
            total_turns=total,
        )

    return turn_averages

