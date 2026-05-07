import jsonargparse
import json
import random
import re
import asyncio
from openai import AsyncOpenAI, Timeout

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from typing import Optional, Dict, Any

from visualization import _clean_model_name, CUSTOM_RD_YL_GN

from conversation import (
    RunConversationRequest,
    Message,
    run_conversations,
    _run_completion,
    _save_conversation,
    run_file,
    _run_all,
)
from constraints import load_constraints, load_constraint_groups, Constraint, constraint_to_data

from test_prompts import (
    generate_test_prompts_single,
    generate_test_prompts_tuples,
    generate_test_prompts_replace_add,
    generate_test_prompts_everything,
    generate_detailed_experiment_plan,
    save_test_prompts,
    load_testset,
    read_tasks,
)

from evaluation import (
    EvaluationReport,
    _evaluate_one,
    load_evaluation_report,
    JudgeConfig,
    compute_overall_constraint_rate,
    compute_per_turn_constraint_rate,
    compute_avg_turns_without_error,
    compute_cumulative_constraint_rate_per_turn,
)

from visualization import (
    save_overall_constraint_rate,
    save_avg_turns_without_error,
    save_per_turn_constraint_rate,
    save_cumulative_constraint_rate_per_turn,
    create_overall_constraint_rate_chart,
    create_avg_turns_without_error_chart,
    create_per_turn_constraint_rate_plot,
    create_per_turn_constraint_rate_heatmap,
    create_cumulative_constraint_rate_per_turn_plot,
    prepare_heatmap_data,
    create_heatmap,
)




def _select_tasks_and_constraints_flat(
    *, tasks_path: Path, constraints_path: Path, num_conversations: Optional[int] = None
) -> tuple[list[Path], list[Constraint]]:
    """Select task files and load flat constraints.

    If num_conversations is provided and tasks_path is a directory, sample that many
    files. If num_conversations is omitted, return all files in the directory. When
    tasks_path is a single file, num_conversations must be 1 or None.
    """

    if tasks_path.is_dir():
        candidates = [p for p in tasks_path.iterdir() if p.is_file()]
        if not candidates:
            raise ValueError(f"No files found in tasks_path directory: {tasks_path}")

        if num_conversations is None:
            selected_files = candidates
        else:
            if num_conversations < 1:
                raise ValueError("num_conversations must be at least 1")
            if len(candidates) < num_conversations:
                raise ValueError(
                    f"Requested {num_conversations} conversations but only "
                    f"{len(candidates)} task files available in {tasks_path}"
                )
            selected_files = random.sample(candidates, num_conversations)
    else:
        if num_conversations is not None and num_conversations > 1:
            raise ValueError("num_conversations > 1 requires tasks_path to be a directory")
        selected_files = [tasks_path]

    constraints = load_constraints(constraints_path)
    if num_conversations is not None:
        constraints = random.sample(constraints, min(num_conversations, len(constraints)))

    return selected_files, constraints


def _select_tasks_and_constraint_groups(
    *, tasks_path: Path, constraints_path: Path, num_conversations: int
) -> tuple[list[Path], list[list[Constraint]]]:
    if num_conversations < 1:
        raise ValueError("num_conversations must be at least 1")

    if tasks_path.is_dir():
        candidates = [p for p in tasks_path.iterdir() if p.is_file()]
        if not candidates:
            raise ValueError(f"No files found in tasks_path directory: {tasks_path}")
        if len(candidates) < num_conversations:
            raise ValueError(
                f"Requested {num_conversations} conversations but only "
                f"{len(candidates)} task files available in {tasks_path}"
            )
        selected_files = random.sample(candidates, num_conversations)
    else:
        if num_conversations > 1:
            raise ValueError("num_conversations > 1 requires tasks_path to be a directory")
        selected_files = [tasks_path]

    constraint_groups = load_constraint_groups(constraints_path)
    if len(constraint_groups) < num_conversations:
        raise ValueError(
            f"Requested {num_conversations} conversations but only "
            f"{len(constraint_groups)} constraint groups available in {constraints_path}"
        )
    constraint_groups = random.sample(constraint_groups, num_conversations)

    return selected_files, constraint_groups


def generate_testset_single(*,
    testset_dir: Path,
    tasks_path: Path,
    constraints_path: Path,
    num_prompts: Optional[int] = None,
    num_constraints: Optional[int] = None,
    replace: bool = False,
    repetitions: int = 1,
):
    """Generate one testset for each constraint; the constraint is given in the first turn."""

    selected_files, constraints = _select_tasks_and_constraints_flat(
        tasks_path=tasks_path,
        constraints_path=constraints_path,
        num_conversations=num_constraints,
    )

    testset_dir.mkdir(parents=True, exist_ok=True)

    for repetition in range(1, repetitions + 1):
        # selected_file = selected_files[(repetition - 1) % len(selected_files)]
        # print(f"Using tasks file: {selected_file}")
        # tasks = read_tasks(selected_file)

        for idx, constraint in enumerate(constraints):

            selected_file = selected_files[idx]
            print(f"Using tasks file: {selected_file}")
            tasks = read_tasks(selected_file)
            _num_prompts = len(tasks) if num_prompts is None else num_prompts

            prompts = generate_test_prompts_single(
                constraint=constraint,
                tasks=tasks,
                num_prompts=_num_prompts,
                replace=replace,
            )

            out_path = testset_dir / f"{constraint.slug}_{repetition}.jsonl"

            save_test_prompts(out_path, prompts)

            print(f"✅ Wrote {len(prompts)} turns to {out_path}")

    print("Batch generation completed!")


def generate_testset_tuples(*,
    testset_dir: Path,
    tasks_path: Path,
    constraints_path: Path,
    num_prompts: Optional[int] = None,
    num_conversations: Optional[int] = None,
    replace: bool = False,
    repetitions: int = 1,
):
    """Generate num_conversations testsets; each uses a constraint tuple all injected in turn 1."""

    selected_files, constraint_groups = _select_tasks_and_constraint_groups(
        tasks_path=tasks_path,
        constraints_path=constraints_path,
        num_conversations=num_conversations,
    )

    base_tag = constraints_path.stem
    digit_tag = "".join(ch for ch in base_tag if ch.isdigit())
    tuple_tag = digit_tag or base_tag

    testset_dir.mkdir(parents=True, exist_ok=True)

    for repetition in range(1, repetitions + 1):

        for idx in range(num_conversations):
            selected_file = selected_files[idx]
            constraints = constraint_groups[idx]

            print(f"Using tasks file: {selected_file}")
            tasks = read_tasks(selected_file)
            _num_prompts = len(tasks) if num_prompts is None else num_prompts

            prompts = generate_test_prompts_tuples(
                constraint=constraints,
                tasks=tasks,
                num_prompts=_num_prompts,
                replace=replace,
            )

            out_path = testset_dir / f"{tuple_tag}" / f"tuple_{idx + 1:02d}_{repetition}.jsonl"

            save_test_prompts(out_path, prompts)

            print(f"✅ Wrote {len(prompts)} turns to {out_path}")

    print("Batch generation completed!")


def generate_testset_replace(*,
    testset_dir: Path,
    tasks_path: Path,
    constraints_path: Path,
    step_sizes: list[int],
    num_conversations: int,
    seed: int = 42,
    seed_increment: int = 15,
    num_prompts: Optional[int] = None,
    replace: bool = False,
    repetitions: int = 1,
):
    """Generate testsets that replace constraints every step_size turns, forgetting previous ones.

    Accepts a flat constraints JSONL file (same shape as generate_testset_single).
    """

    selected_files, constraints = _select_tasks_and_constraints_flat(
        tasks_path=tasks_path,
        constraints_path=constraints_path,
        num_conversations=num_conversations,
    )

    tasks_payload = []
    for file_idx, selected_file in enumerate(selected_files):
        tasks = read_tasks(selected_file)
        _num_prompts = len(tasks) if num_prompts is None else num_prompts
        tasks_payload.append((file_idx, selected_file, tasks, _num_prompts))

    testset_dir.mkdir(parents=True, exist_ok=True)

    current_seed = seed
    for step_size in step_sizes:
        step_dir = testset_dir / f"step_size_{step_size}"
        step_dir.mkdir(parents=True, exist_ok=True)

        for repetition in range(1, repetitions + 1):
            for idx, (file_idx, selected_file, tasks, _num_prompts) in enumerate(tasks_payload):
                random.seed(current_seed)

                needed_constraints = max(1, (_num_prompts + step_size - 1) // step_size)
                sample_size = min(len(constraints), needed_constraints)
                constraints_for_run = random.sample(constraints, sample_size)

                prompts = generate_test_prompts_replace_add(
                    constraints=constraints_for_run,
                    tasks=tasks,
                    experiment="replace",
                    step_size=step_size,
                    min_constraints=1,
                    num_prompts=_num_prompts,
                    replace=replace,
                )

                out_path = step_dir / f"conversation_{file_idx}_{repetition}.jsonl"
                save_test_prompts(out_path, prompts)

                print(
                    f"✅ Generated conversation {selected_file}, repetition {repetition} for "
                    f"step_size {step_size} with seed {current_seed}"
                )

                current_seed += seed_increment

    print("Batch generation completed!")


def generate_testset_add(*,
    testset_dir: Path,
    tasks_path: Path,
    constraints_path: Path,
    step_sizes: list[int],
    num_conversations: int,
    seed: int = 42,
    seed_increment: int = 15,
    num_prompts: Optional[int] = None,
    replace: bool = False,
    repetitions: int = 1,
):
    """Generate testsets that add a pre-grouped list of constraints every step_size turns."""

    selected_files, constraint_groups = _select_tasks_and_constraint_groups(
        tasks_path=tasks_path,
        constraints_path=constraints_path,
        num_conversations=num_conversations,
    )

    tasks_payload = []
    for file_idx, selected_file in enumerate(selected_files):
        tasks = read_tasks(selected_file)
        _num_prompts = len(tasks) if num_prompts is None else num_prompts
        tasks_payload.append((file_idx, selected_file, tasks, _num_prompts))

    testset_dir.mkdir(parents=True, exist_ok=True)

    current_seed = seed
    for step_size in step_sizes:
        step_dir = testset_dir / f"step_size_{step_size}"
        step_dir.mkdir(parents=True, exist_ok=True)

        for repetition in range(1, repetitions + 1):
            groups_for_run = random.sample(constraint_groups, num_conversations)

            for idx, (file_idx, selected_file, tasks, _num_prompts) in enumerate(tasks_payload):
                random.seed(current_seed)

                prompts = generate_test_prompts_replace_add(
                    constraints=groups_for_run[idx],
                    tasks=tasks,
                    experiment="add",
                    step_size=step_size,
                    min_constraints=len(groups_for_run[idx]),
                    num_prompts=_num_prompts,
                    replace=replace,
                )

                out_path = step_dir / f"conversation_{file_idx}_{repetition}.jsonl"
                save_test_prompts(out_path, prompts)

                print(
                    f"✅ Generated conversation {selected_file}, repetition {repetition} for "
                    f"step_size {step_size} with seed {current_seed}"
                )

                current_seed += seed_increment

    print("Batch generation completed!")


def _resolve_constraint_files(constraints_dir: Path) -> tuple[Path, dict[int, Path]]:
    """Locate single.jsonl and tuple constraint files inside a directory."""

    if not constraints_dir.is_dir():
        raise ValueError(f"constraints_path must be a directory: {constraints_dir}")

    single_candidates = [constraints_dir / "single.jsonl", constraints_dir / "single"]
    single_file = next((p for p in single_candidates if p.exists()), None)
    if single_file is None:
        single_file = next((p for p in constraints_dir.iterdir() if p.is_file() and p.stem == "single"), None)
    if single_file is None:
        raise ValueError(f"No single constraints file found in {constraints_dir} (expected 'single.jsonl')")

    tuple_files: dict[int, Path] = {}
    for p in constraints_dir.iterdir():
        if not p.is_file():
            continue
        try:
            size = int(p.stem)
        except ValueError:
            continue
        if size >= 2:
            tuple_files[size] = p

    if not tuple_files:
        raise ValueError(f"No tuple constraint files found in {constraints_dir}")

    return single_file, tuple_files
    

def generate_testset_everything(*,
    testset_dir: Path,
    tasks_path: Path,
    constraints_path: Path,
    num_conversations: int,
    seed: int = 42,
    seed_increment: int = 15,
    min_step_size: int = 1,
    max_step_size: int = 8,
    min_constraints: int = 3,
    max_constraints: int = 8,
    num_prompts: Optional[int] = None,
    replace: bool = False,
    repetitions: int = 1,
):
    """Generate testsets with random add/replace decisions and random step sizes using single constraints and tuple files."""

    single_file, tuple_files = _resolve_constraint_files(constraints_dir=constraints_path)

    # Use the helper function to select task files and constraints
    selected_files, _ = _select_tasks_and_constraints_flat(
        tasks_path=tasks_path,
        constraints_path=single_file,
        num_conversations=num_conversations,
    )

    # Load single constraints
    single_constraints = load_constraints(single_file)

    # Load only 3.jsonl for tuple constraints
    all_tuple_constraints: list[list[Constraint]] = []
    if 3 in tuple_files:
        tuple_groups = load_constraint_groups(tuple_files[3])
        all_tuple_constraints.extend(tuple_groups)

    tasks_payload = []
    for file_idx, selected_file in enumerate(selected_files):
        tasks = read_tasks(selected_file)
        _num_prompts = len(tasks) if num_prompts is None else num_prompts
        tasks_payload.append((file_idx, selected_file, tasks, _num_prompts))

    testset_dir.mkdir(parents=True, exist_ok=True)

    # Generate a separate detailed plan for each conversation (file_idx)
    # Each plan will be used across all repetitions of that conversation
    conversation_plans = {}
    for file_idx, _, _, _num_prompts in tasks_payload:
        # Set seed for this specific conversation's plan
        random.seed(seed + file_idx)
        conversation_plans[file_idx] = generate_detailed_experiment_plan(
            min_step_size=min_step_size,
            max_step_size=max_step_size,
            num_prompts=_num_prompts,
            max_tuple_size=3,  # Match constraints/everything/3.jsonl
        )

    current_seed = seed
    for repetition in range(1, repetitions + 1):
        for file_idx, selected_file, tasks, _num_prompts in tasks_payload:
            # Use seed for selecting constraints and tasks (varies per repetition)
            random.seed(current_seed)

            # Use the fixed plan for this conversation (same across all repetitions)
            prompts = generate_test_prompts_everything(
                single_constraints=single_constraints,
                tuple_constraints=all_tuple_constraints,
                tasks=tasks,
                min_step_size=min_step_size,
                max_step_size=max_step_size,
                num_prompts=_num_prompts,
                replace=replace,
                experiment_plan=conversation_plans[file_idx],
            )

            out_path = testset_dir / f"conversation_{file_idx}_{repetition}.jsonl"
            save_test_prompts(out_path, prompts)

            print(f"✅ Generated conversation {selected_file}, repetition {repetition} with seed {current_seed}")

            current_seed += seed_increment

    print("Batch generation completed!")


def generate_responses(*, 
    run_dir: Path,
    testset_dir: Path,
    output_dir: Path,
    api_model: str,
    max_turns: int = 60,
    api_url: str = "http://localhost:30000/v1",
    api_key: str = "None",
    max_tokens: int = 8192,
    batch_size: int = 10,
    max_concurrent: int = 8,
):
    """
    For each testset in testset_dir, generate responses using the specified model and save the conversations in output_dir.
    Processes files in batches to avoid overwhelming the system with too many concurrent tasks.
    """
    
    # Load all requests
    all_requests = []
    for file, prompts in load_testset(testset_dir):
        relative_path = file.relative_to(testset_dir)
        clean_name = str(relative_path.with_suffix("")).replace("/", "_").replace("\\", "_")
        user_msgs = [
            Message(role="user", content=prompt["prompt"]) for prompt in prompts
        ]
        user_msgs = user_msgs[:max_turns]
        request = RunConversationRequest(
            user_msgs=user_msgs,
            save_path=output_dir / f"{clean_name}.jsonl"
        )
        all_requests.append(request)
    
    total_requests = len(all_requests)
    print(f"📊 Processing {total_requests} conversations in batches of {batch_size}")
    
    # Process in batches
    for batch_idx in range(0, total_requests, batch_size):
        batch_requests = all_requests[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        total_batches = (total_requests + batch_size - 1) // batch_size
        
        print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_requests)} conversations)")
        
        asyncio.run(run_conversations(
            requests=batch_requests,
            api_url=api_url,
            api_model=api_model,
            api_key=api_key,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
        ))
        
        print(f"✅ Completed batch {batch_num}/{total_batches}")
    
    print(f"🎉 All {total_requests} conversations completed!")
    

def generate_responses_baselines(*,
    run_dir: Path,
    testset_dir: Path,
    output_dir: Path,
    api_model: str,
    max_turns: int = 60,
    api_url: str = "http://localhost:30000/v1",
    api_key: str = "None",
    max_tokens: int = 8192,
    batch_size: int = 50,
    max_concurrent: int = 100,
    max_retries: int = 5,
    initial_retry_delay: float = 1.0,
):
    """
    For each baseline file in `testset_dir`, send each prompt as a separate
    single-turn request (no conversation history) but save the sequence of
    prompts+responses as a conversation JSON list in `output_dir` with the
    same basename as the baseline file.
    Processes files in batches to avoid overwhelming the system.
    """
    
    async def _run_batches():
        
        client = AsyncOpenAI(
            base_url=api_url,
            api_key=api_key,
            timeout=Timeout(600.0, connect=10.0),
        )

        try:
            # Build per-file tasks (one task per baseline file)
            all_tasks = []
            semaphore = asyncio.Semaphore(max_concurrent)
            output_dir.mkdir(parents=True, exist_ok=True)

            for file, prompts in load_testset(testset_dir):
                # Optionally limit number of lines/prompts per file
                if max_turns is not None:
                    prompts = prompts[:max_turns]
                relative_path = file.relative_to(testset_dir)
                base_name = str(relative_path.with_suffix("")).replace("/", "_").replace("\\", "_")
                save_path = output_dir / f"{base_name}.jsonl"
                all_tasks.append(run_file(file, prompts, save_path, semaphore, client, api_model, max_tokens, run_completion_fn=_run_completion, save_conversation_fn=_save_conversation, max_retries=max_retries, initial_retry_delay=initial_retry_delay))

            if not all_tasks:
                print(f"No baseline prompts found in {testset_dir}")
                return

            total_tasks = len(all_tasks)
            print(f"📊 Processing {total_tasks} baseline files in batches of {batch_size}")
            
            # Process in batches
            for batch_idx in range(0, total_tasks, batch_size):
                batch_tasks = all_tasks[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1
                total_batches = (total_tasks + batch_size - 1) // batch_size
                
                print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_tasks)} files)")
                
                await _run_all(batch_tasks)
                
                print(f"✅ Completed batch {batch_num}/{total_batches}")
            
            print(f"🎉 All {total_tasks} baseline files completed!")
        finally:
            await client.close()
    
    asyncio.run(_run_batches())

    
def evaluate(*,
    run_dir: Path,
    testset_dir: Path,
    output_dir: Path,
    api_model: str,
    max_turns: int = 60,
    api_url: str = "http://localhost:30000/v1",
    api_key: str = "None",
    max_tokens: int = 8192,
    batch_size: int = 50,
    max_concurrent: int = 100,
):
    """Evaluate a single run's `chat/` folder using the given testset.

    Simplified assumptions:
    - `run_dir` is a specific run folder containing a `chat/` subfolder.
    - `testset_dir` is the correct testset folder to evaluate against.

    Launch one evaluation task per conversation file concurrently.
    batch_size: Number of files to process per batch
    max_concurrent: Maximum number of concurrent evaluation requests
    """

    chat_dir = run_dir / "chat"
    if not chat_dir.is_dir():
        raise ValueError(f"No chat folder found in run_dir: {run_dir} (expected {chat_dir})")

    judge = JudgeConfig(
        api_url=api_url,
        api_model=api_model,
        api_key=api_key,
        max_tokens=max_tokens,
    )

    evaluation_judge_dir = output_dir / "judge"
    evaluation_judge_dir.mkdir(parents=True, exist_ok=True)

    conversations: list[tuple[Path, list[dict]] | tuple[Path, list]] = []
    for file, prompts in load_testset(testset_dir):
        prompts = prompts[:max_turns]
        conversations.append((file.relative_to(testset_dir), prompts))

    # Use provided max_concurrent for limiting concurrent evaluation requests
    file_semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(rel, prm):
        async with file_semaphore:
            await _evaluate_one(
                rel,
                prm,
                chat_dir=chat_dir,
                evaluation_judge_dir=evaluation_judge_dir,
                judge=judge,
                run_name=run_dir.name,
                max_lines=max_turns,
            )

    # Process in batches
    total_files = len(conversations)
    print(f"📊 Processing {total_files} files in batches of {batch_size}")
    
    for batch_idx in range(0, total_files, batch_size):
        batch_conversations = conversations[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        print(f"🔄 Processing evaluation batch {batch_num}/{total_batches} ({len(batch_conversations)} files)")
        
        tasks = []
        for rel, prm in batch_conversations:
            tasks.append(evaluate_with_semaphore(rel, prm))

        if tasks:
            asyncio.run(_run_all(tasks))
        
        print(f"✅ Completed evaluation batch {batch_num}/{total_batches}")



def evaluate_task(*,
    run_dir: Path,
    testset_dir: Path,
    output_dir: Path,
    api_model: str,
    max_turns: int = 60,
    api_url: str = "http://localhost:30000/v1",
    api_key: str = "None",
    max_tokens: int = 8192,
    batch_size: int = 50,
    max_concurrent: int = 100,
):
    chat_dir = run_dir / "chat"
    if not chat_dir.is_dir():
        raise ValueError(f"No chat folder found in run_dir: {run_dir} (expected {chat_dir})")

    judge = JudgeConfig(
        api_url=api_url,
        api_model=api_model,
        api_key=api_key,
        max_tokens=max_tokens,
    )

    evaluation_judge_dir = output_dir / "judge"
    evaluation_judge_dir.mkdir(parents=True, exist_ok=True)

    conversations: list[tuple[Path, list[dict]] | tuple[Path, list]] = []
    for file, prompts in load_testset(testset_dir):
        prompts = prompts[:max_turns]
        conversations.append((file.relative_to(testset_dir), prompts))

    file_semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_task_with_semaphore(rel, prm):
        async with file_semaphore:
            await _evaluate_one(
                rel,
                prm,
                chat_dir=chat_dir,
                evaluation_judge_dir=evaluation_judge_dir,
                judge=judge,
                run_name=run_dir.name,
                max_lines=max_turns,
                evaluate_task_correctness=True,
            )

    total_files = len(conversations)
    print(f"📊 Processing {total_files} task evals in batches of {batch_size}")
    
    for batch_idx in range(0, total_files, batch_size):
        batch_conversations = conversations[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        print(f"🔄 Processing task eval batch {batch_num}/{total_batches} ({len(batch_conversations)} files)")
        
        tasks = []
        for rel, prm in batch_conversations:
            tasks.append(evaluate_task_with_semaphore(rel, prm))

        if tasks:
            asyncio.run(_run_all(tasks))
        
        print(f"✅ Completed task eval batch {batch_num}/{total_batches}")


def compute_eval_scores(*, 
    evaluation_dir: Path,
    output_dir: Path
):
    """
    Load all evaluation reports in evaluation_dir, group files by common base name (everything before the last underscore), compute evaluation metrics for each group, and save per-group JSON score files into output_dir.

    Returns a dict mapping group base name -> computed scores.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group files by the substring before the first "_conversation".
    # All files without "_conversation" go into a single "ungrouped" group.
    groups: Dict[str, list[Path]] = {}
    # Additionally, collect per-tuple-size groups when file stems look like "<size>_tuple_*".
    tuple_size_groups: Dict[str, list[Path]] = {}

    tuple_re = re.compile(r"^(?P<size>\d+)_tuple")

    for file in evaluation_dir.glob("*.jsonl"):
        stem = file.stem
        idx = stem.find("_conversation")
        if idx != -1:
            base = stem[:idx] or "root"
        else:
            base = "single"
        groups.setdefault(base, []).append(file)

        m = tuple_re.match(stem)
        if m:
            size = m.group("size")
            tuple_size_groups.setdefault(size, []).append(file)

    if not groups:
        print(f"No evaluation report files found in {evaluation_dir}")
        return {}

    all_scores: Dict[str, Any] = {}

    def _compute_and_save(label: str, files: list[Path]):
        eval_reports: list[EvaluationReport] = []
        for file in sorted(files):
            eval_reports.append(load_evaluation_report(file))

        scores = {
            "overall_constraint_rate": compute_overall_constraint_rate(eval_reports),
            "per_turn_constraint_rate": compute_per_turn_constraint_rate(eval_reports),
            "avg_turns_without_error": compute_avg_turns_without_error(eval_reports),
            "cumulative_constraint_rate_per_turn": compute_cumulative_constraint_rate_per_turn(eval_reports),
        }

        out_file = output_dir / f"{label}_eval_scores.json"
        with open(out_file, "w") as f:
            json.dump(scores, f, indent=2)

        print(f"✅ Saved evaluation scores for group '{label}' to {out_file}")
        all_scores[label] = scores

    # Save scores for groups, but skip "single" if we have tuple_size_groups (tuples folder)
    for base, files in groups.items():
        if base == "single" and tuple_size_groups:
            # Skip saving aggregate "single" scores for tuples folders
            continue
        _compute_and_save(base, files)

    for size, files in tuple_size_groups.items():
        _compute_and_save(f"tuple_size_{size}", files)

    return all_scores


def compute_eval_scores_tree(*,
    root_dir: Path,
    output_dir: Path,
):
    """
    Walk a directory tree of evaluation reports and
    compute scores.

    If root_dir has JSONL files, write scores into
    output_dir.

    For each immediate subfolder, write scores into
    output_dir/<subfolder_name>.
    """

    if not root_dir.exists():
        print(f"Root dir not found: {root_dir}")
        return {}

    any_work = False

    has_files = any(p.is_file() for p in root_dir.iterdir())
    if has_files:
        any_work = True
        compute_eval_scores(
            evaluation_dir=root_dir,
            output_dir=output_dir,
        )

    for sub in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        any_work = True
        compute_eval_scores(
            evaluation_dir=sub,
            output_dir=output_dir / sub.name,
        )

    if not any_work:
        print(f"No files or subfolders in {root_dir}")
    return {}


def compare_models_scores(*, 
    scores_dir: Path,
    output_dir: Path
):
    """
    scores_dir contains one subfolder per model.
    Each model folder contains score files with the same filenames across models.
    For each distinct filename found across all model folders, collect that file from every model
    (if present), build a mapping model_name -> scores, and produce the text files and visualizations for that filename into a per-method subfolder of output_dir.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect model folders
    model_dirs = [p for p in scores_dir.iterdir() if p.is_dir()]
    if not model_dirs:
        print(f"No model subfolders found in {scores_dir}")
        return

    # Map: model_name -> { filename -> Path }
    model_files: Dict[str, Dict[str, Path]] = {}
    all_filenames = set()

    for model_dir in model_dirs:
        model_name = model_dir.name
        files = {}
        for score_path in model_dir.glob("*.json*"):
            if not score_path.is_file():
                continue
            fname = score_path.name
            files[fname] = score_path
            all_filenames.add(fname)
        model_files[model_name] = files

    if not all_filenames:
        print(f"No score files found in any model subfolder of {scores_dir}")
        return

    # For each distinct filename (method), gather scores from all models and produce outputs
    for fname in sorted(all_filenames):
        model_scores: Dict[str, Any] = {}
        missing_models = []
        for model_name, files in model_files.items():
            path = files.get(fname)
            if path and path.is_file():
                with open(path, "r") as f:
                    try:
                        scores = json.load(f)
                    except Exception as e:
                        print(f"⚠️ Failed to load {path}: {e}")
                        continue
                model_scores[model_name] = scores
            else:
                missing_models.append(model_name)

        if not model_scores:
            print(f"No valid score files found for '{fname}', skipping.")
            continue

        if missing_models:
            print(f"⚠️ For '{fname}', missing files for models: {', '.join(missing_models)}")

        # Create an output subfolder for this method (use stem to drop extension)
        method_base = Path(fname).stem
        method_output = output_dir / method_base
        method_output.mkdir(parents=True, exist_ok=True)

        # Save text files and visualizations using the existing helpers
        save_overall_constraint_rate(model_scores, method_output)
        save_avg_turns_without_error(model_scores, method_output)
        save_per_turn_constraint_rate(model_scores, method_output)
        save_cumulative_constraint_rate_per_turn(model_scores, method_output)

        create_overall_constraint_rate_chart(model_scores, method_output)
        create_avg_turns_without_error_chart(model_scores, method_output)
        create_per_turn_constraint_rate_plot(model_scores, method_output)
        create_per_turn_constraint_rate_heatmap(model_scores, method_output)
        create_cumulative_constraint_rate_per_turn_plot(model_scores, method_output)

        print(f"✅ Saved files and visualizations for '{fname}' to {method_output}")


def _infer_experiment(model_name: str) -> str | None:
    name = model_name.lower()
    if "single" in name:
        return "single"
    if "tuples" in name:
        return "tuples"
    if "tuples_first" in name:
        return "tuples_first"
    if "add" in name:
        return "add"
    if "replace" in name:
        return "replace"
    if "replace_tuples" in name:
        return "replace_tuples"
    if "everything" in name:
        return "everything"
    return None


def compare_models_scores_by_experiment(*,
    scores_dir: Path,
    output_dir: Path,
):
    """Compare models within each experiment bucket.

    Buckets are inferred from model folder names:
    single, tuples_first, add, replace, everything.
    Outputs are written to output_dir/<bucket>/<method>/.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = [p for p in scores_dir.iterdir() if p.is_dir()]
    if not model_dirs:
        print(f"No model subfolders found in {scores_dir}")
        return

    buckets: dict[str, dict[str, Path]] = {}
    for model_dir in model_dirs:
        exp = _infer_experiment(model_dir.name)
        if exp is None:
            print(f"⚠️ Could not infer experiment for {model_dir.name}, skipping")
            continue
        buckets.setdefault(exp, {})[model_dir.name] = model_dir

    if not buckets:
        print(f"No experiment buckets inferred from {scores_dir}")
        return

    for exp, models in sorted(buckets.items()):
        # Gather all filenames seen in this bucket
        all_filenames = set()
        per_model_files: dict[str, dict[str, Path]] = {}
        for model_name, model_dir in models.items():
            files = {}
            for score_path in model_dir.glob("*.json*"):
                if score_path.is_file():
                    files[score_path.name] = score_path
                    all_filenames.add(score_path.name)
            per_model_files[model_name] = files

        if not all_filenames:
            print(f"⚠️ No score files found for bucket {exp}")
            continue

        for fname in sorted(all_filenames):
            model_scores: Dict[str, Any] = {}
            missing_models = []
            for model_name, files in per_model_files.items():
                path = files.get(fname)
                if path and path.is_file():
                    with open(path, "r") as f:
                        try:
                            scores = json.load(f)
                        except Exception as e:
                            print(f"⚠️ Failed to load {path}: {e}")
                            continue
                    model_scores[model_name] = scores
                else:
                    missing_models.append(model_name)

            if not model_scores:
                print(f"No valid score files found for '{fname}' in bucket {exp}, skipping.")
                continue

            if missing_models:
                print(
                    f"⚠️ [{exp}] For '{fname}', missing files for: {', '.join(missing_models)}"
                )

            method_base = Path(fname).stem
            method_output = output_dir / exp / method_base
            method_output.mkdir(parents=True, exist_ok=True)

            # Join experiment bucket name with method filename to pass the full context (e.g., "add/step_size_10_eval_scores")
            # so the heatmap creation can correctly filter turns according to the step size.
            exp_context = f"{exp}/{method_base}"

            save_overall_constraint_rate(model_scores, method_output)
            save_avg_turns_without_error(model_scores, method_output)
            save_per_turn_constraint_rate(model_scores, method_output)
            save_cumulative_constraint_rate_per_turn(model_scores, method_output)

            create_overall_constraint_rate_chart(model_scores, method_output)
            create_avg_turns_without_error_chart(model_scores, method_output)
            create_per_turn_constraint_rate_plot(model_scores, method_output, experiment_name=exp_context)
            create_per_turn_constraint_rate_heatmap(model_scores, method_output, experiment_name=exp_context)
            create_cumulative_constraint_rate_per_turn_plot(model_scores, method_output)

            print(f"✅ [{exp}] Saved files and visualizations for '{fname}' to {method_output}")


def plot_per_turn_accuracy_comparison(*,
    scores_dir: Path,
    output_dir: Path,
    step_size_limit: Optional[int] = None,
):
    """
    Generate two line plots:
    1. Agregated per-turn accuracy across all models for each experiment bucket.
    2. Per-turn accuracy for the best model (via Borda count) across each experiment bucket.
    """

    scores_dir = Path(scores_dir)
    model_dirs = [p for p in scores_dir.iterdir() if p.is_dir()]
    
    # Priority for choosing which score file to use if multiple exist in an experiment
    # For 'single' it's single_eval_scores.json
    # For 'tuples' it's tuple_size_3_eval_scores.json
    # For 'add' and 'replace' we use step_size_5_eval_scores.json if it exists, else largest step size
    # For 'everything' it's single_eval_scores.json
    
    def get_best_score_file(exp: str, files: list[Path]) -> Path | None:
        if not files: return None
        if exp == "single":
            f = next((p for p in files if p.name == "single_eval_scores.json"), None)
            return f or files[0]
        if exp == "tuples":
            f = next((p for p in files if p.name == "tuple_size_3_eval_scores.json"), None)
            return f or files[0]
        if exp == "everything":
            f = next((p for p in files if p.name == "single_eval_scores.json"), None)
            return f or files[0]
        return files[0]

    # exp -> model_clean_name -> turn -> list of rates (to be averaged if multiple steps)
    turn_rates_accum: dict[str, dict[str, dict[int, list[float]]]] = {}
    models_list = set()
    
    for model_dir in model_dirs:
        exp = _infer_experiment(model_dir.name)
        if not exp: continue
        
        score_files = list(model_dir.glob("*.json"))
        
        files_to_process = []
        if exp in ["add", "replace"]:
            if step_size_limit is not None:
                target_file = next((p for p in score_files if p.name == f"step_size_{step_size_limit}_eval_scores.json"), None)
                if target_file:
                    files_to_process.append(target_file)
            else:
                # For add and replace, take step 5 and step 10 and average them
                step5 = next((p for p in score_files if p.name == "step_size_5_eval_scores.json"), None)
                step10 = next((p for p in score_files if p.name == "step_size_10_eval_scores.json"), None)
                if step5: files_to_process.append(step5)
                if step10: files_to_process.append(step10)
            
            if not files_to_process and score_files:
                files_to_process.append(get_best_score_file(exp, score_files))
        else:
            best_file = get_best_score_file(exp, score_files)
            if best_file: files_to_process.append(best_file)

        if not files_to_process: continue
        
        model_name = _clean_model_name(model_dir.name)
        models_list.add(model_name)

        for score_file in files_to_process:
            with open(score_file, "r") as f:
                scores = json.load(f)
            
            per_turn = scores.get("per_turn_constraint_rate", {})
            if not per_turn: continue
            
            turn_rates_accum.setdefault(exp, {}).setdefault(model_name, {})
            for turn_str, turn_data in per_turn.items():
                turn_rates_accum[exp][model_name].setdefault(int(turn_str), []).append(turn_data["rate"])

    # Now average the accumulated rates (e.g. across steps)
    # Average across the accumulated rates
    data_by_exp: dict[str, dict[str, dict[int, float]]] = {}
    for exp, models in turn_rates_accum.items():
        for model_name, turns in models.items():
            for turn, rates in turns.items():
                avg_rate = sum(rates) / len(rates)
                data_by_exp.setdefault(exp, {}).setdefault(model_name, {})[turn] = avg_rate

    if not data_by_exp:
        print("No data found to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Palatino font (or fallback)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'Palatino LT STD', 'Book Antiqua', 'Georgia', 'DejaVu Serif']

    # Define requested order and unique markers
    ordered_exps = ["single", "tuples", "replace", "add", "everything"]
    markers = {
        "single": "o",       # Circles
        "tuples": "s",       # Squares
        "replace": "^",      # Triangles
        "add": "x",          # x
        "everything": "*"    # Stars
    }
    
    # Filter available experiments that match the requested order
    exps_to_plot = [e for e in ordered_exps if e in data_by_exp]

    # Borda count to find best model overall
    model_borda_scores: dict[str, float] = {m: 0.0 for m in models_list}
    for exp in data_by_exp:
        exp_model_scores = []
        for model_name in data_by_exp[exp]:
            rates = list(data_by_exp[exp][model_name].values())
            avg_rate = sum(rates) / len(rates) if rates else 0
            exp_model_scores.append((model_name, avg_rate))
        exp_model_scores.sort(key=lambda x: x[1], reverse=True)
        num_models_in_exp = len(exp_model_scores)
        for rank, (model_name, _) in enumerate(exp_model_scores):
            model_borda_scores[model_name] += (num_models_in_exp - 1 - rank)
    
    best_model = "None"
    if model_borda_scores:
        best_model = max(model_borda_scores, key=model_borda_scores.get)
        print(f"Best model via Borda count: {best_model} (Score: {model_borda_scores[best_model]})")

    # Override model name for display if it matches Gemini
    best_model_display = "Gemini 3.1 Flash Lite" if "gemini" in best_model.lower() or "google" in best_model.lower() else best_model

    # 1. Combined Line Plot (Side by Side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    def _plot_lines(ax, data, title):
        for exp in exps_to_plot:
            if exp not in data: continue
            
            # turn -> list of rates (if aggregated or multiple models)
            turns_data = data[exp]
            
            if title == "Average across models":
                # For aggregated, turns_data is already dict[model, dict[turn, rate]]
                turn_rates_per_model = {}
                for model in turns_data:
                    for turn, rate in turns_data[model].items():
                        turn_rates_per_model.setdefault(turn, []).append(rate)
                
                sorted_turns = sorted(turn_rates_per_model.keys())
                avg_rates = [np.mean(turn_rates_per_model[t]) for t in sorted_turns]
                std_rates = [np.std(turn_rates_per_model[t]) for t in sorted_turns]
                
                # Plot average line
                ax.plot(sorted_turns, [r * 100 for r in avg_rates], marker=markers.get(exp, "x"), label=exp)
                # Plot shadow removed (standard deviation)
            else:
                # Single model - for "Best model", we don't have multiple models to compute std across,
                # but the user said "see the performance of each model on each line and compute the standard deviation"
                # which typically applies to the aggregated view. For the best model plot, 
                # we show only that model's performance without a shadow unless there are multiple runs/seeds.
                # Assuming shadow only for "Average across models".
                sorted_turns = sorted(turns_data.keys())
                rates = [turns_data[t] for t in sorted_turns]
                ax.plot(sorted_turns, [r * 100 for r in rates], marker=markers.get(exp, "x"), label=exp)
        
        ax.set_title(title, fontsize=19)
        ax.set_xlabel("Turn", fontsize=19)
        ax.set_ylabel("Accuracy (%)", fontsize=19)
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.tick_params(labelleft=True)

    _plot_lines(ax1, data_by_exp, "Average across models")
    best_model_data = {exp: data_by_exp[exp][best_model] for exp in exps_to_plot if best_model in data_by_exp[exp]}
    _plot_lines(ax2, best_model_data, f"{best_model_display}")

    # Shared Legend at bottom (single legend for both)
    handles, labels = ax1.get_legend_handles_labels()
    # capitalize names for legend
    labels = [l.capitalize() if l != "everything" else "Everything" for l in labels]

    legend = fig.legend(handles, labels, loc='lower center', ncol=len(exps_to_plot), frameon=False, 
                        bbox_to_anchor=(0.5, 0.05), fontsize=19)
    
    # Ensure legend title also has fontsize 19 if it existed
    if legend.get_title():
        legend.get_title().set_fontsize(19)
    
    # Standardize tick label sizes and axis labels for line plots
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=19)
        ax.xaxis.label.set_size(19)
        ax.yaxis.label.set_size(19)
        ax.title.set_size(19)

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    combined_line_path = output_dir / "combined_lines.pdf"
    plt.savefig(combined_line_path)
    print(f"✅ Saved combined line plot to {combined_line_path}")
    plt.close()

    # 2. Combined Heatmaps (One on Top of the Other)
    
    all_turns_raw = set()
    for exp in exps_to_plot:
        for model in data_by_exp[exp]:
            for turn in data_by_exp[exp][model].keys():
                all_turns_raw.add(turn)
    
    def _get_sampled_turns(all_turns_set):
        return sorted([t for t in all_turns_set if t == 1 or t % 5 == 0])
    
    sampled_turns = _get_sampled_turns(all_turns_raw)
    
    def _get_heatmap_df(data_source, is_agg=True):
        rows = []
        for exp in exps_to_plot:
            if exp not in data_source: continue
            row = {"Experiment": exp}
            for t in sampled_turns:
                if is_agg:
                    turn_rates = [data_source[exp][m][t] for m in data_source[exp] if t in data_source[exp][m]]
                    row[str(t)] = sum(turn_rates) / len(turn_rates) if turn_rates else np.nan
                else:
                    row[str(t)] = data_source[exp].get(t, np.nan)
            rows.append(row)
        return pd.DataFrame(rows).set_index("Experiment")

    df_agg = _get_heatmap_df(data_by_exp, is_agg=True)
    df_best = _get_heatmap_df(best_model_data, is_agg=False)

    fig, (ax_h1, ax_h2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Common colorbar axis (thinner)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])

    sns.heatmap(df_agg, annot=True, annot_kws={"size": 17}, cmap=CUSTOM_RD_YL_GN, fmt=".2f", vmin=0, vmax=1.0, 
                cbar=True, cbar_ax=cbar_ax, cbar_kws={'label': 'Accuracy'}, ax=ax_h1,
                linewidths=0.1, linecolor='lightgray')
    ax_h1.set_title("Average across models", fontsize=17)
    ax_h1.set_ylabel("")
    ax_h1.tick_params(axis='both', which='major', labelsize=17)
    
    sns.heatmap(df_best, annot=True, annot_kws={"size": 17}, cmap=CUSTOM_RD_YL_GN, fmt=".2f", vmin=0, vmax=1.0, 
                cbar=False, ax=ax_h2, linewidths=0.2, linecolor='lightgray')
    ax_h2.set_title(f"{best_model_display}", fontsize=17)
    ax_h2.set_ylabel("")
    ax_h2.set_xlabel("Turn", fontsize=17)
    ax_h2.tick_params(axis='both', which='major', labelsize=17)

    # Update colorbar tick label size
    cbar_ax.tick_params(labelsize=17)
    cbar_ax.yaxis.label.set_size(17)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    combined_heatmap_path = output_dir / "combined_heatmaps.pdf"
    plt.savefig(combined_heatmap_path)
    print(f"✅ Saved combined heatmap to {combined_heatmap_path}")
    plt.close()

    # (Keep previous logic if needed, but the user asked for these specific combinations)
    # The existing individual plots will still be generated by the rest of the function if not removed.
    # To keep the response clean, I'll stop here or just return.
    return # Skip the old individual plot generation for now to avoid cluttering if that was the intent.


def plot_add_replace_comparison(*,
    scores_dir: Path,
    output_path: Path,
):
    """
    Generate a plot with four lines:
    - Average across models
    - Lines: add 5, add 10, replace 5, replace 10
    - add and replace use different colors
    - 5 and 10 use different line styles (solid vs dotted)
    """

    scores_dir = Path(scores_dir)
    model_dirs = [p for p in scores_dir.iterdir() if p.is_dir()]
    
    # (exp, step) -> model_name -> turn -> rate
    data_accum: dict[tuple[str, int], dict[str, dict[int, float]]] = {}
    
    for model_dir in model_dirs:
        exp = _infer_experiment(model_dir.name)
        if exp not in ["add", "replace"]:
            continue
            
        model_name = _clean_model_name(model_dir.name)
        score_files = list(model_dir.glob("*.json"))
        
        for score_file in score_files:
            # Extract step size from filename like step_size_5_eval_scores.json
            match = re.search(r"step_size_(\d+)_eval_scores.json", score_file.name)
            if not match:
                continue
            step = int(match.group(1))
            if step not in [5, 10]:
                continue
                
            with open(score_file, "r") as f:
                scores = json.load(f)
            
            per_turn = scores.get("per_turn_constraint_rate", {})
            if not per_turn:
                continue
                
            key = (exp, step)
            data_accum.setdefault(key, {}).setdefault(model_name, {})
            for turn_str, turn_data in per_turn.items():
                data_accum[key][model_name][int(turn_str)] = turn_data["rate"]

    if not data_accum:
        print("No add/replace data for step 5 or 10 found.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'Palatino LT STD', 'Book Antiqua', 'Georgia', 'DejaVu Serif']
    plt.rcParams['font.size'] = 24

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define styles
    # Use different colors for all four lines
    config_styles = {
        ("add", 5): {"color": "tab:orange", "linestyle": "-", "marker": "o"},
        ("replace", 5): {"color": "tab:blue", "linestyle": "-", "marker": "o"},
        ("add", 10): {"color": "tab:red", "linestyle": "--", "marker": "s"},
        ("replace", 10): {"color": "tab:green", "linestyle": "--", "marker": "s"},
    }

    configs = [("add", 5), ("replace", 5), ("add", 10), ("replace", 10)]
    
    # To split into two columns: Add 5, Add 10 in Col 1; Replace 5, Replace 10 in Col 2
    # The order in 'configs' determines the order of labels in the handle list.
    # matplotlib fills ncol columns row by row by default.
    # With ncol=2 and 4 items:
    # Row 1: Item 0, Item 1
    # Row 2: Item 2, Item 3
    # So we want [Add 5, Replace 5, Add 10, Replace 10]
    
    for exp, step in configs:
        key = (exp, step)
        if key not in data_accum:
            continue
            
        models_data = data_accum[key]
        # Calculate average across models per turn
        turn_rates_per_model = {}
        for model in models_data:
            for turn, rate in models_data[model].items():
                turn_rates_per_model.setdefault(turn, []).append(rate)
        
        sorted_turns = sorted(turn_rates_per_model.keys())
        avg_rates = [np.mean(turn_rates_per_model[t]) for t in sorted_turns]
        
        label = f"{exp.capitalize()} {step}"
        style = config_styles[key]
        ax.plot(
            sorted_turns, 
            [r * 100 for r in avg_rates], 
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2
        )

    ax.set_xlabel("Turn", fontsize=29)
    ax.set_ylabel("Accuracy (%)", fontsize=29)
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Legend at the bottom, without a box, two columns: add on one, replace on the other
    ax.legend(fontsize=29, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=2, frameon=False, columnspacing=2.5, handletextpad=0.4)
    
    ax.tick_params(axis='both', which='major', labelsize=29)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()


TurnSets = list[int]


def generate_heatmaps_from_reports(*, 
    reports_dir: Path,
    output_dir: Path,
    turns: TurnSets = list(range(1, 21)),
):
    """
    Generate heatmaps from evaluation reports.
    """

    # Load evaluation reports
    eval_reports = {}
    for report_file in reports_dir.glob("*.jsonl"):
        eval_reports[report_file.name] = load_evaluation_report(report_file)

    # Prepare data for heatmap
    heatmap_data = prepare_heatmap_data(eval_reports, turns)

    heatmap_filename = f"{reports_dir.name}.pdf"
    heatmap_path = output_dir / heatmap_filename

    # Generate and save heatmap
    create_heatmap(
        heatmap_data,
        output_path=heatmap_path
    )

    print(f"✅ Saved heatmap to {heatmap_path}")


def _compute_relative_value(eval_val, baseline_val):
    """Compute ratio for a single value, handling edge cases."""
    if isinstance(eval_val, (int, float)) and isinstance(baseline_val, (int, float)):
        if baseline_val == 0:
            if eval_val == 0:
                return 1.0  # Both zero, consider ratio as 1
            else:
                return None # float('inf')  # Eval > 0, baseline = 0
        else:
            return eval_val / baseline_val
    elif isinstance(eval_val, dict) and isinstance(baseline_val, dict):
        # Recursively handle nested dictionaries
        return _relative_scores(eval_val, baseline_val)
    elif isinstance(eval_val, list) and isinstance(baseline_val, list):
        # For lists, compute element-wise ratios if same length
        if len(eval_val) == len(baseline_val):
            return [_compute_relative_value(e, b) for e, b in zip(eval_val, baseline_val)]
        else:
            # Different lengths, skip
            return None
    else:
        # Incompatible types, skip
        return None


def _relative_scores(eval_scores, baseline_scores):
    """Recursively compute relative scores for nested dictionaries."""
    relative_scores = {}
    
    for key in eval_scores:
        eval_val = eval_scores[key]
        baseline_val = baseline_scores.get(key)
        
        if baseline_val is None:
            # Key missing in baseline, skip
            continue
        
        relative_val = _compute_relative_value(eval_val, baseline_val)
        if relative_val is not None:
            relative_scores[key] = relative_val
    
    return relative_scores


if __name__ == "__main__":
    jsonargparse.CLI(
        [
            generate_testset_single,
            generate_testset_add,
            generate_testset_replace,
            generate_testset_everything,
            generate_testset_tuples,
            generate_responses,
            generate_responses_baselines,
            evaluate,
            evaluate_task,
            compute_eval_scores,
            compute_eval_scores_tree,
            compare_models_scores,
            compare_models_scores_by_experiment,
            plot_per_turn_accuracy_comparison,
            plot_add_replace_comparison,
            generate_heatmaps_from_reports,
        ],
        as_positional=False,
    )