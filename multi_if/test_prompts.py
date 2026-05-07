from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Generator, TypedDict

import numpy as np

from constraints import Constraint, ConstraintData, constraint_to_data


Task = str


class StepPlan(TypedDict):
    """Plan for a single step in the conversation."""
    step_size: int
    introduce_new: bool


class ExperimentPlan(TypedDict):
    """Complete plan for constraint introductions in a conversation."""
    start_with_tuple: bool
    initial_num_constraints: int
    steps: list[StepPlan]


class ConstraintIntroductionPlan(TypedDict):
    """Plan for how to introduce constraints at a specific turn."""
    is_continuation: bool  # True if continuing to introduce from pending tuple
    use_tuple: bool  # Only relevant if not continuation - whether to sample tuple or single
    num_to_introduce: int  # How many constraints to introduce this turn


class DetailedStepPlan(TypedDict):
    """Detailed plan for a single step in the conversation."""
    step_size: int
    introduce_new: bool
    introduction_plan: ConstraintIntroductionPlan | None  # Only present if introduce_new=True


class DetailedExperimentPlan(TypedDict):
    """Complete detailed plan for constraint introductions that can be replayed consistently."""
    use_tuple_initial: bool
    num_initial: int  # How many constraints to introduce initially (1 for single, 1-N for tuple)
    steps: list[DetailedStepPlan]


def generate_experiment_plan(
    min_step_size: int,
    max_step_size: int,
    num_prompts: int,
    max_constraint_changes: int = 20,
) -> ExperimentPlan:
    """Generate a deterministic plan for WHEN constraints are introduced (but not what/how)."""
    
    # Decide initial setup
    start_with_tuple = random.choice([True, False])
    initial_num_constraints = random.randint(1, 3)  # Will be adjusted in main function
    
    steps: list[StepPlan] = []
    current_idx = 0
    constraint_changes = 0
    introduce_new = False
    
    while current_idx < num_prompts:
        step_size = random.randint(min_step_size, max_step_size)
        actual_step = min(step_size, num_prompts - current_idx)
        
        if introduce_new and constraint_changes < max_constraint_changes:
            steps.append(StepPlan(
                step_size=actual_step,
                introduce_new=True,
            ))
            constraint_changes += 1
            introduce_new = False
        else:
            steps.append(StepPlan(
                step_size=actual_step,
                introduce_new=False,
            ))
            introduce_new = True
        
        current_idx += actual_step
    
    return ExperimentPlan(
        start_with_tuple=start_with_tuple,
        initial_num_constraints=initial_num_constraints,
        steps=steps,
    )


def generate_detailed_experiment_plan(
    min_step_size: int,
    max_step_size: int,
    num_prompts: int,
    max_constraint_changes: int = 20,
    max_tuple_size: int = 3,
) -> DetailedExperimentPlan:
    """
    Generate a detailed, deterministic plan that captures all random decisions
    for constraint introductions. This plan can be replayed with different
    constraints/tasks but the same structure.
    """
    
    # Decide initial setup: tuple or single?
    use_tuple_initial = random.random() < 0.5
    if use_tuple_initial:
        # If tuple, decide how many to introduce initially (1 to max_tuple_size)
        num_initial = random.randint(1, max_tuple_size)
    else:
        num_initial = 1  # Single constraint
    
    steps: list[DetailedStepPlan] = []
    current_idx = 0
    constraint_changes = 0
    introduce_new = False
    pending_count = 0  # Track how many constraints are pending from a tuple
    
    # If we started with a tuple and didn't introduce all, we have pending
    if use_tuple_initial and num_initial < max_tuple_size:
        # Assume we might have pending (will be determined by actual tuple size)
        pending_count = max_tuple_size - num_initial
    
    while current_idx < num_prompts:
        step_size = random.randint(min_step_size, max_step_size)
        actual_step = min(step_size, num_prompts - current_idx)
        
        introduction_plan = None
        if introduce_new and constraint_changes < max_constraint_changes:
            # Decide what to introduce at this turn
            if pending_count > 0:
                # Continue introducing from pending tuple
                num_to_introduce = random.randint(1, pending_count)
                pending_count -= num_to_introduce
                introduction_plan = ConstraintIntroductionPlan(
                    is_continuation=True,
                    use_tuple=False,  # Not relevant for continuation
                    num_to_introduce=num_to_introduce,
                )
            else:
                # Start new: decide single or tuple
                use_tuple = random.random() < 0.5
                if use_tuple:
                    # Decide how many to introduce from the tuple
                    num_to_introduce = random.randint(1, max_tuple_size)
                    # The rest become pending
                    pending_count = max(0, max_tuple_size - num_to_introduce)
                else:
                    num_to_introduce = 1  # Single constraint
                    pending_count = 0
                
                introduction_plan = ConstraintIntroductionPlan(
                    is_continuation=False,
                    use_tuple=use_tuple,
                    num_to_introduce=num_to_introduce,
                )
            
            steps.append(DetailedStepPlan(
                step_size=actual_step,
                introduce_new=True,
                introduction_plan=introduction_plan,
            ))
            constraint_changes += 1
            introduce_new = False
        else:
            steps.append(DetailedStepPlan(
                step_size=actual_step,
                introduce_new=False,
                introduction_plan=None,
            ))
            introduce_new = True
        
        current_idx += actual_step
    
    return DetailedExperimentPlan(
        use_tuple_initial=use_tuple_initial,
        num_initial=num_initial,
        steps=steps,
    )


def read_tasks(path: Path) -> list[Task]:
    tasks = []
    with path.open("r") as f:
        for line in f:
            data = json.loads(line)
            tasks.append(data["prompt"])
    return tasks


class TestPrompt(TypedDict):
    """Prompt in a test conversation with the current constraints."""

    prompt: str
    constraints_data: list[ConstraintData]


TestPrompts = list[TestPrompt]


def generate_test_prompts_single(
    constraint: Constraint,
    tasks: list[Task],
    num_prompts: int,
    replace: bool,
) -> TestPrompts:
    
    prompt_text, constraints_data, _ = get_prompt_and_constraint_data(
        mode="start", constraints=[constraint], constraints_data=[]
    )

    selected = np.random.default_rng().choice(tasks, size=num_prompts, replace=replace).tolist()
    tasks = [t for t in tasks if t in selected]
    tasks[0] = f"{prompt_text}\n\n\n{tasks[0]}"

    return [
        TestPrompt(prompt=task, constraints_data=constraints_data.copy())
        for task in tasks
    ]


def generate_test_prompts_tuples(
    constraint: list[Constraint],
    tasks: list[Task],
    num_prompts: int,
    replace: bool,
) -> TestPrompts:

    prompt_text, constraints_data, _ = get_prompt_and_constraint_data(
        mode="tuples", constraints=[constraint], constraints_data=[]
    )

    _num_prompts = len(tasks) if num_prompts is None else num_prompts

    selected = np.random.default_rng().choice(tasks, size=_num_prompts, replace=replace).tolist()
    chosen_tasks = [t for t in tasks if t in selected]
    if not chosen_tasks:
        raise ValueError("No tasks selected; check num_prompts and replace settings")

    chosen_tasks[0] = f"{prompt_text}\n\n\n{chosen_tasks[0]}"

    prompts = [
        TestPrompt(prompt=task, constraints_data=constraints_data.copy())
        for task in chosen_tasks
    ]
    
    return prompts


def generate_test_prompts_replace_add(
    constraints: list[Constraint],
    tasks: list[Task],
    experiment: str,
    step_size: int,
    min_constraints: int,
    num_prompts: int,
    replace: bool,
) -> TestPrompts:
    if not constraints:
        return []

    constraints = constraints.copy()
    random.shuffle(constraints)

    if experiment == "add":
        constraints = constraints[: max(min_constraints, 1)]

    prompt_text, constraints_data, last_template = get_prompt_and_constraint_data(
        mode="start", constraints=[constraints[0]], constraints_data=[]
    )

    selected = np.random.default_rng().choice(tasks, size=num_prompts, replace=replace).tolist()
    tasks = [t for t in tasks if t in selected]

    start_idx = 0
    end_idx = 0
    constraint_idx = 1
    conversation_prompts: TestPrompts = []

    while end_idx < num_prompts:
        if step_size is None:
            step_size = random.randint(1, num_prompts)

        end_idx = min(end_idx + step_size, num_prompts)
        chunk = tasks[start_idx:end_idx]
        if not chunk:
            break

        # Track if we introduce a new constraint in this iteration
        constraint_introduced = False
        if start_idx > 0 and constraint_idx < len(constraints):
            prompt_text, constraints_data, last_template = get_prompt_and_constraint_data(
                mode=experiment,
                constraints=[constraints[constraint_idx]],
                constraints_data=constraints_data,
                last_template=last_template,
            )
            constraint_idx += 1
            constraint_introduced = True

        # Only prepend constraint text if it's the first chunk or a new constraint was introduced
        if start_idx == 0 or constraint_introduced:
            chunk[0] = f"{prompt_text}\n\n\n{chunk[0]}"

        conversation_prompts.extend(
            TestPrompt(prompt=task, constraints_data=constraints_data.copy())
            for task in chunk
        )

        start_idx = end_idx

    return conversation_prompts


def generate_test_prompts_everything(
    single_constraints: list[Constraint],
    tuple_constraints: list[list[Constraint]],
    tasks: list[Task],
    min_step_size: int,
    max_step_size: int,
    num_prompts: int,
    replace: bool,
    experiment_plan: ExperimentPlan | DetailedExperimentPlan | None = None,
) -> TestPrompts:
    """
    Generate test prompts with dynamic constraint introduction following a detailed plan.
    
    If a DetailedExperimentPlan is provided, it will be followed exactly to ensure
    consistent structure across repetitions (with different constraints/tasks).
    If None or ExperimentPlan, a new detailed plan is generated.
    """
    if not single_constraints and not tuple_constraints:
        return []

    # Check if we have a detailed plan or need to generate one
    if experiment_plan is None or "steps" not in experiment_plan or (
        experiment_plan["steps"] and "introduction_plan" not in experiment_plan["steps"][0]
    ):
        # Generate new detailed plan
        detailed_plan = generate_detailed_experiment_plan(
            min_step_size=min_step_size,
            max_step_size=max_step_size,
            num_prompts=num_prompts,
            max_tuple_size=3,  # Match constraints/everything/3.jsonl
        )
    else:
        # Use provided detailed plan
        detailed_plan = experiment_plan

    # Prepare initial constraint(s) following the plan
    use_tuple_initial = detailed_plan["use_tuple_initial"]
    num_initial = detailed_plan["num_initial"]
    
    if use_tuple_initial and tuple_constraints:
        # Start with a tuple
        initial_tuple = random.choice(tuple_constraints)
        # Introduce num_initial constraints from it
        num_to_use = min(num_initial, len(initial_tuple))
        initial_constraints_list = initial_tuple[:num_to_use]
        pending_constraints = initial_tuple[num_to_use:] if num_to_use < len(initial_tuple) else []
        
        if len(initial_constraints_list) == 1:
            start_mode = "start"
            initial_constraints = initial_constraints_list[0]
        else:
            start_mode = "tuples"
            initial_constraints = initial_constraints_list
        
        # Track the tuple we're introducing
        current_tuple_id = tuple([c.id for c in initial_tuple])
        active_constraint_ids = {c.id for c in initial_constraints_list}
    else:
        # Start with a single constraint
        initial_constraint = random.choice(single_constraints)
        initial_constraints = initial_constraint
        start_mode = "start"
        active_constraint_ids = {initial_constraint.id}
        pending_constraints = []
        current_tuple_id = None

    prompt_text, constraints_data, last_template = get_prompt_and_constraint_data(
        mode=start_mode, 
        constraints=[initial_constraints] if start_mode == "start" else [initial_constraints],
        constraints_data=[]
    )

    selected = np.random.default_rng().choice(tasks, size=num_prompts, replace=replace).tolist()
    tasks = [t for t in tasks if t in selected]

    start_idx = 0
    conversation_prompts: TestPrompts = []

    for step_plan in detailed_plan["steps"]:
        constraint_introduced = False
        
        # Check if we need to introduce new constraints
        if step_plan["introduce_new"]:
            intro_plan = step_plan["introduction_plan"]
            
            if intro_plan["is_continuation"]:
                # Continue introducing from pending tuple
                if pending_constraints:
                    num_to_introduce = min(intro_plan["num_to_introduce"], len(pending_constraints))
                    constraints_to_introduce = pending_constraints[:num_to_introduce]
                    pending_constraints = pending_constraints[num_to_introduce:]
                    
                    # Always use "add" mode when continuing a tuple
                    active_constraint_ids.update(c.id for c in constraints_to_introduce)
                    
                    # Generate prompt with randomly selected template
                    if len(constraints_to_introduce) == 1:
                        prompt_text, constraints_data, last_template = get_prompt_and_constraint_data(
                            mode="add",
                            constraints=constraints_to_introduce,
                            constraints_data=constraints_data,
                            last_template=None,
                        )
                    else:
                        prompt_text, constraints_data, last_template = get_prompt_and_constraint_data(
                            mode="add_tuples",
                            constraints=[constraints_to_introduce],
                            constraints_data=constraints_data,
                            last_template=None,
                        )
                    
                    constraint_introduced = True
            
            else:
                # Start new constraint/tuple - always use REPLACE mode
                use_tuple = intro_plan["use_tuple"]
                
                if use_tuple and tuple_constraints:
                    # Sample a tuple different from current one
                    available_tuples = [
                        t for t in tuple_constraints
                        if tuple([c.id for c in t]) != current_tuple_id
                    ]
                    
                    # If no different tuples available, allow repeating from full pool
                    if not available_tuples:
                        available_tuples = tuple_constraints
                    
                    sampled_tuple = random.choice(available_tuples)
                    current_tuple_id = tuple([c.id for c in sampled_tuple])
                    
                    # Introduce the number specified in the plan
                    num_to_introduce = min(intro_plan["num_to_introduce"], len(sampled_tuple))
                    constraints_to_introduce = sampled_tuple[:num_to_introduce]
                    pending_constraints = sampled_tuple[num_to_introduce:] if num_to_introduce < len(sampled_tuple) else []
                    
                    # ALWAYS use "replace" when starting new
                    active_constraint_ids = {c.id for c in constraints_to_introduce}
                    
                    # Generate prompt
                    if len(constraints_to_introduce) == 1:
                        prompt_text, constraints_data, last_template = get_prompt_and_constraint_data(
                            mode="replace",
                            constraints=constraints_to_introduce,
                            constraints_data=[],
                            last_template=None,
                        )
                    else:
                        prompt_text, constraints_data, last_template = get_prompt_and_constraint_data(
                            mode="replace_tuples",
                            constraints=[constraints_to_introduce],
                            constraints_data=[],
                            last_template=None,
                        )
                    
                    constraint_introduced = True
                
                elif single_constraints:
                    # Sample a single constraint, avoiding currently active ones
                    available_singles = [
                        c for c in single_constraints
                        if c.id not in active_constraint_ids
                    ]
                    
                    # If no different singles available, allow repeating from full pool
                    if not available_singles:
                        available_singles = single_constraints
                    
                    sampled_constraint = random.choice(available_singles)
                    current_tuple_id = None
                    
                    # ALWAYS use "replace" when starting new
                    active_constraint_ids = {sampled_constraint.id}
                    
                    # Generate prompt
                    prompt_text, constraints_data, last_template = get_prompt_and_constraint_data(
                        mode="replace",
                        constraints=[sampled_constraint],
                        constraints_data=[],
                        last_template=None,
                    )
                    
                    constraint_introduced = True
        
        # Get this step's chunk of tasks
        end_idx = min(start_idx + step_plan["step_size"], len(tasks))
        chunk = tasks[start_idx:end_idx]
        if not chunk:
            break

        # Only prepend constraint text if it's the first chunk or a constraint was introduced
        if start_idx == 0 or constraint_introduced:
            chunk[0] = f"{prompt_text}\n\n\n{chunk[0]}"

        conversation_prompts.extend(
            TestPrompt(prompt=task, constraints_data=constraints_data.copy())
            for task in chunk
        )

        start_idx = end_idx

    return conversation_prompts


def get_prompt_and_constraint_data(
    *, mode: str, constraints: list[Constraint], constraints_data: list[ConstraintData],
    last_template: str | None = None
) -> tuple[str, list[ConstraintData], str]:
    """Generate constraint prompt text. Returns (prompt_text, constraints_data, selected_template)."""

    if mode == "start":
        base_prompt = constraints[0].text
        start_templates = [
            "Throughout the following conversation, always follow this constraint: {prompt}",
            "In all your responses, make sure to adhere to this rule: {prompt}",
            "For the duration of this chat, follow this constraint: {prompt}",
            "During this conversation, ensure you follow this directive: {prompt}",
            "As we talk, always comply with this constraint: {prompt}",
            "In every reply, abide by this rule: {prompt}",
        ]
        # Avoid picking the same template as last time
        available = [t for t in start_templates if t != last_template] if last_template else start_templates
        template = random.choice(available if available else start_templates)
        prompt = template.format(prompt=base_prompt)
        constraints_data = [constraint_to_data(constraints[0])]
        return prompt, constraints_data, template

    elif mode == "tuples":
        tuples_templates = [
            "Throughout the following conversation, always follow these constraints:\n",
            "In all your responses, make sure to adhere to these rules:\n",
            "For the duration of this chat, follow these constraints:\n",
            "During this conversation, ensure you follow these directives:\n",
            "As we talk, always comply with these constraints:\n",
            "In every reply, abide by these rules:\n",
        ]
        # Avoid picking the same template as last time
        available = [t for t in tuples_templates if t != last_template] if last_template else tuples_templates
        template = random.choice(available if available else tuples_templates)
        prompt_lines = [f"{c_idx + 1}. {c.text}" for c_idx, c in enumerate(constraints[0])]
        prompt_text = template + "\n".join(prompt_lines)
        constraints_data = [constraint_to_data(c) for c in constraints[0]]
        return prompt_text, constraints_data, template

    elif mode == "replace":
        base_prompt = constraints[0].text
        replace_templates = [
            "Forget all constraints provided earlier. From now on, follow only this one: {prompt}",
            "Disregard previous constraints. The only rule to follow from here on is: {prompt}",
            "Erase earlier directives. The new and sole constraint for the following turns is: {prompt}",
            "Cancel all past guidelines. The only constraint to adhere from now on is: {prompt}",
            "Forget prior constraints. From here on, the only rule is: {prompt}",
            "Override earlier constraints. In the next turns, follow only this one instead: {prompt}",
        ]
        available = [t for t in replace_templates if t != last_template] if last_template else replace_templates
        template = random.choice(available if available else replace_templates)
        prompt = template.format(prompt=base_prompt)
        constraints_data = [constraint_to_data(constraints[0])]
        return prompt, constraints_data, template

    elif mode == "replace_tuples":
        replace_tuples_templates = [
            "Forget all constraints provided earlier. From now on, follow only these ones:\n",
            "Disregard previous constraints. The only rules to follow from here on are:\n",
            "Erase earlier directives. The new and sole constraints for the following turns are:\n",
            "Cancel all past guidelines. The only constraints to adhere from now on are:\n",
            "Forget prior constraints. From here on, the only rules are:\n",
            "Override earlier constraints. In the next turns, follow only these ones instead:\n",
        ]
        available = [t for t in replace_tuples_templates if t != last_template] if last_template else replace_tuples_templates
        template = random.choice(available if available else replace_tuples_templates)
        prompt_lines = [f"{c_idx + 1}. {c.text}" for c_idx, c in enumerate(constraints[0])]
        prompt_text = template + "\n".join(prompt_lines)
        constraints_data = [constraint_to_data(c) for c in constraints[0]]
        return prompt_text, constraints_data, template

    elif mode == "add":
        base_prompt = constraints[0].text
        add_templates = [
            "In addition to the previous constraints, also follow this one from now on: {prompt}",
            "Along with the earlier directives, from here on also follow this new constraint: {prompt}",
            "Do not forget the existing rules; in the next turns follow also this new one: {prompt}",
            "Building on the earlier constraints, adhere to this as well in the following turns: {prompt}",
            "Keep in mind the previous constraints and, in addition, follow this new one from here on: {prompt}",
        ]
        available = [t for t in add_templates if t != last_template] if last_template else add_templates
        template = random.choice(available if available else add_templates)
        prompt = template.format(prompt=base_prompt)
        constraints_data = constraints_data + [constraint_to_data(constraints[0])]
        return prompt, constraints_data, template

    elif mode == "add_tuples":
        add_tuples_templates = [
            "In addition to the previous constraints, also follow these ones from now on:\n",
            "Along with the earlier directives, from here on also follow these new constraints:\n",
            "Do not forget the existing rules; in the next turns follow also these new ones:\n",
            "Building on the earlier constraints, adhere to these as well in the following turns:\n",
            "Keep in mind the previous constraints and, in addition, follow these new ones from here on:\n",
        ]
        available = [t for t in add_tuples_templates if t != last_template] if last_template else add_tuples_templates
        template = random.choice(available if available else add_tuples_templates)
        prompt_lines = [f"{c_idx + 1}. {c.text}" for c_idx, c in enumerate(constraints[0])]
        prompt_text = template + "\n".join(prompt_lines)
        constraints_data = constraints_data + [constraint_to_data(c) for c in constraints[0]]
        return prompt_text, constraints_data, template

    raise ValueError(f"Unknown mode: {mode}")


def load_testset(dirname: Path) -> Generator[tuple[Path, TestPrompts]]:
    for file in dirname.rglob("*.jsonl"):
        yield file, load_test_prompts(file)


def load_test_prompts(filename: Path) -> TestPrompts:
    prompts: TestPrompts = []
    with filename.open("r") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data)
    return prompts


def save_test_prompts(filename: Path, prompts: TestPrompts) -> None:
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    with filename.open("w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")