# All templates used in the original prompts - for matching/stripping
ALL_SINGLE_TEMPLATES = [
    # start templates
    "Throughout the following conversation, always follow this constraint: ",
    "In all your responses, make sure to adhere to this rule: ",
    "For the duration of this chat, follow this constraint: ",
    "During this conversation, ensure you follow this directive: ",
    "As we talk, always comply with this constraint: ",
    "In every reply, abide by this rule: ",    
    # replace templates
    "Forget all constraints provided earlier. From now on, follow only this one: ",
    "Disregard previous constraints. The only rule to follow from here on is: ",
    "Erase earlier directives. The new and sole constraint for the following turns is: ",
    "Cancel all past guidelines. The only constraint to adhere from now on is: ",
    "Forget prior constraints. From here on, the only rule is: ",
    "Override earlier constraints. In the next turns, follow only this one instead: ",
    # add templates
    "In addition to the previous constraints, also follow this one from now on: ",
    "Along with the earlier directives, from here on also follow this new constraint: ",
    "Do not forget the existing rules; in the next turns follow also this new one: ",
    "Building on the earlier constraints, adhere to this as well in the following turns: ",
    "Keep in mind the previous constraints and, in addition, follow this new one from here on: ",
]

ALL_TUPLES_TEMPLATES = [
    # tuples templates
    "Throughout the following conversation, always follow these constraints:\n",
    "In all your responses, make sure to adhere to these rules:\n",
    "For the duration of this chat, follow these constraints:\n",
    "During this conversation, ensure you follow these directives:\n",
    "As we talk, always comply with these constraints:\n",
    "In every reply, abide by these rules:\n",    
    # replace_tuples templates
    "Forget all constraints provided earlier. From now on, follow only these ones:\n",
    "Disregard previous constraints. The only rules to follow from here on are:\n",
    "Erase earlier directives. The new and sole constraints for the following turns are:\n",
    "Cancel all past guidelines. The only constraints to adhere from now on are:\n",
    "Forget prior constraints. From here on, the only rules are:\n",
    "Override earlier constraints. In the next turns, follow only these ones instead:\n",
    # add_tuples templates
    "In addition to the previous constraints, also follow these ones from now on:\n",
    "Along with the earlier directives, from here on also follow these new constraints:\n",
    "Do not forget the existing rules; in the next turns follow also these new ones:\n",
    "Building on the earlier constraints, adhere to these as well in the following turns:\n",
    "Keep in mind the previous constraints and, in addition, follow these new ones from here on:\n",
]

# NEW baseline templates - TODO: Replace these with your new templates
BASELINE_SINGLE_TEMPLATES = [
    "Follow this constraint: ",
    "Make sure to adhere to this rule: ",
    "Ensure you follow this directive: ",
    "Comply with this constraint: ",
    "Abide by this rule: ",
] 

BASELINE_TUPLES_TEMPLATES = [
    "Follow these constraints:\n",
    "Make sure to adhere to these rules:\n",
    "Ensure you follow these directives:\n",
    "Comply with these constraints:\n",
    "Abide by these rules:\n",
]

def process_prompt(prompt):
    """
    If prompt starts with any known constraint template, remove everything up to and including the first '\n\n\n'.
    Tries both single and tuples templates.
    Returns (task, has_constraint_template).
    """
    # Try single templates
    for s in ALL_SINGLE_TEMPLATES:
        if prompt.startswith(s):
            idx = prompt.find('\n\n\n')
            if idx != -1:
                return prompt[idx+3:], True
    
    # Try tuples templates
    for s in ALL_TUPLES_TEMPLATES:
        if prompt.startswith(s):
            idx = prompt.find('\n\n\n')
            if idx != -1:
                return prompt[idx+3:], True
    
    return prompt, False

