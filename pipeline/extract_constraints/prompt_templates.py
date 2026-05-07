one_turn_template = '''
Identify user prompts with constraints.

A constraint is a restriction or condition that limits how the model should generate its output, rather than what task it performs. It guides the form, style, content, or structure of the response — ensuring it adheres to specific requirements or rules. 

We classify constraints into six main categories: 
1) Content Constraints: These define the specific topics or details to be addressed in the LLM's response, such as discussing "climate change impacts" or mentioning "the main characters in Harry Potter";
2) Linguistic Guidelines: These dictate the use of particular language structures and terms, including grammatical styles, syntax, and specific dialects, like "Victorian English" or "technical jargon"; 
3) Style Rules: These direct the overall tone and audience of the text, varying from formal to persuasive or sophisticated, as in writing with a "respectful tone" or for "a young audience"; 
4) Format Specifications: These instruct the LLM on the structural presentation of its response, such as "crafting a sonnet" or "list ideas bullet-wise"; 
5) Number Limitations: These involve numeric-related instructions, like producing "a 500-word essay" or presenting "three arguments for renewable energy".
6) Other Specific Constraints: Any other restrictions that do not fall into the above categories but still limit how the model should generate its output.

Below, you are given a user prompt taken from a conversation. Your job is to identify all constraints in the user prompt and classify them into their categories. Each constraint should be classified into only one of the six categories listed above.

You can first reason about the user prompt and its context. At the end, present your final answer as a list of dictionaries where each dictionary contains the constraint and the constraint type.


For example:

User prompt:
"I want to write an email to my boss about the crazy amount of meetings he's scheduling. 
Write me a formal email of 300 words to explain the situation and ask for him to be more understanding on the number of meetings he schedules."

Output:
{{"constraint": "write in formal tone", "type": "Style Rules"}}
{{"constraint": "write in 300 words", "type": "Number Limitations"}}
{{"constraint": "discuss the crazy amount of meetings", "type": "Content Constraints"}}
{{"constraint": "ask for more understanding on the number of meetings", "type": "Content Constraints"}}


User prompt:
"{user_turn}"

Output:

'''




all_turns_template = '''
Identify user prompts with constraints.

A constraint is a restriction or condition that limits how the model should generate its output, rather than what task it performs. It guides the form, style, content, or structure of the response — ensuring it adheres to specific requirements or rules. 

We classify constraints into six main categories: 
1) Content Constraints: These define the specific topics or details to be addressed in the LLM's response, such as discussing "climate change impacts" or mentioning "the main characters in Harry Potter";
2) Linguistic Guidelines: These dictate the use of particular language structures and terms, including grammatical styles, syntax, and specific dialects, like "Victorian English" or "technical jargon"; 
3) Style Rules: These direct the overall tone and audience of the text, varying from formal to persuasive or sophisticated, as in writing with a "respectful tone" or for "a young audience"; 
4) Format Specifications: These instruct the LLM on the structural presentation of its response, such as "crafting a sonnet" or "list ideas bullet-wise"; 
5) Number Limitations: These involve numeric-related instructions, like producing "a 500-word essay" or presenting "three arguments for renewable energy".
6) Other Specific Constraints: Any other restrictions that do not fall into the above categories but still limit how the model should generate its output.

Below, you are given a sequence of user prompts taken from a conversation. Your job is to identify all constraints in the user prompts and classify them into their categories. Each constraint should be classified into only one of the six categories listed above.

You can first reason about the user prompts and their context. At the end, present your final answer as a list of dictionaries where each dictionary contains the constraint, the constraint type, and the respective turn number.


For example:

User prompt:
"Turn 1:
I want to write an email to my boss about the crazy amount of meetings he's scheduling. 
Write me a formal email of 300 words to explain the situation and ask for him to be more understanding on the number of meetings he schedules.

Turn 2:
Could you make sure to include some suggestions in a bullet-point list on how to manage meetings better?"

Turn 3:
Rewrite the email in a more polite tone."

Output:
{{"constraint": "write in formal tone", "type": "Style Rules", "turn": 1}}
{{"constraint": "write in 300 words", "type": "Number Limitations", "turn": 1}}
{{"constraint": "discuss the crazy amount of meetings", "type": "Content Constraints", "turn": 1}}
{{"constraint": "ask for more understanding on the number of meetings", "type": "Content Constraints", "turn": 1}}
{{"constraint": "include suggestions on how to manage meetings better", "type": "Content Constraints", "turn": 2}}
{{"constraint": "write a bullet-point list", "type": "Format Specifications", "turn": 2}}
{{"constraint": "rewrite in a more polite tone", "type": "Style Rules", "turn": 3}}


User prompts:
"{user_turn}"

Output:

'''




all_turns_revised_template = '''
Identify tasks and constraints in user prompts.

A task is a directive that specifies an action or goal. It tells a model to provide information or do something, such as "What is the capital of France?", "Summarize this text", "Translate this sentence", "Who are you?", "Generate a list of ideas", or "Why is the sky blue?".

A user prompt might contain no task! It can simply be a statement or expression, without any specific request for information or action. Examples of such user prompts include greetings ("Hello!"), expressions of emotion ("I'm feeling great today."), or sharing information ("I went to the park yesterday.").

A constraint is a restriction or condition that limits how the model should generate its output, rather than what task it performs. It guides the form, style, or structure of the response — ensuring it adheres to specific requirements or rules. 

We classify constraints into four main categories: 
1) Linguistic Guidelines: These dictate the use of particular language structures and terms, including grammatical styles, syntax, and specific dialects, like "Victorian English" or "technical jargon"; 
2) Style Rules: These direct the overall tone and audience of the text, varying from formal to persuasive or sophisticated, as in writing with a "respectful tone" or for "a young audience"; 
3) Format Specifications: These instruct the LLM on the structural presentation of its response, such as "write your answer as a sonnet" or "list ideas bullet-wise"; 
4) Number Limitations: These involve numeric-related instructions, like producing "a 500-word essay" or presenting "three arguments for your answer".

Below, you are given a sequence of user prompts taken from a conversation. Your job is to identify all tasks and constraints in the user prompts. In addition, classify all constraints into their categories. Each constraint should be classified into one and only one of the four categories listed above.

You can first reason about the user prompts and their context. At the end, present your final answer as a valid json output, ie, as a list of dictionaries where each dictionary contains the turn number, the task defined in the user turn (if any, otherwise ""),and a list of dictionaries for the constraints found (if any, otherwise []), where each constraint dictionary contains the constraint and the constraint type.


For example:

User prompts:
"Turn 1:
Hello!

Turn 2:
I want to write an email to my boss about the crazy amount of meetings he's scheduling. 
Write me a formal email of 300 words to explain the situation and ask for him to be more understanding on the number of meetings he schedules.

Turn 3:
Could you make sure to include some suggestions in a bullet-point list on how to manage meetings better?"

Turn 4:
Rewrite the email in a more polite tone."

Output:
[
    {{
        "turn": 1,
        "task": "",
        "constraints": []
    }},
    {{
        "turn": 2,
        "task": "I want to write an email to my boss about the crazy amount of meetings he's scheduling. Write me an email to explain the situation and ask for him to be more understanding on the number of meetings he schedules.",
        "constraints": [
            {{
                "constraint": "Write in formal tone",
                "type": "Style Rules"
            }},
            {{
                "constraint": "Write in 300 words",
                "type": "Number Limitations"
            }}
        ]
    }},
    {{
        "turn": 3,
        "task": "Make sure to include some suggestions on how to manage meetings better.",
        "constraints": [
            {{
                "constraint": "Write a bullet-point list",
                "type": "Format Specifications"
            }}
        ]
    }},
    {{
        "turn": 4,
        "task": "Rewrite the email.",
        "constraints": [
            {{
                "constraint": "Write in a more polite tone",
                "type": "Style Rules"
            }}
        ]
    }}
]


User prompts:
"{user_turn}"

Output:
'''

