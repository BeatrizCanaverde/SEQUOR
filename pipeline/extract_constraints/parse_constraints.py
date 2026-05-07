"""
parse_constraints_revised.py

Reads a JSONL file where each line contains an 'output' field with structured
constraint data (already in JSON format with turn, task, and constraints fields).
Extracts all constraints and groups them by type.

The input format is:
{
    "prompt": "...",
    "input_user_turn": "...",
    "finish_reason": "...",
    "output": "{\"turn\": 1, \"task\": \"...\", \"constraints\": [{\"constraint\": \"...\", \"type\": \"...\"}]}"
}

Usage:

python pipeline/extract_constraints/parse_constraints.py \
    --constraints-file pipeline/extract_constraints/outputs/lmsys-chat-1m_all_turns_revised_output_103398.jsonl \
    --output-dir pipeline/extract_constraints/outputs
"""

import argparse
import json
import os
import re
from pathlib import Path


def parse_output_field(output_text):
    """Parse the output field which contains structured JSON with constraints.
    
    The expected format is a list of turn objects:
    [
        {"turn": 1, "task": "", "constraints": [...]},
        {"turn": 2, "task": "...", "constraints": [...]},
        ...
    ]
    
    Returns: 
        tuple: (constraints_list, success)
        - constraints_list: list of constraint dicts (each with 'constraint' and 'type' keys)
        - success: bool indicating if parsing was successful
    """
    constraints = []
    
    try:
        # Try to parse as JSON directly
        data = json.loads(output_text)
        
        # The expected format is a list of turn objects
        if isinstance(data, list):
            for turn_data in data:
                if isinstance(turn_data, dict) and 'constraints' in turn_data:
                    constraints.extend(turn_data['constraints'])
            return constraints, True
        # Handle single turn output (for backward compatibility)
        elif isinstance(data, dict) and 'constraints' in data:
            constraints.extend(data['constraints'])
            return constraints, True
        else:
            # Unexpected format
            return [], False
            
    except json.JSONDecodeError:
        # Failed to parse JSON
        return [], False
    
    return constraints, True


def parse_file(input_path, output_path):
    constraints_by_type = {}
    total_lines = 0
    total_constraints = 0
    lines_with_constraints = 0
    malformed_jsonl_lines = 0
    incorrect_format_lines = 0

    with open(input_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # skip malformed JSONL line
                malformed_jsonl_lines += 1
                continue

            # Extract the 'output' field
            output_text = rec.get('output')
            if not output_text:
                continue

            # Parse constraints from the output
            constraints, success = parse_output_field(output_text)
            
            if not success:
                incorrect_format_lines += 1
                continue
            
            if constraints:
                lines_with_constraints += 1
            
            for constraint_dict in constraints:
                if not isinstance(constraint_dict, dict):
                    continue
                
                constraint_text = constraint_dict.get('constraint')
                constraint_type = constraint_dict.get('type', 'Unknown')
                
                if not constraint_text:
                    continue
                
                total_constraints += 1
                constraints_by_type.setdefault(constraint_type, set()).add(constraint_text)

    # Convert sets to sorted lists
    constraints_out = {k: sorted(list(v)) for k, v in constraints_by_type.items()}

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as outf:
        json.dump(constraints_out, outf, indent=2, ensure_ascii=False)

    print(f"Processed {total_lines} lines")
    print(f"Found constraints in {lines_with_constraints} lines")
    print(f"Malformed JSONL lines: {malformed_jsonl_lines}")
    print(f"Lines with incorrect format: {incorrect_format_lines}")
    print(f"Extracted {total_constraints} total constraint entries")
    print(f"Wrote {sum(len(v) for v in constraints_out.values())} unique constraints to {output_path}")
    print(f"\nConstraints by type:")
    for constraint_type, constraint_list in constraints_out.items():
        print(f"  {constraint_type}: {len(constraint_list)} unique constraints")


def main(args):
    input_path = Path(args.constraints_file)

    output_filename = "parsed_constraints.json"
    
    output_path = Path(args.output_dir) / output_filename

    parse_file(input_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse constraints extracted by LLM from conversations (revised format).")
    parser.add_argument('--constraints-file', type=str, required=True, help='Path to JSONL file with extracted constraints in revised format.' )
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output results.')
    args = parser.parse_args()
    main(args)
