import argparse
import json
import sys
import jsonargparse
from pathlib import Path


def json_to_jsonl(input_json_path: str, output_jsonl_path: str):
    """
    Read a JSON file with constraint types as keys and lists of constraints as values.
    Convert it to JSONL format where each line contains a constraint and its type.
    
    Args:
        input_json_path: Path to input JSON file
        output_jsonl_path: Path to output JSONL file
    """
    input_path = Path(input_json_path)
    output_path = Path(output_jsonl_path)
    
    # Read the JSON file
    print(f"Reading from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to JSONL format
    print(f"Writing to: {output_path}")
    constraint_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for constraint_type, constraints in data.items():
            if constraint_type == "Content Constraints":
                continue  # Skip content constraints
            for constraint in constraints:
                entry = {
                    "text": constraint,
                    "type": constraint_type
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                constraint_count += 1
    
    print(f"✓ Converted {constraint_count} constraints from {len(data)} types")
    print(f"✓ Output saved to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert JSON to JSONL format for constraint data.")
    parser.add_argument("--input_json_path", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_jsonl_path", type=str, required=True, help="Path to output JSONL file.")
    args = parser.parse_args()
    
    json_to_jsonl(input_json_path=args.input_json_path, output_jsonl_path=args.output_jsonl_path)