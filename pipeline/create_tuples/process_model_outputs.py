import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

# Regex to capture the final verdict token (Yes/No) inside double brackets.
VERDICT_REGEX = re.compile(r"Final Verdict[^\[]*\[\[\s*(Yes|No)[^\]]*\]\]", re.IGNORECASE)

# Flexible fallback regex: allows missing colon and brackets
FLEXIBLE_VERDICT_REGEX = re.compile(r"Final\s+Verdict:?\s*(Yes|No)", re.IGNORECASE)


def extract_verdict(text: str) -> Optional[str]:
    """Return the normalized verdict ('Yes' or 'No') if present, else None."""
    # First try the strict regex with brackets
    matches = VERDICT_REGEX.findall(text)
    
    # If no match, try the flexible regex (missing colon/brackets are OK)
    if not matches:
        matches = FLEXIBLE_VERDICT_REGEX.findall(text)
    
    # If still no match, return None
    if not matches:
        return None
    
    # Use the last match found
    verdict = matches[-1].strip().lower()
    return "Yes" if verdict.startswith("yes") else "No"


def process_outputs(raw_outputs: Iterable[str]) -> List[Optional[str]]:
    """Convert raw model outputs into a list of verdicts, preserving missing as None."""
    verdicts: List[Optional[str]] = []
    for output in raw_outputs:
        verdicts.append(extract_verdict(output))
    return verdicts


def process_file(input_file: Path, output_dir: Path) -> Path:
    """
    Read a JSONL file with keys {constraints, tasks, model_outputs},
    extract verdicts from model_outputs, and write a new JSONL file
    containing only {constraints, model_outputs} where model_outputs is the
    list of parsed verdicts (Yes/No/None).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_file.stem}_processed.jsonl"

    total_lines = 0
    missing_verdicts = 0

    with input_file.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            record = json.loads(line)
            constraints = record.get("constraints", [])
            raw_outputs = record.get("model_outputs", [])

            verdicts = process_outputs(raw_outputs)
            missing_verdicts += sum(1 for v in verdicts if v is None)

            processed = {
                "constraints": constraints,
                "model_outputs": verdicts,
            }
            json.dump(processed, dst, ensure_ascii=False)
            dst.write("\n")

    print(f"Processed {total_lines} lines from {input_file}")
    print(f"Missing verdicts: {missing_verdicts}")
    print(f"Wrote processed file to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract verdicts from model outputs and keep constraints only.")
    parser.add_argument("--input-file", type=Path, help="Path to the input JSONL file (e.g., create_tuples/tuples/model_name/3.jsonl)")
    parser.add_argument("--output-dir", type=Path, help="Directory where the processed JSONL will be saved")
    args = parser.parse_args()
    process_file(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
