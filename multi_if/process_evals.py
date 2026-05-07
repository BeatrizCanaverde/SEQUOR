#!/usr/bin/env python3
"""Process evaluation judge files to add a `followed` field.

- Reads from local/evals
- Writes to local/processed_evals (mirrors directory layout)
- Skips copying log and shell script files
- For each JSON/JSONL judge file, adds `followed` to every
  constraint_evaluation entry based on `Final Verdict: [[Yes]]`/`[[No]]`
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable, Tuple
from collections import defaultdict
import argparse

SKIP_EXTENSIONS = {".log", ".sh", ".err", ".out"}
JSON_EXTENSIONS = {".json", ".jsonl"}

# folder -> {'yes': int, 'no': int, 'none': int, 'empty': int}
VERDICT_COUNTS: dict[str, dict[str, int]] = defaultdict(lambda: {"yes": 0, "no": 0, "none": 0, "empty": 0})


def verdict_from_text(text: str) -> bool | None:
    """Return True/False if a Final Verdict is found, else None.
    
    Tolerates variant formats like '[Yes]', '[[ Yes ]]', or 'Final Verdict: Yes'.
    Uses the last match in the text if multiple are found.
    """
    # 1. Try strict matching first.
    if "Final Verdict: [[Yes]]" in text:
        return True
    if "Final Verdict: [[No]]" in text:
        return False

    # 2. Look for patterns with optional brackets and optional 'Final Verdict:' prefix.
    # Handle both single and double brackets with potential spaces.
    pattern = r"\[*(?:Final Verdict:\s*)?\[* *(Yes|No) *\]*\]*"
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    last_verdict = None
    for match in matches:
        last_verdict = match.group(1).lower() == "yes"
        
    return last_verdict


def load_records(path: Path) -> Tuple[Iterable[dict], bool]:
    """Load a JSON/JSONL file.

    Returns (records, is_lines). `records` is iterable of dicts; `is_lines`
    indicates whether the file was line-delimited.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # If it's a single dict, wrap in list for uniform handling
        if isinstance(data, dict):
            return [data], False
        if isinstance(data, list):
            return data, False
        raise ValueError(f"Unexpected JSON type in {path}: {type(data)}")
    except json.JSONDecodeError:
        # Fallback: jsonl (one object per line)
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records, True


def save_records(path: Path, records: Iterable[dict], is_lines: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_lines:
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
    else:
        recs = list(records)
        obj: dict | list
        if len(recs) == 1:
            obj = recs[0]
        else:
            obj = recs
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")


def process_record(record: dict) -> dict:
    ce_list = record.get("constraint_evaluations")
    if not isinstance(ce_list, list):
        return record
    for ce in ce_list:
        if not isinstance(ce, dict):
            continue
        # Prefer judge_content (final response) over judge_response_raw (which includes thinking)
        # This ensures verdict extraction works correctly for models with thinking traces
        text = ce.get("judge_content") or ce.get("judge_response_raw")
        if not isinstance(text, str):
            ce["followed"] = None
            continue
        verdict = verdict_from_text(text)
        ce["followed"] = verdict
    return record


def process_file(src: Path, dst: Path, folder_key: str) -> None:
    records, is_lines = load_records(src)
    processed = [process_record(dict(rec)) for rec in records]

    # Tally verdicts per folder
    for rec in processed:
        ce_list = rec.get("constraint_evaluations")
        if not isinstance(ce_list, list):
            continue
        for ce in ce_list:
            if not isinstance(ce, dict):
                continue
            # Check if the judge response is present but empty/whitespace-only
            # Use judge_content if available, otherwise fall back to judge_response_raw
            text = ce.get("judge_content") or ce.get("judge_response_raw")
            if isinstance(text, str) and text.strip() == "":
                VERDICT_COUNTS[folder_key]["empty"] += 1
                continue
            v = ce.get("followed")
            if v is True:
                VERDICT_COUNTS[folder_key]["yes"] += 1
            elif v is False:
                VERDICT_COUNTS[folder_key]["no"] += 1
            else:
                VERDICT_COUNTS[folder_key]["none"] += 1

    save_records(dst, processed, is_lines)


def mirror_tree(src_root: Path, dst_root: Path) -> None:
    for src_path in src_root.rglob("*"):
        if src_path.is_dir():
            # Directories created lazily when writing files
            continue
        if "logs" in src_path.parts:
            #print(f"Skipping logs directory: {src_path}")
            continue
        elif "config.json" in src_path.parts:
            continue
        if src_path.suffix in SKIP_EXTENSIONS:
            continue
        rel = src_path.relative_to(src_root)
        parts = list(rel.parts)
        # compute folder key ignoring any 'judge' component
        parts_no_judge = [p for p in parts if p != "judge"]
        folder_key = parts_no_judge[0] if parts_no_judge else "."
        if "judge" in rel.parts:
            parts.remove("judge")
            rel = Path(*parts)
        dst_path = dst_root / rel
        if src_path.suffix in JSON_EXTENSIONS:
            process_file(src_path, dst_path, folder_key)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process evaluation judge files to add a `followed` field.")
    parser.add_argument("--src-root", type=Path, help="Source root directory (default: local/evals)")
    parser.add_argument("--dst-root", type=Path, help="Destination root directory (default: local/processed_evals)")
    args = parser.parse_args()

    mirror_tree(args.src_root, args.dst_root)
    # Print per-folder verdict summary
    if VERDICT_COUNTS:
        print("Judge verdict summary (per folder):")
        for folder in sorted(VERDICT_COUNTS.keys()):
            counts = VERDICT_COUNTS[folder]
            print(f"{folder}: Yes={counts['yes']} No={counts['no']} None={counts['none']} Empty={counts['empty']}")
    else:
        print("No verdicts found.")


if __name__ == "__main__":
    main()
