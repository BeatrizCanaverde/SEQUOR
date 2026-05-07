import json
import re
import argparse


def extract_activities(synth_text):
    # Split by \n### (at least one newline, at least three #, optional whitespace)
    parts = re.split(r"\n#{3,}\s*", synth_text)
    # Remove any empty strings and strip whitespace
    activities = [p.strip() for p in parts[1:] if p.strip()]
    return activities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract activities from synthesized text.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as fin, open(args.output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
                persona = obj.get("input persona", "")
                synth_text = obj.get("synthesized text", "")
                activities = extract_activities(synth_text)
                out_obj = {
                    "input persona": persona,
                    "activities": activities
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            except Exception as e:
                continue  # skip malformed lines

