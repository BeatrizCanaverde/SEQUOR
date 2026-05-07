import os
import json
import re
import argparse

def extract_persona_activity(prompt_text):
    persona_match = re.search(r"\nPersona:(.*?)\nActivity/Task:", prompt_text, re.DOTALL)
    activity_match = re.search(r"\nActivity/Task:(.*?)\n\nNote:", prompt_text, re.DOTALL)
    persona = persona_match.group(1).strip() if persona_match else ""
    activity = activity_match.group(1).strip() if activity_match else ""
    return persona, activity

def extract_questions(response_text):
    # Split by \n### (at least one newline, at least three #, optional whitespace)
    parts = re.split(r"\n#{3,}\s*", response_text)
    questions = [p.strip() for p in parts[1:] if p.strip()]
    return questions

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    persona_map = {}
    persona_order = []

    with open(args.input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            prompt_text = entry["prompt"]
            questions_text = entry["questions"]
            persona, activity = extract_persona_activity(prompt_text)
            questions = extract_questions(questions_text)

            # Use persona text as key, keep order for file naming
            if persona not in persona_map:
                persona_map[persona] = []
                persona_order.append(persona)
            persona_map[persona].append((activity, questions))

    # Write one file per persona
    for idx, persona in enumerate(persona_order, start=1):
        filename = f"persona_{idx}.jsonl"
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as fout:
            for activity, questions in persona_map[persona]:
                for question in questions:
                    out = {
                        "persona": persona,
                        "activity": activity,
                        "prompt": question
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract activities from synthesized text.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()
    main(args)