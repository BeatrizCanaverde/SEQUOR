import argparse
import json
import sys
import jsonargparse
from pathlib import Path
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter


def count_instances(file_path: Path) -> int:
    """
    Count the number of instances (lines) in a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
    """
    input_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                input_count += 1
    return input_count


def run_language_filter(input_jsonl_path: str, output_dir: str):
    """
    Filter the JSONL file to keep only English constraints using datatrove.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_dir: Directory for output files
    """
    input_path = Path(input_jsonl_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Running language identification on: {input_path}")
    print(f"Output directory: {output_path}")
    
    # Count input instances
    input_count = count_instances(input_path)
    
    # Create pipeline with language filter
    pipeline_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=str(input_path.parent),
                glob_pattern=input_path.name,
                text_key="text",
                id_key="id",
                default_metadata={}
            ),
            LanguageFilter(
                language_threshold=0.65,  # Confidence threshold for language detection
                languages=["en"],  # Keep only English
                exclusion_writer=JsonlWriter(
                    output_folder=str(output_path / "language_excluded"),
                    output_filename="excluded.jsonl",
                    compression=None
                )
            ),
            JsonlWriter(
                output_folder=str(output_path),
                output_filename="language_filtered.jsonl",
                compression=None  # Disable compression to save as plain .jsonl
            )
        ],
        tasks=1,
        logging_dir=str(output_path / "logs")
    )
    
    # Execute the pipeline
    pipeline_executor.run()
    
    # Count output instances
    output_file = output_path / 'language_filtered.jsonl'
    output_count = count_instances(output_file)
    
    removed_count = input_count - output_count
    print()
    print(f"✓ Language filtering complete")
    print(f"✓ Output: {output_file}")
    print(f"✓ Input instances: {input_count}")
    print(f"✓ Instances kept: {output_count}")
    print(f"✓ Instances removed: {removed_count}")
    print(f"✓ Removal rate: {removed_count/input_count*100:.2f}%")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert JSON to JSONL format for constraint data.")
    parser.add_argument("--input_jsonl_path", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files.")
    args = parser.parse_args()
    
    run_language_filter(input_jsonl_path=args.input_jsonl_path, output_dir=args.output_dir)