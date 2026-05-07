import argparse
import re
from pathlib import Path
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.io import cached_asset_path_or_download
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.data import Document


_EN_BADWORDS_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/25e679f03d96baa721cde20db9944649e8d0a844/en"

# Add your custom bad words here
CUSTOM_BADWORDS = [
    "sex",
    "porn",
    "nud",
]


class BadWordsFilter(BaseFilter):
    """
    Filter documents containing bad words from the English badwords list.
    """
    
    name = "🚫 Bad Words Filter"
    
    def __init__(self, exclusion_writer=None):
        super().__init__(exclusion_writer)
        self._badwords_regex = None
    
    def _get_badwords_regex(self):
        """Load and compile badwords regex pattern."""
        if self._badwords_regex is None:
            local_path = cached_asset_path_or_download(
                _EN_BADWORDS_URL,
                namespace="filters",
                subfolder="badwords",
            )
            badwords = set()
            # Load badwords from file
            with open(local_path, "rt") as f:
                badwords.update(line.strip() for line in f if line.strip())
            
            # Add custom badwords
            badwords.update(CUSTOM_BADWORDS)
            
            # Escape special regex characters and create pattern
            words = [re.escape(w) for w in badwords]
            # Match only when flanked by non-word chars
            self._badwords_regex = re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))
        
        return self._badwords_regex
    
    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """
        Filter out documents containing bad words.
        
        Returns:
            True to keep the document, False to filter it out
        """
        badwords_regex = self._get_badwords_regex()
        badwords_found = badwords_regex.search(doc.text.lower())
        
        if badwords_found is not None:
            self.stat_update("documents_with_badwords")
            return False, "contains_badwords"
        
        return True


class BadSequenceFilter(BaseFilter):
    """
    Filter documents containing any bad sequence anywhere in the text.
    """
    
    name = "🚫 Bad Sequence Filter"
    
    def __init__(self, exclusion_writer=None):
        super().__init__(exclusion_writer)
    
    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """
        Filter out documents containing any bad sequence.
        
        Returns:
            True to keep the document, False to filter it out
        """
        for bad_sequence in CUSTOM_BADWORDS:
            if bad_sequence in doc.text.lower():
                self.stat_update("documents_with_bad_sequence")
                return False, "contains_bad_sequence"
        
        return True


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


def run_badwords_filter(input_jsonl_path: str, output_dir: str):
    """
    Filter the JSONL file to remove constraints containing bad words.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_dir: Directory for output files
    """
    input_path = Path(input_jsonl_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Running bad words filtering on: {input_path}")
    print(f"Output directory: {output_path}")
    
    # Count input instances
    input_count = count_instances(input_path)
    
    # Create pipeline with bad words filter
    pipeline_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=str(input_path.parent),
                glob_pattern=input_path.name,
                text_key="text",
                id_key="id",
                default_metadata={}
            ),
            BadWordsFilter(
                exclusion_writer=JsonlWriter(
                    output_folder=str(output_path / "badwords_excluded"),
                    output_filename="excluded.jsonl",
                    compression=None
                )
            ),
            BadSequenceFilter(
                exclusion_writer=JsonlWriter(
                    output_folder=str(output_path / "bad_sequence_excluded"),
                    output_filename="excluded.jsonl",
                    compression=None
                )
            ),
            JsonlWriter(
                output_folder=str(output_path),
                output_filename="badwords_filtered.jsonl",
                compression=None  # Disable compression to save as plain .jsonl
            )
        ],
        tasks=1,
        logging_dir=str(output_path / "logs")
    )
    
    # Execute the pipeline
    pipeline_executor.run()
    
    # Count output instances
    output_file = output_path / 'badwords_filtered.jsonl'
    output_count = count_instances(output_file)
    
    removed_count = input_count - output_count
    print()
    print(f"✓ Bad words filtering complete")
    print(f"✓ Output: {output_file}")
    print(f"✓ Input instances: {input_count}")
    print(f"✓ Instances kept: {output_count}")
    print(f"✓ Instances removed: {removed_count}")
    print(f"✓ Removal rate: {removed_count/input_count*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter constraints containing bad words.")
    parser.add_argument("--input_jsonl_path", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files.")
    args = parser.parse_args()
    
    run_badwords_filter(input_jsonl_path=args.input_jsonl_path, output_dir=args.output_dir)
