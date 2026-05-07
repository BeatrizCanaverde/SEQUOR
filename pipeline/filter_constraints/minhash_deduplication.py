import os
import argparse
from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages


# MinHash configuration - adjust parameters as needed
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=50,
    hashes_per_bucket=4,
    n_grams=3,
)  # better precision -> fewer false positions (collisions)


def run_minhash_deduplication(input_jsonl_path: str, output_dir: str):
    """
    Run MinHash deduplication on constraint data.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_dir: Directory for output files
    """
    
    signatures_dir="minhash_sigs"
    buckets_dir="minhash_buckets"
    clusters_dir="minhash_clusters"
    excluded_dir="minhash_excluded"

    # Construct full paths
    signatures_path = os.path.join(output_dir, signatures_dir)
    buckets_path = os.path.join(output_dir, buckets_dir)
    clusters_path = os.path.join(output_dir, clusters_dir)
    excluded_path = os.path.join(output_dir, excluded_dir)

    # Stage 1: Compute MinHash signatures
    pipeline_1 = [
        JsonlReader(data_folder=os.path.dirname(input_jsonl_path) or ".", 
                   glob_pattern=os.path.basename(input_jsonl_path)),
        MinhashDedupSignature(
            output_folder=signatures_path, 
            config=minhash_config, 
            language=Languages.english
        ),
    ]

    # Stage 2: Find matches between signatures in each bucket
    pipeline_2 = [
        MinhashDedupBuckets(
            input_folder=signatures_path,
            output_folder=buckets_path,
            config=minhash_config,
        ),
    ]

    # Stage 3: Create clusters of duplicates
    pipeline_3 = [
        MinhashDedupCluster(
            input_folder=buckets_path,
            output_folder=clusters_path,
            config=minhash_config,
        ),
    ]

    # Stage 4: Filter duplicates and write final output
    pipeline_4 = [
        JsonlReader(data_folder=os.path.dirname(input_jsonl_path) or ".", 
                   glob_pattern=os.path.basename(input_jsonl_path)),
        MinhashDedupFilter(
            input_folder=clusters_path,
            exclusion_writer=JsonlWriter(excluded_path, output_filename="excluded.jsonl", compression=None),
        ),
        JsonlWriter(output_dir, output_filename="minhash_deduplicated.jsonl", compression=None),
    ]

    executor_1: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_1, workers=1, tasks=1)
    executor_2: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_2, workers=1, tasks=minhash_config.num_buckets)
    executor_3: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_3, workers=1, tasks=1)
    executor_4: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_4, workers=1, tasks=1)

    print("=" * 80)
    print("Stage 1: Computing MinHash signatures...")
    print("=" * 80)
    stats_1 = executor_1.run()
    print(stats_1)
    
    print("\n" + "=" * 80)
    print("Stage 2: Finding matches in buckets...")
    print("=" * 80)
    stats_2 = executor_2.run()
    print(stats_2)
    
    print("\n" + "=" * 80)
    print("Stage 3: Creating duplicate clusters...")
    print("=" * 80)
    stats_3 = executor_3.run()
    print(stats_3)
    
    print("\n" + "=" * 80)
    print("Stage 4: Filtering duplicates...")
    print("=" * 80)
    stats_4 = executor_4.run()
    print(stats_4)
    
    # Extract document counts from stats
    input_docs = 0
    output_docs = 0
    
    for s in stats_1.stats:
        if 'READER' in s.name and 'documents' in s.stats:
            input_docs = s.stats['documents'].total
            
    for s in stats_4.stats:
        if 'WRITER' in s.name and 'total' in s.stats:
            output_docs = s.stats['total'].total
    
    removed_docs = input_docs - output_docs
    duplicate_rate = (removed_docs / input_docs * 100) if input_docs > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"✓ MinHash deduplication complete!")
    print(f"✓ Output: {output_dir}")
    print(f"✓ Instances kept: {output_docs}")
    print(f"✓ Instances removed (duplicates): {removed_docs}")
    print(f"✓ Duplicate rate: {duplicate_rate:.2f}%")
    print("=" * 80)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run MinHash deduplication on constraint data.")
    parser.add_argument("--input_jsonl_path", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files.")
    args = parser.parse_args()
    
    run_minhash_deduplication(input_jsonl_path=args.input_jsonl_path, output_dir=args.output_dir)
