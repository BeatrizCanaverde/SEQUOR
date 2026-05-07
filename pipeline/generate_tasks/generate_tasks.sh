#!/bin/bash

# Run question synthesis from activities
template=activities
in_path=/tasks/extracted_activities.jsonl  # activities with personas
out_path=/tasks/vllm_questions.jsonl
model_path=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
tensor_parallel_size=2


python pipeline/generate_tasks/vllm_synthesize_questions.py \
    --input_path $in_path \
    --model_path $model_path \
    --output_path $out_path \
    --tensor_parallel_size $tensor_parallel_size

