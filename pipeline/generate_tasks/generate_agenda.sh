#!/bin/bash

# Run agenda/activity synthesis from personas
template=agenda
sample_size=1000  # set to 0 to process the full persona set
out_path=/tasks/vllm_agenda.jsonl
model_path=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
tensor_parallel_size=2


python pipeline/generate_tasks/vllm_synthesize.py \
    --sample_size $sample_size \
    --model_path $model_path \
    --output_path $out_path \
    --template $template \
    --tensor_parallel_size $tensor_parallel_size

