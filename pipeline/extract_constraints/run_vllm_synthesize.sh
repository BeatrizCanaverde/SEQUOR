#!/bin/bash

dataset=lmsys-chat-1m  # lmsys-chat-1m  WildChat-4.8M
template=all_turns_revised  # one_turn  all_turns  all_turns_revised
sample_size=50000  # set sample_size=0 if you want to use the full version of 200k personas
out_path=/constraints_extracted/${dataset}_${template}_output.jsonl
model_path="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
seed=2025


python /pipeline/extract_constraints/vllm_synthesize.py \
    --model_path $model_path \
    --dataset $dataset \
    --template $template \
    --sample_size $sample_size  \
    --output_path $out_path  \
    --seed $seed

