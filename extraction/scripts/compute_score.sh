#!/bin/bash

python extraction/compute_score.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --sae_path andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts \
    --dataset_path andreuka18/OpenThoughts-10k-DeepSeek-R1 \
    --sae_id blocks.19.hook_resid_post \
    --tokens_str_path extraction/reason_tokens.json \
    --expand_range 1,2 \
    --ignore_tokens 128000,128001 \
    --n_samples 4096 \
    --alpha 0.7 \
    --minibatch_size_features 48 \
    --minibatch_size_tokens 64 \
    --output_dir extraction/scores \
    --num_chunks 1 \
    --chunk_num 0
