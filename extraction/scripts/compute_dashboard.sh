#!/bin/bash

python extraction/compute_dashboard.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --sae_path andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts \
    --dataset_path andreuka18/OpenThoughts-10k-DeepSeek-R1 \
    --scores_dir extraction/scores \
    --sae_id blocks.19.hook_resid_post \
    --topk 100 \
    --n_samples 10000 \
    --minibatch_size_features 128 \
    --minibatch_size_tokens 64 \
    --output_dir extraction/dashboards \
