# BASE MODEL

# aime
lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,dtype=bfloat16,tensor_parallel_size=1 --tasks aime2024_nofigures --batch_size auto --apply_chat_template --output_path deepseek-llama-8b-aime --log_samples --gen_kwargs "max_gen_toks=32768,cutoff_token=128014"
# math-500
lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,dtype=bfloat16,tensor_parallel_size=1 --tasks math-500 --batch_size auto --apply_chat_template --output_path deepseek-llama-8b-math-500 --log_samples --gen_kwargs "max_gen_toks=32768,cutoff_token=128014"
# gpqa
lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,dtype=bfloat16,tensor_parallel_size=1 --tasks openai_gpqa_diamond --batch_size auto --apply_chat_template --output_path deepseek-llama-8b-gpqa --log_samples --gen_kwargs "max_gen_toks=32768,cutoff_token=128014"

# Intervention (feature 46379)

# aime
lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.85,sae_release=andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts,sae_id=blocks.19.hook_resid_post,sae_feature_idx=46379,sae_strength=2.0,sae_max_activation=5.840 --tasks aime2024_nofigures --batch_size auto --apply_chat_template --output_path deepseek-llama-8b-aime-sae-46379 --log_samples --gen_kwargs "max_gen_toks=32768,cutoff_token=128014"
# math-500
lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.85,sae_release=andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts,sae_id=blocks.19.hook_resid_post,sae_feature_idx=46379,sae_strength=2.0,sae_max_activation=5.840 --tasks math-500 --batch_size auto --apply_chat_template --output_path deepseek-llama-8b-math-500-sae-46379 --log_samples --gen_kwargs "max_gen_toks=32768,cutoff_token=128014"
# gpqa
lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.85,sae_release=andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts,sae_id=blocks.19.hook_resid_post,sae_feature_idx=46379,sae_strength=2.0,sae_max_activation=5.840 --tasks openai_gpqa_diamond --batch_size auto --apply_chat_template --output_path deepseek-llama-8b-gpqa-sae-46379 --log_samples --gen_kwargs "max_gen_toks=32768,cutoff_token=128014"
