# I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders

This code is the official implementation of [Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders](https://arxiv.org/abs/2503.18878).

Right now we continue our experiments with different *reasoning* sets and hyperparameters to improve our results. We will release top feature indices and corresponding dashboards soon. Stay tuned!

## Preliminaries

### Installation

1. Create a virtual environment and activate it (e.g conda environment):
```bash
conda create -n sae_reasoning python=3.11
conda activate sae_reasoning
```
2. Install build requirements:
```bash
pip install -r requirements.txt
```
3. We cloned `TransformerLens` at commit `e65fafb4791c66076bc54ec9731920de1e8c676f` and modified it to support deepseek distilled models (Llama-8B, Qwen-1.5B, Qwen-7B).
Install our version:
```bash
cd TransformerLens
pip install -e .
```
4. Install `sae_lens` and `sae-dashboard`:
```bash
pip install sae_lens==5.5.2 sae-dashboard
```

### Repository Structure

- `training/`: SAE training scripts
- `extraction/`: extraction scripts
- `evaluation/`: evaluation scripts

### Artifacts

- **SAE**: https://huggingface.co/andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts
- **Data**:
    - SAE training data: https://huggingface.co/datasets/andreuka18/DeepSeek-R1-Distill-Llama-8B-lmsys-openthoughts-tokenized
    - Extraction data: https://huggingface.co/datasets/andreuka18/OpenThoughts-10k-DeepSeek-R1

## SAE Training

### Data

To train our SAE, we use the [LMSYS-Chat-1M](lmsys/lmsys-chat-1m) and the [OpenThoughts-114k](open-thoughts/OpenThoughts-114k) datasets. We provide scripts to convert this datasets into tokenized version w.r.t. the model and compatible to `SAELens`:
- `prepare_lmsys_dataset.py` - convert lmsys-chat-1m dataset to tokens, push to hf
- `prepare_openthoughts_dataset.py` - convert openthoughts-114k dataset to tokens, push to hf

`SAELens` doesn't support passing multiple datasets. To merge obtained tokens into one dataset - use `datasets.concatenate_datasets`.

### Training

We use [SAELens](https://github.com/jbloomAus/SAELens) to train SAE. To run training use `training/train_sae.py` script by passing the `.yaml` configuration file. We train our SAE using the `training/configs/r1-distill-llama-8b.yaml` config on 1 H100, 80GB. 

Command to run training:
```bash
WANDB_API_KEY="YOUR API KEY" python training/train_sae.py 'training/configs/r1-distill-llama-8b.yaml'
```

After training, you can upload your SAE following this [guide](https://jbloomaus.github.io/SAELens/training_saes/#uploading-saes-to-huggingface).

## Extraction of Reasoning Features

### Data

We use a subset of OpenThoughts-114k dataset to collect statistics and construct feature interfaces. To construct this dataset - use `extraction/prepare_openthoughts_subset.py` script by passing number of samples and your `huggingface` credentials.

### 1. Compute `ReasonScore`

Use `extraction/compute_score.py` to calculate `ReasonScore` for each of the SAE features. 

To run calculation with the parameters as in the paper, use:
```bash
bash extraction/scripts/compute_score.sh
```

### 2. Compute feature interfaces

We utilize [SAEDashboard](https://github.com/jbloomAus/SAEDashboard) to obtain interfaces for SAE features. Use `extraction/compute_dashboard.py` to get the `.html` interfaces for `topk` features, sorted by `ReasonScore`.

We provide an example with filled parameters, use:
```bash
bash extraction/scripts/compute_dashboard.sh
```

## Evaluation on reasoning benchmarks

We cloned `lm-evaluation-harness` at commit `a87fe425ec55d90083510fc8b2a07596b76e57b3` and modified it to support single-feature intervention.

Setup:
```bash
cd evaluation/lm-evaluation-harness
pip install -e '.[vllm]'
```

All commands are in `evaluation/evaluate.sh`.

**NOTE:** Some benchmarks (e.g. AIME-2024 and MATH-500) require a verifier (separate LLM) to correctly score the results. By default it is disabled. In our evaluation experiments. we have used the `openrouter` API and set `meta-llama/llama-3.3-70b-instruct` as a verifier. To enable the verifier, you should specify your openrouter API key and verifier as environment variables, e.g.:
`OPENROUTER_API_KEY="YOUR KEY" PROCESSOR=meta-llama/llama-3.3-70b-instruct ./evaluation/evaluate.sh`

## Citation

If you find this repository and our work useful, please consider giving a star and please cite as:

```bash
@misc{galichin2025icoveredbaseshere,
      title={I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders}, 
      author={Andrey Galichin and Alexey Dontsov and Polina Druzhinina and Anton Razzhigaev and Oleg Y. Rogov and Elena Tutubalina and Ivan Oseledets},
      year={2025},
      eprint={2503.18878},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.18878}, 
}
```
