import os
import fire
import json

from datasets import load_dataset
import torch

from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.data_writing_fns import save_feature_centric_vis


def compute_dashboard(
    model_path: str,
    sae_path: str,
    dataset_path: str,
    scores_dir: str,
    output_dir: str,
    sae_id: str = None,
    topk: int = 100,
    column_name: str = "text",
    minibatch_size_features: int = 256,
    minibatch_size_tokens: int = 64,
    n_samples: int = 5000,
    separate_files: bool = False
):
    """Compute `sae_dashboard` interfaces for each feature."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(">>> Loading SAE and LLM")
    if sae_id is None:
        sae = SAE.load_from_pretrained(sae_path, device=device)
    else:
        sae, _, _ = SAE.from_pretrained(sae_path, sae_id, device=device)

    model = HookedTransformer.from_pretrained_no_processing(
        model_path,
        dtype=torch.bfloat16,
        device=device,
    )
    # make pad token different from `bos` and `eos` to prevent removing `bos`/`eos` token during slicing
    if model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
        model.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    print(">>> Loading dataset")
    dataset = load_dataset(dataset_path, streaming=False, split="train")
    if column_name == "tokens":
        token_dataset = dataset
    else:
        print(">>> Tokenize dataset")
        token_dataset = tokenize_and_concatenate(
            dataset=dataset,
            tokenizer=model.tokenizer,
            streaming=False,
            max_length=sae.cfg.context_size,
            column_name=column_name,
            add_bos_token=sae.cfg.prepend_bos,
            num_proc=4
        )

    # We could've processed features in separate chunks; read all results and merge into single
    if not os.path.exists(os.path.join(scores_dir, "feature_scores.pt")):
        filenames = [n for n in os.listdir(scores_dir) if "feature_scores" in n]
        filenames.sort()

        print(">>> Found chunks of feature scores: {}. Merging and saving".format(filenames))
        
        scores = torch.concat([torch.load(os.path.join(scores_dir, n), weights_only=True) for n in filenames])
        torch.save(scores, os.path.join(scores_dir, "feature_scores.pt"))
    
    feature_scores = torch.load(os.path.join(scores_dir, "feature_scores.pt"), weights_only=True, map_location="cpu")
    topk_features = feature_scores.topk(k=topk).indices.tolist()

    feature_vis_config = SaeVisConfig(
        hook_point=sae.cfg.hook_name,
        features=topk_features,
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        verbose=True,
        device=device
    )

    visualization_data = SaeVisRunner(
        feature_vis_config
    ).run(
        encoder=sae,
        model=model,
        tokens=token_dataset[:n_samples]["tokens"]
    )

    os.makedirs(output_dir, exist_ok=True)
    dashboard_path = os.path.join(output_dir, f'topk-{topk}.html')
    save_feature_centric_vis(
        sae_vis_data=visualization_data, 
        filename=dashboard_path, 
        separate_files=separate_files
    )


if __name__ == "__main__":
    fire.Fire(compute_dashboard)
