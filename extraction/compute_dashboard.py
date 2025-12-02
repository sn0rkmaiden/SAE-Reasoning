import os
import json

from pathlib import Path
from datetime import datetime
import sys
import inspect
import re

import fire

from datasets import load_dataset
import torch

from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.data_writing_fns import save_feature_centric_vis


def extract_layer_from_hook_name(hook_name: str) -> int:
    m = re.search(r"blocks\.(\d+)\.", hook_name)
    if not m:
        raise ValueError(f"Could not extract layer from hook_name {hook_name!r}")
    return int(m.group(1))


def sanitize_model_name(model_path: str) -> str:
    return model_path.replace("/", "__")


def save_config(config_path: Path, script_name: str, args_dict: dict, derived: dict, device: str):
    config = {
        "script": script_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "args": args_dict,
        "derived": derived,
        "environment": {
            "device": device,
            "torch_version": torch.__version__,
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def compute_dashboard(
    model_path: str,
    sae_path: str,
    dataset_path: str,
    exp_root: str,
    vocab_name: str,
    sae_id: str = None,
    topk: int = 100,
    column_name: str = "text",
    minibatch_size_features: int = 256,
    minibatch_size_tokens: int = 64,
    n_samples: int = 5000,
    separate_files: bool = False,
    chunk_num: int = -1,
):
    """Compute `sae_dashboard` interfaces for top-k features, tied to a specific ReasonScore set.

    Directory layout (scores_dir is chosen automatically):

        exp_root /
          vocab_name /
            <sanitized model_path> /
              layer_<L> /
                reason_scores /
                  chunk_XXXX / feature_scores.pt
                  merged / feature_scores.pt

    - If chunk_num >= 0 → use that chunk directory.
    - If chunk_num < 0  → use /merged, creating it (by merging chunks) if needed.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(">>> Loading SAE and LLM")
    if sae_id is None:
        sae = SAE.load_from_pretrained(sae_path, device=device)
    else:
        sae, _, _ = SAE.from_pretrained(sae_path, sae_id, device=device)

    # hook name & layer from SAE
    if hasattr(sae.cfg, "hook_name"):
        hook_name = sae.cfg.hook_name
    elif hasattr(sae.cfg, "metadata") and hasattr(sae.cfg.metadata, "hook_name"):
        hook_name = sae.cfg.metadata.hook_name
    else:
        raise ValueError("SAE config does not contain hook_name or metadata.hook_name")

    layer = extract_layer_from_hook_name(hook_name)
    model_dir_name = sanitize_model_name(model_path)

    base_dir = Path(exp_root) / vocab_name / model_dir_name / f"layer_{layer}" / "reason_scores"
    if chunk_num >= 0:
        scores_dir = base_dir / f"chunk_{chunk_num:04d}"
        mode = "chunk"
    else:
        scores_dir = base_dir / "merged"
        mode = "merged"

    if not scores_dir.exists():
        print(f"[info] Creating scores_dir: {scores_dir}")
        scores_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Using scores_dir ({mode}): {scores_dir}")

    feature_scores_path = scores_dir / "feature_scores.pt"

    # If merged mode and feature_scores.pt doesn't exist, merge chunk scores into merged.
    if mode == "merged" and not feature_scores_path.exists():
        print(">>> feature_scores.pt not found in merged dir, attempting to merge chunks...")
        if not base_dir.exists():
            raise FileNotFoundError(f"Reason scores base_dir does not exist: {base_dir}")

        chunk_dirs = sorted(
            [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")]
        )
        if not chunk_dirs:
            raise FileNotFoundError(
                f"No chunk_* dirs found under {base_dir}, nothing to merge for merged scores."
            )

        all_scores = []
        for cd in chunk_dirs:
            fs_path = cd / "feature_scores.pt"
            if not fs_path.exists():
                raise FileNotFoundError(f"Missing feature_scores.pt in {cd}")
            print(f"  - Loading chunk scores from {fs_path}")
            all_scores.append(torch.load(fs_path, weights_only=True, map_location="cpu"))

        merged_scores = torch.concat(all_scores, dim=0)
        torch.save(merged_scores, feature_scores_path)
        print(f">>> Saved merged feature_scores.pt to {feature_scores_path}")

    if not feature_scores_path.exists():
        raise FileNotFoundError(f"feature_scores.pt not found at {feature_scores_path}")

    feature_scores = torch.load(feature_scores_path, weights_only=True, map_location="cpu")
    topk_features = feature_scores.topk(k=topk).indices.tolist()

    print(">>> Loading model...")
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

    feature_vis_config = SaeVisConfig(
        hook_point=hook_name,
        features=topk_features,
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        verbose=True,
        device=device
    )

    print(">>> Running SaeVisRunner...")
    visualization_data = SaeVisRunner(
        feature_vis_config
    ).run(
        encoder=sae,
        model=model,
        tokens=torch.tensor(token_dataset[:n_samples]["tokens"], dtype=torch.long)
    )

    dashboard_dir = scores_dir / "dashboard" / f"topk_{topk}"
    if dashboard_dir.exists():
        print(f"[warning] Dashboard dir already exists — reusing: {dashboard_dir}")
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    dashboard_path = dashboard_dir / f"topk-{topk}.html"
    print(f">>> Saving dashboard to: {dashboard_path}")
    save_feature_centric_vis(
        sae_vis_data=visualization_data,
        filename=str(dashboard_path),
        separate_files=separate_files
    )

    # Save config
    arg_names = inspect.getfullargspec(compute_dashboard).args
    frame_locals = locals()
    args_dict = {name: frame_locals[name] for name in arg_names}

    derived = {
        "hook_name": hook_name,
        "layer": layer,
        "scores_dir": str(scores_dir),
        "dashboard_dir": str(dashboard_dir),
        "mode": mode,
        "n_features_total": int(feature_scores.shape[0]),
        "topk": topk,
        "topk_feature_ids": topk_features,
        "feature_scores_path": str(feature_scores_path),
        "dashboard_path": str(dashboard_path),
    }
    save_config(dashboard_dir / "config.json", "compute_dashboard.py", args_dict, derived, device)


if __name__ == "__main__":
    fire.Fire(compute_dashboard)
