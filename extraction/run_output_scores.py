import argparse
import json
import torch
import os

from pathlib import Path
from datetime import datetime
import sys
import re

from sae_lens import SAE, ActivationsStore
from transformer_lens import HookedTransformer


def extract_layer_from_hook_name(hook_name: str) -> int:
    m = re.search(r"blocks\.(\d+)\.", hook_name)
    if not m:
        raise ValueError(f"Could not extract layer from hook_name {hook_name!r}")
    return int(m.group(1))


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


def compute_output_scores(
    model: HookedTransformer,
    sae: SAE,
    hook_name: str,
    feature_ids: list[int],
    a_max: torch.Tensor,
    s: float = 6.0,
    top_k: int = 10,
    prompt: str = "In my experience,",
    device: torch.device = torch.device("cuda"),
) -> dict[int, dict]:

    model = model.to(device)
    sae = sae.to(device)
    model.eval()

    W_U = model.W_U.to(device)

    print("Precomputing top tokens...")
    top_tokens = {}
    for i in feature_ids:
        f_i = sae.W_dec[i, :].to(device)
        try:
            f_i_ln = model.ln_final(f_i.unsqueeze(0)).squeeze(0)
        except Exception:
            f_i_ln = f_i

        scores = f_i_ln @ W_U
        _, top_idx = torch.topk(scores, k=top_k)
        top_tokens[i] = top_idx.cpu().tolist()

    print("Computing output scores...")
    results = {}
    for i in feature_ids:
        l_star = top_tokens[i][0]

        def hook_fn(value, hook):
            batch, seq_len, d_model = value.shape
            h_flat = value.reshape(-1, d_model)
            a = sae.encode(h_flat)
            a[:, i] = a[:, i] + s * a_max[i].to(device)
            h_mod = sae.decode(a) + sae.b_dec.to(device)
            return h_mod.reshape(batch, seq_len, d_model)

        tokens = model.to_tokens(prompt).to(device)
        logits_after = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, hook_fn)], return_type="logits"
        )

        probs = torch.softmax(logits_after[0, -1], dim=-1)
        prob_star = probs[l_star].item()
        rank_star = int((probs > prob_star).sum().item()) + 1
        vocab_size = probs.shape[0]

        output_score = prob_star * (1 - (rank_star / vocab_size))

        results[i] = {
            "top_tokens": top_tokens[i],
            "prob": prob_star,
            "rank": rank_star,
            "output_score": output_score,
        }

    return results


def compute_a_max_streaming(
    model,
    sae,
    hook_name,
    dataset,
    device=torch.device("cuda"),
    total_tokens: int = 1_000_000,
    store_batch_size_prompts: int = 4,
    train_batch_size_tokens: int = 512,
    n_batches: int = 100,
):
    model = model.to(device)
    sae = sae.to(device)
    model.eval()

    store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        dataset=dataset,
        streaming=True,
        store_batch_size_prompts=store_batch_size_prompts,
        train_batch_size_tokens=train_batch_size_tokens,
        total_tokens=total_tokens,
        device=device,
    )
    print("Loaded activation store")

    d_sae = sae.cfg.d_sae
    a_max = torch.zeros(d_sae, device="cpu")

    for i in range(n_batches):
        print(f"Batch {i+1}/{n_batches}")
        batch_tokens = store.get_batch_tokens(batch_size=1).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_tokens, names_filter=[hook_name]
            )

        h = cache[hook_name]
        h_flat = h.reshape(-1, h.shape[-1])
        latent = sae.encode(h_flat)
        batch_max = latent.cpu().max(dim=0).values

        a_max = torch.maximum(a_max, batch_max)

        del batch_tokens, h, h_flat, latent
        torch.cuda.empty_cache()

    return a_max


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_path",
        type=str,
        required=True,
        help="Path to feature_scores.pt used to select top-k features."
    )

    parser.add_argument("--topk", type=int, required=True,
                        help="Number of top features to use for steering / output scores.")

    parser.add_argument("--model_name", type=str, default="gemma-2b-it")
    parser.add_argument("--sae_release", type=str, required=True)
    parser.add_argument("--sae_id", type=str, required=True)
    parser.add_argument(
        "--hook_name",
        type=str,
        default=None,
        help="Optional override. If not provided, hook_name is taken from SAE config."
    )

    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--n_batches", type=int, default=2)
    parser.add_argument("--total_tokens", type=int, default=1_000_000)
    parser.add_argument("--store_batch_size_prompts", type=int, default=4)
    parser.add_argument("--train_batch_size_tokens", type=int, default=512)

    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--top_k_tokens", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="In my experience,")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_path = Path(args.feature_path).resolve()
    if not feature_path.exists():
        raise FileNotFoundError(f"feature_path not found: {feature_path}")

    # Output directory is derived from the feature_scores location:
    #   <.../reason_scores/chunk_XXXX or merged>/output_scores/topk_<topk>/
    feature_dir = feature_path.parent
    out_dir = feature_dir / "output_scores" / f"topk_{args.topk}"
    if out_dir.exists():
        print(f"[warning] Output dir already exists — reusing: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> Output scores will be saved under: {out_dir}")

    print("Loading model...")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=device,
        dtype=torch.float16
    )

    print("Loading SAE...")
    sae, _, _ = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=device
    )

    if args.hook_name is None:
        if hasattr(sae.cfg, "hook_name"):
            hook_name = sae.cfg.hook_name
        elif hasattr(sae.cfg, "metadata") and hasattr(sae.cfg.metadata, "hook_name"):
            hook_name = sae.cfg.metadata.hook_name
        else:
            raise ValueError("Cannot determine hook_name from SAE config.")
        print(f"Using hook_name from SAE config: {hook_name}")
    else:
        hook_name = args.hook_name
        print(f"Using user-provided hook_name: {hook_name}")

    layer = extract_layer_from_hook_name(hook_name)

    print("Computing a_max...")
    a_max = compute_a_max_streaming(
        model,
        sae,
        hook_name=hook_name,
        dataset=args.dataset,
        n_batches=args.n_batches,
        total_tokens=args.total_tokens,
        store_batch_size_prompts=args.store_batch_size_prompts,
        train_batch_size_tokens=args.train_batch_size_tokens,
        device=device,
    )

    a_max_path = out_dir / "a_max.pt"
    torch.save(a_max, a_max_path)
    print(f"Saved a_max to: {a_max_path}")

    print("Loading feature scores...")
    feature_scores = torch.load(feature_path, map_location="cpu", weights_only=True)
    topk_features = feature_scores.topk(k=args.topk).indices.tolist()
    feature_ids = sorted(topk_features)

    print("Computing output scores...")
    results = compute_output_scores(
        model,
        sae,
        hook_name,
        feature_ids,
        a_max,
        s=args.s,
        top_k=args.top_k_tokens,
        prompt=args.prompt,
        device=device,
    )

    output_path = out_dir / "output_scores.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {output_path}")

    # Save config.json
    args_dict = vars(args).copy()
    derived = {
        "hook_name": hook_name,
        "layer": layer,
        "feature_path": str(feature_path),
        "output_dir": str(out_dir),
        "a_max_path": str(a_max_path),
        "output_scores_path": str(output_path),
        "n_features_total": int(feature_scores.shape[0]),
        "n_features_used": len(feature_ids),
        "topk_features": feature_ids,
    }
    save_config(out_dir / "config.json", "run_output_scores.py", args_dict, derived, device)


if __name__ == "__main__":
    main()
