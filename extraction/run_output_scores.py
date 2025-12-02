import argparse
import json
import os
import torch
from pathlib import Path
from sae_lens import SAE, ActivationsStore
from transformer_lens import HookedTransformer


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
        print(f"feature = {i}")
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

    parser.add_argument("--feature_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--topk", type=int, required=True)

    parser.add_argument("--model_name", type=str, default="gemma-2b-it")
    parser.add_argument("--sae_release", type=str, required=True)
    parser.add_argument("--sae_id", type=str, required=True)
    parser.add_argument("--hook_name", type=str, default=None,
                    help="Optional override. If not provided, the hook_name from the SAE metadata is used.")

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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = HookedTransformer.from_pretrained_no_processing(args.model_name, device=device, dtype=torch.float16)

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
    feature_scores = torch.load(args.feature_path, map_location="cpu", weights_only=True)
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


if __name__ == "__main__":
    main()