#!/usr/bin/env python3
"""Profile a complete OutputScore job (a_max + top-k interventions).

Run from SAE-Reasoning-main. This uses the repository's own OutputScore functions,
but writes results under a separate profiling directory.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def unwrap_sae(value):
    return value[0] if isinstance(value, tuple) else value


def get_hook_name(sae, override: str | None) -> str:
    if override:
        return override
    if hasattr(sae.cfg, "hook_name"):
        return sae.cfg.hook_name
    md = getattr(sae.cfg, "metadata", None)
    if md is not None and hasattr(md, "hook_name"):
        return md.hook_name
    raise ValueError("Cannot infer SAE hook name")


def gpu_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "available": True,
        "name": torch.cuda.get_device_name(),
        "total_memory_bytes": int(p.total_memory),
        "compute_capability": f"{p.major}.{p.minor}",
        "cuda_runtime": torch.version.cuda,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_path", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--sae_release", required=True)
    ap.add_argument("--sae_id", required=True)
    ap.add_argument("--hook_name", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_batches", type=int, default=2)
    ap.add_argument("--total_tokens", type=int, default=1_000_000)
    ap.add_argument("--store_batch_size_prompts", type=int, default=4)
    ap.add_argument("--train_batch_size_tokens", type=int, default=512)
    ap.add_argument("--s", type=float, default=10.0)
    ap.add_argument("--top_k_tokens", type=int, default=50)
    ap.add_argument("--prompt", default="In my experience,")
    ap.add_argument("--reuse_a_max", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    try:
        from extraction.run_output_scores import compute_a_max_streaming, compute_output_scores
    except Exception as e:
        raise RuntimeError("Run from SAE-Reasoning-main or add it to PYTHONPATH") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    sync(); total_t0 = time.perf_counter()

    sync(); t0 = time.perf_counter()
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name, device=str(device), dtype=torch.float16
    )
    sync(); model_load_seconds = time.perf_counter() - t0

    sync(); t0 = time.perf_counter()
    sae = unwrap_sae(SAE.from_pretrained(
        release=args.sae_release, sae_id=args.sae_id, device=str(device)
    ))
    sync(); sae_load_seconds = time.perf_counter() - t0

    hook_name = get_hook_name(sae, args.hook_name)
    m = re.search(r"blocks\.(\d+)\.", hook_name)
    layer = int(m.group(1)) if m else None
    prompt_token_length = int(model.to_tokens(args.prompt).shape[-1])
    d_sae = int(getattr(sae.cfg, "d_sae", sae.W_dec.shape[0]))
    d_model = int(sae.W_dec.shape[1])

    if args.reuse_a_max:
        sync(); t0 = time.perf_counter()
        a_max = torch.load(args.reuse_a_max, map_location="cpu", weights_only=True)
        sync(); amax_seconds = time.perf_counter() - t0
        amax_measured = False
    else:
        sync(); t0 = time.perf_counter()
        a_max = compute_a_max_streaming(
            model=model,
            sae=sae,
            hook_name=hook_name,
            dataset=args.dataset,
            device=device,
            total_tokens=args.total_tokens,
            store_batch_size_prompts=args.store_batch_size_prompts,
            train_batch_size_tokens=args.train_batch_size_tokens,
            n_batches=args.n_batches,
        )
        sync(); amax_seconds = time.perf_counter() - t0
        torch.save(a_max, out_dir / "a_max.pt")
        amax_measured = True

    feature_scores = torch.load(args.feature_path, map_location="cpu", weights_only=True)
    feature_ids = sorted(feature_scores.topk(k=args.topk).indices.tolist())

    sync(); t0 = time.perf_counter()
    results = compute_output_scores(
        model=model,
        sae=sae,
        hook_name=hook_name,
        feature_ids=feature_ids,
        a_max=a_max,
        s=args.s,
        top_k=args.top_k_tokens,
        prompt=args.prompt,
        device=device,
    )
    sync(); outputscore_seconds = time.perf_counter() - t0

    sync(); total_seconds = time.perf_counter() - total_t0
    (out_dir / "output_scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    profile = {
        "kind": "outputscore_profile",
        "command_arguments": vars(args),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": gpu_metadata(),
        },
        "model": {"hook_name": hook_name, "layer": layer},
        "workload": {
            "candidate_features": len(feature_ids),
            "candidate_feature_ids": feature_ids,
            "a_max_forward_calls": 0 if args.reuse_a_max else args.n_batches,
            "approx_a_max_token_positions": 0 if args.reuse_a_max else args.n_batches * args.train_batch_size_tokens,
            "intervention_forward_calls": len(feature_ids),
            "representative_prompt": args.prompt,
            "representative_prompt_token_length": prompt_token_length,
            "d_sae": d_sae,
            "d_model": d_model,
        },
        "measured": {
            "model_load_seconds": model_load_seconds,
            "sae_load_seconds": sae_load_seconds,
            "a_max_seconds": amax_seconds,
            "a_max_was_computed": amax_measured,
            "outputscore_50_candidates_seconds": outputscore_seconds,
            "total_seconds": total_seconds,
            "total_gpu_hours_one_v100": total_seconds / 3600.0,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        },
    }
    (out_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps(profile, indent=2))
    print(f"\nSaved under: {out_dir}")


if __name__ == "__main__":
    main()
