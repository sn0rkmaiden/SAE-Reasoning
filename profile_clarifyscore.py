#!/usr/bin/env python3
"""Profile a representative ClarifyScore/ReasonScore run and extrapolate full cost.

Run this script from the root of SAE-Reasoning-main so that
`extraction.compute_score` and the modified TransformerLens are importable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(i),
        "total_memory_bytes": int(p.total_memory),
        "compute_capability": f"{p.major}.{p.minor}",
        "bf16_supported_by_pytorch": bool(torch.cuda.is_bf16_supported()),
        "cuda_runtime": torch.version.cuda,
    }


def split_sizes(n: int, parts: int) -> list[int]:
    k, m = divmod(n, parts)
    return [k + (1 if i < m else 0) for i in range(parts)]


def newest_config(root: Path, after: float) -> Path | None:
    candidates = [p for p in root.rglob("config.json") if p.stat().st_mtime >= after - 2]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--sae_path", required=True)
    ap.add_argument("--sae_id", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--tokens_str_path", required=True)
    ap.add_argument("--vocab_name", required=True)
    ap.add_argument("--n_samples", type=int, required=True, choices=[128, 256, 512, 1024])
    ap.add_argument("--full_n_samples", type=int, default=4096)
    ap.add_argument("--column_name", default="tokens")
    ap.add_argument("--minibatch_size_features", type=int, default=256)
    ap.add_argument("--minibatch_size_tokens", type=int, default=16)
    ap.add_argument("--num_chunks", type=int, default=20)
    ap.add_argument("--chunk_num", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--exp_root", default="cost_profiles/clarifyscore_runs")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    # Import after parsing so a missing repository dependency gives a clear location hint.
    try:
        from extraction import compute_score as score_module
    except Exception as e:
        raise RuntimeError(
            "Run this script from SAE-Reasoning-main (or add that directory to PYTHONPATH)."
        ) from e

    state: dict[str, float] = {}
    original_run = score_module.SaeSelectionRunner.run

    def timed_run(*run_args, **run_kwargs):
        encoder = run_kwargs.get("encoder")
        if encoder is None and len(run_args) >= 2:
            encoder = run_args[1]
        if encoder is not None:
            state["sae_activation_fn"] = type(getattr(encoder, "activation_fn", None)).__name__
            state["d_sae"] = int(getattr(getattr(encoder, "cfg", None), "d_sae", 0))
            state["d_in"] = int(getattr(getattr(encoder, "cfg", None), "d_in", 0))
        tokens = run_kwargs.get("tokens")
        if tokens is None and len(run_args) >= 4:
            tokens = run_args[3]
        if tokens is not None:
            state["token_rows"] = int(tokens.shape[0])
            state["sequence_length"] = int(tokens.shape[1])
            state["padded_token_positions_profile"] = int(tokens.numel())
        sync()
        t0 = time.perf_counter()
        result = original_run(*run_args, **run_kwargs)
        sync()
        state["score_compute_seconds"] = time.perf_counter() - t0
        return result

    score_module.SaeSelectionRunner.run = timed_run

    out_json = Path(args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    exp_root = Path(args.exp_root).resolve()
    exp_root.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started_epoch = time.time()
    sync()
    total_t0 = time.perf_counter()
    try:
        score_module.compute_score(
            model_path=args.model_path,
            sae_path=args.sae_path,
            dataset_path=args.dataset_path,
            tokens_str_path=str(Path(args.tokens_str_path).resolve()),
            exp_root=str(exp_root),
            vocab_name=args.vocab_name,
            sae_id=args.sae_id,
            n_samples=args.n_samples,
            alpha=args.alpha,
            column_name=args.column_name,
            minibatch_size_features=args.minibatch_size_features,
            minibatch_size_tokens=args.minibatch_size_tokens,
            num_chunks=args.num_chunks,
            chunk_num=args.chunk_num,
        )
    finally:
        score_module.SaeSelectionRunner.run = original_run
    sync()
    total_seconds = time.perf_counter() - total_t0

    config_path = newest_config(exp_root, started_epoch)
    config: dict[str, Any] = {}
    if config_path is not None:
        config = json.loads(config_path.read_text(encoding="utf-8"))

    derived = config.get("derived", {})
    n_features_total = int(derived.get("num_features_total", 16384))
    n_features_chunk = int(derived.get("num_features_in_chunk", 0))
    if not n_features_chunk:
        n_features_chunk = split_sizes(n_features_total, args.num_chunks)[args.chunk_num]

    profile_token_batches = math.ceil(args.n_samples / args.minibatch_size_tokens)
    profile_feature_batches = math.ceil(n_features_chunk / args.minibatch_size_features)
    profile_forward_calls = profile_token_batches * profile_feature_batches

    all_chunk_sizes = split_sizes(n_features_total, args.num_chunks)
    full_feature_batches = sum(
        math.ceil(size / args.minibatch_size_features) for size in all_chunk_sizes
    )
    full_token_batches = math.ceil(args.full_n_samples / args.minibatch_size_tokens)
    full_forward_calls = full_feature_batches * full_token_batches
    scale = full_forward_calls / profile_forward_calls

    compute_seconds = float(state.get("score_compute_seconds", total_seconds))
    setup_io_seconds = max(0.0, total_seconds - compute_seconds)

    # The historical code launches one process per chunk, so setup/load happens 20 times.
    historical_compute_seconds = compute_seconds * scale
    historical_setup_seconds = setup_io_seconds * args.num_chunks
    historical_end_to_end_seconds = historical_compute_seconds + historical_setup_seconds

    # This is useful if all chunks are later processed in one process with one model load.
    one_load_end_to_end_seconds = historical_compute_seconds + setup_io_seconds

    result = {
        "kind": "clarifyscore_profile",
        "command_arguments": vars(args),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": gpu_metadata(),
            "pid": os.getpid(),
        },
        "measured": {
            "total_seconds": total_seconds,
            "score_compute_seconds": compute_seconds,
            "setup_dataset_and_io_seconds": setup_io_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        },
        "workload": {
            "n_samples_profile": args.n_samples,
            "n_samples_full": args.full_n_samples,
            "n_features_total": n_features_total,
            "n_features_profile_chunk": n_features_chunk,
            "sae_activation_fn": state.get("sae_activation_fn"),
            "sae_d_in": int(state.get("d_in", 0)),
            "sae_d_sae": int(state.get("d_sae", n_features_total)),
            "num_chunks": args.num_chunks,
            "profile_token_minibatches": profile_token_batches,
            "profile_feature_minibatches": profile_feature_batches,
            "profile_llm_forward_calls": profile_forward_calls,
            "sequence_length": int(state.get("sequence_length", 0)),
            "padded_token_positions_profile": int(state.get("padded_token_positions_profile", 0)),
            "padded_token_positions_full_corpus_once": int(args.full_n_samples * state.get("sequence_length", 0)),
            "full_token_minibatches": full_token_batches,
            "full_feature_minibatches_across_chunks": full_feature_batches,
            "full_llm_forward_calls": full_forward_calls,
            "compute_scale_factor": scale,
        },
        "estimated_full_one_vocabulary_single_v100": {
            "historical_separate_chunk_jobs_seconds": historical_end_to_end_seconds,
            "historical_separate_chunk_jobs_gpu_hours": historical_end_to_end_seconds / 3600.0,
            "compute_only_seconds": historical_compute_seconds,
            "compute_only_gpu_hours": historical_compute_seconds / 3600.0,
            "one_model_load_seconds": one_load_end_to_end_seconds,
            "one_model_load_gpu_hours": one_load_end_to_end_seconds / 3600.0,
        },
        "generated_score_config": str(config_path) if config_path else None,
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved profile: {out_json}")


if __name__ == "__main__":
    main()
