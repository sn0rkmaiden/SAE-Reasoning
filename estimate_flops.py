#!/usr/bin/env python3
"""Approximate matrix-multiplication FLOPs for ClarifyScore or OutputScore.

Conventions:
- one multiply-add = 2 FLOPs;
- counts transformer linear layers, attention QK/AV products, gated MLP,
  LM head when applicable, and SAE encoder/decoder matrix products;
- omits embeddings, normalization, nonlinearities, masks, softmax, sorting,
  tokenization, and data loading.

The result must be reported as *estimated FLOPs*, not an exact hardware counter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoConfig


def get(cfg: Any, *names: str, default=None):
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if value is not None:
                return value
    return default


def model_dims(cfg: Any) -> dict[str, int]:
    d = int(get(cfg, "hidden_size", "d_model"))
    n_heads = int(get(cfg, "num_attention_heads", "n_heads"))
    n_kv = int(get(cfg, "num_key_value_heads", default=n_heads))
    head_dim = int(get(cfg, "head_dim", default=d // n_heads))
    d_mlp = int(get(cfg, "intermediate_size", "d_mlp"))
    n_layers = int(get(cfg, "num_hidden_layers", "n_layers"))
    vocab = int(get(cfg, "vocab_size", "d_vocab"))
    return {
        "d_model": d,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "d_mlp": d_mlp,
        "n_layers": n_layers,
        "vocab_size": vocab,
    }


def transformer_forward_flops(
    dims: dict[str, int], batch: int, seq: int, layers: int, include_lm_head: bool
) -> dict[str, int]:
    d = dims["d_model"]
    h = dims["n_heads"]
    kv = dims["n_kv_heads"]
    dh = dims["head_dim"]
    m = dims["d_mlp"]
    v = dims["vocab_size"]

    q_dim = h * dh
    kv_dim = kv * dh
    # Q, K, V and output projections.
    attn_proj_per_layer = 2 * batch * seq * d * (q_dim + 2 * kv_dim + q_dim)
    # QK^T and softmax(QK)V. Softmax itself is omitted.
    attn_matmul_per_layer = 4 * batch * h * seq * seq * dh
    # Gemma-style gated MLP: gate_proj, up_proj, down_proj.
    mlp_per_layer = 6 * batch * seq * d * m

    layers_total = layers * (attn_proj_per_layer + attn_matmul_per_layer + mlp_per_layer)
    lm_head = 2 * batch * seq * d * v if include_lm_head else 0
    return {
        "attention_projection_flops": layers * attn_proj_per_layer,
        "attention_qk_av_flops": layers * attn_matmul_per_layer,
        "mlp_flops": layers * mlp_per_layer,
        "lm_head_flops": lm_head,
        "total": layers_total + lm_head,
    }


def human(x: float) -> str:
    units = [(1e18, "EFLOPs"), (1e15, "PFLOPs"), (1e12, "TFLOPs"), (1e9, "GFLOPs")]
    for scale, name in units:
        if abs(x) >= scale:
            return f"{x/scale:.4g} {name}"
    return f"{x:.4g} FLOPs"


def clarify(profile: dict, dims: dict[str, int], hook_layer: int) -> dict:
    w = profile["workload"]
    args = profile["command_arguments"]
    seq = int(w["sequence_length"])
    batch = int(args["minibatch_size_tokens"])
    calls = int(w["full_llm_forward_calls"])
    d_sae = int(w.get("sae_d_sae") or w["n_features_total"])
    d = int(w.get("sae_d_in") or dims["d_model"])
    n_samples = int(w["n_samples_full"])
    act = str(w.get("sae_activation_fn") or "unknown")

    one = transformer_forward_flops(dims, batch, seq, hook_layer + 1, False)
    llm = one["total"] * calls

    if act.lower() == "topk":
        # The code computes all SAE features for every feature minibatch.
        sae_encode = 2 * calls * batch * seq * d * d_sae
        sae_assumption = "TopK branch: full SAE encoding repeated for every feature minibatch"
    else:
        # FeatureMaskingContext computes each feature once across the full chunk sweep.
        sae_encode = 2 * n_samples * seq * d * d_sae
        sae_assumption = "masked SAE branch: each SAE feature encoded once per corpus token"

    total = llm + sae_encode
    return {
        "stage": "ClarifyScore, one vocabulary",
        "convention": "1 multiply-add = 2 FLOPs",
        "included_operations": [
            "transformer Q/K/V/O projections through hook layer",
            "attention QK and AV matrix multiplications",
            "gated MLP projections",
            "SAE encoder matrix multiplication",
        ],
        "omitted_operations": ["normalization", "rotary embeddings", "softmax", "nonlinearities", "mask construction", "entropy/statistics", "I/O"],
        "dimensions": dims,
        "hook_layer_zero_indexed": hook_layer,
        "executed_transformer_layers": hook_layer + 1,
        "sequence_length": seq,
        "token_minibatch_size": batch,
        "llm_forward_calls": calls,
        "sae_assumption": sae_assumption,
        "llm_flops": llm,
        "sae_encode_flops": sae_encode,
        "total_estimated_flops_one_vocabulary": total,
        "total_estimated_flops_two_separate_vocabularies": 2 * total,
        "human_readable": {
            "one_vocabulary": human(total),
            "two_vocabularies": human(2 * total),
        },
    }


def outputscore(profile: dict, dims: dict[str, int]) -> dict:
    w = profile["workload"]
    args = profile["command_arguments"]
    d = int(w.get("d_model") or dims["d_model"])
    d_sae = int(w["d_sae"])
    candidates = int(w["candidate_features"])
    prompt_len = int(w["representative_prompt_token_length"])
    n_batches = int(w["a_max_forward_calls"])
    amax_seq = int(args["train_batch_size_tokens"])

    full_layers = dims["n_layers"]
    amax_one = transformer_forward_flops(dims, 1, amax_seq, full_layers, True)["total"]
    amax_llm = n_batches * amax_one
    amax_sae = 2 * n_batches * amax_seq * d * d_sae

    prompt_model_one = transformer_forward_flops(dims, 1, prompt_len, full_layers, True)["total"]
    prompt_llm = candidates * prompt_model_one
    # Hook computes a full SAE encode and decode for every candidate prompt.
    prompt_sae = candidates * 4 * prompt_len * d * d_sae
    decoder_to_vocab = candidates * 2 * d * dims["vocab_size"]

    total = amax_llm + amax_sae + prompt_llm + prompt_sae + decoder_to_vocab
    return {
        "stage": "OutputScore, one top-k candidate set",
        "convention": "1 multiply-add = 2 FLOPs",
        "included_operations": [
            "full transformer and LM head for a_max batches",
            "SAE encoder for a_max",
            "full transformer and LM head for each intervention prompt",
            "full SAE encoder and decoder inside each intervention hook",
            "decoder-direction to vocabulary projection",
        ],
        "omitted_operations": ["normalization", "softmax", "top-k/sorting", "nonlinearities", "I/O"],
        "dimensions": dims,
        "candidate_features": candidates,
        "prompt_token_length": prompt_len,
        "a_max_batches": n_batches,
        "a_max_sequence_length": amax_seq,
        "a_max_llm_flops": amax_llm,
        "a_max_sae_flops": amax_sae,
        "candidate_prompt_llm_flops": prompt_llm,
        "candidate_prompt_sae_flops": prompt_sae,
        "decoder_to_vocab_flops": decoder_to_vocab,
        "total_estimated_flops": total,
        "human_readable": human(total),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["clarifyscore", "outputscore"], required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model_config", required=True, help="e.g. google/gemma-2b-it")
    ap.add_argument("--hook_layer", type=int, default=None, help="required for ClarifyScore")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    cfg = AutoConfig.from_pretrained(args.model_config)
    dims = model_dims(cfg)

    if args.kind == "clarifyscore":
        if args.hook_layer is None:
            raise ValueError("--hook_layer is required for ClarifyScore")
        result = clarify(profile, dims, args.hook_layer)
    else:
        result = outputscore(profile, dims)

    result["model_config"] = args.model_config
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
