#!/usr/bin/env python3
"""
Cluster SAE features by decoder-direction similarity.

What it does
------------
Given an experiments folder laid out like:

  <exp_root>/<vocab>/<model>/layer_<L>/reason_scores/chunk_XXXX/...

it will:
  1) collect candidate feature IDs + scores from output_scores.json files
  2) load the SAE specified in the chunk config.json (or via CLI overrides)
  3) take each feature's decoder direction (W_dec column), L2-normalize
  4) cluster them with cosine-distance agglomerative clustering
  5) write per-feature assignments + cluster summaries + suggested steering groups

Outputs (in --out_dir)
----------------------
- features_clustered.csv
- cluster_summary.csv
- steering_groups.json

Example
-------
python cluster_sae_decoders.py \
  --exp_root experiments \
  --model_size 9b \
  --vocab clar \
  --layer 31 \
  --topk 50 \
  --distance_threshold 0.25 \
  --top_n_per_cluster 3 \
  --out_dir clusters/9b_clar_l31

Notes
-----
- cosine distance = 1 - cosine similarity
- lower distance_threshold -> more (smaller) clusters
"""

import argparse
import ast
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering


# ----------------------------
# Utilities
# ----------------------------

VOCAB_ALIASES = {
    "clar": "clar_vocab2",
    "clar_vocab2": "clar_vocab2",
    "question": "question",
    "q": "question",
    "combined": "combined",
    "comb": "combined",
}

MODEL_ALIASES = {
    "2b": "gemma-2b-it",
    "gemma-2b-it": "gemma-2b-it",
    "9b": "gemma-2-9b-it",
    "gemma-2-9b-it": "gemma-2-9b-it",
}


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def safe_json_load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_topk_dir(output_scores_dir: Path, requested_topk: int) -> Optional[Path]:
    """
    Returns the folder output_scores/topk_<k> to use.
    If requested doesn't exist, picks the largest available topk_<k>.
    """
    req = output_scores_dir / f"topk_{requested_topk}"
    if req.exists():
        return req
    cands = []
    for p in output_scores_dir.glob("topk_*"):
        try:
            k = int(p.name.split("_")[1])
            cands.append((k, p))
        except Exception:
            continue
    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0][1]


def parse_inspection_txt(path: Path) -> Dict[int, List[str]]:
    """
    Returns mapping: feature_id -> top_tokens_str list
    """
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="replace")
    mapping: Dict[int, List[str]] = {}
    blocks = txt.split("\nFeature ")
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        # blk starts with "<id>:"
        head, *rest = blk.split("\n", 1)
        try:
            fid = int(head.split(":")[0].strip())
        except Exception:
            continue
        m = None
        if rest:
            body = rest[0]
            # line like: top_tokens str = [...]
            for line in body.splitlines():
                if line.strip().startswith("top_tokens str"):
                    m = line.split("=", 1)[1].strip()
                    break
        if m:
            try:
                toks = ast.literal_eval(m)
                if isinstance(toks, list):
                    mapping[fid] = [str(t) for t in toks]
            except Exception:
                pass
    return mapping


# ----------------------------
# SAE loading
# ----------------------------

def load_sae(sae_path: str, sae_id: str, device: str):
    """
    Tries a few SAE Lens APIs, returning an object with .W_dec (decoder weights).
    """
    # SAE Lens has changed APIs across versions. Try common patterns.
    last_err = None
    for import_path in ["sae_lens", "sae_lens.sae", "sae_lens.training.sae"]:
        try:
            mod = __import__(import_path, fromlist=["SAE"])
            SAE = getattr(mod, "SAE")
            # Try from_pretrained signatures
            if hasattr(SAE, "from_pretrained"):
                try:
                    out = SAE.from_pretrained(release=sae_path, sae_id=sae_id, device=device)
                    # sometimes returns (sae, cfg_dict, sparsity); sometimes (sae, cfg)
                    sae = out[0] if isinstance(out, (tuple, list)) else out
                    return sae
                except TypeError:
                    # older signature
                    out = SAE.from_pretrained(sae_path, sae_id, device=device)
                    sae = out[0] if isinstance(out, (tuple, list)) else out
                    return sae
            # Fallback: classmethod load
            if hasattr(SAE, "load"):
                out = SAE.load(sae_path, sae_id, device=device)
                sae = out[0] if isinstance(out, (tuple, list)) else out
                return sae
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        "Could not import/load SAE from sae_lens. "
        "Install sae-lens (SAE Lens) or adjust load_sae() for your setup. "
        f"Last error: {last_err}"
    )


def extract_decoder_matrix(sae) -> torch.Tensor:
    """
    Returns W_dec in shape [n_features, d_model] as a torch tensor on CPU.
    """
    if not hasattr(sae, "W_dec"):
        raise AttributeError("SAE object has no attribute W_dec")
    W = sae.W_dec
    if isinstance(W, np.ndarray):
        W = torch.from_numpy(W)
    if not torch.is_tensor(W):
        W = torch.tensor(W)
    W = W.detach().float().cpu()

    # Common conventions:
    # - [d_model, n_features]  -> transpose
    # - [n_features, d_model]  -> keep
    if W.ndim != 2:
        raise ValueError(f"Unexpected W_dec shape: {tuple(W.shape)}")
    d0, d1 = W.shape
    # Heuristic: width_16k SAEs have n_features=16384, which is usually the larger dim.
    if d1 > d0:
        # likely [d_model, n_features]
        return W.T
    else:
        # likely [n_features, d_model]
        return W


# ----------------------------
# Experiment parsing
# ----------------------------

@dataclass
class RunSpec:
    exp_root: Path
    vocab_dir: str
    model_dir: str
    layer: int

    @property
    def run_dir(self) -> Path:
        return self.exp_root / self.vocab_dir / self.model_dir / f"layer_{self.layer}" / "reason_scores"


def infer_sae_from_run(run_dir: Path) -> Tuple[str, str]:
    """
    Reads chunk_*/config.json and returns (sae_path, sae_id).
    """
    for cfg_path in sorted(run_dir.glob("chunk_*/config.json")):
        cfg = safe_json_load(cfg_path)
        args = cfg.get("args", {})
        sae_path = args.get("sae_path")
        sae_id = args.get("sae_id")
        if sae_path and sae_id:
            return sae_path, sae_id
    raise FileNotFoundError(f"Could not find sae_path/sae_id in any chunk config.json under {run_dir}")


def collect_features(run_dir: Path, topk: int, min_output_score: float, global_top_n: Optional[int]) -> pd.DataFrame:
    """
    Collects candidate features across chunks. Returns a dataframe with:
      feature_id, output_score, rank, prob, chunk, inspection_tokens_str (optional)
    """
    rows = []
    for chunk_dir in sorted(run_dir.glob("chunk_*")):
        out_dir = chunk_dir / "output_scores"
        if not out_dir.exists():
            continue

        topk_dir = pick_topk_dir(out_dir, topk)
        if topk_dir is None:
            continue

        js_path = topk_dir / "output_scores.json"
        if not js_path.exists():
            continue

        data = safe_json_load(js_path)
        for fid_str, obj in data.items():
            try:
                fid = int(fid_str)
            except Exception:
                continue
            oscore = float(obj.get("output_score", 0.0))
            if oscore < min_output_score:
                continue
            rows.append({
                "feature_id": fid,
                "output_score": oscore,
                "rank": int(obj.get("rank", -1)),
                "prob": float(obj.get("prob", float("nan"))),
                "chunk": chunk_dir.name,
                "top_tokens_ids": obj.get("top_tokens", []),
                "output_scores_path": str(js_path),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Merge inspection tokens (strings) where available for interpretation
    # One inspection per chunk topk dir.
    ins_map: Dict[int, List[str]] = {}
    for chunk_dir in sorted(run_dir.glob("chunk_*")):
        out_dir = chunk_dir / "output_scores"
        topk_dir = pick_topk_dir(out_dir, topk)
        if topk_dir is None:
            continue
        ins_path = topk_dir / "inspection.txt"
        ins_map.update(parse_inspection_txt(ins_path))

    df["top_tokens_str"] = df["feature_id"].map(lambda fid: ins_map.get(fid, []))

    # If duplicates exist (unlikely), keep the best output_score
    df = df.sort_values("output_score", ascending=False).drop_duplicates("feature_id")

    if global_top_n is not None and global_top_n > 0:
        df = df.sort_values("output_score", ascending=False).head(global_top_n)

    return df.reset_index(drop=True)


# ----------------------------
# Clustering
# ----------------------------

def cluster_decoder_directions(
    W_dec_nf_dm: np.ndarray,
    feature_ids: np.ndarray,
    distance_threshold: float,
    n_clusters: Optional[int],
) -> np.ndarray:
    """
    W_dec_nf_dm: [n_features_total, d_model] unit-normalized or not
    feature_ids: [n_selected]
    returns: cluster labels [n_selected]
    """
    # Select and normalize
    V = W_dec_nf_dm[feature_ids]  # [n_selected, d_model]
    V = l2_normalize_rows(V)

    if n_clusters is not None and n_clusters > 0:
        cl = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
    else:
        cl = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    labels = cl.fit_predict(V)
    return labels


def make_cluster_summary(df: pd.DataFrame, top_n_per_cluster: int) -> Tuple[pd.DataFrame, dict]:
    """
    Returns (cluster_summary_df, steering_groups_dict)
    """
    groups = []
    steering = {}

    for cid, g in df.groupby("cluster"):
        g2 = g.sort_values("output_score", ascending=False)
        rep = g2.iloc[0]
        steering_feats = g2.head(top_n_per_cluster)["feature_id"].astype(int).tolist()
        steering[str(int(cid))] = steering_feats

        groups.append({
            "cluster": int(cid),
            "size": int(len(g2)),
            "mean_output_score": float(g2["output_score"].mean()),
            "max_output_score": float(g2["output_score"].max()),
            "rep_feature_id": int(rep["feature_id"]),
            "rep_output_score": float(rep["output_score"]),
            "rep_top_tokens_str": rep["top_tokens_str"][:20] if isinstance(rep["top_tokens_str"], list) else [],
            "features_sorted": g2["feature_id"].astype(int).tolist(),
            "steering_features": steering_feats,
        })

    summary = pd.DataFrame(groups).sort_values(["size", "max_output_score"], ascending=[False, False])
    return summary, steering


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", type=str, required=True, help="Path to experiments folder (the one containing vocab subfolders).")
    ap.add_argument("--model_size", type=str, required=True, help="2b or 9b (or full folder name like gemma-2b-it).")
    ap.add_argument("--vocab", type=str, required=True, help="clar / question / combined (or exact folder name).")
    ap.add_argument("--layer", type=int, required=True)

    ap.add_argument("--topk", type=int, default=50, help="Use output_scores/topk_<k>. If missing, picks largest available.")
    ap.add_argument("--min_output_score", type=float, default=0.0)
    ap.add_argument("--global_top_n", type=int, default=0, help="If >0, keep only the global top-N by output_score after unioning chunks.")

    ap.add_argument("--distance_threshold", type=float, default=0.25, help="Cosine distance threshold for agglomerative clustering (ignored if --n_clusters is set).")
    ap.add_argument("--n_clusters", type=int, default=0, help="If >0, fixes number of clusters instead of using distance threshold.")
    ap.add_argument("--top_n_per_cluster", type=int, default=3, help="How many features to suggest per cluster for multi-feature steering.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--sae_path", type=str, default="", help="Override SAE release/path (otherwise inferred from run config.json).")
    ap.add_argument("--sae_id", type=str, default="", help="Override SAE id (otherwise inferred from run config.json).")

    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    vocab_dir = VOCAB_ALIASES.get(args.vocab.lower(), args.vocab)
    model_dir = MODEL_ALIASES.get(args.model_size.lower(), args.model_size)

    spec = RunSpec(exp_root=exp_root, vocab_dir=vocab_dir, model_dir=model_dir, layer=args.layer)
    run_dir = spec.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Collect candidate features
    global_top_n = args.global_top_n if args.global_top_n and args.global_top_n > 0 else None
    df = collect_features(run_dir, topk=args.topk, min_output_score=args.min_output_score, global_top_n=global_top_n)
    if df.empty:
        raise RuntimeError(f"No features found under {run_dir} (check --topk and --min_output_score).")

    # Load SAE
    sae_path = args.sae_path.strip() or None
    sae_id = args.sae_id.strip() or None
    if sae_path is None or sae_id is None:
        inf_path, inf_id = infer_sae_from_run(run_dir)
        sae_path = sae_path or inf_path
        sae_id = sae_id or inf_id

    print(f"[info] Using SAE: sae_path={sae_path}  sae_id={sae_id}")
    print(f"[info] Candidates: {len(df)} features")

    sae = load_sae(sae_path, sae_id, device=args.device)
    W_nf_dm = extract_decoder_matrix(sae).numpy()  # [n_features_total, d_model]

    # Validate feature IDs
    fids = df["feature_id"].to_numpy(dtype=np.int64)
    ok = (fids >= 0) & (fids < W_nf_dm.shape[0])
    if not np.all(ok):
        bad = fids[~ok]
        raise ValueError(f"Some feature IDs are out of range for SAE (n_features={W_nf_dm.shape[0]}). Bad: {bad[:20]}")

    # Cluster
    labels = cluster_decoder_directions(
        W_dec_nf_dm=W_nf_dm,
        feature_ids=fids,
        distance_threshold=args.distance_threshold,
        n_clusters=(args.n_clusters if args.n_clusters and args.n_clusters > 0 else None),
    )
    df = df.copy()
    df["cluster"] = labels.astype(int)

    # Output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = df.sort_values(["cluster", "output_score"], ascending=[True, False])
    df_out.to_csv(out_dir / "features_clustered.csv", index=False)

    summary, steering = make_cluster_summary(df_out, top_n_per_cluster=args.top_n_per_cluster)
    summary.to_csv(out_dir / "cluster_summary.csv", index=False)

    with (out_dir / "steering_groups.json").open("w", encoding="utf-8") as f:
        json.dump(steering, f, indent=2)

    print(f"[done] Wrote:\n  {out_dir/'features_clustered.csv'}\n  {out_dir/'cluster_summary.csv'}\n  {out_dir/'steering_groups.json'}")


if __name__ == "__main__":
    main()
