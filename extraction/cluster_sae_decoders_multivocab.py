#!/usr/bin/env python3
"""
Cluster SAE features by decoder-direction similarity across *multiple vocabs* for a given model+layer,
using ONLY the features that appear in each topk folder's inspection.txt and have output_score > 0.

This matches the typical workflow where:
- inspection.txt defines "the features you found / looked at"
- output_scores.json provides the Output Score used for filtering and ranking

Folder layout expected
----------------------
<exp_root>/<vocab>/<model>/layer_<L>/reason_scores/chunk_XXXX/output_scores/topk_<K>/output_scores.json
and inspection.txt in the same topk folder.

Vocab folder names (aliases)
----------------------------
- clar      -> clar_vocab2
- question  -> question
- combined  -> combined

Model folder names (aliases)
----------------------------
- 2b -> gemma-2b-it
- 9b -> gemma-2-9b-it

Outputs (in --out_dir)
----------------------
- features_clustered.csv
- cluster_summary.csv
- steering_groups.json
- clusters.md               (human-readable cluster report)

Example
-------
python cluster_sae_decoders_multivocab.py \
  --exp_root experiments \
  --model_size 9b \
  --vocabs all \
  --layer 20 \
  --topk 50 \
  --distance_threshold 0.55 \
  --top_n_per_cluster 3 \
  --out_dir clusters/9b_allvocabs_l20
"""

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering


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


def safe_json_load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def cosine_distance_quantiles(V: np.ndarray) -> Dict[str, float]:
    """
    V: [n, d] unit-normalized rows.
    Returns quantiles of cosine distance (1 - cosine sim).
    Uses full pairwise if n<=1200, else random sample pairs.
    """
    n = V.shape[0]
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    if n <= 1200:
        sims = V @ V.T
        iu = np.triu_indices(n, k=1)
        dists = 1.0 - sims[iu]
    else:
        rng = np.random.default_rng(0)
        m = min(2_000_000, n * 200)  # cap
        i = rng.integers(0, n, size=m)
        j = rng.integers(0, n, size=m)
        mask = i != j
        i, j = i[mask], j[mask]
        sims = (V[i] * V[j]).sum(axis=1)
        dists = 1.0 - sims
    out = {f"q{int(q*100):02d}": float(np.quantile(dists, q)) for q in qs}
    return out


def pick_topk_dir(output_scores_dir: Path, requested_topk: int) -> Optional[Path]:
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


# ---------- Robust inspection parsing ----------
INS_BLOCK_RE = re.compile(r"(?:^|\n)Feature\s+(\d+):\s*(.*?)(?=\nFeature\s+\d+:|\Z)", re.S)
TOK_STR_RE = re.compile(r"top_tokens\s+str\s*=\s*(\[[^\]]*\])", re.S)
TOK_ID_RE = re.compile(r"top_tokens\s+IDs\s*=\s*(\[[^\]]*\])", re.S)

def parse_inspection_txt(path: Path) -> Dict[int, Dict[str, list]]:
    """
    Returns mapping:
      feature_id -> {"top_tokens_str": [...], "top_tokens_ids": [...]}
    """
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="replace")
    out: Dict[int, Dict[str, list]] = {}
    for m in INS_BLOCK_RE.finditer(txt):
        fid = int(m.group(1))
        body = m.group(2)

        toks_str = []
        toks_ids = []

        ms = TOK_STR_RE.search(body)
        if ms:
            try:
                toks_str = ast.literal_eval(ms.group(1))
            except Exception:
                toks_str = []
        mi = TOK_ID_RE.search(body)
        if mi:
            try:
                toks_ids = ast.literal_eval(mi.group(1))
            except Exception:
                toks_ids = []

        out[fid] = {"top_tokens_str": toks_str if isinstance(toks_str, list) else [],
                    "top_tokens_ids": toks_ids if isinstance(toks_ids, list) else []}
    return out


# ---------- SAE loading ----------
def load_sae(sae_path: str, sae_id: str, device: str):
    """
    Tries common SAE Lens APIs, returning an object with .W_dec.
    """
    last_err = None
    for import_path in ["sae_lens", "sae_lens.sae", "sae_lens.training.sae"]:
        try:
            mod = __import__(import_path, fromlist=["SAE"])
            SAE = getattr(mod, "SAE")

            if hasattr(SAE, "from_pretrained"):
                try:
                    out = SAE.from_pretrained(release=sae_path, sae_id=sae_id, device=device)
                    sae = out[0] if isinstance(out, (tuple, list)) else out
                    return sae
                except TypeError:
                    out = SAE.from_pretrained(sae_path, sae_id, device=device)
                    sae = out[0] if isinstance(out, (tuple, list)) else out
                    return sae

            if hasattr(SAE, "load"):
                out = SAE.load(sae_path, sae_id, device=device)
                sae = out[0] if isinstance(out, (tuple, list)) else out
                return sae

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Could not import/load SAE from sae_lens. "
        "Install sae-lens (SAE Lens) or adapt load_sae() to your loader. "
        f"Last error: {last_err}"
    )


def extract_decoder_matrix(sae) -> torch.Tensor:
    """
    Returns W_dec as torch.Tensor with shape [n_features, d_model] on CPU.
    """
    if not hasattr(sae, "W_dec"):
        raise AttributeError("SAE object has no attribute W_dec")
    W = sae.W_dec
    if isinstance(W, np.ndarray):
        W = torch.from_numpy(W)
    if not torch.is_tensor(W):
        W = torch.tensor(W)
    W = W.detach().float().cpu()
    if W.ndim != 2:
        raise ValueError(f"Unexpected W_dec shape: {tuple(W.shape)}")
    d0, d1 = W.shape
    # Heuristic: width_16k SAEs: n_features likely the larger dimension.
    if d1 > d0:
        return W.T
    return W


# ---------- Experiment parsing ----------
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
    Reads first chunk_*/config.json and returns (sae_path, sae_id).
    """
    for cfg_path in sorted(run_dir.glob("chunk_*/config.json")):
        cfg = safe_json_load(cfg_path)
        args = cfg.get("args", {})
        sae_path = args.get("sae_path")
        sae_id = args.get("sae_id")
        if sae_path and sae_id:
            return sae_path, sae_id
    raise FileNotFoundError(f"Could not find sae_path/sae_id in any chunk config.json under {run_dir}")


def collect_features_for_vocab(
    run_dir: Path,
    vocab_name: str,
    topk: int,
    min_output_score: float,
    require_in_inspection: bool,
) -> pd.DataFrame:
    """
    Collects features from output_scores.json but keeps ONLY those that also appear in inspection.txt
    (if require_in_inspection=True) and satisfy output_score > min_output_score.

    With default min_output_score=0.0 and a strict comparison, this enforces Output Score > 0.
    """
    rows = []

    for chunk_dir in sorted(run_dir.glob("chunk_*")):
        out_dir = chunk_dir / "output_scores"
        if not out_dir.exists():
            continue

        topk_dir = pick_topk_dir(out_dir, topk)
        if topk_dir is None:
            continue

        ins_map = parse_inspection_txt(topk_dir / "inspection.txt")  # feature_id -> token previews

        js_path = topk_dir / "output_scores.json"
        if not js_path.exists():
            continue
        data = safe_json_load(js_path)

        for fid_str, obj in data.items():
            try:
                fid = int(fid_str)
            except Exception:
                continue

            if require_in_inspection and fid not in ins_map:
                continue

            oscore = float(obj.get("output_score", 0.0))
            # Strictly greater than (min_output_score); with default 0.0 this means Output Score > 0.
            if oscore <= min_output_score:
                continue

            ins = ins_map.get(fid, {})
            rows.append({
                "feature_id": fid,
                "output_score": oscore,
                "rank": int(obj.get("rank", -1)),
                "prob": float(obj.get("prob", float("nan"))),
                "chunk": chunk_dir.name,
                "vocab": vocab_name,
                "top_tokens_ids": obj.get("top_tokens", ins.get("top_tokens_ids", [])),
                "top_tokens_str": ins.get("top_tokens_str", []),
                "output_scores_path": str(js_path),
                "inspection_path": str(topk_dir / "inspection.txt"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # within-vocab dedup: keep best score per feature
    df = df.sort_values("output_score", ascending=False).drop_duplicates("feature_id")
    return df.reset_index(drop=True)


def union_across_vocabs(
    dfs: List[pd.DataFrame],
    aggregate: str = "max"
) -> pd.DataFrame:
    """
    Merge per-vocab tables into a single per-feature table.

    Keeps:
      - output_score_agg (max or mean across vocabs where it appears)
      - vocabs_present (list)
      - per_vocab_scores (json string)
      - top_tokens_str (first non-empty)
    """
    df_all = pd.concat([d for d in dfs if d is not None and not d.empty], ignore_index=True)
    if df_all.empty:
        return df_all

    def agg_scores(g: pd.DataFrame) -> float:
        if aggregate == "mean":
            return float(g["output_score"].mean())
        return float(g["output_score"].max())

    out_rows = []
    for fid, g in df_all.groupby("feature_id"):
        g = g.sort_values("output_score", ascending=False)
        vocabs = g["vocab"].tolist()
        per_vocab = {r["vocab"]: float(r["output_score"]) for _, r in g.iterrows()}
        toks = []
        for _, r in g.iterrows():
            if isinstance(r.get("top_tokens_str", None), list) and len(r["top_tokens_str"]) > 0:
                toks = r["top_tokens_str"]
                break
        out_rows.append({
            "feature_id": int(fid),
            "output_score_agg": agg_scores(g),
            "best_output_score": float(g["output_score"].iloc[0]),
            "best_vocab": str(g["vocab"].iloc[0]),
            "vocabs_present": ",".join(sorted(set(vocabs))),
            "per_vocab_scores_json": json.dumps(per_vocab),
            "top_tokens_str": toks,
        })

    out = pd.DataFrame(out_rows).sort_values("output_score_agg", ascending=False).reset_index(drop=True)
    return out


# ---------- Cluster labeling ----------
TOKEN_KEEP_RE = re.compile(r"[A-Za-z0-9]")

def clean_token(t: str) -> Optional[str]:
    if t is None:
        return None
    s = str(t).strip()
    if not s:
        return None
    # remove purely punctuation/whitespace
    if not TOKEN_KEEP_RE.search(s):
        return None
    # trim very long weird tokens
    if len(s) > 40:
        return None
    return s

def cluster_keywords(df: pd.DataFrame, topn: int = 8) -> Dict[int, List[str]]:
    """
    Build TF-IDF-ish keywords per cluster using top_tokens_str lists.
    """
    df = df.copy()
    df["clean_tokens"] = df["top_tokens_str"].apply(
        lambda toks: [ct for ct in (clean_token(x) for x in (toks if isinstance(toks, list) else [])) if ct]
    )

    clusters = sorted(df["cluster"].unique().tolist())
    nC = len(clusters)

    dfreq: Dict[str, int] = {}
    docs: Dict[int, List[str]] = {}
    for cid in clusters:
        toks = []
        for lst in df.loc[df["cluster"] == cid, "clean_tokens"]:
            toks.extend(lst)
        docs[cid] = toks
        for tok in set(toks):
            dfreq[tok] = dfreq.get(tok, 0) + 1

    out: Dict[int, List[str]] = {}
    for cid in clusters:
        toks = docs[cid]
        if not toks:
            out[int(cid)] = []
            continue
        counts: Dict[str, int] = {}
        for tok in toks:
            counts[tok] = counts.get(tok, 0) + 1
        total = sum(counts.values())

        scores = []
        for tok, c in counts.items():
            tf = c / max(1, total)
            idf = math.log((nC + 1) / (dfreq.get(tok, 1) + 1)) + 1.0
            scores.append((tf * idf, tok))
        scores.sort(reverse=True)
        out[int(cid)] = [tok for _, tok in scores[:topn]]
    return out


# ---------- Clustering ----------
def cluster_decoder_directions(
    W_nf_dm: np.ndarray,
    fids: np.ndarray,
    distance_threshold: float,
    n_clusters: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (labels, V) where V are the selected, unit-normalized decoder directions.
    """
    V = W_nf_dm[fids]
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
    return labels.astype(int), V


def write_markdown_report(
    out_path: Path,
    df_feat: pd.DataFrame,
    df_sum: pd.DataFrame,
):
    lines = []
    lines.append("# SAE decoder-direction clusters (multi-vocab)\n\n")
    lines.append(f"Total features clustered: **{len(df_feat)}**\n\n")
    lines.append(f"Total clusters: **{df_feat['cluster'].nunique()}**\n\n")

    for _, row in df_sum.sort_values(["size", "max_output_score"], ascending=[False, False]).iterrows():
        cid = int(row["cluster"])
        label = row.get("label", "")
        lines.append(f"\n## Cluster {cid}" + (f" — {label}" if label else "") + "\n\n")
        lines.append(f"- size: {int(row['size'])}\n")
        lines.append(f"- max output_score_agg: {row['max_output_score']:.6g}\n")
        lines.append(f"- representative feature: {int(row['rep_feature_id'])}\n")
        kws = row.get("keywords", "")
        if isinstance(kws, str) and kws:
            lines.append(f"- keywords: {kws}\n")

        top = df_feat[df_feat["cluster"] == cid].sort_values("output_score_agg", ascending=False).head(10)
        lines.append("\nTop features:\n")
        for _, fr in top.iterrows():
            toks = fr["top_tokens_str"]
            toks_preview = ""
            if isinstance(toks, list) and toks:
                toks_preview = ", ".join([str(t) for t in toks[:12]])
            lines.append(
                f"- f{int(fr['feature_id'])}  score={fr['output_score_agg']:.6g}  vocabs={fr['vocabs_present']}  toks=[{toks_preview}]\n"
            )

    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", type=str, required=True, help="Path to experiments folder (contains vocab subfolders).")
    ap.add_argument("--model_size", type=str, required=True, help="2b or 9b (or full folder name).")
    ap.add_argument("--layer", type=int, required=True)

    ap.add_argument("--vocabs", type=str, nargs="+", default=["all"],
                    help="Which vocabs to include: clar question combined, or 'all'.")

    ap.add_argument("--topk", type=int, default=50,
                    help="Use output_scores/topk_<k>. If missing, picks largest available per chunk.")
    ap.add_argument("--min_output_score", type=float, default=0.0,
                    help="Strict threshold; features kept only if output_score > min_output_score. Default 0.0 => Output Score > 0.")
    ap.add_argument("--require_in_inspection", action="store_true",
                    help="Keep only features that appear in inspection.txt (recommended). Default: ON.")
    ap.set_defaults(require_in_inspection=True)

    ap.add_argument("--aggregate", type=str, default="max", choices=["max", "mean"],
                    help="How to aggregate output_score across vocabs for the same feature.")

    ap.add_argument("--global_top_n", type=int, default=0,
                    help="If >0, keep only the global top-N by output_score_agg after unioning vocabs.")
    ap.add_argument("--distance_threshold", type=float, default=0.55,
                    help="Cosine distance threshold (ignored if --n_clusters > 0).")
    ap.add_argument("--n_clusters", type=int, default=0, help="If >0, fixes number of clusters.")
    ap.add_argument("--top_n_per_cluster", type=int, default=3,
                    help="How many features to suggest per cluster for steering_groups.json.")
    ap.add_argument("--keywords_topn", type=int, default=8, help="Number of keywords to surface per cluster.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--sae_path", type=str, default="", help="Override SAE release/path (otherwise inferred from run config.json).")
    ap.add_argument("--sae_id", type=str, default="", help="Override SAE id (otherwise inferred from run config.json).")

    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    model_dir = MODEL_ALIASES.get(args.model_size.lower(), args.model_size)

    voc_in = [v.lower() for v in args.vocabs]
    if len(voc_in) == 1 and voc_in[0] == "all":
        vocab_dirs = ["clar_vocab2", "question", "combined"]
    else:
        vocab_dirs = [VOCAB_ALIASES.get(v, v) for v in voc_in]

    per_vocab_dfs = []
    run_dirs = []
    for vdir in vocab_dirs:
        run_dir = exp_root / vdir / model_dir / f"layer_{args.layer}" / "reason_scores"
        if not run_dir.exists():
            print(f"[warn] Missing run dir (skipping): {run_dir}")
            continue
        run_dirs.append(run_dir)
        df_v = collect_features_for_vocab(
            run_dir,
            vocab_name=vdir,
            topk=args.topk,
            min_output_score=args.min_output_score,
            require_in_inspection=args.require_in_inspection,
        )
        if df_v.empty:
            print(f"[warn] No features found under {run_dir} after filtering (inspection+output_score)")
        else:
            print(f"[info] {vdir}: {len(df_v)} features (after filtering & per-vocab dedup)")
        per_vocab_dfs.append(df_v)

    if not per_vocab_dfs:
        raise RuntimeError("No valid vocab runs found. Check --exp_root, --model_size, --layer, and --vocabs.")

    df = union_across_vocabs(per_vocab_dfs, aggregate=args.aggregate)
    if df.empty:
        raise RuntimeError("Union across vocabs produced 0 features. Try lowering --min_output_score or disabling --require_in_inspection.")

    if args.global_top_n and args.global_top_n > 0:
        df = df.sort_values("output_score_agg", ascending=False).head(args.global_top_n).reset_index(drop=True)

    print(f"[info] Unioned candidates: {len(df)} features across vocabs")

    sae_path = args.sae_path.strip() or None
    sae_id = args.sae_id.strip() or None
    if sae_path is None or sae_id is None:
        inf_path, inf_id = infer_sae_from_run(run_dirs[0])
        sae_path = sae_path or inf_path
        sae_id = sae_id or inf_id
    print(f"[info] Using SAE: sae_path={sae_path}  sae_id={sae_id}")

    sae = load_sae(sae_path, sae_id, device=args.device)
    W_nf_dm = extract_decoder_matrix(sae).numpy()

    fids = df["feature_id"].to_numpy(dtype=np.int64)
    ok = (fids >= 0) & (fids < W_nf_dm.shape[0])
    if not np.all(ok):
        bad = fids[~ok]
        raise ValueError(f"Some feature IDs out of range for SAE (n_features={W_nf_dm.shape[0]}). Bad: {bad[:20]}")

    n_clusters = args.n_clusters if args.n_clusters and args.n_clusters > 0 else None
    labels, V = cluster_decoder_directions(
        W_nf_dm=W_nf_dm,
        fids=fids,
        distance_threshold=args.distance_threshold,
        n_clusters=n_clusters,
    )
    df = df.copy()
    df["cluster"] = labels.astype(int)

    q = cosine_distance_quantiles(V)
    qstr = ", ".join([f"{k}={v:.3f}" for k, v in q.items()])
    print(f"[diag] Cosine-distance quantiles among selected features: {qstr}")
    print(f"[diag] Clustering uses distance_threshold={args.distance_threshold} (or n_clusters={n_clusters})")

    kw = cluster_keywords(df, topn=args.keywords_topn)

    summary_rows = []
    steering_groups = {}
    for cid, g in df.groupby("cluster"):
        g2 = g.sort_values("output_score_agg", ascending=False)
        rep = g2.iloc[0]
        keywords = kw.get(int(cid), [])
        label = " / ".join(keywords[:3]) if keywords else ""
        steering = g2.head(args.top_n_per_cluster)["feature_id"].astype(int).tolist()
        steering_groups[str(int(cid))] = steering

        summary_rows.append({
            "cluster": int(cid),
            "label": label,
            "keywords": ", ".join(keywords),
            "size": int(len(g2)),
            "mean_output_score": float(g2["output_score_agg"].mean()),
            "max_output_score": float(g2["output_score_agg"].max()),
            "rep_feature_id": int(rep["feature_id"]),
            "rep_output_score": float(rep["output_score_agg"]),
            "rep_vocabs_present": rep["vocabs_present"],
            "rep_top_tokens_preview": ", ".join([str(t) for t in (rep["top_tokens_str"][:16] if isinstance(rep["top_tokens_str"], list) else [])]),
            "steering_features": json.dumps(steering),
        })

    df_sum = pd.DataFrame(summary_rows).sort_values(["size", "max_output_score"], ascending=[False, False]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.sort_values(["cluster", "output_score_agg"], ascending=[True, False]).to_csv(out_dir / "features_clustered.csv", index=False)
    df_sum.to_csv(out_dir / "cluster_summary.csv", index=False)
    with (out_dir / "steering_groups.json").open("w", encoding="utf-8") as f:
        json.dump(steering_groups, f, indent=2)

    write_markdown_report(out_dir / "clusters.md", df, df_sum)

    print(f"[done] Wrote:\n  {out_dir/'features_clustered.csv'}\n  {out_dir/'cluster_summary.csv'}\n  {out_dir/'steering_groups.json'}\n  {out_dir/'clusters.md'}")


if __name__ == "__main__":
    main()
