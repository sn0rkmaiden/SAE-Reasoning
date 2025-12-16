import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

"""
Example usage:

    python plot_outputscore_hist.py \
  --roots experiments/question/gemma-2-9b-it \
  --bins 50 \
  --out outputscore_hist.png
  --logy

"""


def iter_output_score_files(roots, pattern="output_scores.json"):
    for root in roots:
        root = Path(root)
        if root.is_file() and root.name == pattern:
            yield root
        elif root.is_dir():
            yield from root.rglob(pattern)


def extract_scores(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))

    scores = []
    # Your format is typically: {"<feature_id>": {"output_score": float, ...}, ...}
    if isinstance(obj, dict):
        for _, v in obj.items():
            if isinstance(v, dict) and "output_score" in v:
                scores.append(float(v["output_score"]))
    # Allow list-of-dicts format too (just in case)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict) and "output_score" in v:
                scores.append(float(v["output_score"]))

    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Directories (or specific output_scores.json files) to scan.")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--title", default="Distribution of OutputScore")
    ap.add_argument("--out", default="outputscore_hist.png")
    ap.add_argument("--min", type=float, default=None, help="Optional lower cutoff")
    ap.add_argument("--max", type=float, default=None, help="Optional upper cutoff")
    ap.add_argument("--logy", action="store_true", help="Log scale on Y axis")
    args = ap.parse_args()

    all_scores = []
    files = list(iter_output_score_files(args.roots))
    if not files:
        raise SystemExit("No output_scores.json files found under given roots.")

    for f in files:
        all_scores.extend(extract_scores(f))

    if args.min is not None:
        all_scores = [x for x in all_scores if x >= args.min]
    if args.max is not None:
        all_scores = [x for x in all_scores if x <= args.max]

    if not all_scores:
        raise SystemExit("No scores found after filtering.")

    plt.figure(figsize=(10, 5))
    plt.hist(all_scores, bins=args.bins)
    plt.title(args.title)
    plt.xlabel("OutputScore")
    plt.ylabel("Количество признаков")
    plt.grid(True, alpha=0.3)
    if args.logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[ok] saved {args.out}")
    print(f"[info] files={len(files)} scores={len(all_scores)}")


if __name__ == "__main__":
    main()
