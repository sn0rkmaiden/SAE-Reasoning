import argparse
import json
from pathlib import Path
import re

import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

import torch
import matplotlib.pyplot as plt



"""
Usage examples:

python plot_reason_vs_output.py \
  --pair experiments/.../chunk_0000/feature_scores.pt experiments/.../chunk_0000/output_scores/topk_50/output_scores.json \
  --pair experiments/.../chunk_0001/feature_scores.pt experiments/.../chunk_0001/output_scores/topk_50/output_scores.json \
  --pair experiments/.../chunk_0002/feature_scores.pt experiments/.../chunk_0002/output_scores/topk_50/output_scores.json \
  --use_only_output_features \
  --grid_cols 3 \
  --out grid_reason_vs_output.png

"""


def extract_metadata(reason_path: str, output_path: str) -> str:
    full = f"{reason_path} {output_path}"
    def grab(pattern):
        m = re.search(pattern, full)
        return m.group(1) if m else None

    model = grab(r"experiments/[^/]+/([^/]+)/")
    layer = grab(r"layer_(\d+)")
    chunk = grab(r"chunk_(\d+)")
    topk  = grab(r"topk_(\d+)")

    parts = []
    if model: parts.append(model)
    if layer: parts.append(f"Layer {layer}")
    if chunk: parts.append(f"Chunk {chunk}")
    if topk:  parts.append(f"Top-{topk}")
    return " · ".join(parts) if parts else "ReasonScore vs OutputScore"


def load_points(reason_pt: str, output_json: str, use_only_output_features: bool):
    reason_scores = torch.load(reason_pt, map_location="cpu", weights_only=True)
    if isinstance(reason_scores, dict) and "scores" in reason_scores:
        reason_scores = reason_scores["scores"]
    reason_scores = reason_scores.detach().cpu().float()

    output = json.loads(Path(output_json).read_text(encoding="utf-8"))

    xs, ys = [], []
    if use_only_output_features:
        for k, v in output.items():
            i = int(k)
            if 0 <= i < len(reason_scores) and isinstance(v, dict) and "output_score" in v:
                xs.append(float(reason_scores[i].item()))
                ys.append(float(v["output_score"]))
    else:
        for i in range(len(reason_scores)):
            v = output.get(str(i))
            if v is None:
                continue
            xs.append(float(reason_scores[i].item()))
            ys.append(float(v["output_score"]))

    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason_pt", help="Single plot: feature_scores.pt")
    ap.add_argument("--output_json", help="Single plot: output_scores.json")
    ap.add_argument(
        "--pair", action="append", nargs=2, metavar=("REASON_PT", "OUTPUT_JSON"),
        help="Add a subplot: provide REASON_PT OUTPUT_JSON. Can be used multiple times."
    )
    ap.add_argument("--use_only_output_features", action="store_true")
    ap.add_argument("--grid_cols", type=int, default=3)
    ap.add_argument("--out", default="reason_vs_output.pdf")
    ap.add_argument("--share_axes", action="store_true", help="Share x/y axes across subplots (recommended).")
    args = ap.parse_args()

    # Build list of pairs
    pairs = []
    if args.pair:
        pairs = args.pair
    elif args.reason_pt and args.output_json:
        pairs = [(args.reason_pt, args.output_json)]
    else:
        raise SystemExit("Provide either --reason_pt + --output_json OR one/more --pair arguments.")

    # Load all points first (lets us share axis limits consistently)
    all_data = []
    x_all, y_all = [], []
    for reason_pt, output_json in pairs:
        xs, ys = load_points(reason_pt, output_json, args.use_only_output_features)
        if not xs:
            raise RuntimeError(f"No overlapping features for:\n  {reason_pt}\n  {output_json}")
        all_data.append((reason_pt, output_json, xs, ys))
        x_all.extend(xs)
        y_all.extend(ys)

    n = len(all_data)
    cols = max(1, args.grid_cols)
    rows = (n + cols - 1) // cols

    figsize = (cols * 6.0, rows * 4.8)
    fig, axes = plt.subplots(rows, cols, figsize=figsize,
                             sharex=args.share_axes, sharey=args.share_axes)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # Set common limits if sharing (so comparisons are meaningful)
    if args.share_axes:
        xmin, xmax = min(x_all), max(x_all)
        ymin, ymax = min(y_all), max(y_all)

    # Draw
    for idx, (reason_pt, output_json, xs, ys) in enumerate(all_data):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.scatter(xs, ys, s=28)
        ax.set_title(extract_metadata(reason_pt, output_json))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("ReasonScore")
        ax.set_ylabel("OutputScore")
        if args.share_axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    # Hide unused axes if grid has empty slots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[ok] saved {args.out} with {n} subplot(s)")


if __name__ == "__main__":
    main()
