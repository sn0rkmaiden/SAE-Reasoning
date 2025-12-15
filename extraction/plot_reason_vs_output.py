import argparse
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt


"""
Usage example:

python plot_reason_vs_output.py \
  --reason_pt path/to/reason_scores/.../feature_scores.pt \
  --output_json path/to/reason_scores/.../output_scores/topk_XXX/output_scores.json \
  --use_only_output_features \
  --out reason_vs_output.png

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason_pt", required=True, help="Path to feature_scores.pt (ReasonScore tensor)")
    ap.add_argument("--output_json", required=True, help="Path to output_scores.json (dict with output_score)")
    ap.add_argument("--out", default="reason_vs_output.png")
    ap.add_argument("--title", default="Выбор признаков: метрики редко максимальны одновременно")
    ap.add_argument("--use_only_output_features", action="store_true",
                    help="Plot only features present in output_scores.json (recommended).")
    args = ap.parse_args()

    reason_path = Path(args.reason_pt)
    out_path = Path(args.out)

    # ReasonScore: tensor where index == feature id
    reason_scores = torch.load(reason_path, map_location="cpu", weights_only=True)
    if isinstance(reason_scores, dict) and "scores" in reason_scores:
        reason_scores = reason_scores["scores"]
    reason_scores = reason_scores.detach().cpu().float()

    # OutputScore: dict[str(feature_id)] -> {"output_score": ...}
    output = json.loads(Path(args.output_json).read_text(encoding="utf-8"))

    xs, ys = [], []
    if args.use_only_output_features:
        # Most consistent: output_scores were computed only for a selected top-k set
        for k, v in output.items():
            i = int(k)
            if i < 0 or i >= len(reason_scores):
                continue
            if isinstance(v, dict) and "output_score" in v:
                xs.append(float(reason_scores[i].item()))
                ys.append(float(v["output_score"]))
    else:
        # Only works if output contains nearly all features
        for i in range(len(reason_scores)):
            v = output.get(str(i))
            if v is None:
                continue
            xs.append(float(reason_scores[i].item()))
            ys.append(float(v["output_score"]))

    if not xs:
        raise RuntimeError("No overlapping features found between ReasonScore and Output Score files.")

    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, s=18)
    plt.title(args.title)
    plt.xlabel("ReasonScore")
    plt.ylabel("OutputScore")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[ok] saved {out_path} with n={len(xs)} points")


if __name__ == "__main__":
    main()
