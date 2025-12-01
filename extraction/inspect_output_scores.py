import argparse
import json
import torch
from pathlib import Path
from transformer_lens import HookedTransformer

"""
Example usage:

python inspect_output_scores.py \
  --scores_path results/output_scores.json \
  --a_max_path results/a_max.pt \
  --model_name gemma-2b-it \
  --sort_by output_score

or sort by rank:

python inspect_output_scores.py \
  --scores_path results/output_scores.json \
  --a_max_path results/a_max.pt \
  --sort_by rank

"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scores_path", type=str, required=True,
                        help="Path to output_scores.json produced by run_output_scores.py")

    parser.add_argument("--a_max_path", type=str, required=True,
                        help="Path to a_max.pt saved during the compute process")

    parser.add_argument("--model_name", type=str, default="gemma-2b-it",
                        help="Model name for token decoding")

    parser.add_argument("--sort_by", type=str, default="output_score",
                        choices=["output_score", "prob", "rank", "feature_id"],
                        help="Sorting metric")

    args = parser.parse_args()

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(args.model_name, device=device, dtype=torch.float16)

    # Load results
    scores_path = Path(args.scores_path)
    with open(scores_path, "r") as f:
        results = json.load(f)

    a_max = torch.load(args.a_max_path, map_location="cpu")

    results = {int(k): v for k, v in results.items()}

    if args.sort_by == "feature_id":
        ordered = sorted(results.items(), key=lambda x: x[0])
    elif args.sort_by == "rank":
        ordered = sorted(results.items(), key=lambda x: x[1]["rank"])
    else:
        ordered = sorted(results.items(), key=lambda x: x[1][args.sort_by], reverse=True)

    print(f"\n=== INSPECTION REPORT (sorted by {args.sort_by}) ===\n")

    for fid, res in ordered:
        top_tokens = res["top_tokens"][:5]
        top_strings = [model.to_string(t) for t in top_tokens]

        print(f"Feature {fid}:")
        print(f"  - top_tokens IDs = {res['top_tokens']}")
        print(f"  - top_tokens str = {top_strings}")
        print(f"  - prob        = {res['prob']:.6f}")
        print(f"  - rank        = {res['rank']}")
        print(f"  - output_score = {res['output_score']:.6f}")
        print(f"  - a_max[{fid}] = {a_max[fid].item():.6f}")
        print("")

    print("Done.")


if __name__ == "__main__":
    main()
