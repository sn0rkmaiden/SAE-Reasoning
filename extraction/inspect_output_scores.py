import argparse
import json
import torch
from pathlib import Path
from transformer_lens import HookedTransformer

"""
Example usage:

python inspect_output_scores.py \
  --scores_path /.../output_scores/topk_200/output_scores.json \
  --a_max_path   /.../output_scores/topk_200/a_max.pt \
  --model_name gemma-2b-it \
  --sort_by output_score

or sort by rank:

python inspect_output_scores.py \
  --scores_path ... \
  --a_max_path ... \
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
    
    parser.add_argument("--save_text", type=str, default=None,
                    help="Optional: path to save human-readable text report")

    parser.add_argument("--save_json", type=str, default=None,
                        help="Optional: save enriched JSON including decoded tokens")

    args = parser.parse_args()

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained_no_processing(args.model_name, device=device, dtype=torch.float16)

    # Load results
    scores_path = Path(args.scores_path)
    with open(scores_path, "r") as f:
        results = json.load(f)

    a_max = torch.load(args.a_max_path, map_location="cpu")

    print("Loaded a_max.")

    results = {int(k): v for k, v in results.items()}

    if args.sort_by == "feature_id":
        ordered = sorted(results.items(), key=lambda x: x[0])
    elif args.sort_by == "rank":
        ordered = sorted(results.items(), key=lambda x: x[1]["rank"])
    else:
        ordered = sorted(results.items(), key=lambda x: x[1][args.sort_by], reverse=True)

    print(f"\n=== INSPECTION REPORT (sorted by {args.sort_by}) ===\n")

    text_lines = []
    enriched_json = {}

    for fid, res in ordered:
        top_tokens = res["top_tokens"][:5]
        top_strings = [model.to_string(t) for t in top_tokens]

        block = (
            f"Feature {fid}:\n"
            f"  - top_tokens IDs = {res['top_tokens']}\n"
            f"  - top_tokens str = {top_strings}\n"
            f"  - prob = {res['prob']:.6f}\n"
            f"  - rank = {res['rank']}\n"
            f"  - output_score = {res['output_score']:.6f}\n"
            f"  - a_max[{fid}] = {a_max[fid].item():.6f}\n"
        )

        print(block)
        text_lines.append(block)

        enriched_json[fid] = {
            **res,
            "top_tokens_str": top_strings,
            "a_max": float(a_max[fid].item())
        }

    # Save text report
    if args.save_text:
        text_path = Path(args.save_text)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("\n".join(text_lines), encoding="utf-8")
        print(f"\nSaved text report to: {text_path}")

    # Save enriched JSON
    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(enriched_json, f, indent=2)
        print(f"Saved enriched JSON to: {json_path}")

    print("Done.")


if __name__ == "__main__":
    main()
