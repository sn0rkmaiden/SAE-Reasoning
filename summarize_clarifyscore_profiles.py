#!/usr/bin/env python3
"""Compare 256/512-sample ClarifyScore profiles and produce a concise summary."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('profiles', nargs='+')
    ap.add_argument('--output_json', required=True)
    args=ap.parse_args()
    rows=[json.loads(Path(p).read_text()) for p in args.profiles]
    estimates=[r['estimated_full_one_vocabulary_single_v100']['historical_separate_chunk_jobs_seconds'] for r in rows]
    compute_est=[r['estimated_full_one_vocabulary_single_v100']['compute_only_seconds'] for r in rows]
    throughputs=[]
    for r in rows:
        calls=r['workload']['profile_llm_forward_calls']
        sec=r['measured']['score_compute_seconds']
        throughputs.append(calls/sec)
    out={
      'n_profiles':len(rows),
      'profile_files':args.profiles,
      'full_one_vocab_seconds_mean':statistics.mean(estimates),
      'full_one_vocab_gpu_hours_mean':statistics.mean(estimates)/3600,
      'full_one_vocab_seconds_min':min(estimates),
      'full_one_vocab_seconds_max':max(estimates),
      'compute_only_seconds_mean':statistics.mean(compute_est),
      'profile_forward_calls_per_second':throughputs,
      'relative_spread_percent':100*(max(estimates)-min(estimates))/statistics.mean(estimates) if len(estimates)>1 else 0,
      'recommended_two_vocab_gpu_hours_if_C_and_Q_are_profiled_separately':2*statistics.mean(estimates)/3600,
      'note':'Use separate C and Q summaries if their runtimes differ; do not blindly double when both were measured.'
    }
    Path(args.output_json).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
