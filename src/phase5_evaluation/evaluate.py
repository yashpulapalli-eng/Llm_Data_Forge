"""
Phase 5: Evaluation & Analysis — evaluate a single trained checkpoint.

Run once per dataset-variant model, then aggregate with compare_results.py.
"""

import argparse
import json
from pathlib import Path


def load_checkpoint(checkpoint_path: Path):
    """Load a trained model checkpoint for evaluation. TODO: implement."""
    raise NotImplementedError


def run_benchmark(model, benchmark_name: str) -> dict:
    """
    Run one downstream benchmark/eval task against `model`.

    TODO:
        - Pick benchmark(s) appropriate to a small/medium pretrained
          model (e.g. held-out perplexity, a small subset of a
          standard benchmark like LAMBADA/HellaSwag/a simple QA set).
        - Return a dict of metric name -> value.
    """
    raise NotImplementedError


def evaluate(checkpoint_path: Path, run_name: str, benchmarks: list, output_path: Path):
    model = load_checkpoint(checkpoint_path)
    results = {"run_name": run_name}
    for bench in benchmarks:
        results[bench] = run_benchmark(model, bench)

    output_path.write_text(json.dumps(results, indent=2))
    print(f"[{run_name}] wrote results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate one trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True, help="e.g. raw / filtered / dedup / filtered_dedup.")
    parser.add_argument("--benchmarks", nargs="+", default=["perplexity"])
    parser.add_argument("--output", required=True, help="Where to write this run's results JSON.")
    args = parser.parse_args()

    evaluate(Path(args.checkpoint), args.run_name, args.benchmarks, Path(args.output))


if __name__ == "__main__":
    main()
