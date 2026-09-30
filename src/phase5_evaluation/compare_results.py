"""
Phase 5: Evaluation & Analysis — aggregate and compare results across
all four dataset variants (raw / filtered / deduplicated / filtered_deduplicated).

Reads the per-run JSON files written by evaluate.py, builds the
comparison table, and answers the project's core question: does
curation reduce data/compute needs without hurting performance?
"""

import argparse
import json
from pathlib import Path


def load_all_results(results_dir: Path) -> list:
    """Load every run's results JSON (written by evaluate.py) from results_dir."""
    results = []
    for path in sorted(results_dir.glob("*.json")):
        results.append(json.loads(path.read_text()))
    return results


def build_comparison_table(results: list):
    """
    Build the Section 5 comparison table for the Final Project Report:
    dataset variant | size vs. raw | training compute | downstream performance.

    TODO: pull dataset-size and training-compute numbers in from the
    Phase 1-4 logs (not stored in the eval JSON) and merge them here.
    """
    raise NotImplementedError


def plot_comparison(results: list, output_path: Path):
    """
    Plot performance vs. compute (or vs. dataset size) across variants.

    TODO: implement with matplotlib once real results exist.
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Aggregate and compare eval results across variants.")
    parser.add_argument("--results-dir", required=True, help="Dir of per-run JSON files from evaluate.py.")
    parser.add_argument("--plot-output", default="plots/comparison.png")
    args = parser.parse_args()

    results = load_all_results(Path(args.results_dir))
    table = build_comparison_table(results)
    print(table)
    plot_comparison(results, Path(args.plot_output))


if __name__ == "__main__":
    main()
