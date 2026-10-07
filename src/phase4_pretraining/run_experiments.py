"""
Phase 4: Model Pretraining & Experimentation -- run all four variants.

1. Measures how fast this computer trains the model (tokens/second).
2. Turns "--minutes N per run" into ONE fixed token budget, saved to
   results/phase4/budget.json and reused for every run (and for later reruns).
3. Trains each variant for each seed with that same budget.
   Runs that already have a result file are skipped, so you can stop and resume.

Project root, example (about 5 minutes per run, 2 seeds = 8 runs):
    python src/phase4_pretraining/run_experiments.py --minutes 5 --seeds 1 2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import benchmark, train  # noqa: E402

VARIANTS = ["raw", "filtered", "deduplicated", "filtered_deduplicated"]


def main():
    p = argparse.ArgumentParser(description="Train all dataset variants with one fixed token budget.")
    p.add_argument("--minutes", type=float, default=5.0, help="Target training minutes per run.")
    p.add_argument("--train-tokens", type=int, default=None, help="Set the token budget directly (skips the benchmark).")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--tokenized-dir", default="data/tokenized")
    p.add_argument("--results-dir", default="results/phase4")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    a = p.parse_args()

    results_dir = Path(a.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    budget_file = results_dir / "budget.json"

    if a.train_tokens:
        budget = {"train_tokens": a.train_tokens, "note": "set manually"}
        budget_file.write_text(json.dumps(budget, indent=2))
    elif budget_file.exists():
        budget = json.loads(budget_file.read_text())
        print(f"Using the saved budget: {budget['train_tokens']:,} tokens per run (delete {budget_file} to re-measure).")
    else:
        print("Measuring training speed on this computer (about 20 seconds) ...")
        tps = benchmark()
        tokens = int(tps * a.minutes * 60 // 8192 * 8192)
        budget = {"train_tokens": tokens, "benchmark_tokens_per_second": tps, "target_minutes": a.minutes}
        budget_file.write_text(json.dumps(budget, indent=2))
        print(f"Speed: {tps:,.0f} tokens/second -> budget {tokens:,} tokens per run (~{a.minutes:g} min).")

    for seed in a.seeds:
        for variant in a.variants:
            if (results_dir / f"{variant}_seed{seed}.json").exists():
                print(f"Skipping {variant} seed {seed} (already done).")
                continue
            train(variant, seed, budget["train_tokens"], a.tokenized_dir, a.results_dir, a.checkpoint_dir)
    print("All requested runs finished.")


if __name__ == "__main__":
    main()