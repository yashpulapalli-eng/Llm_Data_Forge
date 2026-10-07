"""
Phase 5: Evaluation & Analysis -- compare the four dataset variants.

Reads:
  results/phase4/*_seed*.json   training logs (curves, tokens/sec, time)
  results/phase5/eval_*.json    held-out evaluation (from evaluate.py)
  data/tokenized/meta.json      dataset sizes after tokenization
Writes (to results/phase5/):
  comparison.json, comparison.md, figures/learning_curves.png, figures/final_loss.png

How to read it: every variant gets the same model, same token budget and the
same starting weights (per seed). So a difference in held-out loss comes from
the DATA. A difference only counts as real if it is bigger than the
seed-to-seed spread (the "noise" a rerun would give).

Run from the project root:   python src/phase5_evaluation/compare_results.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VARIANTS = ["raw", "filtered", "deduplicated", "filtered_deduplicated"]
LABELS = {"raw": "Raw", "filtered": "Filtered", "deduplicated": "Deduplicated",
          "filtered_deduplicated": "Filtered + dedup"}
COLORS = {"raw": "#2a78d6", "filtered": "#eb6834", "deduplicated": "#1baf7a", "filtered_deduplicated": "#eda100"}
MARKERS = {"raw": "o", "filtered": "s", "deduplicated": "^", "filtered_deduplicated": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load_json_files(folder: Path, pattern: str):
    return [json.loads(p.read_text()) for p in sorted(folder.glob(pattern))]


def tokens_to_reach(curve, target):
    """Tokens needed to first reach a loss <= target (linear interpolation); None if never."""
    pts = [(c["tokens_seen"], c["val_clean"]) for c in curve]
    for (t0, l0), (t1, l1) in zip(pts, pts[1:]):
        if l1 <= target < l0:
            return t0 + (t1 - t0) * (l0 - target) / (l0 - l1)
    return pts[0][0] if pts[0][1] <= target else None


def main():
    p = argparse.ArgumentParser(description="Compare dataset variants.")
    p.add_argument("--phase4-dir", default="results/phase4")
    p.add_argument("--phase5-dir", default="results/phase5")
    p.add_argument("--tokenized-dir", default="data/tokenized")
    a = p.parse_args()
    p4, p5 = Path(a.phase4_dir), Path(a.phase5_dir)
    (p5 / "figures").mkdir(parents=True, exist_ok=True)

    runs = [r for r in load_json_files(p4, "*_seed*.json")]
    evals = {e["run_name"]: e for e in load_json_files(p5, "eval_*.json")}
    meta_path = Path(a.tokenized_dir) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"variants": {}}
    if not runs:
        raise SystemExit("No training results found -- run Phase 4 first.")

    by_variant = defaultdict(dict)                # variant -> seed -> run
    for r in runs:
        by_variant[r["variant"]][r["seed"]] = r
    variants = [v for v in VARIANTS if v in by_variant]
    seeds = sorted(set.intersection(*[set(by_variant[v]) for v in variants]))

    def final_loss(run, which):                  # prefers the full held-out evaluation
        ev = evals.get(run["run_name"])
        return ev[f"heldout_{which}"]["loss"] if ev else run[f"final_val_{which}"]

    table = {}
    for v in variants:
        row = {"documents": meta["variants"].get(v, {}).get("documents"),
               "tokens_available": by_variant[v][seeds[0]]["dataset_tokens_available"],
               "train_tokens": by_variant[v][seeds[0]]["train_tokens"],
               "train_seconds_mean": float(np.mean([by_variant[v][s]["train_seconds"] for s in seeds])),
               "tokens_per_second_mean": float(np.mean([by_variant[v][s]["tokens_per_second"] for s in seeds]))}
        for which in ("clean", "raw"):
            vals = [final_loss(by_variant[v][s], which) for s in seeds]
            row[which] = {"per_seed": dict(zip(map(str, seeds), vals)), "mean": float(np.mean(vals)),
                          "min": float(np.min(vals)), "max": float(np.max(vals)),
                          "perplexity": float(np.exp(np.mean(vals)))}
        table[v] = row

    # noise = the largest seed-to-seed range seen in any variant
    noise = {w: max(table[v][w]["max"] - table[v][w]["min"] for v in variants) for w in ("clean", "raw")}
    for v in variants:
        for which in ("clean", "raw"):
            deltas = [final_loss(by_variant[v][s], which) - final_loss(by_variant["raw"][s], which) for s in seeds]
            d = float(np.mean(deltas))
            table[v][which]["delta_vs_raw"] = d
            table[v][which]["same_sign_all_seeds"] = bool(all(x < 0 for x in deltas) or all(x > 0 for x in deltas))
            table[v][which]["beyond_noise"] = bool(v != "raw" and abs(d) > noise[which])

    # tokens needed to match the raw model's final clean loss (training efficiency)
    raw_target = np.mean([by_variant["raw"][s]["final_val_clean"] for s in seeds])
    for v in variants:
        reach = [tokens_to_reach(by_variant[v][s]["curve"], raw_target) for s in seeds]
        reach = [x for x in reach if x is not None]
        table[v]["tokens_to_match_raw"] = float(np.mean(reach)) if len(reach) == len(seeds) else None

    result = {"seeds": seeds, "noise_range": noise, "variants": table,
              "note": "noise_range = largest seed-to-seed spread of final loss in any variant."}
    (p5 / "comparison.json").write_text(json.dumps(result, indent=2))

    # ---------- markdown table ----------
    L = []
    L.append(f"Seeds: {seeds} | seed-to-seed spread (noise): clean {noise['clean']:.4f}, raw {noise['raw']:.4f}\n")
    L.append("| Variant | Docs | Tokens | Clean loss (ppl) | vs raw | Raw-test loss (ppl) | vs raw | Train time |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v in variants:
        t = table[v]
        def cell(w):
            d = t[w]["delta_vs_raw"]
            flag = "" if v == "raw" else (" (beyond noise)" if t[w]["beyond_noise"] else " (within noise)")
            return (f"{t[w]['mean']:.4f} ({t[w]['perplexity']:.1f})", "-" if v == "raw" else f"{d:+.4f}{flag}")
        c, r = cell("clean"), cell("raw")
        docs = f"{t['documents']:,}" if t["documents"] else "?"
        L.append(f"| {LABELS[v]} | {docs} | {t['tokens_available']:,} | {c[0]} | {c[1]} | {r[0]} | {r[1]} | "
                 f"{t['train_seconds_mean']:.0f}s |")
    (p5 / "comparison.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ---------- figures ----------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, which, title in ((axes[0], "val_clean", "Held-out clean text"), (axes[1], "val_raw", "Held-out raw text")):
        for v in variants:
            toks = [c["tokens_seen"] for c in by_variant[v][seeds[0]]["curve"]]
            ys = np.mean([[c[which] for c in by_variant[v][s]["curve"]] for s in seeds], axis=0)
            ax.plot(np.array(toks) / 1e6, ys, color=COLORS[v], marker=MARKERS[v], lw=2, ms=6, label=LABELS[v])
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Training tokens seen (millions)")
        ax.set_ylabel("Validation loss (lower is better)")
        ax.grid(color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Learning curves (mean over seeds)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(p5 / "figures" / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, which, title in ((axes[0], "clean", "Held-out clean text"), (axes[1], "raw", "Held-out raw text")):
        for i, v in enumerate(variants):
            vals = [final_loss(by_variant[v][s], which) for s in seeds]
            ax.scatter([i] * len(vals), vals, color=COLORS[v], marker=MARKERS[v], s=40, alpha=0.55, zorder=3)
            m = table[v][which]["mean"]
            ax.hlines(m, i - 0.28, i + 0.28, color=COLORS[v], lw=3, zorder=4)
            ax.text(i + 0.32, m, f"{m:.3f}", va="center", fontsize=9)
        ax.set_xlim(-0.5, len(variants) - 0.2)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([LABELS[v] for v in variants], fontsize=9)
        ax.set_ylabel("Final validation loss (lower is better)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Final loss by dataset variant (dots = individual seeds, line = mean)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(p5 / "figures" / "final_loss.png", dpi=160)
    plt.close(fig)
    print(f"Saved comparison.json, comparison.md and figures to {p5}/")


if __name__ == "__main__":
    main()