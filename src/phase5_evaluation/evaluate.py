"""
Phase 5: Evaluation & Analysis -- evaluate every trained checkpoint.

For each checkpoint saved by Phase 4 (checkpoints/<variant>_seed<k>.pt) this
measures, on the SAME fixed held-out text for every model:
  - loss / perplexity on heldout_clean (500 docs that pass the quality filters)
  - loss / perplexity on heldout_raw   (500 random raw docs, incl. low-quality text)
and writes a few generated text samples (same prompts and random seed for all).

Lower loss = the model predicts unseen text better.

Run from the project root:
    python src/phase5_evaluation/evaluate.py
Outputs: results/phase5/eval_<run>.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4_pretraining"))
from model_config import ModelConfig  # noqa: E402
from train import GPT, evaluate_loss, load_tokens, make_eval_windows  # noqa: E402

PROMPTS = ["The", "In the United States,", "If you want to", "One of the best"]


def evaluate_checkpoint(ckpt_path: Path, tokenizer: Tokenizer, heldout: dict, ctx: int, seed: int = 0):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = GPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    result = {"run_name": ckpt_path.stem}
    for name, windows in heldout.items():
        loss = evaluate_loss(model, windows)
        result[name] = {"loss": loss, "perplexity": math.exp(loss), "tokens": int(windows[0].numel())}

    torch.manual_seed(seed)
    samples = []
    for prompt in PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt).ids])
        out = model.generate(ids, max_new_tokens=60)[0].tolist()
        samples.append({"prompt": prompt, "text": tokenizer.decode(out)})
    result["samples"] = samples
    return result


def main():
    p = argparse.ArgumentParser(description="Evaluate all trained checkpoints on the held-out sets.")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--tokenized-dir", default="data/tokenized")
    p.add_argument("--results-dir", default="results/phase5")
    p.add_argument("--max-eval-tokens", type=int, default=250_000, help="Max tokens per held-out set.")
    a = p.parse_args()

    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir = Path(a.tokenized_dir)
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    ctx = ModelConfig().context_length
    heldout = {name: make_eval_windows(load_tokens(tok_dir / f"{name}.npy"), ctx, a.max_eval_tokens)
               for name in ("heldout_clean", "heldout_raw")}

    ckpts = sorted(Path(a.checkpoint_dir).glob("*.pt"))
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {a.checkpoint_dir}/ -- run Phase 4 first.")
    for ck in ckpts:
        res = evaluate_checkpoint(ck, tokenizer, heldout, ctx)
        (out_dir / f"eval_{ck.stem}.json").write_text(json.dumps(res, indent=2))
        print(f"{ck.stem:<28} clean loss {res['heldout_clean']['loss']:.4f} "
              f"(ppl {res['heldout_clean']['perplexity']:.1f}) | raw loss {res['heldout_raw']['loss']:.4f} "
              f"(ppl {res['heldout_raw']['perplexity']:.1f})")
    print(f"Saved results to {out_dir}/")


if __name__ == "__main__":
    main()