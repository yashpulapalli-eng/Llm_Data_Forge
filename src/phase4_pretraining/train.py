"""
Phase 4: Model Pretraining & Experimentation -- model + training loop.

Trains one small GPT-style model on ONE dataset variant for a FIXED number of
training tokens. Every variant gets the same model, the same settings and the
same token budget, so only the data differs.

You normally do not call this directly; run_experiments.py runs all variants.
Single run example (project root):
    python src/phase4_pretraining/train.py --variant raw --train-tokens 3000000 --seed 1
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_config import ModelConfig, TrainConfig


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))
        x = x + self.drop(self.fc_proj(F.gelu(self.fc(self.ln2(x)))))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # tie input/output embeddings
        self.apply(self._init)
        for name, p in self.named_parameters():          # scale residual projections
            if name.endswith("proj.weight") or name.endswith("fc_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.context_length:])
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def load_tokens(path: Path) -> np.ndarray:
    return np.load(path).astype(np.int64)


def make_eval_windows(tokens: np.ndarray, ctx: int, max_tokens: int):
    """Fixed, non-overlapping evaluation windows (identical for every run)."""
    n = min(len(tokens) - 1, max_tokens) // ctx
    x = np.stack([tokens[i * ctx:(i + 1) * ctx] for i in range(n)])
    y = np.stack([tokens[i * ctx + 1:(i + 1) * ctx + 1] for i in range(n)])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate_loss(model, windows, batch_size=64):
    model.eval()
    x, y = windows
    total, count = 0.0, 0
    for i in range(0, len(x), batch_size):
        _, loss = model(x[i:i + batch_size], y[i:i + batch_size])
        total += loss.item() * x[i:i + batch_size].numel()
        count += x[i:i + batch_size].numel()
    model.train()
    return total / count


def get_batch(tokens: np.ndarray, rng, batch_size: int, ctx: int):
    starts = rng.integers(0, len(tokens) - ctx - 1, size=batch_size)
    x = np.stack([tokens[s:s + ctx] for s in starts])
    y = np.stack([tokens[s + 1:s + ctx + 1] for s in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def lr_at(step, total_steps, tc: TrainConfig):
    warm = max(1, int(tc.warmup_fraction * total_steps))
    if step < warm:
        return tc.learning_rate * (step + 1) / warm
    progress = (step - warm) / max(1, total_steps - warm)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return tc.learning_rate * (tc.min_lr_ratio + (1 - tc.min_lr_ratio) * cosine)


# --------------------------------------------------------------------------
# Benchmark: how many tokens/second does this computer train at?
# --------------------------------------------------------------------------
def benchmark(steps: int = 15):
    mc, tc = ModelConfig(), TrainConfig()
    torch.manual_seed(0)
    model = GPT(mc)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate)
    rng = np.random.default_rng(0)
    fake = rng.integers(0, mc.vocab_size, size=200_000).astype(np.int64)
    for i in range(steps + 3):
        if i == 3:                                   # skip 3 warm-up steps
            t0 = time.time()
        x, y = get_batch(fake, rng, tc.batch_size, mc.context_length)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return steps * tc.batch_size * mc.context_length / (time.time() - t0)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(variant, seed, train_tokens, tokenized_dir, results_dir, checkpoint_dir):
    mc, tc = ModelConfig(), TrainConfig()
    tokenized_dir, results_dir, checkpoint_dir = Path(tokenized_dir), Path(results_dir), Path(checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{variant}_seed{seed}"

    data = load_tokens(tokenized_dir / f"{variant}.npy")
    clean = make_eval_windows(load_tokens(tokenized_dir / "heldout_clean.npy"), mc.context_length, tc.eval_tokens)
    raw = make_eval_windows(load_tokens(tokenized_dir / "heldout_raw.npy"), mc.context_length, tc.eval_tokens)

    tokens_per_step = tc.batch_size * mc.context_length
    total_steps = max(1, train_tokens // tokens_per_step)
    eval_every = max(1, total_steps // tc.num_evals)

    torch.manual_seed(seed)                   # same starting weights for every variant with this seed
    model = GPT(mc)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc.learning_rate, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)

    print(f"[{run_name}] {model.num_params():,} params | {len(data):,} training tokens available | "
          f"budget {total_steps * tokens_per_step:,} tokens ({total_steps} steps)")

    curve, train_time = [], 0.0
    run_loss, run_n = 0.0, 0
    for step in range(total_steps):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, tc)
        x, y = get_batch(data, rng, tc.batch_size, mc.context_length)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        train_time += time.time() - t0
        run_loss += loss.item()
        run_n += 1

        if (step + 1) % eval_every == 0 or step + 1 == total_steps:
            point = {"step": step + 1, "tokens_seen": (step + 1) * tokens_per_step,
                     "train_loss": run_loss / run_n,
                     "val_clean": evaluate_loss(model, clean),
                     "val_raw": evaluate_loss(model, raw),
                     "train_seconds": train_time}
            curve.append(point)
            run_loss, run_n = 0.0, 0
            print(f"[{run_name}] step {step + 1:>5}/{total_steps}  train {point['train_loss']:.3f}  "
                  f"val_clean {point['val_clean']:.3f}  val_raw {point['val_raw']:.3f}  ({train_time:.0f}s)")

    torch.save({"model": model.state_dict(), "model_config": mc.to_dict()}, checkpoint_dir / f"{run_name}.pt")
    result = {
        "run_name": run_name, "variant": variant, "seed": seed,
        "params": model.num_params(),
        "model_config": mc.to_dict(), "train_config": tc.to_dict(),
        "train_tokens": total_steps * tokens_per_step, "steps": total_steps,
        "dataset_tokens_available": int(len(data)),
        "epochs_over_data": total_steps * tokens_per_step / len(data),
        "train_seconds": train_time,
        "tokens_per_second": total_steps * tokens_per_step / train_time,
        "final_val_clean": curve[-1]["val_clean"], "final_val_raw": curve[-1]["val_raw"],
        "curve": curve,
    }
    (results_dir / f"{run_name}.json").write_text(json.dumps(result, indent=2))
    print(f"[{run_name}] done in {train_time:.0f}s | val_clean {result['final_val_clean']:.3f} | "
          f"val_raw {result['final_val_raw']:.3f}")
    return result


def main():
    p = argparse.ArgumentParser(description="Pretrain the model on one dataset variant.")
    p.add_argument("--variant", required=True, choices=["raw", "filtered", "deduplicated", "filtered_deduplicated"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--train-tokens", type=int, required=True)
    p.add_argument("--tokenized-dir", default="data/tokenized")
    p.add_argument("--results-dir", default="results/phase4")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    a = p.parse_args()
    train(a.variant, a.seed, a.train_tokens, a.tokenized_dir, a.results_dir, a.checkpoint_dir)


if __name__ == "__main__":
    main()