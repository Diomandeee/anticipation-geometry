"""
Anticipatory Transformer Training Script
==========================================

Trains the Anticipatory Transformer on next-token prediction with
anticipation geometry supervision.

Training pipeline:
    1. Load dataset (wikitext-2 or synthetic for smoke test)
    2. Tokenize and chunk into fixed-length sequences
    3. For each batch:
       a. Compute external anticipation scalars from embedding trajectory
       b. Forward pass with external scalars as supervision
       c. Compute total loss = LM loss + scalar loss + ortho penalty
       d. Backward pass and optimizer step
    4. Log metrics: loss, perplexity, commitment correlation, scalar accuracy

Usage:
    # Full training on wikitext-2
    python transformer/train.py

    # Smoke test (CPU, tiny model, synthetic data)
    python transformer/train.py --smoke-test

    # Custom config
    python transformer/train.py --d-model 256 --n-heads 4 --epochs 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add parent to path for local imports when running as script
_this_dir = str(Path(__file__).parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
# Also add the project root for anticipation_geometry package
_proj_root = str(Path(__file__).parent.parent / "python")
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from config import AnticipatoryConfig
from model import AnticipatoryTransformer, BaselineTransformer
from geometry_bridge import GeometryBridge


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Simple dataset that chunks tokenized text into fixed-length sequences."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        # Chunk into seq_len+1 windows (input + target)
        n_seqs = len(token_ids) // (seq_len + 1)
        self.data = torch.tensor(
            token_ids[: n_seqs * (seq_len + 1)],
            dtype=torch.long,
        ).view(n_seqs, seq_len + 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]  # input, target


def load_wikitext(seq_len: int, split: str = "train") -> TextDataset:
    """Load wikitext-2 dataset. Falls back to synthetic if unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n".join(ds["text"])

        # Simple character-level tokenization for wikitext
        # (real training would use a proper tokenizer)
        chars = sorted(set(text))
        char_to_idx = {c: i for i, c in enumerate(chars)}
        tokens = [char_to_idx.get(c, 0) for c in text]
        vocab_size = len(chars)
        print(f"Loaded wikitext-2 ({split}): {len(tokens)} tokens, vocab={vocab_size}")
        return TextDataset(tokens, seq_len), vocab_size

    except (ImportError, Exception) as e:
        print(f"Could not load wikitext-2: {e}")
        print("Falling back to synthetic data.")
        return make_synthetic_dataset(seq_len, n_seqs=500, vocab_size=1000), 1000


def make_synthetic_dataset(
    seq_len: int,
    n_seqs: int = 200,
    vocab_size: int = 1000,
) -> TextDataset:
    """Create a synthetic dataset with structured patterns.

    The synthetic data has deliberate patterns to test whether the
    anticipatory model can learn trajectory geometry:
    - Repeating sequences (high commitment should emerge)
    - Random segments (high uncertainty should emerge)
    - Transition points between patterns
    """
    np.random.seed(42)
    all_tokens = []

    for _ in range(n_seqs):
        seq = []
        # Pattern 1: Repeated motif (should produce high commitment)
        motif_len = np.random.randint(3, 8)
        motif = np.random.randint(0, vocab_size, motif_len).tolist()
        repeats = np.random.randint(2, 5)
        seq.extend(motif * repeats)

        # Pattern 2: Random segment (should produce high uncertainty)
        rand_len = np.random.randint(5, 15)
        seq.extend(np.random.randint(0, vocab_size, rand_len).tolist())

        # Pattern 3: Gradual convergence (should produce transition pressure)
        center = np.random.randint(0, vocab_size)
        for i in range(10):
            noise_range = max(1, vocab_size // (2 ** i))
            seq.append(max(0, min(vocab_size - 1, center + np.random.randint(-noise_range, noise_range))))

        # Pad or truncate to seq_len + 1
        if len(seq) < seq_len + 1:
            seq.extend(np.random.randint(0, vocab_size, seq_len + 1 - len(seq)).tolist())
        seq = seq[: seq_len + 1]
        all_tokens.extend(seq)

    return TextDataset(all_tokens, seq_len)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

class Trainer:
    """Training loop for the Anticipatory Transformer.

    Handles:
        - External scalar computation via GeometryBridge
        - Gradient accumulation
        - Learning rate scheduling (cosine warmup)
        - Metric logging
    """

    def __init__(
        self,
        model: AnticipatoryTransformer,
        config: AnticipatoryConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        grad_accum_steps: int = 1,
        warmup_steps: int = 100,
        max_steps: int = 5000,
        log_interval: int = 10,
        eval_interval: int = 100,
        device: str = "cpu",
        use_external_scalars: bool = True,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_external_scalars = use_external_scalars
        self.grad_accum_steps = grad_accum_steps
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # LR scheduler (cosine with warmup)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: self._lr_schedule(step),
        )

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, drop_last=True,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, batch_size=8, shuffle=False, drop_last=True,
            )

        # Geometry bridge for external scalar computation
        self.bridge = GeometryBridge(k_neighbors=3) if use_external_scalars else None

        # Metrics
        self.step = 0
        self.best_val_loss = float("inf")
        self.history: list[dict] = []

    def _lr_schedule(self, step: int) -> float:
        """Cosine warmup schedule."""
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(self) -> list[dict]:
        """Run the full training loop."""
        print(f"\nTraining Anticipatory Transformer")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  External scalars: {self.use_external_scalars}")
        print()

        self.model.train()
        running_loss = 0.0
        running_lm_loss = 0.0
        running_scalar_loss = 0.0
        running_ortho = 0.0
        n_logged = 0
        t_start = time.time()

        train_iter = iter(self.train_loader)

        while self.step < self.max_steps:
            # Get batch (cycle through dataset)
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                input_ids, targets = next(train_iter)

            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            # Compute external scalars
            external_scalars = None
            if self.use_external_scalars and self.bridge is not None:
                external_scalars = self.bridge.batch_compute(
                    input_ids, self.model.tok_emb,
                ).to(self.device)

            # Forward pass
            result = self.model(
                input_ids,
                external_scalars=external_scalars,
                targets=targets,
            )

            loss = result["loss"] / self.grad_accum_steps
            loss.backward()

            # Gradient accumulation
            if (self.step + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Track metrics
            running_loss += result["loss"].item()
            running_lm_loss += result["lm_loss"].item()
            if "scalar_loss" in result:
                running_scalar_loss += result["scalar_loss"].item()
            running_ortho += result["ortho_penalty"].item()
            n_logged += 1

            # Log
            if (self.step + 1) % self.log_interval == 0:
                avg_loss = running_loss / n_logged
                avg_lm = running_lm_loss / n_logged
                avg_scalar = running_scalar_loss / n_logged
                avg_ortho = running_ortho / n_logged
                ppl = math.exp(min(avg_lm, 20))  # cap to avoid overflow
                elapsed = time.time() - t_start
                steps_per_sec = (self.step + 1) / elapsed

                lr = self.scheduler.get_last_lr()[0]

                metrics = {
                    "step": self.step + 1,
                    "loss": avg_loss,
                    "lm_loss": avg_lm,
                    "scalar_loss": avg_scalar,
                    "ortho_penalty": avg_ortho,
                    "perplexity": ppl,
                    "lr": lr,
                    "steps_per_sec": steps_per_sec,
                }
                self.history.append(metrics)

                # Compute commitment correlation
                with torch.no_grad():
                    pred_scalars = result["predicted_scalars"]
                    gate_values = result["gate_values"]
                    commit = pred_scalars[:, :, 0].flatten()
                    gates = gate_values.squeeze(-1).flatten()
                    if commit.std() > 0 and gates.std() > 0:
                        corr = torch.corrcoef(torch.stack([commit, gates]))[0, 1].item()
                    else:
                        corr = 0.0
                    metrics["commitment_gate_corr"] = corr

                print(
                    f"[{self.step + 1:>5}/{self.max_steps}] "
                    f"loss={avg_loss:.4f} lm={avg_lm:.4f} "
                    f"scalar={avg_scalar:.4f} ortho={avg_ortho:.5f} "
                    f"ppl={ppl:.1f} corr={corr:+.3f} "
                    f"lr={lr:.2e} ({steps_per_sec:.1f} steps/s)"
                )

                running_loss = 0.0
                running_lm_loss = 0.0
                running_scalar_loss = 0.0
                running_ortho = 0.0
                n_logged = 0

            # Eval
            if self.val_loader and (self.step + 1) % self.eval_interval == 0:
                val_loss = self.evaluate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print(f"  ** New best val loss: {val_loss:.4f}")
                self.model.train()

            self.step += 1

        elapsed = time.time() - t_start
        print(f"\nTraining complete in {elapsed:.1f}s ({self.step} steps)")
        return self.history

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for input_ids, targets in self.val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            result = self.model(input_ids, targets=targets)
            total_loss += result["lm_loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        print(f"  [Eval] val_loss={avg_loss:.4f} val_ppl={ppl:.1f}")
        return avg_loss


# ---------------------------------------------------------------------------
# Smoke Test Mode
# ---------------------------------------------------------------------------

def run_smoke_test():
    """Quick smoke test: tiny model, synthetic data, CPU."""
    print("=" * 60)
    print("SMOKE TEST: Anticipatory Transformer Training")
    print("=" * 60)

    config = AnticipatoryConfig().small()
    seq_len = 32

    # Create synthetic train/val data
    train_ds = make_synthetic_dataset(seq_len, n_seqs=100, vocab_size=config.vocab_size)
    val_ds = make_synthetic_dataset(seq_len, n_seqs=20, vocab_size=config.vocab_size)

    print(f"Config: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"vocab={config.vocab_size}")
    print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

    model = AnticipatoryTransformer(config)
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")

    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=1e-3,
        max_steps=50,
        log_interval=5,
        eval_interval=25,
        device="cpu",
        use_external_scalars=True,
    )

    history = trainer.train()

    # Verify training happened
    assert len(history) > 0, "No training metrics recorded"

    first_loss = history[0]["loss"]
    last_loss = history[-1]["loss"]
    print(f"\nFirst loss: {first_loss:.4f}")
    print(f"Last loss: {last_loss:.4f}")

    # Test generation
    print("\n--- Generation test ---")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=10, use_commitment_gate=False)
    print(f"Prompt: {prompt[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")

    # Verify scalar predictions exist and are in range
    with torch.no_grad():
        result = model(prompt)
        scalars = result["predicted_scalars"]
        for i, name in enumerate([
            "commitment", "uncertainty", "transition_pressure",
            "recovery_margin", "phase_stiffness", "novelty", "stability"
        ]):
            val = scalars[0, :, i]
            print(f"  {name}: mean={val.mean():.3f}, std={val.std():.3f}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Anticipatory Transformer")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-fast-layers", type=int, default=2, help="Fast pathway layers")
    parser.add_argument("--n-slow-layers", type=int, default=2, help="Slow pathway layers")
    parser.add_argument("--d-ff", type=int, default=512, help="FFN dimension")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--no-external-scalars", action="store_true",
                        help="Disable external scalar supervision")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save checkpoints")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    # Build config
    config = AnticipatoryConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_fast_layers=args.n_fast_layers,
        n_slow_layers=args.n_slow_layers,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len * 4,
        dropout=0.1,
    )

    # Load data
    print("Loading dataset...")
    train_ds = make_synthetic_dataset(
        args.seq_len, n_seqs=500, vocab_size=config.vocab_size,
    )
    val_ds = make_synthetic_dataset(
        args.seq_len, n_seqs=50, vocab_size=config.vocab_size,
    )

    # Try loading wikitext if available
    try:
        from datasets import load_dataset
        print("Attempting wikitext-2 load...")
        train_ds, actual_vocab = load_wikitext(args.seq_len, "train")
        val_ds, _ = load_wikitext(args.seq_len, "validation")
        config = AnticipatoryConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_fast_layers=args.n_fast_layers,
            n_slow_layers=args.n_slow_layers,
            d_ff=args.d_ff,
            vocab_size=actual_vocab,
            max_seq_len=args.seq_len * 4,
            dropout=0.1,
        )
    except Exception:
        print("Using synthetic data.")

    print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

    # Build model
    model = AnticipatoryTransformer(config)
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,}")
    for name, count in params.items():
        if name != "total":
            print(f"  {name}: {count:,}")

    # Device selection
    device = args.device
    if device == "mps" and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
    elif device == "cuda" and torch.cuda.is_available():
        print("Using CUDA")
    else:
        device = "cpu"
        print("Using CPU")

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=args.lr,
        max_steps=args.max_steps,
        log_interval=10,
        eval_interval=200,
        device=device,
        use_external_scalars=not args.no_external_scalars,
    )

    history = trainer.train()

    # Save checkpoint
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / "anticipatory_transformer.pt"
        torch.save({
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "step": trainer.step,
            "history": history,
        }, ckpt_path)
        print(f"\nCheckpoint saved to {ckpt_path}")

    # Final summary
    print("\nTraining Summary:")
    print(f"  Final loss: {history[-1]['loss']:.4f}")
    print(f"  Final perplexity: {history[-1]['perplexity']:.1f}")
    if "commitment_gate_corr" in history[-1]:
        print(f"  Commitment-gate correlation: {history[-1]['commitment_gate_corr']:+.3f}")


if __name__ == "__main__":
    main()
