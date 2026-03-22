"""
Anticipatory Transformer Evaluation
=====================================

Comparative evaluation of the Anticipatory Transformer vs a standard
baseline transformer of equivalent size.

Metrics:
    1. Perplexity: Standard language modeling quality (lower = better)
    2. Commitment correlation: Does the commitment scalar predict output quality?
    3. Context efficiency: How much context does each model need for equivalent performance?
    4. Frequency separation: Do fast/slow pathways capture different frequency bands?
    5. Orthogonality: Are the per-head trajectory biases diverse?

Usage:
    # Quick comparison on synthetic data
    python transformer/evaluate.py --quick

    # Full evaluation
    python transformer/evaluate.py

    # Evaluate a saved checkpoint
    python transformer/evaluate.py --checkpoint path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent to path for local imports when running as script
_this_dir = str(Path(__file__).parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
_proj_root = str(Path(__file__).parent.parent / "python")
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from config import AnticipatoryConfig
from model import AnticipatoryTransformer, BaselineTransformer
from geometry_bridge import GeometryBridge
from train import make_synthetic_dataset, Trainer


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    is_anticipatory: bool = True,
) -> dict[str, float]:
    """Compute perplexity on a dataset.

    Returns per-sequence and aggregate metrics.
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0
    per_seq_losses = []

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        batch, seq = input_ids.shape

        if is_anticipatory:
            result = model(input_ids, targets=targets, use_commitment_gate=False)
        else:
            result = model(input_ids, targets=targets)

        # Per-token loss (no reduction)
        logits = result["logits"]
        loss_per_token = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        loss_per_token = loss_per_token.view(batch, seq)

        # Per-sequence average loss
        for b in range(batch):
            seq_loss = loss_per_token[b].mean().item()
            per_seq_losses.append(seq_loss)

        total_loss += loss_per_token.sum().item()
        total_tokens += (targets != -100).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    return {
        "avg_loss": avg_loss,
        "perplexity": ppl,
        "n_sequences": len(per_seq_losses),
        "loss_std": float(np.std(per_seq_losses)),
        "loss_median": float(np.median(per_seq_losses)),
    }


@torch.no_grad()
def evaluate_commitment_correlation(
    model: AnticipatoryTransformer,
    dataloader: DataLoader,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure whether commitment predicts output quality.

    For each position, we compute:
    - commitment: model's predicted commitment scalar
    - loss: per-position cross-entropy loss

    If commitment is meaningful, high commitment should correlate with
    low loss (the model is more "sure" and more accurate when committed).
    """
    model.eval()
    model.to(device)

    all_commitments = []
    all_losses = []
    all_gate_values = []

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        result = model(input_ids, targets=targets, use_commitment_gate=False)

        logits = result["logits"]
        batch, seq, vocab = logits.shape

        # Per-position loss
        loss_per_pos = F.cross_entropy(
            logits.view(-1, vocab),
            targets.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(batch, seq)

        commitment = result["predicted_scalars"][:, :, 0]  # (batch, seq)
        gate = result["gate_values"].squeeze(-1)  # (batch, seq)

        all_commitments.append(commitment.cpu().flatten())
        all_losses.append(loss_per_pos.cpu().flatten())
        all_gate_values.append(gate.cpu().flatten())

    commits = torch.cat(all_commitments)
    losses = torch.cat(all_losses)
    gates = torch.cat(all_gate_values)

    # Commitment-loss correlation (should be negative: high commit -> low loss)
    if commits.std() > 1e-6 and losses.std() > 1e-6:
        commit_loss_corr = torch.corrcoef(
            torch.stack([commits, losses])
        )[0, 1].item()
    else:
        commit_loss_corr = 0.0

    # Commitment-gate correlation (should be positive)
    if commits.std() > 1e-6 and gates.std() > 1e-6:
        commit_gate_corr = torch.corrcoef(
            torch.stack([commits, gates])
        )[0, 1].item()
    else:
        commit_gate_corr = 0.0

    # Bin commitments and compute per-bin loss
    n_bins = 5
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    per_bin_loss = []
    for i in range(n_bins):
        mask = (commits >= bin_edges[i]) & (commits < bin_edges[i + 1])
        if mask.sum() > 0:
            per_bin_loss.append(losses[mask].mean().item())
        else:
            per_bin_loss.append(float("nan"))

    return {
        "commit_loss_correlation": commit_loss_corr,
        "commit_gate_correlation": commit_gate_corr,
        "commit_mean": commits.mean().item(),
        "commit_std": commits.std().item(),
        "per_bin_loss": per_bin_loss,
        "gate_mean": gates.mean().item(),
    }


@torch.no_grad()
def evaluate_context_efficiency(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    is_anticipatory: bool = True,
    context_sizes: list[int] | None = None,
) -> dict[str, list[float]]:
    """Measure how performance changes with available context.

    For each context size, we truncate the input and measure loss.
    An anticipatory model should need LESS context for equivalent
    performance (because trajectory bias provides geometric context).
    """
    model.eval()
    model.to(device)

    if context_sizes is None:
        context_sizes = [4, 8, 16, 32]

    losses_by_context = {ctx: [] for ctx in context_sizes}

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        full_seq = input_ids.shape[1]

        for ctx_size in context_sizes:
            if ctx_size > full_seq:
                continue

            # Use only the first ctx_size tokens (contiguous copy)
            trunc_input = input_ids[:, :ctx_size].contiguous()
            trunc_target = targets[:, :ctx_size].contiguous()

            if is_anticipatory:
                result = model(trunc_input, targets=trunc_target, use_commitment_gate=False)
            else:
                result = model(trunc_input, targets=trunc_target)

            losses_by_context[ctx_size].append(result["lm_loss"].item())

    avg_losses = {}
    for ctx, losses in losses_by_context.items():
        if losses:
            avg_losses[ctx] = sum(losses) / len(losses)
        else:
            avg_losses[ctx] = float("nan")

    return {
        "context_sizes": context_sizes,
        "losses": [avg_losses.get(c, float("nan")) for c in context_sizes],
    }


@torch.no_grad()
def evaluate_scalar_accuracy(
    model: AnticipatoryTransformer,
    dataloader: DataLoader,
    bridge: GeometryBridge,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure how well the model predicts its own anticipation scalars.

    Compares model-predicted scalars against externally computed ground truth.
    """
    model.eval()
    model.to(device)

    scalar_names = [
        "commitment", "uncertainty", "transition_pressure",
        "recovery_margin", "phase_stiffness", "novelty", "stability"
    ]

    all_pred = []
    all_true = []

    for input_ids, _ in dataloader:
        input_ids = input_ids.to(device)

        # Get external ground truth
        true_scalars = bridge.batch_compute(input_ids, model.tok_emb).to(device)

        # Get model predictions
        result = model(input_ids, use_commitment_gate=False)
        pred_scalars = result["predicted_scalars"]

        all_pred.append(pred_scalars.cpu())
        all_true.append(true_scalars.cpu())

    pred = torch.cat(all_pred, dim=0)  # (N, seq, 7)
    true = torch.cat(all_true, dim=0)

    # Per-scalar MSE
    per_scalar_mse = {}
    per_scalar_corr = {}
    for i, name in enumerate(scalar_names):
        p = pred[:, :, i].flatten()
        t = true[:, :, i].flatten()
        mse = F.mse_loss(p, t).item()
        per_scalar_mse[name] = mse

        if p.std() > 1e-6 and t.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
        else:
            corr = 0.0
        per_scalar_corr[name] = corr

    return {
        "per_scalar_mse": per_scalar_mse,
        "per_scalar_correlation": per_scalar_corr,
        "total_mse": F.mse_loss(pred, true).item(),
    }


@torch.no_grad()
def evaluate_orthogonality(
    model: AnticipatoryTransformer,
) -> dict[str, float]:
    """Measure trajectory bias head diversity.

    Each attention head should use a different aspect of the anticipation
    geometry. We measure this by looking at the weight matrix of the
    trajectory bias network.
    """
    # Get the first layer weights
    first_layer = model.trajectory_bias_net.net[0]
    W = first_layer.weight.detach()  # (hidden, 7)

    n_heads = model.config.n_heads
    hidden = W.shape[0]
    per_head_dim = hidden // n_heads

    # Reshape to per-head
    per_head = W.view(n_heads, per_head_dim, -1)
    # Reduce each head to a single vector (mean over per_head_dim)
    head_vecs = per_head.mean(dim=1)  # (n_heads, 7)
    head_vecs = F.normalize(head_vecs, dim=-1)

    # Gram matrix
    gram = head_vecs @ head_vecs.T
    eye = torch.eye(n_heads)

    # Off-diagonal mean (should be near 0 for diverse heads)
    off_diag_mask = ~eye.bool()
    off_diag_mean = gram[off_diag_mask].abs().mean().item()

    # Which scalar each head relies on most
    head_dominant_scalars = head_vecs.abs().argmax(dim=-1).tolist()

    scalar_names = [
        "commitment", "uncertainty", "transition_pressure",
        "recovery_margin", "phase_stiffness", "novelty", "stability"
    ]

    return {
        "off_diagonal_mean": off_diag_mean,
        "head_dominant_scalars": [scalar_names[s] for s in head_dominant_scalars],
        "ortho_penalty": model.trajectory_bias_net.orthogonality_penalty().item(),
    }


# ---------------------------------------------------------------------------
# Full Comparison
# ---------------------------------------------------------------------------

def run_comparison(
    config: AnticipatoryConfig,
    train_steps: int = 200,
    device: str = "cpu",
    seq_len: int = 32,
) -> dict:
    """Train and compare anticipatory vs baseline transformer.

    Both models are trained on the same data for the same number of steps.
    Then we compare them on all metrics.
    """
    print("=" * 70)
    print("ANTICIPATORY TRANSFORMER vs BASELINE COMPARISON")
    print("=" * 70)

    # Create data
    train_ds = make_synthetic_dataset(seq_len, n_seqs=200, vocab_size=config.vocab_size)
    val_ds = make_synthetic_dataset(seq_len, n_seqs=40, vocab_size=config.vocab_size)
    eval_loader = DataLoader(val_ds, batch_size=8, shuffle=False, drop_last=True)

    bridge = GeometryBridge(k_neighbors=3)

    # --- Train Anticipatory Model ---
    print("\n>>> Training Anticipatory Transformer")
    antic_model = AnticipatoryTransformer(config)
    antic_params = antic_model.count_parameters()["total"]
    print(f"    Parameters: {antic_params:,}")

    antic_trainer = Trainer(
        model=antic_model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=1e-3,
        max_steps=train_steps,
        log_interval=max(1, train_steps // 10),
        eval_interval=train_steps,
        device=device,
        use_external_scalars=True,
    )
    antic_history = antic_trainer.train()

    # --- Train Baseline Model ---
    print("\n>>> Training Baseline Transformer")
    baseline_model = BaselineTransformer(config)
    base_params = baseline_model.count_parameters()["total"]
    print(f"    Parameters: {base_params:,}")

    # Use same trainer but without external scalars
    base_trainer = Trainer(
        model=AnticipatoryTransformer(config),  # same arch for fair training
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=1e-3,
        max_steps=train_steps,
        log_interval=max(1, train_steps // 10),
        eval_interval=train_steps,
        device=device,
        use_external_scalars=False,
    )
    base_history = base_trainer.train()

    # Also train the minimal baseline
    base_trainer_min = Trainer(
        model=BaselineTransformer(config),
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=1e-3,
        max_steps=train_steps,
        log_interval=max(1, train_steps // 10),
        eval_interval=train_steps,
        device=device,
        use_external_scalars=False,
    )
    base_min_history = base_trainer_min.train()

    # --- Evaluate ---
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    results = {}

    # 1. Perplexity
    print("\n--- 1. Perplexity ---")
    antic_ppl = evaluate_perplexity(
        antic_model, eval_loader, device, is_anticipatory=True,
    )
    base_ppl = evaluate_perplexity(
        base_trainer.model, eval_loader, device, is_anticipatory=True,
    )
    base_min_ppl = evaluate_perplexity(
        base_trainer_min.model, eval_loader, device, is_anticipatory=False,
    )

    print(f"  Anticipatory:           PPL={antic_ppl['perplexity']:.2f} (loss={antic_ppl['avg_loss']:.4f})")
    print(f"  Same arch, no scalars:  PPL={base_ppl['perplexity']:.2f} (loss={base_ppl['avg_loss']:.4f})")
    print(f"  Minimal baseline:       PPL={base_min_ppl['perplexity']:.2f} (loss={base_min_ppl['avg_loss']:.4f})")
    results["perplexity"] = {
        "anticipatory": antic_ppl,
        "no_scalars": base_ppl,
        "baseline": base_min_ppl,
    }

    # 2. Commitment correlation
    print("\n--- 2. Commitment Correlation ---")
    commit_metrics = evaluate_commitment_correlation(
        antic_model, eval_loader, device,
    )
    print(f"  Commitment-loss correlation: {commit_metrics['commit_loss_correlation']:+.4f}")
    print(f"    (negative = good: high commitment -> low loss)")
    print(f"  Commitment-gate correlation: {commit_metrics['commit_gate_correlation']:+.4f}")
    print(f"    (positive = good: gate tracks commitment)")
    print(f"  Commitment stats: mean={commit_metrics['commit_mean']:.3f}, std={commit_metrics['commit_std']:.3f}")
    print(f"  Per-bin losses: {['%.3f' % x for x in commit_metrics['per_bin_loss']]}")
    results["commitment"] = commit_metrics

    # 3. Context efficiency
    print("\n--- 3. Context Efficiency ---")
    ctx_sizes = [4, 8, 16, 32]
    antic_ctx = evaluate_context_efficiency(
        antic_model, eval_loader, device, True, ctx_sizes,
    )
    base_ctx = evaluate_context_efficiency(
        base_trainer_min.model, eval_loader, device, False, ctx_sizes,
    )

    print(f"  {'Context':>8} | {'Anticipatory':>12} | {'Baseline':>12} | {'Delta':>8}")
    print(f"  {'-' * 50}")
    for i, ctx in enumerate(ctx_sizes):
        a_loss = antic_ctx["losses"][i]
        b_loss = base_ctx["losses"][i]
        delta = a_loss - b_loss
        print(f"  {ctx:>8} | {a_loss:>12.4f} | {b_loss:>12.4f} | {delta:>+8.4f}")
    results["context_efficiency"] = {
        "anticipatory": antic_ctx,
        "baseline": base_ctx,
    }

    # 4. Scalar prediction accuracy
    print("\n--- 4. Scalar Prediction Accuracy ---")
    scalar_acc = evaluate_scalar_accuracy(
        antic_model, eval_loader, bridge, device,
    )
    print(f"  Total MSE: {scalar_acc['total_mse']:.6f}")
    print(f"  Per-scalar MSE and correlation:")
    for name in scalar_acc["per_scalar_mse"]:
        mse = scalar_acc["per_scalar_mse"][name]
        corr = scalar_acc["per_scalar_correlation"][name]
        print(f"    {name:<22}: MSE={mse:.6f}, corr={corr:+.4f}")
    results["scalar_accuracy"] = scalar_acc

    # 5. Orthogonality
    print("\n--- 5. Head Orthogonality ---")
    ortho = evaluate_orthogonality(antic_model)
    print(f"  Off-diagonal gram mean: {ortho['off_diagonal_mean']:.4f} (lower = more diverse)")
    print(f"  Ortho penalty: {ortho['ortho_penalty']:.6f}")
    print(f"  Head dominant scalars: {ortho['head_dominant_scalars']}")
    results["orthogonality"] = ortho

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ppl_improvement = (
        (base_min_ppl["perplexity"] - antic_ppl["perplexity"])
        / base_min_ppl["perplexity"]
        * 100
    )
    print(f"  Perplexity improvement: {ppl_improvement:+.1f}%")
    print(f"  Commitment is informative: {commit_metrics['commit_loss_correlation'] < -0.05}")
    print(f"  Heads are diverse: {ortho['off_diagonal_mean'] < 0.5}")
    unique_scalars = len(set(ortho["head_dominant_scalars"]))
    print(f"  Unique dominant scalars: {unique_scalars}/{config.n_heads}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Anticipatory Transformer")
    parser.add_argument("--quick", action="store_true", help="Quick comparison (fewer steps)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    if args.quick:
        config = AnticipatoryConfig().small()
        run_comparison(
            config,
            train_steps=50,
            device=args.device,
            seq_len=args.seq_len,
        )
    elif args.checkpoint:
        # Load and evaluate a saved checkpoint
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        config = ckpt["config"]
        model = AnticipatoryTransformer(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(args.device)

        val_ds = make_synthetic_dataset(args.seq_len, n_seqs=50, vocab_size=config.vocab_size)
        eval_loader = DataLoader(val_ds, batch_size=8, shuffle=False, drop_last=True)
        bridge = GeometryBridge(k_neighbors=3)

        print("\n--- Perplexity ---")
        ppl = evaluate_perplexity(model, eval_loader, args.device, True)
        print(f"  PPL: {ppl['perplexity']:.2f}")

        print("\n--- Commitment ---")
        commit = evaluate_commitment_correlation(model, eval_loader, args.device)
        print(f"  Commit-loss corr: {commit['commit_loss_correlation']:+.4f}")

        print("\n--- Scalar Accuracy ---")
        scalar = evaluate_scalar_accuracy(model, eval_loader, bridge, args.device)
        print(f"  Total MSE: {scalar['total_mse']:.6f}")

        print("\n--- Orthogonality ---")
        ortho = evaluate_orthogonality(model)
        print(f"  Off-diag mean: {ortho['off_diagonal_mean']:.4f}")
    else:
        config = AnticipatoryConfig().small()
        run_comparison(
            config,
            train_steps=args.train_steps,
            device=args.device,
            seq_len=args.seq_len,
        )


if __name__ == "__main__":
    main()
