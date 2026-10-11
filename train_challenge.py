"""
Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import os
import json
from pathlib import Path
import shutil
import matplotlib.pyplot as plt

from dataset import get_train_val_test_loaders
from model.challenge import Challenge
from train_common import (
    evaluate_epoch,
    early_stopping,
    restore_checkpoint,
    save_checkpoint,
    select_checkpoint,
)
from train_target import freeze_layers
from utils import config, set_random_seed, make_training_plot


def transfer_source_weights(
    challenge_model: Challenge, source_ckpt_dir: str
) -> Challenge:
    """Load the best source convolutional weights and retain the binary head."""
    challenge_model, _, _ = restore_checkpoint(
        challenge_model, source_ckpt_dir, force=True, pretrain=True
    )
    return challenge_model


# Augmentation (training only)
def augment_batch(imgs: torch.Tensor) -> torch.Tensor:
    """Apply RandomHorizontalFlip to a batch of standardized tensors."""
    hflip = T.RandomHorizontalFlip(p=0.5)
    return torch.stack([hflip(img) for img in imgs])


# Per-experiment training
def run_experiment(
    exp_name: str,
    n_freeze: int,
    dropout: float,
    weight_decay: float,
    lr: float,
    patience: int,
) -> tuple[float, int]:
    set_random_seed()

    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("challenge.batch_size"),
    )

    # Build model
    model = Challenge(dropout=dropout)
    print(f"\n[{exp_name}] Transferring source weights")
    model = transfer_source_weights(model, config("source.checkpoint"))

    # Freeze the requested number of source conv layers
    freeze_layers(model, n_freeze)
    frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
    print(f"[{exp_name}] Frozen params: {frozen_names}")

    # Optimizer only updates unfrozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    # Per-experiment checkpoint directory
    ckpt_dir = f"checkpoints/challenge_{exp_name}/"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Start fresh (do not restore old challenge checkpoints)
    stats = []
    start_epoch = 0

    axes = make_training_plot()

    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
    )
    save_checkpoint(model, start_epoch, ckpt_dir, stats)

    prev_val_loss = stats[0][1]
    curr_patience = 0
    epoch = start_epoch

    while curr_patience < patience:
        model.train()
        for X, y in tr_loader:
            X = augment_batch(X)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
        )

        save_checkpoint(model, epoch + 1, ckpt_dir, stats)

        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        epoch += 1

    # History starts with the untrained target head at epoch zero.
    best_idx = min(range(len(stats)), key=lambda i: stats[i][1])
    best_val_loss = stats[best_idx][1]
    best_val_auroc = stats[best_idx][2]
    best_epoch = best_idx

    print(
        f"[{exp_name}] Finished | best epoch={best_epoch} | "
        f"val_loss={best_val_loss:.4f} | val_auroc={best_val_auroc:.4f}"
    )

    plt.savefig(f"challenge_training_{exp_name}.png", dpi=200)
    plt.close()

    return best_val_auroc, best_epoch


def main():
    set_random_seed()

    # Freeze
    print("Step1: Freeze strategy")

    freeze_experiments = [
        # (exp_name,       n_freeze, dropout, weight_decay, lr,   patience)
        ("freeze3_all", 3, 0.3, 1e-4, 1e-3, 10),
        ("freeze2", 2, 0.3, 1e-4, 1e-3, 10),
        ("freeze1", 1, 0.3, 1e-4, 1e-3, 10),
        ("freeze0_full", 0, 0.3, 1e-4, 5e-4, 10),
    ]

    results1 = []
    for args in freeze_experiments:
        auroc, epoch = run_experiment(*args)
        results1.append(
            {"name": args[0], "n_freeze": args[1], "auroc": auroc, "epoch": epoch}
        )

    best_freeze_result = max(results1, key=lambda r: r["auroc"])
    best_n_freeze = best_freeze_result["n_freeze"]
    print(
        f"\nBest freeze setting: n_freeze={best_n_freeze} "
        f"(val AUROC={best_freeze_result['auroc']:.4f})"
    )

    # Regularization
    print("Step 2: Regularization")

    reg_experiments = []
    for dropout in [0.0, 0.3, 0.5]:
        for wd in [1e-4, 3e-4]:
            name = f"f{best_n_freeze}_do{int(dropout * 10)}_wd{wd:.0e}"
            reg_experiments.append((name, best_n_freeze, dropout, wd, 1e-3, 10))

    results2 = []
    for args in reg_experiments:
        auroc, epoch = run_experiment(*args)
        results2.append(
            {
                "name": args[0],
                "dropout": args[2],
                "weight_decay": args[3],
                "auroc": auroc,
                "epoch": epoch,
            }
        )

    all_results = [
        (r["name"], f"freeze={r['n_freeze']}", "—", "—", r["auroc"], r["epoch"])
        for r in results1
    ] + [
        (
            r["name"],
            f"freeze={best_n_freeze}",
            str(r["dropout"]),
            f"{r['weight_decay']:.0e}",
            r["auroc"],
            r["epoch"],
        )
        for r in results2
    ]

    print("Summary of all experiments: \n")

    print(
        f"{'Name':<28} {'Setting':<12} {'DO':>5} {'WD':>6} "
        f"{'BestEp':>7} {'ValAUROC':>10}"
    )

    for name, setting, do, wd, auroc, ep in all_results:
        print(f"{name:<28} {setting:<12} {do:>5} {wd:>6} {ep:>7d} {auroc:>10.4f}")

    overall_best = max(all_results, key=lambda r: r[4])
    print(
        f"\nBest overall: {overall_best[0]} "
        f"(val AUROC={overall_best[4]:.4f}, epoch {overall_best[5]})"
    )
    print("\nFinished all experiments.")

    best_ckpt_src = select_checkpoint(f"checkpoints/challenge_{overall_best[0]}/")
    best_ckpt_dst = Path(config("challenge.checkpoint"))
    best_ckpt_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt_src, best_ckpt_dst / "best.checkpoint.pth.tar")
    results_path = Path("results/experiments.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "freeze_experiments": results1,
                "regularization_experiments": results2,
                "selected_experiment": overall_best[0],
                "selected_epoch": overall_best[5],
                "validation_auroc": overall_best[4],
            },
            indent=2,
        )
        + "\n"
    )

    # Evaluate the selected configuration once on held-out test data.
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target", batch_size=config("challenge.batch_size")
    )
    selected_model, selected_epoch, _ = restore_checkpoint(
        Challenge(), str(best_ckpt_dst), force=True
    )
    evaluate_epoch(
        None,
        tr_loader,
        va_loader,
        te_loader,
        selected_model,
        nn.CrossEntropyLoss(),
        selected_epoch,
        [],
        include_test=True,
        update_plot=False,
    )


if __name__ == "__main__":
    main()
