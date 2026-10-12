"""Reproduce model comparisons in a new directory without replacing old weights."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from dataset import get_train_val_test_datasets
from inference import DogBreedPredictor
from model.source import Source
from model.target import Target
from model.vit import ViT
from train_common import (
    count_parameters,
    early_stopping,
    evaluate_epoch,
    restore_checkpoint,
    save_checkpoint,
    train_epoch,
)
from train_target import freeze_layers
from utils import config, set_random_seed


def loaders(task: str):
    train, val, test, standardizer = get_train_val_test_datasets(task)
    batch_size = config(f"{task}.batch_size")
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
        standardizer,
    )


def assess(model, data, criterion, epoch, stats, multiclass=False, test=False):
    with redirect_stdout(StringIO()):
        evaluate_epoch(
            None,
            *data[:3],
            model,
            criterion,
            epoch,
            stats,
            multiclass=multiclass,
            include_test=test,
            update_plot=False,
        )


def train_run(
    name,
    model,
    data,
    output,
    max_epochs,
    patience,
    learning_rate,
    weight_decay=0.0,
    multiclass=False,
):
    checkpoint_dir = output / name
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    stats = []
    started = perf_counter()
    assess(model, data, criterion, 0, stats, multiclass)
    save_checkpoint(model, 0, str(checkpoint_dir), stats)
    count, best_loss = 0, stats[0][1]
    epoch = 0
    while epoch < max_epochs and count < patience:
        epoch += 1
        train_epoch(data[0], model, criterion, optimizer)
        assess(model, data, criterion, epoch, stats, multiclass)
        save_checkpoint(model, epoch, str(checkpoint_dir), stats)
        count, best_loss = early_stopping(stats, count, best_loss)
        if epoch == 1 or epoch % 10 == 0 or count >= patience or epoch == max_epochs:
            print(
                f"{name}: epoch={epoch} val_loss={stats[-1][1]:.4f} val_auroc={stats[-1][2]:.4f}",
                flush=True,
            )
    model, best_epoch, _ = restore_checkpoint(model, str(checkpoint_dir), force=True)
    final = []
    assess(model, data, criterion, best_epoch, final, multiclass, test=True)
    row = final[0]
    result = {
        "model": name,
        "epoch": best_epoch,
        "epochs_trained": epoch,
        "stopped_early": count >= patience,
        "validation_accuracy": row[0],
        "validation_loss": row[1],
        "validation_auroc": row[2],
        "test_accuracy": row[6],
        "test_loss": row[7],
        "test_auroc": row[8],
        "trainable_parameters": count_parameters(model),
        "elapsed_seconds": round(perf_counter() - started, 2),
    }
    result = {
        key: float(value) if isinstance(value, (float,)) else value
        for key, value in result.items()
    }
    print(f"{name}: selected_epoch={best_epoch} test_auroc={row[8]:.4f}", flush=True)
    return model, result


def run(
    output: Path,
    max_epochs: int = 200,
    threads: int = 1,
    export_dir: Path | None = None,
) -> dict:
    if max_epochs <= 0 or threads <= 0:
        raise ValueError("Epoch and thread counts must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty experiment directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    set_random_seed()
    target_data = loaders("target")
    source_data = loaders("source")
    results = []

    def train(name, model, data, patience, multiclass=False):
        return train_run(
            name,
            model,
            data,
            output,
            max_epochs,
            patience,
            learning_rate=config(
                f"{'source' if multiclass else 'target'}.learning_rate"
            ),
            weight_decay=0.01 if multiclass else 0.0,
            multiclass=multiclass,
        )

    set_random_seed()
    baseline, baseline_result = train("cnn_from_scratch", Target(), target_data, 5)
    results.append(baseline_result)
    set_random_seed()
    _, source_result = train("source", Source(), source_data, 10, multiclass=True)
    for frozen in range(4):
        set_random_seed()
        model, _, _ = restore_checkpoint(
            Target(), str(output / "source"), force=True, pretrain=True
        )
        freeze_layers(model, frozen)
        model, result = train(f"transfer_freeze_{frozen}", model, target_data, 5)
        results.append(result)
        if export_dir and frozen == 3:
            DogBreedPredictor(
                model,
                target_data[3],
                "transfer",
                "Frozen CNN",
                "target",
                result["epoch"],
            ).export(export_dir / "transfer.bundle.pt")
    set_random_seed()
    vit, vit_result = train(
        "vit_from_scratch",
        ViT(num_patches=16, num_blocks=2, num_hidden=16, num_heads=2),
        target_data,
        5,
    )
    results.append(vit_result)
    if export_dir:
        DogBreedPredictor(
            baseline,
            target_data[3],
            "cnn",
            "CNN from scratch",
            "target",
            baseline_result["epoch"],
        ).export(export_dir / "cnn.bundle.pt")
        DogBreedPredictor(
            vit, target_data[3], "vit", "Vision Transformer", "vit", vit_result["epoch"]
        ).export(export_dir / "vit.bundle.pt")
    summary = {
        "description": "Reproduced comparisons after evaluation and checkpoint fixes",
        "target_classes": ["golden_retriever", "collie"],
        "target_split_sizes": {
            name: len(loader.dataset)
            for name, loader in zip(("train", "val", "test"), target_data[:3])
        },
        "protocol": {
            "seed": 445,
            "target_cnn_initialization_seed": 42,
            "device": "cpu",
            "threads": threads,
            "max_epochs": max_epochs,
            "selection": "minimum validation loss per run",
            "test_evaluation": "once per run, after checkpoint selection",
            "torch_version": str(torch.__version__),
        },
        "source_result": source_result,
        "results": results,
        "note": "Single-seed reproduction. Each model uses the same target splits and train-only normalization. Test AUROC is reported for comparison, not used to select a checkpoint. These runs are separate from the historical experiments.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Results saved to {output / 'summary.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--export-dir", type=Path, help="optional directory for inference bundles"
    )
    args = parser.parse_args()
    run(args.output, args.max_epochs, args.threads, args.export_dir)


if __name__ == "__main__":
    main()
