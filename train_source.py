"""
Train Source CNN
    Train a convolutional neural network to classify images.
    Periodically output training information, and save model checkpoints.
    Usage: python train_source.py
"""

import torch
import matplotlib.pyplot as plt

from train_common import (
    count_parameters,
    evaluate_epoch,
    early_stopping,
    restore_checkpoint,
    save_checkpoint,
    train_epoch,
)
from dataset import get_train_val_test_loaders
from model.source import Source
from utils import config, set_random_seed, make_training_plot


def main() -> None:
    """Train source model on multiclass data."""
    set_random_seed()

    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="source",
        batch_size=config("source.batch_size"),
    )

    # Model
    model = Source()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config("source.learning_rate"), weight_decay=0.01
    )

    print("Number of float-valued parameters:", count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print("Loading source...")
    model, start_epoch, stats = restore_checkpoint(
        model, config("source.checkpoint"), selection="latest"
    )

    axes = make_training_plot("Source Training")

    # Evaluate the randomly initialized model
    if not stats:
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            start_epoch,
            stats,
            multiclass=True,
        )
        save_checkpoint(model, start_epoch, config("source.checkpoint"), stats)

    # initial val loss for early stopping
    prev_val_loss = min(row[1] for row in stats)

    patience = 10
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            multiclass=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("source.checkpoint"), stats)

        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        epoch += 1

    # Save figure and keep plot open
    print("Finished Training")
    model, best_epoch, _ = restore_checkpoint(
        model, config("source.checkpoint"), force=True
    )
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        best_epoch,
        [],
        include_test=True,
        update_plot=False,
        multiclass=True,
    )
    plt.savefig(f"source_training_plot_patience={patience}.png", dpi=200)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
