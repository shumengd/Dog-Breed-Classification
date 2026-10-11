"""
Train Target
    Train a convolutional neural network to classify images.
    Periodically output training information, and saves model checkpoints
    Usage: python train_target.py
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import get_train_val_test_loaders
from model.target import Target
from train_common import (
    evaluate_epoch,
    early_stopping,
    restore_checkpoint,
    save_checkpoint,
    train_epoch,
)
from utils import config, make_training_plot, set_random_seed


__all__ = ["freeze_layers", "train"]


def freeze_layers(model: torch.nn.Module, num_layers: int = 0) -> None:
    """
    This function modifies 'model' settings to stop tracking gradients on selected layers.
    The first num_layers convolutions are frozen; remaining layers stay trainable.

    Args:
        model: subclass of nn.Module
        num_layers: int, the number of conv layers to freeze
    """
    if num_layers not in range(4):
        raise ValueError("num_layers must be between 0 and 3")
    conv_layer_names = [f"conv{i}" for i in range(1, num_layers + 1)]
    for name, param in model.named_parameters():
        layer_name = name.split(".")[0]
        param.requires_grad = layer_name not in conv_layer_names


def train(
    tr_loader: DataLoader,
    va_loader: DataLoader,
    te_loader: DataLoader,
    model: torch.nn.Module,
    model_name: str,
    num_layers: int = 0,
) -> None:
    """
    This function trains the target model. Only the weights of unfrozen layers of the model passed
    into this function will be updated in training.

    Args:
        tr_loader: DataLoader for training data
        va_loader: DataLoader for validation data
        te_loader: DataLoader for test data
        model: subclass of torch.nn.Module, model to train on
        model_name: str, checkpoint path for the model
        num_layers: int, the number of source model layers to freeze
    """
    set_random_seed()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config("target.learning_rate"),
    )

    print("Loading target model with", num_layers, "layers frozen")
    model, start_epoch, stats = restore_checkpoint(
        model, model_name, selection="latest"
    )

    axes = make_training_plot("Target Training")

    if not stats:
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats
        )
        save_checkpoint(model, start_epoch, model_name, stats)

    # initial val loss for early stopping
    prev_val_loss = min(row[1] for row in stats)

    patience = 5
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
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)

        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        epoch += 1

    print("Finished Training")
    model, best_epoch, _ = restore_checkpoint(model, model_name, force=True)
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
    )

    # Keep plot open
    print(
        f"Saving training plot to target_training_plot_frozen_layers={num_layers}.png..."
    )
    plt.savefig(f"target_training_plot_frozen_layers={num_layers}.png", dpi=200)
    plt.ioff()
    plt.show()


def main() -> None:
    """
    Train transfer learning model and display training plots.

    Train four different models with {0, 1, 2, 3} layers frozen.
    """
    set_random_seed()
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("target.batch_size"),
    )

    freeze_none = Target()
    print("Loading source...")
    freeze_none, _, _ = restore_checkpoint(
        freeze_none,
        config("source.checkpoint"),
        force=True,
        pretrain=True,
    )

    freeze_one = deepcopy(freeze_none)
    freeze_two = deepcopy(freeze_none)
    freeze_three = deepcopy(freeze_none)

    freeze_layers(freeze_one, 1)
    freeze_layers(freeze_two, 2)
    freeze_layers(freeze_three, 3)

    train(
        tr_loader,
        va_loader,
        te_loader,
        freeze_none,
        config("target.frozen_checkpoint").format(layer=0),
        0,
    )
    train(
        tr_loader,
        va_loader,
        te_loader,
        freeze_one,
        config("target.frozen_checkpoint").format(layer=1),
        1,
    )
    train(
        tr_loader,
        va_loader,
        te_loader,
        freeze_two,
        config("target.frozen_checkpoint").format(layer=2),
        2,
    )
    train(
        tr_loader,
        va_loader,
        te_loader,
        freeze_three,
        config("target.frozen_checkpoint").format(layer=3),
        3,
    )


if __name__ == "__main__":
    main()
