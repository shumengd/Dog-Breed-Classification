"""
Helper file for common training functions.
"""

import os
from pathlib import Path
import re

import matplotlib
import torch
from torch import nn
from torch.nn.functional import softmax
from sklearn import metrics

import utils


__all__ = [
    "count_parameters",
    "save_checkpoint",
    "restore_checkpoint",
    "select_checkpoint",
    "clear_checkpoint",
    "early_stopping",
    "evaluate_epoch",
    "train_epoch",
    "predictions",
]


def count_parameters(model: torch.nn.Module) -> int:
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module, epoch: int, checkpoint_dir: str, stats: list
) -> None:
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": [[float(value) for value in row] for row in stats],
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, filename)
    if stats and (len(stats) == 1 or stats[-1][1] < min(row[1] for row in stats[:-1])):
        torch.save(state, os.path.join(checkpoint_dir, "best.checkpoint.pth.tar"))


def select_checkpoint(
    checkpoint_dir: str, selection: str = "best", epoch: int | None = None
) -> Path:
    """Select a saved epoch by validation loss, latest epoch, or explicit number."""
    directory = Path(checkpoint_dir)
    if selection not in {"best", "latest"}:
        raise ValueError("Checkpoint selection must be 'best' or 'latest'")
    if epoch is not None:
        path = directory / f"epoch={epoch}.checkpoint.pth.tar"
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if selection == "best" and (directory / "best.checkpoint.pth.tar").is_file():
        return directory / "best.checkpoint.pth.tar"
    files = [
        path
        for path in directory.glob("epoch=*.checkpoint.pth.tar")
        if re.fullmatch(r"epoch=\d+\.checkpoint\.pth\.tar", path.name)
    ]
    if not files:
        raise FileNotFoundError(
            f"No checkpoints in {directory}. Train the model first."
        )
    if selection == "latest":
        return max(files, key=lambda path: int(path.name.split("=")[1].split(".")[0]))
    # Legacy runs have no best-checkpoint file. Each epoch stores its validation loss.
    return min(
        files,
        key=lambda path: (
            float(
                torch.load(path, map_location="cpu", weights_only=False)["stats"][-1][1]
            ),
            int(path.name.split("=")[1].split(".")[0]),
        ),
    )


def restore_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str,
    cuda: bool = False,
    force: bool = False,
    pretrain: bool = False,
    selection: str = "best",
    epoch: int | None = None,
) -> tuple[nn.Module, int, list]:
    """Load a local checkpoint without prompting.

    Evaluation defaults to the lowest validation loss. Training callers can request
    the latest epoch. Pretraining copies only the three matching convolutions and
    starts a new history for the target task. Only load trusted local checkpoints.
    """
    try:
        filename = select_checkpoint(checkpoint_dir, selection, epoch)
    except FileNotFoundError:
        if force or pretrain or epoch is not None:
            raise
        print("No saved model parameters found; starting from scratch")
        return model, 0, []

    device = "cuda" if cuda else "cpu"
    checkpoint = torch.load(filename, map_location=device, weights_only=False)
    state = checkpoint["state_dict"]
    if pretrain:
        target_state = model.state_dict()
        conv_state = {
            key: value
            for key, value in state.items()
            if key.split(".")[0] in {"conv1", "conv2", "conv3"}
        }
        expected = {
            key
            for key in target_state
            if key.split(".")[0] in {"conv1", "conv2", "conv3"}
        }
        if (
            not expected
            or set(conv_state) != expected
            or any(conv_state[key].shape != target_state[key].shape for key in expected)
        ):
            raise ValueError("Source and target convolutional layers must match")
        model.load_state_dict(conv_state, strict=False)
        print(f"Transferred convolutional weights from {filename}")
        return model, 0, []
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {filename}")
    return model, int(checkpoint["epoch"]), checkpoint["stats"]


def clear_checkpoint(checkpoint_dir: str) -> None:
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def early_stopping(
    stats: list, curr_count_to_patience: int, prev_val_loss: float
) -> tuple[int, float]:
    """Calculate new patience and validation loss.

    Increment curr_count_to_patience by one if new loss is not less than prev_val_loss
    Otherwise, update prev_val_loss with the current val loss, and reset curr_count_to_patience to 0

    Returns: new values of curr_count_to_patience and prev_val_loss
    """
    curr_val_loss = stats[-1][1]
    if curr_val_loss < prev_val_loss:
        curr_count_to_patience = 0
        prev_val_loss = curr_val_loss
    else:
        curr_count_to_patience += 1
    return curr_count_to_patience, prev_val_loss


def evaluate_epoch(
    axes: matplotlib.axes.Axes,
    tr_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    te_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    epoch: int,
    stats: list,
    include_test: bool = False,
    update_plot: bool = True,
    multiclass: bool = False,
) -> None:
    """Evaluate the `model` on the train and validation set."""

    def _get_metrics(loader):
        y_true, y_score = [], []
        correct, total = 0, 0
        running_loss = 0.0
        for X, y in loader:
            with torch.no_grad():
                output = model(X)
                predicted = predictions(output)
                y_true.append(y)
                if not multiclass:
                    y_score.append(softmax(output, dim=1)[:, 1])
                else:
                    y_score.append(softmax(output, dim=1))
                total += y.size(0)
                correct += (predicted == y).sum().item()
                running_loss += criterion(output, y).item() * y.size(0)

        y_true = torch.cat(y_true)
        y_score = torch.cat(y_score)
        loss = running_loss / total
        acc = correct / total
        if not multiclass:
            auroc = metrics.roc_auc_score(y_true, y_score)
        else:
            auroc = metrics.roc_auc_score(y_true, y_score, multi_class="ovo")
        return acc, loss, auroc

    was_training = model.training
    model.eval()
    try:
        train_acc, train_loss, train_auc = _get_metrics(tr_loader)
        val_acc, val_loss, val_auc = _get_metrics(val_loader)
        test_metrics = _get_metrics(te_loader) if include_test else None
    finally:
        model.train(was_training)

    stats_at_epoch = [
        val_acc,
        val_loss,
        val_auc,
        train_acc,
        train_loss,
        train_auc,
    ]
    if include_test:
        stats_at_epoch += list(test_metrics)

    stats.append(stats_at_epoch)
    utils.log_training(epoch, stats)
    if update_plot:
        utils.update_training_plot(axes, epoch, stats)


def train_epoch(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Train the `model` for one epoch of data from `data_loader`.

    Args:
        data_loader: DataLoader providing batches of input data and corresponding labels.
        model: The model to be trained. This is one of the model classes in the 'model' folder.
        criterion: The loss function used to compute the model's loss.
        optimizer: The optimizer used to update the model parameters.

    Description:
        This function sets the model to training mode and use the data loader to iterate through the entire dataset.
        For each batch, it performs the following steps:
        1. Resets the gradient calculations in the optimizer.
        2. Performs a forward pass to get the model predictions.
        3. Computes the loss between predictions and true labels using the specified `criterion`.
        4. Performs a backward pass to calculate gradients.
        5. Updates the model weights using the `optimizer`.
    """
    model.train()
    for X, y in data_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def predictions(logits: torch.Tensor) -> torch.Tensor:
    """Determine predicted class index given logits.

    args:
        logits: The model's output logits. It is a 2D tensor of shape (batch_size, num_classes).

    Returns:
        the predicted class output that has the highest probability. This should be of size (batch_size,).
    """
    return torch.argmax(logits, dim=1)
