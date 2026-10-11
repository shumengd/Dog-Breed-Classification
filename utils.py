"""
Utility functions
"""

import json
import random
from pathlib import Path
from typing import Any

import torch
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    "set_random_seed",
    "config",
    "denormalize_image",
    "log_training",
    "make_training_plot",
    "update_training_plot",
]


SEED = 445


def set_random_seed() -> None:
    """Seed random generators and enable deterministic algorithms."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config(attr: str) -> Any:
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.

    Args:
        attr: the attribute to retrieve from the config file

    Returns:
        the value of the attribute in the config file
    """
    if not hasattr(config, "config"):
        with Path(__file__).with_name("config.json").open() as f:
            config.config = json.load(f)
    node = config.config
    for part in attr.split("."):
        node = node[part]
    return node


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Rescale the image's color space from (min, max) to (0, 1)

    Args:
        image: the image to denormalize

    Returns:
        the denormalized image
    """
    ptp = np.max(image, axis=(0, 1)) - np.min(image, axis=(0, 1))
    return (image - np.min(image, axis=(0, 1))) / ptp


def log_training(epoch: int, stats: list[list[float]]) -> None:
    """Print the train, validation, test accuracy/loss/auroc.

    Args:
        stats: A cumulative list to store the model accuracy, loss, and AUC for every epoch.
            Usage: stats[epoch][0] = validation accuracy, stats[epoch][1] = validation loss, stats[epoch][2] = validation AUC
                    stats[epoch][3] = training accuracy, stats[epoch][4] = training loss, stats[epoch][5] = training AUC
                    stats[epoch][6] = test accuracy, stats[epoch][7] = test loss, stats[epoch][8] = test AUC (test only appears when we are finetuning our target model)

        epoch: The current epoch number.

    Note: Test accuracy is optional and will only be logged if stats is length 9.
    """
    splits = ["Validation", "Train", "Test"]
    metrics = ["Accuracy", "Loss", "AUROC"]
    print("Epoch {}".format(epoch))
    for j, split in enumerate(splits):
        for i, metric in enumerate(metrics):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue
            print(f"\t{split} {metric}:{round(stats[-1][idx], 4)}")


def make_training_plot(name: str = "CNN Training") -> plt.Axes:
    """
    Set up an interactive matplotlib graph to log metrics during training.

    Args:
        name: The name of the training plot.

    Returns:
        axes: The axes of the training
    """
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    plt.suptitle(name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUROC")

    return axes


def update_training_plot(axes: plt.Axes, epoch: int, stats: list[list[float]]) -> None:
    """Update the training plot with a new data point for loss and accuracy."""
    splits = ["Validation", "Train", "Test"]
    metrics = ["Accuracy", "Loss", "AUROC"]
    colors = ["r", "b", "y"]
    styles = ["o", "x", "^"]
    for i, metric in enumerate(metrics):
        for j, split in enumerate(splits):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue
            axes[i].plot(
                range(epoch - len(stats) + 1, epoch + 1),
                [stat[idx] for stat in stats],
                linestyle="--",
                marker=styles[j],
                color=colors[j],
            )
        axes[i].legend(splits[: int(len(stats[-1]) / len(metrics))])
    plt.pause(0.00001)
