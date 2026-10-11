"""
Predict Challenge
    Runs the challenge model inference on the test dataset and saves the
    predictions to disk
    Usage: python predict_challenge.py --output results/predictions.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.nn.functional import softmax

from dataset import get_challenge
from model.challenge import Challenge
from train_common import restore_checkpoint
from utils import config


def predict_challenge(
    data_loader: torch.utils.data.DataLoader, model: torch.nn.Module
) -> torch.Tensor:
    """Runs the model inference on the test set and outputs the predictions."""
    y_score = []
    model.eval()
    with torch.inference_mode():
        for X, _ in data_loader:
            output = model(X)
            y_score.append(softmax(output, dim=1)[:, 1].cpu())
    return torch.cat(y_score)


def main(output: str) -> None:
    """Export positive-class probabilities from the best selected model."""
    # data loaders
    ch_loader, _ = get_challenge(
        task="target",
        batch_size=config("challenge.batch_size"),
    )

    model = Challenge()

    # Load the checkpoint selected by minimum validation loss.
    model, _, _ = restore_checkpoint(model, config("challenge.checkpoint"), force=True)

    # Evaluate model
    model_pred = predict_challenge(ch_loader, model)

    print("Saving challenge predictions...")

    output_path = Path(output)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Prediction output must use a .csv extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predictions": model_pred.numpy()}).to_csv(output_path, index=False)
    print(f"Saved {len(model_pred)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/predictions.csv", help="output CSV path"
    )
    args = parser.parse_args()
    main(args.output)
