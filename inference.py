"""Single-image inference and portable model bundles for the binary target task."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps
import torch

from dataset import DogsDataset, ImageStandardizer
from model.challenge import Challenge
from model.target import Target
from model.vit import ViT
from train_common import select_checkpoint
from utils import config


PROJECT_ROOT = Path(__file__).resolve().parent
LABELS = ("Golden Retriever", "Collie")
IMAGE_SIZE = 64


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    checkpoint_dir: Path


def model_specs() -> dict[str, ModelSpec]:
    return {
        "transfer": ModelSpec(
            "Frozen CNN",
            "target",
            PROJECT_ROOT / config("target.frozen_checkpoint").format(layer=3),
        ),
        "vit": ModelSpec(
            "Vision Transformer", "vit", PROJECT_ROOT / config("vit.checkpoint")
        ),
        "challenge": ModelSpec(
            "Regularized CNN",
            "challenge",
            PROJECT_ROOT / config("challenge.checkpoint"),
        ),
        "cnn": ModelSpec(
            "CNN from scratch", "target", PROJECT_ROOT / config("target.checkpoint")
        ),
    }


def make_model(architecture: str) -> torch.nn.Module:
    if architecture == "target":
        return Target()
    if architecture == "challenge":
        return Challenge()
    if architecture == "vit":
        return ViT(num_patches=16, num_blocks=2, num_hidden=16, num_heads=2)
    raise ValueError(f"Unknown model architecture: {architecture}")


def fit_target_standardizer() -> ImageStandardizer:
    """Reconstruct legacy preprocessing using the target training split only."""
    standardizer = ImageStandardizer()
    standardizer.fit(DogsDataset("train", "target").X)
    return standardizer


def preprocess_image(
    image: Image.Image, standardizer: ImageStandardizer
) -> torch.Tensor:
    """Match RGB channel order and training normalization after resizing uploads."""
    if image.width * image.height > 25_000_000:
        raise ValueError("Use an image with no more than 25 million pixels")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    resized = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    normalized = standardizer.transform(np.asarray(resized)[None, ...])
    return torch.from_numpy(normalized.transpose(0, 3, 1, 2).copy()).float()


class DogBreedPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        standardizer: ImageStandardizer,
        model_id: str,
        model_name: str,
        architecture: str,
        epoch: int,
        labels: tuple[str, str] = LABELS,
    ) -> None:
        self.model = model.cpu().eval()
        self.standardizer = standardizer
        self.model_id = model_id
        self.model_name = model_name
        self.architecture = architecture
        self.epoch = epoch
        self.labels = labels
        self.lock = Lock()

    @classmethod
    def from_checkpoint(
        cls,
        model_id: str = "transfer",
        standardizer: ImageStandardizer | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> "DogBreedPredictor":
        specs = model_specs()
        if model_id not in specs:
            raise ValueError(f"Unknown model: {model_id}")
        spec = specs[model_id]
        path = select_checkpoint(str(checkpoint_dir or spec.checkpoint_dir))
        # Legacy checkpoints contain NumPy statistics. These are trusted local files.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = make_model(spec.architecture)
        model.load_state_dict(checkpoint["state_dict"])
        return cls(
            model,
            standardizer or fit_target_standardizer(),
            model_id,
            spec.name,
            spec.architecture,
            int(checkpoint["epoch"]),
        )

    @classmethod
    def from_bundle(cls, path: str | Path) -> "DogBreedPredictor":
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle.get("format_version") != 1 or bundle.get("image_size") != IMAGE_SIZE:
            raise ValueError("Unsupported model bundle format or image size")
        normalization = bundle["normalization"]
        mean = np.asarray(normalization["mean"], dtype=np.float64)
        std = np.asarray(normalization["std"], dtype=np.float64)
        if (
            mean.shape != (3,)
            or std.shape != (3,)
            or not np.isfinite([mean, std]).all()
            or (std < 0).any()
        ):
            raise ValueError(
                "Model bundle must contain valid RGB normalization statistics"
            )
        labels = tuple(bundle["labels"])
        if len(labels) != 2 or any(
            not isinstance(label, str) or not label for label in labels
        ):
            raise ValueError("Model bundle must contain two class labels")
        standardizer = ImageStandardizer()
        standardizer.image_mean, standardizer.image_std = mean, std
        model = make_model(bundle["architecture"])
        model.load_state_dict(bundle["state_dict"])
        return cls(
            model,
            standardizer,
            bundle["model_id"],
            bundle["model_name"],
            bundle["architecture"],
            int(bundle["epoch"]),
            labels,
        )

    def export(self, path: str | Path) -> Path:
        """Store weights, class order, and preprocessing without dataset files."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_id": self.model_id,
                "model_name": self.model_name,
                "architecture": self.architecture,
                "epoch": self.epoch,
                "image_size": IMAGE_SIZE,
                "labels": list(self.labels),
                "normalization": {
                    "mean": self.standardizer.image_mean.tolist(),
                    "std": self.standardizer.image_std.tolist(),
                },
                "state_dict": self.model.state_dict(),
            },
            destination,
        )
        return destination

    def predict(self, image: Image.Image, filename: str = "image") -> dict:
        started = perf_counter()
        tensor = preprocess_image(image, self.standardizer)
        with self.lock, torch.inference_mode():
            scores = self.model(tensor).softmax(dim=1)[0].tolist()
        predicted = int(np.argmax(scores))
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "epoch": self.epoch,
            "filename": filename,
            "predicted_label": self.labels[predicted],
            "confidence": scores[predicted],
            "probabilities": [
                {"label": label, "probability": score}
                for label, score in zip(self.labels, scores)
            ],
            "image_size": IMAGE_SIZE,
            "inference_ms": round((perf_counter() - started) * 1000, 2),
        }
