"""Tests for single-image inference, portable preprocessing, and demo requests."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image
import pytest
import torch

from dataset import ImageStandardizer
from demo import DemoApplication
from inference import DogBreedPredictor, preprocess_image
from model.target import Target


@pytest.fixture
def predictor():
    standardizer = ImageStandardizer()
    standardizer.fit(np.arange(48, dtype=np.uint8).reshape(1, 4, 4, 3))
    return DogBreedPredictor(
        Target(), standardizer, "transfer", "Frozen CNN", "target", 5
    )


def encoded_image(mode="RGB", image_format="PNG"):
    image = Image.new(mode, (90, 120))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return image, base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_uploaded_images_become_normalized_rgb_tensors(predictor, mode):
    tensor = preprocess_image(Image.new(mode, (90, 120)), predictor.standardizer)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == torch.float32
    expected = -predictor.standardizer.image_mean / predictor.standardizer.image_std
    assert np.allclose(tensor[0, :, 0, 0], expected)


def test_model_bundle_round_trip_needs_no_dataset(predictor, tmp_path, monkeypatch):
    path = predictor.export(tmp_path / "classifier.bundle.pt")

    def unavailable():
        raise AssertionError("Inference bundles must not access the training dataset")

    monkeypatch.setattr("inference.fit_target_standardizer", unavailable)
    restored = DogBreedPredictor.from_bundle(path)
    image = Image.new("RGB", (128, 128), (160, 80, 20))
    original = predictor.predict(image)
    result = restored.predict(image)
    assert result["probabilities"] == original["probabilities"]
    assert result["predicted_label"] == original["predicted_label"]
    assert sum(
        item["probability"] for item in result["probabilities"]
    ) == pytest.approx(1)
    assert result["epoch"] == 5
    assert not restored.model.training


@pytest.mark.parametrize(
    "normalization",
    [
        {"mean": [0, 0], "std": [1, 1, 1]},
        {"mean": [0, 0, 0], "std": [-1, 1, 1]},
        {"mean": [float("nan"), 0, 0], "std": [1, 1, 1]},
    ],
)
def test_bundle_rejects_invalid_normalization(predictor, tmp_path, normalization):
    path = predictor.export(tmp_path / "bad.bundle.pt")
    bundle = torch.load(path, weights_only=True)
    bundle["normalization"] = normalization
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="normalization"):
        DogBreedPredictor.from_bundle(path)


def test_demo_image_prediction_matches_direct_inference(
    predictor, tmp_path, monkeypatch
):
    monkeypatch.setattr("demo.PROJECT_ROOT", tmp_path)
    application = DemoApplication(tmp_path / "bundles")
    application.predictors["transfer"] = predictor
    image, encoded = encoded_image()
    result = application.predict(
        {"model": "transfer", "image": encoded, "filename": "dog.png"}
    )
    assert result["probabilities"] == predictor.predict(image)["probabilities"]
    assert result["filename"] == "dog.png"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"model": "unknown", "image": "AA=="},
        {"image": "invalid-base64!"},
        {"image": "AA=="},
        {"sample": "missing"},
        {"sample": ["invalid"]},
    ],
)
def test_demo_rejects_invalid_requests(tmp_path, monkeypatch, payload):
    monkeypatch.setattr("demo.PROJECT_ROOT", tmp_path)
    application = DemoApplication(tmp_path / "bundles")
    with pytest.raises((ValueError, OSError)):
        application.predict(payload)


def test_demo_is_unavailable_without_weights_or_training_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("demo.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("inference.PROJECT_ROOT", tmp_path)
    application = DemoApplication(tmp_path / "bundles")
    status = application.status()
    assert not status["ready"]
    assert status["default_model"] is None
    assert status["samples"] == []
    assert all(not model["available"] for model in status["models"])


def test_demo_can_use_bundle_without_csv(predictor, tmp_path, monkeypatch):
    monkeypatch.setattr("demo.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("inference.PROJECT_ROOT", tmp_path)
    predictor.export(tmp_path / "bundles/transfer.bundle.pt")
    application = DemoApplication(tmp_path / "bundles")
    assert application.status()["ready"]
    assert application.status()["default_model"] == "transfer"
    image, encoded = encoded_image()
    assert (
        application.predict({"image": encoded})["probabilities"]
        == predictor.predict(image)["probabilities"]
    )
