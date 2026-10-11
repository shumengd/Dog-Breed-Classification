"""Regression coverage for preprocessing, transfer, evaluation, and selection."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import ImageStandardizer, get_train_val_test_datasets
from model.challenge import Challenge
from model.source import Source
from model.target import Target
from model.vit import ViT, patchify
from predict_challenge import predict_challenge
from train_common import (
    early_stopping,
    evaluate_epoch,
    restore_checkpoint,
    save_checkpoint,
    select_checkpoint,
)
from train_target import freeze_layers
import train_challenge


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_standardizer_handles_constant_channels_and_unfitted_transform():
    standardizer = ImageStandardizer()
    with pytest.raises(RuntimeError, match="fit"):
        standardizer.transform(np.ones((1, 2, 2, 3)))
    standardizer.fit(np.full((2, 2, 2, 3), 5))
    assert np.array_equal(
        standardizer.transform(np.full((1, 2, 2, 3), 5)), 0 * np.ones((1, 2, 2, 3))
    )
    assert np.isfinite(standardizer.transform(np.full((1, 2, 2, 3), 9))).all()


@pytest.mark.parametrize("images", [np.array([]), np.ones((2, 3, 3))])
def test_standardizer_rejects_invalid_training_data(images):
    with pytest.raises(ValueError):
        ImageStandardizer().fit(images)


def test_validation_and_test_use_training_statistics(monkeypatch):
    class SyntheticDataset:
        def __init__(self, partition, task):
            offset = {"train": 0, "val": 50, "test": 100}[partition]
            self.X = np.arange(24, dtype=float).reshape(2, 2, 2, 3) + offset

    monkeypatch.setattr("dataset.DogsDataset", SyntheticDataset)
    train, val, test, standardizer = get_train_val_test_datasets()
    assert np.allclose(train.X.mean(axis=(0, 2, 3)), 0)
    assert np.allclose(val.X.mean(axis=(0, 2, 3)), 50 / standardizer.image_std)
    assert np.allclose(test.X.mean(axis=(0, 2, 3)), 100 / standardizer.image_std)


def test_checkpoint_selects_validation_loss_and_handles_missing_epochs(tmp_path):
    model = Target()
    stats = [[0.5, 0.8, 0.6, 0.5, 0.8, 0.6]]
    save_checkpoint(model, 0, str(tmp_path), stats)
    stats.append([0.5, 0.4, 0.6, 0.5, 0.4, 0.6])
    save_checkpoint(model, 2, str(tmp_path), stats)
    stats.append([0.5, 0.4, 0.6, 0.5, 0.4, 0.6])
    save_checkpoint(model, 3, str(tmp_path), stats)
    stats.append([0.5, 0.7, 0.6, 0.5, 0.7, 0.6])
    save_checkpoint(model, 5, str(tmp_path), stats)
    _, epoch, _ = restore_checkpoint(Target(), str(tmp_path), force=True)
    assert epoch == 2
    assert (
        select_checkpoint(str(tmp_path), "latest").name == "epoch=5.checkpoint.pth.tar"
    )
    assert (
        select_checkpoint(str(tmp_path), epoch=0).name == "epoch=0.checkpoint.pth.tar"
    )
    # Older experiment directories contain only epoch checkpoints.
    (tmp_path / "best.checkpoint.pth.tar").unlink()
    assert select_checkpoint(str(tmp_path)).name == "epoch=2.checkpoint.pth.tar"


def test_required_checkpoint_cannot_silently_use_random_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        restore_checkpoint(Target(), str(tmp_path), force=True)
    with pytest.raises(FileNotFoundError):
        restore_checkpoint(Target(), str(tmp_path), pretrain=True)


@pytest.mark.parametrize(
    "target_type,head_name", [(Target, "fc_1"), (Challenge, "fc1")]
)
def test_source_transfer_copies_convolutions_and_preserves_binary_head(
    tmp_path, target_type, head_name
):
    source = Source()
    save_checkpoint(source, 10, str(tmp_path), [[0.5, 0.3, 0.8, 0.5, 0.2, 0.9]])
    target = target_type()
    head_before = getattr(target, head_name).weight.detach().clone()
    target, epoch, stats = restore_checkpoint(target, str(tmp_path), pretrain=True)
    assert epoch == 0 and stats == []
    for name in ("conv1", "conv2", "conv3"):
        assert torch.equal(getattr(target, name).weight, getattr(source, name).weight)
        assert torch.equal(getattr(target, name).bias, getattr(source, name).bias)
    assert torch.equal(getattr(target, head_name).weight, head_before)


def test_source_transfer_rejects_incompatible_convolutions(tmp_path):
    source = Source()
    source.conv1 = nn.Conv2d(3, 32, 5)
    save_checkpoint(source, 1, str(tmp_path), [[0.5, 0.4, 0.6]])
    with pytest.raises(ValueError, match="must match"):
        restore_checkpoint(Target(), str(tmp_path), pretrain=True)


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_freezing_controls_gradients_and_keeps_head_trainable(count):
    model = Target()
    freeze_layers(model, count)
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    for index in range(1, 4):
        conv = getattr(model, f"conv{index}")
        assert conv.weight.requires_grad == (index > count)
        assert (conv.weight.grad is None) == (index <= count)
    assert model.fc_1.weight.grad is not None
    freeze_layers(model, 0)
    assert all(param.requires_grad for param in model.parameters())


def test_freezing_rejects_invalid_layer_count():
    with pytest.raises(ValueError):
        freeze_layers(Target(), 4)


def test_evaluation_disables_dropout_weights_loss_by_samples_and_restores_mode():
    images = torch.tensor([[4.0, 0.0], [0.0, 4.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = nn.Sequential(nn.Dropout(0.9), nn.Identity())
    criterion = nn.CrossEntropyLoss()
    stats = []
    evaluate_epoch(
        None,
        loader,
        loader,
        loader,
        model,
        criterion,
        0,
        stats,
        include_test=True,
        update_plot=False,
    )
    assert model.training
    assert stats[0][1] == pytest.approx(criterion(images, labels).item())
    assert stats[0][0] == pytest.approx(2 / 3)
    assert stats[0][2] == pytest.approx(1.0)
    assert stats[0][:3] == stats[0][3:6] == stats[0][6:]


def test_prediction_is_deterministic_and_does_not_track_gradients():
    images = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    loader = DataLoader(TensorDataset(images, torch.tensor([-1, -1])), batch_size=1)
    model = nn.Sequential(nn.Dropout(0.9), nn.Linear(2, 2))
    first = predict_challenge(loader, model)
    assert torch.equal(first, predict_challenge(loader, model))
    assert not first.requires_grad
    assert ((first >= 0) & (first <= 1)).all()


def test_patchify_preserves_pixel_order_dtype_and_gradients():
    images = torch.arange(32, dtype=torch.float64).reshape(1, 2, 4, 4).requires_grad_()
    patches = patchify(images, 2)
    expected = torch.stack(
        [images[0, :, i : i + 2, j : j + 2].flatten() for i in (0, 2) for j in (0, 2)]
    ).unsqueeze(0)
    assert torch.equal(patches, expected)
    assert patches.dtype == images.dtype
    patches.sum().backward()
    assert torch.equal(images.grad, torch.ones_like(images))
    with pytest.raises(ValueError):
        patchify(images, 3)


def test_vit_backward_reaches_patch_embedding_and_attention():
    model = ViT(
        num_patches=2, num_blocks=1, num_hidden=8, num_heads=2, chw_shape=(3, 8, 8)
    )
    logits = model(torch.randn(2, 3, 8, 8))
    assert logits.shape == (2, 2)
    nn.CrossEntropyLoss()(logits, torch.tensor([0, 1])).backward()
    assert model.patch_to_token.weight.grad.abs().sum() > 0
    assert (
        model.transformer_blocks[0]
        .multi_head_attention.Q_mappers[0]
        .weight.grad.abs()
        .sum()
        > 0
    )


def test_early_stopping_uses_best_loss_instead_of_previous_epoch():
    count, best = early_stopping([[0, 0.4]], 2, 0.5)
    assert (count, best) == (0, 0.4)
    count, best = early_stopping([[0, 0.6]], count, best)
    count, best = early_stopping([[0, 0.5]], count, best)
    assert (count, best) == (2, 0.4)


def test_sweep_returns_validation_auroc_at_correct_checkpoint_epoch(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "source"
    save_checkpoint(Source(), 10, str(source_dir), [[0.5, 0.4, 0.7]])
    images = torch.randn(2, 3, 64, 64)
    loader = DataLoader(TensorDataset(images, torch.tensor([0, 1])), batch_size=2)
    monkeypatch.setattr(
        train_challenge,
        "get_train_val_test_loaders",
        lambda **kw: (loader, loader, loader, None),
    )
    monkeypatch.setattr(
        train_challenge,
        "config",
        lambda key: str(source_dir) if key == "source.checkpoint" else 2,
    )
    monkeypatch.setattr(train_challenge, "make_training_plot", lambda: None)
    monkeypatch.setattr(train_challenge.plt, "savefig", lambda *a, **kw: None)
    rows = [
        [0.5, 0.9, 0.6, 0.5, 0.8, 0.55],
        [0.8, 0.4, 0.85, 0.9, 0.2, 0.99],
        [0.8, 0.6, 0.95, 0.9, 0.2, 0.99],
    ]

    def evaluate(*args, **kwargs):
        assert not kwargs.get("include_test", False)
        stats = args[7]
        stats.append(rows[len(stats)])

    monkeypatch.setattr(train_challenge, "evaluate_epoch", evaluate)
    auroc, epoch = train_challenge.run_experiment("synthetic", 3, 0.3, 1e-4, 1e-3, 1)
    assert auroc == 0.85
    assert epoch == 1
    selected = torch.load(
        Path("checkpoints/challenge_synthetic/best.checkpoint.pth.tar"),
        weights_only=True,
    )
    assert selected["epoch"] == 1
