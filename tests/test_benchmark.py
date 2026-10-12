"""Check isolated training and held-out evaluation in the reproduction runner."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import run_benchmarks


def test_reproduction_trains_selects_checkpoint_then_evaluates_test(
    tmp_path, monkeypatch
):
    model = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.zeros_(model.weight)
    loader = DataLoader(TensorDataset(torch.eye(2), torch.tensor([0, 1])), batch_size=2)
    events = []
    original_assess = run_benchmarks.assess

    def assess(model, data, criterion, epoch, stats, multiclass=False, test=False):
        events.append((epoch, test))
        return original_assess(model, data, criterion, epoch, stats, multiclass, test)

    monkeypatch.setattr(run_benchmarks, "assess", assess)
    trained, result = run_benchmarks.train_run(
        "synthetic",
        model,
        (loader, loader, loader),
        tmp_path,
        max_epochs=1,
        patience=5,
        learning_rate=0.01,
    )
    assert events == [(0, False), (1, False), (1, True)]
    assert result["epoch"] == 1 and result["epochs_trained"] == 1
    assert result["test_auroc"] == 1
    assert not result["stopped_early"]
    assert trained.weight[0, 0] > trained.weight[0, 1]
    checkpoint = torch.load(
        tmp_path / "synthetic/best.checkpoint.pth.tar", weights_only=True
    )
    assert checkpoint["epoch"] == 1


def test_reproduction_does_not_overwrite_existing_experiment(tmp_path):
    marker = tmp_path / "existing-checkpoint.txt"
    marker.write_text("keep this")
    with pytest.raises(FileExistsError, match="empty experiment directory"):
        run_benchmarks.run(tmp_path)
    assert marker.read_text() == "keep this"


def test_reproduction_requires_a_positive_epoch_limit(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run_benchmarks.run(tmp_path, max_epochs=0)
