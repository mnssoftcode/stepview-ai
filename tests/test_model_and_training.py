"""Unit tests for StepView segmentation model, loss functions, metrics, and inference.

Validates software engineering correctness of model forward/backward passes,
loss calculations, metrics on missing classes, and checkpoint serialization.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from stepview.data.schema import NUM_CLASSES, VALID_CLASS_IDS, TerrainClass
from stepview.inference.segmenter import TerrainSegmenter
from stepview.models.segmentation import StepViewSegmentationModel, build_segmentation_model
from stepview.training.losses import (
    CombinedSegmentationLoss,
    DiceLoss,
    compute_median_frequency_weights,
)
from stepview.training.metrics import SegmentationMetrics
from stepview.training.trainer import StepViewTrainer


def test_model_instantiation_and_forward() -> None:
    """Verify that MobileNetV3-Small LR-ASPP model instantiates and executes forward pass."""
    model = build_segmentation_model(num_classes=6, pretrained_backbone=False)
    total_params = sum(p.numel() for p in model.parameters())

    # Param count should be around 1.08M
    assert 1_000_000 < total_params < 1_200_000

    # Test forward pass with batch of 2
    dummy_input = torch.randn(2, 3, 128, 128)
    logits = model(dummy_input)

    assert logits.shape == (2, 6, 128, 128)
    assert logits.dtype == torch.float32


def test_model_predict_contract() -> None:
    """Verify that model.predict returns softmax probabilities and class predictions."""
    model = build_segmentation_model(num_classes=6, pretrained_backbone=False)
    dummy_input = torch.randn(1, 3, 64, 64)

    probs, preds = model.predict(dummy_input)

    assert probs.shape == (1, 6, 64, 64)
    assert preds.shape == (1, 64, 64)
    assert preds.dtype == torch.int64
    assert 0 <= preds.min() and preds.max() < 6


def test_losses_computation() -> None:
    """Verify that CombinedSegmentationLoss computes valid loss with non-zero gradients."""
    criterion = CombinedSegmentationLoss()
    logits = torch.randn(2, 6, 32, 32, requires_grad=True)
    targets = torch.randint(0, 6, (2, 32, 32))

    total_loss, loss_dict = criterion(logits, targets)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() > 0.0
    assert "loss_ce" in loss_dict
    assert "loss_dice" in loss_dict

    total_loss.backward()
    assert logits.grad is not None


def test_metrics_robustness_with_missing_class() -> None:
    """Verify that SegmentationMetrics handles absent classes (e.g. HOLE = 0% pixels)."""
    metrics = SegmentationMetrics(num_classes=6)

    # Simulated targets with only classes 0, 1, 3 (classes 2, 4, 5 are absent)
    targets = np.array([[0, 1], [3, 1]], dtype=np.int64)
    preds = np.array([[0, 1], [3, 0]], dtype=np.int64)

    metrics.update(preds, targets)
    results = metrics.compute()

    assert "mean_iou" in results
    assert "all_class_mean_iou" in results
    assert "iou_ground" in results
    assert "iou_hole" in results

    # Absent class 4 (hole) IoU should be 0.0 without causing ZeroDivisionError
    assert results["iou_hole"] == 0.0
    # Active mean IoU should only average present classes (0, 1, 3)
    assert results["mean_iou"] > results["all_class_mean_iou"]


def test_median_frequency_weights() -> None:
    """Verify median frequency weight calculation."""
    counts = {0: 1000, 1: 500, 2: 100, 3: 5000, 4: 0, 5: 200}
    weights = compute_median_frequency_weights(counts, num_classes=6)

    assert weights.shape == (6,)
    # Absent class 4 must have 0 weight
    assert weights[4].item() == 0.0
    # Infrequent class 2 should have higher weight than frequent class 3
    assert weights[2].item() > weights[3].item()


def test_trainer_checkpoint_cycle(tmp_path: Path) -> None:
    """Verify that trainer trains one step, saves checkpoint, and reloads state."""
    model = build_segmentation_model(num_classes=6, pretrained_backbone=False)

    # Create dummy DataLoader
    images = torch.randn(4, 3, 32, 32)
    masks = torch.randint(0, 6, (4, 32, 32))
    dataset = [
        {"image": images[i], "mask": masks[i], "sample_id": f"s_{i}", "scene_id": "sc_0", "metadata": {}}
        for i in range(4)
    ]

    from stepview.data.dataset import stepview_collate_fn
    loader = DataLoader(dataset, batch_size=2, collate_fn=stepview_collate_fn)

    trainer = StepViewTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        num_classes=6,
    )

    train_stats = trainer.train_epoch(epoch=1)
    assert "train_loss" in train_stats
    assert train_stats["train_loss"] > 0.0

    eval_stats = trainer.evaluate()
    assert "mean_iou" in eval_stats

    # Save checkpoint
    ckpt_file = tmp_path / "test_ckpt.pt"
    trainer.save_checkpoint(ckpt_file, epoch=1, metrics=eval_stats)
    assert ckpt_file.is_file()

    # Create new model and load checkpoint
    new_model = build_segmentation_model(num_classes=6, pretrained_backbone=False)
    new_trainer = StepViewTrainer(model=new_model, train_loader=loader, num_classes=6)
    loaded = new_trainer.load_checkpoint(ckpt_file)

    assert loaded["epoch"] == 1

    # Verify weights are identical
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_terrain_segmenter_inference(tmp_path: Path) -> None:
    """Verify TerrainSegmenter inference output shapes and types."""
    model = build_segmentation_model(num_classes=6, pretrained_backbone=False)
    segmenter = TerrainSegmenter(model=model)

    dummy_rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    dummy_rgb[10:30, 10:30] = (100, 150, 200)

    result = segmenter.segment(dummy_rgb)

    assert "class_mask" in result
    assert "confidence_map" in result
    assert "probabilities" in result
    assert "unique_classes" in result

    mask = result["class_mask"]
    assert mask.shape == (48, 64)
    assert mask.dtype == np.uint8
    assert result["confidence_map"].shape == (48, 64)
    assert result["probabilities"].shape == (6, 48, 64)
    assert result["unique_classes"].issubset(VALID_CLASS_IDS)
