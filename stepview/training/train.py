#!/usr/bin/env python3
"""StepView — Model Training Script.

Trains StepView semantic segmentation models on configured dataset splits.

Usage:
    python stepview/training/train.py --data-dir data --epochs 3 --batch-size 4 --output-dir experiments/runs/smoke_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader

from stepview.data.dataset import StepViewDataset, stepview_collate_fn
from stepview.data.schema import NUM_CLASSES
from stepview.models.segmentation import build_segmentation_model
from stepview.training.losses import CombinedSegmentationLoss, compute_median_frequency_weights
from stepview.training.trainer import StepViewTrainer


def run_training(
    data_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    output_dir: Path = Path("experiments/runs/smoke_test"),
    weight_mode: str = "none",
    pretrained_backbone: bool = False,
    device_name: str = "auto",
) -> Dict[str, float]:
    """Execute training and validation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device selection
    if device_name == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Executing on device: {device}")

    # 2. Data Loaders
    train_dataset = StepViewDataset(root_dir=data_dir, split="train")
    val_dataset = StepViewDataset(root_dir=data_dir, split="val")

    print(f"Loaded train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=stepview_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=stepview_collate_fn,
        num_workers=0,
    )

    # 3. Model instantiation
    model = build_segmentation_model(
        num_classes=NUM_CLASSES,
        pretrained_backbone=pretrained_backbone,
    )

    # 4. Class weighting configuration
    class_weights: Optional[torch.Tensor] = None
    if weight_mode == "median_freq":
        stats_file = data_dir / "metadata" / "class_stats.json"
        if stats_file.is_file():
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
                pixel_counts = {int(k): v["pixel_count"] for k, v in stats["classes"].items()}
                class_weights = compute_median_frequency_weights(pixel_counts, num_classes=NUM_CLASSES).to(device)
                print(f"Applied median frequency class weights: {class_weights.tolist()}")

    # 5. Loss and Optimizer
    criterion = CombinedSegmentationLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 6. Trainer
    trainer = StepViewTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=NUM_CLASSES,
    )

    # 7. Training Loop
    history = []
    best_miou = -1.0
    latest_metrics = {}

    print(f"\nStarting training for {epochs} epoch(s)...")
    for epoch in range(1, epochs + 1):
        train_stats = trainer.train_epoch(epoch)
        val_stats = trainer.evaluate()

        combined = {**train_stats, **val_stats}
        history.append(combined)
        latest_metrics = combined

        miou = val_stats.get("mean_iou", 0.0)
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_stats['train_loss']:.4f} | "
            f"Val Loss: {val_stats.get('val_loss', 0.0):.4f} | "
            f"mIoU: {miou:.4f} | "
            f"Ground IoU: {val_stats.get('iou_ground', 0.0):.4f} | "
            f"Pixel Acc: {val_stats.get('pixel_accuracy', 0.0):.4f}"
        )

        # Save latest checkpoint
        trainer.save_checkpoint(
            path=output_dir / "last_checkpoint.pt",
            epoch=epoch,
            metrics=combined,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_mode": weight_mode,
            },
        )

        # Save best checkpoint
        if miou > best_miou:
            best_miou = miou
            trainer.save_checkpoint(
                path=output_dir / "best_checkpoint.pt",
                epoch=epoch,
                metrics=combined,
            )

    # Save training history
    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Checkpoints saved to: {output_dir.resolve()}")
    return latest_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StepView Segmentation Model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to StepView dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/runs/smoke_test"), help="Output run directory")
    parser.add_argument("--weight-mode", type=str, default="none", choices=["none", "median_freq"], help="Class weighting mode")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'mps')")

    args = parser.parse_args()
    run_training(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        weight_mode=args.weight_mode,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
