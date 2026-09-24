"""StepView Trainer Implementation.

Coordinates forward/backward optimization, evaluation, checkpoint serialization,
and metric tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stepview.training.metrics import SegmentationMetrics


class StepViewTrainer:
    """Trainer orchestrator for StepView semantic segmentation models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        num_classes: int = 6,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes

        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = scheduler
        self.metrics_tracker = SegmentationMetrics(num_classes=num_classes)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)

            # Resize logits if spatial dimensions differ from mask
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            loss, _ = self.criterion(logits, masks) if hasattr(self.criterion, "ce_weight") else (self.criterion(logits, masks), {})
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss, "epoch": float(epoch)}

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate model on given or default validation dataloader."""
        target_loader = dataloader or self.val_loader
        if target_loader is None:
            return {}

        self.model.eval()
        self.metrics_tracker.reset()
        total_val_loss = 0.0
        num_batches = 0

        for batch in target_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            logits = self.model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            loss, _ = self.criterion(logits, masks) if hasattr(self.criterion, "ce_weight") else (self.criterion(logits, masks), {})
            total_val_loss += float(loss.item())
            num_batches += 1

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            self.metrics_tracker.update(preds, targets)

        results = self.metrics_tracker.compute()
        results["val_loss"] = total_val_loss / max(num_batches, 1)
        return results

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the accumulated confusion matrix from the last evaluation."""
        return self.metrics_tracker.get_confusion_matrix()

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serialize model weights, optimizer state, and training metrics."""
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "num_classes": self.num_classes,
            "config": config or {},
        }
        torch.save(checkpoint, str(path))

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load model state from checkpoint."""
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint
