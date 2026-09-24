"""StepView Segmentation Evaluation Metrics.

Computes confusion matrices, per-class IoU, active mIoU, and class-specific metrics
with robust handling for absent classes (e.g. missing HOLE class in RUGD).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from stepview.data.schema import CLASS_NAMES, NUM_CLASSES, TerrainClass


class SegmentationMetrics:
    """Accumulates confusion matrix across batches and computes IoU and accuracy metrics."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Reset the confusion matrix accumulator."""
        self.confusion_matrix.fill(0)

    def update(self, preds: np.ndarray, targets: np.ndarray) -> None:
        """Update confusion matrix with batch predictions and ground truth.

        Args:
            preds: Array of predicted class IDs (H, W) or (B, H, W).
            targets: Array of true class IDs with same shape as preds.
        """
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)

        # Filter out any ignore indices (-1 or >= num_classes)
        valid_mask = (targets_flat >= 0) & (targets_flat < self.num_classes)
        preds_valid = preds_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]

        # Compute 2D histogram (row = target, col = predicted)
        indices = self.num_classes * targets_valid + preds_valid
        bincount = np.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += bincount.reshape((self.num_classes, self.num_classes))

    def compute(self) -> Dict[str, float]:
        """Compute IoU metrics from accumulated confusion matrix.

        Returns:
            Dictionary containing:
                - 'mean_iou': Average IoU over active (present) classes
                - 'all_class_mean_iou': Average IoU over all num_classes
                - 'iou_{class_name}': Per-class IoU
                - 'pixel_accuracy': Overall global pixel accuracy
        """
        cm = self.confusion_matrix
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp

        denominator = tp + fp + fn
        iou_per_class = np.full(self.num_classes, np.nan, dtype=np.float64)

        # Compute IoU for classes where denominator > 0
        valid_classes = denominator > 0
        iou_per_class[valid_classes] = tp[valid_classes] / denominator[valid_classes]

        results: Dict[str, float] = {}

        # Per-class IoU
        for cid in range(self.num_classes):
            cname = CLASS_NAMES.get(cid, f"class_{cid}")
            iou_val = iou_per_class[cid]
            results[f"iou_{cname}"] = float(iou_val) if not np.isnan(iou_val) else 0.0

        # Mean IoU over active classes (present in GT or predictions)
        active_ious = [val for val in iou_per_class if not np.isnan(val)]
        results["mean_iou"] = float(np.mean(active_ious)) if active_ious else 0.0

        # All-class mean IoU (treating absent classes as 0.0)
        results["all_class_mean_iou"] = float(np.nanmean(np.nan_to_num(iou_per_class, nan=0.0)))

        # Global Pixel Accuracy
        total_pixels = cm.sum()
        results["pixel_accuracy"] = float(tp.sum() / total_pixels) if total_pixels > 0 else 0.0

        return results

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the accumulated confusion matrix."""
        return self.confusion_matrix.copy()
