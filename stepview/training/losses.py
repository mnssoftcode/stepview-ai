"""StepView Loss Functions for Semantic Segmentation.

Supports class-weighted Cross-Entropy, multi-class Soft Dice Loss,
and combined loss formulation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from stepview.data.schema import NUM_CLASSES


def compute_median_frequency_weights(
    class_pixel_counts: Dict[int, int],
    num_classes: int = NUM_CLASSES,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute median frequency class weights from class pixel distribution.

    weight_c = median(frequencies) / (frequency_c + eps)
    """
    total_pixels = sum(class_pixel_counts.values())
    if total_pixels == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    freqs = []
    for cid in range(num_classes):
        cnt = class_pixel_counts.get(cid, 0)
        freqs.append(cnt / total_pixels)

    freqs_arr = torch.tensor(freqs, dtype=torch.float32)
    # Only consider non-zero frequencies for median calculation
    non_zero = freqs_arr[freqs_arr > 0]
    median_val = float(torch.median(non_zero)) if len(non_zero) > 0 else 1.0

    weights = median_val / (freqs_arr + eps)
    # Zero out weight for classes with 0 count to prevent gradient explosions
    weights[freqs_arr == 0] = 0.0
    return weights


class DiceLoss(nn.Module):
    """Multi-class Soft Dice Loss."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -100) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-class Dice loss.

        Args:
            logits: (B, C, H, W)
            targets: (B, H, W) integer class labels
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets.clamp(min=0, max=num_classes - 1), num_classes=num_classes)
        # Permute to (B, C, H, W)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Compute Dice per class
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_score)
        return dice_loss


class CombinedSegmentationLoss(nn.Module):
    """Combined Cross-Entropy and Soft Dice loss with optional class weighting."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
        ce_weight: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss.

        Returns:
            Tuple of (total_loss_tensor, loss_dict_with_floats).
        """
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        total_loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_dice": float(loss_dice.item()),
        }
