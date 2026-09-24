"""StepView Semantic Segmentation Model Architectures.

Provides lightweight, edge-compatible segmentation models based on
MobileNetV3-Small backbone with Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP) head.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.segmentation.lraspp import LRASPP

from stepview.data.schema import NUM_CLASSES


class StepViewSegmentationModel(nn.Module):
    """StepView Terrain Segmentation Model based on MobileNetV3-Small + LR-ASPP.

    Designed for low-latency edge execution (~1.08M parameters).
    Produces 6-class segmentation logits matching StepView canonical taxonomy.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = False,
        inter_channels: int = 128,
    ) -> None:
        """Initialize the model.

        Args:
            num_classes: Number of semantic output classes (default 6).
            pretrained_backbone: If True, attempts to load ImageNet-pretrained weights.
            inter_channels: Intermediate channels in LR-ASPP head (default 128).
        """
        super().__init__()
        self.num_classes = num_classes

        # Load MobileNetV3-Small backbone
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
        try:
            backbone_full = models.mobilenet_v3_small(weights=weights)
        except Exception:
            # Fallback if offline / weights unavailable
            backbone_full = models.mobilenet_v3_small(weights=None)

        backbone_features = backbone_full.features

        # Extract low-level (layer 1: 16 channels) and high-level (layer 12: 576 channels) features
        return_nodes = {
            "1": "low",
            "12": "high",
        }
        low_channels = 16
        high_channels = 576

        feature_extractor = create_feature_extractor(backbone_features, return_nodes=return_nodes)

        # LR-ASPP Head
        self.lraspp = LRASPP(
            backbone=feature_extractor,
            low_channels=low_channels,
            high_channels=high_channels,
            num_classes=num_classes,
            inter_channels=inter_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W) normalized to [0, 1].

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        out_dict = self.lraspp(x)
        logits = out_dict["out"]
        return logits

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform inference returning softmax probabilities and class predictions.

        Args:
            x: Input tensor of shape (B, 3, H, W) or (3, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Softmax probabilities: (B, num_classes, H, W)
                - Predicted class index: (B, H, W) long
        """
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)

        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return probs, preds


def build_segmentation_model(
    num_classes: int = NUM_CLASSES,
    pretrained_backbone: bool = False,
) -> StepViewSegmentationModel:
    """Factory function for instantiating the baseline segmentation model."""
    return StepViewSegmentationModel(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
    )
