"""StepView Inference Segmenter Interface.

Provides a clean, standalone inference interface for executing trained
StepView segmentation models on images or video frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stepview.data.schema import NUM_CLASSES, TerrainClass, VALID_CLASS_IDS
from stepview.models.segmentation import StepViewSegmentationModel, build_segmentation_model


class TerrainSegmenter:
    """Inference wrapper for StepView semantic terrain segmentation."""

    def __init__(
        self,
        model: StepViewSegmentationModel,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> TerrainSegmenter:
        """Instantiate segmenter by loading a trained checkpoint."""
        resolved_path = Path(checkpoint_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {resolved_path}")

        dev = device or torch.device("cpu")
        checkpoint = torch.load(str(resolved_path), map_location=dev)
        num_classes = checkpoint.get("num_classes", NUM_CLASSES)

        model = build_segmentation_model(num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model=model, device=dev)

    @torch.no_grad()
    def segment(
        self,
        image_input: Union[np.ndarray, str, Path],
        input_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Perform semantic terrain segmentation on an input image.

        Args:
            image_input: RGB image array (H, W, 3) uint8 or Path to image file.
            input_size: Optional (width, height) tuple to resize input prior to inference.

        Returns:
            Dict containing:
                - 'class_mask': 2D uint8 numpy array with class indices 0..5
                - 'confidence_map': 2D float32 numpy array of prediction probabilities [0, 1]
                - 'probabilities': 3D float32 numpy array (NUM_CLASSES, H, W)
                - 'unique_classes': Set of class IDs detected
        """
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"Could not load image from: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input

        orig_h, orig_w = img_rgb.shape[:2]

        # Optional preprocessing resize
        proc_rgb = img_rgb
        if input_size is not None:
            proc_rgb = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)

        # Convert HWC uint8 -> CHW float32 [0.0, 1.0]
        tensor = torch.from_numpy(proc_rgb.transpose((2, 0, 1))).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(tensor)

        # If spatial size differs from original input, interpolate back to original resolution
        if logits.shape[-2:] != (orig_h, orig_w):
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        probs = F.softmax(logits, dim=1).squeeze(0)  # (C, H, W)
        conf_map, class_mask = torch.max(probs, dim=0)  # (H, W), (H, W)

        mask_np = class_mask.cpu().numpy().astype(np.uint8)
        conf_np = conf_map.cpu().numpy().astype(np.float32)
        probs_np = probs.cpu().numpy().astype(np.float32)

        return {
            "class_mask": mask_np,
            "confidence_map": conf_np,
            "probabilities": probs_np,
            "unique_classes": set(np.unique(mask_np).tolist()),
        }
