"""StepView ONNX Runtime Inference Segmenter.

High-performance, standalone ONNX Runtime wrapper for StepView semantic terrain segmentation.
Requires no PyTorch dependency for execution.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import onnxruntime as ort

from stepview.data.schema import NUM_CLASSES


class ONNXTerrainSegmenter:
    """ONNX Runtime wrapper for StepView semantic terrain segmentation."""

    def __init__(self, model_path: Path, providers: Optional[List[str]] = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        chosen_providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=chosen_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def predict(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Execute segmentation inference on an RGB image.

        Args:
            image_rgb: RGB image array (H, W, 3) uint8.

        Returns:
            Dict containing:
                - 'class_mask': 2D uint8 numpy array with class indices 0..5
                - 'confidence_map': 2D float32 numpy array of prediction probabilities [0, 1]
                - 'probabilities': 3D float32 numpy array (NUM_CLASSES, H, W)
                - 'latency_ms': Float execution time for ONNX forward pass
                - 'raw_output_shape': List of dimensions for raw model output
        """
        orig_h, orig_w = image_rgb.shape[:2]

        # 1. Preprocess: resize to fixed 256x256
        target_h, target_w = 256, 256
        resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Transpose HWC -> CHW and normalize to [0, 1] float32
        tensor_np = (resized.transpose((2, 0, 1)).astype(np.float32) / 255.0)[np.newaxis, ...]

        # 2. Run ONNX inference with latency tracking
        t0 = time.perf_counter()
        raw_out = self.session.run([self.output_name], {self.input_name: tensor_np})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # raw_out is logits shape (1, 6, 256, 256)
        logits = raw_out.squeeze(0)  # (6, 256, 256)

        # 3. Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs_256 = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)  # (6, 256, 256)

        # 4. Upsample continuous probability map to original resolution
        probs_full = np.zeros((NUM_CLASSES, orig_h, orig_w), dtype=np.float32)
        for c in range(NUM_CLASSES):
            probs_full[c] = cv2.resize(probs_256[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Renormalize channels
        prob_sum = np.sum(probs_full, axis=0, keepdims=True) + 1e-7
        probs_full = probs_full / prob_sum

        # 5. Extract class predictions and confidence
        class_mask = np.argmax(probs_full, axis=0).astype(np.uint8)
        confidence_map = np.max(probs_full, axis=0).astype(np.float32)

        return {
            "class_mask": class_mask,
            "confidence_map": confidence_map,
            "probabilities": probs_full,
            "latency_ms": latency_ms,
            "raw_output_shape": list(raw_out.shape),
        }
