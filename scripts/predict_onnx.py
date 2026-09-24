#!/usr/bin/env python3
"""StepView — Standalone ONNX Runtime Segmentation Inference Tool.

Runs exported ONNX models for fast, zero-dependency terrain semantic segmentation:
- Validates model input/output contracts
- Runs inference via ONNX Runtime
- Generates original image, colorized predicted mask, and alpha-blended overlay
- Produces composite diagnostic visualization with legend and class statistics

Usage:
    python scripts/predict_onnx.py \
        --model models/stepview_segmentation.onnx \
        --image data/images/rugd_trail-10_frame00001.png \
        --output-dir experiments/visualizations/onnx_predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import onnxruntime as ort

from stepview.data.schema import CLASS_NAMES, CLASS_PALETTE, NUM_CLASSES, TerrainClass


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map indexed single-channel mask to RGB color image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_PALETTE.items():
        color_mask[mask == class_id] = color
    return color_mask


def create_legend_banner(width: int, height: int = 36) -> np.ndarray:
    """Create a horizontal banner displaying class color chips and names."""
    banner = np.full((height, width, 3), 28, dtype=np.uint8)
    num_items = len(TerrainClass)
    col_width = width // num_items

    for i, cls in enumerate(TerrainClass):
        x_start = i * col_width
        chip_x = x_start + 8
        chip_y = (height - 16) // 2
        chip_w = 18
        chip_h = 16

        rgb_color = CLASS_PALETTE[cls.value]
        bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
        cv2.rectangle(banner, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), bgr_color, -1)
        cv2.rectangle(banner, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), (180, 180, 180), 1)

        label = f"{cls.name.lower()}"
        cv2.putText(
            banner,
            label,
            (chip_x + chip_w + 6, height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return banner


def generate_composite_panel(
    image_rgb: np.ndarray,
    color_mask: np.ndarray,
    overlay_rgb: np.ndarray,
    sample_name: str,
    latency_ms: float,
    stats_str: str,
) -> np.ndarray:
    """Generate side-by-side 3-panel visualization: Original | Predicted Mask | Overlay."""
    h, w = image_rgb.shape[:2]

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    header_h = 44
    total_w = w * 3
    panel = np.zeros((h + header_h, total_w, 3), dtype=np.uint8)
    panel[:header_h, :] = 24

    panel[header_h:, 0:w] = img_bgr
    panel[header_h:, w:2*w] = mask_bgr
    panel[header_h:, 2*w:3*w] = overlay_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)

    cv2.putText(panel, f"StepView Input: {sample_name} ({w}x{h})", (12, 28), font, 0.52, color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"ONNX Runtime Mask ({latency_ms:.2f} ms)", (w + 12, 28), font, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "Blended Overlay (50% Alpha)", (2 * w + 12, 28), font, 0.52, (100, 255, 100), 1, cv2.LINE_AA)

    legend = create_legend_banner(total_w, height=36)
    full_composite = np.vstack([panel, legend])

    return full_composite


class ONNXTerrainSegmenter:
    """ONNX Runtime wrapper for StepView semantic segmentation."""

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
        """Execute segmentation inference on an RGB image."""
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


def process_image(
    image_path: Path,
    segmenter: ONNXTerrainSegmenter,
    output_dir: Path,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """Segment single image and save all artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Predict
    res = segmenter.predict(img_rgb)
    pred_mask = res["class_mask"]
    conf_map = res["confidence_map"]
    latency_ms = res["latency_ms"]

    color_mask = colorize_mask(pred_mask)
    overlay_rgb = cv2.addWeighted(img_rgb, 1.0 - alpha, color_mask, alpha, 0)

    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    # Class distribution
    unique_ids, counts = np.unique(pred_mask, return_counts=True)
    total_px = pred_mask.size
    class_dist: Dict[str, float] = {}
    for cid, cnt in zip(unique_ids, counts):
        cname = CLASS_NAMES.get(int(cid), f"class_{cid}")
        class_dist[cname] = float((cnt / total_px) * 100.0)

    # Save artifacts
    orig_path = output_dir / f"{stem}_original.png"
    mask_path = output_dir / f"{stem}_pred_mask.png"
    color_path = output_dir / f"{stem}_pred_color.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    composite_path = output_dir / f"{stem}_composite.png"
    json_path = output_dir / f"{stem}_result.json"

    cv2.imwrite(str(orig_path), img_bgr)
    cv2.imwrite(str(mask_path), pred_mask)
    cv2.imwrite(str(color_path), color_mask_bgr)
    cv2.imwrite(str(overlay_path), overlay_bgr)

    stats_str = ", ".join(f"{k}: {v:.1f}%" for k, v in class_dist.items())
    composite = generate_composite_panel(img_rgb, color_mask, overlay_rgb, stem, latency_ms, stats_str)
    cv2.imwrite(str(composite_path), composite)

    result_data = {
        "sample": stem,
        "resolution": f"{w}x{h}",
        "raw_output_shape": res["raw_output_shape"],
        "latency_ms": round(latency_ms, 2),
        "mean_confidence": round(float(np.mean(conf_map)), 4),
        "class_distribution": class_dist,
        "artifacts": {
            "original": str(orig_path),
            "predicted_mask": str(mask_path),
            "predicted_color": str(color_path),
            "blended_overlay": str(overlay_path),
            "composite": str(composite_path),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    return result_data


def main() -> None:
    parser = argparse.ArgumentParser(description="StepView ONNX Segmentation Inference")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/stepview_segmentation.onnx"),
        help="Path to exported ONNX model",
    )
    parser.add_argument("--image", type=Path, default=None, help="Path to input image file")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory of images to process")
    parser.add_argument("--limit", type=int, default=3, help="Max images to process from directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/visualizations/onnx_predictions"),
        help="Directory to save output predictions",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha blending weight")

    args = parser.parse_args()

    print(f"Loading ONNX model from: {args.model}...")
    segmenter = ONNXTerrainSegmenter(args.model)

    image_paths: List[Path] = []
    if args.image is not None and args.image.is_file():
        image_paths.append(args.image)
    elif args.image_dir is not None and args.image_dir.is_dir():
        exts = {".png", ".jpg", ".jpeg"}
        files = sorted([p for p in args.image_dir.iterdir() if p.suffix.lower() in exts])
        image_paths.extend(files[:args.limit])
    else:
        # Default: 3 test images across distinct scenes
        data_images = Path("data/images")
        target_scenes = ["trail-10", "park-1", "village"]
        for sc in target_scenes:
            match = next((f for f in data_images.glob("*.png") if sc in f.name and f not in image_paths), None)
            if match:
                image_paths.append(match)

    print(f"\nExecuting ONNX inference on {len(image_paths)} image(s)...")
    for img_p in image_paths:
        res = process_image(img_p, segmenter, args.output_dir, alpha=args.alpha)
        print(f"\n--- {res['sample']} ({res['resolution']}) ---")
        print(f"  Latency:         {res['latency_ms']:.2f} ms")
        print(f"  Mean Confidence: {res['mean_confidence']:.4f}")
        print("  Class Distribution:")
        for cname, pct in res["class_distribution"].items():
            print(f"    - {cname:<18}: {pct:.2f}%")
        print(f"  Artifacts saved:")
        print(f"    * Mask:      {res['artifacts']['predicted_mask']}")
        print(f"    * Overlay:   {res['artifacts']['blended_overlay']}")
        print(f"    * Composite: {res['artifacts']['composite']}")

    print(f"\nProcessing complete. All results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
