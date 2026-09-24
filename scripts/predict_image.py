#!/usr/bin/env python3
"""StepView — Terrain Segmentation Inference Tool.

Runs trained StepView segmentation models on input images to generate:
- Original image copy
- Colorized predicted segmentation mask
- Blended alpha overlay (image + mask)
- Composite 3-panel visualization with legend and class statistics

Usage:
    python scripts/predict_image.py --image data/images/rugd_trail_frame00001.png
    python scripts/predict_image.py --checkpoint experiments/runs/smoke_test/best_checkpoint.pt --image-dir data/images --limit 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch

from stepview.data.schema import CLASS_NAMES, CLASS_PALETTE, NUM_CLASSES, TerrainClass
from stepview.inference.segmenter import TerrainSegmenter


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map indexed single-channel mask to RGB color image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_PALETTE.items():
        color_mask[mask == class_id] = color
    return color_mask


def create_legend_banner(width: int, height: int = 36) -> np.ndarray:
    """Create a horizontal banner displaying class color chips and names."""
    banner = np.full((height, width, 3), 30, dtype=np.uint8)
    num_items = len(TerrainClass)
    col_width = width // num_items

    for i, cls in enumerate(TerrainClass):
        x_start = i * col_width
        chip_x = x_start + 10
        chip_y = (height - 18) // 2
        chip_w = 20
        chip_h = 18

        rgb_color = CLASS_PALETTE[cls.value]
        bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
        cv2.rectangle(banner, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), bgr_color, -1)
        cv2.rectangle(banner, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), (200, 200, 200), 1)

        label = f"{cls.name.lower()}"
        cv2.putText(
            banner,
            label,
            (chip_x + chip_w + 6, height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return banner


def generate_composite_panel(
    image_rgb: np.ndarray,
    color_mask: np.ndarray,
    overlay_rgb: np.ndarray,
    sample_name: str,
    stats_text: str,
) -> np.ndarray:
    """Generate side-by-side 3-panel visualization: Original | Predicted Mask | Overlay."""
    h, w = image_rgb.shape[:2]

    # Convert RGB arrays to BGR for OpenCV drawing
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    header_h = 44
    total_w = w * 3
    panel = np.zeros((h + header_h, total_w, 3), dtype=np.uint8)
    panel[:header_h, :] = 25

    panel[header_h:, 0:w] = img_bgr
    panel[header_h:, w:2*w] = mask_bgr
    panel[header_h:, 2*w:3*w] = overlay_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    color = (255, 255, 255)

    cv2.putText(panel, f"StepView Prediction: {sample_name} ({w}x{h})", (12, 28), font, font_scale, color, 1, cv2.LINE_AA)
    cv2.putText(panel, "Predicted Semantic Mask", (w + 12, 28), font, font_scale, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "Blended Overlay (50% Alpha)", (2 * w + 12, 28), font, font_scale, (100, 255, 100), 1, cv2.LINE_AA)

    # Attach legend at the bottom
    legend = create_legend_banner(total_w, height=36)
    full_composite = np.vstack([panel, legend])

    return full_composite


def predict_single_image(
    image_path: Path,
    segmenter: TerrainSegmenter,
    output_dir: Path,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """Run inference on one image and save all artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Run inference
    result = segmenter.segment(img_rgb)
    pred_mask = result["class_mask"]
    conf_map = result["confidence_map"]
    mean_conf = float(np.mean(conf_map))

    # Colorize mask
    color_mask = colorize_mask(pred_mask)

    # Blended overlay
    overlay_rgb = cv2.addWeighted(img_rgb, 1.0 - alpha, color_mask, alpha, 0)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

    # Compute class distribution
    unique_ids, counts = np.unique(pred_mask, return_counts=True)
    total_pixels = pred_mask.size
    class_dist: Dict[str, float] = {}
    for cid, cnt in zip(unique_ids, counts):
        cname = CLASS_NAMES.get(int(cid), f"class_{cid}")
        class_dist[cname] = float((cnt / total_pixels) * 100.0)

    # Save output artifacts
    orig_save = output_dir / f"{stem}_original.png"
    mask_save = output_dir / f"{stem}_pred_mask.png"
    color_save = output_dir / f"{stem}_pred_color.png"
    overlay_save = output_dir / f"{stem}_overlay.png"
    composite_save = output_dir / f"{stem}_composite.png"

    cv2.imwrite(str(orig_save), img_bgr)
    cv2.imwrite(str(mask_save), pred_mask)
    cv2.imwrite(str(color_save), color_mask_bgr)
    cv2.imwrite(str(overlay_save), overlay_bgr)

    stats_str = ", ".join(f"{k}: {v:.1f}%" for k, v in class_dist.items())
    composite = generate_composite_panel(img_rgb, color_mask, overlay_rgb, stem, stats_str)
    cv2.imwrite(str(composite_save), composite)

    return {
        "sample": stem,
        "resolution": f"{w}x{h}",
        "mean_confidence": mean_conf,
        "class_distribution": class_dist,
        "artifacts": {
            "original": str(orig_save),
            "predicted_mask": str(mask_save),
            "predicted_color": str(color_save),
            "blended_overlay": str(overlay_save),
            "composite": str(composite_save),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="StepView Terrain Segmentation Inference")
    parser.add_argument("--image", type=Path, default=None, help="Path to single image file")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory containing images to segment")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of images to segment from directory")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/runs/smoke_test/best_checkpoint.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/visualizations/predictions"),
        help="Directory to save output predictions",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha blending weight for overlay")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'mps')")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        dev = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint} on {dev}...")
    segmenter = TerrainSegmenter.load_from_checkpoint(args.checkpoint, device=dev)

    # Collect images
    image_paths: List[Path] = []
    if args.image is not None and args.image.is_file():
        image_paths.append(args.image)
    elif args.image_dir is not None and args.image_dir.is_dir():
        exts = {".png", ".jpg", ".jpeg"}
        files = sorted([p for p in args.image_dir.iterdir() if p.suffix.lower() in exts])
        image_paths.extend(files[:args.limit])
    else:
        # Default: pick 3 sample images from data/images/
        data_images = Path("data/images")
        if data_images.is_dir():
            files = sorted(list(data_images.glob("*.png")))
            # Pick one trail, one park, one creek if available
            picked: List[Path] = []
            for kw in ["trail", "park", "creek"]:
                match = next((f for f in files if kw in f.name and f not in picked), None)
                if match:
                    picked.append(match)
            if not picked:
                picked = files[:3]
            image_paths.extend(picked)

    if not image_paths:
        print("No images found to process. Please specify --image or --image-dir.")
        return

    print(f"\nRunning inference on {len(image_paths)} image(s)...")
    for img_p in image_paths:
        res = predict_single_image(img_p, segmenter, args.output_dir, alpha=args.alpha)
        print(f"\n--- {res['sample']} ({res['resolution']}) ---")
        print(f"  Mean Confidence: {res['mean_confidence']:.4f}")
        print("  Class Distribution:")
        for cname, pct in res["class_distribution"].items():
            print(f"    - {cname:<18}: {pct:.2f}%")
        print(f"  Artifacts saved:")
        print(f"    * Original:    {res['artifacts']['original']}")
        print(f"    * Mask:        {res['artifacts']['predicted_mask']}")
        print(f"    * Color Mask:  {res['artifacts']['predicted_color']}")
        print(f"    * Overlay:     {res['artifacts']['blended_overlay']}")
        print(f"    * Composite:   {res['artifacts']['composite']}")

    print(f"\nInference complete. Visualizations available in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
