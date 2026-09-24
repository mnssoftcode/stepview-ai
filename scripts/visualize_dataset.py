#!/usr/bin/env python3
"""StepView Dataset Inspection & Visualization Tool.

Loads images and masks using the StepView dataset contract, validates classes
and resolutions, computes pixel statistics, and generates side-by-side
alpha-blended overlays with class legends.

Usage:
    python scripts/visualize_dataset.py --data-dir data --split all --save-dir experiments/visualizations
    python scripts/visualize_dataset.py --sample-id scene001_frame0001
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from stepview.data.schema import (
    CLASS_NAMES,
    CLASS_PALETTE,
    NUM_CLASSES,
    TerrainClass,
    VALID_CLASS_IDS,
    validate_mask_classes,
)
from stepview.data.dataset import StepViewDataset


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map single-channel indexed mask to 3-channel RGB image.

    Args:
        mask: 2D uint8 numpy array containing class IDs.

    Returns:
        RGB image of shape (H, W, 3) with dtype uint8.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_PALETTE.items():
        color_mask[mask == class_id] = color
    return color_mask


def create_legend_banner(width: int, height: int = 40) -> np.ndarray:
    """Create a horizontal banner showing class colors and names.

    Args:
        width: Desired banner width in pixels.
        height: Banner height in pixels.

    Returns:
        RGB numpy array of shape (height, width, 3).
    """
    banner = np.full((height, width, 3), 30, dtype=np.uint8)
    num_items = len(TerrainClass)
    col_width = max(width // num_items, 1)

    for i, class_item in enumerate(TerrainClass):
        class_id = class_item.value
        name = CLASS_NAMES[class_id]
        color = CLASS_PALETTE[class_id]

        x_start = i * col_width
        x_end = (i + 1) * col_width if i < num_items - 1 else width

        # Color patch
        patch_w = 16
        patch_h = 16
        y_offset = (height - patch_h) // 2
        x_patch = x_start + 8

        if x_patch + patch_w < x_end:
            banner[y_offset : y_offset + patch_h, x_patch : x_patch + patch_w] = color
            cv2.rectangle(
                banner,
                (x_patch, y_offset),
                (x_patch + patch_w, y_offset + patch_h),
                (200, 200, 200),
                1,
            )

        # Text label (convert color to BGR for OpenCV putText)
        text_x = x_patch + patch_w + 6
        text_y = y_offset + 12
        if text_x < x_end:
            cv2.putText(
                banner,
                f"{class_id}:{name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

    return banner


def build_composite_panel(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    sample_id: str = "",
) -> np.ndarray:
    """Create a side-by-side composite: [Original Image | Color Mask | Overlay] with header and legend.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        mask: (H, W) uint8 indexed class mask.
        alpha: Overlay blending weight (0.0 to 1.0).
        sample_id: Sample identifier string.

    Returns:
        Composite RGB image array.
    """
    h, w = image_rgb.shape[:2]
    color_mask = colorize_mask(mask)

    # Compute alpha blend
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, color_mask, alpha, 0)

    # Draw titles onto sub-panels
    def add_title(panel: np.ndarray, text: str) -> np.ndarray:
        p = panel.copy()
        cv2.rectangle(p, (0, 0), (w, 24), (20, 20, 20), -1)
        cv2.putText(
            p,
            text,
            (10, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return p

    p1 = add_title(image_rgb, f"Original: {sample_id}")
    p2 = add_title(color_mask, "Semantic Ground Truth")
    p3 = add_title(overlay, f"Overlay (alpha={alpha:.1f})")

    # Concatenate horizontally
    row = np.hstack([p1, p2, p3])
    total_w = row.shape[1]

    # Create legend banner
    legend = create_legend_banner(width=total_w, height=36)

    # Stack vertically: row + legend
    composite = np.vstack([row, legend])
    return composite


def analyze_sample_classes(mask: np.ndarray) -> Tuple[Dict[int, int], Dict[int, float], List[int]]:
    """Compute pixel counts and percentages for each class in the mask.

    Args:
        mask: 2D integer numpy array.

    Returns:
        Tuple of (counts_by_class, percentage_by_class, invalid_classes_found).
    """
    total_pixels = mask.size
    unique_ids, counts = np.unique(mask, return_counts=True)
    counts_map = dict(zip(unique_ids.tolist(), counts.tolist()))

    invalid_ids = [cid for cid in unique_ids if cid not in VALID_CLASS_IDS]

    pct_map = {}
    for cid in VALID_CLASS_IDS:
        cnt = counts_map.get(cid, 0)
        pct_map[cid] = (cnt / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    return counts_map, pct_map, invalid_ids


def inspect_dataset(
    data_dir: Path,
    split: str = "all",
    sample_id_filter: Optional[str] = None,
    index_filter: Optional[int] = None,
    num_samples: Optional[int] = None,
    save_dir: Optional[Path] = None,
    alpha: float = 0.5,
    show_gui: bool = False,
) -> int:
    """Inspect dataset samples, print class metrics, detect defects, and optionally render overlays."""
    print("=" * 80)
    print(f"StepView Dataset Inspection Tool")
    print(f"Dataset root: {data_dir.resolve()}")
    print(f"Target split: {split}")
    print("=" * 80)

    if not data_dir.is_dir():
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        return 1

    dataset = StepViewDataset(root_dir=data_dir, split=split, validate_classes_on_load=False)
    total_count = len(dataset)
    print(f"Discovered {total_count} sample(s) for split '{split}'.")

    if total_count == 0:
        print("[WARNING] No samples discovered. Verify directory contents under images/ and masks/.")
        return 0

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visualizations will be saved to: {save_dir.resolve()}")

    indices_to_process: List[int] = []

    if sample_id_filter:
        matches = [i for i, s in enumerate(dataset.samples) if s.sample_id == sample_id_filter]
        if not matches:
            print(f"[ERROR] Sample ID '{sample_id_filter}' not found in dataset.")
            return 1
        indices_to_process = matches
    elif index_filter is not None:
        if index_filter < 0 or index_filter >= total_count:
            print(f"[ERROR] Index {index_filter} out of range [0, {total_count - 1}].")
            return 1
        indices_to_process = [index_filter]
    else:
        limit = num_samples if num_samples is not None else total_count
        indices_to_process = list(range(min(limit, total_count)))

    aggregate_counts: Dict[int, int] = {cid: 0 for cid in VALID_CLASS_IDS}
    total_analyzed_pixels = 0
    issues_found: List[str] = []

    print(f"\nProcessing {len(indices_to_process)} sample(s)...\n")

    for idx in indices_to_process:
        meta = dataset.get_metadata(idx)
        sid = meta.sample_id
        img_path = data_dir / meta.image_rel_path
        mask_path = data_dir / meta.mask_rel_path

        print(f"--- Sample [{idx + 1}/{total_count}]: {sid} ---")
        print(f"  Scene ID:        {meta.scene_id}")
        print(f"  Split:           {meta.split}")
        print(f"  Image file:      {img_path.name} (exists: {img_path.is_file()})")
        print(f"  Mask file:       {mask_path.name} (exists: {mask_path.is_file()})")

        # 1. Missing file check
        if not img_path.is_file():
            msg = f"Missing image for sample {sid}: {img_path}"
            print(f"  [ERROR] {msg}")
            issues_found.append(msg)
            continue
        if not mask_path.is_file():
            msg = f"Missing mask for sample {sid}: {mask_path}"
            print(f"  [ERROR] {msg}")
            issues_found.append(msg)
            continue

        # 2. Load arrays
        try:
            image_rgb = dataset.load_image(img_path)
            mask = dataset.load_mask(mask_path)
        except Exception as e:
            msg = f"Failed to load sample {sid}: {e}"
            print(f"  [ERROR] {msg}")
            issues_found.append(msg)
            continue

        h_img, w_img = image_rgb.shape[:2]
        h_mask, w_mask = mask.shape[:2]

        print(f"  Image resolution: {w_img}x{h_img} ({image_rgb.shape[2]} channels)")
        print(f"  Mask resolution:  {w_mask}x{h_mask} (1 channel)")

        # 3. Shape mismatch check
        if (h_img, w_img) != (h_mask, w_mask):
            msg = f"Dimension mismatch for {sid}: image is {w_img}x{h_img}, mask is {w_mask}x{h_mask}"
            print(f"  [ERROR] {msg}")
            issues_found.append(msg)
            continue

        # 4. Class analysis
        counts, pcts, invalid_ids = analyze_sample_classes(mask)
        if invalid_ids:
            msg = f"Invalid class IDs found in {sid}: {invalid_ids} (expected {sorted(list(VALID_CLASS_IDS))})"
            print(f"  [ERROR] {msg}")
            issues_found.append(msg)

        total_pixels = mask.size
        total_analyzed_pixels += total_pixels

        print("  Class distribution:")
        for cid in sorted(VALID_CLASS_IDS):
            cname = CLASS_NAMES[cid]
            cnt = counts.get(cid, 0)
            pct = pcts.get(cid, 0.0)
            aggregate_counts[cid] += cnt
            bar = "#" * int(pct // 5)
            print(f"    [{cid}] {cname:<18}: {cnt:>8} px ({pct:>5.1f}%) {bar}")

        # 5. Composite creation and save
        composite = build_composite_panel(image_rgb, mask, alpha=alpha, sample_id=sid)

        if save_dir:
            out_name = f"inspect_{sid}.png"
            out_file = save_dir / out_name
            # OpenCV imwrite expects BGR format
            cv2.imwrite(str(out_file), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            print(f"  Saved inspection panel: {out_file}")

        if show_gui:
            cv2.imshow("StepView Dataset Inspector", cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0)
            if key == 27:  # ESC to exit
                break

    if show_gui:
        cv2.destroyAllWindows()

    # Aggregate Summary
    print("\n" + "=" * 80)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 80)
    print(f"Samples analyzed:      {len(indices_to_process)}")
    print(f"Total pixels examined: {total_analyzed_pixels:,}")

    if total_analyzed_pixels > 0:
        print("\nOverall Class Distribution:")
        for cid in sorted(VALID_CLASS_IDS):
            cname = CLASS_NAMES[cid]
            cnt = aggregate_counts[cid]
            pct = (cnt / total_analyzed_pixels) * 100.0
            print(f"  [{cid}] {cname:<18}: {cnt:>10,} px ({pct:>5.2f}%)")

    print(f"\nDefects / Issues Detected: {len(issues_found)}")
    for issue in issues_found:
        print(f"  - {issue}")

    return 0 if not issues_found else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StepView Dataset Inspection & Visualization Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to dataset root")
    parser.add_argument("--split", type=str, default="all", help="Dataset split ('train', 'val', 'test', 'all')")
    parser.add_argument("--sample-id", type=str, default=None, help="Inspect specific sample ID")
    parser.add_argument("--index", type=int, default=None, help="Inspect specific zero-based index")
    parser.add_argument("--num-samples", type=int, default=None, help="Maximum number of samples to inspect")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory to save rendered inspection panels")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha blending weight for mask overlay")
    parser.add_argument("--show", action="store_true", help="Display interactive GUI window (press any key to advance)")

    args = parser.parse_args()
    exit_code = inspect_dataset(
        data_dir=args.data_dir,
        split=args.split,
        sample_id_filter=args.sample_id,
        index_filter=args.index,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        alpha=args.alpha,
        show_gui=args.show,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
