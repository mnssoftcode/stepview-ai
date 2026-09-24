#!/usr/bin/env python3
"""StepView — RUGD Bootstrap Data Preprocessing & Conversion Pipeline.

Reads curated RUGD sample data, converts RGB color annotations to StepView
canonical 8-bit indexed masks (0..5), assigns scene-level train/val/test splits,
validates integrity, and outputs StepView-contract data.

Usage:
    python scripts/prepare_rugd_bootstrap.py --raw-dir data/raw/rugd/RUGD_sample-data --output-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from stepview.data.schema import (
    NUM_CLASSES,
    SampleMetadata,
    TerrainClass,
    VALID_CLASS_IDS,
    validate_mask_classes,
)


# Official RUGD 24-bit RGB values mapped to StepView TerrainClass
# Format: (R, G, B) -> TerrainClass
RUGD_RGB_TO_STEPVIEW: Dict[Tuple[int, int, int], TerrainClass] = {
    (0, 0, 0): TerrainClass.BACKGROUND,         # 0 void
    (108, 64, 20): TerrainClass.GROUND,         # 1 dirt
    (255, 229, 204): TerrainClass.UNCERTAIN_SURFACE, # 2 sand
    (0, 102, 0): TerrainClass.VEGETATION,       # 3 grass
    (0, 255, 0): TerrainClass.VEGETATION,       # 4 tree
    (0, 153, 153): TerrainClass.OBSTACLE,       # 5 pole
    (0, 128, 255): TerrainClass.UNCERTAIN_SURFACE, # 6 water
    (0, 0, 255): TerrainClass.BACKGROUND,       # 7 sky
    (255, 255, 0): TerrainClass.BACKGROUND,     # 8 vehicle
    (255, 0, 127): TerrainClass.OBSTACLE,       # 9 container/generic-object
    (64, 64, 64): TerrainClass.GROUND,          # 10 asphalt
    (255, 128, 0): TerrainClass.GROUND,         # 11 gravel
    (255, 0, 0): TerrainClass.BACKGROUND,       # 12 building
    (153, 76, 0): TerrainClass.UNCERTAIN_SURFACE, # 13 mulch
    (102, 102, 0): TerrainClass.UNCERTAIN_SURFACE, # 14 rock-bed
    (102, 0, 0): TerrainClass.OBSTACLE,         # 15 log
    (0, 255, 128): TerrainClass.BACKGROUND,     # 16 bicycle
    (204, 153, 255): TerrainClass.BACKGROUND,   # 17 person
    (102, 0, 204): TerrainClass.OBSTACLE,       # 18 fence
    (255, 153, 204): TerrainClass.VEGETATION,   # 19 bush
    (0, 102, 102): TerrainClass.OBSTACLE,       # 20 sign
    (153, 204, 255): TerrainClass.OBSTACLE,     # 21 rock
    (102, 255, 255): TerrainClass.GROUND,       # 22 bridge
    (101, 101, 11): TerrainClass.GROUND,        # 23 concrete
    (114, 85, 47): TerrainClass.OBSTACLE,       # 24 picnic-table
}


# Scene-level split partition (Strictly no scene overlap between splits)
SCENE_SPLITS: Dict[str, str] = {
    # Train Partition (11 scenes, ~61.5% of samples)
    "trail": "train",
    "trail-3": "train",
    "trail-4": "train",
    "trail-5": "train",
    "trail-6": "train",
    "trail-7": "train",
    "trail-9": "train",
    "trail-10": "train",
    "trail-12": "train",
    "trail-14": "train",
    "park-1": "train",

    # Validation Partition (3 scenes, ~19.2% of samples)
    "creek": "val",
    "park-2": "val",
    "trail-13": "val",

    # Test Partition (4 scenes, ~19.2% of samples)
    "trail-11": "test",
    "park-8": "test",
    "trail-15": "test",
    "village": "test",
}


def build_fast_color_lut() -> np.ndarray:
    """Precompute a 24-bit RGB lookup table for O(1) pixel mapping."""
    # 256^3 is 16MB array, very fast
    lut = np.full(256 * 256 * 256, TerrainClass.BACKGROUND.value, dtype=np.uint8)
    for (r, g, b), target_class in RUGD_RGB_TO_STEPVIEW.items():
        key = (r << 16) | (g << 8) | b
        lut[key] = target_class.value
    return lut


COLOR_LUT = build_fast_color_lut()


def convert_rgb_mask_to_indexed(rgb_mask: np.ndarray) -> Tuple[np.ndarray, Set[Tuple[int, int, int]]]:
    """Convert a 3-channel RGB annotation mask into a 1-channel uint8 StepView indexed mask.

    Args:
        rgb_mask: (H, W, 3) uint8 array in RGB channel order.

    Returns:
        Tuple of (indexed_mask, unmapped_colors_detected).
    """
    h, w = rgb_mask.shape[:2]
    # Pack RGB into 24-bit int keys
    packed_keys = (
        (rgb_mask[:, :, 0].astype(np.uint32) << 16)
        | (rgb_mask[:, :, 1].astype(np.uint32) << 8)
        | rgb_mask[:, :, 2].astype(np.uint32)
    )

    indexed_mask = COLOR_LUT[packed_keys]

    # Detect any unmapped colors
    unique_keys = np.unique(packed_keys)
    unmapped_colors: Set[Tuple[int, int, int]] = set()
    for key in unique_keys:
        r = int((key >> 16) & 0xFF)
        g = int((key >> 8) & 0xFF)
        b = int(key & 0xFF)
        if (r, g, b) not in RUGD_RGB_TO_STEPVIEW:
            unmapped_colors.add((r, g, b))

    return indexed_mask, unmapped_colors


def parse_rugd_filename(filename: str) -> Tuple[str, str]:
    """Extract scene ID and frame ID from RUGD filename (e.g. 'trail-11_02001.png')."""
    stem = Path(filename).stem
    if "_" in stem:
        scene_id, frame_id = stem.rsplit("_", 1)
    else:
        scene_id, frame_id = stem, "00000"
    return scene_id, frame_id


def prepare_bootstrap_dataset(
    raw_dir: Path,
    output_dir: Path,
) -> Dict[str, int]:
    """Execute the RUGD bootstrap preparation and conversion pipeline."""
    print("=" * 80)
    print("StepView — RUGD Bootstrap Data Preprocessing")
    print(f"Raw source directory:   {raw_dir.resolve()}")
    print(f"Output destination:     {output_dir.resolve()}")
    print("=" * 80)

    images_src = raw_dir / "images"
    annotations_src = raw_dir / "annotations"

    if not images_src.is_dir() or not annotations_src.is_dir():
        raise FileNotFoundError(
            f"Expected images/ and annotations/ under {raw_dir}. Please verify RUGD extraction."
        )

    # Destination directories
    dest_images = output_dir / "images"
    dest_masks = output_dir / "masks"
    dest_metadata = output_dir / "metadata"
    dest_splits = output_dir / "splits"

    for d in (dest_images, dest_masks, dest_metadata, dest_splits):
        d.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(images_src.glob("*.png")) + list(images_src.glob("*.jpg")))
    print(f"Found {len(image_files)} raw candidate frame(s).")

    manifest_records: List[Dict] = []
    split_samples: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    class_pixel_totals: Dict[int, int] = {c.value: 0 for c in TerrainClass}

    stats = {
        "processed": 0,
        "train_count": 0,
        "val_count": 0,
        "test_count": 0,
        "rejected": 0,
    }

    for img_path in image_files:
        stem = img_path.stem
        ann_path = annotations_src / f"{stem}.png"

        if not ann_path.is_file():
            print(f"[SKIP] Missing annotation for image: {img_path.name}")
            stats["rejected"] += 1
            continue

        scene_id, frame_id = parse_rugd_filename(img_path.name)
        assigned_split = SCENE_SPLITS.get(scene_id, "train")

        # Load raw RGB image (OpenCV loads BGR, convert to RGB)
        bgr_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr_img is None:
            print(f"[ERROR] Corrupted image: {img_path}")
            stats["rejected"] += 1
            continue
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # Load raw annotation (OpenCV loads BGR, convert to RGB)
        bgr_ann = cv2.imread(str(ann_path), cv2.IMREAD_COLOR)
        if bgr_ann is None:
            print(f"[ERROR] Corrupted annotation: {ann_path}")
            stats["rejected"] += 1
            continue
        rgb_ann = cv2.cvtColor(bgr_ann, cv2.COLOR_BGR2RGB)

        # Dimension alignment check
        if rgb_img.shape[:2] != rgb_ann.shape[:2]:
            print(f"[ERROR] Dimension mismatch for {stem}: img {rgb_img.shape[:2]} vs ann {rgb_ann.shape[:2]}")
            stats["rejected"] += 1
            continue

        # Convert annotation to single-channel uint8 StepView indexed mask
        indexed_mask, unmapped_colors = convert_rgb_mask_to_indexed(rgb_ann)
        if unmapped_colors:
            print(f"[WARN] Sample {stem} contained unmapped colors: {unmapped_colors}")

        # Validate mask classes
        validate_mask_classes(indexed_mask)

        # Standardize sample naming: rugd_{scene_id}_frame{frame_id}
        sample_id = f"rugd_{scene_id}_frame{frame_id}"
        out_img_name = f"{sample_id}.png"
        out_mask_name = f"{sample_id}.png"

        out_img_path = dest_images / out_img_name
        out_mask_path = dest_masks / out_mask_name

        # Save image as PNG (lossless)
        cv2.imwrite(str(out_img_path), bgr_img)
        # Save mask as 8-bit single-channel indexed PNG
        cv2.imwrite(str(out_mask_path), indexed_mask)

        # Pixel statistics
        u_classes, counts = np.unique(indexed_mask, return_counts=True)
        for u_c, cnt in zip(u_classes, counts):
            class_pixel_totals[int(u_c)] += int(cnt)

        # Build metadata record
        meta = SampleMetadata(
            sample_id=sample_id,
            image_rel_path=f"images/{out_img_name}",
            mask_rel_path=f"masks/{out_mask_name}",
            scene_id=f"rugd_{scene_id}",
            split=assigned_split,
            camera_pitch_deg=20.0, # Robot platform orientation (initial estimate)
            camera_height_m=0.6,   # Ground rover height
            terrain_type=scene_id,
            lighting_condition="daylight_outdoor",
            weather="dry",
            extra_attributes={
                "source": "RUGD",
                "original_filename": img_path.name,
                "annotation_type": "converted_semantic_mask",
                "foot_placement_verified": False,
            },
        )

        manifest_records.append(meta.to_dict())
        split_samples[assigned_split].append(sample_id)
        stats["processed"] += 1

    # Write split files
    for split_name, s_ids in split_samples.items():
        split_file = dest_splits / f"{split_name}.txt"
        with open(split_file, "w", encoding="utf-8") as f:
            for sid in sorted(s_ids):
                f.write(f"{sid}\n")
        stats[f"{split_name}_count"] = len(s_ids)
        print(f"Wrote {len(s_ids)} sample(s) to splits/{split_name}.txt")

    # Write manifest.json
    manifest_data = {
        "dataset_name": "StepView RUGD Bootstrap",
        "version": "0.1.0",
        "total_samples": stats["processed"],
        "samples": manifest_records,
    }
    with open(dest_metadata / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Wrote manifest with {len(manifest_records)} records to metadata/manifest.json")

    # Write class distribution stats
    total_pixels = sum(class_pixel_totals.values())
    class_stats_data = {
        "total_pixels": total_pixels,
        "classes": {
            cid: {
                "name": TerrainClass(cid).name.lower(),
                "pixel_count": class_pixel_totals[cid],
                "percentage": (class_pixel_totals[cid] / total_pixels * 100.0) if total_pixels > 0 else 0.0,
            }
            for cid in sorted(class_pixel_totals.keys())
        },
    }
    with open(dest_metadata / "class_stats.json", "w", encoding="utf-8") as f:
        json.dump(class_stats_data, f, indent=2)

    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"Total processed:  {stats['processed']}")
    print(f"Train samples:    {stats['train_count']} ({stats['train_count']/stats['processed']*100:.1f}%)")
    print(f"Val samples:      {stats['val_count']} ({stats['val_count']/stats['processed']*100:.1f}%)")
    print(f"Test samples:     {stats['test_count']} ({stats['test_count']/stats['processed']*100:.1f}%)")
    print("\nOverall Class Pixel Distribution:")
    for cid, data in class_stats_data["classes"].items():
        print(f"  [{cid}] {data['name']:<18}: {data['pixel_count']:>10,} px ({data['percentage']:>5.2f}%)")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RUGD sample data to StepView format")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/rugd/RUGD_sample-data"),
        help="Path to raw extracted RUGD data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Path to output StepView dataset root directory",
    )
    args = parser.parse_args()
    prepare_bootstrap_dataset(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
