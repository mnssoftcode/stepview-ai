#!/usr/bin/env python3
"""StepView — Scaled RUGD Dataset B Preparation Pipeline.

Selects 600 diverse, non-consecutive frames across 18 RUGD scenes,
fetches RGB frames selectively via HTTP Range requests without downloading the 5.6 GB archive,
converts annotation masks to 8-bit StepView indexed format (0..5),
generates scene-isolated splits (train: 380, val: 110, test: 110),
and writes metadata manifests.

Usage:
    python scripts/prepare_rugd_dataset_b.py --target-count 600
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import urllib.request
import zipfile
import zlib
import cv2
import numpy as np

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stepview.data.schema import NUM_CLASSES, SampleMetadata, TerrainClass, validate_mask_classes
from scripts.prepare_rugd_bootstrap import (
    COLOR_LUT,
    RUGD_RGB_TO_STEPVIEW,
    convert_rgb_mask_to_indexed,
    parse_rugd_filename,
)


RUGD_FRAMES_URL = "http://rugd.vision/data/RUGD_frames-with-annotations.zip"
# Central directory offset and size from previous inspection
CD_OFFSET = 5670298978
CD_SIZE = 999518

# Per-scene sampling quotas for 600 total frames
SCENE_QUOTAS: Dict[str, Tuple[str, int]] = {
    # Train Partition (380 samples, 11 scenes)
    "trail": ("train", 45),
    "trail-3": ("train", 40),
    "trail-4": ("train", 45),
    "trail-5": ("train", 35),
    "trail-6": ("train", 35),
    "trail-7": ("train", 25),
    "trail-9": ("train", 25),
    "trail-10": ("train", 25),
    "trail-12": ("train", 35),
    "trail-14": ("train", 30),
    "park-1": ("train", 40),

    # Validation Partition (110 samples, 3 scenes)
    "creek": ("val", 50),
    "park-2": ("val", 35),
    "trail-13": ("val", 25),

    # Test Partition (110 samples, 4 scenes)
    "trail-11": ("test", 40),
    "park-8": ("test", 25),
    "trail-15": ("test", 25),
    "village": ("test", 20),
}


def fetch_central_directory() -> Dict[str, Tuple[int, int, int, int]]:
    """Download and parse central directory from remote zip over HTTP range request.

    Returns:
        Dict mapping zip internal path -> (comp_method, comp_size, uncomp_size, local_offset)
    """
    print(f"Fetching ZIP Central Directory ({CD_SIZE / 1024:.1f} KB) from remote archive...")
    req = urllib.request.Request(
        RUGD_FRAMES_URL,
        headers={"Range": f"bytes={CD_OFFSET}-{CD_OFFSET + CD_SIZE - 1}"},
    )
    with urllib.request.urlopen(req) as resp:
        cd_data = resp.read()

    pos = 0
    file_index: Dict[str, Tuple[int, int, int, int]] = {}

    while pos < len(cd_data) and cd_data[pos : pos + 4] == b"PK\x01\x02":
        comp_method = struct.unpack("<H", cd_data[pos + 10 : pos + 12])[0]
        comp_size, uncomp_size = struct.unpack("<II", cd_data[pos + 20 : pos + 28])
        fname_len, extra_len, comment_len = struct.unpack("<HHH", cd_data[pos + 28 : pos + 34])
        local_offset = struct.unpack("<I", cd_data[pos + 42 : pos + 46])[0]
        fname = cd_data[pos + 46 : pos + 46 + fname_len].decode("utf-8", errors="ignore")

        extra = cd_data[pos + 46 + fname_len : pos + 46 + fname_len + extra_len]
        if local_offset == 0xFFFFFFFF or comp_size == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF:
            epos = 0
            while epos < len(extra):
                tag, elen = struct.unpack("<HH", extra[epos : epos + 4])
                if tag == 1:  # zip64
                    zpos = epos + 4
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = struct.unpack("<Q", extra[zpos : zpos + 8])[0]
                        zpos += 8
                    if comp_size == 0xFFFFFFFF:
                        comp_size = struct.unpack("<Q", extra[zpos : zpos + 8])[0]
                        zpos += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack("<Q", extra[zpos : zpos + 8])[0]
                        zpos += 8
                    break
                epos += 4 + elen

        file_index[fname] = (comp_method, comp_size, uncomp_size, local_offset)
        pos += 46 + fname_len + extra_len + comment_len

    print(f"Parsed {len(file_index)} entries from central directory.")
    return file_index


def download_single_frame(
    fname: str,
    file_info: Tuple[int, int, int, int],
) -> np.ndarray:
    """Download single frame from zip archive over HTTP range."""
    comp_method, comp_size, uncomp_size, local_offset = file_info

    # 1. Fetch local header to determine exact data start
    local_req = urllib.request.Request(
        RUGD_FRAMES_URL,
        headers={"Range": f"bytes={local_offset}-{local_offset + 30 + len(fname) + 256}"},
    )
    with urllib.request.urlopen(local_req) as resp:
        local_hdr = resp.read()

    lfname_len, lextra_len = struct.unpack("<HH", local_hdr[26:30])
    data_start = local_offset + 30 + lfname_len + lextra_len

    # 2. Fetch compressed payload
    data_req = urllib.request.Request(
        RUGD_FRAMES_URL,
        headers={"Range": f"bytes={data_start}-{data_start + comp_size - 1}"},
    )
    with urllib.request.urlopen(data_req) as resp:
        raw_data = resp.read()

    # 3. Decompress
    if comp_method == 8:
        decompressed = zlib.decompress(raw_data, -15)
    elif comp_method == 0:
        decompressed = raw_data
    else:
        raise ValueError(f"Unsupported compression method: {comp_method}")

    # 4. Decode with OpenCV
    arr = np.frombuffer(decompressed, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to decode image: {fname}")

    return img_bgr


def prepare_dataset_b(
    annotations_zip: Path,
    output_dir: Path,
    max_workers: int = 12,
) -> Dict[str, Any]:
    """Execute complete Dataset B curation and conversion."""
    print("=" * 80)
    print("StepView — Dataset B (Scaled RUGD) Pipeline")
    print(f"Annotations source: {annotations_zip.resolve()}")
    print(f"Output directory:   {output_dir.resolve()}")
    print("=" * 80)

    if not annotations_zip.is_file():
        raise FileNotFoundError(f"Annotations zip not found at {annotations_zip}")

    dest_images = output_dir / "images"
    dest_masks = output_dir / "masks"
    dest_metadata = output_dir / "metadata"
    dest_splits = output_dir / "splits"

    for d in (dest_images, dest_masks, dest_metadata, dest_splits):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Fetch remote central directory
    remote_index = fetch_central_directory()

    # 2. Open local annotations archive and discover available annotations by scene
    print("Scanning local annotations archive...")
    ann_by_scene: Dict[str, List[str]] = {scene: [] for scene in SCENE_QUOTAS}

    with zipfile.ZipFile(annotations_zip, "r") as z:
        for zname in z.namelist():
            if zname.endswith(".png") and "/" in zname:
                parts = zname.split("/")
                if len(parts) >= 3 and parts[0] == "RUGD_annotations":
                    scene = parts[1]
                    if scene in ann_by_scene:
                        ann_by_scene[scene].append(zname)

    # 3. Select frames per scene according to quotas with uniform temporal stride
    selected_targets: List[Tuple[str, str, str, str]] = []  # (scene, split, ann_zip_path, remote_img_path)

    for scene, (split, quota) in SCENE_QUOTAS.items():
        available = sorted(ann_by_scene[scene])
        num_avail = len(available)
        if num_avail == 0:
            print(f"[WARN] No annotations found for scene {scene}")
            continue

        stride = max(num_avail / quota, 1.0)
        selected_indices = [int(i * stride) for i in range(min(quota, num_avail))]

        for idx in selected_indices:
            ann_path = available[idx]
            frame_filename = Path(ann_path).name  # e.g. 'trail-4_00121.png'
            remote_img_path = f"RUGD_frames-with-annotations/{scene}/{frame_filename}"

            if remote_img_path in remote_index:
                selected_targets.append((scene, split, ann_path, remote_img_path))
            else:
                print(f"[SKIP] Remote image {remote_img_path} not found in zip index.")

    print(f"\nTotal frames selected for Dataset B: {len(selected_targets)}")

    # 4. Extract annotations from zipfile in memory
    print("Extracting annotation masks from local zip...")
    ann_images_map: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(annotations_zip, "r") as z:
        for scene, split, ann_path, remote_img_path in selected_targets:
            ann_bytes = z.read(ann_path)
            ann_arr = np.frombuffer(ann_bytes, dtype=np.uint8)
            ann_bgr = cv2.imdecode(ann_arr, cv2.IMREAD_COLOR)
            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            ann_images_map[remote_img_path] = ann_rgb

    # 5. Concurrently download RGB images over HTTP Range requests and convert
    print(f"Downloading {len(selected_targets)} RGB frames using {max_workers} threads...")
    manifest_records: List[Dict] = []
    split_samples: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    class_pixel_totals: Dict[int, int] = {c.value: 0 for c in TerrainClass}

    def process_item(item: Tuple[str, str, str, str]) -> Optional[Dict]:
        scene, split, ann_path, remote_img_path = item
        file_info = remote_index[remote_img_path]

        try:
            bgr_img = download_single_frame(remote_img_path, file_info)
        except Exception as e:
            print(f"[ERROR] Failed downloading {remote_img_path}: {e}")
            return None

        rgb_ann = ann_images_map[remote_img_path]
        if bgr_img.shape[:2] != rgb_ann.shape[:2]:
            print(f"[ERROR] Shape mismatch for {remote_img_path}")
            return None

        indexed_mask, _ = convert_rgb_mask_to_indexed(rgb_ann)
        validate_mask_classes(indexed_mask)

        # Standard StepView naming
        stem = Path(remote_img_path).stem
        scene_id, frame_id = parse_rugd_filename(stem)
        sample_id = f"rugd_{scene_id}_frame{frame_id}"
        out_name = f"{sample_id}.png"

        # Save files
        cv2.imwrite(str(dest_images / out_name), bgr_img)
        cv2.imwrite(str(dest_masks / out_name), indexed_mask)

        # Compute counts
        u_classes, counts = np.unique(indexed_mask, return_counts=True)
        counts_dict = {int(k): int(v) for k, v in zip(u_classes, counts)}

        meta = SampleMetadata(
            sample_id=sample_id,
            image_rel_path=f"images/{out_name}",
            mask_rel_path=f"masks/{out_name}",
            scene_id=f"rugd_{scene_id}",
            split=split,
            camera_pitch_deg=20.0,
            camera_height_m=0.6,
            terrain_type=scene_id,
            lighting_condition="daylight_outdoor",
            weather="dry",
            extra_attributes={
                "source": "RUGD_Dataset_B",
                "original_filename": stem,
                "annotation_type": "converted_semantic_mask",
                "foot_placement_verified": False,
            },
        )

        return {
            "meta": meta.to_dict(),
            "sample_id": sample_id,
            "split": split,
            "counts": counts_dict,
        }

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in selected_targets}
        for future in as_completed(future_to_item):
            res = future.result()
            if res is not None:
                manifest_records.append(res["meta"])
                split_samples[res["split"]].append(res["sample_id"])
                for k, v in res["counts"].items():
                    class_pixel_totals[k] += v

            completed_count += 1
            if completed_count % 50 == 0 or completed_count == len(selected_targets):
                print(f"  Processed [{completed_count}/{len(selected_targets)}] frames...")

    # 6. Write split files
    for split_name, s_ids in split_samples.items():
        split_file = dest_splits / f"{split_name}.txt"
        with open(split_file, "w", encoding="utf-8") as f:
            for sid in sorted(s_ids):
                f.write(f"{sid}\n")
        print(f"Wrote {len(s_ids)} sample(s) to splits/{split_name}.txt")

    # 7. Write manifest.json
    manifest_data = {
        "dataset_name": "StepView RUGD Dataset B (Pre-training)",
        "version": "0.2.0",
        "total_samples": len(manifest_records),
        "samples": manifest_records,
    }
    with open(dest_metadata / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Wrote master manifest with {len(manifest_records)} records to metadata/manifest.json")

    # 8. Write class stats
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
    print("DATASET B INGESTION SUMMARY")
    print("=" * 80)
    print(f"Total processed:  {len(manifest_records)}")
    print(f"Train samples:    {len(split_samples['train'])} ({len(split_samples['train'])/len(manifest_records)*100:.1f}%)")
    print(f"Val samples:      {len(split_samples['val'])} ({len(split_samples['val'])/len(manifest_records)*100:.1f}%)")
    print(f"Test samples:     {len(split_samples['test'])} ({len(split_samples['test'])/len(manifest_records)*100:.1f}%)")
    print("\nOverall Class Pixel Distribution:")
    for cid, data in class_stats_data["classes"].items():
        print(f"  [{cid}] {data['name']:<18}: {data['pixel_count']:>12,} px ({data['percentage']:>5.2f}%)")

    return {
        "total": len(manifest_records),
        "train": len(split_samples["train"]),
        "val": len(split_samples["val"]),
        "test": len(split_samples["test"]),
        "stats": class_stats_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare StepView Dataset B from RUGD")
    parser.add_argument(
        "--annotations-zip",
        type=Path,
        default=Path("data/raw/rugd/RUGD_annotations.zip"),
        help="Path to local RUGD_annotations.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="StepView dataset root destination",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of concurrent download worker threads",
    )
    args = parser.parse_args()
    prepare_dataset_b(
        annotations_zip=args.annotations_zip,
        output_dir=args.output_dir,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
