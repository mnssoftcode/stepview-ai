# StepView — Dataset Contract & Manifest Specification

**Version:** 1.0  
**Date:** 2026-09-24  
**Applies to:** StepView Data Ingestion, Storage, and Loader Interfaces

---

## 1. Directory Structure Contract

The StepView dataset directory adheres to the following layout:

```text
data/
├── raw/                      # Unprocessed video recordings, original raw high-res images
│   └── 2026-09-24_trail_walk_01.mp4
├── images/                   # Preprocessed RGB image frames (standardized naming)
│   ├── scene001_frame0001.jpg
│   ├── scene001_frame0002.jpg
│   └── scene002_frame0001.jpg
├── masks/                    # Single-channel 8-bit indexed segmentation masks
│   ├── scene001_frame0001.png
│   ├── scene001_frame0002.png
│   └── scene002_frame0001.png
├── metadata/                 # Dataset manifests and quality audit records
│   ├── manifest.json         # Master metadata manifest
│   └── class_stats.json      # Precomputed class frequency stats
└── splits/                   # Train/Val/Test split definitions (one sample_id per line)
    ├── train.txt
    ├── val.txt
    └── test.txt
```

---

## 2. Naming Conventions & Integrity Rules

1. **Strict 1-to-1 Correspondence:**  
   Every image in `data/images/` must have an identically named mask in `data/masks/` with a `.png` extension:
   * Image: `data/images/{scene_id}_{frame_id}.jpg` (or `.png`)
   * Mask: `data/masks/{scene_id}_{frame_id}.png`
2. **Deterministic Sample IDs:**  
   The `sample_id` is the stem of the file without extension (e.g. `scene001_frame0001`).
3. **Zero Padding:**  
   Frame IDs must use 4-digit zero padding (`frame0001`, `frame0002`, ..., `frame9999`) to preserve chronological and lexical sort ordering.
4. **Resolution Identity:**  
   The pixel width and height of an image and its corresponding mask must match exactly.
5. **Mask Encoding:**  
   Masks must be saved as uncompressed or PNG lossless 8-bit single-channel (grayscale) images. Pixel values must strictly be integer IDs in the range `0..5`.

---

## 3. Metadata Manifest Schema (`manifest.json`)

The master metadata file `data/metadata/manifest.json` provides rich scene context for split stratification, camera calibration experiments, and auditability.

### Schema Fields

| Field Name | Type | Requirement | Description |
|---|---|---|---|
| `sample_id` | string | **Required** | Unique identifier (e.g. `scene001_frame0001`). |
| `scene_id` | string | **Required** | Physical location / continuous shot ID (prevents train/val leakage). |
| `image_rel_path` | string | **Required** | Path relative to dataset root (e.g. `images/scene001_frame0001.jpg`). |
| `mask_rel_path` | string | **Required** | Path relative to dataset root (e.g. `masks/scene001_frame0001.png`). |
| `split` | string | **Required** | Assignment: `"train"`, `"val"`, or `"test"`. |
| `source` | string | Optional | Origin: `"self_collected"`, `"synthetic"`, or public dataset name. |
| `capture_device` | string | Optional | Hardware sensor (e.g. `"iPhone 14"`, `"Pixel 7"`). |
| `image_resolution` | [int, int] | Optional | `[width, height]` in pixels (e.g. `[1280, 720]`). |
| `camera_height_m` | float | Optional | Approximate height of camera above ground plane in meters (~1.0–1.4m). |
| `camera_pitch_deg` | float | Optional | Approximate downward pitch angle from horizontal (~35.0–50.0°). |
| `terrain_type` | string | Optional | Category: `"dirt_trail"`, `"rocky_path"`, `"gravel"`, `"forest_floor"`. |
| `lighting_condition` | string | Optional | Category: `"direct_sun"`, `"dappled_shade"`, `"overcast"`, `"golden_hour"`. |
| `weather` | string | Optional | Description: `"dry"`, `"recent_rain"`, `"foggy"`. |
| `annotation_status` | string | Optional | `"draft"`, `"reviewed"`, `"approved"`. |
| `annotator` | string | Optional | Name / ID of annotator or tool. |
| `annotation_version` | string | Optional | Annotation guidelines version (e.g. `"v1.0"`). |
| `extra_attributes` | object | Optional | Key-value store for experimental tags. |

---

## 4. Concrete Manifest Example (`data/metadata/manifest.json`)

```json
{
  "dataset_name": "StepView Bootstrap",
  "version": "0.1.0",
  "created_at": "2026-09-24",
  "samples": [
    {
      "sample_id": "scene001_frame0001",
      "scene_id": "scene001",
      "image_rel_path": "images/scene001_frame0001.jpg",
      "mask_rel_path": "masks/scene001_frame0001.png",
      "split": "train",
      "source": "self_collected",
      "capture_device": "Pixel 7 Pro",
      "image_resolution": [1280, 720],
      "camera_height_m": 1.25,
      "camera_pitch_deg": 42.0,
      "terrain_type": "rocky_dirt_trail",
      "lighting_condition": "dappled_shade",
      "weather": "dry",
      "annotation_status": "approved",
      "annotator": "lead_ml_engineer",
      "annotation_version": "v1.0",
      "extra_attributes": {
        "has_puddles": false,
        "hazard_present": true
      }
    },
    {
      "sample_id": "scene002_frame0045",
      "scene_id": "scene002",
      "image_rel_path": "images/scene002_frame0045.jpg",
      "mask_rel_path": "masks/scene002_frame0045.png",
      "split": "val",
      "source": "self_collected",
      "capture_device": "iPhone 14",
      "image_resolution": [1920, 1080],
      "camera_height_m": 1.30,
      "camera_pitch_deg": 38.5,
      "terrain_type": "forest_roots_and_mud",
      "lighting_condition": "overcast",
      "weather": "recent_rain",
      "annotation_status": "approved",
      "annotator": "lead_ml_engineer",
      "annotation_version": "v1.0",
      "extra_attributes": {
        "has_puddles": true,
        "hazard_present": true
      }
    }
  ]
}
```

---

## 5. Train/Val/Test Split Rule (Leakage Prevention)

To eliminate data leakage between train, validation, and test partitions:
* **Splits are partitioned strictly by `scene_id`**, never by random sampling of consecutive frames.
* Frames originating from the same physical trail recording must share the same `scene_id` and must all reside in the same split file (`train.txt`, `val.txt`, or `test.txt`).
