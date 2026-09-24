# StepView — Dataset B (RUGD Pre-training) Selection & Sampling Specification

**Version:** 1.0  
**Date:** 2026-09-24  
**Applies to:** Phase 2 Semantic Segmentation Pre-training Dataset

---

## 1. Objective & Scope

Dataset B scales StepView's training assets from the 26-frame bootstrap set to **600 carefully curated real-terrain frames** from the official RUGD repository.

> [!CAUTION]
> **Semantic Pre-training Only:**  
> Dataset B is utilized strictly to train general visual feature representation (distinguishing ground, obstacles, vegetation, water/sand, and background). It is **NOT** human foot-placement ground truth and must not be used to claim physical safety.

---

## 2. Source & Provenance

* **Source Repository:** Official RUGD Project (`http://rugd.vision/`, US Army Research Laboratory).
* **Archive Assets:**
  * Raw Image Frames: `http://rugd.vision/data/RUGD_frames-with-annotations.zip` (selective HTTP Range extraction; multi-gigabyte raw archive preserved on S3).
  * Annotation Masks: `http://rugd.vision/data/RUGD_annotations.zip` (58.7 MB, preserved in `data/raw/rugd/RUGD_annotations.zip`).
* **Citation:** Wigness et al., *A Robot Unstructured Ground Driving (RUGD) Dataset for Semantic Segmentation in Natural Environments*, IROS 2019.
* **Licensing:** Academic / Non-Commercial Research License.

---

## 3. Sampling Strategy & Scene Diversity

To prevent frame-to-frame redundancy (which provides zero gradient signal while artificially inflating validation scores), samples are selected using a **uniform stride across each scene's temporal timeline**.

Total frames selected: **600 frames** across 18 scenes.

### 3.1 Split Allocation & Scene Breakdown

| Split | Scene ID | Available Annotations | Sampled Frames | Sampling Stride | Terrain Characteristics & Contribution |
|---|---|---|---|---|---|
| **Train** | `trail` | 693 | **45** | Every ~15th frame | Natural compacted dirt path, varied sun/shade transitions. |
| **Train** | `trail-3` | 580 | **40** | Every ~14th frame | Uneven woodland floor, encroaching roots, dense tree cover. |
| **Train** | `trail-4` | 758 | **45** | Every ~16th frame | Exposed root systems, uneven trail tread, leaf clusters. |
| **Train** | `trail-5` | 376 | **35** | Every ~10th frame | Broad dirt path, gravel shoulders, open lighting. |
| **Train** | `trail-6` | 439 | **35** | Every ~12th frame | Rocky trail sections, loose stones, uneven footings. |
| **Train** | `trail-7` | 289 | **25** | Every ~11th frame | Heavy leaf litter, sparse path boundary, forest floor. |
| **Train** | `trail-9` | 59 | **25** | Every ~2nd frame | Deep woodland corridor, dark understory lighting. |
| **Train** | `trail-10` | 49 | **25** | Every ~2nd frame | Narrow single-track trail, heavy side brush. |
| **Train** | `trail-12` | 352 | **35** | Every ~10th frame | Moderate slope descent, dirt berms, dry soil. |
| **Train** | `trail-14` | 314 | **30** | Every ~10th frame | Overhanging tree branches, canopy-filtered lighting. |
| **Train** | `park-1` | 627 | **40** | Every ~15th frame | Open grass lawns, paved park transitions, tree trunks. |
| *Subtotal* | **Train (11 scenes)** | 4,536 | **380 (63.3%)** | — | Comprehensive coverage of dirt, roots, rocks, and trees. |
| **Val** | `creek` | 836 | **50** | Every ~16th frame | Uneven riverbed rocks, running water, wet stone textures. |
| **Val** | `park-2` | 656 | **35** | Every ~18th frame | Park pathways, curbs, manicured grass shoulders. |
| **Val** | `trail-13` | 172 | **25** | Every ~7th frame | Rough path, high vegetation, challenging uneven ground. |
| *Subtotal* | **Val (3 scenes)** | 1,664 | **110 (18.3%)** | — | Held-out scenes testing generalization to water, rocks, paths. |
| **Test** | `trail-11` | 438 | **40** | Every ~11th frame | Rocky path with loose stones, rough natural path tread. |
| **Test** | `park-8` | 357 | **25** | Every ~14th frame | Broad open field, distant tree line, direct sunlight. |
| **Test** | `trail-15` | 324 | **25** | Every ~13th frame | Shaded woodland track, subtle soil texture differences. |
| **Test** | `village` | 117 | **20** | Every ~5th frame | Gravel road, concrete curbs, posts, buildings. |
| *Subtotal* | **Test (4 scenes)** | 1,236 | **110 (18.3%)** | — | Completely unseen held-out test scenes. |
| **Total** | **All 18 scenes** | 7,436 | **600 (100.0%)** | — | Zero frame or scene leakage across splits. |

---

## 4. Class Treatment & The Missing `HOLE` Class

* StepView Class 4 (`HOLE`): Has **0% ground-truth representation** in RUGD.
* **Treatment Rules:**
  1. No synthetic hole labels are introduced.
  2. In evaluation, the active mean IoU (`mean_iou`) is computed over classes present in the validation split (`{0, 1, 2, 3, 5}`). The metric `all_class_mean_iou` is also reported for reference.
  3. Class weights dynamically assign weight 0.0 to Class 4 to prevent gradient distortion.

---

## 5. Directory Contract for Dataset B

```text
data/
├── raw/
│   └── rugd/
│       ├── RUGD_sample-data.zip
│       └── RUGD_annotations.zip
├── images/
│   ├── rugd_trail_frame00001.png
│   └── ... (600 frames total)
├── masks/
│   ├── rugd_trail_frame00001.png
│   └── ... (600 8-bit single-channel indexed PNGs)
├── metadata/
│   ├── manifest.json         # Master manifest tracking all 600 records
│   └── class_stats.json      # Complete class pixel frequencies
└── splits/
    ├── train.txt             # 380 samples
    ├── val.txt               # 110 samples
    └── test.txt              # 110 samples
```
