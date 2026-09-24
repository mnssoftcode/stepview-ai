# StepView — RUGD Bootstrap Ingestion & Visual Inspection Report

**Date:** 2026-09-24  
**Author:** Lead AI/ML Engineer (Technical Project Agent)  
**Status:** Ingestion Complete & Validated  
**Artifacts Generated:** 26 images, 26 masks, 1 manifest, 3 split files, 26 inspection panels

---

## 1. Executive Summary

A small, high-diversity bootstrap subset of the public **RUGD (Robot Unstructured Ground Driving)** dataset was acquired, converted to StepView's canonical 6-class segmentation format, partitioned by scene, and visually audited using `scripts/visualize_dataset.py`.

* **Raw Data Download:** Utilized the official RUGD sample distribution (`http://rugd.vision/data/RUGD_sample-data.zip`, 19.9 MB), entirely avoiding the unneeded 5.6 GB full video archive.
* **Preserved Raw Source:** Intact under `data/raw/rugd/RUGD_sample-data/`.
* **Converted Samples:** 26 frames across 18 distinct physical scenes placed into `data/images/` and `data/masks/` with the `rugd_` namespace prefix.
* **Integrity Gate:** 0 dimension mismatches, 0 unmapped pixels, 0 missing files. All 15 unit tests pass.

---

## 2. Selected Scenes & Diversity Breakdown

The 26 frames were drawn from 18 distinct scenes representing varied natural and semi-structured outdoor terrain:

| Terrain Category | Selected Scenes | Frame Count | Key Visual Characteristics |
|---|---|---|---|
| **Rocky / Creekbed** | `creek`, `trail-6`, `trail-11` | 6 | Loose river stones, rocky trail patches, jagged rock beds, water margins. |
| **Dirt & Trail Path** | `trail`, `trail-5`, `trail-12` | 4 | Compacted dirt, sloped trail descent, gravel tracks. |
| **Woodland & Vegetation** | `trail-7`, `trail-9`, `trail-14`, `trail-15`, `park-1` | 6 | Dense forest floor, leaf litter, dappled shadows, tall grass shoulders. |
| **Uneven / Rough Path** | `trail-3`, `trail-4`, `trail-10`, `trail-13` | 6 | Root intrusions, uneven trail tread, brush encroachment. |
| **Mixed / Transition** | `park-2`, `park-8`, `village` | 4 | Manicured park paths, wide fields, gravel roads with buildings/posts. |

---

## 3. Scene-Level Split Allocation

To prevent data leakage, splits were strictly assigned by **scene ID** (no frames from the same scene appear across multiple partitions):

* **Train Partition (16 samples, 61.5%):**
  * Scenes: `trail` (2), `trail-3` (2), `trail-4` (2), `trail-5` (1), `trail-6` (2), `trail-7` (1), `trail-9` (1), `trail-10` (1), `trail-12` (1), `trail-14` (1), `park-1` (2).
* **Validation Partition (5 samples, 19.2%):**
  * Scenes: `creek` (2), `park-2` (2), `trail-13` (1).
* **Test Partition (5 samples, 19.2%):**
  * Scenes: `trail-11` (2), `park-8` (1), `trail-15` (1), `village` (1).

Split text files generated: `data/splits/train.txt`, `data/splits/val.txt`, `data/splits/test.txt`.

---

## 4. Class Distribution & Statistical Analysis

Analysis across the 9,838,400 pixels in the 26 converted masks:

| Class ID | StepView Class Name | Pixel Count | Percentage | Observations |
|---|---|---|---|---|
| **0** | `background` | 1,051,397 px | **10.69%** | Sky, distant trees, vehicles, buildings. |
| **1** | `ground` | 1,235,788 px | **12.56%** | Walkable dirt paths, gravel, asphalt, bridges. |
| **2** | `obstacle` | 284,614 px | **2.89%** | Rocks, tree roots, fallen logs, posts, furniture. |
| **3** | `vegetation` | 6,037,811 px | **61.37%** | Trees, bushes, tall grass (dominant due to low robot camera tilt). |
| **4** | `hole` | 0 px | **0.00%** | **Not represented in RUGD taxonomy.** |
| **5** | `uncertain_surface` | 1,228,790 px | **12.49%** | Riverbeds (`rock-bed`), standing water, mulch, sand. |

### Key Observations:
1. **Vegetation Imbalance:** 61.37% of all pixels are vegetation due to the low-angle, forward-looking camera capturing forest canopy and tall roadside trees.
2. **Obstacle Sparsity:** Obstacles account for only 2.89% of pixels, requiring weighted cross-entropy or focal loss during training.
3. **Absence of Class 4 (`hole`):** RUGD has no drop-off/hole annotation class. This gap must be addressed during Phase 1 self-collected smartphone data.

---

## 5. Visual Inspection of Converted Overlays

Generated composite panels in `experiments/visualizations/rugd_bootstrap/`:

1. **`inspect_rugd_creek_frame00001.png` (Rocky / Water):**
   * *Alignment:* Riverbed rocks accurately mapped to `uncertain_surface` (amber); trees and bushes to `vegetation` (green); sky and vehicle to `background` (gray).
   * *Assessment:* Clean semantic boundary without color bleed.
2. **`inspect_rugd_trail-11_frame00001.png` (Rough Trail):**
   * *Alignment:* Foreground gravel path clearly demarcated as `ground` (bright green), bordered by tall grass `vegetation` (dark green).
   * *Assessment:* Demonstrates realistic path segmentation in natural lighting.
3. **`inspect_rugd_village_frame00003.png` (Semi-structured Ground):**
   * *Alignment:* Roadway is `ground`; posts, benches, crates are `obstacle` (red); buildings/sky are `background` (black).
   * *Assessment:* Excellent separation between man-made ground and obstacles.

---

## 6. Critical Domain Limitations & Non-Safety Notice

> [!WARNING]
> Converted RUGD masks are **strictly a proxy for testing the ML segmentation engineering pipeline**. They have three major limitations:
> 1. **Camera Angle:** Captured at ~0.6m height looking nearly horizontal (~15° pitch), whereas StepView targets handheld/chest height (1.0–1.4m) looking down at 35°–50°.
> 2. **Robot vs. Human Sizing:** Obstacles were labeled for rover wheel clearance, not human footstep landing stability.
> 3. **Missing Drop-Offs:** Zero representations of holes or cliff edges.
> 
> **Decision Rule:** Converted RUGD data is validated and suitable for **Phase 2 baseline segmentation pipeline development**, but must be augmented with real pedestrian smartphone data before evaluating candidate foot-placement logic.

---

## 7. Next Recommended Task

Now that real terrain data is ingested, verified, and passing all tests, the data pipeline proof is **complete**.

Proceed to **Phase 2 (Segmentation Baseline)**:
1. Design and implement a compact PyTorch segmentation model baseline (MobileNetV3-Small encoder + LR-ASPP segmentation head).
2. Implement training loop with class-weighted Cross-Entropy + Dice loss.
3. Train baseline model on `data/splits/train.txt` and validate on `data/splits/val.txt`.
