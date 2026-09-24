# StepView — Public Terrain Dataset Evaluation

**Version:** 1.0  
**Date:** 2026-09-24  
**Status:** Evaluation Report (No Public Data Downloaded Yet)

---

## 1. Executive Summary & Policy

Public datasets offer valuable pre-training diversity and early bootstrapping potential. However, **public datasets cannot replace self-collected StepView data** because:
1. Camera viewpoints in existing datasets are predominantly robot- or vehicle-mounted (forward-facing, lower camera height ~0.5–0.8m) rather than human walking perspectives (handheld/chest ~1.0–1.4m pitched 35°–50° downward).
2. Existing class taxonomies are designed for vehicle navigability (e.g. "can a 2-ton truck drive over this?") rather than human foot placement (e.g. "can an adult foot step safely here without twisting an ankle?").

> [!IMPORTANT]
> **Data Policy:** No public datasets are downloaded or placed in `data/` at this stage. This document serves as a comparative assessment for future Phase 2 supplemental pre-training decisions.

---

## 2. Comparative Evaluation Matrix

| Dataset | Primary Source | License | Viewpoint | Class Compatibility | Suitability Rating |
|---|---|---|---|---|---|
| **RUGD** | US Army Research Lab (2019) | Academic / Non-Commercial | Ground Robot (Low) | High (24 classes mapped to 6) | **Strong Candidate (Supplemental)** |
| **RELLIS-3D** | Texas A&M (2021) | MIT / Open Research | UGV (Medium-Low) | High (20 classes mapped to 6) | **Strong Candidate (Supplemental)** |
| **WildScenes** | CSIRO Data61 (2024) | CC BY-NC-SA 4.0 | UGV / Backpack (Mixed) | High (Natural bushland) | **Moderate Candidate (Transfer Benchmark)** |
| **Freiburg Forest** | Univ. of Freiburg (2016) | CC BY-NC 4.0 | Robot (Low) | High (6 simple classes) | **Moderate Candidate (Quick Baseline)** |
| **Trail-Dataset** | Various Trail Studies | Academic / Non-Commercial | Pedestrian (Handheld) | Low (Binary Path / Non-Path) | **Weak (Lacks Hazard Detail)** |

---

## 3. Detailed Dataset Assessments

### 3.1 RUGD (Robot Unstructured Ground Driving)
* **Source:** Wigness et al., IEEE/RSJ IROS 2019.
* **License:** Free for non-commercial research/academic use.
* **Scene Characteristics:** Natural hiking trails, parks, creeks, rocky paths, gravel roads, and woods.
* **Available Labels:** 24 dense semantic classes including `dirt`, `sand`, `grass`, `tree`, `rock`, `water`, `mud`, `asphalt`, `rubble`, `void`.
* **Mapping to StepView:**
  * `dirt`, `sand`, `gravel`, `concrete` → **`ground` (1)**
  * `rock`, `rock-bed`, `log`, `tree-trunk`, `pole` → **`obstacle` (2)**
  * `grass`, `bush`, `tree-leaves` → **`vegetation` (3)**
  * (No explicit hole class; steep drop-offs labeled as void or background) → **`background` (0)** or **`hole` (4)**
  * `water`, `mud`, `creek` → **`uncertain_surface` (5)**
  * `sky`, `building`, `person`, `void` → **`background` (0)**
* **Advantages:** Extensive diversity of real unstructured trails; dense, high-quality pixel annotations.
* **Limitations:** Robot platform camera sits lower (~0.6m) with near-horizontal pitch; lacks specific human footstep annotations.
* **Recommendation:** **Candidate for Phase 2 pre-training.** Requires class remapping script.

---

### 3.2 RELLIS-3D
* **Source:** Jiang et al., IEEE RA-L 2021.
* **License:** Open Research / MIT License.
* **Scene Characteristics:** Highly challenging off-road trails with puddles, mud, dense grass, rubble, and woods.
* **Available Labels:** 20 classes (e.g. `dirt`, `grass`, `tree`, `bush`, `mud`, `puddle`, `rubble`, `barrier`, `sky`).
* **Mapping to StepView:**
  * `dirt` → **`ground` (1)**
  * `rubble`, `barrier`, `tree` → **`obstacle` (2)**
  * `grass`, `bush` → **`vegetation` (3)**
  * `mud`, `puddle`, `water` → **`uncertain_surface` (5)**
  * `sky`, `object` → **`background` (0)**
* **Advantages:** Open permissive licensing; excellent representation of deceptive surfaces (`mud`, `puddle`).
* **Limitations:** High frame redundancy; captured from an aggressive off-road vehicle platform.
* **Recommendation:** **Candidate for Phase 2 pre-training (hard-negative enrichment).**

---

### 3.3 WildScenes
* **Source:** Vidas et al., CVPR 2024 / CSIRO Data61.
* **License:** CC BY-NC-SA 4.0.
* **Scene Characteristics:** Unstructured Australian native forests and bush tracks, heavy leaf litter, uneven boulders, fallen logs.
* **Available Labels:** Dense multi-modal 2D & 3D semantic annotations.
* **Mapping to StepView:**
  * `ground-surface`, `trail` → **`ground` (1)**
  * `rock`, `boulder`, `trunk` → **`obstacle` (2)**
  * `foliage`, `shrub` → **`vegetation` (3)**
  * `water` → **`uncertain_surface` (5)**
* **Advantages:** High fidelity, modern sensor resolutions, realistic wilderness paths.
* **Limitations:** Non-commercial restriction (NC clause); large download footprint (> 100 GB).
* **Recommendation:** **Retain as potential validation benchmark**, do not ingest into core repository data yet.

---

### 3.4 Freiburg Forest
* **Source:** Valada et al., 2016.
* **License:** CC BY-NC 4.0.
* **Scene Characteristics:** European wooded trails, gravel roads, grass shoulders.
* **Available Labels:** 6 classes: `Obstacle`, `Trail`, `Grass`, `Tree`, `Sky`, `Void`.
* **Mapping to StepView:**
  * `Trail` → **`ground` (1)**
  * `Obstacle` → **`obstacle` (2)**
  * `Grass`, `Tree` → **`vegetation` (3)**
  * `Sky`, `Void` → **`background` (0)**
* **Advantages:** Direct 1-to-1 conceptual alignment with StepView's core baseline.
* **Limitations:** Small sample size (228 annotated images); limited variety in trail types.
* **Recommendation:** **Useful for quick pipeline unit test benchmarking**, but insufficient alone for model robustness.

---

## 4. Required Class Remapping Specifications

Before any public dataset can be ingested as supplemental training data in Phase 2, a deterministic mapping script (`scripts/remap_public_dataset.py`) must be executed to convert third-party mask IDs to StepView's canonical 8-bit values:

```python
# Example: RUGD to StepView Canonical Class Map
RUGD_TO_STEPVIEW = {
    0: TerrainClass.BACKGROUND,       # void
    1: TerrainClass.GROUND,           # dirt
    2: TerrainClass.UNCERTAIN_SURFACE,# sand
    3: TerrainClass.VEGETATION,       # grass
    4: TerrainClass.VEGETATION,       # tree
    5: TerrainClass.OBSTACLE,         # pole
    6: TerrainClass.UNCERTAIN_SURFACE,# water
    7: TerrainClass.BACKGROUND,       # sky
    8: TerrainClass.GROUND,           # vehicle (ignored/bg)
    9: TerrainClass.BACKGROUND,       # container/building
    10: TerrainClass.GROUND,          # asphalt
    11: TerrainClass.GROUND,          # gravel
    12: TerrainClass.BACKGROUND,      # building
    13: TerrainClass.UNCERTAIN_SURFACE,# mulch
    14: TerrainClass.OBSTACLE,        # rock-bed
    15: TerrainClass.OBSTACLE,        # rock
    16: TerrainClass.OBSTACLE,        # log
    17: TerrainClass.BACKGROUND,      # bicycle
    18: TerrainClass.BACKGROUND,      # person
    19: TerrainClass.BACKGROUND,      # fence
    20: TerrainClass.VEGETATION,      # bush
    21: TerrainClass.BACKGROUND,      # sign
    22: TerrainClass.OBSTACLE,        # rock
    23: TerrainClass.BACKGROUND,      # bridge
}
```

---

## 5. Conclusion & Next Steps

* **Immediate Decision:** Do not download public datasets during the current Bootstrap phase.
* **Phase 2 Ingestion Target:** If self-collected data collection is limited, RUGD and RELLIS-3D are the highest-priority candidates for pre-training, provided they are converted to canonical StepView 8-bit masks and split strictly by scene ID.
