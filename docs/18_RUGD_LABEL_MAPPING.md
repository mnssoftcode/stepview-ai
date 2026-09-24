# StepView — RUGD Ontology to StepView Label Mapping Specification

**Version:** 1.0  
**Date:** 2026-09-24  
**Applies to:** RUGD Dataset Conversion for StepView Pre-training & Bootstrap Validation

---

## 1. Critical Annotation Limitation Notice

> [!CAUTION]
> **Fundamental Domain Difference:**  
> RUGD annotations were designed for **off-road autonomous robot navigation (unmanned ground vehicles)**. They categorize surfaces for vehicle tractability, clearance, and obstacle avoidance.
> 
> **THEY ARE NOT FOOT-PLACEMENT ANNOTATIONS.**
> 
> * RUGD masks can be used strictly to validate the **computer-vision terrain-segmentation pipeline** (separating sky, trees, dirt paths, rocks, and water).
> * They must **NEVER** be treated as ground truth for human footstep suitability.
> * Converted RUGD masks must **NEVER** be used to claim that StepView can identify safe human foot placement.

---

## 2. RUGD to StepView Canonical Class Mapping

The official RUGD ontology contains 25 semantic categories (IDs 0–24) encoded as 24-bit RGB values in annotation PNGs. StepView condenses semantic terrain into 6 canonical integer classes:

* `0: BACKGROUND`
* `1: GROUND`
* `2: OBSTACLE`
* `3: VEGETATION`
* `4: HOLE`
* `5: UNCERTAIN_SURFACE`

### Detailed Mapping Table

| RUGD ID | RUGD Class Name | RUGD RGB Value | StepView Class ID | StepView Class Name | Confidence | Rationale & Nuance |
|---|---|---|---|---|---|---|
| **0** | `void` | `(0, 0, 0)` | **0** | `BACKGROUND` | High | Unannotated boundary, ego-vehicle hood, or invalid pixels; must not be proposed as walkable ground. |
| **1** | `dirt` | `(108, 64, 20)` | **1** | `GROUND` | High | Standard packed-soil or dirt trail surface suitable for walking. |
| **2** | `sand` | `(255, 229, 204)` | **5** | `UNCERTAIN_SURFACE` | High | Loose sand yields under foot pressure, easily causing slippage or ankle rolling, especially on slopes. |
| **3** | `grass` | `(0, 102, 0)` | **3** | `VEGETATION` | High | Grass cover obscures underlying ground flatness, concealed roots, and small holes. |
| **4** | `tree` | `(0, 255, 0)` | **3** | `VEGETATION` | High | Foliage and tree branches. Trunks are rigid obstacles, but RUGD groups trunks and canopy together; mapping to vegetation is visually consistent. |
| **5** | `pole` | `(0, 153, 153)` | **2** | `OBSTACLE` | High | Vertical rigid post obstructing movement. |
| **6** | `water` | `(0, 128, 255)` | **5** | `UNCERTAIN_SURFACE` | High | Creek, river, or puddle; unknown depth, hidden slippery stones, and slip hazard. |
| **7** | `sky` | `(0, 0, 255)` | **0** | `BACKGROUND` | High | Atmospheric background. |
| **8** | `vehicle` | `(255, 255, 0)` | **0** | `BACKGROUND` | High | Robot chassis or external vehicles; non-terrain context. |
| **9** | `container/generic-object` | `(255, 0, 127)` | **2** | `OBSTACLE` | High | Man-made debris, boxes, or bins blocking foot landing. |
| **10** | `asphalt` | `(64, 64, 64)` | **1** | `GROUND` | High | Firm, paved, load-bearing walking surface. |
| **11** | `gravel` | `(255, 128, 0)` | **1** | `GROUND` | Medium | Packed gravel trail surface. Note: loose steep gravel may slip, but typical trail gravel functions as walkable ground. |
| **12** | `building` | `(255, 0, 0)` | **0** | `BACKGROUND` | High | Architectural structure. |
| **13** | `mulch` | `(153, 76, 0)` | **5** | `UNCERTAIN_SURFACE` | High | Deformable loose organic bed with unknown load support. |
| **14** | `rock-bed` | `(102, 102, 0)` | **5** | `UNCERTAIN_SURFACE` | Medium | Riverbed rocks and jagged boulder fields. For a rover this is rough terrain; for a pedestrian it is highly irregular and slippery, warranting caution. |
| **15** | `log` | `(102, 0, 0)` | **2** | `OBSTACLE` | High | Fallen tree trunk or log across path; trip hazard. |
| **16** | `bicycle` | `(0, 255, 128)` | **0** | `BACKGROUND` | High | Non-terrain object. |
| **17** | `person` | `(204, 153, 255)` | **0** | `BACKGROUND` | High | Pedestrian in scene. |
| **18** | `fence` | `(102, 0, 204)` | **2** | `OBSTACLE` | High | Vertical barrier. |
| **19** | `bush` | `(255, 153, 204)` | **3** | `VEGETATION` | High | Dense shrubs and thickets obscuring ground. |
| **20** | `sign` | `(0, 102, 102)` | **2** | `OBSTACLE` | High | Trail marker or signage post. |
| **21** | `rock` | `(153, 204, 255)` | **2** | `OBSTACLE` | High | Discrete protruding boulder or rock on or beside path. |
| **22** | `bridge` | `(102, 255, 255)` | **1** | `GROUND` | High | Wooden footbridge or paved pedestrian bridge surface. |
| **23** | `concrete` | `(101, 101, 11)` | **1** | `GROUND` | High | Solid sidewalk or concrete slab. |
| **24** | `picnic-table` | `(114, 85, 47)` | **2** | `OBSTACLE` | High | Man-made outdoor furniture. |

---

## 3. Unmapped Colors & Fallback Policy

* **Strict Exact Color Match:** Any pixel whose RGB value does not exactly match one of the 25 official colors above is evaluated against nearest Euclidean distance in RGB space. If distance > 5.0 (tolerance for compression artifacts), it is assigned to `0: BACKGROUND` and logged for review.
* **No `HOLE` class in RUGD:** The RUGD dataset lacks an explicit drop-off/hole annotation class (steep edges were labeled as `void` or `dirt`). Therefore, StepView Class 4 (`HOLE`) will have 0% representation in the RUGD bootstrap subset. This gap must be filled by self-collected data or targeted hard-negative collections.
