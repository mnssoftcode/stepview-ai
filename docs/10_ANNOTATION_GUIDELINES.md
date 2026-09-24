# StepView — Terrain Annotation Guidelines

**Version:** 1.0  
**Date:** 2026-09-24  
**Target:** Semantic Pixel Segmentation & Foot-Placement Candidate Identification

---

## 1. Objective and Core Principles

The objective of StepView annotation is to produce accurate, consistent, and reproducible ground-truth labels for natural and semi-structured outdoor walking surfaces.

### Foundational Annotation Rules:
1. **Annotate Strictly What Is Visually Evident:** Do not assume a rock is firm or that a pile of leaves covers solid soil. If visual stability cannot be determined, classify as uncertain.
2. **Never Label Physical Safety:** StepView labels reflect *visual suitability* based on surface geometry and texture, NOT certified geotechnical safety.
3. **No Overlapping Mask Pixels:** Every pixel in the image must belong to exactly one integer class ID (0 through 5).

> [!IMPORTANT]
> **Prototype Heuristic Notice:**  
> Numerical thresholds in these guidelines (e.g. foot dimension ~25–30 cm × 10 cm, slope < 25°, obstacle clearance ~3 cm) are **INITIAL PROTOTYPE HEURISTICS**, not scientifically proven physical invariants. They serve to standardize initial annotator consistency and will be calibrated through empirical data collection and gait biomechanics literature.

---

## 2. Semantic Class Taxonomy & Definitions

| Class ID | Class Name | Mask Value | Palette (RGB) | Description |
|---|---|---|---|---|
| **0** | `background` | `0` | `(0, 0, 0)` | Sky, horizon, trees above head height, buildings, people, non-ground scene context. |
| **1** | `ground` | `1` | `(46, 204, 113)` | Visually contiguous, walkable trail surfaces, soil, flat stone, packed gravel, stairs. |
| **2** | `obstacle` | `2` | `(231, 76, 60)` | Protruding rocks, boulders, exposed tree roots, logs, vertical curbs, large debris. |
| **3** | `vegetation` | `3` | `(39, 174, 96)` | Dense shrubs, thick high grass, underbrush, leaves covering underlying ground. |
| **4** | `hole` | `4` | `(142, 68, 173)` | Drop-offs, trenches, erosion gullies, cavities, gaps between boulders, cliff edges. |
| **5** | `uncertain_surface` | `5` | `(241, 196, 15)` | Standing water, deep mud, ice, wet algae on rock, loose sliding scree, dense leaf litter. |

---

## 3. Detailed Class Labeling Criteria

### Class 0: Background (`0`)
* **Includes:** Anything not part of the immediately navigable terrain in front of the user:
  * Sky, clouds, distant mountains.
  * Trees, trunks, and foliage elevated above the walking surface.
  * People, pets, backpacks, gear visible in frame.
  * Walls, distant structures, fences.
* **Boundary rule:** The boundary between the walking surface horizon and the background must be drawn tightly along the top contour of the walkable terrain.

### Class 1: Ground (`1`)
* **Includes:** Surfaces visually suitable for supporting a step:
  * Firm, compacted soil, packed dirt pathways.
  * Solid, embedded, relatively flat rock slabs flush with the trail.
  * Compact gravel or crushed stone path.
  * Well-maintained stairs or paved steps.
* **Excludes:** Loose scree that visibly slides, sharp protruding rocks, or wet muddy patches.

### Class 2: Obstacle (`2`)
* **Includes:** Rigid physical protrusions that represent trip hazards or block foot placement:
  * Rocks or boulders protruding more than approximately 3 cm above the surrounding ground level (prototype heuristic).
  * Exposed tree roots running across the path.
  * Fallen logs, branches, metal spikes, pipes, or debris.
* **Boundary rule:** Include the full visible contour of the rock/root up to the ground contact line. If a root has soil underneath but arches upward, label the root as obstacle.

### Class 3: Vegetation (`3`)
* **Includes:** Organic plant matter that obscures the actual ground surface:
  * Thick or tall grass (> 5 cm high) where the soil beneath cannot be seen.
  * Bushes, brambles, ferns, low tree branches encroaching on the path.
* **Distinction:** Short, sparse turf where the solid ground plane is clearly continuous may be labeled `ground`; dense cover must be labeled `vegetation`.

### Class 4: Hole / Drop-off (`4`)
* **Includes:** Sudden downward discontinuities in the terrain:
  * Trail edges with sharp drop-offs (> 15 cm step-down).
  * Erosion gullies, deep washouts, sinkholes.
  * Crevices or spaces between rocks where a foot could become wedged.
* **Visual indicator:** Often accompanied by sharp occlusion edges and deep shadows.

### Class 5: Uncertain Surface (`5`)
* **Includes:** Surfaces whose footing stability is questionable from visual appearance alone:
  * Standing puddles, stream crossings, submerged rocks.
  * Thick, churned, wet mud or boggy peat.
  * Ice, frost, or snow patches.
  * Rocks with wet green algae or moss sheen.
  * Steep accumulation of loose scree or pea gravel on an incline that appears prone to sliding.
  * Deep leaf litter where the presence of hidden rocks or holes cannot be ruled out.

---

## 4. Specific Edge Cases & Decision Rules

### 4.1 Shadows
* **Rule:** Shadows do NOT change the underlying physical class.
* **Example:** If a tree shadow falls across a flat dirt trail, the shaded dirt remains `ground` (Class 1).
* **Exception:** If a shadow is so dark (clipped black pixels) that surface texture cannot be discerned, and the annotator cannot differentiate flat ground from a drop-off or hole, label the indistinguishable region as `uncertain_surface` (Class 5).

### 4.2 Rocks: Embedded vs. Protruding
* **Embedded flat rocks:** Rocks flush with the trail surface that offer a level, non-wobbly footing surface >= 25 cm across are labeled `ground` (Class 1).
* **Protruding rocks:** Any rock protruding > 3 cm above ground or with sharp angled facets is labeled `obstacle` (Class 2).
* **Loose gravel:** Fine, packed gravel is `ground` (Class 1). Loose, angled stones on a slope that look prone to roll underfoot are `uncertain_surface` (Class 5).

### 4.3 Tree Roots
* **Rule:** Exposed roots are `obstacle` (Class 2).
* If a network of roots forms a "pocket" of flat soil between them, the soil pocket may be labeled `ground` if it is large enough for a foot (~25×10 cm); otherwise, label the entire cluttered zone as `obstacle`.

### 4.4 Partially Occluded Ground
* If grass or leaves partially obscure ground:
  * If > 70% of the ground is covered: label `vegetation` (Class 3) or `uncertain_surface` (Class 5).
  * If sparse cover (< 30%) with clear soil continuity: label `ground` (Class 1).

### 4.5 Steep Slopes
* Terrain with apparent slope > 25° relative to the walking angle where an ordinary step would slide should be labeled `uncertain_surface` (Class 5) or `obstacle` (Class 2) rather than standard `ground`.

### 4.6 Visually Deceptive Surfaces
* Dry mud with visible surface cracking: `ground` (Class 1) if baked solid, but `uncertain_surface` (Class 5) if soft/deformable.
* Wet glistening rocks: `uncertain_surface` (Class 5).

---

## 5. Foot-Placement Candidate Heuristics (Level 2 Annotation)

When annotating Level 2 candidate landing zones (bounding ellipses or polygons):
1. **Target Area:** Must encompass an area roughly corresponding to a standard adult footwear footprint (~25–30 cm length × ~10 cm width) in perspective.
2. **Clearance:** Must have at least ~3 cm clearance from nearest `obstacle`, `hole`, or `uncertain_surface`.
3. **Flatness:** Must lie entirely within a contiguous, uniform `ground` region.
4. **Ranking Preference:**
   - **Recommended:** Dry, flat, broad ground patch in direct line of sight.
   - **Caution/Uncertain:** Narrow ground patch close to roots or rocks.
   - **Reject:** Insufficient area, excessive slope, or proximity to hazard.
