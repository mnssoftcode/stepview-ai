# StepView — Documentation Audit & Specification Review

**Date:** 2026-09-24  
**Author:** Lead AI/ML Engineer (Technical Project Agent)  
**Status:** Completed & Approved with Prototype Assumptions Clarification  
**Reference Version:** Documentation v0.1 (`docs/00` to `docs/07`)

---

## 1. Executive Summary

A comprehensive technical audit was conducted across all existing documentation files in `docs/` (`00_PROJECT_MASTER.md` through `07_DOCUMENT_CREATION_PROMPT.md`) against current workspace realities and machine learning engineering standards.

While the conceptual vision of StepView is clear, focused, and appropriately safety-conscious, several technical contradictions, underspecified requirements, missing evaluation thresholds, and architectural assumptions require resolution before proceeding into training and model export.

---

## 2. Initial Prototype Assumptions & Empirical Validation Policy

> [!IMPORTANT]
> The following parameters are **INITIAL PROTOTYPE ASSUMPTIONS**, not established ground truths:
> - **Human foot bounding area:** Approximately 25–30 cm × 10 cm.
> - **Surface slope limit:** Apparent visual slope < 25°.
> - **Obstacle clearance threshold:** Vertical obstacle/protrusion height clearance around 3 cm.
> - **Camera capture pitch angle:** Handheld downward pitch around 35°–50° relative to horizontal.
> - **Forward lookahead distance:** Forward walking zone around 1–3 meters.
>
> **Mandatory Rule:** These values serve as initial engineering baselines for early software design and heuristic scaffolding. They **must be validated, calibrated, and revised** through empirical data collection, sensor measurements, user experiments, and real-world field testing.

> [!NOTE]
> **Synthetic Dataset Policy:**
> Synthetic dataset tests (mock masks, procedural geometric images) validate **software engineering correctness only** (e.g. data loader robustness, pipeline I/O, tensor conversions, boundary checks). They do **NOT** validate machine learning performance, generalization, or real-world safety.

---

## 3. Detailed Audit Findings

### Category A: Contradictions

#### Issue A1: Product and Project Naming Inconsistency
* **Location:** All docs (`00_PROJECT_MASTER.md`, `01_PRD.md`, `02_ML_SPEC.md`, `03_DATASET_AND_ANNOTATION.md`, `04_SYSTEM_ARCHITECTURE.md`, `05_DEVELOPMENT_ROADMAP.md`, `06_AI_AGENT_RULES.md`, `07_DOCUMENT_CREATION_PROMPT.md`)
* **Description:** Documents consistently refer to the project as "StepView AI" or "StepView-ai", whereas the official user-designated name is now **StepView**.
* **Why it matters:** Inconsistent naming causes confusion in package names, CLI commands, repository structures, configuration files, and end-user documentation.
* **Recommended resolution:** Standardize on **StepView** across all new documentation, code packages, comments, and application titles.
* **Status:** Non-blocking.

#### Issue A2: Pipeline Ordering Discrepancy (Depth vs. Candidate Generation)
* **Location:** `docs/00_PROJECT_MASTER.md` Section 5 vs. `docs/05_DEVELOPMENT_ROADMAP.md` Phases 3 & 4
* **Description:** `00_PROJECT_MASTER.md` places "Depth / geometry estimation" *before* "Candidate foot-placement generation" in the master processing pipeline. However, `05_DEVELOPMENT_ROADMAP.md` introduces 2D candidate placement in Phase 3, and only introduces depth/geometry in Phase 4.
* **Why it matters:** Causes ambiguity for ML engineers about whether candidate placement depends strictly on a depth network or whether a 2D segmentation + morphological/heuristic candidate generator serves as the Phase 3 baseline.
* **Recommended resolution:** Explicitly clarify that Phase 3 implements candidate generation on 2D segmentation masks (using connected components, morphology, and perspective-scaled heuristics). Phase 4 adds monocular depth/geometry as an optional or supportive feature enhancement.
* **Status:** Non-blocking (clarified in roadmap).

#### Issue A3: Proposed Repository Structure vs. Physical Directory State
* **Location:** `docs/04_SYSTEM_ARCHITECTURE.md` Section "Recommended repository"
* **Description:** The document recommends a root called `StepView-ai/` with top-level `src/`, `data/`, `notebooks/`, `models/`, `web/`, `mobile/`. The actual workspace root is `/Users/mns/Mns_Data/code/AiMl/StepViewProject/` with an empty subfolder `stepview/` and `docs/`.
* **Why it matters:** Ambiguity in Python module paths, imports, and relative paths (e.g. `import stepview` vs `import src`).
* **Recommended resolution:** Adopt standard clean Python package architecture within the existing workspace:
  - Root: `StepViewProject/`
  - Python package: `stepview/` (contains `data/`, `models/`, `geometry/`, `postprocess/`, `inference/`, `utils/`)
  - Subprojects/artifacts: `data/`, `experiments/`, `models/`, `web/`, `mobile/`, `tests/`, `scripts/`, `docs/`.
* **Status:** Non-blocking.

---

### Category B: Missing Requirements

#### Issue B1: Underspecified Camera Geometry, Mounting, and Field of View
* **Location:** `docs/01_PRD.md`, `docs/03_DATASET_AND_ANNOTATION.md`
* **Description:** The PRD states "Camera points toward the walking surface," but does not specify camera height (e.g. waist vs. chest vs. handheld), pitch/tilt angle relative to horizontal ground (e.g. 30° to 60° down), or targeted forward lookahead distance (e.g. 0.8m to 3.0m).
* **Why it matters:** Monocular camera images without orientation constraints suffer from severe scale variance. Foot placement 1 meter ahead occupies orders of magnitude more pixels than terrain 4 meters ahead. Data collection and annotation without a defined capture protocol will yield uncalibrated, noisy models.
* **Recommended resolution:** Specify the baseline capture protocol: Handheld smartphone at waist/chest level, pitched downward at an assumed 35°–50°, focusing on terrain around 1 to 3 meters in front of the walker. These assumptions must be validated through data collection and field trials.
* **Status:** Blocking for dataset collection (Phase 1).

#### Issue B2: Physical Metric Scale vs. Pixel Footprint Definition
* **Location:** `docs/02_ML_SPEC.md` Section "Candidate generation"
* **Description:** "Candidate regions should be sufficiently large... geometrically plausible for a foot." No translation between 2D pixel area and real-world foot footprint (~25–30 cm length, ~10 cm width) is defined.
* **Why it matters:** An algorithm filtering regions by fixed pixel area will reject valid distant steps or accept tiny foreground pebbles.
* **Recommended resolution:** In 2D, define a row-dependent minimum pixel area function (perspective scaling heuristic where threshold increases towards bottom rows), or use homography / perspective transformation assuming typical camera pitch angle.
* **Status:** Non-blocking for initial segmentation; blocking for candidate placement (Phase 3).

#### Issue B3: Lack of Numerical Real-Time Performance & Latency Targets
* **Location:** `docs/00_PROJECT_MASTER.md` Section 7, `docs/01_PRD.md` Section "Acceptance criteria"
* **Description:** The documents mention "practical frame rate" and "acceptable latency" without numeric Service Level Objectives (SLOs).
* **Why it matters:** Without an FPS/latency budget, model selection cannot be objectively judged (e.g. MobileNetV4-Small vs ConvNeXt vs SegFormer).
* **Recommended resolution:** Establish concrete targets:
  - Web browser prototype: >= 10 FPS, <= 100 ms inference latency on desktop/laptop Chrome; >= 5 FPS on mobile web.
  - Mobile on-device target (Phase 6): >= 15 FPS, <= 65 ms total frame processing latency on mid-range Android / iOS.
* **Status:** Non-blocking for Phase 1; must be locked before model architecture selection in Phase 2.

#### Issue B4: Missing Annotation Format Specification
* **Location:** `docs/03_DATASET_AND_ANNOTATION.md`
* **Description:** Annotation classes are outlined (ground, rock, vegetation, hole, water/mud), but the serialization format (COCO format, PNG indexed segmentation masks, YOLO-seg, or CVAT polygon XML/JSON) is not specified.
* **Why it matters:** Blocks dataset pipeline implementation and data loader construction.
* **Recommended resolution:** Adopt standard PNG indexed 8-bit masks (single-channel uint8 where pixel value = class ID 0..5) accompanied by a standardized metadata JSON index pointing to image-mask pairs and scene IDs. This format is lightweight, standard across PyTorch/torchvision, and avoids complex polygon parsing.
* **Status:** Blocking for Phase 1 dataset loader.

---

### Category C: Unclear Terminology & Taxonomies

#### Issue C1: Operational Ambiguity of "Visual Terrain Suitability"
* **Location:** `docs/00_PROJECT_MASTER.md`, `docs/01_PRD.md`, `docs/02_ML_SPEC.md`
* **Description:** "Visually suitable" is subjective. Annotators and validation testers will disagree unless operational criteria are formalized.
* **Why it matters:** Inconsistent ground-truth annotations degrade model convergence and inflate validation variance.
* **Recommended resolution:** Define initial operational criteria for "suitable foot placement":
  1. Continuous ground surface approximately >= 25 cm × 10 cm.
  2. Free of sharp rocks (> 3 cm height), roots, loose debris, deep cracks, or holes.
  3. Apparent visual slope < 25 degrees.
  4. Non-slippery appearance (free of standing water, thick mud, wet algae, loose scree).
  *Note:* All geometric parameters (25×10 cm, 3 cm obstacle clearance, 25° slope) are initial prototype assumptions to be calibrated by experiment.
* **Status:** Blocking for Phase 1 annotation.

#### Issue C2: Distinction Between "Ground Mask" and "Placement Candidate"
* **Location:** `docs/02_ML_SPEC.md`
* **Description:** The spec uses "candidate ground" as segmentation Class 1, and "candidate foot-placement" as Model C output.
* **Why it matters:** Risk of conflating pixel-level semantic terrain classification with instance-level footstep landing sites.
* **Recommended resolution:** Use distinct terms:
  - **Walkable Terrain Mask (Semantic Segmentation):** Continuous pixel mask of ground vs. obstacle/vegetation/hole.
  - **Footstep Candidate Region (Spatial Proposal):** Discrete, ranked polygon/ellipse representing an individual foot-sized landing zone extracted from the walkable mask.
* **Status:** Non-blocking.

---

### Category D: Unrealistic Assumptions & Architecture Feasibility

#### Issue D1: Dual Heavy Neural Networks in Mobile Browser
* **Location:** `docs/02_ML_SPEC.md` (Model A + Model B), `docs/04_SYSTEM_ARCHITECTURE.md`
* **Description:** Running a full semantic segmentation network *plus* a separate monocular depth estimation network concurrently inside a browser via ONNX Runtime Web on mobile hardware will cause high latency, memory pressure, and thermal throttling.
* **Why it matters:** Mobile browser WebAssembly/WebGL environments have strict memory bounds (often < 1GB) and limited GPU compute compared to native.
* **Recommended resolution:**
  - Phase 2–3: Focus on a lightweight, efficient 2D segmentation backbone (e.g. MobileNetV3/V4, Fast-SCNN, or lightweight SegFormer/LR-ASPP) with geometric heuristics.
  - Phase 4: If depth is required, investigate either a single multi-task head (sharing the encoder for segmentation + depth) or a very compact depth model (e.g. FastDepth / lightweight MiDaS).
* **Status:** Non-blocking for early phases; critical architecture constraint for Phase 4+.

#### Issue D2: Workspace Version Control Absence
* **Location:** Repository environment
* **Description:** The project directory is currently not initialized as a Git repository (`fatal: not a git repository`).
* **Why it matters:** Violates Rule 6 of `docs/06_AI_AGENT_RULES.md` ("Preserve reproducibility") and prevents version tracking, branching, and commit auditing.
* **Recommended resolution:** Initialize a git repository with a sensible `.gitignore` immediately.
* **Status:** Blocking for code development.

---

### Category E: Missing Evaluation Criteria & Milestone Targets

#### Issue E1: Evaluation Metrics & Milestone Targets Clarification
* **Location:** `docs/05_DEVELOPMENT_ROADMAP.md` Phase 2 & Phase 3 exit conditions
* **Description:** Milestone targets require explicit operational definitions to prevent conflating engineering metrics with safety proof.
* **Why it matters:** High segmentation overlap does not automatically guarantee safety; metrics must be bounded to engineering benchmarks on held-out terrain.
* **Recommended resolution:** Define concrete validation gates:
  - **Phase 2 Engineering Target:** Mean Intersection-over-Union (mIoU) on Walkable Ground >= 0.70 on held-out scenes. *Note: This is an initial engineering target for segmentation quality only, NOT proof that the system can identify safe foot placements.*
  - **Phase 3 Validation Metric:** False Recommendation Rate <= 10% measured strictly on a held-out real-terrain test set (frequency with which the top-ranked candidate overlaps an annotator-marked hazard or unsuitable region).
* **Status:** Non-blocking for Phase 1 setup; blocking for Phase 2/3 sign-off.

---

### Category F: Scope & Engineering Discipline

#### Issue F1: Premature Features (Video Mode & Temporal Smoothing)
* **Location:** `docs/01_PRD.md` Section "V1 modes", `docs/02_ML_SPEC.md`
* **Description:** Video processing with temporal smoothing is listed under V1.
* **Why it matters:** Multi-frame temporal state management introduces unnecessary engineering overhead before single-image spatial prediction is sound.
* **Recommended resolution:** Constrain the immediate early milestone strictly to static image inference (Photo mode / offline frame evaluation). Video and temporal filtering should only be added after single-frame prediction meets acceptance criteria.
* **Status:** Non-blocking.

---

## 4. Summary of Audit Action Items

| ID | Issue | Severity | Action |
|---|---|---|---|
| A1 | Naming inconsistency ("StepView AI" vs "StepView") | Non-blocking | Standardize name to `StepView` in all new artifacts and metadata. |
| A2 | Pipeline ordering (Depth vs Candidate generation) | Non-blocking | Clarify Phase 3 as 2D candidate generation; Phase 4 as optional geometry enhancement. |
| A3 | Directory structure mismatch | Non-blocking | Establish `stepview/` as primary package in workspace root. |
| B1 | Underspecified camera orientation | **Blocking (P1)** | Adopt initial prototype assumption: 35°–50° pitch, 1–3m lookahead (to be empirically validated). |
| B2 | Metric scale definition in 2D | Non-blocking | Adopt initial prototype assumption: ~25–30×10 cm foot area with row perspective scaling. |
| B3 | Missing numerical FPS/latency targets | Non-blocking | Set Web target >= 10 FPS, Mobile native target >= 15 FPS. |
| B4 | Missing annotation data format | **Blocking (P1)** | Adopt single-channel 8-bit PNG masks + index JSON with scene IDs. |
| C1 | Subjective "suitability" definition | **Blocking (P1)** | Define operational criteria (< 25° slope, < 3cm clearance); mark as initial prototype assumptions. |
| C2 | Ground mask vs Placement candidate | Non-blocking | Formalize semantic terrain mask vs discrete footstep candidate. |
| D1 | Dual heavy networks in browser | Non-blocking | Prioritize single lightweight backbone before dual-model evaluation. |
| D2 | Missing Git repository | **Blocking (Dev)** | Initialize git repository and `.gitignore`. |
| E1 | Milestone metrics & engineering targets | Non-blocking | `mIoU >= 0.70` is Phase 2 engineering target only; `False Rec <= 10%` measured on held-out real data. |
| F1 | Premature video mode | Non-blocking | Scope V1 Alpha strictly to single-image evaluation first. |
