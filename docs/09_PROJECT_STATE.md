# StepView — Current Project State

**Last Updated:** 2026-09-24  
**Project Name:** StepView  
**Author:** Lead AI/ML Engineer (Technical Project Agent)

---

## 1. Project Overview & Current Phase

* **Current Phase:** **Phase 1 — Dataset Proof & Pipeline Verification** (Exit Condition Met; Ready to Transition to **Phase 2 — Segmentation Baseline**).
* **Official Project Name:** **StepView**.
* **State Assessment:** Complete end-to-end data pipeline established. Real terrain bootstrap data (26 frames across 18 scenes) converted from RUGD, validated with strict class schemas, partitioned into leakage-free scene splits, inspected with visual overlays, and test-covered across 15 automated pytest cases.

---

## 2. Completed Work

* [x] **Documentation & Strategy Foundations:**
  * `docs/00_PROJECT_MASTER.md` to `docs/07_DOCUMENT_CREATION_PROMPT.md`: Architecture, PRD, ML spec.
  * `docs/08_DOCUMENTATION_AUDIT.md`: Audit with prototype assumptions, empirical validation rules, and testing boundaries.
  * `docs/10_ANNOTATION_GUIDELINES.md`: Comprehensive labeling rules for 6 classes, edge cases, and candidate footstep heuristics.
  * `docs/11_DATASET_MANIFEST.md`: Data directory contract (`images/`, `masks/`, `metadata/`, `splits/`) and JSON manifest schema.
  * `docs/12_BOOTSTRAP_DATA_WORKFLOW.md`: 10-step capture-to-commit workflow, tooling recommendation, and bootstrap boundaries.
  * `docs/13_PUBLIC_DATASET_EVALUATION.md`: Comparative evaluation of RUGD, RELLIS-3D, WildScenes, Freiburg Forest.
  * `docs/18_RUGD_LABEL_MAPPING.md`: Detailed 25-to-6 class conversion ontology and non-safety limitations notice.
  * `docs/19_RUGD_BOOTSTRAP_REPORT.md`: Comprehensive audit report on the 26 converted RUGD samples, scene distributions, and visual panels.
* [x] **Version Control & Repository Setup:**
  * Git repository initialized on `main` branch with clean conventional commit history.
  * Comprehensive `.gitignore` protecting binary assets.
* [x] **Reproducible Python 3.12 Environment (`uv`):**
  * `pyproject.toml` with pinned dependencies (`torch==2.2.2`, `torchvision==0.17.2`, `opencv-python-headless==4.11.0.86`, `numpy==1.26.4`, `pillow==12.3.0`, `pytest==9.1.1`).
  * `.venv` created via `uv` using system Python 3.12.13.
* [x] **Package Structure & Loader (`stepview/`):**
  * `stepview/data/schema.py`: Explicit `TerrainClass` enum (0..5), color palette, `SampleMetadata`, `validate_mask_classes()`.
  * `stepview/data/dataset.py`: `StepViewDataset` and `stepview_collate_fn` separating image loading, mask loading, class validation, transforms, and metadata.
* [x] **Data Pipeline & Inspection Tooling:**
  * `scripts/prepare_rugd_bootstrap.py`: Deterministic conversion script mapping RUGD RGB masks to 8-bit single-channel indexed masks.
  * `scripts/visualize_dataset.py`: CLI inspection tool generating side-by-side composite panels (Original, Semantic Mask, Alpha Overlay, Legend), class statistics, and defect detection.
* [x] **Bootstrap Data Ingestion:**
  * Curated 26 real terrain frames across 18 scenes from RUGD into `data/images/` and `data/masks/` with `rugd_` namespace.
  * Generated `data/metadata/manifest.json` and `data/metadata/class_stats.json`.
  * Partitioned into leakage-free scene splits: Train (16 samples, 11 scenes), Val (5 samples, 3 scenes), Test (5 samples, 4 scenes).
* [x] **Verification & Tests:**
  * 15/15 unit and integration tests passing in pytest (including schema invariants, shape validation, synthetic tests, inspection script execution, and real RUGD bootstrap data integrity).

---

## 3. Missing Work

* [ ] **Baseline Segmentation Model (Phase 2):**
  * PyTorch baseline architecture (MobileNetV3-Small encoder + LR-ASPP segmentation head).
  * Training pipeline with class-weighted Cross-Entropy + Dice loss, optimizer, and learning rate scheduler.
  * Experiment tracking and metric logging (`experiments/runs/`).
* [ ] **Evaluation Pipeline (Phase 2):**
  * Evaluation script measuring mIoU on held-out scenes (initial engineering target: mIoU >= 0.70 on Ground).
* [ ] **Candidate Placement & Scoring (Phase 3):**
  * 2D morphological candidate generation, perspective scaling heuristics, obstacle clearance filtering.
* [ ] **Depth & Geometry (Phase 4):**
  * Monocular depth estimation / geometric surface assessment.
* [ ] **Export & Web/Mobile Prototypes (Phases 5 & 6):**
  * ONNX export, Web prototype (`web/`), Mobile application (`mobile/`).

---

## 4. Current Technical Stack

| Component | Selected Version | Notes |
|---|---|---|
| **OS** | macOS | Host system |
| **Python** | 3.12.13 | Managed in `.venv` |
| **Package Manager** | `uv 0.12.18` | Fast reproducible venv and resolver |
| **ML Framework** | PyTorch 2.2.2 | CPU runtime on macOS |
| **Vision Ecosystem** | Torchvision 0.17.2, OpenCV 4.11.0.86, Pillow 12.3.0 | Headless OpenCV |
| **Numerical Engine** | NumPy 1.26.4 | Pinned `<2.0.0` for PyTorch 2.2 compatibility |
| **Testing** | pytest 9.1.1 | 15 unit/integration tests passing |

---

## 5. Current Component Statuses

* **Dataset Tooling:** Complete (`scripts/visualize_dataset.py`, `scripts/prepare_rugd_bootstrap.py`).
* **Dataset Contract:** Complete (`data/raw/rugd/`, `data/images/`, `data/masks/`, `data/metadata/`, `data/splits/`).
* **Dataset Assets:** 26 real terrain frames converted and validated.
* **Model Status:** None. Checkpoints directory `models/` ready.
* **Application Status:** None. UI decoupled from ML core.
* **Testing Status:** 15/15 tests passing.

---

## 6. Current Blockers

* **None for Phase 1.** Phase 1 exit condition ("Dataset pipeline works end-to-end with real images and masks") is met.

---

## 7. Next Milestone & Exact Next Action

* **Next Milestone:** **Phase 2 — Baseline Terrain Segmentation Model**
* **Exact Next Action:**
  * Design the compact PyTorch segmentation baseline model (`stepview/models/segmentation.py`) using a lightweight MobileNetV3 backbone and implement the training loop (`stepview/training/train.py`).
