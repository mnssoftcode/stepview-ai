# StepView — Current Project State

**Last Updated:** 2026-09-24  
**Project Name:** StepView  
**Author:** Lead AI/ML Engineer (Technical Project Agent)

---

## 1. Project Overview & Current Phase

* **Current Phase:** **Phase 1 — Dataset Proof & Pipeline Verification** (In Progress).
* **Official Project Name:** **StepView**.
* **State Assessment:** Tooling, contracts, and guidelines fully operational. Dataset contract, inspection visualizer (`scripts/visualize_dataset.py`), annotation guidelines, manifest specification, and bootstrap data workflow are locked. All 14 software correctness tests pass. Ready for ingestion of the initial 20–30 real-world terrain frames.

---

## 2. Completed Work

* [x] **Documentation & Strategy Foundations:**
  * `docs/00_PROJECT_MASTER.md` to `docs/07_DOCUMENT_CREATION_PROMPT.md`: Architecture, PRD, ML spec.
  * `docs/08_DOCUMENTATION_AUDIT.md`: Audit with prototype assumptions, empirical validation rules, and testing boundaries.
  * `docs/10_ANNOTATION_GUIDELINES.md`: Comprehensive labeling rules for 6 classes, edge cases (shadows, roots, scree), and candidate footstep heuristics.
  * `docs/11_DATASET_MANIFEST.md`: Data directory contract (`images/`, `masks/`, `metadata/`, `splits/`) and JSON manifest schema.
  * `docs/12_BOOTSTRAP_DATA_WORKFLOW.md`: 10-step capture-to-commit workflow, tooling recommendation (Labelme/AnyLabeling), and dataset capacity boundaries.
  * `docs/13_PUBLIC_DATASET_EVALUATION.md`: Comparative evaluation of RUGD, RELLIS-3D, WildScenes, Freiburg Forest, and class remapping rules.
* [x] **Version Control & Repository Setup:**
  * Git repository initialized on `main` branch with conventional commit history.
  * Comprehensive `.gitignore` created.
* [x] **Reproducible Python 3.12 Environment (`uv`):**
  * `pyproject.toml` with pinned dependencies (`torch==2.2.2`, `torchvision==0.17.2`, `opencv-python-headless==4.11.0.86`, `numpy==1.26.4`, `pillow==12.3.0`, `pytest==9.1.1`).
  * `.venv` created via `uv` using system Python 3.12.13.
* [x] **Package Structure & Loader (`stepview/`):**
  * `stepview/data/schema.py`: Explicit `TerrainClass` enum (0..5), color palette, `SampleMetadata`, `validate_mask_classes()`.
  * `stepview/data/dataset.py`: `StepViewDataset` separating image loading, mask loading, class validation, transforms, and metadata.
* [x] **Dataset Tooling:**
  * `scripts/visualize_dataset.py`: CLI tool for dataset inspection, image-mask overlay rendering, class pixel counting, percentage distribution, and defect detection (invalid IDs, dimension mismatches, missing files).
* [x] **Software Correctness Verification:**
  * `tests/test_dataset.py`: 14 automated unit tests passing via pytest covering schema invariants, validation failures, shape mismatches, split filtering, manifest parsing, batching, and inspection script execution.

---

## 3. Missing Work

* [ ] **Bootstrap Dataset Ingestion (Phase 1):**
  * Collect/extract 20–30 real-world terrain image frames across 4–6 varied scenes (dirt, rocks, roots, slopes, shade).
  * Generate ground-truth indexed 8-bit PNG masks adhering to `docs/10_ANNOTATION_GUIDELINES.md`.
  * Create `splits/train.txt`, `splits/val.txt`, and `metadata/manifest.json`.
  * Execute `scripts/visualize_dataset.py` on real data to verify visual alignment and class distribution.
* [ ] **Baseline Segmentation Model (Phase 2):**
  * PyTorch baseline architecture (MobileNetV3-Small + LR-ASPP head).
  * Training loop with cross-entropy / Dice loss, learning rate schedule, and metric logging.
* [ ] **Evaluation Pipeline (Phase 2):**
  * Evaluation script measuring mIoU on held-out scenes (engineering target: mIoU >= 0.70 on Ground).
* [ ] **Candidate Placement & Scoring (Phase 3):**
  * 2D morphological candidate generation, perspective scaling heuristics, obstacle clearance filtering.
* [ ] **Depth & Geometry (Phase 4):**
  * Monocular depth estimation / geometric surface assessment.
* [ ] **Export & Interfaces (Phases 5 & 6):**
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
| **Testing** | pytest 9.1.1 | 14 unit tests passing |

---

## 5. Current Component Statuses

* **Dataset Tooling:** Complete (`scripts/visualize_dataset.py`).
* **Dataset Contract:** Complete (`data/raw/`, `data/images/`, `data/masks/`, `data/metadata/`, `data/splits/`).
* **Dataset Assets:** 0 real images. Directory placeholders with `.gitkeep` active.
* **Model Status:** None. Checkpoints directory `models/` ready.
* **Application Status:** None. UI decoupled from ML core.
* **Testing Status:** 14/14 unit tests passing on synthetic verification dataset.

---

## 6. Current Blockers

1. **Acquisition of 20–30 Real-World Terrain Frames:** Need initial sample video/images conforming to the capture protocol (downward pitch 35°–50°, handheld/chest height 1.0–1.4m) across varied path types to run through the annotation and visual inspection workflow.

---

## 7. Next Milestone & Exact Next Action

* **Next Milestone:** **Phase 1 Completion — Bootstrap Real-Terrain Verification**
* **Exact Next Action:**
  * Obtain 20–30 candidate frames of real terrain, generate initial single-channel 8-bit masks following [docs/10_ANNOTATION_GUIDELINES.md](file:///Users/mns/Mns_Data/code/AiMl/StepViewProject/docs/10_ANNOTATION_GUIDELINES.md), place them into `data/images/` and `data/masks/`, create `data/metadata/manifest.json`, and run `scripts/visualize_dataset.py` to confirm real-world pipeline integrity.
