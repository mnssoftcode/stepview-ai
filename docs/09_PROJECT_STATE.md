# StepView — Current Project State

**Last Updated:** 2026-09-24  
**Project Name:** StepView  
**Author:** Lead AI/ML Engineer (Technical Project Agent)

---

## 1. Project Overview & Current Phase

* **Current Phase:** **Phase 1 — Dataset Proof & Pipeline Verification** (In Progress).
* **Official Project Name:** **StepView**.
* **State Assessment:** Foundation established. Git repository initialized, reproducible Python 3.12 environment active, `stepview` package scaffolded, dataset contract & loader implemented with strict class validation, and all unit tests passing on synthetic sample data.

---

## 2. Completed Work

* [x] **Documentation Package & Corrections:**
  * `docs/00_PROJECT_MASTER.md` to `docs/07_DOCUMENT_CREATION_PROMPT.md` reviewed.
  * `docs/08_DOCUMENTATION_AUDIT.md`: Updated with prototype assumption boundaries, validation policy, and target definitions.
  * `docs/05_DEVELOPMENT_ROADMAP.md`: Updated with explicit Phase 1, Phase 2, and Phase 3 exit condition definitions.
* [x] **Version Control & Repository Setup:**
  * Git repository initialized on `main` branch.
  * Comprehensive `.gitignore` created (ignoring `.venv`, caches, weights, raw data binaries).
  * Root `README.md` created.
* [x] **Reproducible Python Environment:**
  * Configured `pyproject.toml` with Python 3.12 constraint and minimal dependencies (`torch==2.2.2`, `torchvision==0.17.2`, `opencv-python-headless==4.11.0.86`, `numpy==1.26.4`, `pillow==12.3.0`, `pytest==9.1.1`).
  * Virtual environment `.venv` created via `uv` using system Python 3.12.13.
  * Package installed in editable mode (`pip install -e ".[dev]"`).
* [x] **Package Structure (`stepview/`):**
  * Scaffolded: `stepview/`, `stepview/data/`, `stepview/models/`, `stepview/geometry/`, `stepview/postprocess/`, `stepview/inference/`, `stepview/utils/`.
* [x] **Dataset Contract & Loader:**
  * `stepview/data/schema.py`: Explicit `TerrainClass` enum (0 to 5), color palette, `SampleMetadata` dataclass, and `validate_mask_classes()` validator.
  * `stepview/data/dataset.py`: `StepViewDataset` separating image loading (RGB), mask loading (indexed uint8), class validation, transforms, and metadata extraction. Supports manifest JSON and split files.
* [x] **Dataset Directory Layout:**
  * Created `data/raw/`, `data/images/`, `data/masks/`, `data/metadata/`, `data/splits/`, `models/`, `experiments/runs/` with `.gitkeep`.
* [x] **Software Correctness Verification:**
  * Created `tests/test_dataset.py`.
  * Verified 9 unit tests passing via pytest (schema invariants, validation failures on invalid class IDs, shape mismatches, manifest parsing, split filtering, PyTorch DataLoader batching).

---

## 3. Missing Work

* [ ] **Dataset Acquisition (Phase 1):**
  * Ingest or record initial real-world terrain image sample set (bootstrap set of ~20–30 frames).
  * Create ground-truth indexed segmentation masks for bootstrap samples.
  * Add train/val split text files and manifest JSON.
* [ ] **Data Pipeline Visualizer (Phase 1):**
  * Lightweight CLI script to load samples and render RGB image + color-coded mask overlays to verify visual annotation fidelity.
* [ ] **Baseline Segmentation Model (Phase 2):**
  * PyTorch baseline model architecture (e.g. lightweight MobileNetV3-Small with segmentation head).
  * Training loop with cross-entropy / Dice loss, learning rate schedule, and metric logging.
* [ ] **Evaluation Pipeline (Phase 2):**
  * Evaluation script computing mIoU (target: mIoU >= 0.70 on Ground class on held-out scenes).
* [ ] **Candidate Placement & Scoring (Phase 3):**
  * Morphological connected components, perspective scaling heuristics, obstacle clearance filtering.
* [ ] **Depth & Geometry (Phase 4):**
  * Monocular depth estimation investigation / geometric surface assessment.
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
| **Testing** | pytest 9.1.1 | Unit test runner |

---

## 5. Current Component Statuses

* **Dataset Status:** Scaffolded. Directory layout ready (`data/images/`, `data/masks/`, `data/metadata/`, `data/splits/`). 0 real terrain images collected yet.
* **Model Status:** None. Checkpoints directory `models/` is ready.
* **Application Status:** None. UI decoupled from ML core.
* **Testing Status:** 9/9 unit tests passing on synthetic verification dataset.

---

## 6. Current Blockers

1. **Lack of Bootstrap Real Terrain Data:** Need a small bootstrap sample (20–30 images or frames) representing varied paths (dirt, rock, obstacles) to test real image-mask loading and train the first baseline.

---

## 7. Next Milestone & Immediate Next Task

* **Next Milestone:** **Phase 1 Completion — Bootstrap Dataset Ingestion & Visual Inspection Tool**
  * Target: Populate `data/` with a small representative bootstrap set of real terrain images and masks, and provide a lightweight inspection script (`scripts/visualize_dataset.py`) to verify real image-mask alignment and label balance.
* **Immediate Next Task:**
  * Define bootstrap dataset acquisition plan or ingest a small curated public trail/terrain subset (e.g., from RUGD or self-collected walking frames) into `data/images/` and `data/masks/`.
