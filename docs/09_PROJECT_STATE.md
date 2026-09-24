# StepView — Current Project State

**Last Updated:** 2026-09-24  
**Project Name:** StepView  
**Author:** Lead AI/ML Engineer (Technical Project Agent)

---

## 1. Project Overview & Current Phase

* **Current Phase:** **Phase 0 — Specification & Project Initialization** (Transitioning into **Phase 1 — Dataset Proof**).
* **Official Project Name:** **StepView** (formerly referenced as StepView AI).
* **State Assessment:** Clean slate. The project has comprehensive conceptual documentation, but zero executable code, zero dataset assets, and zero trained models in the repository.

---

## 2. Completed Work

* [x] **Documentation Package:**
  * `docs/00_PROJECT_MASTER.md`: Master vision, scope, non-goals, and research questions.
  * `docs/01_PRD.md`: Target user, user experience, safety language, and acceptance criteria.
  * `docs/02_ML_SPEC.md`: Machine learning specification, initial class taxonomy, candidate scoring formula, evaluation strategy.
  * `docs/03_DATASET_AND_ANNOTATION.md`: Data collection protocol, annotation levels, and hard-negative cases.
  * `docs/04_SYSTEM_ARCHITECTURE.md`: Training, Web, and Mobile inference pipelines.
  * `docs/05_DEVELOPMENT_ROADMAP.md`: Phase-by-phase roadmap (Phases 0 through 9).
  * `docs/06_AI_AGENT_RULES.md`: Engineering discipline and execution constraints.
  * `docs/07_DOCUMENT_CREATION_PROMPT.md`: Reference prompt.
  * `docs/08_DOCUMENTATION_AUDIT.md`: In-depth audit covering contradictions, gaps, and recommended resolutions.
* [x] **Environment Discovery:**
  * Host OS: macOS.
  * Python runtime: Python 3.12.13 available.
  * Package manager: `uv` installed (`/Users/mns/.local/bin/uv`).
  * Node runtime: Node v20.19.6 / npm 10.8.2 available.
  * Repository directory: `/Users/mns/Mns_Data/code/AiMl/StepViewProject/` inspected.

---

## 3. Missing Work

* [ ] **Version Control & Package Foundation:**
  * Git repository not initialized.
  * No `.gitignore` or `.editorconfig`.
  * Python virtual environment (`.venv`) and dependency definitions (`pyproject.toml`) not created.
* [ ] **Package Skeleton (`stepview/`):**
  * Core modules (`data/`, `models/`, `geometry/`, `postprocess/`, `inference/`, `utils/`) not yet scaffolded.
* [ ] **Dataset & Annotation Assets:**
  * No terrain images or video frames collected.
  * No annotation masks or metadata index files present.
  * No PyTorch Dataset / DataLoader implemented.
* [ ] **Model Assets:**
  * No baseline segmentation model implemented or trained.
  * No baseline weights or ONNX export artifacts.
* [ ] **Applications & Interfaces:**
  * No CLI test scripts or offline visualization tools.
  * No Web prototype (`web/`) or mobile application (`mobile/`).
* [ ] **Test Suite:**
  * No automated unit tests (`tests/`) or continuous integration scripts.

---

## 4. Current Technical Stack

| Component | Current State | Planned Target |
|---|---|---|
| **Version Control** | None (uninitialized) | Git with conventional commits |
| **Python Environment** | System Python 3.12.13 | Managed `.venv` via `uv` |
| **ML Framework** | None installed | PyTorch 2.x, Torchvision |
| **Vision & Math** | None installed | OpenCV (`opencv-python-headless`), NumPy, Albumentations, Pillow |
| **Model Interop** | None | ONNX, ONNX Runtime |
| **Web Runtime** | Node 20.19.6 / npm 10.8.2 | Vite + React + Vanilla CSS + ONNX Runtime Web |
| **Mobile Runtime** | Not started | React Native (Expo or Bare) + ONNX Runtime Mobile |

---

## 5. Current Component Statuses

* **Dataset Status:** Empty. 0 images, 0 masks. Directory structure for dataset versioning (`data/v0.1/`) needs initialization.
* **Model Status:** No model. Baseline model architecture needs to be selected (recommended: MobileNetV3-Large or Small with LR-ASPP head for terrain segmentation).
* **Application Status:** None. `stepview/` is an empty directory.
* **Inference Pipeline:** None.

---

## 6. Current Blockers

1. **Lack of Version Control:** Workspace is not a Git repo. Any edits risk untracked state or irreversible changes.
2. **Missing Python Project Environment:** No virtual environment or locked dependency specification exists.
3. **No Labeled Bootstrap Data:** Model training cannot begin until a minimal representative image set (even 20–30 bootstrap terrain images with masks) or public domain proxy dataset (e.g. RUGD, Trailscape, or WildScenes subset) is ingested.

---

## 7. Next Milestone & Immediate Next Task

* **Next Milestone:** **Phase 1 — Dataset Proof & Pipeline Verification**
  * Target: Establish an end-to-end reproducible data ingestion pipeline that loads image-mask pairs, verifies classes, applies augmentations, and visualizes overlays.
* **Immediate Next Task:**
  1. Initialize Git repository with a complete `.gitignore`.
  2. Scaffold Python project configuration (`pyproject.toml`) using `uv`.
  3. Set up the `stepview` package directory structure with minimal modules and `__init__.py` files.
  4. Create a dataset schema and loader script with a synthetic/mock dataset verification test to ensure end-to-end data pipeline integrity before real data ingestion.
