# StepView — Current Project State

**Last Updated:** 2026-09-24  
**Project Name:** StepView  
**Author:** Lead AI/ML Engineer (Technical Project Agent)

---

## 1. Project Overview & Current Phase

* **Current Phase:** **Phase 5 (Step 2) — ONNX Footstep Pipeline & Web Prototype Foundation Complete**
* **Official Project Name:** **StepView**.
* **Strategic Product Architecture:**
  * **Web-First & Web-Only:** StepView is designed strictly as a browser-first web application running on-device inference via **ONNX Runtime Web** (WebGPU / WebAssembly).
  * **No Native Apps:** No React Native, native Android, or native iOS application codebases. The website works on Android phones, iPhones, tablets, and laptops directly via standard web browsers.
  * **Zero Server Transmission:** Video frames remain on the client device for privacy, latency, and reliability.
* **State Assessment:**
  * **End-to-End ONNX Footstep Pipeline:** `scripts/propose_footsteps_onnx.py` connects `models/stepview_segmentation.onnx` directly into `FootstepProposalEngine`. Successfully executed across multiple diverse terrain test images with hazard refusal and candidate ranking.
  * **Web Application Foundation (`web/`):** TypeScript, Vite, ONNX Runtime Web (`onnxruntime-web`), and HTML5 Canvas renderer implemented. Supports live camera (`getUserMedia`) with graceful fallback to sample terrain test images (`trail.png`, `park.png`).
  * **Browser Footstep Engine:** `FootstepProposalEngineWeb` implemented in TypeScript with two-pass distance transform, perspective scaling, and Non-Maximum Suppression.
  * **Production Build Verified:** `npm run build` cleanly packages the application in 1.80s.
  * **Dev Server Active:** Running on `http://127.0.0.1:5173/` with COOP/COEP headers for multi-threaded WASM support.
  * **Software Engineering Status:** 33/33 automated pytest tests passing cleanly.

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
  * `docs/20_TRAINING_DATA_STRATEGY.md`: Three-dataset strategy (A: bootstrap, B: semantic pretraining, C: smartphone validation), median frequency class weights, and missing class handling.
  * `docs/21_RUGD_DATASET_B_SELECTION.md`: 600-frame selection matrix across 18 scenes with uniform temporal strides and strict scene isolation.
* [x] **Version Control & Repository Setup:**
  * Git repository initialized on `main` branch with clean conventional commit history.
  * Comprehensive `.gitignore` protecting binary assets.
* [x] **Python Environment & Package Structure:**
  * Python 3.12 managed via `uv` with pinned packages in `pyproject.toml`.
  * `stepview/data/`: `TerrainClass` enum, datasets, collate functions, validation.
  * `stepview/models/`: `StepViewSegmentationModel` (MobileNetV3-Small + LR-ASPP, ~1.08M params).
  * `stepview/inference/`: PyTorch `TerrainSegmenter` and standalone `ONNXTerrainSegmenter`.
  * `stepview/geometry/`: `FootstepProposalEngine` (Euclidean distance transform, perspective scaling, clearance checks, NMS).
* [x] **Model Checkpoints & ONNX Models:**
  * `models/stepview_segmentation.onnx`: 4.12 MB, opset 17, $256 \times 256$ input, 4.84 ms / 206 FPS CPU inference.
  * `experiments/runs/dataset_b_baseline/best_checkpoint.pt`: PyTorch baseline (0.4464 Val mIoU, 0.5006 Test Ground IoU).
* [x] **Deployment Scripts:**
  * `scripts/export_onnx.py`: CLI checkpoint-to-ONNX exporter with graph verification.
  * `scripts/predict_onnx.py`: Standalone ONNX segmentation CLI.
  * `scripts/propose_footsteps_onnx.py`: Complete ONNX-to-footstep pipeline generating overlays and JSON metadata.
* [x] **Web Product Prototype (`web/`):**
  * `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`.
  * `web/src/types.ts`: TypeScript contracts for terrain classes, candidates, segmentation results.
  * `web/src/segmenter.ts`: ONNX Runtime Web session management, WebGPU / WASM execution.
  * `web/src/footstepEngine.ts`: Browser-side 2D Footstep Candidate Engine with perspective scaling and NMS.
  * `web/src/renderer.ts`: HTML5 Canvas rendering for video, mask overlay, candidate footprints, and status indicators.
  * `web/src/main.ts`: Application coordinator, camera stream management, frame loop, and live metrics.
  * `web/index.html`: Responsive high-contrast user interface.
* [x] **Verification & Tests:**
  * 33/33 automated tests passing in pytest.
  * Web build verification: `npm run build` passes with zero errors.

---

## 3. Current Technical Stack

| Component | Technology | Version / Notes |
|---|---|---|
| **OS** | macOS | Host system |
| **Python** | Python 3.12.13 | Managed in `.venv` |
| **Package Manager** | `uv 0.12.18` / `npm 10.8.2` | Fast reproducible environments |
| **ML Framework** | PyTorch 2.2.2 / ONNX 1.23.0 | Export & validation runtime |
| **Inference Runtime** | ONNX Runtime 1.23.2 (Python) / ONNX Runtime Web 1.20+ | CPU / WebGPU / WASM execution |
| **Frontend Foundation**| TypeScript 5.4.5, Vite 5.2.11 | Modern ESM web development |
| **Testing** | pytest 9.1.1 | 33 unit/integration tests passing (100% pass rate) |

---

## 4. Current Component Statuses

* **Dataset Assets:** Dataset A (26 frames) and Dataset B (600 frames) available.
* **ONNX Model:** `models/stepview_segmentation.onnx` active and served in `web/public/models/`.
* **Python Pipeline:** `scripts/propose_footsteps_onnx.py` tested and operational.
* **Web Prototype:** Ready for browser access at `http://127.0.0.1:5173/`.
* **Testing Status:** 33/33 tests passing.

---

## 5. Next Milestone & Immediate Focus

* **Next Milestone:** Test and iterate on real device web camera performance, add perspective calibration controls, and evaluate field ergonomics.
* **Immediate Focus:** Await user direction on the next single task.
