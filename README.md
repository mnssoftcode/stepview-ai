# StepView — Web-First AI Footstep Recommendation System

StepView is a web-first, on-device computer vision application that identifies visually suitable, hazard-free ground locations for human foot placement while walking or hiking outdoors.

The website is the **main product**, designed to run on-device inside modern web browsers across Android phones, iPhones, tablets, and laptops without requiring native app installation or cloud server inference.

---

## Architecture Overview

```text
Camera / Video Stream
       │
       ▼
Canvas Frame Capture
       │
       ▼
ONNX Runtime Web (WebAssembly / WebGPU)
       │
       ▼
Terrain Semantic Segmentation (256x256 MobileNetV3-Small + LR-ASPP)
[Classes: Background, Ground, Obstacle, Vegetation, Hole/Drop, Uncertain]
       │
       ▼
2D Footstep Candidate Engine
[Euclidean Hazard Clearance Transform + Perspective Footprint Scaling]
       │
       ▼
Non-Maximum Suppression (NMS) & Scoring
       │
       ▼
Real-Time Canvas Overlay
[Top Step #1 Gold Badge, Emerald Green Alternatives, Hazard Refusal Banner]
```

All video processing and model evaluation occurs **100% locally on-device**. No camera video or sensor frames are transmitted to any server.

---

## Browser & Hardware Requirements

| Requirement | Specification | Notes |
|---|---|---|
| **Browsers** | Safari (iOS 16+), Chrome (Android & Desktop), Edge, Firefox | Modern ESM + WebAssembly support |
| **Acceleration** | WebGPU (where available) or WebAssembly (SIMD) | Automatic fallback to WASM |
| **Camera Access** | HTTPS or `localhost` context required | Mobile browsers enforce HTTPS for camera hardware access |
| **Device Orientation**| Portrait (handheld walking) or Landscape | Automatically adapts canvas and lookahead geometry |

---

## Web Application Setup & Run

### 1. Prerequisites
* **Node.js**: v18+ (tested on Node v20.19+)
* **npm**: v9+ (tested on npm 10.8+)

### 2. Local Development Server

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173/` in your browser.

* Click **Start Camera** to activate live rear-camera inference.
* Or click **Trail Sample**, **Park Sample**, or **Village Sample** to test instantly without a camera.
* Or click **Upload / Photo** to run inference on any photo from your device camera roll.

### 3. Production Build & Local Preview

```bash
cd web
npm run build
npm run preview
```

The production bundle is generated into `web/dist/`.

---

## Deployment Instructions (Render)

StepView is configured for one-click static site deployment on **Render**:

1. Push your repository to **GitHub**.
2. Log into [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** $\rightarrow$ **Static Site** (or **Blueprint** using `render.yaml`).
4. Configure the service:
   * **Name:** `stepview-web`
   * **Branch:** `main`
   * **Root Directory:** leave empty (or repository root)
   * **Build Command:** `cd web && npm install && npm run build`
   * **Publish Directory:** `./web/dist`
5. Render automatically provides free TLS certificates (`https://<your-app>.onrender.com`), enabling full mobile camera hardware access on iOS and Android.

---

## Python Research & Verification Pipeline

StepView includes a Python 3.12 research, training, and offline verification suite:

### 1. Python Environment Setup (`uv`)

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Run Automated Test Suite (33 Tests)

```bash
pytest -v
```

### 3. Offline ONNX Footstep Proposal Script

Run the ONNX segmentation + candidate proposal pipeline on local images:

```bash
python scripts/propose_footsteps_onnx.py \
    --model models/stepview_segmentation.onnx \
    --image data/images/rugd_park-1_frame03056.png \
    --output-dir experiments/visualizations/onnx_footsteps
```

---

## Project Structure

```
StepViewProject/
├── render.yaml               # Render static web deployment configuration
├── README.md                 # Project documentation and deployment guide
├── pyproject.toml            # Python packaging and dependencies
├── models/
│   └── stepview_segmentation.onnx  # Exported MobileNetV3-Small LR-ASPP model (4.12 MB)
├── web/                      # Production Web Application (TypeScript + Vite)
│   ├── index.html            # Main web UI and viewport
│   ├── package.json          # Dependencies (onnxruntime-web, vite, typescript)
│   ├── vite.config.ts        # Server and COOP/COEP headers
│   ├── public/
│   │   ├── models/           # Static ONNX model for browser fetch
│   │   └── samples/          # Built-in trail, park, and village test images
│   ├── scripts/
│   │   └── copy-wasm.js      # WASM asset synchronization from node_modules
│   └── src/
│       ├── types.ts          # Core terrain and candidate data contracts
│       ├── segmenter.ts      # ONNX Runtime Web session and tensor inference
│       ├── footstepEngine.ts # 2D Footstep Candidate Engine (distance map + NMS)
│       ├── renderer.ts       # HTML5 Canvas overlay rendering
│       └── main.ts           # App lifecycle, camera stream, and frame pacing
├── stepview/                 # Python package (data, models, inference, geometry)
├── scripts/                  # Python CLI tools (export_onnx, predict_onnx, etc.)
├── tests/                    # Automated unit & integration tests
└── docs/                     # Specifications, architecture, and project state
```

---

## License

Apache 2.0 / Proprietary Research Prototype. See project documentation for details.
