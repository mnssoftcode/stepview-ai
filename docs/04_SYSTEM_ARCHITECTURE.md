# StepView AI — System Architecture

## Stage 1 — research/training

Python
  ↓
Dataset loader
  ↓
Augmentation
  ↓
PyTorch model
  ↓
Evaluation
  ↓
Export
  ↓
ONNX

## Stage 2 — web prototype

Browser camera
  ↓
Frame extraction
  ↓
Pre-processing
  ↓
ONNX Runtime Web
  ↓
Model inference
  ↓
Post-processing
  ↓
Canvas/WebGL overlay

ONNX Runtime Web supports in-browser inference and can use WebAssembly, WebGL, WebGPU and other execution paths depending on browser/platform support.

## Stage 3 — mobile

React Native
  ↓
Native camera/frame pipeline
  ↓
ONNX model
  ↓
On-device inference
  ↓
Overlay

The model contract should stay stable between web and mobile wherever practical.

## Recommended repository

StepView-ai/
  docs/
  data/
  notebooks/
  src/
    data/
    training/
    inference/
    geometry/
    postprocess/
  models/
  web/
  mobile/
  tests/
  scripts/
  experiments/

## Separation of concerns

ML code:
- training
- evaluation
- export

Inference code:
- preprocessing
- model execution
- postprocessing

UI code:
- camera
- overlays
- controls
- debugging

Never put model training logic inside the mobile/web application.

## Model format

Preferred interoperability target:
ONNX.

Reason:
- browser deployment
- mobile deployment
- language/runtime separation
- easier deployment experimentation

Keep the original PyTorch checkpoint too.
