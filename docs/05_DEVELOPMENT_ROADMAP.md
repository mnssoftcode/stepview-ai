# StepView AI — Development Roadmap

## Phase 0 — specification

Deliverables:
- PRD
- ML specification
- dataset plan
- architecture
- evaluation plan

Exit condition:
All major terminology and success criteria are written down.

## Phase 1 — dataset proof

Tasks:
1. Collect initial images/videos.
2. Annotate a small representative set.
3. Build data loader.
4. Visualize labels.
5. Check imbalance and leakage.

Exit condition:
Dataset pipeline works end-to-end. Synthetic dataset tests validate software correctness only, NOT machine learning performance.

## Phase 2 — segmentation baseline

Tasks:
1. Train a simple segmentation baseline.
2. Measure validation metrics (initial engineering target: mIoU >= 0.70 on ground class; note that mIoU is an engineering baseline metric, not proof of physical safety).
3. Visualize predictions.
4. Identify failure cases.
5. Improve labels/augmentation.

Exit condition:
Model reaches baseline segmentation convergence on held-out scenes without overfitting.

## Phase 3 — candidate placement

Tasks:
1. Convert masks into candidate regions.
2. Filter candidates by size (initial prototype assumption: ~25-30 cm x 10 cm, perspective scaled).
3. Add local geometry/flatness features (initial prototype assumption: slope < 25°, clearance ~3 cm).
4. Add obstacle-distance filtering.
5. Produce ranked candidate regions.
6. Evaluate against initial prototype target: False Recommendation Rate <= 10% measured strictly on a held-out real-terrain test set.

Exit condition:
The system reliably highlights plausible foot-placement candidates on held-out still images without a live camera.

## Phase 4 — depth and geometry

Tasks:
1. Integrate depth estimation.
2. Estimate local slope/flatness.
3. Measure candidate surface geometry.
4. Compare geometry-aware vs segmentation-only results.

Exit condition:
Geometry improves or at least does not materially worsen validation metrics.

## Phase 5 — browser prototype

Tasks:
1. Build React web app.
2. Add image upload.
3. Add recorded-video support.
4. Export model to ONNX.
5. Run inference with ONNX Runtime Web.
6. Add live camera.
7. Measure browser FPS and latency.

Exit condition:
A user can point a browser camera at terrain and see real-time candidate regions.

## Phase 6 — mobile prototype

Tasks:
1. Build React Native camera screen.
2. Load ONNX model on device.
3. Implement frame throttling.
4. Add overlay.
5. Measure latency, memory, battery.
6. Compare device results with web results.

Exit condition:
Real-time on-device inference on the selected Android test phone.

## Phase 7 — field validation

Test categories:
- sunny
- cloudy
- shade
- evening
- rocky
- gravel
- dirt
- slope
- stairs
- vegetation
- wet-looking terrain
- crowded/complex scenes

For every failure:
- save frame
- save prediction
- record expected behavior
- classify root cause
- add to hard-negative dataset when appropriate

## Phase 8 — optimization

Only after quality is acceptable:
- quantization
- smaller input resolution
- model pruning/distillation if needed
- frame skipping
- temporal smoothing
- execution provider optimization
- model size reduction

## Phase 9 — portfolio release

Deliver:
- GitHub repository
- technical report
- architecture diagram
- dataset methodology
- model metrics
- failure analysis
- demo video
- web demo
- mobile demo
- README

The portfolio should emphasize engineering decisions and measured results, not just the UI.
