# StepView AI — Project Master Specification

## 1. Project identity

**Project name:** StepView AI  
**One-line description:** A camera-based computer-vision system that identifies visually suitable terrain regions for human foot placement while walking or hiking.

## 2. Product goal

Given a live camera view of terrain in front of a person, the system should:
1. Understand the visible ground.
2. Identify candidate regions large enough for a foot.
3. Estimate whether each candidate region is visually suitable for foot placement.
4. Rank candidate regions.
5. Overlay the recommendation on the camera view.

The product must communicate uncertainty. It must never claim that a surface is guaranteed safe.

## 3. Initial product scope

### V1 — visual terrain suitability prototype
Input:
- Single image or short video frame.
- Camera pointing toward the walking surface.

Output:
- Candidate ground regions.
- Suitability score/confidence.
- Overlay:
  - green = recommended candidate
  - yellow = uncertain/caution
  - red = reject

V1 is a research prototype, not a safety-certified fall-prevention system.

### Out of scope for V1
- Guaranteed fall prevention.
- Medical/safety certification.
- Understanding hidden hazards.
- Reliable prediction of rock stability or soil load-bearing capacity.
- Autonomous control of a person's movement.
- Complex GPS/navigation features.
- User accounts, payments, social features, backend marketplace functionality.

## 4. Target development path

Phase A:
Python + dataset + offline image inference.

Phase B:
Real-time browser prototype.

Phase C:
Depth/geometry-aware candidate ranking.

Phase D:
Mobile on-device inference.

Phase E:
Field validation and optimization.

## 5. Core technical principle

Keep the ML model independent from the UI.

Pipeline:

Camera/Image
    ↓
Pre-processing
    ↓
Terrain understanding model
    ↓
Depth / geometry estimation
    ↓
Candidate foot-placement generation
    ↓
Suitability scoring
    ↓
Temporal smoothing
    ↓
Overlay renderer

## 6. Main technical questions

- What visual cues correlate with suitable foot placement?
- Can the model distinguish walkable ground from obstacles?
- Can depth/geometry improve recommendations?
- How much camera tilt affects performance?
- What minimum candidate area is required for a human foot?
- How can uncertainty be represented?
- What latency is acceptable on a normal phone?
- How does performance change between daylight, shade, rocks, soil, mud, gravel, and vegetation?

## 7. Success criteria

The project is successful when:
- The model consistently identifies candidate ground regions in unseen scenes.
- Recommendations are spatially coherent rather than random pixels.
- The system operates in real time at a practical frame rate on a target device.
- Performance is measured on held-out environments, not just training images.
- False confidence is actively minimized.
- The model and app remain usable offline after model download.

