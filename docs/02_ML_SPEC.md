# StepView AI — Machine Learning Specification

## ML objective

Predict terrain regions that are visually suitable for human foot placement.

## Recommended ML decomposition

Do not force the entire problem into one giant end-to-end model initially.

### Model A — semantic/terrain segmentation

Classes should start simple:

0 background/non-ground
1 candidate ground
2 rock/obstacle
3 vegetation
4 hole/drop
5 water/mud or other uncertain surface

The exact taxonomy must be finalized after dataset inspection.

### Model B — depth estimation

Estimate relative depth/scene geometry where useful.

Required output:
- relative depth map
- optionally surface-normal/geometry features

Depth is supportive evidence, not a direct safety guarantee.

### Model C — candidate foot-placement scorer

Generate candidate regions from the ground mask, then score them using:
- area
- estimated flatness
- slope
- distance
- local visual texture
- obstacle proximity
- confidence
- temporal stability

## Candidate generation

Candidate regions should be:
- sufficiently large
- connected
- away from obvious obstacles
- within a practical distance from the camera/user
- geometrically plausible for a foot

## Candidate score

Initial conceptual score:

score =
    w1 * ground_confidence
  + w2 * flatness_score
  + w3 * geometry_score
  + w4 * size_score
  + w5 * temporal_stability
  - w6 * obstacle_proximity
  - w7 * uncertainty

Weights must be learned/tuned from validation data rather than treated as permanent truths.

## Uncertainty

The system should prefer:
"uncertain — no recommendation"

over:
"confident but wrong"

when confidence is low.

## Training framework

Primary recommendation:
- Python
- PyTorch
- torchvision / suitable vision ecosystem
- OpenCV
- NumPy
- Albumentations or equivalent augmentation pipeline

## Experiment tracking

Every experiment must record:
- model name
- dataset version
- train/validation/test split
- image resolution
- augmentations
- learning rate
- batch size
- epochs
- random seed
- metrics
- qualitative examples
- export version

## Evaluation

Do not rely on accuracy alone.

Track appropriate metrics for:
- segmentation IoU / Dice
- candidate detection precision/recall
- false-positive recommendation rate
- missed-candidate rate
- calibration/confidence behavior
- latency
- memory usage

Most important practical metric:
**How often does the system recommend a visually unsuitable region on unseen terrain?**

## Dataset split

Avoid random frame-only splitting when frames come from the same video.

Preferred:
- train by scene/location
- validation by different scene/location
- test by completely unseen scene/location

This prevents leakage.
