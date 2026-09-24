# StepView AI — Dataset and Annotation Plan

## Dataset strategy

Use three sources:

1. Public datasets for bootstrapping.
2. Self-collected terrain videos/images.
3. Carefully selected difficult cases.

The self-collected dataset becomes increasingly important because the final camera position, terrain style, and application behavior are specific to this product.

## Capture protocol

Record:
- walking trails
- rocky ground
- dirt
- gravel
- stairs
- uneven pavement
- slopes
- roots
- vegetation
- shadows
- low light
- wet-looking surfaces
- cluttered terrain

Capture at different:
- camera heights
- camera tilts
- walking speeds
- lighting conditions
- distances
- terrain types

## Annotation levels

### Level 1 — terrain segmentation
Draw regions for ground and major non-ground hazards.

### Level 2 — candidate placement
Mark regions a human annotator would consider visually suitable for placing a foot.

### Level 3 — exclusion reason
For rejected candidates, optionally record reason:
- too small
- too steep
- obstacle
- hole
- uncertain
- cluttered
- low confidence

## Annotation rules

Annotators must judge only what is visible.

Do not label a region "safe" because it looks like solid rock if hidden geometry cannot be known.

Use:
- recommended-looking
- uncertain
- reject

## Dataset quality checks

Before training:
- remove duplicates
- inspect corrupted images
- verify labels
- check class imbalance
- inspect scene diversity
- check train/test leakage
- maintain dataset version

## Hard-negative collection

Specifically collect:
- visually flat but unsuitable surfaces
- shiny/wet surfaces
- loose gravel
- shadows that look like holes
- rocks that look like ground
- vegetation-covered ground
- strong perspective distortions

These are valuable because naive models often become overconfident on them.

## Dataset versioning

Example:

dataset/
  v0.1/
  v0.2/
  v0.3/

Never silently replace labels in an existing experiment.
