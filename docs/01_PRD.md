# StepView AI — Product Requirements Document

## Problem

A person walking on irregular terrain cannot always visually identify the best immediate location for the next foot placement. The objective is to use computer vision to highlight visually suitable terrain regions.

## Primary user

A person walking/hiking on:
- trails
- rocky paths
- uneven natural ground
- moderately irregular terrain

## User experience

1. User opens camera.
2. Camera points toward the walking surface.
3. System processes the view continuously.
4. Candidate foot-placement regions appear as overlays.
5. A small number of recommendations are shown rather than covering the entire screen.
6. Confidence/uncertainty is visible.

## Product language

Use:
- "Recommended placement"
- "Candidate surface"
- "Uncertain"
- "Avoid"

Do not use:
- "Guaranteed safe"
- "You cannot fall"
- "This surface is definitely stable"

## V1 UI

Screen:
- camera preview
- terrain overlay
- one primary recommended region
- secondary candidate regions
- confidence indicator
- model FPS / latency in debug mode

## V1 modes

### Photo mode
Upload/take an image and visualize predictions.

### Video mode
Process recorded video.

### Live camera mode
Process live camera frames.

## Acceptance criteria

A prototype should:
- load a model successfully
- accept an image
- generate candidate regions
- render overlays correctly
- show confidence
- fail gracefully when confidence is low
