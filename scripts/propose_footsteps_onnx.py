#!/usr/bin/env python3
"""StepView — Direct ONNX Segmentation to Footstep Proposal Pipeline.

Connects the standalone ONNX Runtime segmentation engine directly to the
2D Footstep Candidate Engine:
1. Loads exported ONNX model (`models/stepview_segmentation.onnx`)
2. Executes low-latency on-device inference via ONNX Runtime
3. Passes prediction maps directly to `FootstepProposalEngine`
4. Filters hazard zones and generates ranked candidate footstep footprints
5. Produces multi-panel diagnostic overlays and structured candidate JSON metadata

Usage:
    python scripts/propose_footsteps_onnx.py \
        --model models/stepview_segmentation.onnx \
        --image data/images/rugd_trail_frame00001.png \
        --output-dir experiments/visualizations/onnx_footsteps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from stepview.data.schema import CLASS_NAMES, CLASS_PALETTE, NUM_CLASSES, TerrainClass
from stepview.geometry.footstep_proposals import FootstepCandidate, FootstepProposalEngine
from stepview.inference.onnx_segmenter import ONNXTerrainSegmenter


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map indexed single-channel mask to RGB color image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_PALETTE.items():
        color_mask[mask == class_id] = color
    return color_mask


def draw_footstep_candidates(
    image_bgr: np.ndarray,
    candidates: List[FootstepCandidate],
) -> np.ndarray:
    """Render footstep candidate footprints with distinctive styling for top choice."""
    canvas = image_bgr.copy()
    overlay = canvas.copy()

    # Draw candidates in reverse rank order (so rank 1 is drawn on top)
    for cand in reversed(candidates):
        is_top = (cand.rank == 1)
        center = (int(round(cand.x_center)), int(round(cand.y_center)))
        axes = (max(2, int(round(cand.width / 2.0))), max(2, int(round(cand.height / 2.0))))

        if is_top:
            fill_color = (0, 215, 255)  # BGR gold/amber
        else:
            fill_color = (80, 200, 80)  # BGR emerald green

        cv2.ellipse(overlay, center, axes, 0.0, 0.0, 360.0, fill_color, -1)

    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    # Draw solid borders, labels, and badges
    for cand in reversed(candidates):
        is_top = (cand.rank == 1)
        center = (int(round(cand.x_center)), int(round(cand.y_center)))
        axes = (max(2, int(round(cand.width / 2.0))), max(2, int(round(cand.height / 2.0))))

        if is_top:
            border_color = (0, 220, 255)  # Gold border
            border_thick = 3

            # Outer ring
            cv2.ellipse(canvas, center, axes, 0.0, 0.0, 360.0, border_color, border_thick)
            # Inner white accent ring
            inner_axes = (max(1, axes[0] - 2), max(1, axes[1] - 2))
            cv2.ellipse(canvas, center, inner_axes, 0.0, 0.0, 360.0, (255, 255, 255), 1)

            # Center crosshair
            cv2.drawMarker(canvas, center, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)

            # Prominent badge
            badge_text = f"TOP STEP #1 (Score: {cand.score:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            thickness = 1
            (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)

            bx0 = center[0] - tw // 2 - 6
            by0 = center[1] - axes[1] - th - 10
            bx1 = bx0 + tw + 12
            by1 = by0 + th + 8

            if by0 < 10:
                by0 = center[1] + axes[1] + 6
                by1 = by0 + th + 8

            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (20, 20, 20), -1)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (0, 220, 255), 1)
            cv2.putText(canvas, badge_text, (bx0 + 6, by1 - 5), font, font_scale, (0, 230, 255), thickness, cv2.LINE_AA)
        else:
            border_color = (0, 255, 128)
            cv2.ellipse(canvas, center, axes, 0.0, 0.0, 360.0, border_color, 2)

            label = f"#{cand.rank} ({cand.score:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.40
            cv2.putText(
                canvas,
                label,
                (center[0] - axes[0], center[1] - axes[1] - 4),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return canvas


def create_legend_banner(width: int, height: int = 34) -> np.ndarray:
    """Create a horizontal banner displaying class color chips and names."""
    banner = np.full((height, width, 3), 28, dtype=np.uint8)
    num_items = len(TerrainClass)
    col_width = width // num_items

    for i, cls in enumerate(TerrainClass):
        x_start = i * col_width
        chip_x = x_start + 8
        chip_y = (height - 16) // 2
        chip_w = 18
        chip_h = 16

        rgb_color = CLASS_PALETTE[cls.value]
        bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
        cv2.rectangle(banner, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), bgr_color, -1)
        cv2.rectangle(banner, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), (180, 180, 180), 1)

        label = f"{cls.name.lower()}"
        cv2.putText(
            banner,
            label,
            (chip_x + chip_w + 6, height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return banner


def generate_footstep_composite_panel(
    image_rgb: np.ndarray,
    overlay_bgr: np.ndarray,
    footsteps_bgr: np.ndarray,
    sample_name: str,
    latency_ms: float,
    top_candidate: Optional[FootstepCandidate],
    num_candidates: int,
) -> np.ndarray:
    """Generate 3-panel composite: Original RGB | Segmentation Overlay | Recommended Footsteps."""
    h, w = image_rgb.shape[:2]
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    header_h = 42
    total_w = w * 3
    panel = np.zeros((h + header_h, total_w, 3), dtype=np.uint8)
    panel[:header_h, :] = 24

    panel[header_h:, 0:w] = img_bgr
    panel[header_h:, w:2*w] = overlay_bgr
    panel[header_h:, 2*w:3*w] = footsteps_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)

    cv2.putText(panel, f"StepView Camera: {sample_name} ({w}x{h})", (12, 26), font, 0.50, color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"ONNX Segmentation ({latency_ms:.1f} ms)", (w + 12, 26), font, 0.50, (100, 255, 100), 1, cv2.LINE_AA)

    top_str = f"Top Score: {top_candidate.score:.2f} (Clr: {top_candidate.obstacle_clearance_px:.0f}px)" if top_candidate else "No Safe Footstep Found"
    cv2.putText(panel, f"2D Footsteps ({num_candidates} cands) | {top_str}", (2 * w + 12, 26), font, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

    legend = create_legend_banner(total_w, height=34)
    full_composite = np.vstack([panel, legend])

    return full_composite


def process_image(
    image_path: Path,
    segmenter: ONNXTerrainSegmenter,
    proposal_engine: FootstepProposalEngine,
    output_dir: Path,
    alpha: float = 0.40,
) -> Dict[str, Any]:
    """Execute end-to-end ONNX segmentation, footstep proposal, and visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 1. ONNX Model inference
    t0 = time.perf_counter()
    seg_res = segmenter.predict(img_rgb)
    onnx_infer_ms = seg_res["latency_ms"]

    class_mask = seg_res["class_mask"]
    conf_map = seg_res["confidence_map"]
    probs = seg_res["probabilities"]

    # 2. Footstep proposal generation
    t_prop0 = time.perf_counter()
    prop_res = proposal_engine.propose_footsteps(
        class_mask=class_mask,
        confidence_map=conf_map,
        probabilities=probs,
    )
    prop_ms = (time.perf_counter() - t_prop0) * 1000.0
    total_pipeline_ms = (time.perf_counter() - t0) * 1000.0

    top_cands: List[FootstepCandidate] = prop_res["top_candidates"]
    top_candidate = top_cands[0] if top_cands else None

    # 3. Create visual overlays
    color_mask_rgb = colorize_mask(class_mask)
    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, color_mask_bgr, alpha, 0)
    footsteps_bgr = draw_footstep_candidates(overlay_bgr, top_cands)

    # 4. Save individual artifact images
    orig_path = output_dir / f"{stem}_original.png"
    overlay_path = output_dir / f"{stem}_seg_overlay.png"
    footsteps_path = output_dir / f"{stem}_candidates.png"
    composite_path = output_dir / f"{stem}_composite.png"
    json_path = output_dir / f"{stem}_candidates.json"

    cv2.imwrite(str(orig_path), img_bgr)
    cv2.imwrite(str(overlay_path), overlay_bgr)
    cv2.imwrite(str(footsteps_path), footsteps_bgr)

    composite = generate_footstep_composite_panel(
        image_rgb=img_rgb,
        overlay_bgr=overlay_bgr,
        footsteps_bgr=footsteps_bgr,
        sample_name=stem,
        latency_ms=onnx_infer_ms,
        top_candidate=top_candidate,
        num_candidates=len(top_cands),
    )
    cv2.imwrite(str(composite_path), composite)

    # 5. Build candidate metadata dictionary
    candidates_data = [cand.to_dict() for cand in top_cands]

    unique_ids, counts = np.unique(class_mask, return_counts=True)
    total_px = class_mask.size
    class_dist = {
        CLASS_NAMES.get(int(cid), f"class_{cid}"): round(float((cnt / total_px) * 100.0), 2)
        for cid, cnt in zip(unique_ids, counts)
    }

    result_data = {
        "sample": stem,
        "resolution": f"{w}x{h}",
        "pipeline_timing": {
            "onnx_inference_ms": round(onnx_infer_ms, 2),
            "proposal_engine_ms": round(prop_ms, 2),
            "total_pipeline_ms": round(total_pipeline_ms, 2),
        },
        "class_distribution": class_dist,
        "footstep_proposals": {
            "total_candidates_found": len(top_cands),
            "proposals_evaluated": prop_res["summary"]["total_proposals_evaluated"],
            "valid_proposals": prop_res["summary"]["valid_proposals"],
            "top_candidate": top_candidate.to_dict() if top_candidate else None,
            "all_candidates": candidates_data,
        },
        "artifacts": {
            "original": str(orig_path),
            "seg_overlay": str(overlay_path),
            "footstep_candidates": str(footsteps_path),
            "composite": str(composite_path),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    return result_data


def main() -> None:
    parser = argparse.ArgumentParser(description="StepView ONNX Footstep Proposal Pipeline")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/stepview_segmentation.onnx"),
        help="Path to exported ONNX segmentation model",
    )
    parser.add_argument("--image", type=Path, default=None, help="Path to single input image")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory containing images")
    parser.add_argument("--limit", type=int, default=5, help="Number of images to process")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/visualizations/onnx_footsteps"),
        help="Directory to save output visualizations and JSON",
    )
    parser.add_argument("--alpha", type=float, default=0.40, help="Segmentation overlay alpha")
    parser.add_argument("--max-candidates", type=int, default=5, help="Max footstep candidates to keep")
    args = parser.parse_args()

    print(f"Loading ONNX model from: {args.model}...")
    segmenter = ONNXTerrainSegmenter(args.model)

    print("Initializing FootstepProposalEngine...")
    proposal_engine = FootstepProposalEngine(top_k=args.max_candidates)

    image_paths: List[Path] = []
    if args.image is not None and args.image.is_file():
        image_paths.append(args.image)
    elif args.image_dir is not None and args.image_dir.is_dir():
        exts = {".png", ".jpg", ".jpeg"}
        files = sorted([p for p in args.image_dir.iterdir() if p.suffix.lower() in exts])
        image_paths.extend(files[:args.limit])
    else:
        # Default: 5 diverse terrain images across distinct scenes
        data_images = Path("data/images")
        target_scenes = ["trail-4", "trail-7", "trail_", "park-1", "village"]
        for sc in target_scenes:
            match = next((f for f in data_images.glob("*.png") if sc in f.name and f not in image_paths), None)
            if match:
                image_paths.append(match)

        # Fallback if fewer than limit found
        if len(image_paths) < args.limit:
            all_imgs = sorted(list(data_images.glob("*.png")))
            for img in all_imgs:
                if img not in image_paths and len(image_paths) < args.limit:
                    image_paths.append(img)

    print(f"\nExecuting ONNX Footstep Proposal Pipeline on {len(image_paths)} image(s)...")
    for img_p in image_paths:
        res = process_image(img_p, segmenter, proposal_engine, args.output_dir, alpha=args.alpha)
        props = res["footstep_proposals"]
        timing = res["pipeline_timing"]
        print(f"\n--- {res['sample']} ({res['resolution']}) ---")
        print(f"  Timing: ONNX {timing['onnx_inference_ms']:.1f}ms + Proposal {timing['proposal_engine_ms']:.1f}ms = {timing['total_pipeline_ms']:.1f}ms total")
        print(f"  Proposals: {props['total_candidates_found']} candidate(s) (from {props['valid_proposals']} valid ground proposals)")
        top = props["top_candidate"]
        if top:
            print(f"  Top Candidate #1: Score={top['score']:.2f}, Center=({top['x_center']:.1f}, {top['y_center']:.1f}), Clearance={top['obstacle_clearance_px']:.1f}px")
        else:
            print("  Top Candidate: [NO SAFE FOOTSTEP FOUND — Hazard Refusal]")

    print(f"\nPipeline execution complete. All artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
