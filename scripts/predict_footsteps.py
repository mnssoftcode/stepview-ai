#!/usr/bin/env python3
"""StepView — Footstep Placement Candidate Prediction & Visualization Tool.

Runs semantic segmentation inference and the 2D Footstep Candidate Engine on input
images to generate:
- Original image
- Segmentation overlay
- Candidate footstep regions
- Highlighted top-ranked footstep candidate
- Diagnostic composite panel and candidate metadata JSON

Usage:
    python scripts/predict_footsteps.py \
        --checkpoint experiments/runs/dataset_b_baseline/best_checkpoint.pt \
        --image data/images/rugd_trail_frame00001.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import torch

from stepview.data.schema import CLASS_NAMES, CLASS_PALETTE, NUM_CLASSES, TerrainClass
from stepview.geometry.footstep_proposals import FootstepCandidate, FootstepProposalEngine
from stepview.inference.segmenter import TerrainSegmenter


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
    draw_all_ranks: bool = True,
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
            # Bright Gold / Amber for top candidate
            fill_color = (0, 215, 255)  # BGR gold
            border_color = (0, 230, 255)
            border_thick = 3
            alpha = 0.40
        else:
            # Soft Emerald Green for secondary alternatives
            fill_color = (80, 200, 80)
            border_color = (0, 255, 128)
            border_thick = 2
            alpha = 0.25

        # Filled translucent ellipse
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
            (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

            bx0 = center[0] - tw // 2 - 6
            by0 = center[1] - axes[1] - th - 10
            bx1 = bx0 + tw + 12
            by1 = by0 + th + 8

            # Ensure badge stays inside image
            if by0 < 10:
                by0 = center[1] + axes[1] + 6
                by1 = by0 + th + 8

            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (20, 20, 20), -1)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (0, 220, 255), 1)
            cv2.putText(canvas, badge_text, (bx0 + 6, by1 - 5), font, font_scale, (0, 230, 255), thickness, cv2.LINE_AA)
        else:
            border_color = (0, 255, 128)
            cv2.ellipse(canvas, center, axes, 0.0, 0.0, 360.0, border_color, 2)

            # Small rank label
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


def create_legend_banner(width: int, height: int = 36) -> np.ndarray:
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
    color_mask: np.ndarray,
    overlay_bgr: np.ndarray,
    footsteps_bgr: np.ndarray,
    sample_name: str,
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

    cv2.putText(panel, f"StepView RGB Input: {sample_name}", (12, 26), font, 0.52, color, 1, cv2.LINE_AA)
    cv2.putText(panel, "Terrain Segmentation Overlay (50% Alpha)", (w + 12, 26), font, 0.52, (100, 255, 100), 1, cv2.LINE_AA)

    top_str = f"Top Score: {top_candidate.score:.2f} (Clr: {top_candidate.obstacle_clearance_px:.0f}px)" if top_candidate else "No Safe Footstep Found"
    cv2.putText(panel, f"2D Footstep Proposals ({num_candidates} found) | {top_str}", (2 * w + 12, 26), font, 0.50, (0, 220, 255), 1, cv2.LINE_AA)

    legend = create_legend_banner(total_w, height=34)
    full_composite = np.vstack([panel, legend])

    return full_composite


def process_image(
    image_path: Path,
    segmenter: TerrainSegmenter,
    proposal_engine: FootstepProposalEngine,
    output_dir: Path,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """Execute segmentation, footstep proposal, visualization, and JSON serialization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 1. Model inference
    seg_res = segmenter.segment(img_rgb)
    class_mask = seg_res["class_mask"]
    conf_map = seg_res["confidence_map"]
    probs = seg_res["probabilities"]

    # 2. Footstep proposal generation
    prop_res = proposal_engine.propose_footsteps(
        class_mask=class_mask,
        confidence_map=conf_map,
        probabilities=probs,
    )
    top_cands: List[FootstepCandidate] = prop_res["top_candidates"]
    top_candidate = top_cands[0] if top_cands else None

    # 3. Create visual overlays
    color_mask_rgb = colorize_mask(class_mask)
    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)

    overlay_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, color_mask_bgr, alpha, 0)

    # Footstep proposals rendered on top of subtle terrain overlay
    footsteps_bgr = draw_footstep_candidates(overlay_bgr, top_cands)

    # 4. Save individual artifact images
    orig_path = output_dir / f"{stem}_original.png"
    overlay_path = output_dir / f"{stem}_seg_overlay.png"
    footsteps_path = output_dir / f"{stem}_candidates.png"
    composite_path = output_dir / f"{stem}_footstep_composite.png"
    json_path = output_dir / f"{stem}_candidates.json"

    cv2.imwrite(str(orig_path), img_bgr)
    cv2.imwrite(str(overlay_path), overlay_bgr)
    cv2.imwrite(str(footsteps_path), footsteps_bgr)

    # 5. Composite Panel
    composite = generate_footstep_composite_panel(
        image_rgb=img_rgb,
        color_mask=color_mask_rgb,
        overlay_bgr=overlay_bgr,
        footsteps_bgr=footsteps_bgr,
        sample_name=stem,
        top_candidate=top_candidate,
        num_candidates=len(top_cands),
    )
    cv2.imwrite(str(composite_path), composite)

    # 6. Save JSON candidate summary
    summary_data = {
        "sample": stem,
        "resolution": f"{w}x{h}",
        "summary": prop_res["summary"],
        "top_candidates": [c.to_dict() for c in top_cands],
        "all_candidates_count": len(prop_res["all_candidates"]),
        "valid_candidates_count": prop_res["summary"]["valid_proposals"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return {
        "sample": stem,
        "resolution": f"{w}x{h}",
        "top_candidates_count": len(top_cands),
        "top_candidate": top_candidate.to_dict() if top_candidate else None,
        "artifacts": {
            "original": str(orig_path),
            "seg_overlay": str(overlay_path),
            "candidates": str(footsteps_path),
            "composite": str(composite_path),
            "json": str(json_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="StepView 2D Footstep Candidate Prediction")
    parser.add_argument("--image", type=Path, default=None, help="Path to single image file")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory containing images to process")
    parser.add_argument("--limit", type=int, default=5, help="Number of images to process")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/runs/dataset_b_baseline/best_checkpoint.pt"),
        help="Path to trained segmentation model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/visualizations/footsteps"),
        help="Output directory for footstep visualizations",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates to propose")
    parser.add_argument("--min-clearance", type=float, default=12.0, help="Minimum obstacle clearance in pixels")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'mps')")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        dev = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint} on {dev}...")
    segmenter = TerrainSegmenter.load_from_checkpoint(args.checkpoint, device=dev)
    proposal_engine = FootstepProposalEngine(
        top_k=args.top_k,
        min_clearance_px=args.min_clearance,
    )

    # Collect images
    image_paths: List[Path] = []
    if args.image is not None and args.image.is_file():
        image_paths.append(args.image)
    elif args.image_dir is not None and args.image_dir.is_dir():
        exts = {".png", ".jpg", ".jpeg"}
        files = sorted([p for p in args.image_dir.iterdir() if p.suffix.lower() in exts])
        image_paths.extend(files[:args.limit])
    else:
        # Default: select 5 representative images across distinct scenes
        data_images = Path("data/images")
        if data_images.is_dir():
            files = sorted(list(data_images.glob("*.png")))
            # Pick 5 distinct scenes
            target_scenes = ["trail-10", "village", "park-1", "creek", "trail-11"]
            for sc in target_scenes:
                match = next((f for f in files if sc in f.name and f not in image_paths), None)
                if match:
                    image_paths.append(match)
            if len(image_paths) < 5:
                for f in files:
                    if f not in image_paths:
                        image_paths.append(f)
                    if len(image_paths) >= 5:
                        break

    if not image_paths:
        print("No images found to process. Please specify --image or --image-dir.")
        return

    print(f"\nEvaluating Footstep Proposal Engine on {len(image_paths)} image(s)...")
    for img_p in image_paths:
        res = process_image(img_p, segmenter, proposal_engine, args.output_dir)
        print(f"\n================================================================================")
        print(f"Sample: {res['sample']} ({res['resolution']})")
        print(f"  Top Candidates Found: {res['top_candidates_count']}")
        if res["top_candidate"]:
            top = res["top_candidate"]
            print(f"  TOP CANDIDATE (#1):")
            print(f"    - Center (x, y):     ({top['x_center']}, {top['y_center']})")
            print(f"    - Footprint (w x h): {top['width']} x {top['height']} px")
            print(f"    - Composite Score:   {top['score']:.4f}")
            print(f"    - Ground Support:    {top['ground_confidence']*100:.1f}%")
            print(f"    - Obstacle Clr:      {top['obstacle_clearance_px']:.1f} px")
            print(f"    - Model Confidence:  {top['model_confidence']*100:.1f}%")
        else:
            print("  NO SAFE CANDIDATE PROPOSALS FOUND (Terrain obstructed or uncertain)")
        print(f"  Artifacts saved:")
        print(f"    * Footstep Panel: {res['artifacts']['candidates']}")
        print(f"    * Composite:      {res['artifacts']['composite']}")
        print(f"    * Candidate JSON: {res['artifacts']['json']}")

    print(f"\nProcessing complete. All results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
