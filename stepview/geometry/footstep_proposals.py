"""StepView — 2D Footstep Candidate Engine.

Heuristic candidate foot-placement proposal generator operating on
semantic segmentation masks and confidence maps.

Proposes supportable ground footprints, filters out hazards and tight clearances,
and scores prospective foot placements according to terrain stability, clearance,
and walking progression.

NOTE: This is a prototype heuristic generator and NOT a physical safety guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from stepview.data.schema import TerrainClass


@dataclass
class FootstepCandidate:
    """Represents a prospective 2D foot-placement location."""

    x_center: float
    y_center: float
    width: float
    height: float
    shape: str = "ellipse"
    score: float = 0.0
    ground_confidence: float = 0.0
    model_confidence: float = 0.0
    obstacle_clearance_px: float = 0.0
    compactness: float = 0.0
    area_px: float = 0.0
    distance_from_bottom_norm: float = 0.0
    is_valid: bool = True
    rejection_reason: Optional[str] = None
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return self.ground_confidence

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate properties to dict."""
        return {
            "rank": self.rank,
            "x_center": round(self.x_center, 1),
            "y_center": round(self.y_center, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "shape": self.shape,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "ground_confidence": round(self.ground_confidence, 4),
            "model_confidence": round(self.model_confidence, 4),
            "obstacle_clearance_px": round(self.obstacle_clearance_px, 1),
            "compactness": round(self.compactness, 4),
            "area_px": round(self.area_px, 1),
            "distance_from_bottom_norm": round(self.distance_from_bottom_norm, 4),
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
        }


class FootstepProposalEngine:
    """Proposes and scores prospective 2D foot placements on segmented terrain."""

    def __init__(
        self,
        min_clearance_px: float = 12.0,
        ideal_clearance_px: float = 40.0,
        min_ground_support_ratio: float = 0.85,
        max_hazard_ratio: float = 0.02,
        min_component_area_px: int = 400,
        lookahead_min_y_ratio: float = 0.35,
        lookahead_max_y_ratio: float = 0.95,
        near_footprint_size: Tuple[float, float] = (70.0, 42.0),
        far_footprint_size: Tuple[float, float] = (36.0, 22.0),
        nms_iou_threshold: float = 0.35,
        top_k: int = 5,
    ) -> None:
        """Initialize footstep proposal engine.

        Args:
            min_clearance_px: Hard minimum distance to any hazard pixel.
            ideal_clearance_px: Distance threshold where clearance score saturates at 1.0.
            min_ground_support_ratio: Minimum fraction of footprint that must be GROUND.
            max_hazard_ratio: Maximum allowed hazard fraction within footprint.
            min_component_area_px: Minimum connected ground component area.
            lookahead_min_y_ratio: Top horizon bound (fraction of image height).
            lookahead_max_y_ratio: Bottom camera cutoff (fraction of image height).
            near_footprint_size: (width, height) in pixels near bottom of image.
            far_footprint_size: (width, height) in pixels near horizon.
            nms_iou_threshold: IoU overlap threshold for Non-Maximum Suppression.
            top_k: Maximum number of top candidates to return.
        """
        self.min_clearance_px = min_clearance_px
        self.ideal_clearance_px = ideal_clearance_px
        self.min_ground_support_ratio = min_ground_support_ratio
        self.max_hazard_ratio = max_hazard_ratio
        self.min_component_area_px = min_component_area_px
        self.lookahead_min_y_ratio = lookahead_min_y_ratio
        self.lookahead_max_y_ratio = lookahead_max_y_ratio
        self.near_footprint_size = near_footprint_size
        self.far_footprint_size = far_footprint_size
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k

    def get_perspective_footprint_size(self, y: float, image_h: int) -> Tuple[float, float]:
        """Compute expected footprint pixel dimensions at vertical row y."""
        y_norm = np.clip(y / float(image_h), self.lookahead_min_y_ratio, self.lookahead_max_y_ratio)
        t = (y_norm - self.lookahead_min_y_ratio) / (self.lookahead_max_y_ratio - self.lookahead_min_y_ratio + 1e-6)

        w_far, h_far = self.far_footprint_size
        w_near, h_near = self.near_footprint_size

        width = w_far + t * (w_near - w_far)
        height = h_far + t * (h_near - h_far)
        return float(width), float(height)

    def extract_hazard_distance_map(self, class_mask: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance transform from all hazard classes.

        Hazards include: OBSTACLE (2), HOLE (4), UNCERTAIN_SURFACE (5).
        """
        hazard_mask = np.isin(
            class_mask,
            [
                TerrainClass.OBSTACLE.value,
                TerrainClass.HOLE.value,
                TerrainClass.UNCERTAIN_SURFACE.value,
            ],
        ).astype(np.uint8)

        # Distance transform requires non-zero pixels where distance is measured
        non_hazard = (1 - hazard_mask).astype(np.uint8)
        dist_map = cv2.distanceTransform(non_hazard, cv2.DIST_L2, 5)
        return dist_map

    def extract_safe_ground_components(
        self,
        class_mask: np.ndarray,
        dist_map: np.ndarray,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Isolate contiguous ground patches with sufficient clearance from hazards."""
        ground_mask = (class_mask == TerrainClass.GROUND.value).astype(np.uint8)

        # Enforce minimum clearance
        safe_ground = ground_mask & (dist_map >= self.min_clearance_px).astype(np.uint8)

        # Morphological opening to eliminate isolated pixel spurs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(safe_ground, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        valid_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_component_area_px:
                comp_mask = (labels == i)
                valid_components.append({
                    "id": i,
                    "area": area,
                    "centroid": centroids[i],
                    "bbox": (
                        stats[i, cv2.CC_STAT_LEFT],
                        stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH],
                        stats[i, cv2.CC_STAT_HEIGHT],
                    ),
                    "mask": comp_mask,
                })

        return cleaned, valid_components

    def generate_candidate_footprint_mask(
        self,
        image_h: int,
        image_w: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Rasterize a candidate foot ellipse mask."""
        x0 = max(0, int(round(x_center - width / 2.0)))
        x1 = min(image_w, int(round(x_center + width / 2.0)))
        y0 = max(0, int(round(y_center - height / 2.0)))
        y1 = min(image_h, int(round(y_center + height / 2.0)))

        foot_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        axes = (max(1, int(round(width / 2.0))), max(1, int(round(height / 2.0))))
        center = (int(round(x_center)), int(round(y_center)))
        cv2.ellipse(foot_mask, center, axes, 0.0, 0.0, 360.0, 1, -1)

        return foot_mask, (x0, y0, x1, y1)

    def evaluate_candidate(
        self,
        x: float,
        y: float,
        class_mask: np.ndarray,
        dist_map: np.ndarray,
        confidence_map: Optional[np.ndarray],
        probabilities: Optional[np.ndarray],
    ) -> FootstepCandidate:
        """Evaluate and score a candidate location."""
        h, w = class_mask.shape
        cand_w, cand_h = self.get_perspective_footprint_size(y, h)
        area_px = np.pi * (cand_w / 2.0) * (cand_h / 2.0)
        dist_from_bottom = (h - y) / float(h)

        candidate = FootstepCandidate(
            x_center=x,
            y_center=y,
            width=cand_w,
            height=cand_h,
            area_px=area_px,
            distance_from_bottom_norm=dist_from_bottom,
        )

        # 1. Lookahead range check
        if y < self.lookahead_min_y_ratio * h:
            candidate.is_valid = False
            candidate.rejection_reason = "beyond_lookahead_horizon"
            return candidate

        if y > self.lookahead_max_y_ratio * h:
            candidate.is_valid = False
            candidate.rejection_reason = "too_close_to_camera_edge"
            return candidate

        # 2. Rasterize footprint
        foot_mask, (x0, y0, x1, y1) = self.generate_candidate_footprint_mask(h, w, x, y, cand_w, cand_h)
        foot_pixels = foot_mask > 0
        num_foot_px = int(np.sum(foot_pixels))

        if num_foot_px == 0:
            candidate.is_valid = False
            candidate.rejection_reason = "empty_footprint"
            return candidate

        # 3. Ground support evaluation
        ground_px = np.sum((class_mask == TerrainClass.GROUND.value) & foot_pixels)
        ground_support = ground_px / float(num_foot_px)
        candidate.ground_confidence = float(ground_support)

        if probabilities is not None and probabilities.shape[0] > TerrainClass.GROUND.value:
            ground_probs = probabilities[TerrainClass.GROUND.value]
            mean_ground_prob = float(np.mean(ground_probs[foot_pixels]))
        else:
            mean_ground_prob = float(ground_support)

        # 4. Hazard presence evaluation
        hazards = np.isin(
            class_mask,
            [
                TerrainClass.OBSTACLE.value,
                TerrainClass.HOLE.value,
                TerrainClass.UNCERTAIN_SURFACE.value,
            ],
        )
        hazard_px = np.sum(hazards & foot_pixels)
        hazard_ratio = hazard_px / float(num_foot_px)

        if hazard_ratio > self.max_hazard_ratio:
            candidate.is_valid = False
            candidate.rejection_reason = f"hazard_overlap ({hazard_ratio*100:.1f}%)"
            return candidate

        if ground_support < self.min_ground_support_ratio:
            candidate.is_valid = False
            candidate.rejection_reason = f"insufficient_ground ({ground_support*100:.1f}%)"
            return candidate

        # 5. Clearance evaluation
        min_clearance = float(np.min(dist_map[foot_pixels]))
        candidate.obstacle_clearance_px = min_clearance

        if min_clearance < self.min_clearance_px:
            candidate.is_valid = False
            candidate.rejection_reason = f"low_obstacle_clearance ({min_clearance:.1f}px)"
            return candidate

        # 6. Model confidence
        if confidence_map is not None:
            model_conf = float(np.mean(confidence_map[foot_pixels]))
        else:
            model_conf = float(ground_support)
        candidate.model_confidence = model_conf

        # 7. Local Compactness (measure ratio of ground in bounding box)
        bbox_region = class_mask[y0:y1, x0:x1]
        compactness = float(np.mean(bbox_region == TerrainClass.GROUND.value))
        candidate.compactness = compactness

        # 8. Score Computation
        # Clearance score [0, 1]
        score_clearance = np.clip(min_clearance / self.ideal_clearance_px, 0.0, 1.0)

        # Walking progression preference: sweet spot is 1-2 strides ahead (y around 0.60 - 0.85 * h)
        optimal_y = 0.72 * h
        y_dist = abs(y - optimal_y) / (0.35 * h)
        score_progression = float(np.clip(1.0 - y_dist, 0.0, 1.0))

        # Ground quality score
        score_ground = 0.5 * ground_support + 0.5 * mean_ground_prob

        # Composite score
        total_score = (
            0.35 * score_ground
            + 0.25 * score_clearance
            + 0.15 * score_progression
            + 0.15 * model_conf
            + 0.10 * compactness
        )
        candidate.score = float(total_score)
        candidate.is_valid = True
        candidate.rejection_reason = None

        return candidate

    def compute_iou(self, c1: FootstepCandidate, c2: FootstepCandidate) -> float:
        """Compute 2D bounding box intersection over union."""
        x1_min = c1.x_center - c1.width / 2.0
        x1_max = c1.x_center + c1.width / 2.0
        y1_min = c1.y_center - c1.height / 2.0
        y1_max = c1.y_center + c1.height / 2.0

        x2_min = c2.x_center - c2.width / 2.0
        x2_max = c2.x_center + c2.width / 2.0
        y2_min = c2.y_center - c2.height / 2.0
        y2_max = c2.y_center + c2.height / 2.0

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area1 = c1.width * c1.height
        area2 = c2.width * c2.height
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    def apply_non_maximum_suppression(
        self,
        candidates: List[FootstepCandidate],
    ) -> List[FootstepCandidate]:
        """Eliminate overlapping candidates to provide spatial diversity."""
        sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
        selected: List[FootstepCandidate] = []

        for cand in sorted_cands:
            overlap = False
            for prev in selected:
                if self.compute_iou(cand, prev) > self.nms_iou_threshold:
                    overlap = True
                    break
            if not overlap:
                selected.append(cand)
                if len(selected) >= self.top_k:
                    break

        # Assign ranking
        for idx, item in enumerate(selected):
            item.rank = idx + 1

        return selected

    def propose_footsteps(
        self,
        class_mask: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Propose and rank candidate foot placements.

        Args:
            class_mask: 2D uint8 segmentation mask with class values (0..5).
            confidence_map: Optional 2D float32 prediction probability map.
            probabilities: Optional 3D float32 (num_classes, H, W) probabilities.

        Returns:
            Dict containing:
                - 'top_candidates': Sorted list of valid, non-overlapping FootstepCandidates.
                - 'all_candidates': Complete list of evaluated candidates (including rejected).
                - 'usable_ground_mask': 2D boolean/uint8 mask of safe ground components.
                - 'hazard_distance_map': 2D float32 Euclidean distance transform to hazards.
                - 'summary': High-level statistics.
        """
        h, w = class_mask.shape
        dist_map = self.extract_hazard_distance_map(class_mask)
        safe_ground_mask, components = self.extract_safe_ground_components(class_mask, dist_map)

        all_candidates: List[FootstepCandidate] = []

        # 1. Sample candidate seed locations:
        # A: Local maxima of safe ground distance transform (deep inside open terrain)
        # B: Dense grid sampling across safe ground areas
        stride_y = max(16, int(h * 0.05))
        stride_x = max(16, int(w * 0.05))

        y_range = range(int(h * self.lookahead_min_y_ratio), int(h * self.lookahead_max_y_ratio), stride_y)
        x_range = range(int(w * 0.10), int(w * 0.90), stride_x)

        for y in y_range:
            for x in x_range:
                # Fast pre-check: pixel must be within safe ground
                if safe_ground_mask[y, x] == 0:
                    continue

                candidate = self.evaluate_candidate(
                    x=float(x),
                    y=float(y),
                    class_mask=class_mask,
                    dist_map=dist_map,
                    confidence_map=confidence_map,
                    probabilities=probabilities,
                )
                all_candidates.append(candidate)

        # Also sample component centroids if valid
        for comp in components:
            cx, cy = comp["centroid"]
            cx_int = int(round(cx))
            cy_int = int(round(cy))
            if 0 <= cy_int < h and 0 <= cx_int < w:
                candidate = self.evaluate_candidate(
                    x=float(cx),
                    y=float(cy),
                    class_mask=class_mask,
                    dist_map=dist_map,
                    confidence_map=confidence_map,
                    probabilities=probabilities,
                )
                all_candidates.append(candidate)

        valid_candidates = [c for c in all_candidates if c.is_valid]
        top_candidates = self.apply_non_maximum_suppression(valid_candidates)

        return {
            "top_candidates": top_candidates,
            "all_candidates": all_candidates,
            "usable_ground_mask": safe_ground_mask,
            "hazard_distance_map": dist_map,
            "summary": {
                "total_proposals_evaluated": len(all_candidates),
                "valid_proposals": len(valid_candidates),
                "top_selected": len(top_candidates),
                "top_score": top_candidates[0].score if top_candidates else 0.0,
            },
        }
