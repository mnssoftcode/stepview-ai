"""Unit tests for StepView 2D Footstep Candidate Engine.

Validates candidate footprint generation, distance transform clearance,
hazard filtering, scoring logic, and non-maximum suppression.
"""

import numpy as np
import pytest

from stepview.data.schema import TerrainClass
from stepview.geometry.footstep_proposals import FootstepCandidate, FootstepProposalEngine


def test_footstep_candidate_data_model() -> None:
    """Verify FootstepCandidate serialization and properties."""
    cand = FootstepCandidate(
        x_center=150.5,
        y_center=320.0,
        width=60.0,
        height=35.0,
        score=0.8421,
        ground_confidence=0.95,
        obstacle_clearance_px=24.5,
        compactness=0.88,
        area_px=1649.3,
    )
    d = cand.to_dict()

    assert d["x_center"] == 150.5
    assert d["y_center"] == 320.0
    assert d["width"] == 60.0
    assert d["height"] == 35.0
    assert d["score"] == 0.8421
    assert d["confidence"] == 0.95
    assert d["is_valid"] is True
    assert d["rejection_reason"] is None


def test_perspective_scaling() -> None:
    """Verify that candidate footprints scale up near the bottom of the image."""
    engine = FootstepProposalEngine(
        near_footprint_size=(80.0, 45.0),
        far_footprint_size=(40.0, 20.0),
    )
    image_h = 500

    w_far, h_far = engine.get_perspective_footprint_size(y=180.0, image_h=image_h)
    w_near, h_near = engine.get_perspective_footprint_size(y=450.0, image_h=image_h)

    assert w_near > w_far
    assert h_near > h_far
    assert 40.0 <= w_far < 55.0
    assert 70.0 < w_near <= 80.0


def test_hazard_clearance_and_distance_map() -> None:
    """Verify that hazard distance map accurately reflects proximity to obstacle pixels."""
    engine = FootstepProposalEngine(min_clearance_px=10.0)

    # 100x100 mask: all ground except a single obstacle column at x=50
    mask = np.full((100, 100), TerrainClass.GROUND.value, dtype=np.uint8)
    mask[:, 50] = TerrainClass.OBSTACLE.value

    dist_map = engine.extract_hazard_distance_map(mask)

    assert dist_map[:, 50].max() == 0.0
    assert dist_map[:, 40].min() == pytest.approx(10.0, abs=0.5)
    assert dist_map[:, 30].min() == pytest.approx(20.0, abs=0.5)


def test_candidate_rejection_on_hazard_overlap() -> None:
    """Verify that candidate footprints touching obstacles or holes are rejected."""
    engine = FootstepProposalEngine(min_ground_support_ratio=0.85, max_hazard_ratio=0.02)

    # 200x200 mask: Ground with an obstacle patch at center (100, 100)
    mask = np.full((200, 200), TerrainClass.GROUND.value, dtype=np.uint8)
    mask[90:110, 90:110] = TerrainClass.OBSTACLE.value

    dist_map = engine.extract_hazard_distance_map(mask)

    # Candidate directly over obstacle
    cand_bad = engine.evaluate_candidate(
        x=100.0,
        y=100.0,
        class_mask=mask,
        dist_map=dist_map,
        confidence_map=None,
        probabilities=None,
    )
    assert cand_bad.is_valid is False
    assert "hazard_overlap" in cand_bad.rejection_reason or "insufficient_ground" in cand_bad.rejection_reason

    # Candidate safely to the side with plenty of clearance
    cand_good = engine.evaluate_candidate(
        x=30.0,
        y=140.0,
        class_mask=mask,
        dist_map=dist_map,
        confidence_map=None,
        probabilities=None,
    )
    assert cand_good.is_valid is True
    assert cand_good.score > 0.0


def test_propose_footsteps_end_to_end() -> None:
    """Verify end-to-end proposal generation and ranking on simulated terrain."""
    engine = FootstepProposalEngine(top_k=3, min_clearance_px=8.0)

    # 300x300 image: Top is sky/background, bottom is broad ground with an obstacle on left
    mask = np.full((300, 300), TerrainClass.GROUND.value, dtype=np.uint8)
    mask[:100, :] = TerrainClass.BACKGROUND.value  # Sky
    mask[150:280, 20:60] = TerrainClass.OBSTACLE.value  # Rock on left side

    res = engine.propose_footsteps(mask)

    top_cands = res["top_candidates"]
    assert len(top_cands) > 0
    assert len(top_cands) <= 3

    # Rank 1 candidate must have highest score
    assert top_cands[0].rank == 1
    if len(top_cands) > 1:
        assert top_cands[0].score >= top_cands[1].score

    # All top candidates must be clear of the rock obstacle (x > 70)
    for c in top_cands:
        assert c.is_valid is True
        assert c.obstacle_clearance_px >= 8.0
        assert c.x_center > 60  # Avoids rock on the left
