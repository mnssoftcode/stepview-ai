"""StepView Terrain Segmentation & Dataset Schema.

Defines explicit class constants, color maps, validation rules,
and metadata schemas for terrain segmentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set
import numpy as np


class TerrainClass(IntEnum):
    """Semantic terrain class IDs for StepView segmentation."""
    BACKGROUND = 0        # Non-ground, sky, horizon, distant objects, background
    GROUND = 1            # Walkable ground, trail surface, pathway
    OBSTACLE = 2          # Large rocks, boulders, logs, tree trunks, raised roots, debris
    VEGETATION = 3        # Dense brush, thick grass, shrubs, overhanging branches
    HOLE = 4              # Pit, drop-off, trench, steep cavity, crevasse
    UNCERTAIN_SURFACE = 5 # Standing water, deep mud, ice, slippery algae


# Total number of segmentation classes
NUM_CLASSES: int = len(TerrainClass)

# Valid integer class identifiers
VALID_CLASS_IDS: Set[int] = {c.value for c in TerrainClass}

# Human-readable labels
CLASS_NAMES: Dict[int, str] = {
    TerrainClass.BACKGROUND.value: "background",
    TerrainClass.GROUND.value: "ground",
    TerrainClass.OBSTACLE.value: "obstacle",
    TerrainClass.VEGETATION.value: "vegetation",
    TerrainClass.HOLE.value: "hole",
    TerrainClass.UNCERTAIN_SURFACE.value: "uncertain_surface",
}

# Standard visualization RGB palette (uint8)
CLASS_PALETTE: Dict[int, tuple[int, int, int]] = {
    TerrainClass.BACKGROUND.value: (0, 0, 0),        # Black
    TerrainClass.GROUND.value: (46, 204, 113),       # Green
    TerrainClass.OBSTACLE.value: (231, 76, 60),      # Red
    TerrainClass.VEGETATION.value: (39, 174, 96),    # Dark Green
    TerrainClass.HOLE.value: (142, 68, 173),         # Purple
    TerrainClass.UNCERTAIN_SURFACE.value: (241, 196, 15), # Amber/Yellow
}


@dataclass
class SampleMetadata:
    """Metadata container for a single dataset sample."""
    sample_id: str
    image_rel_path: str
    mask_rel_path: str
    scene_id: str
    split: str = "train"
    camera_pitch_deg: Optional[float] = None
    camera_height_m: Optional[float] = None
    terrain_type: Optional[str] = None
    lighting_condition: Optional[str] = None
    weather: Optional[str] = None
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SampleMetadata:
        """Construct metadata instance from serialized dictionary."""
        return cls(
            sample_id=data["sample_id"],
            image_rel_path=data["image_rel_path"],
            mask_rel_path=data["mask_rel_path"],
            scene_id=data["scene_id"],
            split=data.get("split", "train"),
            camera_pitch_deg=data.get("camera_pitch_deg"),
            camera_height_m=data.get("camera_height_m"),
            terrain_type=data.get("terrain_type"),
            lighting_condition=data.get("lighting_condition"),
            weather=data.get("weather"),
            extra_attributes=data.get("extra_attributes", {}),
        )


def validate_mask_classes(mask: np.ndarray) -> Set[int]:
    """Validate that all pixel values in the mask correspond to valid TerrainClass IDs.

    Args:
        mask: 2D numpy array containing integer class labels.

    Returns:
        Set of unique class IDs present in the mask.

    Raises:
        ValueError: If unexpected/unsupported class IDs are detected.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be a 2D single-channel array, got shape {mask.shape}")

    unique_classes = set(np.unique(mask).tolist())
    invalid_classes = unique_classes - VALID_CLASS_IDS

    if invalid_classes:
        raise ValueError(
            f"Invalid class IDs found in segmentation mask: {sorted(list(invalid_classes))}. "
            f"Expected only values from {sorted(list(VALID_CLASS_IDS))}."
        )

    return unique_classes
