"""Unit tests for StepView dataset loader and schema using synthetic data.

These tests validate software correctness (I/O, tensor shapes, types,
validation, metadata parsing). They do NOT validate ML performance.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from stepview.data import (
    CLASS_NAMES,
    CLASS_PALETTE,
    NUM_CLASSES,
    SampleMetadata,
    StepViewDataset,
    TerrainClass,
    VALID_CLASS_IDS,
    validate_mask_classes,
)


def test_terrain_class_schema() -> None:
    """Verify class taxonomy invariants."""
    assert NUM_CLASSES == 6
    assert TerrainClass.BACKGROUND == 0
    assert TerrainClass.GROUND == 1
    assert TerrainClass.OBSTACLE == 2
    assert TerrainClass.VEGETATION == 3
    assert TerrainClass.HOLE == 4
    assert TerrainClass.UNCERTAIN_SURFACE == 5

    assert len(VALID_CLASS_IDS) == 6
    for class_id in range(6):
        assert class_id in VALID_CLASS_IDS
        assert class_id in CLASS_NAMES
        assert class_id in CLASS_PALETTE


def test_validate_mask_classes_valid() -> None:
    """Valid masks with allowed classes should pass validation."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = TerrainClass.GROUND
    mask[20:30, 20:30] = TerrainClass.OBSTACLE
    mask[30:40, 30:40] = TerrainClass.UNCERTAIN_SURFACE

    detected = validate_mask_classes(mask)
    assert detected == {
        TerrainClass.BACKGROUND,
        TerrainClass.GROUND,
        TerrainClass.OBSTACLE,
        TerrainClass.UNCERTAIN_SURFACE,
    }


def test_validate_mask_classes_invalid() -> None:
    """Masks with unexpected class IDs must raise ValueError."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:10, 5:10] = 99  # Invalid class ID

    with pytest.raises(ValueError, match="Invalid class IDs found"):
        validate_mask_classes(mask)


def test_validate_mask_classes_shape_error() -> None:
    """Masks with incorrect shape (e.g. 3D array) must raise ValueError."""
    mask_3d = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="2D single-channel"):
        validate_mask_classes(mask_3d)


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Generate a clean synthetic StepView dataset on disk for testing."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    metadata_dir = tmp_path / "metadata"
    splits_dir = tmp_path / "splits"

    for d in (images_dir, masks_dir, metadata_dir, splits_dir):
        d.mkdir(parents=True)

    samples_meta = []

    for i in range(4):
        sample_id = f"terrain_sample_{i:03d}"
        split = "train" if i < 3 else "val"

        # Generate synthetic 64x64 RGB image
        img = np.full((64, 64, 3), fill_value=(50 * (i + 1)) % 255, dtype=np.uint8)
        # Add diagonal line pattern
        cv2.line(img, (0, 0), (63, 63), (200, 200, 200), 2)
        img_filename = f"{sample_id}.png"
        cv2.imwrite(str(images_dir / img_filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Generate synthetic 64x64 single-channel mask
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:32, :] = TerrainClass.GROUND
        mask[32:48, :] = TerrainClass.OBSTACLE
        mask[48:, :] = TerrainClass.UNCERTAIN_SURFACE if i % 2 == 0 else TerrainClass.HOLE
        mask_filename = f"{sample_id}.png"
        cv2.imwrite(str(masks_dir / mask_filename), mask)

        meta = SampleMetadata(
            sample_id=sample_id,
            image_rel_path=f"images/{img_filename}",
            mask_rel_path=f"masks/{mask_filename}",
            scene_id=f"scene_{i // 2}",
            split=split,
            camera_pitch_deg=40.0,
            camera_height_m=1.2,
            terrain_type="rocky_trail",
            lighting_condition="daylight",
        )
        samples_meta.append(meta.to_dict())

    # Write manifest
    with open(metadata_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples_meta}, f, indent=2)

    # Write split files
    with open(splits_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("terrain_sample_000\nterrain_sample_001\nterrain_sample_002\n")

    with open(splits_dir / "val.txt", "w", encoding="utf-8") as f:
        f.write("terrain_sample_003\n")

    return tmp_path


def test_dataset_loader_with_manifest_and_splits(synthetic_dataset_dir: Path) -> None:
    """Verify that StepViewDataset correctly filters by split and reads manifests."""
    train_dataset = StepViewDataset(root_dir=synthetic_dataset_dir, split="train")
    assert len(train_dataset) == 3

    val_dataset = StepViewDataset(root_dir=synthetic_dataset_dir, split="val")
    assert len(val_dataset) == 1

    all_dataset = StepViewDataset(root_dir=synthetic_dataset_dir, split="all")
    assert len(all_dataset) == 4


def test_dataset_getitem_contracts(synthetic_dataset_dir: Path) -> None:
    """Verify tensor types, shapes, and metadata in dataset __getitem__."""
    dataset = StepViewDataset(root_dir=synthetic_dataset_dir, split="train")
    item = dataset[0]

    # Verify keys
    assert "image" in item
    assert "mask" in item
    assert "sample_id" in item
    assert "scene_id" in item
    assert "metadata" in item

    # Verify Image Tensor
    image = item["image"]
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert image.shape == (3, 64, 64)
    assert 0.0 <= image.min() and image.max() <= 1.0

    # Verify Mask Tensor
    mask = item["mask"]
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.int64
    assert mask.shape == (64, 64)
    assert mask.min() >= 0 and mask.max() < NUM_CLASSES

    # Verify Metadata
    assert item["sample_id"] == "terrain_sample_000"
    assert item["scene_id"] == "scene_0"
    assert item["metadata"]["camera_pitch_deg"] == 40.0


def test_dataset_dataloader_batching(synthetic_dataset_dir: Path) -> None:
    """Verify that PyTorch DataLoader batches StepViewDataset samples seamlessly."""
    from stepview.data import stepview_collate_fn
    dataset = StepViewDataset(root_dir=synthetic_dataset_dir, split="train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=stepview_collate_fn)

    batch = next(iter(dataloader))
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["mask"].shape == (2, 64, 64)
    assert len(batch["sample_id"]) == 2
    assert len(batch["scene_id"]) == 2
    assert len(batch["metadata"]) == 2


def test_dataset_auto_discovery(tmp_path: Path) -> None:
    """Verify fallback auto-discovery of image-mask pairs without manifest."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Create 1 pair
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "test_pair.png"), img)
    cv2.imwrite(str(masks_dir / "test_pair.png"), mask)

    dataset = StepViewDataset(root_dir=tmp_path, split="all")
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["sample_id"] == "test_pair"
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (32, 32)


def test_dataset_shape_mismatch_error(tmp_path: Path) -> None:
    """Verify that mismatched spatial dimensions between image and mask trigger ValueError."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Image is 64x64, Mask is 32x32
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "mismatch.png"), img)
    cv2.imwrite(str(masks_dir / "mismatch.png"), mask)

    dataset = StepViewDataset(root_dir=tmp_path, split="all")
    with pytest.raises(ValueError, match="Shape mismatch"):
        _ = dataset[0]


def test_metadata_schema_serialization() -> None:
    """Verify SampleMetadata serialization to and from dictionary."""
    meta = SampleMetadata(
        sample_id="scene005_frame0012",
        scene_id="scene005",
        image_rel_path="images/scene005_frame0012.jpg",
        mask_rel_path="masks/scene005_frame0012.png",
        split="train",
        camera_pitch_deg=45.0,
        camera_height_m=1.25,
        terrain_type="rocky_dirt",
        lighting_condition="direct_sun",
        extra_attributes={"annotator": "agent"},
    )
    data_dict = meta.to_dict()
    assert data_dict["sample_id"] == "scene005_frame0012"
    assert data_dict["camera_pitch_deg"] == 45.0
    assert data_dict["extra_attributes"]["annotator"] == "agent"

    reconstructed = SampleMetadata.from_dict(data_dict)
    assert reconstructed.sample_id == meta.sample_id
    assert reconstructed.scene_id == meta.scene_id
    assert reconstructed.camera_pitch_deg == meta.camera_pitch_deg
    assert reconstructed.extra_attributes == meta.extra_attributes


def test_missing_image_file_raises(tmp_path: Path) -> None:
    """Verify that a missing image file raises FileNotFoundError."""
    dataset = StepViewDataset(root_dir=tmp_path, split="all")
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        dataset.load_image(tmp_path / "nonexistent_image.png")


def test_missing_mask_file_raises(tmp_path: Path) -> None:
    """Verify that a missing mask file raises FileNotFoundError."""
    dataset = StepViewDataset(root_dir=tmp_path, split="all")
    with pytest.raises(FileNotFoundError, match="Mask file not found"):
        dataset.load_mask(tmp_path / "nonexistent_mask.png")


def test_visualize_dataset_script_execution(synthetic_dataset_dir: Path, tmp_path: Path) -> None:
    """Verify that visualize_dataset script runs cleanly on valid synthetic dataset."""
    import sys
    from scripts.visualize_dataset import inspect_dataset

    save_dir = tmp_path / "viz_output"
    exit_code = inspect_dataset(
        data_dir=synthetic_dataset_dir,
        split="all",
        save_dir=save_dir,
        alpha=0.5,
        show_gui=False,
    )
    assert exit_code == 0

    # Verify that inspection panels were saved
    saved_panels = list(save_dir.glob("inspect_*.png"))
    assert len(saved_panels) == 4
    for panel_path in saved_panels:
        img = cv2.imread(str(panel_path))
        assert img is not None
        assert img.ndim == 3


def test_visualize_dataset_detects_invalid_class(synthetic_dataset_dir: Path) -> None:
    """Verify that visualize_dataset reports defects when invalid class IDs are present."""
    from scripts.visualize_dataset import inspect_dataset

    # Corrupt one mask with invalid class 42
    corrupt_mask = np.full((64, 64), 42, dtype=np.uint8)
    corrupt_mask_path = synthetic_dataset_dir / "masks" / "terrain_sample_000.png"
    cv2.imwrite(str(corrupt_mask_path), corrupt_mask)

    exit_code = inspect_dataset(
        data_dir=synthetic_dataset_dir,
        split="all",
        show_gui=False,
    )
    # Must detect issue and return non-zero
    assert exit_code == 1


def test_rugd_bootstrap_dataset_integrity() -> None:
    """Verify integrity of real converted RUGD bootstrap data."""
    data_dir = Path("data")
    manifest_file = data_dir / "metadata" / "manifest.json"
    if not manifest_file.is_file():
        pytest.skip("Bootstrap RUGD data not yet converted.")

    # Check splits
    train_ds = StepViewDataset(root_dir=data_dir, split="train")
    val_ds = StepViewDataset(root_dir=data_dir, split="val")
    test_ds = StepViewDataset(root_dir=data_dir, split="test")

    assert len(train_ds) == 16
    assert len(val_ds) == 5
    assert len(test_ds) == 5

    # Check sample item properties
    sample = train_ds[0]
    assert sample["image"].ndim == 3
    assert sample["image"].shape[0] == 3
    assert sample["mask"].ndim == 2
    assert sample["image"].shape[1:] == sample["mask"].shape
    assert sample["sample_id"].startswith("rugd_")

    # Verify no scene leakage across splits
    train_scenes = {s.scene_id for s in train_ds.samples}
    val_scenes = {s.scene_id for s in val_ds.samples}
    test_scenes = {s.scene_id for s in test_ds.samples}

    assert len(train_scenes.intersection(val_scenes)) == 0
    assert len(train_scenes.intersection(test_scenes)) == 0
    assert len(val_scenes.intersection(test_scenes)) == 0


