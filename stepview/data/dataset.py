"""StepView Dataset Loader.

Provides PyTorch Dataset implementation for loading paired terrain images,
semantic segmentation masks, and metadata with explicit class validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from stepview.data.schema import (
    NUM_CLASSES,
    SampleMetadata,
    TerrainClass,
    VALID_CLASS_IDS,
    validate_mask_classes,
)


def stepview_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for StepView DataLoader batches.

    Stacks image and mask tensors into batch tensors, while preserving
    string identifiers and metadata dictionaries as lists.

    Args:
        batch: List of sample dictionaries returned by StepViewDataset.__getitem__.

    Returns:
        Dict with batched 'image' (B, C, H, W) and 'mask' (B, H, W),
        and lists for 'sample_id', 'scene_id', and 'metadata'.
    """
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "scene_id": [item["scene_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


class StepViewDataset(Dataset):
    """PyTorch Dataset for StepView terrain segmentation.

    Separates:
    - Image loading (RGB 3-channel)
    - Mask loading (Indexed uint8 1-channel)
    - Class validation (Strict verification against TerrainClass)
    - Transformations (Separate or joint image/mask transforms)
    - Metadata tracking (Scene ID, split, capture attributes)

    Expected directory structure under root_dir:
        root_dir/
            images/
                sample_001.png
            masks/
                sample_001.png
            metadata/
                manifest.json (optional)
            splits/
                train.txt, val.txt, test.txt (optional)
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        split_file: Optional[Union[str, Path]] = None,
        manifest_file: Optional[Union[str, Path]] = None,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        target_transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        joint_transform: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
        validate_classes_on_load: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initialize StepViewDataset.

        Args:
            root_dir: Base directory containing images/, masks/, metadata/, splits/.
            split: Data split name ('train', 'val', 'test', 'all').
            split_file: Optional path to text file with sample IDs for this split.
            manifest_file: Optional path to JSON manifest file.
            transform: Optional transform callable for RGB image.
            target_transform: Optional transform callable for mask.
            joint_transform: Optional joint transform (e.g. geometric augmentations).
            validate_classes_on_load: If True, checks that mask classes are valid.
            target_size: Optional (height, width) to resize image and mask to.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.validate_classes_on_load = validate_classes_on_load
        self.target_size = target_size

        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"
        self.metadata_dir = self.root_dir / "metadata"
        self.splits_dir = self.root_dir / "splits"

        self.samples: List[SampleMetadata] = []
        self._load_dataset_index(split_file, manifest_file)

    def _load_dataset_index(
        self,
        split_file: Optional[Union[str, Path]],
        manifest_file: Optional[Union[str, Path]],
    ) -> None:
        """Discover and index dataset samples from manifest or directory scans."""
        # 1. Determine allowed sample IDs from split file if present
        allowed_sample_ids: Optional[set[str]] = None
        target_split_path = Path(split_file) if split_file else self.splits_dir / f"{self.split}.txt"
        if target_split_path.is_file() and self.split != "all":
            with open(target_split_path, "r", encoding="utf-8") as f:
                allowed_sample_ids = {
                    line.strip() for line in f if line.strip() and not line.startswith("#")
                }

        # 2. Check for manifest file
        target_manifest = Path(manifest_file) if manifest_file else self.metadata_dir / "manifest.json"
        if target_manifest.is_file():
            with open(target_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data.get("samples", data)
                for item in records:
                    meta = SampleMetadata.from_dict(item)
                    if allowed_sample_ids is not None:
                        if meta.sample_id in allowed_sample_ids:
                            self.samples.append(meta)
                    elif self.split == "all" or meta.split == self.split:
                        self.samples.append(meta)
            return

        # 3. Fallback: Auto-discover paired files from images/ and masks/ directories
        if not self.images_dir.is_dir():
            return

        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        image_files = sorted(
            [f for f in self.images_dir.iterdir() if f.suffix.lower() in image_extensions]
        )

        for img_path in image_files:
            sample_id = img_path.stem
            if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
                continue

            # Look for matching mask file (e.g. sample_id.png)
            mask_candidates = [
                self.masks_dir / f"{sample_id}.png",
                self.masks_dir / f"{sample_id}_mask.png",
            ]
            mask_path = next((m for m in mask_candidates if m.is_file()), None)

            if mask_path is not None:
                meta = SampleMetadata(
                    sample_id=sample_id,
                    image_rel_path=str(img_path.relative_to(self.root_dir)),
                    mask_rel_path=str(mask_path.relative_to(self.root_dir)),
                    scene_id="unassigned",
                    split=self.split,
                )
                self.samples.append(meta)

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load an RGB image from disk.

        Args:
            path: Absolute or relative Path to image.

        Returns:
            np.ndarray: RGB image array of shape (H, W, 3) and dtype uint8.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded as an image.
        """
        resolved_path = Path(path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Image file not found: {resolved_path}")

        img_bgr = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Corrupted or invalid image at: {resolved_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def load_mask(self, path: Union[str, Path]) -> np.ndarray:
        """Load a single-channel indexed segmentation mask from disk.

        Args:
            path: Absolute or relative Path to mask.

        Returns:
            np.ndarray: 2D integer mask array of shape (H, W) and dtype uint8.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded or is not 2D single-channel.
        """
        resolved_path = Path(path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Mask file not found: {resolved_path}")

        mask = cv2.imread(str(resolved_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Corrupted or invalid mask at: {resolved_path}")

        if mask.ndim == 3:
            # If accidentally saved with 3 channels, verify identical channels and take channel 0
            if mask.shape[2] == 3 or mask.shape[2] == 4:
                mask = mask[:, :, 0]
            else:
                raise ValueError(f"Unsupported mask channel dimension: {mask.shape}")

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

        return mask.astype(np.uint8)

    def validate_classes(self, mask: np.ndarray) -> None:
        """Validate mask pixel classes against defined schema.

        Args:
            mask: 2D numpy array of class indices.

        Raises:
            ValueError: If invalid class indices are present.
        """
        validate_mask_classes(mask)

    def apply_transforms(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply joint and separate transforms, converting arrays to PyTorch Tensors.

        Args:
            image: RGB image (H, W, 3) uint8.
            mask: Segmentation mask (H, W) uint8.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Image tensor: shape (3, H, W), float32 normalized [0.0, 1.0].
                - Mask tensor: shape (H, W), int64 (torch.long).
        """
        if self.target_size is not None:
            th, tw = self.target_size
            image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        if self.transform is not None:
            image_t = self.transform(image)
        else:
            # Default transform: HWC uint8 -> CHW float32 [0.0, 1.0]
            image_t = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

        if self.target_transform is not None:
            mask_t = self.target_transform(mask)
        else:
            # Default target transform: HW uint8 -> HW int64 (torch.long)
            mask_t = torch.from_numpy(mask.astype(np.int64)).long()

        return image_t, mask_t

    def get_metadata(self, idx: int) -> SampleMetadata:
        """Retrieve the metadata record for sample at index."""
        return self.samples[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve sample at index.

        Returns:
            Dict containing:
                - 'image': torch.FloatTensor (3, H, W)
                - 'mask': torch.LongTensor (H, W)
                - 'sample_id': str
                - 'scene_id': str
                - 'metadata': Dict[str, Any]
        """
        meta = self.get_metadata(idx)
        img_path = self.root_dir / meta.image_rel_path
        mask_path = self.root_dir / meta.mask_rel_path

        # 1. Load image and mask
        image = self.load_image(img_path)
        mask = self.load_mask(mask_path)

        # 2. Check spatial shape alignment
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Shape mismatch for sample {meta.sample_id}: "
                f"image shape {image.shape[:2]} vs mask shape {mask.shape[:2]}"
            )

        # 3. Class validation
        if self.validate_classes_on_load:
            self.validate_classes(mask)

        # 4. Transformations
        image_t, mask_t = self.apply_transforms(image, mask)

        return {
            "image": image_t,
            "mask": mask_t,
            "sample_id": meta.sample_id,
            "scene_id": meta.scene_id,
            "metadata": meta.to_dict(),
        }
