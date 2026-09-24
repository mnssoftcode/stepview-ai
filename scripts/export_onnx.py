#!/usr/bin/env python3
"""StepView — Export PyTorch Checkpoint to ONNX.

Exports a trained MobileNetV3-Small LR-ASPP checkpoint to a fixed-size or dynamic
batch ONNX model for mobile/edge runtime deployment.

Usage:
    python scripts/export_onnx.py \
        --checkpoint experiments/runs/dataset_b_baseline/best_checkpoint.pt \
        --output models/stepview_segmentation.onnx \
        --input-size 256
"""

from __future__ import annotations

import argparse
from pathlib import Path
import onnx
import torch

from stepview.models.segmentation import StepViewSegmentationModel, build_segmentation_model


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_size: int = 256,
    opset_version: int = 17,
) -> None:
    """Load model checkpoint and export to ONNX graph."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    num_classes = checkpoint.get("config", {}).get("num_classes", 6)
    model = build_segmentation_model(num_classes=num_classes, pretrained_backbone=False)

    # Support both raw state dict and structured trainer checkpoint
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    print(f"Exporting ONNX model to {output_path} (input: 1x3x{input_size}x{input_size}, opset: {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Validate ONNX graph
    print("Validating exported ONNX graph with onnx.checker...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Export successful: {output_path} ({file_size_mb:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export StepView Checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/runs/dataset_b_baseline/best_checkpoint.pt"),
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/stepview_segmentation.onnx"),
        help="Path for destination .onnx file",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Fixed spatial dimension (height and width) for export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_size=args.input_size,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
