#!/usr/bin/env python3
"""StepView — Model Training and Evaluation Pipeline.

Trains StepView semantic segmentation models on configured dataset splits,
evaluates per-class IoU, active mIoU, confusion matrices across train/val/test,
and generates qualitative prediction panels.

Usage:
    python stepview/training/train.py --config experiments/configs/dataset_b_baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stepview.data.dataset import StepViewDataset, stepview_collate_fn
from stepview.data.schema import CLASS_NAMES, CLASS_PALETTE, NUM_CLASSES, TerrainClass
from stepview.inference.segmenter import TerrainSegmenter
from stepview.models.segmentation import build_segmentation_model
from stepview.training.losses import CombinedSegmentationLoss, compute_median_frequency_weights
from stepview.training.metrics import SegmentationMetrics
from stepview.training.trainer import StepViewTrainer


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert indexed 2D class mask to RGB color mask."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_PALETTE.items():
        color_mask[mask == class_id] = color
    return color_mask


def generate_qualitative_panel(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    sample_id: str,
    scene_id: str,
    split: str,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create a 3-column panel: Original RGB | Ground Truth Overlay | Prediction Overlay."""
    h, w = image_rgb.shape[:2]
    if gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    gt_color = colorize_mask(gt_mask)
    pred_color = colorize_mask(pred_mask)

    gt_overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, gt_color, alpha, 0)
    pred_overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, pred_color, alpha, 0)

    # Header bar
    header_h = 36
    panel = np.zeros((h + header_h, w * 3, 3), dtype=np.uint8)
    panel[:header_h, :] = 25

    panel[header_h:, 0:w] = image_rgb
    panel[header_h:, w:2*w] = gt_overlay
    panel[header_h:, 2*w:3*w] = pred_overlay

    # Text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    color = (255, 255, 255)

    cv2.putText(panel, f"[{split.upper()}] {sample_id} ({scene_id}) - RGB", (10, 24), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(panel, "Ground Truth Mask Overlay", (w + 10, 24), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(panel, "Model Prediction Overlay", (2 * w + 10, 24), font, font_scale, (100, 255, 100), thickness, cv2.LINE_AA)

    return panel


def run_training(
    data_dir: Path,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "AdamW",
    scheduler_name: Optional[str] = "CosineAnnealingLR",
    random_seed: int = 42,
    output_dir: Path = Path("experiments/runs/dataset_b_baseline"),
    weight_mode: str = "median_freq",
    pretrained_backbone: bool = False,
    freeze_backbone: bool = False,
    device_name: str = "auto",
    target_size: Optional[Tuple[int, int]] = (480, 480),
    dataset_version: str = "Dataset B (RUGD 600-sample curated subset, v0.2.0)",
) -> Dict[str, Any]:
    """Execute training, validation, multi-split evaluation, and qualitative artifact generation."""
    set_seed(random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = Path("experiments/visualizations/dataset_b_eval")
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device selection
    if device_name == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(device_name)
    print(f"Executing on device: {device}", flush=True)

    # 2. Data Loaders
    train_dataset = StepViewDataset(root_dir=data_dir, split="train", target_size=target_size)
    val_dataset = StepViewDataset(root_dir=data_dir, split="val", target_size=target_size)
    test_dataset = StepViewDataset(root_dir=data_dir, split="test", target_size=target_size)

    print(f"Loaded samples -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=stepview_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=stepview_collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=stepview_collate_fn,
        num_workers=0,
    )

    # 3. Model instantiation
    model = build_segmentation_model(
        num_classes=NUM_CLASSES,
        pretrained_backbone=pretrained_backbone,
    )
    total_params = sum(p.numel() for p in model.parameters())

    if freeze_backbone:
        for p in model.lraspp.backbone.parameters():
            p.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: MobileNetV3-Small + LR-ASPP ({total_params:,} params, {trainable_params:,} trainable; backbone FROZEN)", flush=True)
    else:
        trainable_params = total_params
        print(f"Model: MobileNetV3-Small + LR-ASPP ({total_params:,} parameters; full fine-tuning)", flush=True)

    # 4. Class weighting configuration
    class_weights: Optional[torch.Tensor] = None
    applied_weights_list: Optional[List[float]] = None
    if weight_mode == "median_freq":
        stats_file = data_dir / "metadata" / "class_stats.json"
        if stats_file.is_file():
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
                pixel_counts = {int(k): v["pixel_count"] for k, v in stats["classes"].items()}
                class_weights = compute_median_frequency_weights(pixel_counts, num_classes=NUM_CLASSES).to(device)
                applied_weights_list = [round(float(w), 4) for w in class_weights.cpu().tolist()]
                print(f"Applied median frequency class weights: {applied_weights_list}", flush=True)

    # 5. Loss, Optimizer, Scheduler
    criterion = CombinedSegmentationLoss(class_weights=class_weights)
    trainable_params_iter = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params_iter, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(trainable_params_iter, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    scheduler = None
    if scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 6. Save experiment configuration
    exp_config = {
        "experiment_name": output_dir.name,
        "dataset_version": dataset_version,
        "data_dir": str(data_dir),
        "split_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "model": {
            "architecture": "MobileNetV3-Small + LR-ASPP",
            "num_classes": NUM_CLASSES,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "pretrained_backbone": pretrained_backbone,
            "freeze_backbone": freeze_backbone,
            "input_resolution": list(target_size) if target_size else [550, 688],
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "random_seed": random_seed,
            "weight_mode": weight_mode,
            "class_weights": applied_weights_list,
        },
        "device": str(device),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    # 7. Trainer
    trainer = StepViewTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=NUM_CLASSES,
    )

    # 8. Training Loop
    history = []
    best_miou = -1.0
    best_epoch = 1

    print(f"\n================================================================================")
    print(f"StepView — Dataset B Baseline Training ({epochs} epochs, seed={random_seed})")
    print(f"================================================================================")

    for epoch in range(1, epochs + 1):
        train_stats = trainer.train_epoch(epoch)
        val_stats = trainer.evaluate()

        current_lr = optimizer.param_groups[0]["lr"]
        combined = {
            "epoch": epoch,
            "lr": current_lr,
            **train_stats,
            **val_stats,
        }
        history.append(combined)

        miou = val_stats.get("mean_iou", 0.0)
        pixel_acc = val_stats.get("pixel_accuracy", 0.0)
        ground_iou = val_stats.get("iou_ground", 0.0)
        obs_iou = val_stats.get("iou_obstacle", 0.0)
        veg_iou = val_stats.get("iou_vegetation", 0.0)
        unc_iou = val_stats.get("iou_uncertain_surface", 0.0)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_stats['train_loss']:.4f} | "
            f"Val Loss: {val_stats.get('val_loss', 0.0):.4f} | "
            f"mIoU: {miou:.4f} | "
            f"Ground: {ground_iou:.4f} | "
            f"Obs: {obs_iou:.4f} | "
            f"Veg: {veg_iou:.4f} | "
            f"Uncert: {unc_iou:.4f} | "
            f"Acc: {pixel_acc:.4f}",
            flush=True,
        )

        # Save latest checkpoint
        trainer.save_checkpoint(
            path=output_dir / "last_checkpoint.pt",
            epoch=epoch,
            metrics=combined,
            config=exp_config,
        )

        # Save best checkpoint
        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            trainer.save_checkpoint(
                path=output_dir / "best_checkpoint.pt",
                epoch=epoch,
                metrics=combined,
                config=exp_config,
            )

    # Save training history
    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed. Best Val mIoU: {best_miou:.4f} at epoch {best_epoch}.")

    # 9. Multi-Split Comprehensive Evaluation using Best Checkpoint
    print("\nExecuting comprehensive multi-split evaluation using best checkpoint...")
    trainer.load_checkpoint(output_dir / "best_checkpoint.pt")

    # Evaluate Train Split
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=stepview_collate_fn,
        num_workers=0,
    )
    train_metrics = trainer.evaluate(train_eval_loader)
    train_cm = trainer.get_confusion_matrix()

    # Evaluate Validation Split
    val_metrics = trainer.evaluate(val_loader)
    val_cm = trainer.get_confusion_matrix()

    # Evaluate Test Split
    test_metrics = trainer.evaluate(test_loader)
    test_cm = trainer.get_confusion_matrix()

    final_results = {
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "train": {
            "metrics": train_metrics,
            "confusion_matrix": train_cm.tolist(),
        },
        "val": {
            "metrics": val_metrics,
            "confusion_matrix": val_cm.tolist(),
        },
        "test": {
            "metrics": test_metrics,
            "confusion_matrix": test_cm.tolist(),
        },
    }

    with open(output_dir / "final_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    # 10. Generate Qualitative Visualizations on Val and Test images
    print("\nGenerating qualitative prediction visualizations...")
    segmenter = TerrainSegmenter.load_from_checkpoint(output_dir / "best_checkpoint.pt", device=device)

    # Pick 3 representative samples from val and 3 from test
    val_sample_indices = [0, len(val_dataset) // 2, len(val_dataset) - 1]
    test_sample_indices = [0, len(test_dataset) // 2, len(test_dataset) - 1]

    for idx in val_sample_indices:
        sample_meta = val_dataset.get_metadata(idx)
        img_path = data_dir / sample_meta.image_rel_path
        mask_path = data_dir / sample_meta.mask_rel_path

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        res = segmenter.segment(img_rgb)
        pred_mask = res["class_mask"]

        panel = generate_qualitative_panel(
            image_rgb=img_rgb,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            sample_id=sample_meta.sample_id,
            scene_id=sample_meta.scene_id,
            split="val",
        )
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        save_path = vis_dir / f"val_{sample_meta.sample_id}_pred.png"
        cv2.imwrite(str(save_path), panel_bgr)
        print(f"  Saved qualitative panel: {save_path.name}")

    for idx in test_sample_indices:
        sample_meta = test_dataset.get_metadata(idx)
        img_path = data_dir / sample_meta.image_rel_path
        mask_path = data_dir / sample_meta.mask_rel_path

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        res = segmenter.segment(img_rgb)
        pred_mask = res["class_mask"]

        panel = generate_qualitative_panel(
            image_rgb=img_rgb,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            sample_id=sample_meta.sample_id,
            scene_id=sample_meta.scene_id,
            split="test",
        )
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        save_path = vis_dir / f"test_{sample_meta.sample_id}_pred.png"
        cv2.imwrite(str(save_path), panel_bgr)
        print(f"  Saved qualitative panel: {save_path.name}")

    print("\n================================================================================")
    print("DATASET B BASELINE EVALUATION SUMMARY")
    print("================================================================================")
    print(f"{'Split':<8} | {'Loss':<7} | {'PixelAcc':<8} | {'mIoU (Act)':<10} | {'Ground':<8} | {'Obstacle':<8} | {'Vegetation':<10} | {'Uncertain':<9}")
    print("-" * 88)
    for sname, sdata in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(
            f"{sname:<8} | "
            f"{sdata.get('val_loss', 0.0):<7.4f} | "
            f"{sdata.get('pixel_accuracy', 0.0):<8.4f} | "
            f"{sdata.get('mean_iou', 0.0):<10.4f} | "
            f"{sdata.get('iou_ground', 0.0):<8.4f} | "
            f"{sdata.get('iou_obstacle', 0.0):<8.4f} | "
            f"{sdata.get('iou_vegetation', 0.0):<10.4f} | "
            f"{sdata.get('iou_uncertain_surface', 0.0):<9.4f}"
        )
    print("================================================================================\n")

    return final_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StepView Segmentation Model")
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON experiment config file")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to StepView dataset")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer name ('AdamW', 'SGD')")
    parser.add_argument("--scheduler", type=str, default=None, help="Scheduler name ('CosineAnnealingLR', 'None')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output run directory")
    parser.add_argument("--weight-mode", type=str, default=None, choices=["none", "median_freq"], help="Class weighting mode")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('auto', 'cpu', 'mps')")

    args = parser.parse_args()

    # Default parameters
    cfg: Dict[str, Any] = {
        "data_dir": "data",
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "random_seed": 42,
        "output_dir": "experiments/runs/dataset_b_baseline",
        "weight_mode": "median_freq",
        "device": "auto",
        "pretrained_backbone": False,
        "freeze_backbone": False,
        "target_size": [480, 480],
        "dataset_version": "Dataset B (RUGD 600-sample curated subset, v0.2.0)",
    }

    # Load from config file if provided
    if args.config is not None and args.config.is_file():
        with open(args.config, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
            if "data_dir" in file_cfg:
                cfg["data_dir"] = file_cfg["data_dir"]
            if "training" in file_cfg:
                t = file_cfg["training"]
                cfg["epochs"] = t.get("epochs", cfg["epochs"])
                cfg["batch_size"] = t.get("batch_size", cfg["batch_size"])
                cfg["learning_rate"] = t.get("learning_rate", cfg["learning_rate"])
                cfg["weight_decay"] = t.get("weight_decay", cfg["weight_decay"])
                cfg["optimizer"] = t.get("optimizer", cfg["optimizer"])
                cfg["scheduler"] = t.get("scheduler", cfg["scheduler"])
                cfg["random_seed"] = t.get("random_seed", cfg["random_seed"])
                cfg["weight_mode"] = t.get("weight_mode", cfg["weight_mode"])
            if "model" in file_cfg:
                m = file_cfg["model"]
                cfg["pretrained_backbone"] = m.get("pretrained_backbone", cfg["pretrained_backbone"])
                cfg["freeze_backbone"] = m.get("freeze_backbone", cfg["freeze_backbone"])
                if "input_resolution" in m:
                    cfg["target_size"] = m["input_resolution"]
            if "device" in file_cfg:
                cfg["device"] = file_cfg["device"]
            if "output_dir" in file_cfg:
                cfg["output_dir"] = file_cfg["output_dir"]
            if "dataset_version" in file_cfg:
                cfg["dataset_version"] = file_cfg["dataset_version"]

    # Command line overrides
    if args.data_dir is not None:
        cfg["data_dir"] = str(args.data_dir)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.optimizer is not None:
        cfg["optimizer"] = args.optimizer
    if args.scheduler is not None:
        cfg["scheduler"] = None if args.scheduler.lower() == "none" else args.scheduler
    if args.seed is not None:
        cfg["random_seed"] = args.seed
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    if args.weight_mode is not None:
        cfg["weight_mode"] = args.weight_mode
    if args.device is not None:
        cfg["device"] = args.device

    target_size_tuple = tuple(cfg["target_size"]) if cfg.get("target_size") else None

    run_training(
        data_dir=Path(cfg["data_dir"]),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        optimizer_name=cfg["optimizer"],
        scheduler_name=cfg["scheduler"],
        random_seed=cfg["random_seed"],
        output_dir=Path(cfg["output_dir"]),
        weight_mode=cfg["weight_mode"],
        pretrained_backbone=cfg["pretrained_backbone"],
        freeze_backbone=cfg["freeze_backbone"],
        device_name=cfg["device"],
        target_size=target_size_tuple,
        dataset_version=cfg["dataset_version"],
    )


if __name__ == "__main__":
    main()
