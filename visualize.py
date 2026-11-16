"""Visualization script for medical segmentation results.

Provides two main functions:
- ``plot_learning_curves``: plots validation loss, Dice, and IoU curves for both scratch and pretrained models.
- ``visualize_predictions``: generates side‑by‑side comparisons of original images, ground‑truth masks, and model predictions.

The script can be executed directly via the CLI.
"""

import os
import json
import argparse
from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.unet_scratch import UNetScratch
from models.unet_pretrained import build_unet
from utils.dataset import MedicalSegmentationDataset, find_image_mask_pairs
from utils.metrics import dice_coeff, iou_score


def plot_learning_curves(
    history_scratch_path: str,
    history_pretrained_path: str,
    out_path: str = "results/learning_curves.png",
) -> None:
    """Plot validation loss, Dice and IoU curves for both models.

    Args:
        history_scratch_path: Path to JSON file containing training history of the scratch model.
        history_pretrained_path: Path to JSON file containing training history of the pretrained model.
        out_path: Destination image path for the saved figure.
    """
    if not os.path.exists(history_scratch_path) and not os.path.exists(
        history_pretrained_path
    ):
        return

    plt.figure(figsize=(15, 5))
    data_to_plot: List[tuple] = []
    if os.path.exists(history_scratch_path):
        with open(history_scratch_path, "r", encoding="utf-8") as f:
            data_to_plot.append(("U-Net Scratch", json.load(f), "blue"))
    if os.path.exists(history_pretrained_path):
        with open(history_pretrained_path, "r", encoding="utf-8") as f:
            data_to_plot.append(("U-Net Pretrained (ResNet34)", json.load(f), "green"))

    # Subplot 1: Validation Loss
    plt.subplot(1, 3, 1)
    for name, hist, color in data_to_plot:
        epochs = range(1, len(hist["val_loss"]) + 1)
        plt.plot(epochs, hist["val_loss"], label=name, color=color, linewidth=2)
    plt.title("Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Subplot 2: Validation Dice Score
    plt.subplot(1, 3, 2)
    for name, hist, color in data_to_plot:
        epochs = range(1, len(hist["val_dice"]) + 1)
        plt.plot(
            epochs,
            [d * 100 for d in hist["val_dice"]],
            label=name,
            color=color,
            linewidth=2,
        )
    plt.title("Validation Dice Score (%)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Dice (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Subplot 3: Validation IoU
    plt.subplot(1, 3, 3)
    for name, hist, color in data_to_plot:
        epochs = range(1, len(hist["val_iou"]) + 1)
        plt.plot(
            epochs,
            [iou * 100 for iou in hist["val_iou"]],
            label=name,
            color=color,
            linewidth=2,
        )
    plt.title("Validation IoU (Jaccard) (%)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("IoU (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def visualize_predictions(
    data_dir: str,
    scratch_ckpt: str,
    pretrained_ckpt: str,
    backbone: str = "resnet34",
    num_samples: int = 4,
    img_size: int = 256,
    out_path: str = "results/segmentation_comparison.png",
) -> None:
    """Generate a grid visualising model predictions against ground truth.

    Args:
        data_dir: Directory containing image/mask pairs.
        scratch_ckpt: Path to checkpoint for the scratch model.
        pretrained_ckpt: Path to checkpoint for the pretrained model.
        backbone: Backbone name for the pretrained model (default ``resnet34``).
        num_samples: Number of samples to visualise (max limited by dataset size).
        img_size: Image size used for the dataset loader.
        out_path: Destination path for the saved figure.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths, mask_paths = find_image_mask_pairs(data_dir)
    if not image_paths:
        return

    # Initialise models
    scratch_model = UNetScratch(in_channels=3, num_classes=1).to(device)
    if os.path.exists(scratch_ckpt):
        ckpt = torch.load(scratch_ckpt, map_location=device)
        scratch_model.load_state_dict(ckpt["model_state_dict"])
    scratch_model.eval()

    pretrained_model = build_unet(
        model_type="pretrained", backbone=backbone, pretrained=True, num_classes=1
    ).to(device)
    if os.path.exists(pretrained_ckpt):
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        pretrained_model.load_state_dict(ckpt["model_state_dict"])
    pretrained_model.eval()

    num_samples = min(num_samples, len(image_paths))
    dataset = MedicalSegmentationDataset(
        image_paths[:num_samples],
        mask_paths[:num_samples],
        img_size=(img_size, img_size),
        is_train=False,
        augment=False,
    )

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        img_tensor, mask_tensor = dataset[i]
        input_batch = img_tensor.unsqueeze(0).to(device)
        target_batch = mask_tensor.unsqueeze(0).to(device)

        # Convert image tensor to displayable RGB
        inv_img = img_tensor.clone().permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inv_img = np.clip((inv_img * std + mean), 0, 1)
        gt_mask = mask_tensor.squeeze().cpu().numpy()

        with torch.no_grad():
            out_scratch = scratch_model(input_batch)
            prob_scratch = torch.sigmoid(out_scratch).squeeze().cpu().numpy()
            pred_scratch = (prob_scratch > 0.5).astype(np.float32)
            dice_s = dice_coeff(out_scratch, target_batch)

            out_pretrained = pretrained_model(input_batch)
            prob_pretrained = torch.sigmoid(out_pretrained).squeeze().cpu().numpy()
            pred_pretrained = (prob_pretrained > 0.5).astype(np.float32)
            dice_p = dice_coeff(out_pretrained, target_batch)

        # Original image
        axes[i, 0].imshow(inv_img)
        axes[i, 0].set_title(f"Sample #{i+1}: Image", fontsize=10)
        axes[i, 0].axis("off")

        # Ground truth mask
        axes[i, 1].imshow(gt_mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis("off")

        # Scratch prediction
        axes[i, 2].imshow(pred_scratch, cmap="jet", alpha=0.8)
        axes[i, 2].set_title(f"U-Net Scratch\nDice: {dice_s*100:.1f}%", fontsize=10)
        axes[i, 2].axis("off")

        # Pretrained prediction
        axes[i, 3].imshow(pred_pretrained, cmap="jet", alpha=0.8)
        axes[i, 3].set_title(
            f"U-Net Pretrained ({backbone})\nDice: {dice_p*100:.1f}%",
            fontsize=10,
        )
        axes[i, 3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    """CLI entry point for the visualization utilities."""
    parser = argparse.ArgumentParser(description="Visualize segmentation results")
    parser.add_argument("--data_dir", type=str, default="data/kvasir-seg")
    parser.add_argument("--scratch_ckpt", type=str, default="checkpoints/best_unet_scratch.pth")
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="checkpoints/best_unet_pretrained_resnet34.pth",
    )
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()

    # Plot learning curves
    plot_learning_curves(
        history_scratch_path="checkpoints/history_unet_scratch.json",
        history_pretrained_path=f"checkpoints/history_unet_pretrained_{args.backbone}.json",
        out_path="results/learning_curves.png",
    )

    # Visualize predictions
    visualize_predictions(
        data_dir=args.data_dir,
        scratch_ckpt=args.scratch_ckpt,
        pretrained_ckpt=args.pretrained_ckpt,
        backbone=args.backbone,
        num_samples=args.num_samples,
        out_path="results/segmentation_comparison.png",
    )


if __name__ == "__main__":
    main()
