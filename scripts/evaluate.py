"""
U‑Net evaluation script (cleaned: no prints, minimal comments).
"""

import os
import time
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet_scratch import UNetScratch
from models.unet_pretrained import build_unet
from utils.dataset import get_dataloaders
from losses.losses import BCEDiceLoss
from utils.metrics import dice_coeff, iou_score


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Return loss, Dice, IoU, latency (ms) and FPS on the test set."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_batches = len(dataloader)
    total_images = len(dataloader.dataset)

    # Warm‑up GPU
    dummy = torch.randn(1, 3, 256, 256).to(device)
    for _ in range(5):
        _ = model(dummy)

    start_time = time.time()
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            total_dice += dice_coeff(logits, masks, from_logits=True)
            total_iou += iou_score(logits, masks, from_logits=True)

    total_time = time.time() - start_time
    avg_latency = (total_time / total_images) * 1000.0  # ms per image
    fps = total_images / total_time

    return {
        "test_loss": total_loss / max(1, total_batches),
        "dice": (total_dice / max(1, total_batches)) * 100.0,
        "iou": (total_iou / max(1, total_batches)) * 100.0,
        "latency_ms": avg_latency,
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare U‑Net Scratch vs Pretrained")
    parser.add_argument("--data_dir", type=str, default="data/raw/Kvasir-SEG", help="Dataset path")
    parser.add_argument("--scratch_ckpt", type=str, default="experiments/unet_scratch/checkpoints/best_unet_scratch.pth")
    parser.add_argument("--pretrained_ckpt", type=str, default="experiments/unet_resnet34/checkpoints/best_unet_pretrained_resnet34.pth")
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="experiments/comparison/results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = BCEDiceLoss()

    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    models_to_test = []
    # Scratch model
    scratch_model = UNetScratch(in_channels=3, num_classes=1).to(device)
    if os.path.exists(args.scratch_ckpt):
        ckpt = torch.load(args.scratch_ckpt, map_location=device)
        scratch_model.load_state_dict(ckpt["model_state_dict"])
    models_to_test.append(("U‑Net (Scratch)", scratch_model))

    # Pretrained model
    pretrained_model = build_unet(
        model_type="pretrained",
        backbone=args.backbone,
        pretrained=True,
        in_channels=3,
        num_classes=1,
    ).to(device)
    if os.path.exists(args.pretrained_ckpt):
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        pretrained_model.load_state_dict(ckpt["model_state_dict"])
    models_to_test.append((f"U‑Net (Pretrained {args.backbone.upper()})", pretrained_model))

    results = []
    for name, model in models_to_test:
        params_m = model.count_parameters() / 1e6
        metrics = evaluate_model(model, test_loader, criterion, device)
        results.append({
            "Model": name,
            "Params (M)": f"{params_m:.2f}M",
            "Dice (%)": f"{metrics['dice']:.2f}%",
            "IoU (%)": f"{metrics['iou']:.2f}%",
            "Loss": f"{metrics['test_loss']:.4f}",
            "Latency (ms)": f"{metrics['latency_ms']:.1f}",
            "FPS": f"{metrics['fps']:.1f}",
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.out_dir, "comparison_results.csv")
    md_path = os.path.join(args.out_dir, "comparison_results.md")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
