"""Train the SegFormer-B0 binary segmentation baseline from scratch."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.losses import BCEDiceLoss, DiceLoss, FocalLoss
from models.segformer import build_segformer
from utils.dataset import get_dataloaders
from utils.metrics import dice_coeff, iou_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress = tqdm(dataloader, desc="  [Train]", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dice = dice_coeff(logits, masks, from_logits=True)
            iou = iou_score(logits, masks, from_logits=True)

        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

    batches = max(1, len(dataloader))
    return {
        "loss": running_loss / batches,
        "dice": running_dice / batches,
        "iou": running_iou / batches,
    }


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress = tqdm(dataloader, desc="  [Val]", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        dice = dice_coeff(logits, masks, from_logits=True)
        iou = iou_score(logits, masks, from_logits=True)

        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

    batches = max(1, len(dataloader))
    return {
        "loss": running_loss / batches,
        "dice": running_dice / batches,
        "iou": running_iou / batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SegFormer-B0 from scratch")
    parser.add_argument("--data_dir", type=str, default="data/raw/Kvasir-SEG")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="bce_dice", choices=["bce_dice", "dice", "focal"])
    parser.add_argument("--save_dir", type=str, default="experiments/segformer_b0_scratch/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda được chọn nhưng CUDA không khả dụng.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = build_segformer(num_labels=1).to(device)
    print(f"Trainable parameters: {model.count_parameters():,}")

    if args.loss == "bce_dice":
        criterion = BCEDiceLoss()
    elif args.loss == "dice":
        criterion = DiceLoss()
    else:
        criterion = FocalLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    os.makedirs(args.save_dir, exist_ok=True)
    model_name = "segformer_b0_scratch"
    checkpoint_path = os.path.join(args.save_dir, f"best_{model_name}.pth")
    history_path = os.path.join(args.save_dir, f"history_{model_name}.json")

    history = {
        "model_type": model_name,
        "pretrained": False,
        "parameters": model.count_parameters(),
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }

    best_val_dice = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_result["dice"])

        for split, result in (("train", train_result), ("val", val_result)):
            history[f"{split}_loss"].append(result["loss"])
            history[f"{split}_dice"].append(result["dice"])
            history[f"{split}_iou"].append(result["iou"])

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_result['loss']:.4f} | "
            f"Train Dice: {train_result['dice']:.4f} | "
            f"Train IoU: {train_result['iou']:.4f} | "
            f"Val Loss: {val_result['loss']:.4f} | "
            f"Val Dice: {val_result['dice']:.4f} | "
            f"Val IoU: {val_result['iou']:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        if val_result["dice"] > best_val_dice:
            best_val_dice = val_result["dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_result["dice"],
                    "val_iou": val_result["iou"],
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint: {checkpoint_path}")

        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

    total_time = time.time() - start_time
    print(f"Training finished in {total_time / 60:.2f} minutes")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
