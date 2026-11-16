import os
import time
import json
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.unet_scratch import UNetScratch
from models.unet_pretrained import build_unet
from utils.dataset import get_dataloaders
from utils.losses import BCEDiceLoss, DiceLoss, FocalLoss
from utils.metrics import dice_coeff, iou_score


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total_batches = len(dataloader)
    pbar = tqdm(dataloader, desc="  [Train]", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            dice = dice_coeff(logits, masks)
            iou = iou_score(logits, masks)
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice:.4f}"})
    return {
        "loss": running_loss / max(1, total_batches),
        "dice": running_dice / max(1, total_batches),
        "iou": running_iou / max(1, total_batches),
    }


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total_batches = len(dataloader)
    pbar = tqdm(dataloader, desc="  [Val]", leave=False)
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            dice = dice_coeff(logits, masks)
            iou = iou_score(logits, masks)
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice:.4f}"})
    return {
        "loss": running_loss / max(1, total_batches),
        "dice": running_dice / max(1, total_batches),
        "iou": running_iou / max(1, total_batches),
    }


def main():
    parser = argparse.ArgumentParser(description="U‑Net training")
    parser.add_argument("--model", type=str, default="scratch", choices=["scratch", "pretrained"], help="Model type")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet34", "resnet50"], help="Backbone for pretrained encoder")
    parser.add_argument("--data_dir", type=str, default="data/kvasir-seg", help="Path to dataset (images + masks)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=256, help="Image size (square)")
    parser.add_argument("--loss", type=str, default="bce_dice", choices=["bce_dice", "dice", "focal"], help="Loss function")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    model = build_unet(
        model_type=args.model,
        backbone=args.backbone,
        pretrained=True,
        in_channels=3,
        num_classes=1,
    ).to(device)
    total_params = model.count_parameters()
    if args.loss == "bce_dice":
        criterion = BCEDiceLoss()
    elif args.loss == "dice":
        criterion = DiceLoss()
    else:
        criterion = FocalLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    history = {
        "model_type": args.model,
        "backbone": args.backbone if args.model == "pretrained" else "none",
        "parameters": total_params,
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss": [], "val_dice": [], "val_iou": [],
    }
    best_val_dice = 0.0
    model_name = f"unet_{args.model}" + (f"_{args.backbone}" if args.model == "pretrained" else "")
    best_ckpt_path = os.path.join(args.save_dir, f"best_{model_name}.pth")
    history_path = os.path.join(args.save_dir, f"history_{model_name}.json")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_res["dice"])
        history["train_loss"].append(train_res["loss"])
        history["train_dice"].append(train_res["dice"])
        history["train_iou"].append(train_res["iou"])
        history["val_loss"].append(val_res["loss"])
        history["val_dice"].append(val_res["dice"])
        history["val_iou"].append(val_res["iou"])
        elapsed = time.time() - epoch_start
        if val_res["dice"] > best_val_dice:
            best_val_dice = val_res["dice"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "val_iou": val_res["iou"],
                "args": vars(args),
            }, best_ckpt_path)
    total_time = time.time() - start_time
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    main()
