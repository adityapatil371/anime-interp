"""
Training script for UnetWithDistance fine-tuning on ATD-12K.

Only UnetWithDistance is trained — IFNet is frozen.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 20 --batch-size 4
    python scripts/train.py --resume checkpoints/unet_epoch_5.pth

What this script does:
    1. Loads ATD-12K train split with precomputed distance maps
    2. Initialises pipeline (frozen IFNet + trainable Unet)
    3. Trains Unet with LPIPS + L1 loss
    4. Saves checkpoint after every epoch
    5. Logs train loss to TensorBoard
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ATD12KDataset
from src.losses.losses import AnimeLoss
from src.models.pipeline import AnimeInterpPipeline


def get_device() -> torch.device:
    """Get best available device — MPS for M1, CUDA for Nvidia, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    unet_state: dict,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
) -> str:
    """
    Save UnetWithDistance weights only — not the full pipeline.
    IFNet weights don't change so we never need to save them.

    Args:
        unet_state:     model.unet.state_dict()
        epoch:          current epoch number
        loss:           average training loss this epoch
        checkpoint_dir: directory to save into

    Returns:
        path where checkpoint was saved
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch:03d}.pth")
    torch.save({
        "epoch": epoch,
        "loss":  loss,
        "unet_state_dict": unet_state,
    }, path)
    return path


def train_one_epoch(
    model:     AnimeInterpPipeline,
    loader:    DataLoader,
    loss_fn:   AnimeLoss,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
    writer:    SummaryWriter,
) -> dict:
    """
    Run one full training epoch.

    Returns dict with average losses for the epoch.
    """
    model.unet.train()   # Only Unet in train mode
    model.ifnet.eval()   # IFNet always in eval mode

    total_loss  = 0.0
    total_lpips = 0.0
    total_l1    = 0.0
    n_batches   = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)

    for batch in pbar:
        # Move all tensors to device
        img0    = batch["frame_a"].to(device)   # (B, 3, H, W)
        img1    = batch["frame_b"].to(device)   # (B, 3, H, W)
        gt      = batch["frame_gt"].to(device)  # (B, 3, H, W)
        dist_a  = batch["dist_a"].to(device)    # (B, 1, H, W)
        dist_b  = batch["dist_b"].to(device)    # (B, 1, H, W)

        optimizer.zero_grad()

        # Forward pass
        pred = model(img0, img1, dist_a, dist_b)

        # Compute loss
        loss, loss_dict = loss_fn(pred, gt)

        # Backward pass — only Unet parameters get gradients
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss  += loss_dict["loss_total"]
        total_lpips += loss_dict["loss_lpips"]
        total_l1    += loss_dict["loss_l1"]
        n_batches   += 1

        # Update progress bar
        pbar.set_postfix(
            loss=f"{loss_dict['loss_total']:.4f}",
            lpips=f"{loss_dict['loss_lpips']:.4f}",
            l1=f"{loss_dict['loss_l1']:.4f}",
        )

    # Compute epoch averages
    avg_loss  = total_loss  / n_batches
    avg_lpips = total_lpips / n_batches
    avg_l1    = total_l1    / n_batches

    # Log to TensorBoard
    writer.add_scalar("train/loss_total", avg_loss,  epoch)
    writer.add_scalar("train/loss_lpips", avg_lpips, epoch)
    writer.add_scalar("train/loss_l1",    avg_l1,    epoch)

    return {
        "loss_total": avg_loss,
        "loss_lpips": avg_lpips,
        "loss_l1":    avg_l1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train UnetWithDistance on ATD-12K"
    )
    parser.add_argument("--datasets-root",  default="data/datasets")
    parser.add_argument("--checkpoint",     default="checkpoints/flownet.pkl")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir",        default="runs/train")
    parser.add_argument("--epochs",         type=int,   default=15)
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--num-workers",    type=int,   default=4)
    parser.add_argument("--size",           type=int,   nargs=2,
                        default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--resume",         type=str,   default=None,
                        help="Path to unet checkpoint to resume from")
    args = parser.parse_args()

    device = get_device()
    print(f"Training on: {device}")

    # Dataset and DataLoader
    print("Loading dataset...")
    train_ds = ATD12KDataset(
        root=args.datasets_root,
        split="train",
        size=tuple(args.size),
        use_cached_dist=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train samples: {len(train_ds)}")

    # Model
    print("Initialising model...")
    model = AnimeInterpPipeline(
        checkpoint_path=args.checkpoint,
        device=device,
    ).to(device)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 1

    # Loss — only optimise Unet parameters
    loss_fn   = AnimeLoss(device=device).to(device)
    optimizer = torch.optim.AdamW(
        model.unet.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Cosine annealing — gradually reduces LR to near zero over training
    # Helps fine-tuning converge without overshooting
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Steps per epoch: {len(train_loader)}\n")

    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train
        metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, writer
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Save checkpoint every epoch
        ckpt_path = save_checkpoint(
            model.unet.state_dict(),
            epoch,
            metrics["loss_total"],
            args.checkpoint_dir,
        )

        # Track best
        if metrics["loss_total"] < best_loss:
            best_loss = metrics["loss_total"]
            best_path = os.path.join(args.checkpoint_dir, "unet_best.pth")
            torch.save({
                "epoch": epoch,
                "loss":  metrics["loss_total"],
                "unet_state_dict": model.unet.state_dict(),
            }, best_path)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {metrics['loss_total']:.4f} | "
            f"LPIPS: {metrics['loss_lpips']:.4f} | "
            f"L1: {metrics['loss_l1']:.4f} | "
            f"Time: {epoch_time:.1f}s | "
            f"Saved: {ckpt_path}"
        )

    writer.close()
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()