"""
Evaluate fine-tuned model on ATD-12K test set.

Reports PSNR, SSIM, LPIPS split by difficulty level:
    Level 0 — Easy
    Level 1 — Medium
    Level 2 — Hard

Also compares against baseline RIFE (IFNet only, no Unet refinement).

Usage:
    python scripts/evaluate.py --unet-checkpoint checkpoints/unet_best.pth
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ATD12KDataset
from src.models.pipeline import AnimeInterpPipeline
from src.utils.metrics import MetricsCalculator


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(
    model:      AnimeInterpPipeline,
    loader:     DataLoader,
    metrics_fn: MetricsCalculator,
    device:     torch.device,
    label:      str,
) -> dict:
    """
    Run evaluation loop and return metrics split by difficulty level.

    Returns:
        dict mapping level (0/1/2) → dict of mean metrics
    """
    model.eval()

    # Accumulate per level: {level: {metric: [values]}}
    results = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {label}"):
            img0   = batch["frame_a"].to(device)
            img1   = batch["frame_b"].to(device)
            gt     = batch["frame_gt"].to(device)
            dist_a = batch["dist_a"].to(device)
            dist_b = batch["dist_b"].to(device)
            levels = batch["level"]   # list of ints

            pred = model(img0, img1, dist_a, dist_b)
            
            # Store per sample per level
            # levels is a tensor of shape (B,)
            for i, level in enumerate(levels):
                lv = int(level.item()) if hasattr(level, 'item') else int(level)
                # Compute per-sample metrics
                sample_pred = pred[i:i+1]
                sample_gt   = gt[i:i+1]
                sample_m    = metrics_fn.compute(sample_pred, sample_gt)
                for k, v in sample_m.items():
                    results[lv][k].append(v)

    # Compute means per level
    import numpy as np
    level_names = {0: "Easy", 1: "Medium", 2: "Hard"}
    summary = {}

    for level in sorted(results.keys()):
        name = level_names.get(level, f"Level {level}")
        summary[name] = {
            k: float(np.mean(v))
            for k, v in results[level].items()
        }

    # Overall
    all_vals = defaultdict(list)
    for level_data in results.values():
        for k, v in level_data.items():
            all_vals[k].extend(v)
    summary["Overall"] = {
        k: float(np.mean(v)) for k, v in all_vals.items()
    }

    return summary


def print_table(results: dict, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Split':<12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print(f"  {'-'*40}")
    for split, metrics in results.items():
        print(
            f"  {split:<12} "
            f"{metrics['psnr']:>8.2f} "
            f"{metrics['ssim']:>8.4f} "
            f"{metrics['lpips']:>8.4f}"
        )
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root",    default="data/datasets")
    parser.add_argument("--checkpoint",       default="checkpoints/flownet.pkl")
    parser.add_argument("--unet-checkpoint",  default="checkpoints/unet_best.pth")
    parser.add_argument("--batch-size",       type=int, default=4)
    parser.add_argument("--num-workers",      type=int, default=4)
    parser.add_argument("--size",             type=int, nargs=2,
                        default=[256, 256], metavar=("H", "W"))
    args = parser.parse_args()

    device = get_device()
    print(f"Evaluating on: {device}")

    # Dataset
    test_ds = ATD12KDataset(
        root=args.datasets_root,
        split="test",
        size=tuple(args.size),
        use_cached_dist=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Test samples: {len(test_ds)}")

    # Metrics
    metrics_fn = MetricsCalculator(device)

    # --- Baseline: IFNet only (no Unet refinement) ---
    print("\nEvaluating baseline (IFNet only)...")
    baseline_model = AnimeInterpPipeline(
        checkpoint_path=args.checkpoint,
        device=device,
    ).to(device)

    # Monkey-patch forward to skip Unet
    def baseline_forward(img0, img1, dist_a, dist_b, scale_list=None):
        with torch.no_grad():
            merged, flow, mask, w0, w1 = baseline_model.ifnet(
                img0, img1, scale_list or [4, 2, 1]
            )
        return merged
    baseline_model.forward = baseline_forward

    baseline_results = evaluate(
        baseline_model, test_loader, metrics_fn, device, "Baseline RIFE"
    )
    print_table(baseline_results, "Baseline RIFE (IFNet only)")

    # --- Fine-tuned: IFNet + UnetWithDistance ---
    print("Evaluating fine-tuned model...")
    finetuned_model = AnimeInterpPipeline(
        checkpoint_path=args.checkpoint,
        device=device,
    ).to(device)

    if os.path.exists(args.unet_checkpoint):
        ckpt = torch.load(args.unet_checkpoint, map_location=device)
        finetuned_model.unet.load_state_dict(ckpt["unet_state_dict"])
        print(f"Loaded Unet from {args.unet_checkpoint}")
    else:
        print(f"Warning: {args.unet_checkpoint} not found — using random Unet")

    finetuned_results = evaluate(
        finetuned_model, test_loader, metrics_fn, device, "Fine-tuned"
    )
    print_table(finetuned_results, "Fine-tuned (IFNet + UnetWithDistance)")

    # --- Delta table ---
    print(f"\n{'='*60}")
    print("  Delta (Fine-tuned - Baseline)")
    print(f"{'='*60}")
    print(f"  {'Split':<12} {'ΔPSNR':>8} {'ΔSSIM':>8} {'ΔLPIPS':>8}")
    print(f"  {'-'*40}")
    for split in baseline_results:
        b = baseline_results[split]
        f = finetuned_results[split]
        print(
            f"  {split:<12} "
            f"{f['psnr']-b['psnr']:>+8.2f} "
            f"{f['ssim']-b['ssim']:>+8.4f} "
            f"{f['lpips']-b['lpips']:>+8.4f}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()