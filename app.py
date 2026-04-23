"""
Generate side-by-side comparison images for the README demo.

Usage:
    python app.py --triplet-dir data/datasets/test_2k_540p/Disney_v4_0_000024_s2
    python app.py --frame-a frame1.png --frame-b frame3.png --gt frame2.png
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.pipeline import AnimeInterpPipeline
from src.utils.edge import compute_distance_map, load_frame


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def frame_to_tensor(path, size, device):
    frame = load_frame(path)
    img = Image.fromarray(frame).resize((size[1], size[0]), Image.LANCZOS)
    arr = np.array(img)
    dist = compute_distance_map(arr)
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    d = torch.from_numpy(dist).unsqueeze(0).unsqueeze(0).to(device)
    return t, d, arr


def add_label(img_arr, label):
    """Add a text label below an image."""
    img = Image.fromarray(img_arr)
    labeled = Image.new("RGB", (img.width, img.height + 24), (20, 20, 20))
    labeled.paste(img, (0, 0))
    draw = ImageDraw.Draw(labeled)
    draw.text((img.width // 2, img.height + 4), label, fill=(220, 220, 220), anchor="mt")
    return np.array(labeled)


def make_comparison(frame_a, pred, gt, frame_b, output_path):
    """Create a 4-panel side by side comparison."""
    panels = [
        add_label(frame_a, "Frame A (t=0)"),
        add_label(pred,    "Predicted (t=0.5)"),
        add_label(gt,      "Ground Truth (t=0.5)"),
        add_label(frame_b, "Frame B (t=1)"),
    ]
    combined = np.concatenate(panels, axis=1)
    Image.fromarray(combined).save(output_path)
    print(f"Saved comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplet-dir",     type=str, default=None)
    parser.add_argument("--frame-a",         type=str, default=None)
    parser.add_argument("--frame-b",         type=str, default=None)
    parser.add_argument("--gt",              type=str, default=None)
    parser.add_argument("--output",          type=str, default="comparison.png")
    parser.add_argument("--checkpoint",      default="checkpoints/flownet.pkl")
    parser.add_argument("--unet-checkpoint", default="checkpoints/unet_best.pth")
    parser.add_argument("--size",            type=int, nargs=2,
                        default=[256, 256], metavar=("H", "W"))
    args = parser.parse_args()

    # Resolve paths
    if args.triplet_dir:
        ext = ".png" if os.path.exists(os.path.join(args.triplet_dir, "frame1.png")) else ".jpg"
        path_a  = os.path.join(args.triplet_dir, f"frame1{ext}")
        path_gt = os.path.join(args.triplet_dir, f"frame2{ext}")
        path_b  = os.path.join(args.triplet_dir, f"frame3{ext}")
    else:
        path_a, path_gt, path_b = args.frame_a, args.gt, args.frame_b

    device = get_device()
    size   = tuple(args.size)

    model = AnimeInterpPipeline(
        checkpoint_path=args.checkpoint,
        device=device,
    ).to(device)

    if os.path.exists(args.unet_checkpoint):
        ckpt = torch.load(args.unet_checkpoint, map_location=device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        print(f"Loaded Unet from {args.unet_checkpoint}")

    model.eval()

    img0, dist_a, arr_a = frame_to_tensor(path_a,  size, device)
    img1, dist_b, arr_b = frame_to_tensor(path_b,  size, device)
    _,    _,      arr_gt = frame_to_tensor(path_gt, size, device)

    with torch.no_grad():
        pred = model(img0, img1, dist_a, dist_b)

    pred_np = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    make_comparison(arr_a, pred_np, arr_gt, arr_b, args.output)
    os.system(f"open {args.output}")


if __name__ == "__main__":
    main()