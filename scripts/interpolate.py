"""
CLI inference script — interpolate between two frames.

Usage:
    python scripts/interpolate.py --frame-a input/a.jpg --frame-b input/b.jpg
    python scripts/interpolate.py --frame-a a.png --frame-b b.png --output result.png
    python scripts/interpolate.py --video input.mp4 --output output.mp4
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pipeline import AnimeInterpPipeline
from src.utils.edge import compute_distance_map, load_frame


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_and_prepare(
    path: str,
    size: tuple,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a frame from disk, compute its distance map,
    return both as tensors on device.

    Returns:
        frame_tensor: (1, 3, H, W) float32 [0,1]
        dist_tensor:  (1, 1, H, W) float32 [0,1]
    """
    frame = load_frame(path)
    img = Image.fromarray(frame).resize(
        (size[1], size[0]), Image.LANCZOS
    )
    frame = np.array(img)

    dist = compute_distance_map(frame)

    # (H, W, 3) → (1, 3, H, W)
    frame_t = torch.from_numpy(
        frame.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    # (H, W) → (1, 1, H, W)
    dist_t = torch.from_numpy(dist).unsqueeze(0).unsqueeze(0).to(device)

    return frame_t, dist_t


def interpolate_frames(
    model:   AnimeInterpPipeline,
    path_a:  str,
    path_b:  str,
    size:    tuple,
    device:  torch.device,
) -> np.ndarray:
    """
    Interpolate between two image files.

    Returns:
        np.ndarray (H, W, 3) uint8 — predicted middle frame
    """
    img0, dist_a = load_and_prepare(path_a, size, device)
    img1, dist_b = load_and_prepare(path_b, size, device)

    with torch.no_grad():
        pred = model(img0, img1, dist_a, dist_b)

    # (1, 3, H, W) → (H, W, 3) uint8
    pred_np = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return pred_np


def interpolate_video(
    model:      AnimeInterpPipeline,
    input_path: str,
    output_path: str,
    size:       tuple,
    device:     torch.device,
) -> None:
    """
    Interpolate every consecutive frame pair in a video (2x frame rate).
    """
    try:
        import cv2
    except ImportError:
        print("opencv-python required for video: pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps * 2,
                          (size[1], size[0]))

    ret, prev_frame = cap.read()
    if not ret:
        print("Could not read video")
        return

    prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)

    with tqdm(total=total, desc="Interpolating video") as pbar:
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

            # Write original frame
            out.write(cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2BGR))

            # Compute interpolated frame
            def frame_to_tensor(f):
                img = Image.fromarray(f).resize(
                    (size[1], size[0]), Image.LANCZOS
                )
                arr = np.array(img).astype(np.float32) / 255.0
                dist = compute_distance_map(np.array(img))
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                d = torch.from_numpy(dist).unsqueeze(0).unsqueeze(0).to(device)
                return t, d

            img0, dist_a = frame_to_tensor(prev_rgb)
            img1, dist_b = frame_to_tensor(curr_rgb)

            with torch.no_grad():
                pred = model(img0, img1, dist_a, dist_b)

            interp_np = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            out.write(cv2.cvtColor(interp_np, cv2.COLOR_RGB2BGR))

            prev_rgb = curr_rgb
            pbar.update(1)

    # Write last frame
    out.write(cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2BGR))
    cap.release()
    out.release()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Anime frame interpolation inference"
    )
    parser.add_argument("--frame-a",        type=str, default=None)
    parser.add_argument("--frame-b",        type=str, default=None)
    parser.add_argument("--video",          type=str, default=None)
    parser.add_argument("--output",         type=str, default="output.png")
    parser.add_argument("--checkpoint",     default="checkpoints/flownet.pkl")
    parser.add_argument("--unet-checkpoint",default="checkpoints/unet_best.pth")
    parser.add_argument("--size",           type=int, nargs=2,
                        default=[256, 256], metavar=("H", "W"))
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    model = AnimeInterpPipeline(
        checkpoint_path=args.checkpoint,
        device=device,
    ).to(device)

    if os.path.exists(args.unet_checkpoint):
        ckpt = torch.load(args.unet_checkpoint, map_location=device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        print(f"Loaded Unet from {args.unet_checkpoint}")
    else:
        print("Warning: no Unet checkpoint found — using random weights")

    model.eval()

    if args.video:
        interpolate_video(model, args.video, args.output,
                         tuple(args.size), device)
    elif args.frame_a and args.frame_b:
        result = interpolate_frames(
            model, args.frame_a, args.frame_b,
            tuple(args.size), device
        )
        Image.fromarray(result).save(args.output)
        print(f"Saved interpolated frame to {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()