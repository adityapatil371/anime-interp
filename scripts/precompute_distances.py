"""
Precompute distance transform maps for all frames in ATD-12K.

Run this ONCE before training. Saves .npy files next to each frame:
    frame1.jpg → frame1_dist.npy
    frame3.jpg → frame3_dist.npy

We skip frame2 (ground truth) — we never feed GT into the model,
so it doesn't need a distance map.

Usage:
    python scripts/precompute_distances.py
    python scripts/precompute_distances.py --split train
    python scripts/precompute_distances.py --split test
    python scripts/precompute_distances.py --workers 4
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

# Make src/ importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.edge import compute_distance_map


def get_all_frame_paths(
    datasets_root: str,
    split: str
) -> list[tuple[str, str]]:
    """
    Walk the dataset folder and collect paths to frame1 and frame3
    for every triplet in the given split.

    Args:
        datasets_root: path to data/datasets/
        split:         "train", "test", or "both"

    Returns:
        List of (frame_path, ext) tuples
        e.g. (".../frame1.jpg", ".jpg")
    """
    paths = []

    splits_to_process = []
    if split in ("train", "both"):
        splits_to_process.append(("train_10k", ".jpg"))
    if split in ("test", "both"):
        splits_to_process.append(("test_2k_540p", ".png"))

    for folder, ext in splits_to_process:
        frames_dir = os.path.join(datasets_root, folder)
        triplets = sorted(os.listdir(frames_dir))

        for triplet_name in triplets:
            triplet_dir = os.path.join(frames_dir, triplet_name)
            if not os.path.isdir(triplet_dir):
                continue

            # Only frame1 and frame3 — never frame2 (ground truth)
            for frame_name in (f"frame1{ext}", f"frame3{ext}"):
                frame_path = os.path.join(triplet_dir, frame_name)
                if os.path.exists(frame_path):
                    paths.append((frame_path, ext))

    return paths


def process_one_frame(args: tuple[str, str, tuple[int, int]]) -> str:
    """
    Compute and save distance map for a single frame.

    Args:
        args: (frame_path, ext, size)
            frame_path: path to frame jpg/png
            ext:        file extension
            size:       (H, W) to resize to before computing

    Returns:
        cache_path where .npy was saved, or "SKIP" if already exists
    """
    frame_path, ext, size = args

    # Build output path: frame1.jpg → frame1_dist.npy
    cache_path = os.path.splitext(frame_path)[0] + "_dist.npy"

    # Skip if already computed — allows resuming interrupted runs
    if os.path.exists(cache_path):
        return "SKIP"

    # Load frame and resize
    img = Image.open(frame_path).convert("RGB")
    img = img.resize((size[1], size[0]), Image.LANCZOS)
    frame = np.array(img)

    # Compute distance map
    dist = compute_distance_map(frame)

    # Save as float32 numpy array
    np.save(cache_path, dist)

    return cache_path


def main():
    parser = argparse.ArgumentParser(
        description="Precompute distance transform maps for ATD-12K"
    )
    parser.add_argument(
        "--datasets-root",
        type=str,
        default="data/datasets",
        help="Path to data/datasets/ folder"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Which split to process (default: both)"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="Resize frames to this size before computing (default: 256 256)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    args = parser.parse_args()

    size = tuple(args.size)

    print(f"Collecting frame paths for split='{args.split}'...")
    paths = get_all_frame_paths(args.datasets_root, args.split)
    print(f"Found {len(paths)} frames to process")

    # Build args list for parallel processing
    # Each worker gets (frame_path, ext, size)
    worker_args = [(path, ext, size) for path, ext in paths]

    # Count how many already exist
    already_done = sum(
        1 for path, _ in paths
        if os.path.exists(os.path.splitext(path)[0] + "_dist.npy")
    )
    print(f"Already computed: {already_done} / {len(paths)}")
    print(f"To compute: {len(paths) - already_done}")

    if len(paths) - already_done == 0:
        print("All distance maps already computed. Nothing to do.")
        return

    # Process in parallel with progress bar
    skipped = 0
    computed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one_frame, arg): arg
            for arg in worker_args
        }

        with tqdm(total=len(paths), desc="Computing distance maps") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "SKIP":
                    skipped += 1
                else:
                    computed += 1
                pbar.update(1)
                pbar.set_postfix(computed=computed, skipped=skipped)

    print(f"\nDone. Computed: {computed}, Skipped: {skipped}")
    print("Distance maps saved as _dist.npy next to each frame.")


if __name__ == "__main__":
    main()