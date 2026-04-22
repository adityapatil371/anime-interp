import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Literal

from src.utils.edge import compute_distance_map


class ATD12KDataset(Dataset):
    """
    PyTorch Dataset for the ATD-12K anime frame interpolation dataset.

    Each sample contains:
        - frame_a:      Tensor (3, H, W) — first frame (t=0)
        - frame_b:      Tensor (3, H, W) — last frame (t=1)
        - frame_gt:     Tensor (3, H, W) — ground truth middle frame (t=0.5)
        - dist_a:       Tensor (1, H, W) — distance map for frame_a
        - dist_b:       Tensor (1, H, W) — distance map for frame_b
        - level:        int — difficulty level (0=easy, 1=medium, 2=hard)
                        Only available for test split. -1 for train split.
        - triplet_name: str — folder name, useful for debugging

    Folder structure expected:
        data/datasets/
            train_10k/
                <triplet_name>/
                    frame1.jpg   ← frame_a
                    frame2.jpg   ← frame_gt
                    frame3.jpg   ← frame_b
            test_2k_540p/
                <triplet_name>/
                    frame1.jpg
                    frame2.jpg
                    frame3.jpg
            test_2k_annotations/
                <triplet_name>/
                    <triplet_name>.json   ← contains "level" field
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        size: tuple[int, int] = (256, 256),
        use_cached_dist: bool = True,
    ):
        """
        Args:
            root:             path to data/datasets/ folder
            split:            "train" or "test"
            size:             (H, W) to resize frames to. Default 256x256.
                              Training at full 1080p on M1 is too slow.
            use_cached_dist:  if True, load precomputed .npy distance maps
                              from disk instead of computing on the fly.
                              Run scripts/precompute_distances.py first.
        """
        self.root = root
        self.split = split
        self.size = size
        self.use_cached_dist = use_cached_dist

        # Set folder paths based on split
        if split == "train":
            self.frames_dir = os.path.join(root, "train_10k")
            self.ann_dir = None   # No annotations for train split
        else:
            self.frames_dir = os.path.join(root, "test_2k_540p")
            self.ann_dir = os.path.join(root, "test_2k_annotations")

        # Build list of all triplet folder names
        # sorted() ensures consistent ordering across runs
        self.triplets = sorted(os.listdir(self.frames_dir))
        self.ext = ".jpg" if split == "train" else ".png"

        # Load difficulty annotations for test split
        # Maps triplet_name → level (0, 1, or 2)
        self.annotations = {}
        if self.ann_dir is not None:
            self._load_annotations()

    def _load_annotations(self) -> None:
        """
        Read all JSON annotation files and store level per triplet.
        Only called for test split.
        """
        for triplet_name in self.triplets:
            json_path = os.path.join(
                self.ann_dir,
                triplet_name,
                f"{triplet_name}.json"
            )
            if os.path.exists(json_path):
                with open(json_path) as f:
                    data = json.load(f)
                    # level: 0=easy, 1=medium, 2=hard
                    self.annotations[triplet_name] = data.get("level", -1)

    def _load_frame(self, path: str) -> np.ndarray:
        """
        Load a JPEG frame, resize to self.size, return as uint8 numpy array.

        Args:
            path: absolute path to frame .jpg file

        Returns:
            np.ndarray (H, W, 3) uint8
        """
        img = Image.open(path).convert("RGB")
        # LANCZOS gives best quality for downscaling
        img = img.resize((self.size[1], self.size[0]), Image.LANCZOS)
        return np.array(img)

    def _get_dist_map(self, frame: np.ndarray, frame_path: str) -> np.ndarray:
        """
        Get distance map for a frame — either from cache or computed fresh.

        Cache path is frame_path with .jpg replaced by _dist.npy
        e.g. frame1.jpg → frame1_dist.npy

        Args:
            frame:      np.ndarray (H, W, 3) uint8 — already resized
            frame_path: original path to the .jpg (used to find cache)

        Returns:
            np.ndarray (H, W) float32
        """
        if self.use_cached_dist:
            # Build cache path: replace .jpg with _dist.npy
            cache_path = os.path.splitext(frame_path)[0] + "_dist.npy"
            if os.path.exists(cache_path):
                return np.load(cache_path)
            # Cache miss — compute and warn
            # This means precompute_distances.py hasn't been run yet
            print(f"Cache miss for {cache_path} — computing on the fly")

        return compute_distance_map(frame)

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert uint8 numpy frame (H, W, 3) to float32 tensor (3, H, W)
        normalised to [0, 1].

        Args:
            frame: np.ndarray (H, W, 3) uint8

        Returns:
            torch.Tensor (3, H, W) float32
        """
        # Normalise to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        # (H, W, 3) → (3, H, W) for PyTorch convention
        frame = np.transpose(frame, (2, 0, 1))
        return torch.from_numpy(frame)

    def _dist_to_tensor(self, dist: np.ndarray) -> torch.Tensor:
        """
        Convert float32 distance map (H, W) to tensor (1, H, W).
        Adds channel dimension for concatenation with RGB channels.

        Args:
            dist: np.ndarray (H, W) float32

        Returns:
            torch.Tensor (1, H, W) float32
        """
        # (H, W) → (1, H, W)
        dist = dist[np.newaxis, :, :]
        return torch.from_numpy(dist)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict:
        """
        Load one triplet and return all components as tensors.

        Returns dict with keys:
            frame_a:      Tensor (3, H, W)
            frame_b:      Tensor (3, H, W)
            frame_gt:     Tensor (3, H, W)
            dist_a:       Tensor (1, H, W)
            dist_b:       Tensor (1, H, W)
            level:        int (-1 for train, 0/1/2 for test)
            triplet_name: str
        """
        triplet_name = self.triplets[idx]
        triplet_dir = os.path.join(self.frames_dir, triplet_name)

        # Build paths to all three frames
        # frame1 = A (t=0), frame2 = GT (t=0.5), frame3 = B (t=1)
        path_a  = os.path.join(triplet_dir, f"frame1{self.ext}")
        path_gt = os.path.join(triplet_dir, f"frame2{self.ext}")
        path_b  = os.path.join(triplet_dir, f"frame3{self.ext}")

        # Load and resize all three frames
        frame_a  = self._load_frame(path_a)
        frame_b  = self._load_frame(path_b)
        frame_gt = self._load_frame(path_gt)

        # Get distance maps (from cache or computed)
        dist_a = self._get_dist_map(frame_a, path_a)
        dist_b = self._get_dist_map(frame_b, path_b)

        # Get difficulty level — -1 for train, 0/1/2 for test
        level = self.annotations.get(triplet_name, -1)

        return {
            "frame_a":      self._to_tensor(frame_a),
            "frame_b":      self._to_tensor(frame_b),
            "frame_gt":     self._to_tensor(frame_gt),
            "dist_a":       self._dist_to_tensor(dist_a),
            "dist_b":       self._dist_to_tensor(dist_b),
            "level":        level,
            "triplet_name": triplet_name,
        }