import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image


def compute_distance_map(frame: np.ndarray, edge_threshold: float = 0.1) -> np.ndarray:
    """
    Compute a distance transform map for a single anime frame.
    
    For each pixel, computes the Euclidean distance to the nearest edge
    (outline/line art). Large values = deep inside flat colour region.
    Small values = close to an outline.
    
    This map is fed to RefineNet as an extra input channel so it knows
    where the dangerous flat colour regions are — the regions where
    colour bleeding occurs during interpolation.
    
    Args:
        frame:          np.ndarray of shape (H, W, 3), dtype uint8, range [0, 255]
        edge_threshold: gradient magnitude above this is considered an edge.
                        0.1 works well for anime — lines are high contrast.
    
    Returns:
        distance_map:   np.ndarray of shape (H, W), dtype float32
                        Values normalised to [0, 1] range.
                        0.0 = on an edge, 1.0 = furthest from any edge.
    """
    # Step 1: Convert RGB frame to grayscale
    # We use a standard luminance formula — human eyes are most sensitive
    # to green, least to blue. This gives perceptually accurate edges.
    # Shape goes from (H, W, 3) → (H, W)
    gray = (
        0.299 * frame[:, :, 0] +   # Red channel
        0.587 * frame[:, :, 1] +   # Green channel
        0.114 * frame[:, :, 2]     # Blue channel
    ).astype(np.float32) / 255.0   # Normalise to [0, 1]

    # Step 2: Compute image gradients using numpy
    # np.gradient returns rate of change at each pixel
    # gx = how fast values change left-to-right (horizontal edges)
    # gy = how fast values change top-to-bottom (vertical edges)
    # Both have shape (H, W)
    gy, gx = np.gradient(gray)

    # Step 3: Gradient magnitude — combines both directions
    # At a strong edge, either gx or gy will be large
    # sqrt(gx² + gy²) gives the overall edge strength at each pixel
    # Shape: (H, W)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Step 4: Threshold to get binary edge mask
    # Pixels above threshold = edge (True), below = not edge (False)
    # This is what distance_transform_edt needs as input
    edge_mask = magnitude > edge_threshold   # Shape: (H, W), dtype bool

    # Step 5: Distance transform
    # distance_transform_edt measures distance to nearest True pixel
    # BUT it measures distance to nearest ZERO, not nearest ONE
    # So we invert: edges become 0, non-edges become 1
    # Result: each non-edge pixel gets its distance to the nearest edge
    # Shape: (H, W), dtype float64
    distance_map = distance_transform_edt(~edge_mask)

    # Step 6: Normalise to [0, 1]
    # Divide by max value so all distances are on the same scale
    # regardless of image resolution
    max_val = distance_map.max()
    if max_val > 0:
        distance_map = distance_map / max_val

    return distance_map.astype(np.float32)


def load_frame(path: str) -> np.ndarray:
    """
    Load an image from disk as a numpy array.
    
    Args:
        path: path to image file (jpg or png)
    
    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8, range [0, 255]
    """
    return np.array(Image.open(path).convert("RGB"))


def frames_to_tensor_input(
    frame: np.ndarray,
    distance_map: np.ndarray
) -> np.ndarray:
    """
    Combine an RGB frame and its distance map into a single 4-channel array.
    
    RefineNet expects input as (C, H, W) for PyTorch.
    We return (4, H, W): 3 RGB channels + 1 distance map channel.
    
    The RGB channels are normalised to [0, 1].
    The distance map is already in [0, 1] from compute_distance_map.
    
    Args:
        frame:        np.ndarray (H, W, 3), dtype uint8
        distance_map: np.ndarray (H, W),    dtype float32
    
    Returns:
        np.ndarray (4, H, W), dtype float32
    """
    # Normalise RGB to [0, 1] and transpose from (H, W, 3) to (3, H, W)
    rgb = frame.astype(np.float32) / 255.0
    rgb = rgb.transpose(2, 0, 1)   # (H, W, 3) → (3, H, W)

    # Add channel dimension to distance map: (H, W) → (1, H, W)
    dist = distance_map[np.newaxis, :, :]

    # Stack along channel axis: (3, H, W) + (1, H, W) → (4, H, W)
    return np.concatenate([rgb, dist], axis=0)