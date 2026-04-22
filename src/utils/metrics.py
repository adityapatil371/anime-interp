"""
Evaluation metrics for anime frame interpolation.

Three metrics, each measuring something different:
    PSNR  — pixel accuracy
    SSIM  — structural similarity
    LPIPS — perceptual quality (primary metric)

All functions expect torch tensors (B, 3, H, W) float32 [0, 1].
"""

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


class MetricsCalculator:
    """
    Computes PSNR, SSIM, and LPIPS for a batch of predictions.

    Args:
        device: torch device
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def compute(
        self,
        pred: torch.Tensor,
        gt:   torch.Tensor,
    ) -> dict:
        """
        Compute all three metrics for a batch.

        Args:
            pred: (B, 3, H, W) float32 [0, 1]
            gt:   (B, 3, H, W) float32 [0, 1]

        Returns:
            dict with mean PSNR, SSIM, LPIPS across the batch
        """
        # LPIPS — computed on GPU/MPS directly
        with torch.no_grad():
            pred_lpips = pred * 2 - 1
            gt_lpips   = gt   * 2 - 1
            lpips_val  = self.lpips_fn(pred_lpips, gt_lpips).mean().item()

        # PSNR and SSIM — computed on CPU with skimage
        pred_np = pred.cpu().numpy()   # (B, 3, H, W)
        gt_np   = gt.cpu().numpy()

        psnr_vals = []
        ssim_vals = []

        for i in range(pred_np.shape[0]):
            # skimage expects (H, W, C)
            p = pred_np[i].transpose(1, 2, 0)
            g = gt_np[i].transpose(1, 2, 0)

            psnr_vals.append(
                peak_signal_noise_ratio(g, p, data_range=1.0)
            )
            ssim_vals.append(
                structural_similarity(g, p, data_range=1.0,
                                     channel_axis=2)
            )

        return {
            "psnr":  float(np.mean(psnr_vals)),
            "ssim":  float(np.mean(ssim_vals)),
            "lpips": lpips_val,
        }