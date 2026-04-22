"""
Loss functions for training UnetWithDistance.

We use a combination of two losses:
    1. LPIPS (perceptual loss) — primary loss
       Measures perceptual similarity using deep features from a
       pretrained VGG network. Correlates with human visual quality.
       This is what matters for anime — sharp lines, clean fills.

    2. L1 loss — secondary loss, weight 0.1
       Pixel-level accuracy. Prevents the model from producing
       perceptually plausible but structurally wrong outputs.
       LPIPS alone can sometimes hallucinate plausible-looking
       but geometrically incorrect frames.

Combined: loss = lpips + 0.1 * l1

Why not L2 (MSE)?
    L2 penalises large errors heavily, which encourages blurry
    predictions — the model averages uncertain regions to minimise
    squared error. L1 is more tolerant of outliers and produces
    sharper outputs.

Why LPIPS as primary?
    Our evaluation showed PSNR/SSIM disagree with LPIPS on smear
    frames — the very frames where our model should behave differently
    from baseline RIFE. Training with LPIPS directly optimises for
    the metric we care about most.
"""

import torch
import torch.nn as nn
import lpips


class AnimeLoss(nn.Module):
    """
    Combined LPIPS + L1 loss for anime frame interpolation.

    Args:
        lpips_weight: weight for LPIPS loss (default 1.0)
        l1_weight:    weight for L1 loss (default 0.1)
        device:       torch device
    """

    def __init__(
        self,
        lpips_weight: float = 1.0,
        l1_weight: float = 0.1,
        device: torch.device = None,
    ):
        super(AnimeLoss, self).__init__()

        self.lpips_weight = lpips_weight
        self.l1_weight = l1_weight

        # LPIPS uses pretrained VGG features
        # net='vgg' gives better perceptual quality than 'alex'
        # for fine-grained image details like anime line art
        self.lpips_fn = lpips.LPIPS(net='vgg')

        if device is not None:
            self.lpips_fn = self.lpips_fn.to(device)

        # Freeze LPIPS — it's a fixed perceptual metric, not trained
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

        self.l1_fn = nn.L1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            pred: (B, 3, H, W) float32 [0, 1] — model prediction
            gt:   (B, 3, H, W) float32 [0, 1] — ground truth frame

        Returns:
            total_loss: scalar tensor — backpropagated
            loss_dict:  dict with individual loss values for logging
        """
        # LPIPS expects inputs in [-1, 1] range
        # Our tensors are in [0, 1] so we rescale
        pred_lpips = pred * 2 - 1
        gt_lpips   = gt   * 2 - 1

        # LPIPS returns (B, 1, 1, 1) — mean across batch
        lpips_loss = self.lpips_fn(pred_lpips, gt_lpips).mean()

        # L1 works on [0, 1] directly
        l1_loss = self.l1_fn(pred, gt)

        total_loss = (self.lpips_weight * lpips_loss +
                      self.l1_weight    * l1_loss)

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_lpips": lpips_loss.item(),
            "loss_l1":    l1_loss.item(),
        }