"""
Full anime interpolation pipeline.

Architecture:
    IFNet (frozen, pretrained) → coarse merged frame
    UnetWithDistance (trained from scratch) → refined frame

The UnetWithDistance takes:
    - coarse merged frame from IFNet
    - warped img0 and img1
    - distance maps for img0 and img1
    - flow and mask from IFNet

And outputs a residual correction that sharpens edges and
reduces colour bleeding in flat anime regions.
"""

import torch
import torch.nn as nn

from src.models.ifnet import IFNet
from src.models.refinenet import UnetWithDistance
from src.models.warplayer import warp


class AnimeInterpPipeline(nn.Module):
    """
    Full anime frame interpolation pipeline.

    Training strategy:
        IFNet:            frozen — pretrained on Vimeo90K
        UnetWithDistance: trained from scratch on ATD-12K anime data

    Args:
        checkpoint_path: path to pretrained flownet.pkl
        device:          torch device
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        super(AnimeInterpPipeline, self).__init__()
        self.device = device

        self.ifnet = IFNet()
        self.unet = UnetWithDistance()

        self._load_pretrained(checkpoint_path)
        self._freeze_ifnet()

    def _load_pretrained(self, checkpoint_path: str) -> None:
        print(f"Loading pretrained IFNet from {checkpoint_path}...")
        raw = torch.load(checkpoint_path, map_location=self.device)
        state_dict = {
            k.replace("module.", ""): v for k, v in raw.items()
        }
        missing, unexpected = self.ifnet.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        print("IFNet weights loaded.")

    def _freeze_ifnet(self) -> None:
        for param in self.ifnet.parameters():
            param.requires_grad = False
        self.ifnet.eval()

        frozen    = sum(p.numel() for p in self.ifnet.parameters())
        trainable = sum(p.numel() for p in self.unet.parameters()
                        if p.requires_grad)
        print(f"Frozen (IFNet):      {frozen:,}")
        print(f"Trainable (Unet):    {trainable:,}")

    def forward(
        self,
        img0:       torch.Tensor,
        img1:       torch.Tensor,
        dist_a:     torch.Tensor,
        dist_b:     torch.Tensor,
        scale_list: list = None,
    ) -> torch.Tensor:
        if scale_list is None:
            scale_list = [4, 2, 1]

        # IFNet: frozen, produces coarse frame
        with torch.no_grad():
            merged, flow, mask, warped_img0, warped_img1 = self.ifnet(
                img0, img1, scale_list
            )

        # Build context feature pyramid — 4 levels, matching Contextnet dims
        def make_context(img):
            f1 = torch.nn.functional.avg_pool2d(img, 2)
            f2 = torch.nn.functional.avg_pool2d(f1, 2)
            f3 = torch.nn.functional.avg_pool2d(f2, 2)
            f4 = torch.nn.functional.avg_pool2d(f3, 2)
            f1 = f1.repeat(1, 6, 1, 1)[:, :16, :, :]
            f2 = f2.repeat(1, 11, 1, 1)[:, :32, :, :]
            f3 = f3.repeat(1, 22, 1, 1)[:, :64, :, :]
            f4 = f4.repeat(1, 43, 1, 1)[:, :128, :, :]
            return [f1, f2, f3, f4]

        c0 = make_context(warped_img0)
        c1 = make_context(warped_img1)

        # UnetWithDistance: trainable refinement
        tmp = self.unet(
            img0, img1,
            warped_img0, warped_img1,
            mask, flow,
            c0, c1,
            dist_a, dist_b
        )

        # Residual correction: convert [0,1] → [-1,1] and add to merged
        res = tmp[:, :3] * 2 - 1
        pred = torch.clamp(merged + res, 0, 1)

        return pred