import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.warplayer import warp


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=True),
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    """
    Single IFBlock — estimates residual flow and mask at one scale.

    Architecture reverse-engineered from checkpoint weights:
    - conv0:     two strided convs for downsampling (in_planes → c//2 → c)
    - convblock0-3: four residual conv blocks (c → c, 2 convs each)
    - conv1:     two deconvs for flow output (c → c//2 → 4 channels)
    - conv2:     two deconvs for mask output (c → c//2 → 1 channel)

    All three student blocks take 11 input channels:
        img0(3) + img1(3) + flow(4) + mask(1) = 11
    Flow and mask start as zeros for block0.

    block_tea takes 14 channels (adds gt frame during training):
        img0(3) + img1(3) + flow(4) + mask(1) + gt(3) = 14
    """

    def __init__(self, in_planes, c=90):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))

        # Flow output: c → c//2 → 4 channels
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1)
        )
        # Mask output: c → c//2 → 1 channel
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1)
        )

    def forward(self, x, flow, mask, scale):
        # Rescale input for multi-scale processing
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale,
                              mode="bilinear", align_corners=False)
        if flow is not None:
            flow_scaled = F.interpolate(
                flow, scale_factor=1. / scale,
                mode="bilinear", align_corners=False
            ) * (1. / scale)
            mask_scaled = F.interpolate(
                mask, scale_factor=1. / scale,
                mode="bilinear", align_corners=False
            )
            x = torch.cat((x, flow_scaled, mask_scaled), 1)

        x = self.conv0(x)
        x = self.convblock0(x) + x
        x = self.convblock1(x) + x
        x = self.convblock2(x) + x
        x = self.convblock3(x) + x

        flow_res = self.conv1(x)
        mask_res = self.conv2(x)

        # Rescale outputs back to original resolution
        if scale != 1:
            flow_res = F.interpolate(
                flow_res, scale_factor=scale,
                mode="bilinear", align_corners=False
            ) * scale
            mask_res = F.interpolate(
                mask_res, scale_factor=scale,
                mode="bilinear", align_corners=False
            )

        return flow_res, mask_res


class IFNet(nn.Module):
    """
    Full IFNet: three student IFBlocks at scales [4, 2, 1] + one teacher block.

    Forward pass:
        1. block0 processes at scale 4 with zero flow/mask initialisation
        2. block1 processes at scale 2, refining block0's output
        3. block2 processes at scale 1, final refinement
        4. block_tea (training only): uses gt frame to distill knowledge

    Returns coarse merged frame at finest scale.
    """

    def __init__(self):
        super(IFNet, self).__init__()
        # All student blocks: 11 = img0(3) + img1(3) + flow(4) + mask(1)
        self.block0 = IFBlock(11, c=90)
        self.block1 = IFBlock(11, c=90)
        self.block2 = IFBlock(11, c=90)
        # Teacher block: 14 = 11 + gt(3), only used during training
        self.block_tea = IFBlock(14, c=90)

    def forward(self, img0, img1, scale_list=None, timestep=0.5):
        """
        Args:
            img0:       (B, 3, H, W) float32 [0,1]
            img1:       (B, 3, H, W) float32 [0,1]
            scale_list: list of 3 scales (default [4, 2, 1])
            timestep:   interpolation time (default 0.5)

        Returns:
            merged: (B, 3, H, W) coarse interpolated frame
            flow:   (B, 4, H, W) final optical flow
            mask:   (B, 1, H, W) blending mask
        """
        if scale_list is None:
            scale_list = [4, 2, 1]

        imgs = torch.cat((img0, img1), 1)   # (B, 6, H, W)
        flow = None
        mask = None

        for i, (block, scale) in enumerate(
            zip([self.block0, self.block1, self.block2], scale_list)
        ):
            if flow is None:
                # First block: initialise flow and mask as zeros
                flow = torch.zeros(
                    img0.shape[0], 4, img0.shape[2], img0.shape[3],
                    device=img0.device
                )
                mask = torch.zeros(
                    img0.shape[0], 1, img0.shape[2], img0.shape[3],
                    device=img0.device
                )

            flow_res, mask_res = block(imgs, flow, mask, scale)
            flow = flow + flow_res
            mask = mask + mask_res

        # Final warp using accumulated flow
        mask = torch.sigmoid(mask)
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        merged = warped_img0 * mask + warped_img1 * (1 - mask)

        return merged, flow, mask, warped_img0, warped_img1