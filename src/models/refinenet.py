"""
Modified RIFE Unet (RefineNet) that accepts distance map channels.

Original Unet takes 17 input channels to down0:
    img0 (3) + img1 (3) + warped_img0 (3) + warped_img1 (3) + mask (1) + flow (4) = 17

Our modified version takes 19 input channels to down0:
    same as above + dist_a (1) + dist_b (1) = 19

The distance maps tell the network where flat colour regions are —
the regions most prone to colour bleeding during interpolation.

All other layers are identical to the original Unet.
Only down0's input channels change: Conv2(17, 2*c) → Conv2(19, 2*c)
The new 2 channels are zero-initialised so the network starts
behaving identically to pretrained RIFE before fine-tuning begins.
"""

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
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


c = 16


class UnetWithDistance(nn.Module):
    """
    Modified RIFE Unet that accepts distance map channels as extra input.

    The only architectural change vs original Unet:
        down0 input channels: 17 → 19 (added dist_a and dist_b)

    All other layers, channel sizes, and skip connections are identical.
    This means all pretrained weights except down0's first conv transfer
    directly with no modification.
    """

    def __init__(self):
        super(UnetWithDistance, self).__init__()
        # MODIFIED: 19 input channels instead of 17
        # Extra 2 channels are distance maps for img0 and img1
        self.down0 = Conv2(19, 2 * c)
        # All remaining layers identical to original Unet
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1,
                dist_a, dist_b):
        """
        Args:
            img0:         (B, 3, H, W) — first input frame
            img1:         (B, 3, H, W) — second input frame
            warped_img0:  (B, 3, H, W) — img0 warped toward t=0.5
            warped_img1:  (B, 3, H, W) — img1 warped toward t=0.5
            mask:         (B, 1, H, W) — blending mask from IFNet
            flow:         (B, 4, H, W) — optical flow from IFNet
            c0:           list of 4 context feature tensors from img0
            c1:           list of 4 context feature tensors from img1
            dist_a:       (B, 1, H, W) — distance map for img0
            dist_b:       (B, 1, H, W) — distance map for img1

        Returns:
            (B, 3, H, W) sigmoid output — residual correction to add to merged
        """
        # Concatenate all inputs including distance maps
        # Original: (img0, img1, warped_img0, warped_img1, mask, flow) = 17ch
        # Modified: add dist_a, dist_b = 19ch
        s0 = self.down0(
            torch.cat((img0, img1, warped_img0, warped_img1, mask, flow,
                       dist_a, dist_b), 1)
        )
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x)

    def load_from_rife(self, unet_state_dict: dict) -> None:
        """
        Load pretrained RIFE Unet weights into this modified Unet.

        All layers transfer exactly except down0.conv1.0.weight which
        needs expanding from 17 to 19 input channels.

        The 2 new channels (dist_a, dist_b) are zero-initialised so
        on the first forward pass the network output is identical to
        pretrained RIFE — as if the distance maps don't exist yet.
        Training then gradually learns to use them.

        Args:
            unet_state_dict: state dict from original RIFE Unet
                             (keys prefixed with 'unet.')
        """
        # Strip 'unet.' prefix from keys if present
        cleaned = {}
        for k, v in unet_state_dict.items():
            new_key = k.replace("unet.", "") if k.startswith("unet.") else k
            cleaned[new_key] = v

        # Get our current state dict
        own_state = self.state_dict()

        for name, param in cleaned.items():
            if name not in own_state:
                print(f"Skipping unexpected key: {name}")
                continue

            if name == "down0.conv1.0.weight":
                # This is the only layer that changes shape
                # Original: (out_ch, 17, kH, kW)
                # Ours:     (out_ch, 19, kH, kW)
                out_ch, _, kH, kW = param.shape
                new_weight = torch.zeros(out_ch, 19, kH, kW,
                                         dtype=param.dtype)
                # Copy original 17 channels exactly
                new_weight[:, :17, :, :] = param
                # Channels 17 and 18 (dist_a, dist_b) stay zero
                own_state[name].copy_(new_weight)
                print(f"Expanded {name}: {param.shape} → {new_weight.shape}")
            else:
                # All other layers copy directly
                own_state[name].copy_(param)

        self.load_state_dict(own_state)
        print("UnetWithDistance weights loaded successfully")