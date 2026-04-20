import torch
import torch.nn as nn
import torch.nn.functional as F


class RCFP(nn.Module):


    def __init__(self, channels, num_dirs: int = 4, reduction: int = 4):

        super().__init__()
        c = channels if isinstance(channels, int) else channels[0]
        self.c = c
        self.num_dirs = num_dirs

        self.dir_conv = nn.Conv2d(
            c, c * num_dirs, kernel_size=3, padding=1,
            groups=c, bias=False
        )
        self.dir_bn = nn.BatchNorm2d(c * num_dirs)
        self.dir_act = nn.ReLU(inplace=True)

        self.dir_weight_branch = nn.Sequential(
            nn.Conv2d(c, c // reduction, 1, bias=False),
            nn.BatchNorm2d(c // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // reduction, num_dirs, 1)  # [B, num_dirs, H, W]
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // reduction, c, 1),
            nn.Sigmoid()
        )

        self.proj = nn.Conv2d(c, c, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c)

        self.global_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # dir_feat_raw: [B, C*num_dirs, H, W]
        dir_feat_raw = self.dir_act(self.dir_bn(self.dir_conv(x)))

        dir_feat = dir_feat_raw.view(B, self.num_dirs, C, H, W)

        dir_logits = self.dir_weight_branch(x)

        dir_weights = torch.softmax(dir_logits, dim=1).unsqueeze(2)

        # [B, C, H, W]
        feat_oriented = (dir_feat * dir_weights).sum(dim=1)

        feat_oriented = self.proj_bn(self.proj(feat_oriented))

        ch_gate = self.channel_gate(x)  # [B, C, 1, 1]

        g = torch.sigmoid(self.global_gate)  # (0, 1) 之间

        # out = x + g * ch_gate * (feat_oriented - x)
        out = x + g * ch_gate * (feat_oriented - x)

        return out