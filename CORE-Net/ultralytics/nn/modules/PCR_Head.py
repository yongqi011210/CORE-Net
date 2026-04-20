import math

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules import Detect, Conv
from ultralytics.utils.tal import dist2rbox


class OEB(nn.Module):
    """Orientation Enhancement Block"""

    def __init__(self, in_channels, ratio=0.25):
        super().__init__()
        mid_c = max(int(in_channels * ratio), 16)
        self.conv_angle = nn.Sequential(
            Conv(in_channels, mid_c, 3),
            Conv(mid_c, in_channels, 3)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat_angle = self.conv_angle(x)
        w = self.sigmoid(self.alpha)
        return w * x + (1 - w) * feat_angle


class ChannelAttention(nn.Module):
    """SE-style Channel Attention"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        w = self.fc(y).view(b, c, 1, 1)
        return x * w


class OAREU(nn.Module):
    """
    Orientation-Aware Regression Enhancement Unit
    """

    def __init__(self, channels: tuple, reduction=16):
        super().__init__()
        self.nl = len(channels)

        self.orientation_blocks = nn.ModuleList(OEB(c) for c in channels)
        self.channel_attn = nn.ModuleList(ChannelAttention(c, reduction) for c in channels)

        self.up_map = nn.ModuleList()
        self.down_map = nn.ModuleList()
        self.merge = nn.ModuleList()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        for i in range(self.nl):
            if i < self.nl - 1:
                self.up_map.append(Conv(channels[i + 1], channels[i], 1, 1))
            else:
                self.up_map.append(nn.Identity())

            if i > 0:
                self.down_map.append(Conv(channels[i - 1], channels[i], 3, 2))
            else:
                self.down_map.append(nn.Identity())

            in_cat = channels[i]
            if i < self.nl - 1:
                in_cat += channels[i]
            if i > 0:
                in_cat += channels[i]

            self.merge.append(Conv(in_cat, channels[i], 1, 1))

    def forward(self, feats):
        base_feats = [self.orientation_blocks[i](feats[i]) for i in range(self.nl)]
        outputs = []

        for i in range(self.nl):
            parts = [base_feats[i]]

            # 上采样来自更粗层
            if i < self.nl - 1:
                up = self.up_map[i](base_feats[i + 1])
                up = self.upsample(up)
                if up.shape[-2:] != base_feats[i].shape[-2:]:
                    up = F.interpolate(up, size=base_feats[i].shape[-2:], mode="nearest")
                parts.append(up)

            # 下采样来自更细层
            if i > 0:
                down = self.down_map[i](base_feats[i - 1])
                if down.shape[-2:] != base_feats[i].shape[-2:]:
                    down = F.interpolate(down, size=base_feats[i].shape[-2:], mode="nearest")
                parts.append(down)

            fused = torch.cat(parts, dim=1)
            fused = self.merge[i](fused)
            fused = self.channel_attn[i](fused)
            outputs.append(fused)

        return outputs


class CrossStageAttention(nn.Module):
    """Cross-Stage Attention (CSA)"""

    def __init__(self, channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or max(1, channels // 2)
        self.query_conv = nn.Conv2d(channels, inter_channels, 1)
        self.key_conv = nn.Conv2d(channels, inter_channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, prev_feat):
        if prev_feat is None:
            return x

        if prev_feat.shape[1] != x.shape[1]:
            raise RuntimeError(f"CSA channel mismatch: x={tuple(x.shape)}, prev={tuple(prev_feat.shape)}")

        q = self.query_conv(x)
        k = self.key_conv(prev_feat)
        v = self.value_conv(prev_feat)

        attn = torch.sigmoid(torch.mean(q * k, dim=1, keepdim=True))  # [B,1,H,W]
        return x + self.gamma.to(x.dtype) * (attn * v)


class CSAOrIdentity(nn.Module):
    """统一 CSA / Identity 接口，避免 ModuleList 中放 None"""

    def __init__(self, channels: int, enabled: bool):
        super().__init__()
        self.enabled = bool(enabled)
        self.block = CrossStageAttention(channels) if self.enabled else nn.Identity()

    def forward(self, x, prev_feat=None):
        if (not self.enabled) or (prev_feat is None):
            return x
        return self.block(x, prev_feat)


class PCR_Head(Detect):
    """
    Progressive Cascade Rotation Head
    原 OBB_CascadeHead，现统一命名为 PCR_Head
    """

    def __init__(
        self,
        nc=80,
        ne=1,
        cascade_stages=2,
        use_csa=True,
        use_oareu=True,
        return_all_stages=False,
        ch=(),
        debug=False,
    ):
        super().__init__(nc, ch)

        self.ne = int(ne)
        self.nl = len(ch)
        self.cascade_stages = int(cascade_stages)
        self.use_csa = bool(use_csa)
        self.use_oareu = bool(use_oareu)
        self.return_all_stages = bool(return_all_stages)
        self.debug = bool(debug)

        # Ultralytics build-time stride inference gate
        self._stride_infer = True

        channels = tuple(ch)
        c4 = max(min(channels) // 4, 16, self.ne)

        # Optional OAREU
        self.oareu = OAREU(channels) if self.use_oareu else nn.Identity()

        # Cascade branches
        self.trunks = nn.ModuleList()
        self.angle_preds = nn.ModuleList()
        self.angle_embeds = nn.ModuleList()

        for _ in range(self.cascade_stages):
            trunk_s = nn.ModuleList()
            pred_s = nn.ModuleList()
            emb_s = nn.ModuleList()

            for xch in channels:
                trunk_s.append(nn.Sequential(
                    Conv(xch, c4, 3),
                    Conv(c4, c4, 3),
                ))
                pred_s.append(nn.Conv2d(c4, self.ne, 1))
                emb_s.append(nn.Conv2d(self.ne, xch, 1))

            self.trunks.append(trunk_s)
            self.angle_preds.append(pred_s)
            self.angle_embeds.append(emb_s)

        # CSA blocks
        self.csa_blocks = nn.ModuleList()
        for s in range(self.cascade_stages):
            csa_s = nn.ModuleList()
            for xch in channels:
                csa_s.append(CSAOrIdentity(xch, enabled=(self.use_csa and s > 0)))
            self.csa_blocks.append(csa_s)

        print(f"[INFO] PCR_Head: stages={self.cascade_stages}, CSA={self.use_csa}, OAREU={self.use_oareu}")

    def forward(self, x):
        """
        x: list/tuple of multi-level features
        """
        bs = x[0].shape[0]

        # Step1: optional OAREU
        feats = self.oareu(x)

        # Step2: cascade angle refinement (+ optional CSA)
        prev_embed = None
        all_angles = []

        for s in range(self.cascade_stages):
            angle_list = []
            curr_embed = []

            for i in range(self.nl):
                xi = feats[i]

                # CSA
                xi = self.csa_blocks[s][i](xi, None if prev_embed is None else prev_embed[i])

                trunk = self.trunks[s][i](xi)                  # [B, c4, H, W]
                angle_logits = self.angle_preds[s][i](trunk)   # [B, ne, H, W]

                angle_list.append(angle_logits.view(bs, self.ne, -1))

                emb = torch.tanh(self.angle_embeds[s][i](angle_logits))
                curr_embed.append(emb)

            angle_stage = torch.cat(angle_list, dim=2)
            angle_stage = (angle_stage.sigmoid() - 0.25) * math.pi
            all_angles.append(angle_stage)

            if self.debug:
                mn, mx = angle_stage.min().item(), angle_stage.max().item()
                print(f"[Stage {s}] angle range=({mn:.3f}, {mx:.3f})")

            prev_embed = curr_embed

        final_angle = all_angles[-1]
        if not self.training:
            self.angle = final_angle

        # Step3: YOLO detect output
        det = super().forward(feats)

        # Fix for Ultralytics build-time stride inference
        if self._stride_infer:
            if getattr(self, "stride", None) is not None:
                self._stride_infer = False
            return det

        # Step4: outputs
        if self.training:
            if self.return_all_stages:
                return det, final_angle, all_angles
            return det, final_angle

        # Inference / Export
        if isinstance(det, torch.Tensor):
            return torch.cat([det, final_angle], 1)

        pred = det[0]
        aux = det[1] if len(det) > 1 else None
        out_pred = torch.cat([pred, final_angle], 1)
        return (out_pred, (aux, final_angle))

    def decode_bboxes(self, bboxes, anchors):
        return dist2rbox(bboxes, self.angle, anchors, dim=1)