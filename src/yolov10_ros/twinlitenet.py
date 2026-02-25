"""
twinlitenet.py — TwinLiteNet model architecture
================================================
Lightweight dual-head segmentation network for simultaneous drivable area
and lane line prediction.  Architecture from:

  "TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and
   Lane Segmentation"  — Chau Huy et al.
  https://github.com/chequanghuy/TwinLiteNet

Two output heads share a single ESPNet encoder:
  Head 1 (classifier_1):  Drivable area  — (B, 2, H, W) logits
  Head 2 (classifier_2):  Lane lines     — (B, 2, H, W) logits

Both outputs are 2-class (background / foreground); apply torch.max over
the channel dimension to obtain binary pixel labels.

Input convention:
  • Size:    640 × 360  (W × H)
  • Channels: 3 (RGB order)
  • Range:   [0.0, 1.0]  (divide uint8 by 255.0)

Usage:
    from yolov10_ros.twinlitenet import TwinLiteNet
    import torch

    model = TwinLiteNet()
    state = torch.load('twinlitenet_best.pth', map_location='cpu')
    # Weights were saved with DataParallel — strip 'module.' prefix.
    clean = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(clean)
    model.eval()
"""

import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter, Softmax


# ── Attention modules ─────────────────────────────────────────────────────────

class PAM_Module(Module):
    """Position Attention Module (SAGAN-based)."""

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_dim, in_dim,      kernel_size=1)
        self.gamma      = Parameter(torch.zeros(1))
        self.softmax    = Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        k = self.key_conv(x).view(B, -1, W * H)
        attn = self.softmax(torch.bmm(q, k))
        v = self.value_conv(x).view(B, -1, W * H)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class CAM_Module(Module):
    """Channel Attention Module."""

    def __init__(self, in_dim):
        super().__init__()
        self.gamma   = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = x.view(B, C, -1)
        k = x.view(B, C, -1).permute(0, 2, 1)
        energy = torch.bmm(q, k)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attn = self.softmax(energy)
        v = x.view(B, C, -1)
        out = torch.bmm(attn, v).view(B, C, H, W)
        return self.gamma * out + x


# ── Core building blocks ──────────────────────────────────────────────────────

class CBR(nn.Module):
    """Conv → BatchNorm → PReLU."""

    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        pad = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=pad, bias=False)
        self.bn   = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act  = nn.PReLU(nOut)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CB(nn.Module):
    """Conv → BatchNorm (no activation)."""

    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        pad = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=pad, bias=False)
        self.bn   = nn.BatchNorm2d(nOut, eps=1e-3)

    def forward(self, x):
        return self.bn(self.conv(x))


class C(nn.Module):
    """Plain Conv2d."""

    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        pad = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=pad, bias=False)

    def forward(self, x):
        return self.conv(x)


class CDilated(nn.Module):
    """Dilated Conv2d."""

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        pad = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride,
                              padding=pad, dilation=d, bias=False)

    def forward(self, x):
        return self.conv(x)


class BR(nn.Module):
    """BatchNorm → PReLU."""

    def __init__(self, nOut):
        super().__init__()
        self.bn  = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, x):
        return self.act(self.bn(x))


class UPx2(nn.Module):
    """Transposed conv ×2 → BatchNorm → PReLU."""

    def __init__(self, nIn, nOut):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(nIn, nOut, 2, stride=2, bias=False)
        self.bn     = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act    = nn.PReLU(nOut)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


# ── Encoder blocks ────────────────────────────────────────────────────────────

class InputProjectionA(nn.Module):
    """Pyramid-based average-pool downsampler."""

    def __init__(self, samplingTimes):
        super().__init__()
        self.pool = nn.ModuleList(
            [nn.AvgPool2d(3, stride=2, padding=1) for _ in range(samplingTimes)]
        )

    def forward(self, x):
        for p in self.pool:
            x = p(x)
        return x


class DownSamplerB(nn.Module):
    """Strided dilated parallel block (downsamples by 2)."""

    def __init__(self, nIn, nOut):
        super().__init__()
        n  = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1  = C(nIn, n, 3, 2)
        self.d1  = CDilated(n, n1, 3, 1, 1)
        self.d2  = CDilated(n, n,  3, 1, 2)
        self.d4  = CDilated(n, n,  3, 1, 4)
        self.d8  = CDilated(n, n,  3, 1, 8)
        self.d16 = CDilated(n, n,  3, 1, 16)
        self.bn  = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, x):
        o   = self.c1(x)
        d1  = self.d1(o)
        add1 = self.d2(o)
        add2 = add1 + self.d4(o)
        add3 = add2 + self.d8(o)
        add4 = add3 + self.d16(o)
        return self.act(self.bn(torch.cat([d1, add1, add2, add3, add4], 1)))


class DilatedParllelResidualBlockB(nn.Module):
    """ESP block: reduce → split → transform (dilated) → hierarchical merge."""

    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n  = max(int(nOut / 5), 1)
        n1 = max(nOut - 4 * n, 1)
        self.c1  = C(nIn, n, 1, 1)
        self.d1  = CDilated(n, n1, 3, 1, 1)
        self.d2  = CDilated(n, n,  3, 1, 2)
        self.d4  = CDilated(n, n,  3, 1, 4)
        self.d8  = CDilated(n, n,  3, 1, 8)
        self.d16 = CDilated(n, n,  3, 1, 16)
        self.bn  = BR(nOut)
        self.add = add

    def forward(self, x):
        o    = self.c1(x)
        d1   = self.d1(o)
        add1 = self.d2(o)
        add2 = add1 + self.d4(o)
        add3 = add2 + self.d8(o)
        add4 = add3 + self.d16(o)
        out  = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            out = x + out
        return self.bn(out)


# ── Encoder ───────────────────────────────────────────────────────────────────

class ESPNet_Encoder(nn.Module):
    """Shared encoder producing 32-channel feature maps at 1/8 input resolution."""

    def __init__(self, p=2, q=3):
        super().__init__()
        self.level1   = CBR(3, 16, 3, 2)
        self.sample1  = InputProjectionA(1)
        self.sample2  = InputProjectionA(2)

        self.b1       = CBR(16 + 3, 19, 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)

        self.level2   = nn.ModuleList(
            [DilatedParllelResidualBlockB(64, 64) for _ in range(p)]
        )
        self.b2       = CBR(128 + 3, 131, 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3   = nn.ModuleList(
            [DilatedParllelResidualBlockB(128, 128) for _ in range(q)]
        )

        self.b3       = CBR(256, 32, 3)
        self.sa       = PAM_Module(32)
        self.sc       = CAM_Module(32)
        self.conv_sa  = CBR(32, 32, 3)
        self.conv_sc  = CBR(32, 32, 3)
        self.classifier = CBR(32, 32, 1, 1)

    def forward(self, x):
        out0     = self.level1(x)
        inp1, inp2 = self.sample1(x), self.sample2(x)

        out0_cat = self.b1(torch.cat([out0, inp1], 1))
        out1_0   = self.level2_0(out0_cat)
        out1     = out1_0
        for layer in self.level2:
            out1 = layer(out1)

        out1_cat = self.b2(torch.cat([out1, out1_0, inp2], 1))
        out2_0   = self.level3_0(out1_cat)
        out2     = out2_0
        for layer in self.level3:
            out2 = layer(out2)

        cat_    = torch.cat([out2_0, out2], 1)
        out2_cat = self.b3(cat_)

        out_sa  = self.conv_sa(self.sa(out2_cat))
        out_sc  = self.conv_sc(self.sc(out2_cat))
        return self.classifier(out_sa + out_sc)


# ── Full model ────────────────────────────────────────────────────────────────

class TwinLiteNet(nn.Module):
    """
    Dual-head segmentation network.

    Returns:
        (da_logits, ll_logits) — each (B, 2, H, W):
            da_logits:  drivable area  [background, drivable]
            ll_logits:  lane lines     [background, lane]
    """

    def __init__(self, p=2, q=3):
        super().__init__()
        self.encoder      = ESPNet_Encoder(p, q)

        self.up_1_1       = UPx2(32, 16)
        self.up_2_1       = UPx2(16, 8)
        self.classifier_1 = UPx2(8, 2)

        self.up_1_2       = UPx2(32, 16)
        self.up_2_2       = UPx2(16, 8)
        self.classifier_2 = UPx2(8, 2)

    def forward(self, x):
        feat = self.encoder(x)

        x1 = self.classifier_1(self.up_2_1(self.up_1_1(feat)))
        x2 = self.classifier_2(self.up_2_2(self.up_1_2(feat)))

        return x1, x2
