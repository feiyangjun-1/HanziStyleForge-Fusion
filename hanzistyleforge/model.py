from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (16, 12, 8, 6, 4, 3, 2):
        if channels % value == 0:
            return value
    return 1


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class GlobalStyleFiLM(nn.Module):
    """A small learned per-font style token injected as affine modulation.

    The project trains one model per target font, so this token is deliberately
    global rather than character-specific.  It reduces the pressure on the
    content pathway to memorize roundness, terminal and stroke-weight biases.
    """

    def __init__(self, channels: int, style_dim: int = 96) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, style_dim))
        self.to_scale_shift = nn.Sequential(
            nn.Linear(style_dim, max(32, style_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(max(32, style_dim), channels * 2),
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(self.token).chunk(2, dim=-1)
        scale = scale.view(1, -1, 1, 1)
        shift = shift.view(1, -1, 1, 1)
        return x * (1.0 + 0.15 * torch.tanh(scale)) + 0.15 * shift


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, film: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.se = SqueezeExcite(out_channels)
        self.film = GlobalStyleFiLM(out_channels) if film else nn.Identity()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.se(self.norm2(self.conv2(x)))
        x = self.film(x)
        return self.act(x + residual)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels, dropout),
            ResidualBlock(out_channels, out_channels, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            ResidualBlock(out_channels + skip_channels, out_channels, dropout),
            ResidualBlock(out_channels, out_channels, dropout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.blocks(torch.cat([x, skip], dim=1))


class MultiScaleContentBlock(nn.Module):
    """Aggregate local stroke cues and global glyph layout at the bottleneck."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = max(16, channels // 4)
        self.branches = nn.ModuleList()
        for dilation in (1, 2, 4, 8):
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels),
                    nn.GroupNorm(_groups(channels), channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels, branch_channels, 1),
                )
            )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, 1),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 5, channels, 1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            ResidualBlock(channels, channels, dropout=0.06),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        global_feature = F.interpolate(self.global_branch(x), size=x.shape[-2:], mode="nearest")
        return x + self.fuse(torch.cat([*features, global_feature], dim=1))


class BottleneckAttention(nn.Module):
    """Memory-bounded global reasoning for complex CJK layouts."""

    def __init__(self, channels: int, heads: int = 8, max_tokens: int = 24 * 24) -> None:
        super().__init__()
        heads = max(1, min(int(heads), channels // 16))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=0.03, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(0.03),
            nn.Linear(channels * 2, channels),
        )
        self.max_tokens = int(max_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        source = x
        if h * w > self.max_tokens:
            side = max(8, int(math.sqrt(self.max_tokens)))
            source = F.adaptive_avg_pool2d(x, (side, side))
        sh, sw = source.shape[-2:]
        tokens = source.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ff(tokens)
        result = tokens.transpose(1, 2).reshape(n, c, sh, sw)
        if (sh, sw) != (h, w):
            result = F.interpolate(result, size=(h, w), mode="bilinear", align_corners=False)
        return x + 0.35 * result


class TopologyStem(nn.Module):
    """Emphasize endpoint/junction/edge channels before downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.topology = nn.Sequential(
            nn.Conv2d(5, max(8, out_channels // 2), 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, out_channels // 2), out_channels, 1),
        )
        self.blocks = nn.Sequential(ResidualBlock(out_channels, out_channels), ResidualBlock(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # edge, endpoint, junction, centreline and coarse layout
        topology = torch.cat([x[:, 4:5], x[:, 7:10], x[:, 3:4]], dim=1)
        return self.blocks(self.main(x) + self.topology(topology))


class FontStyleNetFinal(nn.Module):
    """Topology-aware direct renderer without a hard proxy-output shortcut.

    Earlier builds added proxy logits directly to the output.  That made the
    canonical content proxy an attractor and could gradually erase the target
    font style.  This version predicts all four heads directly from learned
    decoder features.  Skip connections still preserve character structure,
    while the target glyph exclusively determines stroke weight and terminals.
    """

    def __init__(self, base: int = 32, in_channels: int = 10, out_channels: int = 4) -> None:
        super().__init__()
        b = int(base)
        self.base = b
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stem = TopologyStem(in_channels, b)
        self.enc2 = DownBlock(b, b * 2)
        self.enc3 = DownBlock(b * 2, b * 4)
        self.enc4 = DownBlock(b * 4, b * 8, dropout=0.02)
        self.enc5 = DownBlock(b * 8, b * 12, dropout=0.04)
        self.content = MultiScaleContentBlock(b * 12)
        self.attention = BottleneckAttention(b * 12, heads=8)
        self.dec4 = UpBlock(b * 12, b * 8, b * 8, dropout=0.03)
        self.dec3 = UpBlock(b * 8, b * 4, b * 4, dropout=0.02)
        self.dec2 = UpBlock(b * 4, b * 2, b * 2)
        self.dec1 = UpBlock(b * 2, b, b)
        self.head = nn.Sequential(
            ResidualBlock(b, b),
            nn.Conv2d(b, b, 3, padding=1),
            nn.GroupNorm(_groups(b), b),
            nn.SiLU(inplace=True),
            nn.Conv2d(b, out_channels, 1),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Start from a sparse neutral renderer rather than copying the proxy.
        final = self.head[-1]
        if final.bias is not None and final.bias.numel() >= 4:
            with torch.no_grad():
                final.bias.copy_(torch.tensor([-1.65, 0.0, -2.4, -2.0], dtype=final.bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.stem(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        bottleneck = self.attention(self.content(e5))
        d4 = self.dec4(bottleneck, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)


class GlyphRefinerFinal(nn.Module):
    """High-resolution artifact refiner with the same four auxiliary heads."""

    def __init__(self, base: int = 24, in_channels: int = 11, out_channels: int = 4) -> None:
        super().__init__()
        b = int(base)
        self.base = b
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stem = nn.Sequential(ResidualBlock(in_channels, b), ResidualBlock(b, b))
        self.enc2 = DownBlock(b, b * 2)
        self.enc3 = DownBlock(b * 2, b * 4)
        self.enc4 = DownBlock(b * 4, b * 8, dropout=0.03)
        self.content = MultiScaleContentBlock(b * 8)
        self.attention = BottleneckAttention(b * 8, heads=4, max_tokens=20 * 20)
        self.dec3 = UpBlock(b * 8, b * 4, b * 4)
        self.dec2 = UpBlock(b * 4, b * 2, b * 2)
        self.dec1 = UpBlock(b * 2, b, b)
        self.head = nn.Sequential(ResidualBlock(b, b), nn.Conv2d(b, out_channels, 1))
        self.residual_gain = nn.Parameter(torch.ones(out_channels) * 0.75)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.head[-1]
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

    @staticmethod
    def _logit(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(0.025, 0.975)
        return torch.log(value) - torch.log1p(-value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.stem(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottleneck = self.attention(self.content(e4))
        d3 = self.dec3(bottleneck, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        residual = self.head(d1) * self.residual_gain.clamp(0.15, 3.0).view(1, -1, 1, 1)
        # x = [candidate ink, expanded proxy...]
        baselines = torch.cat(
            [
                self._logit(x[:, 0:1]),
                self._logit(x[:, 3:4]),
                self._logit(x[:, 2:3]),
                self._logit(x[:, 5:6]),
            ],
            dim=1,
        )
        return baselines + residual


class PatchDiscriminatorFinal(nn.Module):
    """Optional low-weight multi-scale PatchGAN used only after warm-up."""

    def __init__(self, base: int = 32, in_channels: int = 1) -> None:
        super().__init__()
        channels = [base, base * 2, base * 4, base * 6]
        blocks: list[nn.Module] = []
        current = in_channels
        for index, out in enumerate(channels):
            blocks.append(nn.Conv2d(current, out, 4, stride=2, padding=1))
            if index:
                blocks.append(nn.GroupNorm(_groups(out), out))
            blocks.append(nn.LeakyReLU(0.2, inplace=True))
            current = out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Conv2d(current, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features: list[torch.Tensor] = []
        current = x
        for module in self.features:
            current = module(current)
            if isinstance(module, nn.LeakyReLU):
                features.append(current)
        score = self.head(current)
        return (score, features) if return_features else score


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

