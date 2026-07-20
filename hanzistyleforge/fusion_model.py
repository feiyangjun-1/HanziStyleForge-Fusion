from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    # Keep at least two channels in each group. This remains valid for batch=1
    # and a 1x1 feature map, which occurs in small self-tests and can also
    # occur in style encoders for unusually small preview resolutions.
    for value in (32, 24, 16, 12, 8, 6, 4, 3, 2):
        if channels % value == 0 and channels // value >= 2:
            return value
    return 1


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2 if stride == 1 else max(0, (kernel_size - stride) // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = ConvNormAct(out_channels, out_channels, activation=False)
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.act(x + residual)


class GlyphStyleCNN(nn.Module):
    """Encode one rendered target glyph into a style descriptor.

    It deliberately receives the rendered target glyph rather than the content
    proxy.  This prevents the content pathway from becoming the sole carrier of
    stroke weight, terminal shape, roundness and visual-size information.
    """

    def __init__(self, base: int = 32, style_dim: int = 256) -> None:
        super().__init__()
        b = int(base)
        self.features = nn.Sequential(
            ConvNormAct(1, b, 5, stride=2),
            ResBlock(b, b),
            ConvNormAct(b, b * 2, 3, stride=2),
            ResBlock(b * 2, b * 2),
            ConvNormAct(b * 2, b * 4, 3, stride=2),
            ResBlock(b * 4, b * 4),
            ConvNormAct(b * 4, b * 6, 3, stride=2),
            ResBlock(b * 6, b * 6),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(b * 6 * 4, style_dim),
            nn.LayerNorm(style_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, glyphs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(self.features(glyphs)))


class StyleReferenceEncoder(nn.Module):
    """Aggregate multiple target glyphs into global and localized style tokens.

    Learned expert queries attend to different target reference glyphs.  The
    result is analogous to a collection of localized experts: one token can
    specialize in terminals, another in dense intersections, another in round
    counters, without requiring semantic component labels during inference.
    """

    def __init__(
        self,
        base: int = 32,
        style_dim: int = 256,
        expert_count: int = 8,
        heads: int = 8,
        synthetic_parameter_count: int = 7,
    ) -> None:
        super().__init__()
        self.style_dim = int(style_dim)
        self.expert_count = int(expert_count)
        self.glyph_encoder = GlyphStyleCNN(base=base, style_dim=style_dim)
        self.expert_queries = nn.Parameter(torch.randn(expert_count, style_dim) * 0.02)
        self.attention = nn.MultiheadAttention(style_dim, heads, dropout=0.05, batch_first=True)
        self.post = nn.Sequential(
            nn.LayerNorm(style_dim),
            nn.Linear(style_dim, style_dim * 2),
            nn.GELU(),
            nn.Linear(style_dim * 2, style_dim),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(style_dim),
            nn.Linear(style_dim, style_dim),
            nn.SiLU(inplace=True),
        )
        self.synthetic_head = nn.Sequential(
            nn.LayerNorm(style_dim),
            nn.Linear(style_dim, style_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(style_dim // 2, synthetic_parameter_count),
        )

    def forward(self, references: torch.Tensor) -> dict[str, torch.Tensor]:
        if references.ndim != 5:
            raise ValueError(f"style references must be [B,K,1,H,W], got {tuple(references.shape)}")
        batch, count, channels, height, width = references.shape
        if channels != 1:
            raise ValueError("style reference glyphs must be single-channel ink images")
        encoded = self.glyph_encoder(references.reshape(batch * count, 1, height, width))
        glyph_tokens = encoded.reshape(batch, count, self.style_dim)
        queries = self.expert_queries.unsqueeze(0).expand(batch, -1, -1)
        experts, _ = self.attention(queries, glyph_tokens, glyph_tokens, need_weights=False)
        experts = experts + self.post(experts)
        global_style = self.global_projection(experts.mean(dim=1))
        synthetic = self.synthetic_head(global_style)
        return {
            "global": global_style,
            "experts": experts,
            "glyph_tokens": glyph_tokens,
            "synthetic": synthetic,
        }


class MultiScaleContentEncoder(nn.Module):
    """FontDiffuser-inspired multi-scale content aggregation for proxy inputs."""

    def __init__(self, in_channels: int = 10, base: int = 48) -> None:
        super().__init__()
        b = int(base)
        self.stem = nn.Sequential(ConvNormAct(in_channels, b, 5), ResBlock(b, b))
        self.down1 = nn.Sequential(ConvNormAct(b, b * 2, 4, stride=2), ResBlock(b * 2, b * 2))
        self.down2 = nn.Sequential(ConvNormAct(b * 2, b * 4, 4, stride=2), ResBlock(b * 4, b * 4))
        self.down3 = nn.Sequential(ConvNormAct(b * 4, b * 6, 4, stride=2), ResBlock(b * 6, b * 6))
        self.down4 = nn.Sequential(ConvNormAct(b * 6, b * 8, 4, stride=2), ResBlock(b * 8, b * 8))
        self.context_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(b * 8, b * 8, 3, padding=dilation, dilation=dilation, groups=b * 8),
                nn.GroupNorm(_groups(b * 8), b * 8),
                nn.SiLU(inplace=True),
                nn.Conv2d(b * 8, b * 2, 1),
            )
            for dilation in (1, 2, 4, 8)
        ])
        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(b * 8, b * 2, 1), nn.SiLU())
        self.context_fuse = nn.Sequential(
            nn.Conv2d(b * 10, b * 8, 1),
            nn.GroupNorm(_groups(b * 8), b * 8),
            nn.SiLU(inplace=True),
            ResBlock(b * 8, b * 8),
        )
        self.channels = (b, b * 2, b * 4, b * 6, b * 8)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c0 = self.stem(x)
        c1 = self.down1(c0)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        c4 = self.down4(c3)
        branches = [branch(c4) for branch in self.context_branches]
        global_feature = F.interpolate(self.global_branch(c4), size=c4.shape[-2:], mode="nearest")
        c4 = c4 + self.context_fuse(torch.cat([*branches, global_feature], dim=1))
        return [c0, c1, c2, c3, c4]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.projection = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        exponent = exponent / max(1, half - 1)
        angles = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[1]))
        return self.projection(embedding)


class ExpertStyleModulation(nn.Module):
    """Mixture-of-experts FiLM conditioned on local content and diffusion time."""

    def __init__(self, channels: int, style_dim: int, time_dim: int, expert_count: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.expert_count = int(expert_count)
        self.expert_affine = nn.Linear(style_dim, channels * 2)
        self.gate = nn.Sequential(
            nn.Linear(channels + time_dim, max(64, channels // 2)),
            nn.SiLU(inplace=True),
            nn.Linear(max(64, channels // 2), expert_count),
        )
        self.time_affine = nn.Linear(time_dim, channels * 2)
        nn.init.zeros_(self.expert_affine.weight)
        nn.init.zeros_(self.expert_affine.bias)
        nn.init.zeros_(self.time_affine.weight)
        nn.init.zeros_(self.time_affine.bias)

    def forward(self, x: torch.Tensor, time: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gates = torch.softmax(self.gate(torch.cat([pooled, time], dim=1)), dim=1)
        affine = self.expert_affine(experts)
        affine = torch.sum(affine * gates.unsqueeze(-1), dim=1)
        affine = affine + self.time_affine(time)
        scale, shift = affine.chunk(2, dim=1)
        scale = scale.view(x.shape[0], -1, 1, 1)
        shift = shift.view(x.shape[0], -1, 1, 1)
        return x * (1.0 + 0.18 * torch.tanh(scale)) + 0.18 * shift


class DiffusionResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        style_dim: int,
        time_dim: int,
        expert_count: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.modulation = ExpertStyleModulation(out_channels, style_dim, time_dim, expert_count)
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.modulation(x, time, experts)
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + residual


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 8, maximum_tokens: int = 1024) -> None:
        super().__init__()
        heads = max(1, min(int(heads), channels // 16))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0.03, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.maximum_tokens = int(maximum_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        source = x
        if height * width > self.maximum_tokens:
            side = max(8, int(math.sqrt(self.maximum_tokens)))
            source = F.adaptive_avg_pool2d(source, (side, side))
        sh, sw = source.shape[-2:]
        tokens = source.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.feed_forward(tokens)
        result = tokens.transpose(1, 2).reshape(batch, channels, sh, sw)
        if (sh, sw) != (height, width):
            result = F.interpolate(result, size=(height, width), mode="bilinear", align_corners=False)
        return x + 0.35 * result


class LatentDiffusionUNet(nn.Module):
    """Latent denoiser with multi-scale ref content and localized target style."""

    def __init__(
        self,
        latent_channels: int = 32,
        content_channels: int = 10,
        base: int = 96,
        content_base: int = 40,
        style_dim: int = 256,
        expert_count: int = 8,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        b = int(base)
        self.latent_channels = int(latent_channels)
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.content = MultiScaleContentEncoder(content_channels, content_base)
        cc = self.content.channels
        self.content_projection = nn.ModuleList([
            nn.Conv2d(cc[2], b, 1),
            nn.Conv2d(cc[3], b * 2, 1),
            nn.Conv2d(cc[4], b * 4, 1),
        ])
        self.input = nn.Conv2d(latent_channels, b, 3, padding=1)
        self.block0 = DiffusionResBlock(b * 2, b, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count)
        self.down1 = nn.Conv2d(b, b * 2, 4, stride=2, padding=1)
        self.block1 = DiffusionResBlock(b * 4, b * 2, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count)
        self.down2 = nn.Conv2d(b * 2, b * 4, 4, stride=2, padding=1)
        self.block2 = DiffusionResBlock(b * 8, b * 4, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count, dropout=0.04)
        self.mid1 = DiffusionResBlock(b * 4, b * 4, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count, dropout=0.05)
        self.attention = SpatialAttention(b * 4, heads=8, maximum_tokens=1024)
        self.mid2 = DiffusionResBlock(b * 4, b * 4, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count)
        self.up1 = nn.Conv2d(b * 4, b * 2, 3, padding=1)
        self.up_block1 = DiffusionResBlock(b * 4, b * 2, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count)
        self.up0 = nn.Conv2d(b * 2, b, 3, padding=1)
        self.up_block0 = DiffusionResBlock(b * 2, b, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count)
        self.output = nn.Sequential(nn.GroupNorm(_groups(b), b), nn.SiLU(inplace=True), nn.Conv2d(b, latent_channels, 3, padding=1))

    @staticmethod
    def _resize(feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(feature, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        content_proxy: torch.Tensor,
        style_experts: torch.Tensor,
    ) -> torch.Tensor:
        time = self.time(timesteps)
        pyramid = self.content(content_proxy)
        x0 = self.input(noisy_latent)
        c0 = self._resize(self.content_projection[0](pyramid[2]), x0)
        x0 = self.block0(torch.cat([x0, c0], dim=1), time, style_experts)
        x1 = self.down1(x0)
        c1 = self._resize(self.content_projection[1](pyramid[3]), x1)
        x1 = self.block1(torch.cat([x1, c1], dim=1), time, style_experts)
        x2 = self.down2(x1)
        c2 = self._resize(self.content_projection[2](pyramid[4]), x2)
        x2 = self.block2(torch.cat([x2, c2], dim=1), time, style_experts)
        x2 = self.mid2(self.attention(self.mid1(x2, time, style_experts)), time, style_experts)
        up1 = F.interpolate(x2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(up1)
        up1 = self.up_block1(torch.cat([up1, x1], dim=1), time, style_experts)
        up0 = F.interpolate(up1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        up0 = self.up0(up0)
        up0 = self.up_block0(torch.cat([up0, x0], dim=1), time, style_experts)
        return self.output(up0)


class VectorQuantizerEMA(nn.Module):
    """EMA VQ codebook used as a target-stroke prior."""

    def __init__(self, embeddings: int = 1024, dimension: int = 32, decay: float = 0.995, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.embeddings = int(embeddings)
        self.dimension = int(dimension)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        codebook = torch.randn(embeddings, dimension)
        codebook = F.normalize(codebook, dim=1)
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(embeddings))
        self.register_buffer("embedding_average", codebook.clone())

    def nearest(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.dimension)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.codebook.t()
            + self.codebook.square().sum(dim=1).unsqueeze(0)
        )
        indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(indices, self.codebook).view(z.shape[0], z.shape[2], z.shape[3], self.dimension)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, indices.view(z.shape[0], z.shape[2], z.shape[3])

    def forward(self, z: torch.Tensor, update: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized, indices = self.nearest(z)
        if self.training and update:
            flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.dimension)
            flat_indices = indices.reshape(-1)
            one_hot = F.one_hot(flat_indices, self.embeddings).type(flat.dtype)
            counts = one_hot.sum(dim=0)
            sums = one_hot.t() @ flat
            self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
            self.embedding_average.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
            total = self.cluster_size.sum()
            normalized_count = (self.cluster_size + self.epsilon) / (
                total + self.embeddings * self.epsilon
            ) * total
            self.codebook.copy_(self.embedding_average / normalized_count.unsqueeze(1).clamp_min(self.epsilon))
        commitment = F.mse_loss(z, quantized.detach())
        straight_through = z + (quantized - z).detach()
        histogram = torch.bincount(indices.reshape(-1), minlength=self.embeddings).float()
        probability = histogram / histogram.sum().clamp_min(1.0)
        perplexity = torch.exp(-(probability * (probability + 1e-10).log()).sum())
        return straight_through, commitment, perplexity, indices


class GlyphVQVAE(nn.Module):
    """Target-font local-stroke VQ autoencoder.

    The codebook is learned exclusively from target.ttf glyphs.  Diffusion
    outputs are snapped to this codebook before decoding, reducing synthetic
    terminals and local stroke fragments that never occur in the target font.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base: int = 48,
        latent_channels: int = 32,
        embeddings: int = 1024,
        decay: float = 0.995,
    ) -> None:
        super().__init__()
        b = int(base)
        self.latent_channels = int(latent_channels)
        self.encoder = nn.Sequential(
            ConvNormAct(in_channels, b, 5),
            ResBlock(b, b),
            ConvNormAct(b, b * 2, 4, stride=2),
            ResBlock(b * 2, b * 2),
            ConvNormAct(b * 2, b * 4, 4, stride=2),
            ResBlock(b * 4, b * 4),
            ConvNormAct(b * 4, b * 6, 4, stride=2),
            ResBlock(b * 6, b * 6),
            nn.Conv2d(b * 6, latent_channels, 1),
        )
        self.pre_quant = nn.Conv2d(latent_channels, latent_channels, 1)
        self.quantizer = VectorQuantizerEMA(embeddings=embeddings, dimension=latent_channels, decay=decay)
        self.post_quant = nn.Conv2d(latent_channels, b * 6, 1)
        self.decoder = nn.Sequential(
            ResBlock(b * 6, b * 6),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(b * 6, b * 4),
            ResBlock(b * 4, b * 4),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(b * 4, b * 2),
            ResBlock(b * 2, b * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(b * 2, b),
            ResBlock(b, b),
            nn.Conv2d(b, out_channels, 3, padding=1),
        )

    def encode(self, target_aux: torch.Tensor, *, quantize: bool = True, update_codebook: bool = False) -> dict[str, torch.Tensor]:
        latent = self.pre_quant(self.encoder(target_aux))
        if not quantize:
            return {"latent": latent, "quantized": latent}
        quantized, commitment, perplexity, indices = self.quantizer(latent, update=update_codebook)
        return {
            "latent": latent,
            "quantized": quantized,
            "commitment": commitment,
            "perplexity": perplexity,
            "indices": indices,
        }

    def decode(self, latent: torch.Tensor, *, snap_to_codebook: bool = False) -> torch.Tensor:
        if snap_to_codebook:
            latent, _ = self.quantizer.nearest(latent)
        return self.decoder(self.post_quant(latent))

    def forward(self, target_aux: torch.Tensor, *, update_codebook: bool = True) -> dict[str, torch.Tensor]:
        encoded = self.encode(target_aux, quantize=True, update_codebook=update_codebook)
        reconstruction = self.decode(encoded["quantized"])
        return {**encoded, "reconstruction": reconstruction}


class StyleAwareGlyphRefiner(nn.Module):
    """High-resolution target-style residual refiner."""

    def __init__(self, in_channels: int = 11, base: int = 32, style_dim: int = 256, expert_count: int = 8) -> None:
        super().__init__()
        b = int(base)
        time_dim = style_dim
        self.pseudo_time = nn.Parameter(torch.zeros(1, time_dim))
        self.stem = ConvNormAct(in_channels, b, 5)
        self.block0 = DiffusionResBlock(b, b, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count)
        self.down1 = nn.Conv2d(b, b * 2, 4, stride=2, padding=1)
        self.block1 = DiffusionResBlock(b * 2, b * 2, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count)
        self.down2 = nn.Conv2d(b * 2, b * 4, 4, stride=2, padding=1)
        self.block2 = DiffusionResBlock(b * 4, b * 4, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count)
        self.attention = SpatialAttention(b * 4, heads=4, maximum_tokens=1024)
        self.up1 = nn.Conv2d(b * 4, b * 2, 3, padding=1)
        self.up_block1 = DiffusionResBlock(b * 4, b * 2, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count)
        self.up0 = nn.Conv2d(b * 2, b, 3, padding=1)
        self.up_block0 = DiffusionResBlock(b * 2, b, style_dim=style_dim, time_dim=time_dim, expert_count=expert_count)
        self.output = nn.Conv2d(b, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _logit(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(0.02, 0.98)
        return torch.log(value) - torch.log1p(-value)

    def forward(self, candidate_and_proxy: torch.Tensor, style_experts: torch.Tensor) -> torch.Tensor:
        batch = candidate_and_proxy.shape[0]
        time = self.pseudo_time.expand(batch, -1)
        e0 = self.block0(self.stem(candidate_and_proxy), time, style_experts)
        e1 = self.block1(self.down1(e0), time, style_experts)
        e2 = self.block2(self.down2(e1), time, style_experts)
        middle = self.attention(e2)
        u1 = F.interpolate(middle, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up_block1(torch.cat([self.up1(u1), e1], dim=1), time, style_experts)
        u0 = F.interpolate(u1, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.up_block0(torch.cat([self.up0(u0), e0], dim=1), time, style_experts)
        return self._logit(candidate_and_proxy[:, :1]) + self.output(u0)


class ContourSequenceTransformer(nn.Module):
    """DeepVecFont-inspired fixed-length closed-contour denoiser.

    It does not generate topology.  Topology is still supplied by the selected
    raster/SDF candidate; this model only moves resampled contour points toward
    the curvature and terminal statistics learned from target.ttf outlines.
    """

    def __init__(self, points: int = 128, hidden: int = 192, layers: int = 6, heads: int = 8) -> None:
        super().__init__()
        self.points = int(points)
        self.input = nn.Linear(8, hidden)
        self.position = nn.Parameter(torch.randn(1, points, hidden) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.offset = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.corner = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.shape[1] != self.points:
            raise ValueError(f"contour point count mismatch: expected {self.points}, got {features.shape[1]}")
        encoded = self.encoder(self.input(features) + self.position)
        return {"offset": self.offset(encoded), "corner_logits": self.corner(encoded)}


@dataclass(frozen=True)
class FusionModelSpec:
    style_dim: int
    expert_count: int
    latent_channels: int
    vq_embeddings: int
    diffusion_steps: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "FusionModelSpec":
        fusion = cfg.get("fusion", {})
        return cls(
            style_dim=int(fusion.get("style_dim", 256)),
            expert_count=int(fusion.get("expert_count", 8)),
            latent_channels=int(fusion.get("latent_channels", 32)),
            vq_embeddings=int(fusion.get("vq_embeddings", 1024)),
            diffusion_steps=int(fusion.get("diffusion_steps", 1000)),
        )
