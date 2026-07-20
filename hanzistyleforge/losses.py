from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .features import split_prediction


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.reshape(-1).to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _soft_erode(image: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-image, kernel_size=3, stride=1, padding=1)


def _soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def _soft_open(image: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(image))


def _soft_skeleton(image: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    opened = _soft_open(image)
    skeleton = F.relu(image - opened)
    current = image
    for _ in range(max(1, int(iterations))):
        current = _soft_erode(current)
        opened = _soft_open(current)
        delta = F.relu(current - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def _dice_per(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


class FontLossFinal(nn.Module):
    """Multi-task glyph loss with explicit SDF, skeleton and edge supervision."""

    DEFAULT_WEIGHTS = {
        "bce": 0.28,
        "dice": 0.24,
        "edge": 0.13,
        "multiscale": 0.08,
        "projection": 0.075,
        "cldice": 0.085,
        "proxy_skeleton": 0.003,
        "proxy_layout": 0.001,
        "sdf": 0.16,
        "skeleton_head": 0.10,
        "edge_head": 0.085,
        "topology_points": 0.003,
        "boundary_distance": 0.04,
        "style_signature": 0.12,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        super().__init__()
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({str(k): float(v) for k, v in weights.items()})
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edge(self, image: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(image, self.sobel_x, padding=1)
        gy = F.conv2d(image, self.sobel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    @staticmethod
    def _fallback_aux(target: torch.Tensor) -> torch.Tensor:
        # Used only when an auxiliary target cache is missing or damaged.
        small = target
        skeleton = _soft_skeleton(small, iterations=7)
        edge = F.max_pool2d(target, 3, 1, 1) - _soft_erode(target)
        # A differentiable pseudo-SDF is sufficient as a fallback; normal Final
        # datasets store an exact OpenCV distance transform in target_aux_path.
        inside = torch.zeros_like(target)
        current = target
        for index in range(8):
            inside = inside + (current > 0.25).to(target.dtype)
            current = _soft_erode(current)
        outside = torch.zeros_like(target)
        current = 1.0 - target
        for index in range(8):
            outside = outside + (current > 0.25).to(target.dtype)
            current = _soft_erode(current)
        sdf = (0.5 + (inside - outside) / 16.0).clamp(0.0, 1.0)
        return torch.cat([target, sdf, skeleton, edge.clamp(0.0, 1.0)], dim=1)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        content_proxy: torch.Tensor | None = None,
        target_aux: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ink_logits, sdf_logits, skeleton_logits, edge_logits = split_prediction(prediction)
        probability = torch.sigmoid(ink_logits)
        if target_aux is None or target_aux.shape[1] < 4:
            target_aux = self._fallback_aux(target)
        target_aux = target_aux[:, :4]
        target_sdf = target_aux[:, 1:2]
        target_skeleton = target_aux[:, 2:3]
        target_edge = target_aux[:, 3:4]

        foreground = target.mean().detach()
        pos_weight = ((1.0 - foreground) / (foreground + 1e-4)).clamp(1.0, 5.0)
        bce_map = F.binary_cross_entropy_with_logits(ink_logits, target, pos_weight=pos_weight, reduction="none")
        bce_per = bce_map.mean(dim=(1, 2, 3))
        dice_per = _dice_per(probability, target)
        edge_per = (self._edge(probability) - self._edge(target)).abs().mean(dim=(1, 2, 3))

        multi_per = torch.zeros_like(dice_per)
        for scale in (2, 4, 8):
            p = F.avg_pool2d(probability, scale)
            t = F.avg_pool2d(target, scale)
            multi_per = multi_per + (p - t).abs().mean(dim=(1, 2, 3))
        multi_per = multi_per / 3.0

        row_pred = probability.mean(dim=3)
        row_true = target.mean(dim=3)
        col_pred = probability.mean(dim=2)
        col_true = target.mean(dim=2)
        projection_per = (
            (row_pred - row_true).abs().mean(dim=(1, 2))
            + (col_pred - col_true).abs().mean(dim=(1, 2))
        ) / 2.0

        # Font-style signature: global ink mass, visual centre, spread and edge
        # density are supervised only from target.ttf.  This explicitly makes
        # the target font—not the content proxy—the source of stroke weight,
        # spacing and terminal/roundness statistics.
        yy = torch.linspace(-1.0, 1.0, probability.shape[-2], device=probability.device, dtype=probability.dtype)
        xx = torch.linspace(-1.0, 1.0, probability.shape[-1], device=probability.device, dtype=probability.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        grid_x = grid_x.view(1, 1, *grid_x.shape)
        grid_y = grid_y.view(1, 1, *grid_y.shape)
        pred_mass = probability.sum(dim=(1, 2, 3)).clamp_min(1e-5)
        true_mass = target.sum(dim=(1, 2, 3)).clamp_min(1e-5)
        pred_cx = (probability * grid_x).sum(dim=(1, 2, 3)) / pred_mass
        pred_cy = (probability * grid_y).sum(dim=(1, 2, 3)) / pred_mass
        true_cx = (target * grid_x).sum(dim=(1, 2, 3)) / true_mass
        true_cy = (target * grid_y).sum(dim=(1, 2, 3)) / true_mass
        pred_vx = (probability * (grid_x - pred_cx[:, None, None, None]).square()).sum(dim=(1, 2, 3)) / pred_mass
        pred_vy = (probability * (grid_y - pred_cy[:, None, None, None]).square()).sum(dim=(1, 2, 3)) / pred_mass
        true_vx = (target * (grid_x - true_cx[:, None, None, None]).square()).sum(dim=(1, 2, 3)) / true_mass
        true_vy = (target * (grid_y - true_cy[:, None, None, None]).square()).sum(dim=(1, 2, 3)) / true_mass
        pred_edge_density = self._edge(probability).mean(dim=(1, 2, 3))
        true_edge_density = self._edge(target).mean(dim=(1, 2, 3))
        style_signature_per = (
            2.0 * (probability.mean(dim=(1, 2, 3)) - target.mean(dim=(1, 2, 3))).abs()
            + (pred_cx - true_cx).abs()
            + (pred_cy - true_cy).abs()
            + 0.75 * (pred_vx - true_vx).abs()
            + 0.75 * (pred_vy - true_vy).abs()
            + 0.75 * (pred_edge_density - true_edge_density).abs()
        ) / 6.25

        h, w = probability.shape[-2:]
        if max(h, w) > 128:
            p_small = F.interpolate(probability, size=(128, 128), mode="bilinear", align_corners=False)
            t_small = F.interpolate(target, size=(128, 128), mode="bilinear", align_corners=False)
        else:
            p_small, t_small = probability, target
        skel_pred = _soft_skeleton(p_small, iterations=7)
        skel_true = _soft_skeleton(t_small, iterations=7)
        tprec = (skel_pred * t_small).sum(dim=(1, 2, 3)) / skel_pred.sum(dim=(1, 2, 3)).clamp_min(1e-5)
        tsens = (skel_true * p_small).sum(dim=(1, 2, 3)) / skel_true.sum(dim=(1, 2, 3)).clamp_min(1e-5)
        cldice_per = 1.0 - (2.0 * tprec * tsens + 1e-5) / (tprec + tsens + 1e-5)

        proxy_skeleton_per = torch.zeros_like(dice_per)
        proxy_layout_per = torch.zeros_like(dice_per)
        point_per = torch.zeros_like(dice_per)
        if content_proxy is not None and content_proxy.shape[1] >= 4:
            proxy = content_proxy
            if max(h, w) > 128:
                proxy_small = F.interpolate(proxy, size=(128, 128), mode="bilinear", align_corners=False)
            else:
                proxy_small = proxy
            proxy_skeleton = _soft_dilate(proxy_small[:, 1:2].clamp(0.0, 1.0))
            content_precision = (skel_pred * proxy_skeleton).sum(dim=(1, 2, 3)) / skel_pred.sum(
                dim=(1, 2, 3)
            ).clamp_min(1e-5)
            proxy_core_skeleton = _soft_skeleton(proxy_small[:, 0:1], iterations=5)
            content_recall = (proxy_core_skeleton * p_small).sum(dim=(1, 2, 3)) / proxy_core_skeleton.sum(
                dim=(1, 2, 3)
            ).clamp_min(1e-5)
            proxy_skeleton_per = 1.0 - (
                2.0 * content_precision * content_recall + 1e-5
            ) / (content_precision + content_recall + 1e-5)
            pred_layout = F.avg_pool2d(p_small, kernel_size=8, stride=8)
            content_layout = F.avg_pool2d(proxy_small[:, 3:4], kernel_size=8, stride=8)
            proxy_layout_per = (pred_layout - content_layout).abs().mean(dim=(1, 2, 3))

            if proxy_small.shape[1] >= 9:
                endpoint_map = proxy_small[:, 7:8]
                junction_map = proxy_small[:, 8:9]
                coverage = F.max_pool2d(skel_pred, 7, 1, 3)
                endpoint_loss = ((1.0 - coverage) * endpoint_map).sum(dim=(1, 2, 3)) / endpoint_map.sum(
                    dim=(1, 2, 3)
                ).clamp_min(1.0)
                junction_loss = ((1.0 - coverage) * junction_map).sum(dim=(1, 2, 3)) / junction_map.sum(
                    dim=(1, 2, 3)
                ).clamp_min(1.0)
                point_per = (endpoint_loss + junction_loss) / 2.0

        if sdf_logits is None:
            sdf_per = torch.zeros_like(dice_per)
        else:
            sdf_probability = torch.sigmoid(sdf_logits)
            # Weight the zero-level region more strongly because it controls
            # the contour extracted during vectorization.
            contour_weight = 1.0 + 2.5 * torch.exp(-((target_sdf - 0.5) / 0.12).square())
            sdf_per = (F.smooth_l1_loss(sdf_probability, target_sdf, reduction="none") * contour_weight).mean(
                dim=(1, 2, 3)
            )

        if skeleton_logits is None:
            skeleton_head_per = torch.zeros_like(dice_per)
        else:
            skeleton_probability = torch.sigmoid(skeleton_logits)
            skeleton_bce = F.binary_cross_entropy_with_logits(
                skeleton_logits, target_skeleton, reduction="none"
            ).mean(dim=(1, 2, 3))
            skeleton_head_per = 0.55 * skeleton_bce + 0.45 * _dice_per(skeleton_probability, target_skeleton)

        if edge_logits is None:
            edge_head_per = torch.zeros_like(dice_per)
            boundary_distance_per = torch.zeros_like(dice_per)
        else:
            edge_probability = torch.sigmoid(edge_logits)
            edge_head_per = (
                0.55 * F.binary_cross_entropy_with_logits(edge_logits, target_edge, reduction="none").mean(dim=(1, 2, 3))
                + 0.45 * (edge_probability - target_edge).abs().mean(dim=(1, 2, 3))
            )
            target_boundary_distance = (target_sdf - 0.5).abs() * 2.0
            boundary_distance_per = (edge_probability * target_boundary_distance).mean(dim=(1, 2, 3))

        pieces_per = {
            "bce": bce_per,
            "dice_loss": dice_per,
            "edge_loss": edge_per,
            "multiscale_loss": multi_per,
            "projection_loss": projection_per,
            "cldice_loss": cldice_per,
            "proxy_skeleton_loss": proxy_skeleton_per,
            "proxy_layout_loss": proxy_layout_per,
            "sdf_loss": sdf_per,
            "skeleton_head_loss": skeleton_head_per,
            "edge_head_loss": edge_head_per,
            "topology_point_loss": point_per,
            "boundary_distance_loss": boundary_distance_per,
            "style_signature_loss": style_signature_per,
        }
        reduced = {key: _weighted_mean(value, sample_weight) for key, value in pieces_per.items()}
        total = torch.zeros((), device=probability.device, dtype=probability.dtype)
        mapping = {
            "bce": "bce",
            "dice": "dice_loss",
            "edge": "edge_loss",
            "multiscale": "multiscale_loss",
            "projection": "projection_loss",
            "cldice": "cldice_loss",
            "proxy_skeleton": "proxy_skeleton_loss",
            "proxy_layout": "proxy_layout_loss",
            "sdf": "sdf_loss",
            "skeleton_head": "skeleton_head_loss",
            "edge_head": "edge_head_loss",
            "topology_points": "topology_point_loss",
            "boundary_distance": "boundary_distance_loss",
            "style_signature": "style_signature_loss",
        }
        for weight_key, piece_key in mapping.items():
            total = total + float(self.weights.get(weight_key, 0.0)) * reduced[piece_key]
        return total, {key: value.detach() for key, value in reduced.items()}


def batch_binary_dice(probability: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if probability.shape[1] > 1:
        probability = probability[:, :1]
    pred = (probability >= threshold).float()
    truth = (target >= 0.5).float()
    intersection = (pred * truth).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
    return (2.0 * intersection + 1.0) / (denominator + 1.0)

