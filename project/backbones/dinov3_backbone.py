"""DINOv3 dense feature extractor.

DINOv3 exposes the same ``forward_features`` interface as DINOv2 via
``torch.hub``, so this wrapper is almost identical to the v2 one. The
official repo lives at https://github.com/facebookresearch/dinov3 and the
pretrained checkpoints need to be downloaded manually (they are gated).

Pass the path to a ``.pth`` checkpoint via ``weights_path`` if you cannot
use the hub auto-download.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .base import DenseFeatureBackbone


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# DINOv3 uses patch size 16 for the ViT variants we target here.
# Feature dims follow the standard ViT-S/B/L layout.
_FEATURE_DIMS = {
    "dinov3_vits16": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
}


class DINOv3Backbone(DenseFeatureBackbone):
    """Frozen DINOv3 ViT used as a dense feature extractor."""

    def __init__(
        self,
        variant: str = "dinov3_vitb16",
        repo_dir: str = "external/dinov3",
        input_size: int = 512,
        device: str = "cuda",
        weights_path: Optional[str] = None,
    ) -> None:
        if variant not in _FEATURE_DIMS:
            raise ValueError(
                f"Unknown DINOv3 variant '{variant}'. "
                f"Expected one of {list(_FEATURE_DIMS)}."
            )
        self.patch_stride = 16
        if input_size % self.patch_stride != 0:
            raise ValueError(
                f"input_size={input_size} must be a multiple of {self.patch_stride}."
            )
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                f"DINOv3 repo not found at '{repo_dir}'. "
                f"Clone it with: git clone https://github.com/facebookresearch/dinov3"
            )

        self.name = variant
        self.feature_dim = _FEATURE_DIMS[variant]
        self.input_size = input_size
        self.device = device

        pretrained = weights_path is None
        model = torch.hub.load(repo_dir, variant, source="local", pretrained=pretrained)
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W) input, got {tuple(images.shape)}.")

        x = F.interpolate(
            images, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )
        x = TF.normalize(x, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        x = x.to(self.device, non_blocking=True)

        out = self.model.forward_features(x)
        # DINOv3 keeps the same key name as DINOv2 for normalised patch tokens.
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            tokens = out["x_norm_patchtokens"]
        else:
            raise RuntimeError(
                "Unexpected output from DINOv3 forward_features; "
                "check the repo version and adapt the key name."
            )

        B = tokens.shape[0]
        h = w = self.input_size // self.patch_stride
        feat = tokens.transpose(1, 2).reshape(B, self.feature_dim, h, w).contiguous()
        return feat
