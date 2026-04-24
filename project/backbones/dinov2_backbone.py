"""DINOv2 dense feature extractor.

Loads the official ``facebookresearch/dinov2`` code via ``torch.hub`` from a
local clone (no HuggingFace, per the project instructions) and returns the
patch tokens reshaped into a ``(B, C, h, w)`` map.

Variants supported out of the box:
  * ``dinov2_vits14`` -> 384-d features
  * ``dinov2_vitb14`` -> 768-d features
  * ``dinov2_vitl14`` -> 1024-d features
  * ``dinov2_vitg14`` -> 1536-d features
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

_FEATURE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2Backbone(DenseFeatureBackbone):
    """Frozen DINOv2 ViT used as a dense feature extractor.

    Parameters
    ----------
    variant : str
        One of the keys in ``_FEATURE_DIMS``.
    repo_dir : str
        Path to a local clone of ``facebookresearch/dinov2``.
    input_size : int
        Side length the image is resized to. Must be a multiple of 14.
        518 (=37*14) is a common choice and gives a 37x37 feature grid.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    patch_stride = 14

    def __init__(
        self,
        variant: str = "dinov2_vitb14",
        repo_dir: str = "external/dinov2",
        input_size: int = 518,
        device: str = "cuda",
        weights_path: Optional[str] = None,
    ) -> None:
        if variant not in _FEATURE_DIMS:
            raise ValueError(
                f"Unknown DINOv2 variant '{variant}'. "
                f"Expected one of {list(_FEATURE_DIMS)}."
            )
        if input_size % self.patch_stride != 0:
            raise ValueError(
                f"input_size={input_size} must be a multiple of {self.patch_stride}."
            )
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                f"DINOv2 repo not found at '{repo_dir}'. "
                f"Clone it with: git clone https://github.com/facebookresearch/dinov2"
            )

        self.name = variant
        self.feature_dim = _FEATURE_DIMS[variant]
        self.input_size = input_size
        self.device = device

        # Load from local clone of the official repo (not HuggingFace).
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
        """``images``: ``(B, 3, H, W)`` in ``[0, 1]``. Returns ``(B, C, h, w)``."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W) input, got {tuple(images.shape)}.")

        x = F.interpolate(
            images, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )
        x = TF.normalize(x, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        x = x.to(self.device, non_blocking=True)

        # forward_features returns a dict; x_norm_patchtokens is (B, N, C)
        out = self.model.forward_features(x)
        tokens = out["x_norm_patchtokens"]

        B = tokens.shape[0]
        h = w = self.input_size // self.patch_stride
        feat = tokens.transpose(1, 2).reshape(B, self.feature_dim, h, w).contiguous()
        return feat
