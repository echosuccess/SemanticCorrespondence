"""Abstract backbone for dense feature extraction.

Each concrete backbone (DINOv2 / DINOv3 / SAM) must implement ``extract``
which, given an image tensor in ``[0, 1]`` range and arbitrary spatial size,
returns a dense feature map of shape ``(B, C, h, w)``.

The wrapper is responsible for:
  * resizing the input to the backbone's preferred input size,
  * applying the correct normalization statistics,
  * reshaping / selecting the appropriate internal tokens.

This lets the downstream pipeline treat every backbone as a black box that
maps ``(B, 3, H, W) -> (B, C, h, w)``.
"""

from abc import ABC, abstractmethod

import torch


class DenseFeatureBackbone(ABC):
    """Minimal interface shared by all frozen feature extractors."""

    name: str = "backbone"
    feature_dim: int = 0
    patch_stride: int = 0

    @abstractmethod
    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``(B, C, h, w)`` dense features for ``images`` in ``[0, 1]``."""
        raise NotImplementedError

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract(images)
