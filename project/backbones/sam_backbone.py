"""Segment Anything (SAM) image encoder as a dense feature extractor.

We only keep ``sam.image_encoder`` (a ViT-based encoder that maps a
1024x1024 image to a 64x64x256 feature grid) and discard the prompt and
mask decoders.

The official repo is https://github.com/facebookresearch/segment-anything
and checkpoints are listed in its README:
  * ``sam_vit_b_01ec64.pth`` (ViT-B)
  * ``sam_vit_l_0b3195.pth`` (ViT-L)
  * ``sam_vit_h_4b8939.pth`` (ViT-H)
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .base import DenseFeatureBackbone


_SAM_PIXEL_MEAN = [123.675, 116.28, 103.53]
_SAM_PIXEL_STD = [58.395, 57.12, 57.375]

_FEATURE_DIMS = {
    "vit_b": 256,
    "vit_l": 256,
    "vit_h": 256,
}


class SAMBackbone(DenseFeatureBackbone):
    """Frozen SAM image encoder.

    Parameters
    ----------
    variant : str
        ``"vit_b"``, ``"vit_l"`` or ``"vit_h"``.
    checkpoint : str
        Path to the ``.pth`` weights for the chosen variant.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    patch_stride = 16  # 1024 / 64
    input_size = 1024

    def __init__(
        self,
        variant: str = "vit_b",
        checkpoint: str = "external/segment-anything/sam_vit_b_01ec64.pth",
        device: str = "cuda",
    ) -> None:
        if variant not in _FEATURE_DIMS:
            raise ValueError(
                f"Unknown SAM variant '{variant}'. "
                f"Expected one of {list(_FEATURE_DIMS)}."
            )
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"SAM checkpoint not found at '{checkpoint}'. "
                f"Download from the official repo's README."
            )

        # Imported lazily so the project still loads on environments without
        # segment-anything installed (e.g. before the first `pip install`).
        from segment_anything import sam_model_registry

        self.name = f"sam_{variant}"
        self.feature_dim = _FEATURE_DIMS[variant]
        self.device = device

        sam = sam_model_registry[variant](checkpoint=checkpoint)
        self.model = sam.image_encoder
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Pre-scale the normalisation statistics to the [0, 1] input range so
        # we can feed images that are already in [0, 1] (consistent with the
        # other backbones).
        self._mean = (torch.tensor(_SAM_PIXEL_MEAN).view(1, 3, 1, 1) / 255.0).to(device)
        self._std = (torch.tensor(_SAM_PIXEL_STD).view(1, 3, 1, 1) / 255.0).to(device)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W) input, got {tuple(images.shape)}.")

        x = F.interpolate(
            images, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )
        x = x.to(self.device, non_blocking=True)
        x = (x - self._mean) / self._std

        feat = self.model(x)  # (B, 256, 64, 64)
        return feat.contiguous()
