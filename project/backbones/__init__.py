"""Factory for the frozen backbones used in the training-free baseline."""

from __future__ import annotations

from typing import Any, Dict

from .base import DenseFeatureBackbone
from .dinov2_backbone import DINOv2Backbone
from .dinov3_backbone import DINOv3Backbone
from .sam_backbone import SAMBackbone


_BACKBONE_REGISTRY: Dict[str, type] = {
    # DINOv2
    "dinov2_vits14": DINOv2Backbone,
    "dinov2_vitb14": DINOv2Backbone,
    "dinov2_vitl14": DINOv2Backbone,
    "dinov2_vitg14": DINOv2Backbone,
    # DINOv3
    "dinov3_vits16": DINOv3Backbone,
    "dinov3_vitb16": DINOv3Backbone,
    "dinov3_vitl16": DINOv3Backbone,
    # SAM
    "sam_vit_b": SAMBackbone,
    "sam_vit_l": SAMBackbone,
    "sam_vit_h": SAMBackbone,
}


def build_backbone(name: str, **kwargs: Any) -> DenseFeatureBackbone:
    """Instantiate a backbone by short name.

    Any extra kwargs (``repo_dir``, ``input_size``, ``checkpoint`` ...) are
    forwarded to the concrete constructor.
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {list(_BACKBONE_REGISTRY)}."
        )

    cls = _BACKBONE_REGISTRY[name]

    if cls is DINOv2Backbone:
        return DINOv2Backbone(variant=name, **kwargs)
    if cls is DINOv3Backbone:
        return DINOv3Backbone(variant=name, **kwargs)
    if cls is SAMBackbone:
        # strip the "sam_" prefix to get "vit_b" / "vit_l" / "vit_h"
        return SAMBackbone(variant=name.replace("sam_", ""), **kwargs)
    raise RuntimeError(f"Unhandled backbone class for '{name}'.")


__all__ = [
    "DenseFeatureBackbone",
    "DINOv2Backbone",
    "DINOv3Backbone",
    "SAMBackbone",
    "build_backbone",
]
