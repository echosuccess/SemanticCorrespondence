"""Gradient-enabled wrapper for fine-tuning the last N transformer blocks.

Supports DINOv2, DINOv3, and SAM backbones.

DINOv2 / DINOv3  – transformer blocks are in ``model.blocks``, final norm
                   is ``model.norm``.  Forward uses ``forward_features()``.
SAM              – transformer blocks are in ``image_encoder.blocks``, final
                   processing is ``image_encoder.neck`` (a conv neck).
                   Forward calls the encoder directly.

Typical usage
-------------
    from backbones import build_backbone
    from backbones.finetune_wrapper import FinetunableBackbone

    frozen_backbone = build_backbone("dinov2_vitb14", repo_dir="external/dinov2",
                                     input_size=224, device="cuda")
    model = FinetunableBackbone(frozen_backbone, n_unfrozen_layers=2)
    model.set_train_eval_mode()

    # gradient-enabled forward
    feat = model(images)   # (B, C, h, w)
    loss = my_loss(feat, ...)
    loss.backward()
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class FinetunableBackbone(nn.Module):
    """Wraps a frozen DINOv2/DINOv3 backbone for supervised fine-tuning.

    After construction all parameters remain frozen.  Call
    ``unfreeze_last_n(n)`` (or pass ``n_unfrozen_layers`` to the constructor)
    to selectively unfreeze the last *n* transformer blocks plus the final
    layer-norm.

    The ``forward()`` method is identical to ``backbone.extract()`` but
    runs *without* ``torch.no_grad()``, so gradients flow through the
    unfrozen layers.

    Parameters
    ----------
    backbone :
        A ``DINOv2Backbone`` or ``DINOv3Backbone`` instance.
    n_unfrozen_layers : int
        How many of the *last* transformer blocks to unfreeze.
        0 means fully frozen (useful for sanity checks).
    """

    def __init__(self, backbone, n_unfrozen_layers: int = 2) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_unfrozen_layers = n_unfrozen_layers
        # Detect SAM vs DINOv2/DINOv3 once at construction time.
        # SAMBackbone stores image_encoder in self.model and has no
        # forward_features(); DINOv2/DINOv3 models do have it.
        self._is_sam = not hasattr(backbone.model, "forward_features")
        self.unfreeze_last_n(n_unfrozen_layers)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def unfreeze_last_n(self, n: int) -> None:
        """Freeze all params, then unfreeze the last *n* blocks + final layer."""
        self.n_unfrozen_layers = n
        model = self.backbone.model

        for p in model.parameters():
            p.requires_grad_(False)

        blocks = model.blocks
        n_total = len(blocks)
        start = max(0, n_total - n)
        for i in range(start, n_total):
            for p in blocks[i].parameters():
                p.requires_grad_(True)

        # DINOv2/DINOv3: final LayerNorm is model.norm
        # SAM: final processing is model.neck (conv neck)
        final_layer = getattr(model, "norm", None) or getattr(model, "neck", None)
        if final_layer is not None:
            for p in final_layer.parameters():
                p.requires_grad_(True)

    def set_train_eval_mode(self) -> None:
        """Put frozen layers in eval mode and unfrozen layers in train mode.

        Call this at the top of every training epoch so that dropout /
        layer-norm statistics behave correctly.
        """
        model = self.backbone.model
        model.eval()

        blocks = model.blocks
        n_total = len(blocks)
        start = max(0, n_total - self.n_unfrozen_layers)
        for i in range(start, n_total):
            blocks[i].train()

        final_layer = getattr(model, "norm", None) or getattr(model, "neck", None)
        if final_layer is not None:
            final_layer.train()

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.backbone.model.parameters() if p.requires_grad]

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def n_total_blocks(self) -> int:
        return len(self.backbone.model.blocks)

    # ------------------------------------------------------------------
    # Gradient-enabled forward pass
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: ``(B, 3, H, W)`` in ``[0, 1]``. Returns ``(B, C, h, w)``."""
        b = self.backbone
        x = F.interpolate(
            images, size=(b.input_size, b.input_size),
            mode="bilinear", align_corners=False,
        )
        x = x.to(b.device, non_blocking=True)

        if self._is_sam:
            # SAM uses its own pixel-mean/std (stored on the backbone)
            x = (x - b._mean) / b._std
            return b.model(x).contiguous()  # (B, 256, 64, 64) – already shaped
        else:
            # DINOv2 / DINOv3 – ImageNet normalisation
            x = TF.normalize(x, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
            out = b.model.forward_features(x)
            tokens = out["x_norm_patchtokens"]  # (B, h*w, C)
            B = tokens.shape[0]
            h = w = b.input_size // b.patch_stride
            return tokens.transpose(1, 2).reshape(B, b.feature_dim, h, w).contiguous()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, extra: dict | None = None) -> None:
        """Persist the full model state dict (not just the unfrozen layers).

        Loading is straightforward: restore into any backbone of the same
        architecture with ``load_checkpoint_into_backbone()``.
        """
        payload = {
            "model_state_dict": self.backbone.model.state_dict(),
            "backbone_name": self.backbone.name,
            "n_unfrozen_layers": self.n_unfrozen_layers,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[ckpt] Saved checkpoint → {path}")

    @staticmethod
    def load_checkpoint_into_backbone(backbone, path: str | Path) -> dict:
        """Load a saved checkpoint into an existing backbone.

        Returns the full checkpoint dict (contains training metadata).
        """
        ckpt = torch.load(path, map_location="cpu")
        backbone.model.load_state_dict(ckpt["model_state_dict"])
        backbone.model.to(backbone.device)
        backbone.model.eval()
        for p in backbone.model.parameters():
            p.requires_grad_(False)
        return ckpt
