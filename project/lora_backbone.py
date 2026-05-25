"""LoRA (Low-Rank Adaptation) for Vision Foundation Models — Step 4.

Injects trainable low-rank matrices A and B alongside every target
linear layer (e.g. attention qkv and proj).  All original backbone
weights remain strictly frozen.  Only A and B are trained.

    W' = W + (A @ B) * (alpha / r)

where  W  is the frozen pretrained weight,  r  is the rank, and
alpha / r  is the scaling factor.

This is implemented manually (no peft dependency) so the backbone's
own forward methods (get_intermediate_layers, etc.) are never touched.

References
----------
Hu et al., LoRA: Low-Rank Adaptation of Large Language Models,
ICLR 2022.  https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Replaces a frozen nn.Linear with a LoRA-augmented version.

    The original weight is kept and frozen.  Two small matrices A (in×r)
    and B (r×out) are added; only A and B are trained.
    """

    def __init__(
        self,
        original: nn.Linear,
        r: int,
        lora_alpha: float,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original = original
        self.r = r
        self.scaling = lora_alpha / r

        in_f  = original.in_features
        out_f = original.out_features
        device = original.weight.device   # match original layer's device

        self.lora_A = nn.Parameter(torch.empty(in_f, r, device=device))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f, device=device))
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Kaiming uniform for A (standard LoRA initialisation)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze original weights
        for p in original.parameters():
            p.requires_grad_(False)

    # Expose the same attributes as nn.Linear so backbone code stays compatible
    @property
    def in_features(self) -> int:
        return self.original.in_features

    @property
    def out_features(self) -> int:
        return self.original.out_features

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base + lora

    def merge(self) -> nn.Linear:
        """Return a plain nn.Linear with LoRA weights merged in."""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        with torch.no_grad():
            merged.weight.copy_(
                self.original.weight
                + (self.lora_A @ self.lora_B).t() * self.scaling
            )
            if self.original.bias is not None:
                merged.bias.copy_(self.original.bias)
        return merged


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def _get_submodule(model: nn.Module, path: str):
    """Return the submodule and its parent given a dotted path."""
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    child = getattr(parent, parts[-1])
    return parent, parts[-1], child


def inject_lora(
    model: nn.Module,
    r: int,
    lora_alpha: float,
    target_suffixes: List[str],
    lora_dropout: float = 0.0,
) -> int:
    """Replace matching nn.Linear modules with LoRALinear in-place.

    Args:
        model          : The backbone's inner model (e.g. backbone.model).
        r              : LoRA rank.
        lora_alpha     : LoRA scaling (alpha/r applied to the LoRA term).
        target_suffixes: Module names whose *suffix* matches these strings
                         will be replaced (e.g. ["qkv", "proj"]).
        lora_dropout   : Dropout on the LoRA path.

    Returns:
        Number of layers replaced.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name == s or name.endswith(f".{s}") for s in target_suffixes):
                replacements.append(name)

    for name in replacements:
        parent, attr, linear = _get_submodule(model, name)
        setattr(parent, attr, LoRALinear(linear, r, lora_alpha, lora_dropout))

    return len(replacements)


def count_lora_params(model: nn.Module) -> int:
    """Count trainable LoRA parameters."""
    return sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, LoRALinear)
        for p in [m.lora_A, m.lora_B]
    )


# ---------------------------------------------------------------------------
# LoRA backbone wrapper
# ---------------------------------------------------------------------------

class LoRABackbone(nn.Module):
    """Wraps a DenseFeatureBackbone, injecting LoRA into attention layers.

    All original backbone weights are frozen.  Only LoRA matrices A and B
    are trained.  The forward interface matches FinetunableBackbone so
    the same training loop (train_step2_finetune.py) can be reused.
    """

    # Target modules for each backbone family
    _DINO_TARGETS = ["qkv", "proj"]   # DINOv2 / DINOv3 ViT attention
    _SAM_TARGETS  = ["qkv", "proj"]   # SAM image-encoder attention

    def __init__(
        self,
        backbone,
        r: int = 8,
        lora_alpha: float = 32.0,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.r = r
        self.lora_alpha = lora_alpha
        self._is_sam = backbone.__class__.__name__ == "SAMBackbone"

        if target_modules is None:
            target_modules = self._SAM_TARGETS if self._is_sam else self._DINO_TARGETS

        self.target_modules = target_modules

        # Freeze entire backbone first
        for p in backbone.model.parameters():
            p.requires_grad_(False)

        # Inject LoRA
        n = inject_lora(backbone.model, r, lora_alpha, target_modules, lora_dropout)
        n_params = count_lora_params(backbone.model)
        print(
            f"LoRA: injected {n} layers  "
            f"(r={r}, alpha={lora_alpha})  "
            f"trainable params: {n_params:,}"
        )

    # ------------------------------------------------------------------
    # Train / eval management
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Return only the LoRA A/B parameters (the ones that need gradients)."""
        return [p for p in self.backbone.model.parameters() if p.requires_grad]

    def set_train_eval_mode(self) -> None:
        """Set LoRA layers to train, everything else to eval."""
        self.backbone.model.eval()
        for m in self.backbone.model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad_(True)
                m.lora_B.requires_grad_(True)
                m.train()

    # ------------------------------------------------------------------
    # Forward (mirrors FinetunableBackbone)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0,1].  Returns (B, C, h, w) feature map."""
        device = next(self.backbone.model.parameters()).device
        x = x.to(device)

        if self._is_sam:
            # SAM handles its own normalisation internally
            return self.backbone.model(x).contiguous()
        else:
            # DINOv2 / DINOv3: apply ImageNet normalisation first
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x = (x - mean) / std

            # get_intermediate_layers returns a list of (B, N_tokens, C)
            feats = self.backbone.model.get_intermediate_layers(x, n=1)[0]
            B, N, C = feats.shape
            h = w = int(N ** 0.5)
            return feats.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, extra: Optional[dict] = None) -> None:
        """Save only the LoRA parameters + metadata."""
        lora_state = {
            name: param
            for name, param in self.backbone.model.named_parameters()
            if param.requires_grad
        }
        payload = {
            "type":           "lora",
            "backbone_class": self.backbone.__class__.__name__,
            "r":              self.r,
            "lora_alpha":     self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_state":     lora_state,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[LoRA ckpt] saved → {path}  "
              f"({len(lora_state)} tensors, "
              f"{sum(v.numel() for v in lora_state.values()):,} params)")

    @classmethod
    def load_checkpoint_into_backbone(cls, backbone, path) -> dict:
        """Apply saved LoRA weights to *backbone* and return the metadata dict.

        The backbone's own model is modified in-place: LoRALinear layers
        are injected and their weights loaded.  After this call the backbone
        behaves exactly as it did at checkpoint time.
        """
        ckpt = torch.load(path, map_location="cpu")
        assert ckpt.get("type") == "lora", \
            f"Expected a LoRA checkpoint but got type={ckpt.get('type')!r}"

        r              = ckpt["r"]
        lora_alpha     = ckpt["lora_alpha"]
        target_modules = ckpt["target_modules"]

        # Freeze and inject LoRA structure
        for p in backbone.model.parameters():
            p.requires_grad_(False)
        inject_lora(backbone.model, r, lora_alpha, target_modules)

        # Load LoRA weights
        state = ckpt["lora_state"]
        model_state = backbone.model.state_dict()
        model_state.update(state)
        backbone.model.load_state_dict(model_state, strict=False)

        n_params = count_lora_params(backbone.model)
        print(
            f"[LoRA ckpt] loaded from {path}\n"
            f"  r={r}  alpha={lora_alpha}  "
            f"target={target_modules}  params={n_params:,}"
        )
        return ckpt
