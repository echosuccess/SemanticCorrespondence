"""Qualitative visualisation of training-free semantic correspondence.

For each selected pair:
  * left  : source image with annotated keypoints (colour per keypoint),
  * right : target image with the ground-truth keypoint (hollow circle) and
            the model's prediction (filled dot, same colour).

Saves one PNG per pair under ``--output-dir`` and, optionally, a grid PNG.

Example
-------
    python project/visualize.py \
        --backbone dinov2_vitb14 \
        --dino-repo external/dinov2 \
        --data-root ../ \
        --split test \
        --num-pairs 8 \
        --output-dir results/step1/vis_dinov2_vitb14
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _add_sd4match_to_syspath(sd4match_dir: str) -> None:
    sd4match_dir = os.path.abspath(sd4match_dir)
    if sd4match_dir not in sys.path:
        sys.path.insert(0, sd4match_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, required=True)

    p.add_argument("--data-root", type=str, default="../")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--category", type=str, default="all")
    p.add_argument("--img-size", type=int, default=512)

    p.add_argument("--dino-repo", type=str, default="external/dinov2")
    p.add_argument("--sam-checkpoint", type=str,
                   default="external/segment-anything/sam_vit_b_01ec64.pth")
    p.add_argument("--backbone-input-size", type=int, default=None)

    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--pair-ids", type=int, nargs="+", default=None,
                   help="If given, visualise exactly these dataset indices.")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--sd4match-dir", type=str, default="../SD4Match")
    p.add_argument("--output-dir", type=str, default="results/step1/vis")
    return p.parse_args()


def build_backbone(args: argparse.Namespace):
    from backbones import build_backbone as _build

    kwargs = {"device": args.device}
    if args.backbone.startswith("dinov2_"):
        kwargs["repo_dir"] = args.dino_repo
        kwargs["input_size"] = args.backbone_input_size or 518
    elif args.backbone.startswith("dinov3_"):
        kwargs["repo_dir"] = args.dino_repo
        kwargs["input_size"] = args.backbone_input_size or 512
    elif args.backbone.startswith("sam_"):
        kwargs["checkpoint"] = args.sam_checkpoint
    else:
        raise ValueError(f"Unrecognised backbone '{args.backbone}'.")
    return _build(args.backbone, **kwargs)


def build_dataset(args: argparse.Namespace):
    from config.base import get_default_defaults
    from spair_dataset import SafeSPairDataset as SPairDataset

    cfg = get_default_defaults()
    cfg.DATASET.NAME = "spair"
    cfg.DATASET.ROOT = os.path.abspath(args.data_root)
    cfg.DATASET.IMG_SIZE = args.img_size
    cfg.DATASET.MEAN = [0.0, 0.0, 0.0]
    cfg.DATASET.STD = [1.0, 1.0, 1.0]
    return SPairDataset(cfg, split=args.split, category=args.category)


def _to_display_image(img_tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [0, 1] -> (H, W, 3) uint8 for matplotlib."""
    arr = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def predict_matches(
    backbone,
    src_img: torch.Tensor,
    trg_img: torch.Tensor,
    src_kps: torch.Tensor,  # (N, 2) in (x, y), image pixels
) -> torch.Tensor:
    """Argmax matching for a single pair; returns target coords in image px."""
    # Local import so SD4Match's sys.path has already been set.
    from utils.matching import nn_get_matches
    from utils.geometry import scaling_coordinates

    # Features at their native feature-map resolution.
    F_src = backbone(src_img.unsqueeze(0)).float()   # (1, C, h1, w1)
    F_trg = backbone(trg_img.unsqueeze(0)).float()   # (1, C, h2, w2)

    H1, W1 = src_img.shape[1:]
    H2, W2 = trg_img.shape[1:]
    h1, w1 = F_src.shape[2:]
    h2, w2 = F_trg.shape[2:]

    q = scaling_coordinates(src_kps.unsqueeze(0), (H1, W1), (h1, w1))  # (1, N, 2)
    pred_feat = nn_get_matches(F_src, F_trg, q, l2_norm=True)          # (1, N, 2)
    pred_img = scaling_coordinates(pred_feat, (h2, w2), (H2, W2))
    return pred_img.squeeze(0).cpu()                                   # (N, 2)


def plot_pair(sample, preds: torch.Tensor, save_path: Path) -> None:
    n_pts = int(sample["n_pts"].item())
    src_kps = sample["src_kps"][:n_pts]         # (N, 2) xy
    trg_kps = sample["trg_kps"][:n_pts]         # ground truth
    preds = preds[:n_pts]

    src_np = _to_display_image(sample["src_img"])
    trg_np = _to_display_image(sample["trg_img"])

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_pts)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(src_np)
    axes[0].set_title(f"source ({sample['category']})")
    axes[0].axis("off")
    for i in range(n_pts):
        axes[0].scatter(src_kps[i, 0], src_kps[i, 1], c=[colors[i]],
                        s=60, edgecolors="black", linewidths=0.7)

    axes[1].imshow(trg_np)
    axes[1].set_title("target: GT (hollow) vs prediction (filled)")
    axes[1].axis("off")
    for i in range(n_pts):
        axes[1].scatter(trg_kps[i, 0], trg_kps[i, 1],
                        facecolors="none", edgecolors=colors[i],
                        s=120, linewidths=2.0)
        axes[1].scatter(preds[i, 0], preds[i, 1],
                        c=[colors[i]], s=45, edgecolors="black",
                        linewidths=0.6)

    legend_elems = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="none", markeredgecolor="black",
               markersize=10, label="ground truth"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="grey", markeredgecolor="black",
               markersize=8, label="prediction"),
    ]
    axes[1].legend(handles=legend_elems, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _add_sd4match_to_syspath(args.sd4match_dir)

    dataset = build_dataset(args)
    backbone = build_backbone(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pair_ids is not None:
        indices = list(args.pair_ids)
    else:
        step = max(1, len(dataset) // args.num_pairs)
        indices = list(range(0, len(dataset), step))[: args.num_pairs]

    print(f"Visualising {len(indices)} pairs from '{args.split}' split "
          f"with backbone='{args.backbone}'.")

    for idx in indices:
        sample = dataset[idx]
        n_pts = int(sample["n_pts"].item())
        preds = predict_matches(
            backbone,
            sample["src_img"],
            sample["trg_img"],
            sample["src_kps"][:n_pts],
        )
        save_path = out_dir / f"pair_{idx:06d}_{sample['category']}.png"
        plot_pair(sample, preds, save_path)
        print(f"  saved {save_path}")

    print(f"\nAll visualisations written under {out_dir}")


if __name__ == "__main__":
    main()
