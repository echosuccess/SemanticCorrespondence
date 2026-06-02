"""Visualise similarity maps for PPT: shows Argmax vs Window Soft-Argmax.

For a chosen source keypoint, computes cosine similarity against the target
feature map and produces a figure with:
  - Source image with the selected keypoint marked
  - Target image with the full similarity heatmap overlaid
  - Argmax prediction (white cross) and ground-truth (green circle) marked

Example (run in Colab):
    %cd /content/drive/MyDrive/Semantic-Correspondence
    !python project/visualize_simmap.py \
        --backbone dinov2_vitb14 \
        --dino-repo external/dinov2 \
        --data-root /content/drive/MyDrive/Semantic-Correspondence \
        --sd4match-dir SD4Match \
        --split test \
        --pair-id 5 \
        --kp-idx 0 \
        --output-dir project/results/simmap_vis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _add_path(d: str):
    d = os.path.abspath(d)
    if d not in sys.path:
        sys.path.insert(0, d)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--data-root", default="../")
    p.add_argument("--split", default="test")
    p.add_argument("--img-size", type=int, default=518)
    p.add_argument("--pair-id", type=int, default=0,
                   help="Dataset index of the pair to visualise.")
    p.add_argument("--kp-idx", type=int, default=0,
                   help="Which source keypoint to query (0-indexed).")
    p.add_argument("--window-size", type=int, default=7,
                   help="WSA window size.")
    p.add_argument("--wsa-temp", type=float, default=0.1,
                   help="WSA softmax temperature.")
    p.add_argument("--dino-repo", default="external/dinov2")
    p.add_argument("--dinov3-weights",
                   default="external/dinov3_weights/dinov3_vitb16.pth")
    p.add_argument("--sam-checkpoint",
                   default="external/segment-anything/sam_vit_b_01ec64.pth")
    p.add_argument("--sd4match-dir", default="../SD4Match")
    p.add_argument("--output-dir", default="results/simmap_vis")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_backbone(args):
    _add_path(str(Path(args.data_root) / "project"))
    from backbones import build_backbone as _build
    kwargs = {"device": args.device}
    if args.backbone.startswith("dinov2_"):
        kwargs["repo_dir"] = args.dino_repo
        kwargs["input_size"] = args.img_size
    elif args.backbone.startswith("dinov3_"):
        kwargs["repo_dir"] = args.dino_repo
        kwargs["input_size"] = args.img_size
        kwargs["weights_path"] = args.dinov3_weights
    elif args.backbone.startswith("sam_"):
        kwargs["checkpoint"] = args.sam_checkpoint
    return _build(args.backbone, **kwargs)


def build_dataset(args):
    _add_path(args.sd4match_dir)
    _add_path(str(Path(args.data_root) / "project"))
    from config.base import get_default_defaults
    from spair_dataset import SafeSPairDataset as SPairDataset
    cfg = get_default_defaults()
    cfg.DATASET.NAME = "spair"
    cfg.DATASET.ROOT = os.path.abspath(args.data_root)
    cfg.DATASET.IMG_SIZE = args.img_size
    cfg.DATASET.MEAN = [0.0, 0.0, 0.0]
    cfg.DATASET.STD = [1.0, 1.0, 1.0]
    return SPairDataset(cfg, split=args.split, category="all")


def to_np(t):
    return t.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)


def window_softargmax(sim_hw, cx, cy, w, temp):
    """Return sub-pixel (x, y) in feature-map coords using WSA."""
    H, W = sim_hw.shape
    half = w // 2
    x0 = max(0, cx - half); x1 = min(W, cx + half + 1)
    y0 = max(0, cy - half); y1 = min(H, cy + half + 1)
    patch = sim_hw[y0:y1, x0:x1]
    weights = torch.softmax((patch / temp).flatten(), dim=0).reshape(patch.shape)
    ys = torch.arange(y0, y1, dtype=torch.float32)
    xs = torch.arange(x0, x1, dtype=torch.float32)
    pred_y = (weights.sum(dim=1) * ys).sum().item()
    pred_x = (weights.sum(dim=0) * xs).sum().item()
    return pred_x, pred_y


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset  = build_dataset(args)
    backbone = build_backbone(args)

    sample = dataset[args.pair_id]
    n_pts  = int(sample["n_pts"].item())
    kp_idx = min(args.kp_idx, n_pts - 1)

    src_img = sample["src_img"]   # (3, H, W)
    trg_img = sample["trg_img"]
    src_kp  = sample["src_kps"][kp_idx]   # (2,) x, y in image px
    trg_kp  = sample["trg_kps"][kp_idx]
    category = sample["category"]

    H, W = src_img.shape[1:]

    # ---- extract features ------------------------------------------------
    with torch.no_grad():
        F_src = backbone(src_img.unsqueeze(0).to(args.device)).float()  # (1,C,h,w)
        F_trg = backbone(trg_img.unsqueeze(0).to(args.device)).float()
    F_src = F_src.squeeze(0)   # (C, h, w)
    F_trg = F_trg.squeeze(0)
    C, fh, fw = F_trg.shape

    # L2-normalise
    F_src = F.normalize(F_src, dim=0)
    F_trg = F.normalize(F_trg, dim=0)

    # ---- sample source descriptor ----------------------------------------
    # Bilinear sample at keypoint location rescaled to feature grid
    src_fx = src_kp[0].item() / W * fw
    src_fy = src_kp[1].item() / H * fh
    grid = torch.tensor([[[[src_fx / (fw - 1) * 2 - 1,
                            src_fy / (fh - 1) * 2 - 1]]]],
                        dtype=torch.float32, device=args.device)
    desc = F.grid_sample(F_src.unsqueeze(0), grid,
                         mode="bilinear", align_corners=True)
    desc = desc.squeeze().reshape(C)   # (C,)

    # ---- compute similarity map ------------------------------------------
    sim = (F_trg * desc.view(C, 1, 1)).sum(dim=0)   # (fh, fw)
    sim_np = sim.cpu().numpy()

    # ---- argmax prediction -----------------------------------------------
    flat_idx = sim.cpu().flatten().argmax().item()
    ax_fy, ax_fx = divmod(flat_idx, fw)
    # convert back to image px
    ax_px = (ax_fx + 0.5) / fw * W
    ax_py = (ax_fy + 0.5) / fh * H

    # ---- WSA prediction --------------------------------------------------
    wsa_fx, wsa_fy = window_softargmax(
        sim.cpu(), ax_fx, ax_fy, args.window_size, args.wsa_temp)
    wsa_px = (wsa_fx + 0.5) / fw * W
    wsa_py = (wsa_fy + 0.5) / fh * H

    # ---- plot ------------------------------------------------------------
    src_np = to_np(src_img)
    trg_np = to_np(trg_img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Similarity Map — {category}  (pair {args.pair_id}, kp {kp_idx})",
                 fontsize=13, fontweight="bold")

    # Panel 1: source with keypoint
    axes[0].imshow(src_np)
    axes[0].scatter(src_kp[0], src_kp[1], c="red", s=120,
                    edgecolors="white", linewidths=1.5, zorder=5)
    axes[0].set_title("Source keypoint (red)", fontsize=11)
    axes[0].axis("off")

    # Panel 2: similarity heatmap on target
    axes[1].imshow(trg_np)
    sim_up = F.interpolate(
        sim.unsqueeze(0).unsqueeze(0), size=(H, W),
        mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    axes[1].imshow(sim_up, cmap="hot", alpha=0.55)
    # GT
    axes[1].scatter(trg_kp[0], trg_kp[1], c="lime", s=180, marker="o",
                    edgecolors="black", linewidths=1.5, zorder=6, label="Ground truth")
    # Argmax
    axes[1].scatter(ax_px, ax_py, c="white", s=120, marker="x",
                    linewidths=2.5, zorder=7, label=f"Argmax ({ax_px:.1f},{ax_py:.1f})")
    # WSA
    axes[1].scatter(wsa_px, wsa_py, c="cyan", s=120, marker="P",
                    edgecolors="black", linewidths=1.0, zorder=7,
                    label=f"WSA ({wsa_px:.1f},{wsa_py:.1f})")
    axes[1].set_title("Similarity heatmap + predictions", fontsize=11)
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.85)
    axes[1].axis("off")

    # Panel 3: zoomed patch grid around peak (shows quantization)
    zoom = 3
    z0x = max(0, int(ax_px) - zoom * 14)
    z1x = min(W, int(ax_px) + zoom * 14)
    z0y = max(0, int(ax_py) - zoom * 14)
    z1y = min(H, int(ax_py) + zoom * 14)
    axes[2].imshow(trg_np[z0y:z1y, z0x:z1x])
    patch_size = W // fw
    for gx in range(0, z1x - z0x, patch_size):
        axes[2].axvline(gx, color="white", linewidth=0.5, alpha=0.6)
    for gy in range(0, z1y - z0y, patch_size):
        axes[2].axhline(gy, color="white", linewidth=0.5, alpha=0.6)
    axes[2].scatter(ax_px - z0x, ax_py - z0y, c="white", s=150, marker="x",
                    linewidths=2.5, zorder=7, label="Argmax (patch center)")
    axes[2].scatter(wsa_px - z0x, wsa_py - z0y, c="cyan", s=150, marker="P",
                    edgecolors="black", linewidths=1.0, zorder=7, label="WSA (sub-pixel)")
    axes[2].scatter(trg_kp[0] - z0x, trg_kp[1] - z0y, c="lime", s=180,
                    edgecolors="black", linewidths=1.5, zorder=6, label="Ground truth")
    axes[2].set_title("Zoomed: patch grid + quantization", fontsize=11)
    axes[2].legend(loc="lower right", fontsize=8, framealpha=0.85)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"simmap_{category}_pair{args.pair_id}_kp{kp_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    print(f"  Argmax:  ({ax_px:.1f}, {ax_py:.1f})")
    print(f"  WSA:     ({wsa_px:.1f}, {wsa_py:.1f})")
    print(f"  GT:      ({trg_kp[0]:.1f}, {trg_kp[1]:.1f})")
    err_ax  = ((ax_px - trg_kp[0])**2 + (ax_py - trg_kp[1])**2)**0.5
    err_wsa = ((wsa_px - trg_kp[0])**2 + (wsa_py - trg_kp[1])**2)**0.5
    print(f"  Error Argmax: {err_ax:.1f}px   WSA: {err_wsa:.1f}px")


if __name__ == "__main__":
    main()
