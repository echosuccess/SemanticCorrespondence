"""Find the best pair for similarity map visualization.

Scans N pairs and finds ones where:
- Argmax is reasonably close to GT (< 60px error) -- so the prediction is meaningful
- WSA improves over Argmax (wsa_err < argmax_err)
- The improvement delta is large

Run in Colab:
    !python project/find_best_simmap.py \
        --backbone dinov2_vitb14 \
        --dino-repo external/dinov2 \
        --sd4match-dir SD4Match \
        --data-root /content/drive/MyDrive/Semantic-Correspondence \
        --split test --num-scan 200 \
        --output-dir project/results/simmap_vis
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def _add_path(d):
    d = os.path.abspath(d)
    if d not in sys.path:
        sys.path.insert(0, d)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--data-root", default="../")
    p.add_argument("--split", default="test")
    p.add_argument("--img-size", type=int, default=518)
    p.add_argument("--num-scan", type=int, default=200)
    p.add_argument("--window-size", type=int, default=7)
    p.add_argument("--wsa-temp", type=float, default=0.1)
    p.add_argument("--dino-repo", default="external/dinov2")
    p.add_argument("--dinov3-weights", default="external/dinov3_weights/dinov3_vitb16.pth")
    p.add_argument("--sam-checkpoint", default="external/segment-anything/sam_vit_b_01ec64.pth")
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

def window_softargmax(sim_hw, cx, cy, w, temp):
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

def to_np(t):
    return t.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)

def save_figure(sample, pair_id, kp_idx, F_src, F_trg, args, out_dir):
    H, W = sample["src_img"].shape[1:]
    C, fh, fw = F_trg.shape
    src_kp = sample["src_kps"][kp_idx]
    trg_kp = sample["trg_kps"][kp_idx]

    src_fx = src_kp[0].item() / W * fw
    src_fy = src_kp[1].item() / H * fh
    grid = torch.tensor([[[[src_fx / (fw - 1) * 2 - 1, src_fy / (fh - 1) * 2 - 1]]]],
                        dtype=torch.float32, device=args.device)
    desc = F.grid_sample(F_src.unsqueeze(0), grid, mode="bilinear", align_corners=True)
    desc = desc.squeeze().reshape(C)
    sim = (F_trg * desc.view(C, 1, 1)).sum(dim=0)

    flat_idx = sim.cpu().flatten().argmax().item()
    ax_fy, ax_fx = divmod(flat_idx, fw)
    ax_px = (ax_fx + 0.5) / fw * W
    ax_py = (ax_fy + 0.5) / fh * H
    wsa_fx, wsa_fy = window_softargmax(sim.cpu(), ax_fx, ax_fy, args.window_size, args.wsa_temp)
    wsa_px = (wsa_fx + 0.5) / fw * W
    wsa_py = (wsa_fy + 0.5) / fh * H

    src_np = to_np(sample["src_img"])
    trg_np = to_np(sample["trg_img"])
    sim_np = sim.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    category = sample["category"]
    err_ax  = ((ax_px  - trg_kp[0])**2 + (ax_py  - trg_kp[1])**2)**0.5
    err_wsa = ((wsa_px - trg_kp[0])**2 + (wsa_py - trg_kp[1])**2)**0.5
    fig.suptitle(f"{category} | pair {pair_id} kp {kp_idx} | "
                 f"Argmax err={err_ax:.1f}px  WSA err={err_wsa:.1f}px  Δ={err_ax-err_wsa:.1f}px",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(src_np)
    axes[0].scatter(src_kp[0], src_kp[1], c="red", s=120, edgecolors="white", linewidths=1.5, zorder=5)
    axes[0].set_title("Source keypoint", fontsize=11); axes[0].axis("off")

    axes[1].imshow(trg_np)
    sim_up = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(H, W),
                           mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    axes[1].imshow(sim_up, cmap="hot", alpha=0.55)
    axes[1].scatter(trg_kp[0], trg_kp[1], c="lime", s=180, edgecolors="black", linewidths=1.5, zorder=6, label="GT")
    axes[1].scatter(ax_px,  ax_py,  c="white", s=120, marker="x", linewidths=2.5, zorder=7, label=f"Argmax ({err_ax:.1f}px)")
    axes[1].scatter(wsa_px, wsa_py, c="cyan",  s=120, marker="P", edgecolors="black", linewidths=1.0, zorder=7, label=f"WSA ({err_wsa:.1f}px)")
    axes[1].set_title("Similarity heatmap", fontsize=11)
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.85); axes[1].axis("off")

    zoom = 3; patch_size = W // fw
    z0x = max(0, int(ax_px) - zoom * patch_size)
    z1x = min(W, int(ax_px) + zoom * patch_size)
    z0y = max(0, int(ax_py) - zoom * patch_size)
    z1y = min(H, int(ax_py) + zoom * patch_size)
    axes[2].imshow(trg_np[z0y:z1y, z0x:z1x])
    for gx in range(0, z1x - z0x, patch_size):
        axes[2].axvline(gx, color="white", linewidth=0.5, alpha=0.6)
    for gy in range(0, z1y - z0y, patch_size):
        axes[2].axhline(gy, color="white", linewidth=0.5, alpha=0.6)
    axes[2].scatter(ax_px - z0x,  ax_py - z0y,  c="white", s=150, marker="x", linewidths=2.5, zorder=7, label="Argmax")
    axes[2].scatter(wsa_px - z0x, wsa_py - z0y, c="cyan",  s=150, marker="P", edgecolors="black", linewidths=1.0, zorder=7, label="WSA")
    axes[2].scatter(trg_kp[0] - z0x, trg_kp[1] - z0y, c="lime", s=180, edgecolors="black", linewidths=1.5, zorder=6, label="GT")
    axes[2].set_title("Zoomed: patch grid", fontsize=11)
    axes[2].legend(loc="lower right", fontsize=8, framealpha=0.85); axes[2].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"BEST_{category}_pair{pair_id}_kp{kp_idx}_delta{err_ax-err_wsa:.1f}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset  = build_dataset(args)
    backbone = build_backbone(args)

    candidates = []
    print(f"Scanning {args.num_scan} pairs...")

    for pair_id in range(min(args.num_scan, len(dataset))):
        sample = dataset[pair_id]
        n_pts  = int(sample["n_pts"].item())
        H, W   = sample["src_img"].shape[1:]

        with torch.no_grad():
            F_src = backbone(sample["src_img"].unsqueeze(0).to(args.device)).float().squeeze(0)
            F_trg = backbone(sample["trg_img"].unsqueeze(0).to(args.device)).float().squeeze(0)
        F_src = F.normalize(F_src, dim=0)
        F_trg = F.normalize(F_trg, dim=0)
        C, fh, fw = F_trg.shape

        for kp_idx in range(n_pts):
            src_kp = sample["src_kps"][kp_idx]
            trg_kp = sample["trg_kps"][kp_idx]

            src_fx = src_kp[0].item() / W * fw
            src_fy = src_kp[1].item() / H * fh
            grid = torch.tensor([[[[src_fx / (fw-1)*2-1, src_fy / (fh-1)*2-1]]]],
                                dtype=torch.float32, device=args.device)
            desc = F.grid_sample(F_src.unsqueeze(0), grid, mode="bilinear", align_corners=True)
            desc = desc.squeeze().reshape(C)
            sim  = (F_trg * desc.view(C,1,1)).sum(dim=0)

            flat_idx = sim.cpu().flatten().argmax().item()
            ax_fy_f, ax_fx_f = divmod(flat_idx, fw)
            ax_px = (ax_fx_f + 0.5) / fw * W
            ax_py = (ax_fy_f + 0.5) / fh * H
            wsa_fx, wsa_fy = window_softargmax(sim.cpu(), ax_fx_f, ax_fy_f, args.window_size, args.wsa_temp)
            wsa_px = (wsa_fx + 0.5) / fw * W
            wsa_py = (wsa_fy + 0.5) / fh * H

            err_ax  = ((ax_px  - trg_kp[0].item())**2 + (ax_py  - trg_kp[1].item())**2)**0.5
            err_wsa = ((wsa_px - trg_kp[0].item())**2 + (wsa_py - trg_kp[1].item())**2)**0.5
            delta = err_ax - err_wsa

            # Good example: argmax close-ish (5-50px), WSA clearly better
            if 5 < err_ax < 50 and delta > 2:
                candidates.append((delta, pair_id, kp_idx, sample["category"],
                                   err_ax, err_wsa, F_src, F_trg, sample))

        if pair_id % 20 == 0:
            print(f"  [{pair_id}/{args.num_scan}] candidates so far: {len(candidates)}")

    if not candidates:
        print("No good candidates found. Try increasing --num-scan.")
        return

    # Sort by delta (largest improvement first), save top 5
    candidates.sort(key=lambda x: -x[0])
    print(f"\nTop candidates (Argmax err → WSA err, delta):")
    for i, (delta, pid, kid, cat, e_ax, e_wsa, Fs, Ft, samp) in enumerate(candidates[:5]):
        print(f"  {i+1}. pair={pid} kp={kid} cat={cat}  argmax={e_ax:.1f}px  wsa={e_wsa:.1f}px  Δ={delta:.1f}px")
        save_figure(samp, pid, kid, Fs, Ft, args, out_dir)

    print(f"\nDone. Best example: pair {candidates[0][1]}, kp {candidates[0][2]}, category={candidates[0][3]}")
    print(f"Use: --pair-id {candidates[0][1]} --kp-idx {candidates[0][2]}")

if __name__ == "__main__":
    main()
