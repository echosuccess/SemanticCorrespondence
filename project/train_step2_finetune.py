"""Step 2: Light Fine-tuning of the Last Backbone Layers.

Keeps the same training-free pipeline but unfreezes the last N transformer
blocks of a DINOv2 (or DINOv3) backbone and fine-tunes them with an InfoNCE
correspondence loss on the SPair-71k **train** split.

An ablation over the number of unfrozen layers is run automatically when
``--n-unfrozen-layers`` receives more than one value (e.g. ``1 2 4``).
Model selection is performed on the **val** split using PCK@0.1 (by-image).

Outputs (per unfrozen-layer count N)
-------------------------------------
    <output-dir>/n<N>/<backbone>_n<N>_best.pth   – best checkpoint
    <output-dir>/n<N>/train_log.json              – per-epoch loss & val PCK
    <output-dir>/ablation_summary.json            – cross-N comparison table

Usage
-----
    python project/train_step2_finetune.py \\
        --backbone dinov2_vitb14 \\
        --n-unfrozen-layers 1 2 4 \\
        --dino-repo external/dinov2 \\
        --sd4match-dir SD4Match \\
        --data-root /content/drive/MyDrive/Semantic-Correspondence \\
        --epochs 5 \\
        --output-dir project/results/step2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune last N layers of a DINOv2/DINOv3 backbone "
                    "for semantic correspondence (Step 2)."
    )
    # backbone
    p.add_argument("--backbone", type=str, default="dinov2_vitb14",
                   help="Backbone name, e.g. dinov2_vitb14 / dinov3_vitb16.")
    p.add_argument("--dino-repo", type=str, default="external/dinov2",
                   help="Local clone of facebookresearch/dinov2 or dinov3.")
    p.add_argument("--dinov3-weights", type=str,
                   default="external/dinov3_weights/dinov3_vitb16.pth")
    p.add_argument("--sam-checkpoint", type=str,
                   default="external/segment-anything/sam_vit_b_01ec64.pth")

    # ablation
    p.add_argument("--n-unfrozen-layers", type=int, nargs="+", default=[1, 2, 4],
                   help="Number(s) of last transformer blocks to unfreeze. "
                        "Multiple values → run each in sequence.")

    # data
    p.add_argument("--sd4match-dir", type=str, default="../SD4Match")
    p.add_argument("--data-root", type=str, default="../")
    p.add_argument("--train-img-size", type=int, default=224,
                   help="Image size for training (lower = faster, less memory).")
    p.add_argument("--eval-img-size", type=int, default=224,
                   help="Image size used for val-PCK evaluation during training.")
    p.add_argument("--category", type=str, default="all")

    # training hyper-params
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps-per-epoch", type=int, default=1000,
                   help="Cap on gradient updates per epoch. "
                        "-1 means use the full training set.")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Number of image pairs per gradient update.")
    p.add_argument("--accum-steps", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = "
                        "batch-size × accum-steps).")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature.")
    p.add_argument("--num-workers", type=int, default=2)

    # runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sd4match-dir-eval", type=str, default=None,
                   help="SD4Match dir for evaluation (defaults to --sd4match-dir).")

    # output
    p.add_argument("--output-dir", type=str, default="project/results/step2")

    return p.parse_args()


# ---------------------------------------------------------------------------
# InfoNCE correspondence loss
# ---------------------------------------------------------------------------

def correspondence_loss(
    F_src: torch.Tensor,
    F_trg: torch.Tensor,
    src_kps: torch.Tensor,
    trg_kps: torch.Tensor,
    img_size: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE loss for a single image pair.

    For every source keypoint the positive sample is the target feature-map
    patch whose centre is nearest to the ground-truth target keypoint.
    All other ``h*w`` target patches act as negatives.

    Parameters
    ----------
    F_src, F_trg : (1, C, h, w) – feature maps with gradients attached.
    src_kps, trg_kps : (N, 2) – keypoints in image-pixel coords ``(x, y)``.
    img_size : int – spatial side-length of the input images fed to the
        backbone (used to convert pixel coords → patch indices).
    temperature : float

    Returns
    -------
    Scalar loss tensor.  Returns 0.0 (detached) if N == 0.
    """
    N = src_kps.shape[0]
    if N == 0:
        return F_src.sum() * 0.0

    C, h, w = F_src.shape[1], F_src.shape[2], F_src.shape[3]
    device = F_src.device

    # L2-normalise along channel dimension
    F_src_n = F.normalize(F_src, p=2, dim=1)  # (1, C, h, w)
    F_trg_n = F.normalize(F_trg, p=2, dim=1)  # (1, C, h, w)

    # ---- Sample source features at keypoint locations (bilinear) -----------
    # grid_sample expects grid in [-1, 1] relative to the feature-map extent.
    kp = src_kps.float().to(device)
    grid_x = kp[:, 0] / (img_size - 1) * 2 - 1  # (N,)
    grid_y = kp[:, 1] / (img_size - 1) * 2 - 1  # (N,)
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, N, 2)  # (1,1,N,2)

    f_src_kps = F.grid_sample(
        F_src_n, grid,
        mode="bilinear", align_corners=True, padding_mode="border",
    )  # (1, C, 1, N)
    f_src_kps = f_src_kps.squeeze(0).squeeze(1).T  # (N, C)

    # ---- Flatten target feature map to (h*w, C) ----------------------------
    F_trg_flat = F_trg_n.squeeze(0).reshape(C, h * w).T  # (h*w, C)

    # ---- Compute similarity logits (N, h*w) --------------------------------
    logits = (f_src_kps @ F_trg_flat.T) / temperature  # (N, h*w)

    # ---- Positive patch index for each keypoint ----------------------------
    # Map GT target keypoint pixel coord → nearest patch index.
    tk = trg_kps.float().to(device)
    px = (tk[:, 0] / img_size * w).long().clamp(0, w - 1)  # (N,)
    py = (tk[:, 1] / img_size * h).long().clamp(0, h - 1)  # (N,)
    pos_idx = py * w + px  # (N,)

    return F.cross_entropy(logits, pos_idx)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _add_sd4match_to_syspath(sd4match_dir: str) -> None:
    sd4match_dir = os.path.abspath(sd4match_dir)
    if sd4match_dir not in sys.path:
        sys.path.insert(0, sd4match_dir)


def build_dataset(args: argparse.Namespace, split: str, img_size: int):
    from config.base import get_default_defaults
    from spair_dataset import SafeSPairDataset as SPairDataset

    cfg = get_default_defaults()
    cfg.DATASET.NAME = "spair"
    cfg.DATASET.ROOT = os.path.abspath(args.data_root)
    cfg.DATASET.IMG_SIZE = img_size
    cfg.DATASET.MEAN = [0.0, 0.0, 0.0]
    cfg.DATASET.STD = [1.0, 1.0, 1.0]
    cfg.EVALUATOR.ALPHA = [0.05, 0.1, 0.2]
    cfg.EVALUATOR.BY = "image"

    return cfg, SPairDataset(cfg, split=split, category=args.category)


# ---------------------------------------------------------------------------
# Backbone construction
# ---------------------------------------------------------------------------

def build_trainable_backbone(args: argparse.Namespace, n_unfrozen: int):
    """Return a FinetunableBackbone ready for gradient-based training."""
    from backbones import build_backbone as _build
    from backbones.finetune_wrapper import FinetunableBackbone

    if args.backbone.startswith("dinov2_"):
        backbone = _build(
            args.backbone,
            repo_dir=args.dino_repo,
            input_size=args.train_img_size,
            device=args.device,
        )
    elif args.backbone.startswith("dinov3_"):
        backbone = _build(
            args.backbone,
            repo_dir=args.dino_repo,
            input_size=args.train_img_size,
            device=args.device,
            weights_path=args.dinov3_weights,
        )
    elif args.backbone.startswith("sam_"):
        backbone = _build(
            args.backbone,
            checkpoint=args.sam_checkpoint,
            device=args.device,
        )
        # Override input_size so the wrapper uses train_img_size for resizing
        backbone.input_size = args.train_img_size
    else:
        raise ValueError(
            f"Unsupported backbone for fine-tuning: '{args.backbone}'. "
            f"Expected dinov2_*, dinov3_*, or sam_*."
        )

    model = FinetunableBackbone(backbone, n_unfrozen_layers=n_unfrozen)
    n_blocks = model.n_total_blocks()
    n_params = model.n_trainable_params()
    print(
        f"  Backbone : {args.backbone}  ({n_blocks} blocks total)\n"
        f"  Unfrozen : last {n_unfrozen} block(s) + norm\n"
        f"  Trainable: {n_params:,} parameters"
    )
    return model


# ---------------------------------------------------------------------------
# Validation – quick PCK@0.1 (by-image) on the val split
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, cfg, device: str) -> float:
    """Return PCK@0.1 (by-image) over the val split.

    Uses SD4Match's PCKEvaluator with a single threshold (0.1) for speed.
    """
    from utils.evaluation import PCKEvaluator

    evaluator = PCKEvaluator(cfg)
    model.backbone.model.eval()

    for batch in val_loader:
        src_img = batch["src_img"].to(device)
        trg_img = batch["trg_img"].to(device)

        F_src = model(src_img).float().cpu()
        F_trg = model(trg_img).float().cpu()

        batch["src_featmaps"] = F_src
        batch["trg_featmaps"] = F_trg
        evaluator.evaluate_batch(batch)

    results = evaluator.get_results()
    # results is a dict of dicts; extract mean PCK at alpha=0.1
    try:
        pck_01 = float(results["0.1"]["mean"])
    except (KeyError, TypeError):
        # fall back to first available key
        first_alpha = next(iter(results))
        pck_01 = float(results[first_alpha]["mean"])
    return pck_01


# ---------------------------------------------------------------------------
# Training loop for a single N
# ---------------------------------------------------------------------------

def train_one_n(
    args: argparse.Namespace,
    n_unfrozen: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_cfg,
    out_dir: Path,
) -> dict:
    """Train for ``args.epochs`` epochs and return the training log."""
    model = build_trainable_backbone(args, n_unfrozen)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    tag = f"{args.backbone}_n{n_unfrozen}"
    best_pck = -1.0
    log = {"backbone": args.backbone, "n_unfrozen": n_unfrozen, "epochs": []}

    print(f"\n{'='*60}")
    print(f"  Training  {tag}  (epochs={args.epochs})")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        model.set_train_eval_mode()
        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step_in_epoch, batch in enumerate(train_loader):
            # Optionally cap the number of gradient updates per epoch
            if args.steps_per_epoch > 0 and n_steps >= args.steps_per_epoch:
                break

            src_img = batch["src_img"].to(args.device)
            trg_img = batch["trg_img"].to(args.device)
            n_pts_batch = batch["n_pts"]          # (B,)

            # Process each pair in the micro-batch individually
            # (keypoint count varies between pairs)
            loss_accum = torch.tensor(0.0, device=args.device)
            n_valid = 0
            for b_idx in range(src_img.shape[0]):
                n_pts = int(n_pts_batch[b_idx].item())
                if n_pts == 0:
                    continue
                src_kps = batch["src_kps"][b_idx, :n_pts]  # (N, 2)
                trg_kps = batch["trg_kps"][b_idx, :n_pts]  # (N, 2)

                F_src = model(src_img[b_idx].unsqueeze(0))  # (1, C, h, w)
                F_trg = model(trg_img[b_idx].unsqueeze(0))  # (1, C, h, w)

                pair_loss = correspondence_loss(
                    F_src, F_trg,
                    src_kps, trg_kps,
                    img_size=args.train_img_size,
                    temperature=args.temperature,
                )
                loss_accum = loss_accum + pair_loss
                n_valid += 1

            if n_valid == 0:
                continue

            loss_mean = loss_accum / n_valid / args.accum_steps
            loss_mean.backward()
            epoch_loss += loss_mean.item() * args.accum_steps
            n_steps += 1

            if n_steps % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Flush any remaining gradients
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - t0

        # ---- Validation ----
        val_pck = validate(model, val_loader, val_cfg, args.device)

        print(
            f"  Epoch {epoch:2d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"val_PCK@0.1={val_pck:.4f} | "
            f"steps={n_steps} | "
            f"time={elapsed:.0f}s"
        )

        epoch_record = {
            "epoch": epoch,
            "avg_loss": round(avg_loss, 6),
            "val_pck_01": round(val_pck, 6),
            "n_steps": n_steps,
            "elapsed_s": round(elapsed, 1),
        }
        log["epochs"].append(epoch_record)

        # ---- Save best checkpoint ----
        if val_pck > best_pck:
            best_pck = val_pck
            ckpt_path = out_dir / f"{tag}_best.pth"
            model.save_checkpoint(ckpt_path, extra={"best_val_pck_01": best_pck})
            print(f"    ↑ New best  val_PCK@0.1={best_pck:.4f}  saved → {ckpt_path}")

    log["best_val_pck_01"] = round(best_pck, 6)
    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Set up SD4Match on sys.path ----
    _add_sd4match_to_syspath(args.sd4match_dir)
    sd4match_eval = args.sd4match_dir_eval or args.sd4match_dir
    if sd4match_eval != args.sd4match_dir:
        _add_sd4match_to_syspath(sd4match_eval)

    # ---- Datasets & loaders ----
    print("Loading training dataset …")
    _, train_ds = build_dataset(args, split="trn", img_size=args.train_img_size)
    print(f"  train pairs: {len(train_ds)}")

    print("Loading validation dataset …")
    val_cfg, val_ds = build_dataset(args, split="val", img_size=args.eval_img_size)
    print(f"  val   pairs: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Ablation loop ----
    out_root = Path(args.output_dir)
    ablation_logs: List[dict] = []

    for n in args.n_unfrozen_layers:
        n_dir = out_root / f"n{n}"
        n_dir.mkdir(parents=True, exist_ok=True)

        log = train_one_n(args, n, train_loader, val_loader, val_cfg, n_dir)

        # Save per-N log
        log_path = n_dir / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  Log saved → {log_path}")

        ablation_logs.append(
            {
                "n_unfrozen": n,
                "best_val_pck_01": log["best_val_pck_01"],
                "checkpoint": str(n_dir / f"{args.backbone}_n{n}_best.pth"),
            }
        )

    # ---- Ablation summary ----
    summary_path = out_root / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"backbone": args.backbone, "results": ablation_logs},
            f, indent=2,
        )

    print(f"\n{'='*60}")
    print("  Ablation summary (val PCK@0.1 by-image)")
    print(f"{'='*60}")
    print(f"  {'N unfrozen':>12}  {'val PCK@0.1':>12}  Checkpoint")
    for row in ablation_logs:
        print(
            f"  {row['n_unfrozen']:>12}  "
            f"{row['best_val_pck_01']:>12.4f}  "
            f"{row['checkpoint']}"
        )
    print(f"\nFull summary → {summary_path}")
    print(
        "\nNext step: evaluate the best checkpoint on the TEST split with:\n"
        "  python project/run_step1_trainfree.py \\\n"
        f"      --backbone {args.backbone} \\\n"
        "      --finetune-checkpoint <best .pth> \\\n"
        "      --split test  --by image  --output-dir project/results/step2/eval"
    )


if __name__ == "__main__":
    main()
