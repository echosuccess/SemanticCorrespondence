"""Step 4: LoRA Fine-tuning — Efficient Adaptation via Low-Rank Matrices.

Instead of unfreezing entire transformer blocks (Step 2), we inject tiny
trainable matrices A (d×r) and B (r×d) alongside every attention linear
layer.  All original weights stay frozen; only A and B are trained.

    W' x = W x + (A B x) * (alpha / r)

Benefits over Step 2
--------------------
* Far fewer trainable parameters (r=8 → ~0.4 M vs. ~14 M for N=4).
* LoRA touches *all* layers at once; Step 2 only touches the last N.
* Easier to compare parameter-efficiency vs. performance.

Ablation
--------
Runs the full training pipeline for each rank in ``--lora-ranks``
(default: 4 8 16) and writes per-rank checkpoints + a summary table.

Outputs (per rank r)
--------------------
    <output-dir>/r<r>/<backbone>_lora_r<r>_best.pth  – best checkpoint
    <output-dir>/r<r>/train_log.json                  – training log
    <output-dir>/ablation_summary.json                – rank comparison

Usage
-----
    python project/train_step4_lora.py \\
        --backbone dinov2_vitb14 \\
        --lora-ranks 4 8 16 \\
        --lora-alpha 32 \\
        --dino-repo external/dinov2 \\
        --sd4match-dir SD4Match \\
        --data-root /content/drive/MyDrive/Semantic-Correspondence \\
        --epochs 5 --steps-per-epoch 500 \\
        --output-dir project/results/step4/dinov2
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
        description="Step 4: LoRA fine-tuning for semantic correspondence."
    )
    # backbone
    p.add_argument("--backbone", type=str, default="dinov2_vitb14")
    p.add_argument("--dino-repo", type=str, default="external/dinov2")
    p.add_argument("--dinov3-weights", type=str,
                   default="external/dinov3_weights/dinov3_vitb16.pth")
    p.add_argument("--sam-checkpoint", type=str,
                   default="external/segment-anything/sam_vit_b_01ec64.pth")

    # LoRA ablation
    p.add_argument("--lora-ranks", type=int, nargs="+", default=[4, 8, 16],
                   help="LoRA ranks to ablate (default: 4 8 16).")
    p.add_argument("--lora-alpha", type=float, default=32.0,
                   help="LoRA scaling  alpha/r  (default: 32, same as reference).")
    p.add_argument("--lora-dropout", type=float, default=0.0)

    # data
    p.add_argument("--sd4match-dir", type=str, default="../SD4Match")
    p.add_argument("--data-root", type=str, default="../")
    p.add_argument("--train-img-size", type=int, default=224)
    p.add_argument("--eval-img-size", type=int, default=224)
    p.add_argument("--category", type=str, default="all")

    # training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature.")
    p.add_argument("--num-workers", type=int, default=2)

    # runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sd4match-dir-eval", type=str, default=None)

    # output
    p.add_argument("--output-dir", type=str, default="project/results/step4")

    return p.parse_args()


# ---------------------------------------------------------------------------
# InfoNCE loss  (identical to train_step2_finetune.py)
# ---------------------------------------------------------------------------

def correspondence_loss(
    F_src: torch.Tensor,
    F_trg: torch.Tensor,
    src_kps: torch.Tensor,
    trg_kps: torch.Tensor,
    img_size: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    N = src_kps.shape[0]
    if N == 0:
        return F_src.sum() * 0.0

    C, h, w = F_src.shape[1], F_src.shape[2], F_src.shape[3]
    device = F_src.device

    F_src_n = F.normalize(F_src, p=2, dim=1)
    F_trg_n = F.normalize(F_trg, p=2, dim=1)

    kp = src_kps.float().to(device)
    grid_x = kp[:, 0] / (img_size - 1) * 2 - 1
    grid_y = kp[:, 1] / (img_size - 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, N, 2)

    f_src_kps = F.grid_sample(
        F_src_n, grid, mode="bilinear", align_corners=True, padding_mode="border",
    ).squeeze(0).squeeze(1).T  # (N, C)

    F_trg_flat = F_trg_n.squeeze(0).reshape(C, h * w).T  # (h*w, C)
    logits = (f_src_kps @ F_trg_flat.T) / temperature     # (N, h*w)

    tk = trg_kps.float().to(device)
    px = (tk[:, 0] / img_size * w).long().clamp(0, w - 1)
    py = (tk[:, 1] / img_size * h).long().clamp(0, h - 1)
    pos_idx = py * w + px

    return F.cross_entropy(logits, pos_idx)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _add_sd4match_to_syspath(sd4match_dir: str) -> None:
    sd4match_dir = os.path.abspath(sd4match_dir)
    if sd4match_dir not in sys.path:
        sys.path.insert(0, sd4match_dir)


def build_dataset(args, split: str, img_size: int):
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

def build_lora_backbone(args, r: int):
    """Return a LoRABackbone ready for training."""
    from backbones import build_backbone as _build
    from lora_backbone import LoRABackbone

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
        backbone.input_size = args.train_img_size
    else:
        raise ValueError(f"Unsupported backbone: '{args.backbone}'")

    model = LoRABackbone(
        backbone,
        r=r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, cfg, device: str) -> float:
    from utils.evaluator import PCKEvaluator

    evaluator = PCKEvaluator(cfg)
    model.backbone.model.eval()

    for batch in val_loader:
        src_img = batch["src_img"].to(device)
        trg_img = batch["trg_img"].to(device)
        batch["src_featmaps"] = model(src_img).float().cpu()
        batch["trg_featmaps"] = model(trg_img).float().cpu()
        evaluator.evaluate_feature_map(
            batch, softmax_temp=0.04,
            gaussian_suppression_sigma=7, enable_l2_norm=True,
        )

    results = evaluator.summerize_result()
    try:
        return float(results["nn_pck0.1"]["all"])
    except KeyError:
        first_key = next(k for k in results if k.startswith("nn_pck"))
        return float(results[first_key]["all"])


# ---------------------------------------------------------------------------
# Training loop for one rank value
# ---------------------------------------------------------------------------

def train_one_rank(
    args,
    r: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_cfg,
    out_dir: Path,
) -> dict:
    model = build_lora_backbone(args, r)
    model.to(args.device)

    # Only LoRA parameters are trained
    lora_params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        lora_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    tag = f"{args.backbone}_lora_r{r}"
    best_pck = -1.0
    start_epoch = 1
    log = {"backbone": args.backbone, "lora_r": r,
           "lora_alpha": args.lora_alpha, "epochs": []}

    # ---- resume ----
    latest_path = out_dir / f"{tag}_latest.pth"
    if latest_path.exists():
        print(f"[resume] Loading state from {latest_path}")
        state = torch.load(latest_path, map_location=args.device)
        # Load LoRA weights back into the already-injected model
        lora_state = state["lora_state"]
        model_state = model.backbone.model.state_dict()
        model_state.update(lora_state)
        model.backbone.model.load_state_dict(model_state, strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_pck = state["best_pck"]
        log = state["log"]
        print(f"[resume] Epoch {start_epoch}/{args.epochs}, best_pck={best_pck:.4f}")

    print(f"\n{'='*60}")
    print(f"  Training  {tag}  (epochs={args.epochs})")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.set_train_eval_mode()
        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step_in_epoch, batch in enumerate(train_loader):
            if args.steps_per_epoch > 0 and n_steps >= args.steps_per_epoch:
                break

            src_img = batch["src_img"].to(args.device)
            trg_img = batch["trg_img"].to(args.device)
            n_pts_batch = batch["n_pts"]

            loss_accum = torch.tensor(0.0, device=args.device)
            n_valid = 0
            for b_idx in range(src_img.shape[0]):
                n_pts = int(n_pts_batch[b_idx].item())
                if n_pts == 0:
                    continue
                src_kps = batch["src_kps"][b_idx, :n_pts]
                trg_kps = batch["trg_kps"][b_idx, :n_pts]

                F_src = model(src_img[b_idx].unsqueeze(0))
                F_trg = model(trg_img[b_idx].unsqueeze(0))

                pair_loss = correspondence_loss(
                    F_src, F_trg, src_kps, trg_kps,
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

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - t0

        val_pck = validate(model, val_loader, val_cfg, args.device)
        print(
            f"  Epoch {epoch:2d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | val_PCK@0.1={val_pck:.4f} | "
            f"steps={n_steps} | time={elapsed:.0f}s"
        )

        epoch_record = {
            "epoch": epoch, "avg_loss": round(avg_loss, 6),
            "val_pck_01": round(val_pck, 6), "n_steps": n_steps,
            "elapsed_s": round(elapsed, 1),
        }
        log["epochs"].append(epoch_record)

        # ---- Save best checkpoint ----
        if val_pck > best_pck:
            best_pck = val_pck
            ckpt_path = out_dir / f"{tag}_best.pth"
            model.save_checkpoint(
                ckpt_path,
                extra={"best_val_pck_01": best_pck,
                       "lora_r": r, "lora_alpha": args.lora_alpha},
            )
            print(f"    ↑ New best val_PCK@0.1={best_pck:.4f} → {ckpt_path}")

        # ---- Save latest for resume ----
        lora_state = {
            name: param
            for name, param in model.backbone.model.named_parameters()
            if param.requires_grad
        }
        torch.save({
            "epoch":      epoch,
            "lora_state": lora_state,
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "best_pck":   best_pck,
            "log":        log,
        }, latest_path)

    if latest_path.exists():
        latest_path.unlink()

    log["best_val_pck_01"] = round(best_pck, 6)
    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    _add_sd4match_to_syspath(args.sd4match_dir)
    sd4match_eval = args.sd4match_dir_eval or args.sd4match_dir
    if sd4match_eval != args.sd4match_dir:
        _add_sd4match_to_syspath(sd4match_eval)

    print("Loading training dataset …")
    _, train_ds = build_dataset(args, split="trn", img_size=args.train_img_size)
    print(f"  train pairs: {len(train_ds)}")
    print("Loading validation dataset …")
    val_cfg, val_ds = build_dataset(args, split="val", img_size=args.eval_img_size)
    print(f"  val   pairs: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    out_root = Path(args.output_dir)
    ablation_logs: List[dict] = []

    for r in args.lora_ranks:
        r_dir = out_root / f"r{r}"
        r_dir.mkdir(parents=True, exist_ok=True)

        tag = f"{args.backbone}_lora_r{r}"
        best_path   = r_dir / f"{tag}_best.pth"
        latest_path = r_dir / f"{tag}_latest.pth"

        # Skip already-completed ranks
        if best_path.exists() and not latest_path.exists():
            log_path = r_dir / "train_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                best_pck = log.get("best_val_pck_01", "?")
            else:
                log = {"backbone": args.backbone, "lora_r": r,
                       "best_val_pck_01": None, "epochs": []}
                best_pck = "?"
            print(f"\n[skip] {tag} already finished "
                  f"(best_val_PCK@0.1={best_pck})")
            ablation_logs.append({
                "lora_r": r, "lora_alpha": args.lora_alpha,
                "best_val_pck_01": log.get("best_val_pck_01"),
                "checkpoint": str(best_path),
            })
            continue

        log = train_one_rank(args, r, train_loader, val_loader, val_cfg, r_dir)

        log_path = r_dir / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  Log saved → {log_path}")

        ablation_logs.append({
            "lora_r": r, "lora_alpha": args.lora_alpha,
            "best_val_pck_01": log["best_val_pck_01"],
            "checkpoint": str(r_dir / f"{tag}_best.pth"),
        })

    # ---- Ablation summary ----
    summary_path = out_root / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"backbone": args.backbone, "results": ablation_logs}, f, indent=2)

    print(f"\n{'='*60}")
    print("  LoRA ablation summary (val PCK@0.1 by-image)")
    print(f"{'='*60}")
    print(f"  {'r':>8}  {'val PCK@0.1':>12}  Checkpoint")
    for row in ablation_logs:
        pck = row["best_val_pck_01"]
        pck_str = f"{pck:>12.4f}" if pck is not None else f"{'N/A':>12}"
        print(f"  {row['lora_r']:>8}  {pck_str}  {row['checkpoint']}")
    print(f"\nFull summary → {summary_path}")
    print(
        "\nNext: evaluate the best checkpoint on TEST split with:\n"
        "  python project/run_step1_trainfree.py \\\n"
        f"      --backbone {args.backbone} \\\n"
        "      --lora-checkpoint <best .pth> \\\n"
        "      --split test --by image --output-dir project/results/step4/eval"
    )


if __name__ == "__main__":
    main()
