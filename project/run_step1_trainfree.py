"""Step 1 - Training-free semantic correspondence baseline.

Pipeline (no training, no optimiser, only frozen features):

    for each (src_img, trg_img) pair in SPair-71k test split:
        F_src = backbone(src_img)                  # (C, h, w) dense features
        F_trg = backbone(trg_img)
        for each annotated source keypoint p:
            f_p = bilinear_sample(F_src, p)        # (C,)
            sim = cos(f_p, F_trg)                  # (h, w)
            p_hat = argmax(sim)                    # predicted target location
        PCK@alpha(p_hat, p_gt)

Matching + PCK are computed by ``utils.evaluator.PCKEvaluator`` from the
SD4Match repo (which also reports the bilinear / soft-argmax variants for
reference; ``nn`` is the pure argmax baseline required by Step 1).

Example
-------
    python project/run_step1_trainfree.py \
        --backbone dinov2_vitb14 \
        --dino-repo external/dinov2 \
        --data-root /content/drive/MyDrive/AML_Project/data \
        --split test \
        --img-size 512 \
        --batch-size 4 \
        --output-dir results/step1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _add_sd4match_to_syspath(sd4match_dir: str) -> None:
    """Make SD4Match's ``dataset``/``utils``/``config`` packages importable."""
    sd4match_dir = os.path.abspath(sd4match_dir)
    if sd4match_dir not in sys.path:
        sys.path.insert(0, sd4match_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Training-free semantic correspondence on SPair-71k."
    )
    p.add_argument("--backbone", type=str, required=True,
                   help="e.g. dinov2_vitb14 / dinov3_vitb16 / sam_vit_b")

    # dataset
    p.add_argument("--data-root", type=str, default="../",
                   help="Parent dir that contains a 'SPair-71k' folder.")
    p.add_argument("--split", type=str, default="test",
                   choices=["trn", "val", "test"],
                   help="SPair-71k split to evaluate on.")
    p.add_argument("--category", type=str, default="all",
                   help="'all' or one of the 18 SPair-71k categories.")
    p.add_argument("--img-size", type=int, default=512,
                   help="Images are resized to (img_size, img_size) before "
                        "being fed to the backbone; the backbone applies any "
                        "further resize it needs internally.")

    # backbone-specific options
    p.add_argument("--dino-repo", type=str, default="external/dinov2",
                   help="Local clone of facebookresearch/dinov2 or dinov3.")
    p.add_argument("--sam-checkpoint", type=str,
                   default="external/segment-anything/sam_vit_b_01ec64.pth",
                   help="Path to the SAM .pth checkpoint.")
    p.add_argument("--backbone-input-size", type=int, default=None,
                   help="Override the backbone's default input size "
                        "(e.g. 518 for DINOv2, 512 for DINOv3, 1024 for SAM).")

    # evaluation
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.15],
                   help="PCK thresholds (fractions of the bbox side).")
    p.add_argument("--by", type=str, default="image", choices=["image", "point"],
                   help="Whether to average PCK per image or per keypoint.")
    p.add_argument("--softmax-temp", type=float, default=0.04,
                   help="Temperature used by the soft-argmax matchers "
                        "(only affects the 'softmax' / 'kernelsoftmax' rows).")

    # runtime
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-pairs", type=int, default=-1,
                   help="If > 0, only evaluate that many pairs (for debugging).")

    # IO
    p.add_argument("--sd4match-dir", type=str, default="../SD4Match",
                   help="Path to the SD4Match repo clone.")
    p.add_argument("--output-dir", type=str, default="results/step1",
                   help="Where to write the PCK summary and the JSON dump.")

    return p.parse_args()


def build_backbone(args: argparse.Namespace):
    """Instantiate the requested frozen backbone."""
    # Local import so that the SD4Match sys.path insertion happens first.
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
    """Build the SPair-71k dataset using SD4Match's loader."""
    from config.base import get_default_defaults
    from dataset.spair import SPairDataset

    cfg = get_default_defaults()
    cfg.DATASET.NAME = "spair"
    cfg.DATASET.ROOT = os.path.abspath(args.data_root)
    cfg.DATASET.IMG_SIZE = args.img_size
    # Feed raw [0,1] images to the backbones; each backbone applies its own
    # normalization internally. This keeps the pipeline backbone-agnostic.
    cfg.DATASET.MEAN = [0.0, 0.0, 0.0]
    cfg.DATASET.STD = [1.0, 1.0, 1.0]

    cfg.EVALUATOR.ALPHA = list(args.alphas)
    cfg.EVALUATOR.BY = args.by

    dataset = SPairDataset(cfg, split=args.split, category=args.category)
    return cfg, dataset


def main() -> None:
    args = parse_args()
    _add_sd4match_to_syspath(args.sd4match_dir)

    cfg, dataset = build_dataset(args)

    from utils.evaluator import PCKEvaluator  # noqa: E402  (after sys.path)

    evaluator = PCKEvaluator(cfg)
    backbone = build_backbone(args)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    n_pairs_seen = 0
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        src_img = batch["src_img"]  # (B, 3, H, W), in [0, 1]
        trg_img = batch["trg_img"]

        # Inputs stay on CPU here; each backbone moves them to its device.
        batch["src_featmaps"] = backbone(src_img).float()
        batch["trg_featmaps"] = backbone(trg_img).float()

        # The evaluator compares every matcher (nn / bilinear / soft-argmax)
        # against the ground truth and updates its internal counters.
        evaluator.evaluate_feature_map(
            batch,
            softmax_temp=args.softmax_temp,
            gaussian_suppression_sigma=7,
            enable_l2_norm=True,
        )

        n_pairs_seen += src_img.shape[0]
        if i % 50 == 0:
            dt = time.perf_counter() - t0
            ips = n_pairs_seen / max(dt, 1e-6)
            print(f"[{n_pairs_seen}/{len(dataset)}] "
                  f"elapsed={dt:.1f}s  pairs/s={ips:.2f}")

        if args.max_pairs > 0 and n_pairs_seen >= args.max_pairs:
            break

    dt = time.perf_counter() - t0
    print(f"\nFinished. Evaluated {n_pairs_seen} pairs in {dt:.1f}s "
          f"({n_pairs_seen / max(dt, 1e-6):.2f} pairs/s).\n")

    # ---- persist results ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.backbone}_{args.split}_{args.category}_by{args.by}"
    txt_path = out_dir / f"{tag}.txt"
    json_path = out_dir / f"{tag}.json"

    evaluator.print_summarize_result()
    evaluator.save_result(str(txt_path))

    summary = evaluator.summerize_result()
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nSaved per-category report to {txt_path}")
    print(f"Saved machine-readable summary to {json_path}")

    if torch.cuda.is_available():
        max_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Peak GPU memory: {max_mem_mb:.0f} MiB")


if __name__ == "__main__":
    main()
