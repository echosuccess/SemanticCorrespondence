"""Step 3 - Window Soft-Argmax prediction refinement.

Replaces the plain argmax prediction of Step 1 with *window soft-argmax*:

    1. Find the peak location in the similarity map with argmax.
    2. Apply softmax *only within a small window* around that peak.
    3. Return the expected (weighted-mean) coordinate — a sub-pixel prediction.

This makes predictions more robust to noisy similarity maps and allows
sub-pixel accuracy.  All five matching strategies are evaluated
side-by-side so the improvement is immediately visible:

    nn / bilinear / softmax / kernelsoftmax / window_softargmax  (◄ Step 3)

The script is structurally identical to run_step1_trainfree.py but uses
WindowSoftArgmaxEvaluator instead of PCKEvaluator.  It also supports the
same checkpoint/resume mechanism, so it can be safely interrupted and
restarted on Colab.

Usage examples
--------------
# Evaluate DINOv2 ViT-B baseline with window soft-argmax (window=5):
python project/run_step3_prediction.py \\
    --backbone dinov2_vitb14 \\
    --split test --img-size 518 \\
    --window-size 5 \\
    --output-dir project/results/step3/dinov2_vitb14

# Evaluate with Step-2 fine-tuned weights:
python project/run_step3_prediction.py \\
    --backbone dinov2_vitb14 \\
    --finetune-checkpoint project/results/step2/dinov2/n4/dinov2_vitb14_n4_best.pth \\
    --split test --img-size 518 \\
    --window-size 5 \\
    --output-dir project/results/step3/dinov2_vitb14_finetuned
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


# ---------------------------------------------------------------------------
# Helpers (shared with run_step1_trainfree.py)
# ---------------------------------------------------------------------------

def _add_sd4match_to_syspath(sd4match_dir: str) -> None:
    sd4match_dir = os.path.abspath(sd4match_dir)
    if sd4match_dir not in sys.path:
        sys.path.insert(0, sd4match_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 3: window soft-argmax prediction on SPair-71k."
    )
    p.add_argument("--backbone", type=str, required=True,
                   help="e.g. dinov2_vitb14 / dinov3_vitb16 / sam_vit_b")

    # dataset
    p.add_argument("--data-root", type=str, default="../")
    p.add_argument("--split", type=str, default="test",
                   choices=["trn", "val", "test"])
    p.add_argument("--category", type=str, default="all")
    p.add_argument("--img-size", type=int, default=518)

    # backbone-specific
    p.add_argument("--dino-repo", type=str, default="external/dinov2")
    p.add_argument("--dinov3-weights", type=str,
                   default="external/dinov3_weights/dinov3_vitb16.pth")
    p.add_argument("--sam-checkpoint", type=str,
                   default="external/segment-anything/sam_vit_b_01ec64.pth")
    p.add_argument("--backbone-input-size", type=int, default=None)
    p.add_argument("--finetune-checkpoint", type=str, default=None,
                   help="Optional Step-2 checkpoint to load before evaluation.")

    # Step 3 specific
    p.add_argument("--window-size", type=int, default=5,
                   help="Side length of the soft-argmax window (odd; default 5).")
    p.add_argument("--wsa-temperature", type=float, default=1.0,
                   help="Softmax temperature used inside the window "
                        "(lower = sharper; default 1.0 = raw similarity scores).")

    # evaluation
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    p.add_argument("--by", type=str, default="image", choices=["image", "point"])
    p.add_argument("--softmax-temp", type=float, default=0.04,
                   help="Temperature for the SD4Match softmax/kernelsoftmax baselines.")

    # runtime
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-pairs", type=int, default=-1)

    # checkpoint / resume
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")

    # IO
    p.add_argument("--sd4match-dir", type=str, default="../SD4Match")
    p.add_argument("--output-dir", type=str, default="results/step3")

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
        kwargs["weights_path"] = args.dinov3_weights
    elif args.backbone.startswith("sam_"):
        kwargs["checkpoint"] = args.sam_checkpoint
    else:
        raise ValueError(f"Unrecognised backbone '{args.backbone}'.")
    backbone = _build(args.backbone, **kwargs)

    if args.finetune_checkpoint is not None:
        from backbones.finetune_wrapper import FinetunableBackbone
        ckpt = FinetunableBackbone.load_checkpoint_into_backbone(
            backbone, args.finetune_checkpoint
        )
        n_layers = ckpt.get("n_unfrozen_layers", "?")
        best_pck = ckpt.get("best_val_pck_01", "?")
        print(
            f"[finetune] Loaded checkpoint: {args.finetune_checkpoint}\n"
            f"           unfrozen_layers={n_layers}  best_val_PCK@0.1={best_pck}"
        )
    return backbone


def build_dataset(args: argparse.Namespace):
    from config.base import get_default_defaults
    from spair_dataset import SafeSPairDataset as SPairDataset

    cfg = get_default_defaults()
    cfg.DATASET.NAME = "spair"
    cfg.DATASET.ROOT = os.path.abspath(args.data_root)
    cfg.DATASET.IMG_SIZE = args.img_size
    cfg.DATASET.MEAN = [0.0, 0.0, 0.0]
    cfg.DATASET.STD = [1.0, 1.0, 1.0]
    cfg.EVALUATOR.ALPHA = list(args.alphas)
    cfg.EVALUATOR.BY = args.by

    dataset = SPairDataset(cfg, split=args.split, category=args.category)
    return cfg, dataset


# ---------------------------------------------------------------------------
# Checkpoint helpers (identical to Step 1)
# ---------------------------------------------------------------------------

def _ckpt_path(out_dir: Path, tag: str) -> Path:
    return out_dir / f"{tag}.ckpt.json"


def save_checkpoint(out_dir: Path, tag: str, n_pairs_seen: int, evaluator) -> None:
    data = {"n_pairs_seen": n_pairs_seen, "evaluator_result": evaluator.result}
    path = _ckpt_path(out_dir, tag)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, default=float)
    tmp.replace(path)


def load_checkpoint(out_dir: Path, tag: str, evaluator) -> int:
    path = _ckpt_path(out_dir, tag)
    if not path.exists():
        return 0
    with open(path) as f:
        data = json.load(f)
    evaluator.result = data["evaluator_result"]
    n = data["n_pairs_seen"]
    print(f"[resume] Loaded checkpoint: {n} pairs already evaluated.")
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _add_sd4match_to_syspath(args.sd4match_dir)

    cfg, dataset = build_dataset(args)

    from window_softargmax import WindowSoftArgmaxEvaluator
    evaluator = WindowSoftArgmaxEvaluator(
        cfg,
        window_size=args.window_size,
        wsa_temperature=args.wsa_temperature,
    )

    backbone = build_backbone(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = (f"{args.backbone}_{args.split}_{args.category}"
           f"_by{args.by}_w{args.window_size}")

    # ---- resume ----
    n_skip = 0
    if args.resume or _ckpt_path(out_dir, tag).exists():
        n_skip = load_checkpoint(out_dir, tag, evaluator)

    total_pairs = len(dataset)
    if n_skip > 0:
        import torch.utils.data as tud
        dataset = tud.Subset(dataset, list(range(n_skip, total_pairs)))
        print(f"[resume] Resuming from pair {n_skip} "
              f"({len(dataset)} pairs remaining) ...")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    n_pairs_seen = n_skip
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        src_img = batch["src_img"]
        trg_img = batch["trg_img"]

        batch["src_featmaps"] = backbone(src_img).float().cpu()
        batch["trg_featmaps"] = backbone(trg_img).float().cpu()

        evaluator.evaluate_feature_map(
            batch,
            softmax_temp=args.softmax_temp,
            gaussian_suppression_sigma=7,
            enable_l2_norm=True,
        )

        n_pairs_seen += src_img.shape[0]

        if i % 50 == 0:
            dt = time.perf_counter() - t0
            new_pairs = n_pairs_seen - n_skip
            ips = new_pairs / max(dt, 1e-6)
            print(f"[{n_pairs_seen}/{total_pairs}] "
                  f"elapsed={dt:.1f}s  pairs/s={ips:.2f}")

        if (args.checkpoint_every > 0
                and n_pairs_seen % args.checkpoint_every < args.batch_size):
            save_checkpoint(out_dir, tag, n_pairs_seen, evaluator)
            print(f"  [ckpt] saved at {n_pairs_seen} pairs")

        if args.max_pairs > 0 and n_pairs_seen >= args.max_pairs:
            break

    dt = time.perf_counter() - t0
    new_pairs = n_pairs_seen - n_skip
    print(f"\nFinished. Evaluated {new_pairs} new pairs "
          f"({n_pairs_seen} total) in {dt:.1f}s "
          f"({new_pairs / max(dt, 1e-6):.2f} pairs/s).\n")

    # ---- final results ----
    txt_path  = out_dir / f"{tag}.txt"
    json_path = out_dir / f"{tag}.json"

    evaluator.print_summarize_result()
    evaluator.save_result(str(txt_path))

    summary = evaluator.summerize_result()
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nSaved per-category report  → {txt_path}")
    print(f"Saved machine-readable JSON → {json_path}")

    if torch.cuda.is_available():
        max_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Peak GPU memory: {max_mem_mb:.0f} MiB")

    ckpt = _ckpt_path(out_dir, tag)
    if ckpt.exists():
        ckpt.unlink()
        print("[ckpt] Checkpoint deleted (run complete).")


if __name__ == "__main__":
    main()
