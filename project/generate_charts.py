"""Generate result charts and tables as PNG images for the report."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path("project/results/charts")
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLORS = ["#4472C4", "#ED7D31", "#A9D18E", "#FFC000", "#5B9BD5"]


def save(fig, name: str):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path}")


# ── 1. Summary: PCK@0.1 across all steps ──────────────────────────────────
def chart_summary():
    steps = ["Frozen\n(Step 1)", "Full FT\n(Step 2)", "Frozen+WSA\n(Step 3)",
             "FT+WSA\n(S2+3)", "LoRA r=4\n(Step 4)"]
    data = {
        "DINOv2 ViT-B/14": [52.4, 73.4, 54.7, 75.9, 60.7],
        "DINOv3 ViT-B/16": [48.3, 65.8, 48.4, 67.9, 52.4],
        "SAM ViT-B":        [21.6, 22.7, 21.4, 22.8, 23.0],
    }
    x = np.arange(len(steps))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (label, vals) in enumerate(data.items()):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label,
                      color=COLORS[i], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=10)
    ax.set_ylabel("PCK@0.1 (%)")
    ax.set_ylim(0, 88)
    ax.set_title("PCK@0.1 Across All Steps — SPair-71k Test Set", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save(fig, "01_summary_all_steps")


# ── 2. Step 1 – backbone size ablation (DINOv2) ────────────────────────────
def chart_backbone_size():
    models = ["DINOv2\nViT-S/14", "DINOv2\nViT-B/14", "DINOv2\nViT-L/14"]
    p05 = [32.8, 34.5, 35.5]
    p10 = [50.1, 52.4, 53.3]
    p20 = [66.4, 68.2, 69.2]
    x = np.arange(3)
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (vals, label) in enumerate([(p05, "PCK@0.05"), (p10, "PCK@0.1"), (p20, "PCK@0.2")]):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label,
                      color=COLORS[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("PCK (%)")
    ax.set_ylim(0, 80)
    ax.set_title("Step 1 · DINOv2 Backbone Size Ablation (nn, Frozen)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "02_step1_backbone_size")


# ── 3. Step 1 – all backbone comparison (PCK@0.1, nn) ─────────────────────
def chart_backbone_compare():
    models   = ["DINOv2\nViT-S", "DINOv2\nViT-B", "DINOv2\nViT-L", "DINOv3\nViT-B", "SAM\nViT-B"]
    nn_p10   = [50.1, 52.4, 53.3, 48.3, 21.6]
    ksm_p10  = [51.7, 53.5, 54.4, 49.4, 21.8]
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w / 2, nn_p10,  w, label="nn (argmax)",       color=COLORS[0], edgecolor="white")
    b2 = ax.bar(x + w / 2, ksm_p10, w, label="ksm (kernel softargmax)", color=COLORS[1], edgecolor="white")
    for bar, v in list(zip(b1, nn_p10)) + list(zip(b2, ksm_p10)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("PCK@0.1 (%)")
    ax.set_ylim(0, 65)
    ax.set_title("Step 1 · Training-Free Baseline — All Backbones (PCK@0.1)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "03_step1_backbone_compare")


# ── 4. Step 1 – table as figure ────────────────────────────────────────────
def table_step1():
    rows = [
        ["DINOv2 ViT-S/14", "nn",  "32.8", "50.1", "66.4"],
        ["DINOv2 ViT-B/14", "nn",  "34.5", "52.4", "68.2"],
        ["DINOv2 ViT-B/14", "bilinear", "31.3", "48.9", "67.2"],
        ["DINOv2 ViT-B/14", "softmax",  "24.9", "43.4", "65.8"],
        ["DINOv2 ViT-B/14", "ksm", "36.6", "53.5", "68.9"],
        ["DINOv2 ViT-L/14", "nn",  "35.5", "53.3", "69.2"],
        ["DINOv2 ViT-L/14", "ksm", "37.4", "54.4", "69.8"],
        ["DINOv3 ViT-B/16", "nn",  "31.4", "48.3", "63.1"],
        ["DINOv3 ViT-B/16", "ksm", "30.8", "49.4", "64.1"],
        ["SAM ViT-B",        "nn",  "13.5", "21.6", "34.8"],
        ["SAM ViT-B",        "ksm", "13.7", "21.8", "34.9"],
    ]
    cols = ["Backbone", "Matching", "PCK@0.05", "PCK@0.1", "PCK@0.2"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)
    # Header style
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Zebra stripes
    for i in range(1, len(rows) + 1):
        color = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(color)
    ax.set_title("Step 1 · Training-Free Baseline Results (PCK %, SPair-71k Test)",
                 fontweight="bold", pad=16)
    fig.tight_layout()
    save(fig, "04_step1_table")


# ── 5. Step 2 – N layers ablation ─────────────────────────────────────────
def chart_n_ablation():
    n_labels = ["N=0\n(frozen)", "N=1", "N=2", "N=4"]
    data = {
        "DINOv2 ViT-B": [52.4, 50.2, 58.6, 61.5],
        "DINOv3 ViT-B": [48.3, 43.5, 51.1, 56.3],
        "SAM ViT-B":    [21.6, 24.2, 25.5, 26.3],
    }
    x = np.arange(4)
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, vals) in enumerate(data.items()):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label,
                      color=COLORS[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels)
    ax.set_ylabel("Validation PCK@0.1 (%)")
    ax.set_ylim(0, 72)
    ax.set_title("Step 2 · Unfrozen Layers Ablation (Validation PCK@0.1)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "05_step2_n_ablation")


# ── 6. Step 2 – FT vs frozen comparison (test) ────────────────────────────
def chart_ft_compare():
    backbones = ["DINOv2 ViT-B", "DINOv3 ViT-B", "SAM ViT-B"]
    frozen = [52.4, 48.3, 21.6]
    ft4    = [73.4, 65.8, 22.7]
    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b1 = ax.bar(x - w / 2, frozen, w, label="Frozen (Step 1)", color=COLORS[0], edgecolor="white")
    b2 = ax.bar(x + w / 2, ft4,   w, label="Fine-tuned N=4 (Step 2)", color=COLORS[1], edgecolor="white")
    for bar, v in list(zip(b1, frozen)) + list(zip(b2, ft4)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    # delta arrows
    for i in range(3):
        delta = ft4[i] - frozen[i]
        if delta > 1:
            ax.annotate(f"+{delta:.1f}pp",
                        xy=(x[i] + w / 2, ft4[i] + 1.5),
                        ha="center", va="bottom", fontsize=8.5,
                        color="#C00000", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(backbones)
    ax.set_ylabel("PCK@0.1 (%)")
    ax.set_ylim(0, 88)
    ax.set_title("Step 2 · Fine-Tuning vs. Frozen Baseline (Test PCK@0.1)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "06_step2_ft_vs_frozen")


# ── 7. Step 3 – WSA ablation ──────────────────────────────────────────────
def chart_wsa_ablation():
    labels = ["Baseline\n(nn)", "3×3\nτ=1.0", "5×5\nτ=1.0", "7×7\nτ=1.0", "7×7\nτ=0.1"]
    p05 = [34.5,  35.4, 35.7, 35.3, 35.6]
    p10 = [52.4,  52.5, 52.8, 52.8, 54.7]
    x = np.arange(5)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w / 2, p05, w, label="PCK@0.05", color=COLORS[0], edgecolor="white")
    b2 = ax.bar(x + w / 2, p10, w, label="PCK@0.1",  color=COLORS[1], edgecolor="white")
    for bar, v in list(zip(b1, p05)) + list(zip(b2, p10)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PCK (%)")
    ax.set_ylim(0, 62)
    ax.set_title("Step 3 · Window Soft-Argmax Ablation (DINOv2 ViT-B, Frozen)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "07_step3_wsa_ablation")


# ── 8. Step 3 – WSA full results ──────────────────────────────────────────
def chart_wsa_full():
    categories = ["DINOv2 ViT-B\nFrozen", "DINOv2 ViT-B\nFT N=4",
                  "DINOv3 ViT-B\nFrozen", "DINOv3 ViT-B\nFT N=4", "SAM ViT-B\nFrozen"]
    nn_p10  = [52.4, 73.4, 48.3, 65.8, 21.6]
    wsa_p10 = [54.7, 75.9, 48.4, 67.9, 21.4]
    x = np.arange(5)
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - w / 2, nn_p10,  w, label="nn (argmax)",        color=COLORS[0], edgecolor="white")
    b2 = ax.bar(x + w / 2, wsa_p10, w, label="window soft-argmax", color=COLORS[2], edgecolor="white")
    for bar, v in list(zip(b1, nn_p10)) + list(zip(b2, wsa_p10)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_ylabel("PCK@0.1 (%)")
    ax.set_ylim(0, 88)
    ax.set_title("Step 3 · Window Soft-Argmax vs. nn (Test PCK@0.1, w=7, τ=0.1)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "08_step3_wsa_full")


# ── 9. Step 4 – LoRA rank ablation ────────────────────────────────────────
def chart_lora_rank():
    ranks = ["r=4", "r=8", "r=16"]
    data = {
        "DINOv2 ViT-B": [52.8, 49.7, 48.0],
        "DINOv3 ViT-B": [51.4, 45.5, 43.3],
        "SAM ViT-B":    [22.4, 21.7, 21.0],
    }
    x = np.arange(3)
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, vals) in enumerate(data.items()):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label,
                      color=COLORS[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.set_ylabel("Validation PCK@0.1 (%)")
    ax.set_ylim(0, 65)
    ax.set_title("Step 4 · LoRA Rank Ablation (Validation PCK@0.1)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "09_step4_lora_rank")


# ── 10. Step 4 – LoRA vs Frozen vs Full FT ────────────────────────────────
def chart_lora_compare():
    backbones = ["DINOv2 ViT-B", "DINOv3 ViT-B", "SAM ViT-B"]
    frozen = [52.4, 48.3, 21.6]
    lora   = [60.7, 52.4, 23.0]
    full   = [73.4, 65.8, 22.7]
    x = np.arange(3)
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (vals, label, col) in enumerate([
        (frozen, "Frozen (Step 1)",      COLORS[0]),
        (lora,   "LoRA r=4 (Step 4)",    COLORS[1]),
        (full,   "Full FT N=4 (Step 2)", COLORS[2]),
    ]):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=col, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(backbones)
    ax.set_ylabel("PCK@0.1 (%)")
    ax.set_ylim(0, 88)
    ax.set_title("Step 4 · LoRA vs. Frozen vs. Full Fine-Tuning (Test PCK@0.1)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "10_step4_lora_compare")


# ── 11. Step 2 – full table ────────────────────────────────────────────────
def table_step2():
    rows = [
        ["DINOv2 ViT-B/14", "Frozen",   "34.5", "52.4", "68.2"],
        ["DINOv2 ViT-B/14", "FT N=1 (val)", "—",  "50.2*","—"],
        ["DINOv2 ViT-B/14", "FT N=2 (val)", "—",  "58.6*","—"],
        ["DINOv2 ViT-B/14", "FT N=4",   "56.9", "73.4", "83.5"],
        ["DINOv3 ViT-B/16", "Frozen",   "31.4", "48.3", "63.1"],
        ["DINOv3 ViT-B/16", "FT N=4",   "47.5", "65.8", "77.9"],
        ["SAM ViT-B",        "Frozen",   "13.5", "21.6", "34.8"],
        ["SAM ViT-B",        "FT best",  "14.3", "22.7", "36.0"],
    ]
    cols = ["Backbone", "Setting", "PCK@0.05", "PCK@0.1", "PCK@0.2"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#ED7D31")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        color = "#FFF2E6" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(color)
    ax.set_title("Step 2 · Light Fine-Tuning Results (PCK %, SPair-71k Test; *=val only)",
                 fontweight="bold", pad=14)
    fig.tight_layout()
    save(fig, "11_step2_table")


# ── 12. Step 4 – full table ────────────────────────────────────────────────
def table_step4():
    rows = [
        ["DINOv2 ViT-B/14", "Frozen",       "34.5", "52.4", "68.2"],
        ["DINOv2 ViT-B/14", "LoRA r=4",     "44.1", "60.7", "73.7"],
        ["DINOv2 ViT-B/14", "Full FT N=4",  "56.9", "73.4", "83.5"],
        ["DINOv3 ViT-B/16", "Frozen",       "31.4", "48.3", "63.1"],
        ["DINOv3 ViT-B/16", "LoRA r=4",     "35.0", "52.4", "66.4"],
        ["DINOv3 ViT-B/16", "Full FT N=4",  "47.5", "65.8", "77.9"],
        ["SAM ViT-B",        "Frozen",       "13.5", "21.6", "34.8"],
        ["SAM ViT-B",        "LoRA r=4",     "14.4", "23.0", "36.7"],
        ["SAM ViT-B",        "Full FT best", "14.3", "22.7", "36.0"],
    ]
    cols = ["Backbone", "Setting", "PCK@0.05", "PCK@0.1", "PCK@0.2"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#5B9BD5")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        color = "#EBF3FB" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(color)
    ax.set_title("Step 4 · LoRA Fine-Tuning Results (PCK %, SPair-71k Test)",
                 fontweight="bold", pad=14)
    fig.tight_layout()
    save(fig, "12_step4_table")


if __name__ == "__main__":
    print(f"Saving charts to {OUT.resolve()}\n")
    chart_summary()
    chart_backbone_size()
    chart_backbone_compare()
    table_step1()
    chart_n_ablation()
    chart_ft_compare()
    chart_wsa_ablation()
    chart_wsa_full()
    chart_lora_rank()
    chart_lora_compare()
    table_step2()
    table_step4()
    print(f"\nDone! {len(list(OUT.glob('*.png')))} images saved under {OUT.resolve()}")
