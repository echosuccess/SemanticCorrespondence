#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-shot environment setup for the semantic-correspondence project.
#
# Usage (from the repository root):
#   bash setup.sh
#
# What it does:
#   1. pulls the SD4Match submodule (dataset loader + PCK evaluator)
#   2. clones the official backbone repos into external/
#   3. downloads the SAM ViT-B checkpoint
#   4. installs the Python dependencies listed in project/requirements.txt
#
# The SPair-71k dataset is NOT downloaded automatically: it is ~1.3 GB and
# hosted behind a manual agreement. See README.md for the download link
# and place the extracted SPair-71k/ folder at the repository root.
# -----------------------------------------------------------------------------

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> 1/4  Init SD4Match submodule"
if [ -f .gitmodules ] && grep -q "SD4Match" .gitmodules; then
    git submodule update --init --recursive
else
    echo "    (no submodule declared yet; skipping)"
fi

echo "==> 2/4  Clone backbone repos into external/"
mkdir -p external
cd external
[ -d dinov2 ]             || git clone https://github.com/facebookresearch/dinov2
[ -d dinov3 ]             || git clone https://github.com/facebookresearch/dinov3
[ -d segment-anything ]   || git clone https://github.com/facebookresearch/segment-anything
cd "$ROOT"

echo "==> 3/4  Download SAM ViT-B checkpoint (~375 MB)"
SAM_CKPT="external/segment-anything/sam_vit_b_01ec64.pth"
if [ ! -f "$SAM_CKPT" ]; then
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
        -O "$SAM_CKPT"
else
    echo "    (already downloaded)"
fi

echo "==> 4/4  Install Python dependencies"
pip install -q -r project/requirements.txt
pip install -q -e external/segment-anything

echo
echo "Setup done."
echo "Next step: place SPair-71k at ./SPair-71k/ (see README.md), then run"
echo "    python project/run_step1_trainfree.py --backbone dinov2_vitb14 \\"
echo "        --dino-repo external/dinov2 --data-root . --split test"
