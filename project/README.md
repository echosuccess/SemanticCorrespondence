# Semantic Correspondence with Visual Foundation Models

This directory contains our implementation of the project
*Semantic Correspondence with Visual Foundation Models*.

The layout keeps the upstream [SD4Match](https://github.com/ActiveVisionLab/SD4Match)
repo untouched (we only import the `dataset` and `utils` packages from it for
the SPair-71k loader and the PCK evaluator). All the project-specific code
lives in this folder.

```
Exam2/
├── SD4Match/                      # upstream repo - used as a library
├── SPair-71k/                     # or anywhere else; point --data-root at its parent
├── external/                      # local clones of DINOv2/DINOv3/SAM (see below)
└── project/                       # <-- this folder
    ├── backbones/
    │   ├── base.py
    │   ├── dinov2_backbone.py
    │   ├── dinov3_backbone.py
    │   └── sam_backbone.py
    ├── run_step1_trainfree.py     # Step 1 entry point
    ├── visualize.py               # qualitative results
    ├── requirements.txt
    └── README.md
```

## 1. Environment (Colab or local)

```bash
pip install -r project/requirements.txt
# SAM only - skip if you are not running the SAM backbone yet
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 2. Clone the backbone repos (official, NOT HuggingFace)

```bash
cd external
git clone https://github.com/facebookresearch/dinov2
git clone https://github.com/facebookresearch/dinov3
git clone https://github.com/facebookresearch/segment-anything
# SAM weights (pick the size you want, ViT-B is enough to start):
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O segment-anything/sam_vit_b_01ec64.pth
```

## 3. Dataset

We use **SPair-71k** (https://cvlab.postech.ac.kr/research/SPair-71k/).
Unpack it so that the layout is:

```
<data-root>/SPair-71k/
    JPEGImages/...
    ImageAnnotation/...
    PairAnnotation/{trn,val,test}/*.json
    Layout/large/{trn,val,test}.txt
    Segmentation/...
```

On Colab, upload the folder to Google Drive once and pass
`--data-root /content/drive/MyDrive/AML_Project/data` to the scripts.

Per the project instructions we **only report final numbers on the `test`
split**; the `trn`/`val` splits will be used for Step 2 (fine-tuning).

## 4. Running Step 1 - training-free baseline

From the repository root (the `Exam2/` folder):

```bash
# DINOv2 ViT-B, argmax matching, PCK@{0.05,0.1,0.15} per image
python project/run_step1_trainfree.py \
    --backbone dinov2_vitb14 \
    --dino-repo external/dinov2 \
    --data-root . \
    --split test \
    --img-size 512 \
    --batch-size 4 \
    --output-dir project/results/step1
```

Change `--backbone` to `dinov2_vits14` / `dinov2_vitl14` / `dinov3_vitb16` /
`sam_vit_b` etc. to reproduce the other rows of the report.

Each run writes two files to `--output-dir`:

* `<backbone>_<split>_<category>_by<image|point>.txt` - human-readable
  per-category PCK table for every matcher (nn = argmax baseline),
* `<backbone>_<split>_<category>_by<image|point>.json` - same numbers in a
  machine-readable form, so we can aggregate them later in the report.

To report PCK **per keypoint** (as the assignment also asks), repeat the run
with `--by point`.

### Quick sanity check on a handful of pairs

```bash
python project/run_step1_trainfree.py \
    --backbone dinov2_vits14 \
    --dino-repo external/dinov2 \
    --data-root . \
    --split val \
    --max-pairs 50
```

## 5. Qualitative results

```bash
python project/visualize.py \
    --backbone dinov2_vitb14 \
    --dino-repo external/dinov2 \
    --data-root . \
    --split test \
    --num-pairs 10 \
    --output-dir project/results/step1/vis_dinov2_vitb14
```

Each output PNG shows the source with its annotated keypoints and the
target with both the ground-truth keypoints (hollow circles) and the
model's predictions (filled dots, same colour per keypoint).

## 6. What the code does under the hood

The math of the training-free baseline is deliberately simple; all the
interesting choices live inside the backbones:

1. `backbones/*_backbone.py` - resize to the backbone's preferred size,
   apply the right normalisation, run the frozen model and return a
   `(B, C, h, w)` dense feature map.
2. `run_step1_trainfree.py` - build the SPair-71k loader (from SD4Match),
   plug the features into SD4Match's `PCKEvaluator` which already
   implements:
      * `nn`            - **argmax** on cosine similarity (this is Step 1),
      * `bilinear`      - bilinear interpolation of the four nearest cells,
      * `softmax`       - soft-argmax over the full map,
      * `kernelsoftmax` - soft-argmax restricted by a Gaussian kernel
                          around the argmax peak (used again in Step 3).
3. `visualize.py` - runs the same matcher on a handful of pairs and
   plots the result for the report.

Steps 2 (light fine-tuning), 3 (window soft-argmax) and 4 (extension)
will reuse the exact same `backbones/` wrappers.
