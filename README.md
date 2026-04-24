# Semantic Correspondence with Visual Foundation Models

Course project for *Advanced Machine Learning* — Politecnico di Torino.

We establish pixel-level correspondences between pairs of semantically
similar images using features from frozen Vision Foundation Models
(DINOv2, DINOv3, SAM) and evaluate on [SPair-71k](https://cvlab.postech.ac.kr/research/SPair-71k/).

The project is organised around the four stages described in the
assignment:

1. **Training-free baseline** — cosine similarity + argmax on frozen
   features.
2. **Light fine-tuning** — unfreeze the last blocks of the backbone
   under keypoint supervision.
3. **Better prediction rule** — replace argmax with *window soft-argmax*
   (Zhang et al., CVPR 2024).
4. **Extension** — e.g. Stable Diffusion features, PF-Pascal / PF-Willow
   transfer, or LoRA / Adapter fine-tuning.

## Repository layout

```
Exam2/
├── project/                 # All our own code lives here
│   ├── backbones/           # DINOv2 / DINOv3 / SAM wrappers
│   ├── run_step1_trainfree.py
│   ├── visualize.py
│   ├── requirements.txt
│   └── README.md            # Detailed usage
├── SD4Match/                # Submodule: dataset loader + PCK evaluator only
├── external/                # (gitignored) backbone repos + SAM weights
├── SPair-71k/               # (gitignored) dataset, downloaded manually
├── setup.sh                 # One-shot environment setup
├── .gitignore
└── README.md                # This file
```

`SD4Match/` is declared as a git submodule so that we use it *as a
library* without copy-pasting third-party code into our repo.
The only two classes we import from it are
`dataset.spair.SPairDataset` and `utils.evaluator.PCKEvaluator`.

## Quick start

```bash
# 1. Clone (remember --recursive for the SD4Match submodule)
git clone --recursive <this-repo-url>
cd Exam2

# 2. Install dependencies + clone backbone repos + download SAM weights
bash setup.sh

# 3. Place SPair-71k at ./SPair-71k/  (see "Dataset" below)

# 4. Run the training-free baseline (Step 1) on the test split
python project/run_step1_trainfree.py \
    --backbone dinov2_vitb14 \
    --dino-repo external/dinov2 \
    --data-root . \
    --split test \
    --img-size 512 --batch-size 4 \
    --output-dir project/results/step1
```

See `project/README.md` for the full list of command-line flags and for
how to reproduce every row of the final PCK table.

## Dataset

Download **SPair-71k** from
<https://cvlab.postech.ac.kr/research/SPair-71k/> and extract it so that
the layout is:

```
Exam2/SPair-71k/
    JPEGImages/...
    ImageAnnotation/...
    PairAnnotation/{trn,val,test}/*.json
    Layout/large/{trn,val,test}.txt
    Segmentation/...
```

The dataset is **not** committed to this repository (1.3 GB).

## On Google Colab

```python
# Cell 1
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/path/to/Exam2
!bash setup.sh
```

Then call the scripts exactly as in the Quick start section above.

## Attribution

* **SD4Match** (submodule) — Li et al., *Learning to Prompt Stable
  Diffusion Model for Semantic Matching*, 2023.
  <https://github.com/ActiveVisionLab/SD4Match>
  We only use its `SPairDataset` and `PCKEvaluator` classes.
* **DINOv2** — Oquab et al., CVPR 2023.
  <https://github.com/facebookresearch/dinov2>
* **DINOv3** — Simeoni et al., 2025.
  <https://github.com/facebookresearch/dinov3>
* **Segment Anything** — Kirillov et al., ICCV 2023.
  <https://github.com/facebookresearch/segment-anything>
