"""CRLF-tolerant wrapper around ``SD4Match.dataset.spair.SPairDataset``.

The upstream ``SPairDataset`` parses ``Layout/large/{split}.txt`` with::

    self.train_data = open(self.spt_path).read().split('\\n')

which keeps a trailing ``\\r`` on every line if the file was uploaded /
unzipped on Windows. The first ``glob`` look-up then fails because the
filename ``...{name}\\r.json`` does not exist on disk.

Rather than rewrite the dataset on disk (which would touch the original
SPair-71k files we got from the authors), we sub-class ``SPairDataset``
and re-do the layout parsing in a CRLF-safe way *after* the parent
constructor runs. The trick is that the parent's broken parsing happens
before any expensive I/O is needed, so we let it fail-fast on a known
exception, then fix the state ourselves.

The implementation simply replicates the parent constructor with the
parsing line replaced. Everything else (annotation reading, transforms,
keypoint coordinates, segmentation masks ...) is untouched and still
comes from the upstream class.
"""

from __future__ import annotations

import glob
import json
import os
from typing import List

import torch
from tqdm import tqdm

from dataset.spair import SPairDataset
from dataset.dataset import CorrespondenceDataset


_VALID_CATEGORIES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "train", "tvmonitor",
)


def _read_layout_lines(path: str) -> List[str]:
    """Return non-empty layout lines with any CR/LF/whitespace stripped."""
    with open(path, "r") as f:
        raw = f.read()
    # ``splitlines()`` already handles \r, \n and \r\n uniformly.
    return [line.strip() for line in raw.splitlines() if line.strip()]


class SafeSPairDataset(SPairDataset):
    """Drop-in replacement for ``SPairDataset`` that tolerates CRLF layouts.

    The signature, returned items and all behaviour are identical to the
    upstream class; the only difference is how ``Layout/large/*.txt`` is
    parsed.
    """

    def __init__(self, cfg, split: str, category: str = "all") -> None:
        # Skip SPairDataset.__init__ on purpose (it will choke on CRLF) and
        # call its grandparent CorrespondenceDataset.__init__ directly to set
        # up paths, transforms, image size, etc.
        CorrespondenceDataset.__init__(self, cfg, split=split)

        # ---- CRLF-safe layout parsing ----
        self.train_data = _read_layout_lines(self.spt_path)

        if category != "all":
            if category not in _VALID_CATEGORIES:
                raise ValueError(
                    f"{category!r} is not a valid SPair-71k category. "
                    f"Expected 'all' or one of {_VALID_CATEGORIES}."
                )
            self.train_data = [p for p in self.train_data if category in p]

        # ---- the rest is a verbatim copy of SPairDataset.__init__'s tail ----
        self.src_imnames = list(map(
            lambda x: x.split("-")[1] + ".jpg", self.train_data))
        self.trg_imnames = list(map(
            lambda x: x.split("-")[2].split(":")[0] + ".jpg", self.train_data))
        self.seg_path = os.path.abspath(
            os.path.join(self.img_path, os.pardir, "Segmentation"))
        self.cls = sorted(os.listdir(self.img_path))

        anntn_files: List[str] = []
        for data_name in self.train_data:
            matches = glob.glob(f"{self.ann_path}/{data_name}.json")
            if not matches:
                raise FileNotFoundError(
                    f"Annotation file for pair '{data_name}' not found under "
                    f"{self.ann_path}. Check that the dataset layout is intact."
                )
            anntn_files.append(matches[0])

        self.src_kps, self.trg_kps = [], []
        self.src_bbox, self.trg_bbox = [], []
        self.cls_ids = []
        self.vpvar, self.scvar, self.trncn, self.occln = [], [], [], []

        print(f"Reading SPair-71k information ({split} / {category}) ...")
        for anntn_file in tqdm(anntn_files):
            with open(anntn_file) as f:
                anntn = json.load(f)
            self.src_kps.append(torch.tensor(anntn["src_kps"]).t().float())
            self.trg_kps.append(torch.tensor(anntn["trg_kps"]).t().float())
            self.src_bbox.append(torch.tensor(anntn["src_bndbox"]).float())
            self.trg_bbox.append(torch.tensor(anntn["trg_bndbox"]).float())
            self.cls_ids.append(self.cls.index(anntn["category"]))

            self.vpvar.append(torch.tensor(anntn["viewpoint_variation"]))
            self.scvar.append(torch.tensor(anntn["scale_variation"]))
            self.trncn.append(torch.tensor(anntn["truncation"]))
            self.occln.append(torch.tensor(anntn["occlusion"]))

        self.src_identifiers = [
            f"{self.cls[ids]}-{name[:-4]}"
            for ids, name in zip(self.cls_ids, self.src_imnames)
        ]
        self.trg_identifiers = [
            f"{self.cls[ids]}-{name[:-4]}"
            for ids, name in zip(self.cls_ids, self.trg_imnames)
        ]
