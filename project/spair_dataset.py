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
import pickle
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

        # ---- Annotation loading with persistent + resumable cache ----
        # Final cache : {SPair-71k root}/ann_cache_{split}_{category}.pkl
        # Partial cache: same name + ".partial" suffix – saved every 2000 items
        #                so an interrupted run can resume instead of restarting.
        _SAVE_EVERY = 500
        spair_root  = os.path.dirname(os.path.dirname(self.ann_path))
        cache_path  = os.path.join(spair_root, f"ann_cache_{split}_{category}.pkl")
        partial_path = cache_path + ".partial"

        def _pack(lists):
            return {
                "src_kps":  lists[0], "trg_kps":  lists[1],
                "src_bbox": lists[2], "trg_bbox": lists[3],
                "cls_ids":  lists[4], "vpvar":    lists[5],
                "scvar":    lists[6], "trncn":    lists[7],
                "occln":    lists[8],
            }

        def _unpack(cache):
            return (cache["src_kps"], cache["trg_kps"],
                    cache["src_bbox"], cache["trg_bbox"],
                    cache["cls_ids"], cache["vpvar"],
                    cache["scvar"], cache["trncn"], cache["occln"])

        if os.path.exists(cache_path):
            # Complete cache → instant load
            print(f"[cache] Loading annotation cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            (self.src_kps, self.trg_kps, self.src_bbox, self.trg_bbox,
             self.cls_ids, self.vpvar, self.scvar, self.trncn,
             self.occln) = _unpack(cache)
        else:
            # Build the list of annotation file paths only when cache is missing.
            # Some users extract SPair-71k on Windows where ':' in filenames is
            # silently replaced with '_'. We probe both so the code works either way.
            anntn_files: List[str] = []
            for data_name in self.train_data:
                candidates = [
                    os.path.join(self.ann_path, f"{data_name}.json"),
                    os.path.join(self.ann_path,
                                 f"{data_name.replace(':', '_')}.json"),
                ]
                found = next((p for p in candidates if os.path.exists(p)), None)
                if found is None:
                    raise FileNotFoundError(
                        f"Annotation file for pair '{data_name}' not found "
                        f"under {self.ann_path} (tried both ':' and '_' "
                        f"as the category separator). "
                        f"Check that the dataset layout is intact."
                    )
                anntn_files.append(found)

            # Partial cache → resume from last checkpoint
            start_idx = 0
            self.src_kps, self.trg_kps = [], []
            self.src_bbox, self.trg_bbox = [], []
            self.cls_ids = []
            self.vpvar, self.scvar, self.trncn, self.occln = [], [], [], []

            if os.path.exists(partial_path):
                print(f"[cache] Resuming annotation loading from: {partial_path}")
                with open(partial_path, "rb") as f:
                    partial = pickle.load(f)
                (self.src_kps, self.trg_kps, self.src_bbox, self.trg_bbox,
                 self.cls_ids, self.vpvar, self.scvar, self.trncn,
                 self.occln) = _unpack(partial)
                start_idx = partial["n_loaded"]
                print(f"[cache] Resuming from item {start_idx} / {len(anntn_files)}")

            print(f"Reading SPair-71k information ({split} / {category}) ...")
            for idx, anntn_file in enumerate(tqdm(anntn_files[start_idx:],
                                                  initial=start_idx,
                                                  total=len(anntn_files))):
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

                # Save partial cache every _SAVE_EVERY items
                n_done = start_idx + idx + 1
                if n_done % _SAVE_EVERY == 0:
                    partial_data = _pack([
                        self.src_kps, self.trg_kps, self.src_bbox,
                        self.trg_bbox, self.cls_ids, self.vpvar,
                        self.scvar, self.trncn, self.occln,
                    ])
                    partial_data["n_loaded"] = n_done
                    with open(partial_path, "wb") as f:
                        pickle.dump(partial_data, f)

            # All done → save final cache and remove partial
            final_cache = _pack([
                self.src_kps, self.trg_kps, self.src_bbox,
                self.trg_bbox, self.cls_ids, self.vpvar,
                self.scvar, self.trncn, self.occln,
            ])
            with open(cache_path, "wb") as f:
                pickle.dump(final_cache, f)
            if os.path.exists(partial_path):
                os.remove(partial_path)
            print(f"[cache] Saved annotation cache → {cache_path}")

        self.src_identifiers = [
            f"{self.cls[ids]}-{name[:-4]}"
            for ids, name in zip(self.cls_ids, self.src_imnames)
        ]
        self.trg_identifiers = [
            f"{self.cls[ids]}-{name[:-4]}"
            for ids, name in zip(self.cls_ids, self.trg_imnames)
        ]
