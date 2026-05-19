"""Window Soft-Argmax prediction refinement (Step 3).

Standard argmax snaps predictions to discrete patch centres; window
soft-argmax instead applies a local softmax inside a small window
around the argmax peak and returns the resulting expected coordinate
value.  This gives sub-pixel refinement and makes predictions more
robust to noisy similarity maps.

Reference: Zhang et al., "telling left from right" / GeoAware-SC
           https://github.com/Junyi42/geoaware-sc
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core matching function
# ---------------------------------------------------------------------------

def window_softargmax_get_matches(
    src_featmaps: torch.Tensor,
    trg_featmaps: torch.Tensor,
    query: torch.Tensor,
    window_size: int = 5,
    temperature: float = 1.0,
    l2_norm: bool = True,
) -> torch.Tensor:
    """Window soft-argmax matching.

    For each query point, computes the cosine-similarity map between
    the query feature and all target patches, finds the argmax peak,
    then applies softmax *only within a window* around that peak and
    returns the resulting expected coordinate — a sub-pixel prediction.

    Args:
        src_featmaps : (B, C, h1, w1) source feature maps.
        trg_featmaps : (B, C, h2, w2) target feature maps.
        query        : (B, Nq, 2) source keypoints in **feature-map scale**
                       (x, y) — same convention as SD4Match matching.py.
        window_size  : side length of the local window (made odd if even).
        temperature  : softmax temperature; lower → sharper peak.
        l2_norm      : L2-normalise features before similarity (recommended).

    Returns:
        matches : (B, Nq, 2) predicted target coords in feature-map scale.
    """
    # Import from SD4Match — caller must have added SD4Match to sys.path.
    from utils.matching import extract_feature, L2normalization

    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2

    B, C, _h1, _w1 = src_featmaps.shape
    _, _, h2, w2 = trg_featmaps.shape
    Nq = query.shape[1]
    device = src_featmaps.device

    # ---- feature extraction ------------------------------------------------
    feat_q = extract_feature(src_featmaps, query)       # (B, Nq, C)
    if l2_norm:
        feat_q = L2normalization(feat_q, dim=-1)
        trg_norm = L2normalization(trg_featmaps, dim=1)
    else:
        trg_norm = trg_featmaps

    # ---- similarity maps: (B, Nq, h2, w2) ----------------------------------
    feat_flat = trg_norm.view(B, C, -1)                 # (B, C, h2*w2)
    scores = torch.bmm(
        feat_q.view(B, Nq, C),
        feat_flat,
    ).view(B, Nq, h2, w2)

    # ---- window soft-argmax (per keypoint) ----------------------------------
    matches = torch.zeros(B, Nq, 2, device=device)

    for b in range(B):
        for q in range(Nq):
            sim = scores[b, q]                          # (h2, w2)

            # Peak
            peak_flat = int(sim.argmax())
            py = peak_flat // w2
            px = peak_flat % w2

            # Window bounds (clipped to map edges)
            y0, y1 = max(py - half, 0), min(py + half + 1, h2)
            x0, x1 = max(px - half, 0), min(px + half + 1, w2)

            # Softmax weights inside window
            win = sim[y0:y1, x0:x1] / temperature       # (wH, wW)
            weights = F.softmax(win.flatten(), dim=0).view(win.shape)

            # Absolute coordinates inside the window
            ys = torch.arange(y0, y1, dtype=torch.float32, device=device)
            xs = torch.arange(x0, x1, dtype=torch.float32, device=device)

            # Expected coordinate (weighted mean)
            matches[b, q, 1] = (weights.sum(dim=1) * ys).sum()   # y
            matches[b, q, 0] = (weights.sum(dim=0) * xs).sum()   # x

    return matches


# ---------------------------------------------------------------------------
# Extended evaluator
# ---------------------------------------------------------------------------

class WindowSoftArgmaxEvaluator:
    """Drop-in replacement for SD4Match's PCKEvaluator.

    Runs all four original SD4Match methods (nn / bilinear / softmax /
    kernelsoftmax) **plus** window_softargmax so that every result table
    is directly comparable.
    """

    METHOD_OPTIONS = (
        "nn", "bilinear", "softmax", "kernelsoftmax", "window_softargmax",
    )

    def __init__(self, cfg, window_size: int = 5, wsa_temperature: float = 1.0) -> None:
        self.alpha = list(cfg.EVALUATOR.ALPHA)
        self.by = cfg.EVALUATOR.BY
        self.window_size = window_size
        self.wsa_temperature = wsa_temperature

        self.result: dict = {}
        for method in self.METHOD_OPTIONS:
            for alpha in self.alpha:
                self.result[f"{method}_pck{alpha}"] = {"all": []}

    # ------------------------------------------------------------------
    def evaluate_feature_map(
        self,
        batch: dict,
        softmax_temp: float = 0.04,
        gaussian_suppression_sigma: int = 7,
        enable_l2_norm: bool = True,
    ) -> None:
        from utils.matching import (
            nn_get_matches,
            bilinear_get_matches,
            softargmax_get_matches,
            kernel_softargmax_get_matches,
        )
        from utils.geometry import scaling_coordinates

        src_img = batch["src_img"]
        trg_img = batch["trg_img"]
        src_fm = batch["src_featmaps"].clone()
        trg_fm = batch["trg_featmaps"].clone()
        src_kps = batch["src_kps"].clone()
        trg_kps = batch["trg_kps"].clone()
        n_pts = batch["n_pts"].clone()
        categories = batch["category"]
        pckthres = batch["pckthres"].clone()

        H1, W1 = src_img.shape[2:]
        H2, W2 = trg_img.shape[2:]
        h1, w1 = src_fm.shape[2:]
        h2, w2 = trg_fm.shape[2:]

        # Scale source keypoints to feature-map scale
        src_kps_f = scaling_coordinates(src_kps, (H1, W1), (h1, w1))

        # All five matching methods
        matches = {
            "nn":              nn_get_matches(src_fm, trg_fm, src_kps_f, l2_norm=enable_l2_norm),
            "bilinear":        bilinear_get_matches(src_fm, trg_fm, src_kps_f, l2_norm=enable_l2_norm),
            "softmax":         softargmax_get_matches(src_fm, trg_fm, src_kps_f, softmax_temp, l2_norm=enable_l2_norm),
            "kernelsoftmax":   kernel_softargmax_get_matches(src_fm, trg_fm, src_kps_f, softmax_temp, gaussian_suppression_sigma, l2_norm=enable_l2_norm),
            "window_softargmax": window_softargmax_get_matches(
                src_fm, trg_fm, src_kps_f,
                window_size=self.window_size,
                temperature=self.wsa_temperature,
                l2_norm=enable_l2_norm,
            ),
        }

        # Scale all matches back to image scale, then compute PCK
        for method, m in matches.items():
            m_img = scaling_coordinates(m, (h2, w2), (H2, W2))
            self._calculate_pck(trg_kps, m_img, n_pts, categories, pckthres, method)

    # ------------------------------------------------------------------
    def _calculate_pck(self, trg_kps, matches, n_pts, categories, pckthres, method):
        B = trg_kps.shape[0]
        for b in range(B):
            npt = int(n_pts[b].item())
            thres = float(pckthres[b].item())
            cat = categories[b]
            tkps = trg_kps[b, :npt]
            mats = matches[b, :npt]
            diff = torch.norm(tkps - mats, dim=-1)   # (npt,)

            for alpha in self.alpha:
                key = f"{method}_pck{alpha}"
                if cat not in self.result[key]:
                    self.result[key][cat] = []
                if self.by == "image":
                    pck = float((diff <= alpha * thres).float().mean())
                    self.result[key][cat].append(pck)
                    self.result[key]["all"].append(pck)
                else:
                    pck_list = (diff <= alpha * thres).float().tolist()
                    self.result[key][cat].extend(pck_list)
                    self.result[key]["all"].extend(pck_list)

    # ------------------------------------------------------------------
    def summerize_result(self) -> dict:          # keep SD4Match typo
        import numpy as np
        out = {}
        for method in self.METHOD_OPTIONS:
            for alpha in self.alpha:
                key = f"{method}_pck{alpha}"
                out[key] = {k: float(np.mean(v)) for k, v in self.result[key].items()}
        return out

    def print_summarize_result(self) -> None:
        result = self.summerize_result()
        print(" " * 20 + "".join(f"{a:<10}" for a in self.alpha))
        for method in self.METHOD_OPTIONS:
            pcks = [f"{result[f'{method}_pck{a}']['all']:.2%}" for a in self.alpha]
            marker = " ◄" if method == "window_softargmax" else ""
            print(f"{method:<20}" + "".join(f"{p:<10}" for p in pcks) + marker)

    def save_result(self, save_file: str) -> None:
        import numpy as np
        result = self.summerize_result()
        lines = []
        for method in self.METHOD_OPTIONS:
            lines.append(f"{method}:")
            cats = sorted(k for k in result[f"{method}_pck{self.alpha[0]}"] if k != "all")
            header = " " * 12 + "".join(f"{c:<12}" for c in cats) + "all\n"
            lines.append(header)
            for alpha in self.alpha:
                key = f"{method}_pck{alpha}"
                row_vals = [f"{result[key].get(c, 0):.2%}" for c in cats]
                row_vals.append(f"{result[key]['all']:.2%}")
                lines.append(f"{alpha:<12}" + "".join(f"{v:<12}" for v in row_vals) + "\n")
            lines.append("-" * 64 + "\n")
        with open(save_file, "w") as f:
            f.writelines(lines)
