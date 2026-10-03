"""Heuristic automatic checks on a generated signature sheet.

Catches the common failures cheaply (empty cells, crowded cells, signatures touching
a cell edge, tiny images). It does NOT judge whether the signatures look like the same
hand; that still needs a quick human look. Tune the thresholds during the pilot.
"""
import cv2
import numpy as np
from PIL import Image

from src.config import GRID_ROWS, GRID_COLS


def _to_bgr(img):
    if isinstance(img, Image.Image):
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def auto_qc(img, rows=GRID_ROWS, cols=GRID_COLS, margin=0.03,
            min_ink=0.004, max_ink=0.30, band=0.04, max_edge_ink=0.01):
    """Returns (ok: bool, reasons: list[str])."""
    bgr = _to_bgr(img)
    h, w = bgr.shape[:2]
    if min(h, w) < 400:
        return False, ["too_small"]

    my, mx = int(h * margin), int(w * margin)
    gray = cv2.cvtColor(bgr[my:h - my, mx:w - mx], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    H, W = ink.shape
    ch, cw = H // rows, W // cols
    by, bx = max(1, int(ch * band)), max(1, int(cw * band))
    reasons = []
    for r in range(rows):
        for c in range(cols):
            cell = ink[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] > 0
            frac = cell.mean()
            if frac < min_ink:
                reasons.append(f"empty_cell_r{r}c{c}")
                continue
            if frac > max_ink:
                reasons.append(f"crowded_cell_r{r}c{c}")
                continue
            edge = np.concatenate([cell[:by].ravel(), cell[-by:].ravel(),
                                   cell[:, :bx].ravel(), cell[:, -bx:].ravel()])
            if edge.mean() > max_edge_ink:
                reasons.append(f"touches_cell_edge_r{r}c{c}")
    return len(reasons) == 0, reasons
