"""Simulate different capture conditions on cropped signatures (instead of generating them).

Condition A = clean (as generated). B = mild phone-photo look. C = stronger phone-photo look.
Which crops get which condition is fixed by the split plan (see PROJECT file, section 6.2).
Run:  python -m src.simulate_capture      (reads dataset/crops, writes dataset/crops_sim)
"""
import zlib
from pathlib import Path

import cv2
import numpy as np

from src.config import DATASET, SEED

SHEET_CONDITION = {1: "A", 2: "A", 3: "B", 4: "C"}               # enrolled identities
UNKNOWN_CONDITION = {"U01": "B", "U02": "B", "U03": "C", "U04": "C", "U05": "C"}

PARAMS = {
    "B": dict(shift=0.02, shadow=0.10, bright=(0.90, 1.05), noise=3.0, jpeg=80, blur=0.0),
    "C": dict(shift=0.04, shadow=0.22, bright=(0.80, 1.00), noise=6.0, jpeg=60, blur=0.6),
}


def condition_for(identity, sheet, crop_idx=1, has_multiple_sheets=False):
    if identity.startswith("U"):
        return UNKNOWN_CONDITION.get(identity, "C")
    if has_multiple_sheets:
        return SHEET_CONDITION.get(sheet, "A")
    if crop_idx <= 8:
        return "A"
    elif crop_idx <= 10:
        return "B"
    return "C"


def simulate(img, condition, seed):
    if condition == "A":
        return img
    p = PARAMS[condition]
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    # 1) small perspective change (camera not perfectly overhead)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    d = p["shift"] * min(h, w)
    dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

    # 2) soft lighting gradient (shadow)
    ang = rng.uniform(0, 2 * np.pi)
    xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    g = np.cos(ang) * xs + np.sin(ang) * ys
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    out *= (1 - p["shadow"] * g)[..., None]

    # 3) brightness, 4) blur + sensor noise
    out *= rng.uniform(*p["bright"])
    if p["blur"] > 0:
        out = cv2.GaussianBlur(out, (0, 0), p["blur"])
    out += rng.normal(0, p["noise"], out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 5) JPEG compression
    _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def parse(stem):
    """'S01_s1_c4' -> ('S01', 1, 4)"""
    ident, rest = stem.split("_s", 1)
    sheet_str, crop_str = rest.split("_c", 1)
    return ident, int(sheet_str), int(crop_str)


def run():
    src_dir, out_dir = DATASET / "crops", DATASET / "crops_sim"
    # Also produce simulated-augmentation crops for train (c1-c8)
    # so the CNN sees domain-shifted variants at training time.
    aug_dir = DATASET / "crops_sim_aug"
    crops = sorted(src_dir.glob("*/*.png"))
    if not crops:
        print("No crops found in", src_dir)
        return
    sheets_present = set(parse(f.stem)[1] for f in crops if not f.stem.startswith("U"))
    has_multiple = max(sheets_present, default=1) > 1

    n, n_aug = 0, 0
    for f in crops:
        ident, sheet, crop_idx = parse(f.stem)
        cond = condition_for(ident, sheet, crop_idx=crop_idx, has_multiple_sheets=has_multiple)
        img = cv2.imread(str(f))
        if cond == "A":
            # Generate a condition-B sim variant for train augmentation
            if not ident.startswith("U"):
                aug_out = simulate(img, "B", SEED + zlib.crc32((f.stem + "_aug").encode()))
                (aug_dir / ident).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(aug_dir / ident / f.name), aug_out)
                n_aug += 1
        else:
            out = simulate(img, cond, SEED + zlib.crc32(f.stem.encode()))  # deterministic
            (out_dir / ident).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / ident / f.name), out)
            n += 1
    print(f"simulated {n} crops -> {out_dir}")
    print(f"train augmentation sim {n_aug} crops -> {aug_dir}")


if __name__ == "__main__":
    run()
