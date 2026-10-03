"""Split each raw sheet into GRID_ROWS x GRID_COLS single-signature crops.
Run:  python -m src.crop_sheets      (reads dataset/raw_sheets/S01_sheet1.png ...)
Output: dataset/crops/S01/S01_s1_c1.png ...
"""
from pathlib import Path

import cv2

from src.config import DATASET, GRID_ROWS, GRID_COLS


def split_sheet(path, rows=GRID_ROWS, cols=GRID_COLS, margin=0.03):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    my, mx = int(h * margin), int(w * margin)
    img = img[my:h - my, mx:w - mx]                 # trim corner marks / watermark area
    h, w = img.shape[:2]
    ch, cw = h // rows, w // cols
    return [img[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] for r in range(rows) for c in range(cols)]


def run():
    raw, out = DATASET / "raw_sheets", DATASET / "crops"
    for f in sorted(raw.glob("*_sheet*.png")):
        ident, sheet = f.stem.split("_sheet")       # e.g. S01_sheet3
        (out / ident).mkdir(parents=True, exist_ok=True)
        for i, crop in enumerate(split_sheet(f), 1):
            cv2.imwrite(str(out / ident / f"{ident}_s{sheet}_c{i}.png"), crop)
    print("crops written to", out)


if __name__ == "__main__":
    run()
