"""Build dataset/metadata.csv automatically from the crop file names and the split plan.
Put the file names of crops you reject during review (one per line, e.g. S01_s3_c4.png)
in dataset/rejects.txt, then run:  python -m src.build_metadata
"""
from pathlib import Path

import pandas as pd

from src.config import DATASET, PROMPT_VERSION
from src.simulate_capture import condition_for, parse

SPLIT_ENROLLED = {1: "train", 2: "train", 3: "val", 4: "test"}
SPLIT_UNKNOWN = {
    "U01": "unk_val", "U02": "unk_val",
    "U03": "unk_test", "U04": "unk_test", "U05": "unk_test"
}


def get_split(identity, sheet, crop_idx, has_multiple_sheets):
    if identity.startswith("U"):
        return SPLIT_UNKNOWN.get(identity, "unk_test")
    if has_multiple_sheets:
        return SPLIT_ENROLLED.get(sheet, "train")
    if crop_idx <= 8:
        return "train"
    elif crop_idx <= 10:
        return "val"
    return "test"


def main():
    rej_file = DATASET / "rejects.txt"
    rejects = set(rej_file.read_text().split()) if rej_file.exists() else set()
    crops = sorted((DATASET / "crops").glob("*/*.png"))
    if not crops:
        print("No crops found to build metadata.")
        return

    sheets_present = set(parse(f.stem)[1] for f in crops if not f.stem.startswith("U"))
    has_multiple = max(sheets_present, default=1) > 1

    rows = []
    for f in crops:
        ident, sheet, crop_idx = parse(f.stem)
        cond = condition_for(ident, sheet, crop_idx=crop_idx, has_multiple_sheets=has_multiple)
        used = f if cond == "A" else DATASET / "crops_sim" / ident / f.name
        group = "unknown" if ident.startswith("U") else "enrolled"
        split = get_split(ident, sheet, crop_idx, has_multiple)
        rows.append(dict(file=used.relative_to(DATASET).as_posix(), identity=ident, group=group,
                         sheet=sheet, condition=cond, split=split,
                         qc="reject" if f.name in rejects else "ok", prompt_version=PROMPT_VERSION))

        # Also add condition-B sim augmentation variant for train crops (enrolled only)
        if split == "train" and not ident.startswith("U"):
            aug_path = DATASET / "crops_sim_aug" / ident / f.name
            if aug_path.exists():
                rows.append(dict(file=aug_path.relative_to(DATASET).as_posix(), identity=ident, group=group,
                                 sheet=sheet, condition="B", split="train",
                                 qc="reject" if f.name in rejects else "ok", prompt_version=PROMPT_VERSION))

    df = pd.DataFrame(rows)
    df.to_csv(DATASET / "metadata.csv", index=False)
    print(df.groupby(["group", "split", "qc"]).size())



if __name__ == "__main__":
    main()
