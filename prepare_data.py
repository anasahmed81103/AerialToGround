"""
prepare_data.py
---------------
Converts the raw CVPR-subset split CSVs into correctly-pathed manifests
that the CrossNet data loader can use directly.

The split CSVs ship paths like:
    bingmap/19/NNNNNNN.jpg , streetview/panos/NNNNNNN.jpg , streetview/annotations/NNNNNNN.png

But after extraction the real layout is:
    cvpr_subset/bingmap/bingmap/19/NNNNNNN.jpg
    cvpr_subset/streetview/streetview/panos/NNNNNNN.jpg
    cvpr_subset/streetview/streetview/annotations/NNNNNNN.png

This script rewrites them as paths relative to the project root so the data
loader can open them with image_dir="" (empty prefix).

Outputs:
    cvpr_train.csv   — 35,532 training pairs
    cvpr_val.csv     — 8,884  validation pairs

Usage:
    python prepare_data.py
"""

import os
import csv

SUBSET_ROOT  = "cvpr_subset"
AERIAL_BASE  = os.path.join(SUBSET_ROOT, "bingmap",    "bingmap")
GROUND_BASE  = os.path.join(SUBSET_ROOT, "streetview", "streetview")

SPLIT_TRAIN  = os.path.join(SUBSET_ROOT, "splits", "splits", "train-19zl.csv")
SPLIT_VAL    = os.path.join(SUBSET_ROOT, "splits", "splits", "val-19zl.csv")

OUT_TRAIN    = "cvpr_train.csv"
OUT_VAL      = "cvpr_val.csv"


def remap_paths(orig_aerial, orig_ground, orig_label):
    """Convert original relative paths to project-root-relative paths."""
    # orig_aerial : "bingmap/19/NNNNNNN.jpg"  → strip "bingmap/" prefix
    aerial_rel = orig_aerial[len("bingmap/"):]          # "19/NNNNNNN.jpg"
    # orig_ground : "streetview/panos/NNNNNNN.jpg"      → strip "streetview/" prefix
    ground_rel = orig_ground[len("streetview/"):]       # "panos/NNNNNNN.jpg"
    # orig_label  : "streetview/annotations/NNNNNNN.png"
    label_rel  = orig_label[len("streetview/"):]        # "annotations/NNNNNNN.png"

    aerial = os.path.join(AERIAL_BASE, aerial_rel).replace("\\", "/")
    ground = os.path.join(GROUND_BASE, ground_rel).replace("\\", "/")
    label  = os.path.join(GROUND_BASE, label_rel).replace("\\", "/")
    return aerial, ground, label


def convert(src_csv, dst_csv, label):
    kept = 0
    skipped = 0
    with open(src_csv, "r") as fin, open(dst_csv, "w", newline="") as fout:
        for line in fin:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != 3:
                skipped += 1
                continue
            aerial, ground, lbl = remap_paths(*parts)
            # Only include rows where all three files actually exist on disk
            if os.path.exists(aerial) and os.path.exists(ground) and os.path.exists(lbl):
                fout.write(f"{aerial},{ground},{lbl}\n")
                kept += 1
            else:
                skipped += 1

    print(f"[{label}] written {kept:,} pairs -> {dst_csv}  (skipped {skipped:,})")
    return kept


if __name__ == "__main__":
    train_count = convert(SPLIT_TRAIN, OUT_TRAIN, "train")
    val_count   = convert(SPLIT_VAL,   OUT_VAL,   "val")
    print(f"\nTotal: {train_count + val_count:,} usable image pairs")
    print("Done. Now run:  python main.py --data_list=cvpr_train.csv --batch_size=8")
