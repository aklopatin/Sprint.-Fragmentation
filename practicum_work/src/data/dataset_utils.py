"""Загрузка данных и базовые функции для EDA."""
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image


def get_pairs(root, split: str):
    root = Path(root)
    img_dir = root / "img" / split
    ann_dir = root / "labels" / split
    pairs = []
    for p in sorted(img_dir.glob("*.jpg")):
        ann = ann_dir / (p.stem + ".png")
        if ann.exists():
            pairs.append((str(p), str(ann)))
    return pairs


def get_classes(pairs, ignore_index: int = 255):
    classes = set()
    for _, mask_path in pairs:
        arr = np.array(Image.open(mask_path))
        for v in np.unique(arr):
            if v != ignore_index:
                classes.add(int(v))
    return sorted(classes)


def pixel_counts_per_class(mask_path: str, ignore_index: int = 255):
    arr = np.array(Image.open(mask_path))
    counts = defaultdict(int)
    for v in np.unique(arr):
        if v != ignore_index:
            counts[int(v)] = int((arr == v).sum())
    return dict(counts)


def collect_class_stats(pairs):
    total_pixels = defaultdict(int)
    images_with_class = defaultdict(int)
    per_image = []
    for img_path, mask_path in pairs:
        counts = pixel_counts_per_class(mask_path)
        per_image.append(counts)
        for c, n in counts.items():
            total_pixels[c] += n
            images_with_class[c] += 1
    return dict(total_pixels), dict(images_with_class), per_image


def get_image_sizes(pairs):
    widths, heights = [], []
    for img_path, _ in pairs:
        w, h = Image.open(img_path).size
        widths.append(w)
        heights.append(h)
    return np.array(widths), np.array(heights)


def object_areas_per_image(pairs, classes):
    rows = []
    for img_path, mask_path in pairs:
        arr = np.array(Image.open(mask_path))
        h, w = arr.shape
        total_px = h * w
        row = {"img": Path(img_path).name, "W": w, "H": h, "total_px": total_px}
        for c in classes:
            area = (arr == c).sum()
            row[f"area_c{c}"] = area
            row[f"pct_c{c}"] = 100 * area / total_px if total_px else 0
        rows.append(row)
    return pd.DataFrame(rows)
