"""Анализ баланса классов в датасете."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .dataset_utils import get_pairs, get_classes, collect_class_stats


def run_class_balance(root="train_dataset_for_students"):
    root = Path(root)
    train_pairs = get_pairs(root, "train")
    val_pairs = get_pairs(root, "val")
    classes = get_classes(train_pairs + val_pairs)
    total_train, img_train, _ = collect_class_stats(train_pairs)
    total_val, img_val, _ = collect_class_stats(val_pairs)
    print("Классы в масках:", classes)
    print()
    for split_name, total, img_count in [
        ("Train", total_train, img_train),
        ("Val", total_val, img_val),
    ]:
        s = sum(total.values())
        print(f"=== {split_name} ===")
        for c in classes:
            pct = 100 * total.get(c, 0) / s if s else 0
            print(
                f"  Класс {c}: {total.get(c, 0):>12} пикселей ({pct:5.2f}%)  |  в {img_count.get(c, 0)} изображениях"
            )
        print()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (total, title) in zip(
        axes,
        [
            (total_train, "Train: распределение пикселей по классам"),
            (total_val, "Val: распределение пикселей по классам"),
        ],
    ):
        s = sum(total.values())
        labels = [f"Класс {c}" for c in classes]
        sizes = [total.get(c, 0) / s * 100 if s else 0 for c in classes]
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
    total_px = sum(total_train.values())
    min_px = min(total_train.get(c, 1) for c in classes) or 1
    max_px = max(total_train.get(c, 0) for c in classes)
    imbalance = max_px / min_px
    print(f"Соотношение самого частого к самому редкому классу (по пикселям): ~{imbalance:.1f}x")
