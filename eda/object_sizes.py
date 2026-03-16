"""Анализ размеров объектов (площадь по классам)."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .data import get_pairs, get_classes, object_areas_per_image


def run_object_sizes(root="train_dataset_for_students"):
    """Считает площади по классам, выводит describe и гистограммы площадей."""
    root = Path(root)
    train_pairs = get_pairs(root, "train")
    val_pairs = get_pairs(root, "val")
    classes = get_classes(train_pairs + val_pairs)

    df_train = object_areas_per_image(train_pairs, classes)
    df_val = object_areas_per_image(val_pairs, classes)

    area_cols = [f"area_c{c}" for c in classes]
    print("Площадь объектов по классам (пиксели), Train:")
    print(df_train[area_cols].describe().round(0))
    print()
    print("Доля площади изображения по классам (%), Train:")
    print(df_train[[f"pct_c{c}" for c in classes]].describe().round(2))

    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))
    axes = np.atleast_1d(axes).flatten()
    for ax, c in zip(axes, classes):
        col = f"area_c{c}"
        vals = df_train[col].replace(0, np.nan).dropna()
        if len(vals) > 0:
            ax.hist(np.log10(vals), bins=25, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_title(f"Класс {c}: площадь (log10 px)")
        ax.set_xlabel("log10(площадь)")
    plt.tight_layout()
    plt.show()
