"""Анализ размеров изображений."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .dataset_utils import get_pairs, get_image_sizes


def run_image_sizes(root="train_dataset_for_students"):
    root = Path(root)
    train_pairs = get_pairs(root, "train")
    val_pairs = get_pairs(root, "val")
    w_train, h_train = get_image_sizes(train_pairs)
    w_val, h_val = get_image_sizes(val_pairs)
    print("Train:")
    print(f"  Ширина:  min={w_train.min()}, max={w_train.max()}, mean={w_train.mean():.0f}, std={w_train.std():.0f}")
    print(f"  Высота:  min={h_train.min()}, max={h_train.max()}, mean={h_train.mean():.0f}, std={h_train.std():.0f}")
    print(f"  Соотношение сторон: min={(w_train/h_train).min():.2f}, max={(w_train/h_train).max():.2f}")
    print("Val:")
    print(f"  Ширина:  min={w_val.min()}, max={w_val.max()}, mean={w_val.mean():.0f}")
    print(f"  Высота:  min={h_val.min()}, max={h_val.max()}, mean={h_val.mean():.0f}")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(w_train, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Train: распределение ширины")
    axes[0, 0].set_xlabel("Ширина (px)")
    axes[0, 1].hist(h_train, bins=30, color="coral", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Train: распределение высоты")
    axes[0, 1].set_xlabel("Высота (px)")
    axes[1, 0].scatter(w_train, h_train, alpha=0.5, s=20)
    axes[1, 0].set_title("Train: ширина vs высота")
    axes[1, 0].set_xlabel("Ширина")
    axes[1, 0].set_ylabel("Высота")
    aspect_train = w_train / h_train
    axes[1, 1].hist(aspect_train, bins=30, color="seagreen", edgecolor="black", alpha=0.7)
    axes[1, 1].axvline(aspect_train.mean(), color="red", linestyle="--", label=f"mean={aspect_train.mean():.2f}")
    axes[1, 1].set_title("Train: соотношение сторон (W/H)")
    axes[1, 1].set_xlabel("Aspect ratio")
    axes[1, 1].legend()
    plt.tight_layout()
    plt.show()
