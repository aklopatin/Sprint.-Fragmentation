"""Проверка доменных проблем датасета и сводка EDA."""
from pathlib import Path
import numpy as np
from PIL import Image

from .data import (
    get_pairs,
    get_classes,
    pixel_counts_per_class,
    collect_class_stats,
    get_image_sizes,
)


def run_domain_issues(root="train_dataset_for_students", small_area_threshold=100):
    root = Path(root)
    train_pairs = get_pairs(root, "train")
    val_pairs = get_pairs(root, "val")
    classes = get_classes(train_pairs + val_pairs)
    all_pairs = train_pairs + val_pairs

    issues = []
    size_mismatch = []
    for img_path, mask_path in all_pairs:
        wi, hi = Image.open(img_path).size
        wm, hm = Image.open(mask_path).size
        if (wi, hi) != (wm, hm):
            size_mismatch.append((Path(img_path).name, (wi, hi), (wm, hm)))
    issues.append(("Несовпадение размеров img и mask", len(size_mismatch), size_mismatch[:5]))
    missing_class = {c: [] for c in classes}
    for img_path, mask_path in train_pairs:
        counts = pixel_counts_per_class(mask_path)
        for c in classes:
            if counts.get(c, 0) == 0:
                missing_class[c].append(Path(img_path).name)
    for c in classes:
        issues.append(
            (f"Train: нет класса {c} в изображении", len(missing_class[c]), missing_class[c][:3])
        )
    small_objects = []
    for img_path, mask_path in train_pairs:
        arr = np.array(Image.open(mask_path))
        for c in classes:
            area = (arr == c).sum()
            if 0 < area < small_area_threshold:
                small_objects.append((Path(img_path).name, c, int(area)))
    issues.append(
        (
            f"Объекты с площадью < {small_area_threshold} px",
            len(small_objects),
            small_objects[:5],
        )
    )
    only_bg = []
    for img_path, mask_path in train_pairs:
        counts = pixel_counts_per_class(mask_path)
        if sum(counts.get(c, 0) for c in classes if c != 0) == 0:
            only_bg.append(Path(img_path).name)
    issues.append(("Train: изображения только с фоном (класс 0)", len(only_bg), only_bg[:5]))

    for name, count, samples in issues:
        print(f"{name}: {count}")
        if samples:
            print(f"  Примеры: {samples}")
        print()

    return {
        "size_mismatch": size_mismatch,
        "small_objects": small_objects,
        "classes": classes,
        "train_pairs": train_pairs,
    }


def run_summary(root="train_dataset_for_students", small_area_threshold=100):
    root = Path(root)
    train_pairs = get_pairs(root, "train")
    classes = get_classes(train_pairs)
    total_train, _, _ = collect_class_stats(train_pairs)
    w_train, h_train = get_image_sizes(train_pairs)
    size_mismatch = []
    for img_path, mask_path in train_pairs:
        wi, hi = Image.open(img_path).size
        wm, hm = Image.open(mask_path).size
        if (wi, hi) != (wm, hm):
            size_mismatch.append((Path(img_path).name, (wi, hi), (wm, hm)))

    small_objects = []
    for img_path, mask_path in train_pairs:
        arr = np.array(Image.open(mask_path))
        for c in classes:
            area = (arr == c).sum()
            if 0 < area < small_area_threshold:
                small_objects.append((Path(img_path).name, c, int(area)))

    total_px_train = sum(total_train.values())
    pcts = [100 * total_train.get(c, 0) / total_px_train for c in classes] if total_px_train else []
    max_ratio = max(pcts) / (min(pcts) or 1) if pcts else 0

    print("--- Итоги ---")
    print(f"Классов: {len(classes)}")
    print(f"Дисбаланс (макс/мин по пикселям): ~{max_ratio:.1f}x")
    print(f"Размеры изображений: W [{w_train.min()}-{w_train.max()}], H [{h_train.min()}-{h_train.max()}]")
    print(f"Очень маленьких объектов (<{small_area_threshold} px): {len(small_objects)}")
    print()
    print("Рекомендации:")
    if max_ratio > 10:
        print("- Сильный дисбаланс классов: рассмотреть взвешенный loss или oversampling редких классов.")
    if len(small_objects) > 0:
        print("- Есть очень маленькие объекты: аугментации и достаточное разрешение при обучении.")
    if w_train.std() > 200 or h_train.std() > 200:
        print("- Сильный разброс размеров: фиксированный crop или multi-scale обучение.")
    if not size_mismatch:
        print("- Размеры img и mask совпадают — ок.")
    else:
        print("- Внимание: есть несовпадение размеров img и mask.")
