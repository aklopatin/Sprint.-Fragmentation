"""Проверка числа уникальных меток в масках: python -m practicum_work.src.data.check_num_classes"""
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    root = Path("train_dataset_for_students/labels/train")
    if not root.exists():
        print("Папка не найдена:", root.resolve())
        return
    all_values = set()
    for p in root.glob("*.png"):
        arr = np.array(Image.open(p))
        all_values.update(np.unique(arr))
    all_values.discard(255)
    classes = sorted(all_values)
    num_classes = len(classes)
    print("Уникальные значения меток (кроме 255):", classes)
    print("Рекомендуемое num_classes (включая фон 0):", max(classes) + 1 if classes else 1)
    print("Или число уникальных классов:", num_classes)


if __name__ == "__main__":
    main()
