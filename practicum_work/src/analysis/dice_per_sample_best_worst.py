"""Dice score по каждому семплу валидации, топ-K и худшие K по Dice, запись в experiments_log."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def dice_per_image(pred: np.ndarray, gt: np.ndarray, num_classes: int = 3, ignore_index: int = 255) -> float:
    """Средний Dice по классам для одного изображения."""
    pred = np.asarray(pred).ravel()
    gt = np.asarray(gt).ravel()
    mask = gt != ignore_index
    pred, gt = pred[mask], gt[mask]
    dices = []
    for c in range(num_classes):
        a = (pred == c).astype(np.float32)
        b = (gt == c).astype(np.float32)
        inter = (a * b).sum()
        total = a.sum() + b.sum()
        dices.append(2.0 * inter / total if total > 0 else 1.0)
    return float(np.mean(dices))


def run_val_per_image_dice(
    config_path: str | Path,
    checkpoint_path: str | Path,
    data_root: str | Path,
    val_img_subdir: str = "img/val",
    val_labels_subdir: str = "labels/val",
    device: str = "cuda:0",
    num_classes: int = 3,
) -> list[tuple[str, float]]:
    """Инференс по всем изображениям val, возвращает список (имя_файла, dice)."""
    from pathlib import Path
    import cv2
    from mmseg.apis import init_model, inference_model

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    data_root = Path(data_root)
    img_dir = data_root / val_img_subdir
    label_dir = data_root / val_labels_subdir
    if not img_dir.exists() or not label_dir.exists():
        return []
    model = init_model(str(config_path), str(checkpoint_path), device=device)
    results_list = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        name = img_path.name
        label_path = label_dir / (img_path.stem + ".png")
        if not label_path.exists():
            continue
        gt = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        result = inference_model(model, str(img_path))
        pred = result.pred_sem_seg.data
        if hasattr(pred, "cpu"):
            pred = pred.cpu().numpy()
        pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape != gt.shape:
            from PIL import Image
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST
                )
            )
        results_list.append((name, float(dice_per_image(pred, gt, num_classes=num_classes))))
    return results_list


def top_and_worst(
    results_list: list[tuple[str, float]],
    top_k: int = 5,
    worst_k: int = 5,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Топ-K и худшие K по Dice."""
    sorted_list = sorted(results_list, key=lambda x: x[1], reverse=True)
    top = sorted_list[:top_k]
    worst = sorted_list[-worst_k:] if len(sorted_list) >= worst_k else sorted_list[::-1]
    return top, worst


def log_experiment(
    work_dir: str | Path,
    hyperparams: dict[str, Any],
    results: dict[str, Any] | None = None,
    log_path: str | Path | None = None,
) -> Path:
    """Дописывает эксперимент в JSON-лог (гиперпараметры и результаты)."""
    log_path = Path(log_path or Path("artifacts/experiments_log.json"))
    record = {
        "timestamp": datetime.now().isoformat(),
        "work_dir": str(work_dir),
        "hyperparams": hyperparams,
        "results": results or {},
    }
    entries = []
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, Exception):
            entries = []
    if not isinstance(entries, list):
        entries = [entries]
    entries.append(record)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return log_path
