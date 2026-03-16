"""Логирование экспериментов, парсинг логов mmengine, графики, top/worst по Dice."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENTS_JSON = Path("artifacts/experiments_log.json")


def get_best_mdice_checkpoint(work_dir: str | Path) -> Path | None:
    work_dir = Path(work_dir)
    if not work_dir.exists():
        return None
    pattern = re.compile(r"best_mDice_iter_(\d+)\.pth$")
    candidates = []
    for p in work_dir.glob("best_mDice_iter_*.pth"):
        m = pattern.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def log_experiment(
    work_dir: str | Path,
    hyperparams: dict[str, Any],
    results: dict[str, Any] | None = None,
    log_path: str | Path | None = None,
) -> Path:
    log_path = Path(log_path or EXPERIMENTS_JSON)
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


def parse_mmengine_log(work_dir: str | Path) -> dict[str, list[float]]:
    work_dir = Path(work_dir)
    if not work_dir.exists():
        return {"iter": [], "loss": [], "mDice": [], "mIoU": []}

    log_files = list(work_dir.rglob("*.log"))
    if not log_files:
        log_files = list(work_dir.glob("*.log"))
    if not log_files:
        return {"iter": [], "loss": [], "mDice": [], "mIoU": []}
    vis_scalars = work_dir / "vis_data" / "scalars.json"
    if vis_scalars.exists():
        try:
            data = json.loads(vis_scalars.read_text(encoding="utf-8"))
            out = {"iter": [], "loss": [], "mDice": [], "mIoU": []}
            for key, points in data.items():
                if not points:
                    continue
                it = [p[0] for p in points]
                vals = [p[1] for p in points]
                if "loss" in key.lower():
                    out["loss"] = vals
                    if not out["iter"]:
                        out["iter"] = it
                elif "dice" in key.lower():
                    out["mDice"] = vals
                elif "iou" in key.lower() or "mIoU" in key:
                    out["mIoU"] = vals
            if out["iter"] or out["loss"]:
                return out
        except Exception:
            pass
    main_log = max(log_files, key=lambda p: p.stat().st_size)
    text = main_log.read_text(encoding="utf-8", errors="ignore")

    iters: list[float] = []
    losses: list[float] = []
    mDices: list[float] = []
    mIoUs: list[float] = []
    iter_re = re.compile(r"(?:iter|iteration)[:\s]+(\d+)", re.I)
    loss_re = re.compile(r"loss[:\s]+([\d.]+)", re.I)
    dice_re = re.compile(r"mDice[:\s]+([\d.]+)", re.I)
    iou_re = re.compile(r"mIoU[:\s]+([\d.]+)", re.I)

    last_iter = None
    for line in text.splitlines():
        mi = iter_re.search(line)
        ml = loss_re.search(line)
        md = dice_re.search(line)
        mu = iou_re.search(line)
        if mi:
            last_iter = int(mi.group(1))
        if ml and last_iter is not None:
            iters.append(last_iter)
            losses.append(float(ml.group(1)))
        if md and last_iter is not None:
            if not mDices or len(mDices) < len(iters):
                mDices.append(float(md.group(1)))
        if mu and last_iter is not None:
            if not mIoUs or len(mIoUs) < len(iters):
                mIoUs.append(float(mu.group(1)))
    out: dict[str, list[float]] = {
        "iter": iters or [],
        "loss": losses or [],
        "mDice": mDices or [],
        "mIoU": mIoUs or [],
    }
    return out


def plot_training_curves(
    logs: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен, графики не построены")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    iters = logs.get("iter", [])
    if not iters:
        plt.suptitle("Нет данных логов (запустите обучение и укажите work_dir)")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        return
    ax = axes[0]
    ax.plot(iters, logs.get("loss", []), label="loss", color="C0")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Loss")
    ax.set_title("Loss по итерациям")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    n_val = len(logs.get("mDice", []))
    if n_val > 0:
        val_interval = 4000
        max_iter = max(iters) if iters else n_val * val_interval
        val_iters = list(range(val_interval, max_iter + 1, val_interval))[:n_val]
        if len(val_iters) < n_val:
            val_iters = list(range(val_interval, val_interval * (n_val + 1), val_interval))[:n_val]
        ax.plot(val_iters, logs.get("mDice", []), label="mDice", color="C1")
        if len(logs.get("mIoU", [])) == n_val:
            ax.plot(val_iters, logs.get("mIoU", []), label="mIoU", color="C2")
        elif logs.get("mIoU"):
            ax.plot(logs.get("mIoU", []), label="mIoU", color="C2")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Метрика")
    ax.set_title("mDice / mIoU на валидации")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def dice_per_image(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int = 3,
    ignore_index: int = 255,
) -> float:
    pred = np.asarray(pred).ravel()
    gt = np.asarray(gt).ravel()
    mask = gt != ignore_index
    pred = pred[mask]
    gt = gt[mask]
    dices = []
    for c in range(num_classes):
        a = (pred == c).astype(np.float32)
        b = (gt == c).astype(np.float32)
        inter = (a * b).sum()
        total = a.sum() + b.sum()
        if total > 0:
            dices.append(2.0 * inter / total)
        else:
            dices.append(1.0)
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
    results_list: list[tuple[str, float]] = []

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
        dice = dice_per_image(pred, gt, num_classes=num_classes)
        results_list.append((name, float(dice)))

    return results_list


def top_and_worst(
    results_list: list[tuple[str, float]],
    top_k: int = 5,
    worst_k: int = 5,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    sorted_list = sorted(results_list, key=lambda x: x[1], reverse=True)
    top = sorted_list[:top_k]
    worst = sorted_list[-worst_k:] if len(sorted_list) >= worst_k else sorted_list[::-1]
    return top, worst


def export_notebook_report_pdf(
    notebook_path: str | Path,
    output_pdf: str | Path = "artifacts/full_report.pdf",
) -> Path:
    notebook_path = Path(notebook_path)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_basename = output_pdf.stem
    work_dir = output_pdf.parent

    commands = [
        [
            "jupyter",
            "nbconvert",
            "--to",
            "webpdf",
            "--allow-chromium-download",
            str(notebook_path),
            "--output",
            output_basename,
        ],
        [
            "jupyter",
            "nbconvert",
            "--to",
            "pdf",
            str(notebook_path),
            "--output",
            output_basename,
        ],
    ]

    last_error = None
    for cmd in commands:
        try:
            subprocess.run(
                cmd,
                cwd=str(work_dir),
                check=True,
                capture_output=True,
                text=True,
            )
            candidate = work_dir / f"{output_basename}.pdf"
            if candidate.exists():
                return candidate
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        "Не удалось экспортировать PDF через nbconvert. "
        f"Последняя ошибка: {last_error}"
    )
