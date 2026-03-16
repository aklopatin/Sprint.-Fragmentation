"""Построение графиков loss и метрик (mDice, mIoU) по итерациям обучения."""
from __future__ import annotations

from pathlib import Path


def plot_training_curves(
    logs: dict[str, list],
    save_path: str | Path | None = None,
) -> None:
    """Строит графики loss и mDice/mIoU. При указании save_path сохраняет в файл."""
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
    axes[0].plot(iters, logs.get("loss", []), label="loss", color="C0")
    axes[0].set_xlabel("Итерация")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss по итерациям")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    n_val = len(logs.get("mDice", []))
    if n_val > 0:
        val_interval = 4000
        max_iter = max(iters) if iters else n_val * val_interval
        val_iters = list(range(val_interval, max_iter + 1, val_interval))[:n_val]
        if len(val_iters) < n_val:
            val_iters = list(range(val_interval, val_interval * (n_val + 1), val_interval))[:n_val]
        axes[1].plot(val_iters, logs.get("mDice", []), label="mDice", color="C1")
        if logs.get("mIoU") and len(logs["mIoU"]) == n_val:
            axes[1].plot(val_iters, logs["mIoU"], label="mIoU", color="C2")
        elif logs.get("mIoU"):
            axes[1].plot(logs["mIoU"], label="mIoU", color="C2")
    axes[1].set_xlabel("Итерация")
    axes[1].set_ylabel("Метрика")
    axes[1].set_title("mDice / mIoU на валидации")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
