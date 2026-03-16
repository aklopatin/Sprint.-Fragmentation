"""Парсинг логов MMEngine (loss, mDice, mIoU по итерациям) и поиск чекпоинта best_mDice."""
from __future__ import annotations

import json
import re
from pathlib import Path


def get_best_mdice_checkpoint(work_dir: str | Path) -> Path | None:
    """Путь к чекпоинту best_mDice_iter_*.pth с максимальным номером итерации."""
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


def parse_mmengine_log(work_dir: str | Path) -> dict[str, list[float]]:
    """Парсит *.log и vis_data/scalars.json в work_dir. Возвращает iter, loss, mDice, mIoU."""
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
    iters, losses, mDices, mIoUs = [], [], [], []
    iter_re = re.compile(r"(?:iter|iteration)[:\s]+(\d+)", re.I)
    loss_re = re.compile(r"loss[:\s]+([\d.]+)", re.I)
    dice_re = re.compile(r"mDice[:\s]+([\d.]+)", re.I)
    iou_re = re.compile(r"mIoU[:\s]+([\d.]+)", re.I)
    last_iter = None
    for line in text.splitlines():
        mi = iter_re.search(line)
        if mi:
            last_iter = int(mi.group(1))
        ml = loss_re.search(line)
        if ml and last_iter is not None:
            iters.append(last_iter)
            losses.append(float(ml.group(1)))
        md = dice_re.search(line)
        if md and last_iter is not None and (not mDices or len(mDices) < len(iters)):
            mDices.append(float(md.group(1)))
        mu = iou_re.search(line)
        if mu and last_iter is not None and (not mIoUs or len(mIoUs) < len(iters)):
            mIoUs.append(float(mu.group(1)))
    return {"iter": iters, "loss": losses, "mDice": mDices, "mIoU": mIoUs}
