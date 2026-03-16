"""Экспорт ноутбука в PDF через nbconvert (webpdf или pdf)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def export_notebook_report_pdf(
    notebook_path: str | Path,
    output_pdf: str | Path = "artifacts/full_report.pdf",
) -> Path:
    """Конвертирует ipynb в PDF. Пробует webpdf, затем обычный pdf."""
    notebook_path = Path(notebook_path)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_basename = output_pdf.stem
    work_dir = output_pdf.parent
    commands = [
        ["jupyter", "nbconvert", "--to", "webpdf", "--allow-chromium-download", str(notebook_path), "--output", output_basename],
        ["jupyter", "nbconvert", "--to", "pdf", str(notebook_path), "--output", output_basename],
    ]
    last_error = None
    for cmd in commands:
        try:
            subprocess.run(cmd, cwd=str(work_dir), check=True, capture_output=True, text=True)
            candidate = work_dir / f"{output_basename}.pdf"
            if candidate.exists():
                return candidate
        except Exception as e:
            last_error = e
    raise RuntimeError("Не удалось экспортировать PDF через nbconvert. " f"Последняя ошибка: {last_error}")
