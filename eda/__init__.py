"""Пакет исследовательского анализа датасета train_dataset_for_students."""
from pathlib import Path

from .data import (
    get_pairs,
    get_classes,
    pixel_counts_per_class,
    collect_class_stats,
    get_image_sizes,
    object_areas_per_image,
)
from .class_balance import run_class_balance
from .image_sizes import run_image_sizes
from .object_sizes import run_object_sizes
from .domain_issues import run_domain_issues, run_summary

DEFAULT_ROOT = Path("train_dataset_for_students")

__all__ = [
    "get_pairs",
    "get_classes",
    "pixel_counts_per_class",
    "collect_class_stats",
    "get_image_sizes",
    "object_areas_per_image",
    "run_class_balance",
    "run_image_sizes",
    "run_object_sizes",
    "run_domain_issues",
    "run_summary",
    "DEFAULT_ROOT",
]
