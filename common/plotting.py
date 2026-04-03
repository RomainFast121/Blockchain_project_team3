"""Matplotlib helpers used by the reporting scripts."""

from __future__ import annotations

import os
from pathlib import Path

cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.io_utils import ensure_directory


def set_project_style() -> None:
    """Apply a consistent visual style for report-ready figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure with stable formatting."""
    ensure_directory(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
