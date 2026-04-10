"""Small plotting helpers shared by the report modules.

The figures are generated in non-interactive mode because the project is meant to
run from the command line and later from GitHub, not from a notebook session.
"""

from __future__ import annotations

import os
from pathlib import Path

project_cache_dir = Path(__file__).resolve().parents[1] / ".cache"
matplotlib_cache_dir = project_cache_dir / "matplotlib"

# Setting both cache locations upfront avoids noisy cache warnings in constrained
# environments and makes the test/CLI pipeline reproducible.
project_cache_dir.mkdir(parents=True, exist_ok=True)
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(project_cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.io_utils import ensure_directory


def set_project_style() -> None:
    """Apply one consistent figure style across all modules."""

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
    """Save a figure with stable export settings and close it."""

    ensure_directory(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
