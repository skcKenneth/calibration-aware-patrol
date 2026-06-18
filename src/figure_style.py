"""Editorial plotting helpers for publication-quality scientific figures.

The style is intentionally restrained and journal-like: compact dimensions,
colour-blind-safe accents, thin axes, outward ticks, panel labels, and vector
output.  It is inspired by common Nature/Science figure conventions without
copying any proprietary house template.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Compact colour-blind-safe editorial palette.
COLORS = {
    "ink": "#222222",
    "grey": "#7A7A7A",
    "light_grey": "#D9D9D9",
    "pale_grey": "#F2F2F2",
    "blue": "#3C5488",
    "vermillion": "#E64B35",
    "teal": "#00A087",
    "sky": "#4DBBD5",
    "gold": "#E69F00",
    "purple": "#7E6148",
}

LINE_COLORS = [
    COLORS["ink"],
    COLORS["blue"],
    COLORS["vermillion"],
    COLORS["teal"],
    COLORS["gold"],
    COLORS["sky"],
]

PATROL_CMAP = LinearSegmentedColormap.from_list(
    "patrol_teal",
    ["#FFFFFF", "#E8F2F0", "#B7D7D2", "#6AAFA6", "#006D6F"],
)

# Approximate journal widths.
SINGLE_COLUMN = 3.70
DOUBLE_COLUMN = 7.50


def apply_editorial_style() -> None:
    """Apply a compact journal-style Matplotlib configuration."""
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "figure.constrained_layout.use": False,
            "savefig.dpi": 600,
            "savefig.transparent": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Liberation Sans",
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.65,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "legend.handlelength": 1.7,
            "legend.handletextpad": 0.45,
            "legend.borderaxespad": 0.25,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.8,
            "patch.linewidth": 0.5,
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "stixsans",
            "axes.prop_cycle": mpl.cycler(color=LINE_COLORS),
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    """Apply subtle axis styling while preserving data-specific formatting."""
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.tick_params(which="both", color=COLORS["ink"], labelcolor=COLORS["ink"])


def panel_label(axis: plt.Axes, label: str, *, x: float = -0.16, y: float = 1.08) -> None:
    """Place a bold lowercase panel label in axes coordinates."""
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        clip_on=False,
    )


def compact_legend(axis: plt.Axes, *, ncol: int = 1, **kwargs: object) -> None:
    """Create a compact frameless legend."""
    axis.legend(ncol=ncol, frameon=False, **kwargs)


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    """Save publication outputs without clipping labels or panel annotations.

    ``bbox_inches='tight'`` alone can still crop artists that sit very close to
    the canvas boundary, especially panel labels and long axis labels in SVG or
    PDF output.  Drawing the canvas first and retaining a slightly larger pad
    makes all three output formats consistent.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            directory / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.10,
            facecolor="white",
        )


def shade_uncertainty(
    axis: plt.Axes,
    x: Iterable[float],
    mean: Iterable[float],
    sd: Iterable[float],
    *,
    color: str,
    alpha: float = 0.14,
) -> None:
    """Draw a mean ± 1 SD band."""
    import numpy as np

    x_array = np.asarray(list(x), dtype=float)
    mean_array = np.asarray(list(mean), dtype=float)
    sd_array = np.asarray(list(sd), dtype=float)
    axis.fill_between(
        x_array,
        mean_array - sd_array,
        mean_array + sd_array,
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=1,
    )
