"""
Shared style and provenance helpers for molecular publication figures.

All panel scripts are a *presentation layer* over the validated workflow:
they read exported artifacts and never re-run the analysis. This module
centralises the colourblind-safe palette, the vector-quality matplotlib
settings, the three-format save routine, and a provenance footer so every
figure is traceable to the run that produced it.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette (matches molecular_landscape/plotting.py)
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#000000", "#F0E442", "#332288", "#88CCEE")
CLASS_ORDER = ("active", "intermediate", "inactive")
CLASS_COLORS = {"active": "#009E73", "intermediate": "#E69F00", "inactive": "#0072B2"}
CLASS_LABEL = {"active": "Active", "intermediate": "Intermediate", "inactive": "Inactive"}

INK = "#14213D"        # dark navy for emphasis lines/curves
ACCENT = "#D55E00"     # vermillion for medians / references
GRID = "#B8B8B8"
MUTED = "#555555"
FAINT = "#888888"


def apply_style() -> None:
    """Vector-first, publication-grade rcParams (editable embedded text).

    Arial is used consistently for every text element; font sizes are set ~37%
    larger than the previous defaults for readability at print scale.
    """
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        # Arial everywhere, with a safe fallback chain if Arial is ever absent.
        "font.family": ["Arial", "sans-serif"],
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        # Font sizes raised ~37% (10->14, 13->18, 11->15, 9.5->13, 8.5->12).
        "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 15,
        "axes.linewidth": 0.9, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 12, "legend.title_fontsize": 12, "legend.frameon": True,
        "figure.dpi": 150, "savefig.dpi": 600,
        "axes.edgecolor": "#444444", "axes.titleweight": "bold",
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def resolve_run_dir(default_rel: str, base_dir: str) -> str:
    """Use argv[1] if given, else a default relative to the script's base dir."""
    import sys
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    return os.path.abspath(os.path.join(base_dir, default_rel))


def require(path: str, hint: str = "") -> str:
    if not os.path.exists(path):
        import sys
        msg = f"[figutils] Missing required artifact:\n  {path}"
        if hint:
            msg += f"\n{hint}"
        sys.exit(msg)
    return path


def load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def load_manifest(run_dir: str) -> dict:
    p = os.path.join(run_dir, "run_manifest.json")
    return load_json(p) if os.path.exists(p) else {}


def provenance_footer(fig, *, source: str, version: str = "", sha256: str = "",
                      date: str = "", extra: str = "", y: float = 0.012) -> None:
    parts = [f"Source: {source}"]
    if version:
        parts.append(f"molecular-landscape-eda {version}")
    if sha256:
        parts.append(f"input SHA-256 {sha256[:10]}…")
    if date:
        parts.append(f"run {date[:10]}")
    if extra:
        parts.append(extra)
    fig.text(0.035, y, "  ·  ".join(parts), ha="left", fontsize=7.0, color=FAINT)


def save_vector(fig, out_stem: str, formats=("pdf", "svg", "png")) -> None:
    parent = os.path.dirname(out_stem)
    if parent:
        os.makedirs(parent, exist_ok=True)
    for ext in formats:
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight",
                    facecolor="white", pad_inches=0.04)
    plt.close(fig)
