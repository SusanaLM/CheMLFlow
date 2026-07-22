"""Publication-oriented static molecular landscape figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_STYLE = {
    # Arial everywhere (safe fallback chain) with editable embedded vector text.
    "font.family": ["Arial", "sans-serif"],
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Font sizes raised ~35% (9->12, 11->15, 8->11, 7->9.5).
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

ACCESSIBLE_PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#000000",
    "#F0E442",
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#AA4499",
)
MARKERS = ("o", "s", "^", "D", "P", "X")


def axis_labels(method: str, diagnostics: Dict[str, Any]) -> tuple[str, str]:
    if method == "pca":
        ratios = diagnostics.get("explained_variance_ratio", [None, None])
        if ratios[0] is not None:
            return (
                f"PC1 ({100.0 * float(ratios[0]):.1f}%)",
                f"PC2 ({100.0 * float(ratios[1]):.1f}%)",
            )
        return "PC1", "PC2"
    labels = {
        "umap": ("UMAP 1", "UMAP 2"),
        "tsne": ("t-SNE 1", "t-SNE 2"),
        "pacmap": ("PaCMAP 1", "PaCMAP 2"),
        "trimap": ("TriMap 1", "TriMap 2"),
    }
    if method in labels:
        return labels[method]
    title = method.upper()
    return f"{title} 1", f"{title} 2"


def _save_all_formats(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(
            output_stem.parent / f"{output_stem.name}{suffix}",
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "molecular-landscape-eda"},
        )


def plot_cluster_map(
    coordinates: np.ndarray,
    labels: Sequence[int],
    title: str,
    method: str,
    diagnostics: Dict[str, Any],
    output_stem: Path,
    top_clusters: int = 12,
) -> None:
    labels_arr = np.asarray(labels, dtype=int)
    unique, counts = np.unique(labels_arr, return_counts=True)
    ordered = unique[np.argsort(counts)[::-1]]
    featured = ordered[:top_clusters]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
        other = ~np.isin(labels_arr, featured)
        ax.scatter(
            coordinates[other, 0],
            coordinates[other, 1],
            s=8,
            c="#c7c9cc",
            alpha=0.48,
            linewidths=0,
            rasterized=True,
            label="Other clusters",
        )
        for idx, cluster_id in enumerate(featured):
            mask = labels_arr == cluster_id
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=12,
                color=ACCESSIBLE_PALETTE[idx % len(ACCESSIBLE_PALETTE)],
                marker=MARKERS[idx % len(MARKERS)],
                alpha=0.82,
                linewidths=0,
                rasterized=True,
                label=f"Cluster {cluster_id} (n={int(mask.sum())})",
            )
        x_label, y_label = axis_labels(method, diagnostics)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
        )
        _save_all_formats(fig, output_stem)
        plt.close(fig)


def plot_shepard(
    source_distances: np.ndarray,
    coordinates: np.ndarray,
    output_stem: Path,
    random_state: int,
    max_pairs: int = 20000,
) -> None:
    """Shepard diagram: source (Tanimoto) distance vs embedded distance on sampled pairs."""
    n_items = len(source_distances)
    rng = np.random.default_rng(random_state)
    left = rng.integers(0, n_items, size=max_pairs * 2)
    right = rng.integers(0, n_items, size=max_pairs * 2)
    keep = left != right
    left, right = left[keep][:max_pairs], right[keep][:max_pairs]
    source = np.asarray(source_distances)[left, right]
    embedded = np.linalg.norm(
        np.asarray(coordinates)[left] - np.asarray(coordinates)[right], axis=1
    )
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
        ax.scatter(
            source, embedded, s=4, alpha=0.22, linewidths=0,
            rasterized=True, color="#0072B2",
        )
        ax.set(
            xlabel="Source Tanimoto distance",
            ylabel="Embedded distance",
            title="Shepard diagram",
        )
        ax.grid(color="#B8B8B8", alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
        _save_all_formats(fig, output_stem)
        plt.close(fig)


def plot_property_map(
    coordinates: np.ndarray,
    values: Sequence[float],
    property_name: str,
    title: str,
    method: str,
    diagnostics: Dict[str, Any],
    output_stem: Path,
    color_limits: Optional[tuple[float, float]] = None,
) -> None:
    values_arr = np.asarray(values, dtype=float)
    if color_limits is None:
        color_limits = (float(np.nanmin(values_arr)), float(np.nanmax(values_arr)))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values_arr,
            cmap="viridis",
            vmin=color_limits[0],
            vmax=color_limits[1],
            s=11,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label(property_name)
        x_label, y_label = axis_labels(method, diagnostics)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        _save_all_formats(fig, output_stem)
        plt.close(fig)
