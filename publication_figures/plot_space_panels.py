#!/usr/bin/env python3
"""Side-by-side chemical-space panels (e.g. PCA | UMAP | t-SNE) in one figure.

Draws one scatter panel per projection, all coloured by the same property on a
shared colour scale and colourbar, so the projections can be compared directly.
Only real points are drawn. All inputs/columns/colours come from a YAML config.

Usage:
    python plot_space_panels.py CONFIG.yaml
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import yaml

import matplotlib.pyplot as plt

try:  # package use
    from . import figutils as fu
except ImportError:  # direct script use
    import figutils as fu  # type: ignore


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python plot_space_panels.py CONFIG.yaml")
    cfg_path = os.path.abspath(sys.argv[1])
    cfg_dir = os.path.dirname(cfg_path)
    with open(cfg_path) as handle:
        cfg = yaml.safe_load(handle)

    def rel(p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(cfg_dir, p))

    inp = cfg["input"]
    table = fu.require(
        os.path.join(rel(inp["run_dir"]), inp["table"]),
        hint="Check input.run_dir / input.table in the config.",
    )
    lab = cfg.get("labels", {})
    lay = cfg.get("layout", {})
    color_col = inp["color_column"]
    df = pd.read_csv(table, low_memory=False)

    panels = []
    for panel in cfg["panels"]:
        method = panel["method"]
        xcol, ycol = f"structure_{method}_x", f"structure_{method}_y"
        if xcol not in df.columns or ycol not in df.columns:
            print(f"skipping {method}: {xcol}/{ycol} not in table")
            continue
        panels.append((panel.get("label", method.upper()), xcol, ycol))
    if not panels:
        sys.exit("No requested projection columns were found in the table.")

    colormap = lay.get("colormap", "viridis")
    if isinstance(colormap, (list, tuple)):
        from matplotlib.colors import LinearSegmentedColormap
        colormap = LinearSegmentedColormap.from_list("custom", list(colormap))

    values_all = pd.to_numeric(df[color_col], errors="coerce")
    vmin, vmax = float(values_all.min()), float(values_all.max())

    fu.apply_style()
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=tuple(lay.get("figsize", [4.6 * len(panels), 4.8])),
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    scatter = None
    for ax, (label, xcol, ycol) in zip(axes, panels):
        sub = df.dropna(subset=[xcol, ycol]).copy()
        values = pd.to_numeric(sub[color_col], errors="coerce")
        scatter = ax.scatter(
            sub[xcol], sub[ycol], c=values, cmap=colormap, vmin=vmin, vmax=vmax,
            s=lay.get("point_size", 9), alpha=lay.get("alpha", 0.75),
            linewidths=0, rasterized=True,
        )
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#9a9a9a")
            spine.set_linewidth(0.8)

    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.015)
    colorbar.set_label(lab.get("color", color_col))

    out = cfg["output"]
    fmts = tuple(out.get("formats", ["pdf", "svg", "png"]))
    fu.save_vector(fig, rel(out["stem"]), formats=fmts)
    print(f"panels: {', '.join(p[0] for p in panels)}; formats: {','.join(fmts)}")


if __name__ == "__main__":
    main()
