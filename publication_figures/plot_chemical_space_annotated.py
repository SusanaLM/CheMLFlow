#!/usr/bin/env python3
"""Chemical-space map annotated with representative molecule structures.

Draws the structure-only projection scatter and overlays the actual 2D structures
of the molecules nearest the four corners and the centre of the map, with leader
lines to their points -- so the figure shows the chemical diversity the layout
captures, not just dots. Only real points are drawn and annotated; nothing is
estimated. All inputs/columns/colours come from a YAML config.

Usage:
    python plot_chemical_space_annotated.py CONFIG.yaml
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

try:  # package use
    from . import figutils as fu
except ImportError:  # direct script use
    import figutils as fu  # type: ignore


def _mol_image(smiles: str, size: int) -> Image.Image | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    options = drawer.drawOptions()
    options.padding = 0.10
    options.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python plot_chemical_space_annotated.py CONFIG.yaml")
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
    x, y = inp["x_column"], inp["y_column"]
    smiles_col = inp.get("smiles_column", "canonical_smiles")
    color_col = inp.get("color_column")

    df = pd.read_csv(table, low_memory=False).dropna(subset=[x, y]).reset_index(drop=True)

    def nearest(px: float, py: float) -> pd.Series:
        return df.loc[((df[x] - px) ** 2 + (df[y] - py) ** 2).idxmin()]

    xmin, xmax, ymin, ymax = df[x].min(), df[x].max(), df[y].min(), df[y].max()
    cx, cy = float(df[x].mean()), float(df[y].mean())
    # Each structure is placed OUTSIDE the scatter, in a fixed figure-margin slot
    # (figure fraction), so no data point is ever covered. Slots: top-left,
    # top-right, bottom-left, bottom-right, bottom-centre. Override via layout.boxes.
    boxes_fig = lay.get(
        "boxes",
        [[0.13, 0.86], [0.66, 0.86], [0.13, 0.14], [0.66, 0.14], [0.43, 0.11]],
    )
    picks = [
        nearest(xmin, ymax),
        nearest(xmax, ymax),
        nearest(xmin, ymin),
        nearest(xmax, ymin),
        nearest(cx, cy),
    ]

    fu.apply_style()
    fig = plt.figure(figsize=tuple(lay.get("figsize", [10.0, 9.0])))
    ax = fig.add_axes(lay.get("axes_rect", [0.30, 0.22, 0.46, 0.56]))

    if color_col:
        values = pd.to_numeric(df[color_col], errors="coerce")
        colormap = lay.get("colormap", "viridis")
        if isinstance(colormap, (list, tuple)):
            from matplotlib.colors import LinearSegmentedColormap
            colormap = LinearSegmentedColormap.from_list("custom", list(colormap))
        scatter = ax.scatter(
            df[x], df[y], c=values, cmap=colormap,
            s=lay.get("point_size", 14), alpha=lay.get("alpha", 0.7),
            linewidths=0, rasterized=True,
        )
        cax = fig.add_axes(lay.get("cbar_rect", [0.78, 0.22, 0.018, 0.56]))
        colorbar = fig.colorbar(scatter, cax=cax)
        colorbar.set_label(lab.get("color", color_col))
    else:
        ax.scatter(df[x], df[y], s=lay.get("point_size", 14),
                   color=fu.PALETTE[0], alpha=lay.get("alpha", 0.7),
                   linewidths=0, rasterized=True)

    # A thin border frames the scatter so the margin structures read as separate.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#9a9a9a")
        spine.set_linewidth(0.8)

    mol_px = int(lay.get("mol_px", 700))
    zoom = float(lay.get("mol_zoom", 0.20))
    for row, (bx, by) in zip(picks, boxes_fig):
        image = _mol_image(row[smiles_col], mol_px)
        if image is None:
            continue
        ax.scatter([row[x]], [row[y]], s=130, facecolors="none",
                   edgecolors=fu.INK, linewidths=1.6, zorder=6)
        annotation = AnnotationBbox(
            OffsetImage(np.asarray(image), zoom=zoom),
            (row[x], row[y]),
            xybox=(bx, by), xycoords=ax.transData, boxcoords=fig.transFigure,
            box_alignment=(0.5, 0.5), pad=0.3, frameon=True,
            bboxprops=dict(edgecolor="#9a9a9a", linewidth=0.8),
            arrowprops=dict(arrowstyle="<-", color=fu.MUTED, lw=0.9,
                            shrinkA=3, shrinkB=4),
            annotation_clip=False,
        )
        ax.add_artist(annotation)

    # Axis labels sit in the margins, where the structures now live, so they are
    # off by default (UMAP axes are arbitrary units anyway). Re-enable with
    # layout.show_axis_labels: true.
    if lay.get("show_axis_labels", False):
        ax.set_xlabel(lab.get("x", "UMAP 1"))
        ax.set_ylabel(lab.get("y", "UMAP 2"))
        # Nudge titles into clear margin gaps (axes-fraction) so the corner
        # structures never clip them.
        ax.xaxis.set_label_coords(*lay.get("xlabel_coords", [0.34, -0.05]))
        ax.yaxis.set_label_coords(*lay.get("ylabel_coords", [-0.06, 0.5]))
    if lab.get("title"):
        ax.set_title(lab["title"])
    if lay.get("hide_ticks", True):
        ax.set_xticks([])
        ax.set_yticks([])

    out = cfg["output"]
    fmts = tuple(out.get("formats", ["pdf", "svg", "png"]))
    fu.save_vector(fig, rel(out["stem"]), formats=fmts)
    print(
        f"annotated {len(picks)} structures "
        f"({len(df)} points); formats: {','.join(fmts)}"
    )


if __name__ == "__main__":
    main()
