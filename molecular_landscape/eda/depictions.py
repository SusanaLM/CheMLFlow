"""Robust RDKit SVG molecule depiction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from ..io_utils import safe_filename_token


def generate_molecule_svgs(
    structure_df: pd.DataFrame,
    output_dir: Path,
    max_molecules: int,
) -> Tuple[Dict[int, str], List[str]]:
    """Generate stable SVG depictions without allowing one failure to abort a run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[int, str] = {}
    warnings: List[str] = []
    for position, (_, row) in enumerate(structure_df.iterrows()):
        structure_index = int(row["_structure_index"])
        if position >= max_molecules:
            break
        identifier = safe_filename_token(str(row["_compound_id"]), max_length=60)
        filename = f"{identifier}__{structure_index:06d}.svg"
        try:
            mol = Chem.MolFromSmiles(str(row["_canonical_smiles"]))
            if mol is None:
                raise ValueError("RDKit could not reconstruct the canonical SMILES")
            mol = Chem.Mol(mol)
            AllChem.Compute2DCoords(mol)
            drawer = Draw.MolDraw2DSVG(320, 240)
            drawer.drawOptions().clearBackground = False
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            (output_dir / filename).write_text(
                drawer.GetDrawingText(),
                encoding="utf-8",
            )
            paths[structure_index] = f"eda/molecule_svgs/{filename}"
        except Exception as exc:
            warnings.append(
                f"Depiction failed for structure index {structure_index} "
                f"({row['_compound_id']}): {type(exc).__name__}: {exc}"
            )
    if len(structure_df) > max_molecules:
        warnings.append(
            f"SVG generation was limited to {max_molecules} of "
            f"{len(structure_df)} valid molecules."
        )
    return paths, warnings
