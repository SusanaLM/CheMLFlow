"""Optional small-molecule drug-discovery heuristics and structural alerts."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED


DRUGLIKENESS_COLUMNS = [
    "Lipinski_HBD",
    "Lipinski_HBA",
    "Lipinski_MolWt",
    "Lipinski_MolLogP",
    "Lipinski_Violation_Count",
    "Lipinski_Passes",
    "QED",
    "Veber_TPSA_OK",
    "Veber_RotBonds_OK",
    "LeadLike_Flag",
    "FragmentLike_Flag",
    "MacrocycleLike_Flag",
    "PeptideLike_Size_Flag",
]


def _structural_alert_catalog():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="to-Python converter for boost::shared_ptr.*already registered.*",
            category=RuntimeWarning,
        )
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    params = FilterCatalogParams()
    available = []
    for name in ("PAINS", "BRENK", "NIH"):
        try:
            catalog = getattr(FilterCatalogParams.FilterCatalogs, name)
            params.AddCatalog(catalog)
            available.append(name)
        except AttributeError:
            continue
    if not available:
        return None, []
    return FilterCatalog(params), available


def calculate_druglikeness(
    molecule_table: pd.DataFrame,
    mols: Sequence[Chem.Mol],
    *,
    qed_low_threshold: float = 0.35,
    lipinski_violation_warning_threshold: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    """Calculate cautious drug-discovery heuristics and optional alerts."""
    rows: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        catalog, catalog_names = _structural_alert_catalog()
        catalog_status = "available" if catalog is not None else "unavailable"
    except Exception as exc:
        catalog, catalog_names = None, []
        catalog_status = f"unavailable: {type(exc).__name__}: {exc}"
        warnings.append(f"Structural alert catalogue unavailable: {exc}")

    for position, mol in enumerate(mols):
        source = molecule_table.iloc[position]
        row: dict[str, Any] = {
            "structure_index": int(source["structure_index"]),
            "compound_id": source["compound_id"],
        }
        try:
            hbd = float(source["NumHDonors"])
            hba = float(source["NumHAcceptors"])
            mw = float(source["MolWt"])
            logp = float(source["MolLogP"])
            tpsa = float(source["TPSA"])
            rotors = float(source["NumRotatableBonds"])
            violations = int(hbd > 5) + int(hba > 10) + int(mw > 500) + int(logp > 5)
            ring_info = mol.GetRingInfo().AtomRings()
            row.update(
                {
                    "Lipinski_HBD": hbd,
                    "Lipinski_HBA": hba,
                    "Lipinski_MolWt": mw,
                    "Lipinski_MolLogP": logp,
                    "Lipinski_Violation_Count": violations,
                    "Lipinski_Passes": violations == 0,
                    "QED": float(QED.qed(mol)),
                    "Veber_TPSA_OK": tpsa <= 140,
                    "Veber_RotBonds_OK": rotors <= 10,
                    "LeadLike_Flag": 250 <= mw <= 350 and logp <= 3.5 and hbd <= 3 and hba <= 6,
                    "FragmentLike_Flag": mw <= 300 and logp <= 3 and hbd <= 3 and hba <= 3,
                    "MacrocycleLike_Flag": any(len(ring) >= 12 for ring in ring_info),
                    "PeptideLike_Size_Flag": mw >= 800 or int(source["NumHeteroatoms"]) >= 15,
                }
            )
        except Exception as exc:
            row.update({name: np.nan for name in DRUGLIKENESS_COLUMNS})
            warnings.append(
                f"Drug-likeness calculation failed for {source['compound_id']}: "
                f"{type(exc).__name__}: {exc}"
            )
        rows.append(row)
        if catalog is not None:
            try:
                for match in catalog.GetMatches(mol):
                    alerts.append(
                        {
                            "structure_index": int(source["structure_index"]),
                            "compound_id": source["compound_id"],
                            "alert": match.GetDescription(),
                        }
                    )
            except Exception as exc:
                warnings.append(
                    f"Structural alert matching failed for {source['compound_id']}: {exc}"
                )

    frame = pd.DataFrame(rows)
    alert_frame = pd.DataFrame(
        alerts,
        columns=["structure_index", "compound_id", "alert"],
    )
    qed = pd.to_numeric(frame["QED"], errors="coerce")
    violation_counts = pd.to_numeric(
        frame["Lipinski_Violation_Count"], errors="coerce"
    )
    summary = {
        "interpretation": (
            "These are small-molecule drug-discovery heuristics, not universal "
            "filters. They may not apply to macrocycles, peptides, PROTACs, "
            "covalent fragments, materials, QM datasets, or non-oral concepts."
        ),
        "n_molecules": int(len(frame)),
        "lipinski_pass_count": int(frame["Lipinski_Passes"].fillna(False).sum()),
        "lipinski_pass_fraction": float(frame["Lipinski_Passes"].fillna(False).mean()),
        "lipinski_violation_counts": {
            str(int(key)): int(value)
            for key, value in violation_counts.dropna().value_counts().sort_index().items()
        },
        "qed": {
            "available": int(qed.count()),
            "min": float(qed.min()) if qed.count() else None,
            "median": float(qed.median()) if qed.count() else None,
            "mean": float(qed.mean()) if qed.count() else None,
            "max": float(qed.max()) if qed.count() else None,
            "low_threshold": float(qed_low_threshold),
            "below_threshold_count": int((qed < qed_low_threshold).sum()),
        },
        "lipinski_violation_warning_threshold": int(
            lipinski_violation_warning_threshold
        ),
        "lipinski_warning_count": int(
            (violation_counts >= lipinski_violation_warning_threshold).sum()
        ),
        "lead_like_count": int(frame["LeadLike_Flag"].fillna(False).sum()),
        "fragment_like_count": int(frame["FragmentLike_Flag"].fillna(False).sum()),
        "macrocycle_like_count": int(frame["MacrocycleLike_Flag"].fillna(False).sum()),
        "peptide_like_size_count": int(
            frame["PeptideLike_Size_Flag"].fillna(False).sum()
        ),
        "structural_alert_catalog_status": catalog_status,
        "structural_alert_catalogs": catalog_names,
        "structural_alert_records": int(len(alert_frame)),
        "molecules_with_structural_alerts": int(
            alert_frame["structure_index"].nunique() if not alert_frame.empty else 0
        ),
    }
    return frame, alert_frame, summary, warnings
