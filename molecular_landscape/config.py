"""Typed configuration for the molecular landscape workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .eda.config import EDAConfig


_REPRESENTATION_CHOICES = {"morgan", "fcfp", "rdkit", "atompair", "torsion", "maccs"}


@dataclass(frozen=True)
class FingerprintConfig:
    radius: int = 2
    n_bits: int = 2048
    include_chirality: bool = True
    use_features: bool = False
    # Optional representation-sensitivity diagnostic: compare the default Morgan
    # Tanimoto geometry against these alternative fingerprint families.
    representation_sensitivity: bool = False
    comparison_representations: List[str] = field(
        default_factory=lambda: ["fcfp", "rdkit", "atompair", "torsion", "maccs"]
    )

    def validate(self) -> None:
        if self.radius < 1:
            raise ValueError("Fingerprint radius must be at least 1.")
        if self.n_bits < 128:
            raise ValueError("Fingerprint size must be at least 128 bits.")
        invalid = sorted(set(self.comparison_representations) - _REPRESENTATION_CHOICES)
        if invalid:
            raise ValueError(f"Unsupported comparison representations: {invalid}")


@dataclass(frozen=True)
class EmbeddingConfig:
    property_weight: float = 0.20
    property_weight_sensitivity: List[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.30]
    )
    random_state: int = 42
    umap_seed_sensitivity: List[int] = field(default_factory=lambda: [7, 42, 99])
    umap_neighbors: int = 30
    umap_min_dist: float = 0.10
    validation_neighbors: int = 15
    max_pairwise_molecules: int = 5000
    # t-SNE is an optional, additional map method (off by default because it is
    # O(n^2); enable only when requested). PCA and UMAP are always produced.
    include_tsne: bool = False
    tsne_perplexity: float = 30.0
    # PaCMAP and TriMap are optional modern projections (better global/local
    # balance), off by default and requiring the optional pacmap/trimap packages.
    include_pacmap: bool = False
    include_trimap: bool = False
    pacmap_neighbors: int = 10
    # Optional structure-only hyperparameter-selection sweep for the enabled
    # methods, written to diagnostics/map_method_selection.csv.
    map_method_selection: bool = False
    # Optional co-ranking (multi-scale R_NX/LCMC) diagnostics, random-layout
    # baselines, and Shepard-diagram figures for the structure-only maps.
    coranking_diagnostics: bool = False

    def validate(self) -> None:
        weights = [self.property_weight, *self.property_weight_sensitivity]
        if any(not 0.0 <= value <= 1.0 for value in weights):
            raise ValueError("Property weights must be between 0 and 1.")
        if self.umap_neighbors < 2:
            raise ValueError("UMAP neighbors must be at least 2.")
        if not self.umap_seed_sensitivity:
            raise ValueError("At least one UMAP sensitivity seed is required.")
        if not 0.0 <= self.umap_min_dist <= 1.0:
            raise ValueError("UMAP minimum distance must be between 0 and 1.")
        if self.validation_neighbors < 2:
            raise ValueError("Validation neighbors must be at least 2.")
        if self.max_pairwise_molecules < 10:
            raise ValueError("Maximum pairwise molecules must be at least 10.")
        if self.tsne_perplexity <= 1.0:
            raise ValueError("t-SNE perplexity must be greater than 1.")
        if self.pacmap_neighbors < 2:
            raise ValueError("PaCMAP neighbors must be at least 2.")


@dataclass(frozen=True)
class ClusteringConfig:
    butina_similarity_threshold: float = 0.65
    threshold_sensitivity: List[float] = field(
        default_factory=lambda: [0.55, 0.65, 0.75]
    )
    # Optional threshold-free density-based clustering (HDBSCAN) reported
    # alongside Butina, with adjusted Rand index agreement.
    hdbscan: bool = False
    hdbscan_min_cluster_size: int = 5

    def validate(self) -> None:
        thresholds = [
            self.butina_similarity_threshold,
            *self.threshold_sensitivity,
        ]
        if any(not 0.0 < value < 1.0 for value in thresholds):
            raise ValueError("Butina similarity thresholds must be between 0 and 1.")
        if self.hdbscan_min_cluster_size < 2:
            raise ValueError("HDBSCAN minimum cluster size must be at least 2.")


@dataclass(frozen=True)
class WorkflowConfig:
    input_path: Path
    output_dir: Path
    smiles_col: Optional[str] = None
    id_col: Optional[str] = None
    property_cols: Optional[List[str]] = None
    property_transforms: Dict[str, str] = field(default_factory=dict)
    sample_size: Optional[int] = None
    overwrite: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    eda: EDAConfig = field(default_factory=EDAConfig)

    def validate(self) -> None:
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {self.output_dir}")
        if self.input_path.resolve() == self.output_dir.resolve():
            raise ValueError("Input file and output directory must be different paths.")
        if self.sample_size is not None and self.sample_size < 10:
            raise ValueError("Sample size must be at least 10.")
        if self.property_cols is not None:
            duplicates = sorted(
                {
                    name
                    for name in self.property_cols
                    if self.property_cols.count(name) > 1
                }
            )
            if duplicates:
                raise ValueError(f"Duplicate property columns are not allowed: {duplicates}")
            unused_transforms = sorted(
                set(self.property_transforms).difference(self.property_cols)
            )
            if unused_transforms:
                raise ValueError(
                    "Property transforms were provided for unselected columns: "
                    f"{unused_transforms}"
                )
        allowed_transforms = {
            "auto",
            "none",
            "log1p",
            "signed_log1p",
            "quantile",
        }
        invalid = {
            name: transform
            for name, transform in self.property_transforms.items()
            if transform not in allowed_transforms
        }
        if invalid:
            raise ValueError(f"Unsupported property transforms: {invalid}")
        self.fingerprint.validate()
        self.embedding.validate()
        self.clustering.validate()
        self.eda.validate()
        if self.eda.map_method == "tsne" and not self.embedding.include_tsne:
            raise ValueError(
                "EDA map method 'tsne' requires the t-SNE embedding to be enabled "
                "(set embedding.include_tsne / pass --include-tsne)."
            )
        if self.eda.map_method == "pacmap" and not self.embedding.include_pacmap:
            raise ValueError(
                "EDA map method 'pacmap' requires PaCMAP to be enabled."
            )
        if self.eda.map_method == "trimap" and not self.embedding.include_trimap:
            raise ValueError(
                "EDA map method 'trimap' requires TriMap to be enabled."
            )

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        return payload
