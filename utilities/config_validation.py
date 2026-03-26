from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    hint: str | None = None


class ConfigValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        details = []
        for issue in issues:
            hint = f" Hint: {issue.hint}" if issue.hint else ""
            details.append(f"- [{issue.code}] {issue.path}: {issue.message}{hint}")
        msg = "Config validation failed:\n" + "\n".join(details)
        super().__init__(msg)


_NODE_TO_BLOCK = {
    "get_data": "get_data",
    "curate": "curate",
    "label.normalize": "label",
    "split": "split",
    "featurize.none": "featurize",
    "featurize.lipinski": "featurize",
    "featurize.rdkit": "featurize",
    "featurize.rdkit_labeled": "featurize",
    "featurize.morgan": "featurize",
    "preprocess.features": "preprocess",
    "select.features": "preprocess",
    "train": "train",
    "train.tdc": "train_tdc",
}

_CONFIGLESS_NODE_TO_BLOCK = {
    "featurize.none": "featurize",
}

_ALWAYS_ALLOWED_BLOCKS = {"global", "pipeline"}
_KNOWN_TOP_LEVEL_BLOCKS = {
    *_ALWAYS_ALLOWED_BLOCKS,
    "get_data",
    "curate",
    "label",
    "split",
    "featurize",
    "preprocess",
    "train",
    "train_tdc",
}
_FEATURE_INPUT_NODES = {
    "featurize.none",
    "featurize.rdkit",
    "featurize.rdkit_labeled",
    "featurize.morgan",
    "use.curated_features",
}
_CHEMPROP_LIKE_MODELS = {"chemprop", "chemeleon"}
_FEATURE_INPUT_ALIASES = {
    "use.curated_features": "featurize.none",
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _normalize_feature_input(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return _FEATURE_INPUT_ALIASES.get(normalized, normalized)


def _feature_input_from_nodes(nodes: list[str]) -> str | None:
    lowered = [str(node).strip().lower() for node in nodes]
    if "featurize.morgan" in lowered:
        return "featurize.morgan"
    if "featurize.rdkit_labeled" in lowered:
        return "featurize.rdkit_labeled"
    if "featurize.rdkit" in lowered:
        return "featurize.rdkit"
    if "featurize.none" in lowered or "use.curated_features" in lowered:
        return "featurize.none"
    return None


def collect_config_issues(config: dict[str, Any], nodes: list[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    blocks_present = set(config.keys())
    configless_blocks = {
        block for node, block in _CONFIGLESS_NODE_TO_BLOCK.items() if node in nodes
    }
    blocks_required_by_configurable_nodes = {
        block
        for node, block in _NODE_TO_BLOCK.items()
        if node in nodes and node not in _CONFIGLESS_NODE_TO_BLOCK
    }

    allowed_blocks = set(_ALWAYS_ALLOWED_BLOCKS)
    for node in nodes:
        block = _NODE_TO_BLOCK.get(node)
        if block:
            allowed_blocks.add(block)

    # Top-level block validity.
    for block in blocks_present:
        if block == "model":
            issues.append(
                ValidationIssue(
                    code="CFG_LEGACY_MODEL_BLOCK_FORBIDDEN",
                    path="model",
                    message="Top-level model block is no longer supported.",
                    hint="Move model settings to train.model.*",
                )
            )
            continue
        if block not in _KNOWN_TOP_LEVEL_BLOCKS:
            issues.append(
                ValidationIssue(
                    code="CFG_UNKNOWN_TOP_LEVEL_BLOCK",
                    path=block,
                    message="Unknown top-level config block.",
                )
            )
            continue
        if block not in allowed_blocks:
            if block in configless_blocks:
                continue
            issues.append(
                ValidationIssue(
                    code="CFG_BLOCK_NOT_ALLOWED_FOR_PIPELINE",
                    path=block,
                    message=f"Block '{block}' is present but no corresponding node is in pipeline.nodes.",
                )
            )

    # Configless-node specific checks.
    for node, block in _CONFIGLESS_NODE_TO_BLOCK.items():
        if (
            node in nodes
            and block in blocks_present
            and block not in blocks_required_by_configurable_nodes
        ):
            issues.append(
                ValidationIssue(
                    code="CFG_CONFIGLESS_NODE_HAS_BLOCK",
                    path=block,
                    message=(
                        f"Node {node} is configless and does not accept a top-level {block} block."
                    ),
                )
            )

    # Required blocks for specific nodes.
    if "train" in nodes and "train" not in blocks_present:
        issues.append(
            ValidationIssue(
                code="CFG_MISSING_BLOCK_FOR_NODE",
                path="train",
                message="Pipeline contains train node but train block is missing.",
            )
        )
    if "train.tdc" in nodes and "train_tdc" not in blocks_present:
        issues.append(
            ValidationIssue(
                code="CFG_MISSING_BLOCK_FOR_NODE",
                path="train_tdc",
                message="Pipeline contains train.tdc node but train_tdc block is missing.",
            )
        )

    if "train" in blocks_present and not isinstance(config.get("train"), dict):
        issues.append(
            ValidationIssue(
                code="CFG_INVALID_TRAIN_SCHEMA",
                path="train",
                message="train block must be a mapping/object.",
            )
        )
    train_cfg = _as_dict(config.get("train"))
    if "train" in nodes:
        model_value = train_cfg.get("model", None)
        if model_value is None:
            issues.append(
                ValidationIssue(
                    code="CFG_MISSING_TRAIN_MODEL",
                    path="train.model",
                    message="train.model block is required for train node.",
                )
            )
        else:
            model_cfg = _as_dict(model_value)
            if not model_cfg.get("type"):
                issues.append(
                    ValidationIssue(
                        code="CFG_MISSING_TRAIN_MODEL_TYPE",
                        path="train.model.type",
                        message="train.model.type is required.",
                    )
                )

    pipeline_cfg = _as_dict(config.get("pipeline"))
    configured_feature_input = _normalize_feature_input(pipeline_cfg.get("feature_input", ""))
    has_explicit_feature_input = any(node in _FEATURE_INPUT_NODES for node in nodes)
    explicit_feature_input = _feature_input_from_nodes(nodes)
    model_type = str(_as_dict(train_cfg.get("model")).get("type", "")).strip().lower()
    model_cfg = _as_dict(train_cfg.get("model"))
    foundation_mode = str(model_cfg.get("foundation", "none")).strip().lower() or "none"
    foundation_checkpoint = str(model_cfg.get("foundation_checkpoint", "")).strip()
    preprocess_cfg = _as_dict(config.get("preprocess"))
    preprocess_scaler = str(preprocess_cfg.get("scaler", "robust")).strip().lower() or "robust"
    chemprop_like_noop_preprocess = (
        model_type in _CHEMPROP_LIKE_MODELS
        and configured_feature_input == "smiles_native"
        and preprocess_scaler == "none"
        and "preprocess.features" in nodes
        and "select.features" not in nodes
    )
    if configured_feature_input and explicit_feature_input and configured_feature_input != explicit_feature_input:
        issues.append(
            ValidationIssue(
                code="CFG_PIPELINE_FEATURE_INPUT_MISMATCH",
                path="pipeline.feature_input",
                message=(
                    f"pipeline.feature_input={configured_feature_input!r} does not match the explicit feature node "
                    f"{explicit_feature_input!r}."
                ),
            )
        )

    if (
        any(node in nodes for node in ("preprocess.features", "select.features"))
        and not has_explicit_feature_input
        and not chemprop_like_noop_preprocess
    ):
        issues.append(
            ValidationIssue(
                code="CFG_FEATURE_INPUT_NODE_REQUIRED",
                path="pipeline.nodes",
                message=(
                    "preprocess.features/select.features require an explicit feature input node "
                    "(featurize.none/featurize.rdkit/featurize.rdkit_labeled/featurize.morgan), "
                    "except for the Chemprop/CheMeleon no-op preprocess branch."
                ),
            )
        )
    if "train" in nodes and not has_explicit_feature_input:
        if model_type and model_type not in _CHEMPROP_LIKE_MODELS:
            issues.append(
                ValidationIssue(
                    code="CFG_FEATURE_INPUT_NODE_REQUIRED",
                    path="pipeline.nodes",
                    message=(
                        "train for non-SMILES-native models requires an explicit feature input node "
                        "(featurize.none/featurize.rdkit/featurize.rdkit_labeled/featurize.morgan)."
                    ),
                )
            )
    if model_type in _CHEMPROP_LIKE_MODELS:
        resolved_feature_input = explicit_feature_input or configured_feature_input or ""
        if resolved_feature_input != "smiles_native":
            issues.append(
                ValidationIssue(
                    code="CFG_CHEMPROP_FEATURE_INPUT_UNSUPPORTED",
                    path="pipeline.feature_input",
                    message=(
                        "Chemprop/CheMeleon are SMILES-native and must use pipeline.feature_input=smiles_native "
                        "with no explicit tabular featurizer node."
                    ),
                )
            )
        if "select.features" in nodes or ("preprocess.features" in nodes and preprocess_scaler != "none"):
            issues.append(
                ValidationIssue(
                    code="CFG_CHEMPROP_PREPROCESS_UNSUPPORTED",
                    path="pipeline.nodes",
                    message=(
                        "Chemprop/CheMeleon only support the no-op preprocess branch "
                        "(preprocess.scaler=none, no select.features)."
                    ),
                )
            )
        if model_type == "chemeleon":
            foundation_mode = "chemeleon"
        if foundation_mode == "chemeleon" and not foundation_checkpoint:
            issues.append(
                ValidationIssue(
                    code="CFG_CHEMELEON_CHECKPOINT_REQUIRED",
                    path="train.model.foundation_checkpoint",
                    message="CheMeleon runs require train.model.foundation_checkpoint.",
                )
            )

    if "train_tdc" in blocks_present and not isinstance(config.get("train_tdc"), dict):
        issues.append(
            ValidationIssue(
                code="CFG_INVALID_TRAIN_SCHEMA",
                path="train_tdc",
                message="train_tdc block must be a mapping/object.",
            )
        )
    train_tdc_cfg = _as_dict(config.get("train_tdc"))
    if "train.tdc" in nodes:
        model_value = train_tdc_cfg.get("model", None)
        if model_value is None:
            issues.append(
                ValidationIssue(
                    code="CFG_MISSING_TRAIN_MODEL",
                    path="train_tdc.model",
                    message="train_tdc.model block is required for train.tdc node.",
                )
            )
        else:
            model_cfg = _as_dict(model_value)
            if not model_cfg.get("type"):
                issues.append(
                    ValidationIssue(
                        code="CFG_MISSING_TRAIN_MODEL_TYPE",
                        path="train_tdc.model.type",
                        message="train_tdc.model.type is required.",
                    )
                )

    # Legacy preprocess keys that leaked into other nodes.
    if "keep_all_columns" in preprocess_cfg:
        issues.append(
            ValidationIssue(
                code="CFG_LEGACY_PREPROCESS_KEY_FORBIDDEN",
                path="preprocess.keep_all_columns",
                message="preprocess.keep_all_columns is no longer supported.",
                hint="Move to curate.keep_all_columns",
            )
        )
    if "exclude_columns" in preprocess_cfg:
        issues.append(
            ValidationIssue(
                code="CFG_LEGACY_PREPROCESS_KEY_FORBIDDEN",
                path="preprocess.exclude_columns",
                message="preprocess.exclude_columns is no longer supported.",
                hint="Move to train.features.exclude_columns",
            )
        )
    if "scaler" in preprocess_cfg:
        scaler_value = str(preprocess_cfg.get("scaler", "")).strip().lower()
        if scaler_value not in {"robust", "standard", "minmax", "none"}:
            issues.append(
                ValidationIssue(
                    code="CFG_PREPROCESS_SCALER_INVALID",
                    path="preprocess.scaler",
                    message="preprocess.scaler must be one of: robust, standard, minmax, none.",
                )
            )

    return issues


def validate_config_strict(config: dict[str, Any], nodes: list[str]) -> None:
    issues = collect_config_issues(config, nodes)
    if issues:
        raise ConfigValidationError(issues)
