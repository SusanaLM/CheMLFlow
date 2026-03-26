from __future__ import annotations

import pytest

from utilities.config_validation import ConfigValidationError, collect_config_issues, validate_config_strict


def _base_config(nodes: list[str]) -> dict:
    return {
        "global": {
            "pipeline_type": "qm9",
            "base_dir": "data/qm9",
            "thresholds": {"active": 1, "inactive": 2},
        },
        "pipeline": {"nodes": nodes},
    }


def test_strict_rejects_unknown_top_level_block() -> None:
    cfg = _base_config(["train"])
    cfg["train"] = {"model": {"type": "decision_tree"}}
    cfg["mystery"] = {"foo": "bar"}
    with pytest.raises(ConfigValidationError, match="CFG_UNKNOWN_TOP_LEVEL_BLOCK"):
        validate_config_strict(cfg, ["train"])


def test_strict_rejects_block_not_in_pipeline() -> None:
    cfg = _base_config(["train"])
    cfg["train"] = {"model": {"type": "decision_tree"}}
    cfg["split"] = {"strategy": "random"}
    with pytest.raises(ConfigValidationError, match="CFG_BLOCK_NOT_ALLOWED_FOR_PIPELINE"):
        validate_config_strict(cfg, ["train"])


def test_strict_rejects_configless_featurize_none_block() -> None:
    cfg = _base_config(["featurize.none"])
    cfg["featurize"] = {"radius": 2}
    with pytest.raises(ConfigValidationError, match="CFG_CONFIGLESS_NODE_HAS_BLOCK"):
        validate_config_strict(cfg, ["featurize.none"])


def test_configless_featurize_none_allows_shared_featurize_block_with_morgan() -> None:
    cfg = _base_config(["featurize.none", "featurize.morgan"])
    cfg["featurize"] = {"radius": 2, "n_bits": 1024}
    validate_config_strict(cfg, ["featurize.none", "featurize.morgan"])


def test_strict_allows_legacy_use_curated_features_alias_node() -> None:
    cfg = _base_config(["use.curated_features"])
    validate_config_strict(cfg, ["use.curated_features"])


def test_strict_requires_train_model_type() -> None:
    cfg = _base_config(["train"])
    cfg["train"] = {"model": {}}
    with pytest.raises(ConfigValidationError, match="CFG_MISSING_TRAIN_MODEL_TYPE"):
        validate_config_strict(cfg, ["train"])


def test_strict_rejects_legacy_preprocess_keys() -> None:
    cfg = _base_config(["preprocess.features"])
    cfg["preprocess"] = {"keep_all_columns": True, "exclude_columns": ["A"]}
    issues = collect_config_issues(cfg, ["preprocess.features"])
    codes = {issue.code for issue in issues}
    assert "CFG_LEGACY_PREPROCESS_KEY_FORBIDDEN" in codes


def test_strict_requires_feature_input_node_for_preprocess() -> None:
    cfg = _base_config(["split", "preprocess.features"])
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {}
    issues = collect_config_issues(cfg, ["split", "preprocess.features"])
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NODE_REQUIRED" in codes


def test_strict_requires_feature_input_node_for_non_chemprop_train() -> None:
    cfg = _base_config(["split", "train"])
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "random_forest"}}
    issues = collect_config_issues(cfg, ["split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NODE_REQUIRED" in codes


@pytest.mark.parametrize("model_type", ["chemprop", "chemeleon"])
def test_strict_allows_train_without_feature_node_for_chemprop_like(model_type: str) -> None:
    cfg = _base_config(["split", "train"])
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": model_type}}
    issues = collect_config_issues(cfg, ["split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NODE_REQUIRED" not in codes
    assert "CFG_CHEMPROP_FEATURE_INPUT_UNSUPPORTED" not in codes


@pytest.mark.parametrize("model_type", ["chemprop", "chemeleon"])
def test_strict_allows_noop_preprocess_without_feature_node_for_chemprop_like(model_type: str) -> None:
    cfg = _base_config(["split", "preprocess.features", "train"])
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {"scaler": "none"}
    cfg["train"] = {"model": {"type": model_type}}

    issues = collect_config_issues(cfg, ["split", "preprocess.features", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_FEATURE_INPUT_NODE_REQUIRED" not in codes


def test_strict_rejects_invalid_preprocess_scaler() -> None:
    cfg = _base_config(["split", "featurize.rdkit", "preprocess.features", "train"])
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {"scaler": "banana"}
    cfg["train"] = {"model": {"type": "random_forest"}}

    issues = collect_config_issues(cfg, ["split", "featurize.rdkit", "preprocess.features", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_PREPROCESS_SCALER_INVALID" in codes


def test_strict_rejects_chemprop_with_explicit_tabular_featurizer() -> None:
    cfg = _base_config(["split", "featurize.rdkit", "preprocess.features", "train"])
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {"scaler": "none"}
    cfg["train"] = {"model": {"type": "chemprop"}}

    issues = collect_config_issues(cfg, ["split", "featurize.rdkit", "preprocess.features", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_PIPELINE_FEATURE_INPUT_MISMATCH" in codes
    assert "CFG_CHEMPROP_FEATURE_INPUT_UNSUPPORTED" in codes


def test_strict_rejects_chemprop_like_select_features_branch() -> None:
    cfg = _base_config(["split", "preprocess.features", "select.features", "train"])
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["split"] = {"strategy": "random"}
    cfg["preprocess"] = {"scaler": "none"}
    cfg["train"] = {"model": {"type": "chemprop"}}

    issues = collect_config_issues(cfg, ["split", "preprocess.features", "select.features", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_CHEMPROP_PREPROCESS_UNSUPPORTED" in codes


def test_strict_requires_chemeleon_checkpoint() -> None:
    cfg = _base_config(["split", "train"])
    cfg["pipeline"]["feature_input"] = "smiles_native"
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "chemeleon"}}

    issues = collect_config_issues(cfg, ["split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_CHEMELEON_CHECKPOINT_REQUIRED" in codes


def test_strict_requires_smiles_native_for_chemprop_like_models() -> None:
    cfg = _base_config(["split", "train"])
    cfg["split"] = {"strategy": "random"}
    cfg["train"] = {"model": {"type": "chemprop"}}

    issues = collect_config_issues(cfg, ["split", "train"])
    codes = {issue.code for issue in issues}
    assert "CFG_CHEMPROP_FEATURE_INPUT_UNSUPPORTED" in codes
