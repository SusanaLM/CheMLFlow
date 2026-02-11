# an example workflow that uses ChemBL bioactivty data
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

import yaml
import pandas as pd
import joblib

from contracts import (
    ANALYZE_EDA_INPUT_2CLASS_CONTRACT,
    ANALYZE_EDA_INPUT_3CLASS_CONTRACT,
    ANALYZE_EDA_OUTPUT_CONTRACT,
    ANALYZE_STATS_INPUT_CONTRACT,
    ANALYZE_STATS_OUTPUT_CONTRACT,
    CURATE_INPUT_CONTRACT,
    CURATE_OUTPUT_CONTRACT,
    FEATURIZE_LIPINSKI_INPUT_CONTRACT,
    FEATURIZE_LIPINSKI_OUTPUT_CONTRACT,
    FEATURIZE_RDKIT_INPUT_CONTRACT,
    FEATURIZE_RDKIT_OUTPUT_CONTRACT,
    FEATURIZE_RDKIT_LABELED_INPUT_CONTRACT,
    FEATURIZE_RDKIT_LABELED_OUTPUT_LABELS_CONTRACT,
    FEATURIZE_MORGAN_INPUT_CONTRACT,
    FEATURIZE_MORGAN_OUTPUT_CONTRACT,
    PREPROCESS_FEATURES_INPUT_CONTRACT,
    PREPROCESS_FEATURES_OUTPUT_CONTRACT,
    PREPROCESS_ARTIFACTS_CONTRACT,
    SELECT_FEATURES_INPUT_FEATURES_CONTRACT,
    SELECT_FEATURES_OUTPUT_CONTRACT,
    SELECT_FEATURES_LIST_CONTRACT,
    EXPLAIN_INPUT_MODEL_CONTRACT,
    EXPLAIN_OUTPUT_CONTRACT,
    GET_DATA_INPUT_CONTRACT,
    GET_DATA_OUTPUT_CONTRACT,
    DESCRIPTORS_CONTRACT,
    IC50_INPUT_CONTRACT,
    LABEL_IC50_INPUT_CONTRACT,
    LABEL_IC50_OUTPUT_2CLASS_CONTRACT,
    LABEL_IC50_OUTPUT_3CLASS_CONTRACT,
    LIPINSKI_CONTRACT,
    MODEL_LABELS_CONTRACT,
    PIC50_2CLASS_CONTRACT,
    PIC50_3CLASS_CONTRACT,
    PREPROCESSED_CONTRACT,
    TRAIN_INPUT_FEATURES_CONTRACT,
    TRAIN_INPUT_LABELS_CONTRACT,
    TRAIN_OUTPUT_CONTRACT,
    ContractSpec,
    bind_output_path,
    make_target_column_contract,
    validate_contract,
)
from MLModels import data_preprocessing
from MLModels import train_models
from utilities import splitters

def build_paths(base_dir: str) -> dict[str, str]:
    return {
        "raw": os.path.join(base_dir, "raw.csv"),
        "raw_sample": os.path.join(base_dir, "raw_sample.csv"),
        "preprocessed": os.path.join(base_dir, "preprocessed.csv"),
        "curated": os.path.join(base_dir, "curated.csv"),
        "curated_labeled": os.path.join(base_dir, "curated_labeled.csv"),
        "curated_smiles": os.path.join(base_dir, "curated_smiles.csv"),
        "lipinski": os.path.join(base_dir, "lipinski_results.csv"),
        "pic50_3class": os.path.join(base_dir, "bioactivity_3class_pIC50.csv"),
        "pic50_2class": os.path.join(base_dir, "bioactivity_2class_pIC50.csv"),
        "rdkit_descriptors": os.path.join(base_dir, "rdkit_descriptors.csv"),
        "rdkit_labeled": os.path.join(base_dir, "rdkit_descriptors_labeled.csv"),
        "morgan_fingerprints": os.path.join(base_dir, "morgan_fingerprints.csv"),
        "morgan_labeled": os.path.join(base_dir, "morgan_fingerprints_labeled.csv"),
        "morgan_meta": os.path.join(base_dir, "morgan_meta.json"),
        "preprocessed_features": os.path.join(base_dir, "preprocessed_features.csv"),
        "preprocessed_labels": os.path.join(base_dir, "preprocessed_labels.csv"),
        "selected_features": os.path.join(base_dir, "selected_features.csv"),
        "selected_features_list": os.path.join(base_dir, "selected_features.txt"),
        "preprocess_artifacts": os.path.join(base_dir, "preprocess_artifacts.joblib"),
        "split_dir": os.path.join(base_dir, "splits"),
    }


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def sample_csv(input_path: str, output_path: str, max_rows: int) -> None:
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for idx, row in enumerate(reader):
            writer.writerow(row)
            if idx >= max_rows:
                break


def resolve_run_dir(config: dict) -> str:
    global_config = config.get("global", {})
    run_dir = global_config.get("run_dir")
    if run_dir:
        return run_dir
    runs = global_config.get("runs", {})
    if runs.get("enabled"):
        run_id = runs.get("id")
        if not run_id:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return os.path.join("runs", run_id)
    return "results"

def _configure_logging(run_dir: str) -> None:
    """Configure root logger to emit to stdout and to <run_dir>/run.log.

    This must be defined before any pipeline runner calls it.
    """
    log_path = os.path.join(run_dir, "run.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path
        for h in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def run_node_get_data(context: dict) -> None:
    validate_contract(
        bind_output_path(GET_DATA_INPUT_CONTRACT, context["config_path"]),
        warn_only=False,
    )
    script_path = os.path.join("GetData", "get_data.py")
    output_file = context["paths"]["raw"]
    subprocess.run([sys.executable, script_path, output_file, "--config", context["config_path"]])
    validate_contract(bind_output_path(GET_DATA_OUTPUT_CONTRACT, output_file), warn_only=True)


def run_node_curate(context: dict) -> None:
    validate_contract(
        bind_output_path(CURATE_INPUT_CONTRACT, context["paths"]["raw"]),
        warn_only=False,
    )
    raw_data_file = context["paths"]["raw"]
    pipeline_type = context["pipeline_type"]
    get_data_config = context.get("get_data_config", {})
    if pipeline_type == "qm9":
        max_rows = get_data_config.get("max_rows")
        if max_rows:
            sampled_path = context["paths"]["raw_sample"]
            sample_csv(raw_data_file, sampled_path, int(max_rows))
            raw_data_file = sampled_path

    curate_config = context["curate_config"]
    target_column = context["target_column"]
    properties = curate_config.get("properties")
    if not properties:
        if pipeline_type == "qm9" or context.get("task_type") == "classification":
            properties = target_column
        else:
            properties = "standard_value"

    script_path = os.path.join("utilities", "prepareActivityData.py")
    preprocessed_file = context["paths"]["preprocessed"]
    curated_file = context["paths"]["curated"]
    curated_smiles_output = context["paths"]["curated_smiles"]
    cmd = [
        sys.executable,
        script_path,
        raw_data_file,
        preprocessed_file,
        curated_file,
        curated_smiles_output,
        "--active_threshold",
        str(context["active_threshold"]),
        "--inactive_threshold",
        str(context["inactive_threshold"]),
        "--properties",
        properties,
    ]
    smiles_column = curate_config.get("smiles_column")
    if smiles_column:
        cmd.extend(["--smiles_column", smiles_column])
    dedupe_strategy = curate_config.get("dedupe_strategy")
    if dedupe_strategy:
        cmd.extend(["--dedupe_strategy", dedupe_strategy])
    label_column = curate_config.get("label_column")
    if not label_column and context.get("task_type") == "classification":
        label_column = context["target_column"]
    if label_column:
        cmd.extend(["--label_column", label_column])
    if curate_config.get("require_neutral_charge"):
        cmd.append("--require_neutral_charge")
    if "prefer_largest_fragment" in curate_config:
        if curate_config.get("prefer_largest_fragment"):
            cmd.append("--prefer_largest_fragment")
        else:
            cmd.append("--no_prefer_largest_fragment")
    if context["keep_all_columns"]:
        cmd.append("--keep_all_columns")
    subprocess.run(cmd)
    validate_contract(bind_output_path(PREPROCESSED_CONTRACT, preprocessed_file), warn_only=True)
    validate_contract(bind_output_path(CURATE_OUTPUT_CONTRACT, curated_file), warn_only=True)
    # Establish the canonical dataset path for downstream nodes.
    context["curated_path"] = curated_file


def run_node_use_curated_features(context: dict) -> None:
    curated_path = context.get("curated_path", context["paths"]["curated"])
    validate_contract(
        bind_output_path(
            make_target_column_contract(
                name="use_curated_features_input",
                target_column=context["target_column"],
                output_path=curated_path,
            ),
            curated_path,
        ),
        warn_only=False,
    )
    context["feature_matrix"] = curated_path
    context["labels_matrix"] = curated_path


def run_node_featurize_lipinski(context: dict) -> None:
    validate_contract(
        bind_output_path(
            FEATURIZE_LIPINSKI_INPUT_CONTRACT,
            context.get("curated_path", context["paths"]["curated"]),
        ),
        warn_only=False,
    )
    script_path = os.path.join("utilities", "Lipinski_rules.py")
    smiles_file = context.get("curated_path", context["paths"]["curated"])
    output_file = context["paths"]["lipinski"]
    subprocess.run([sys.executable, script_path, smiles_file, output_file])
    validate_contract(
        bind_output_path(FEATURIZE_LIPINSKI_OUTPUT_CONTRACT, output_file),
        warn_only=True,
    )


def run_node_label_ic50(context: dict) -> None:
    validate_contract(
        bind_output_path(LABEL_IC50_INPUT_CONTRACT, context["paths"]["lipinski"]),
        warn_only=False,
    )
    script_path = os.path.join("utilities", "IC50_pIC50.py")
    input_file = context["paths"]["lipinski"]
    output_file_3class = context["paths"]["pic50_3class"]
    output_file_2class = context["paths"]["pic50_2class"]
    subprocess.run(
        [sys.executable, script_path, input_file, output_file_3class, output_file_2class]
    )
    validate_contract(
        bind_output_path(LABEL_IC50_OUTPUT_3CLASS_CONTRACT, output_file_3class),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(LABEL_IC50_OUTPUT_2CLASS_CONTRACT, output_file_2class),
        warn_only=True,
    )
    context["labels_matrix"] = output_file_3class
    # Treat the labeled pIC50 output as the canonical dataset for downstream nodes.
    context["curated_path"] = output_file_3class


def run_node_analyze_stats(context: dict) -> None:
    validate_contract(
        bind_output_path(ANALYZE_STATS_INPUT_CONTRACT, context["paths"]["pic50_2class"]),
        warn_only=False,
    )
    script_path = os.path.join("utilities", "stat_tests.py")
    input_file = context["paths"]["pic50_2class"]
    output_dir = context["base_dir"]
    test_type = ["mannwhitney", "ttest", "chi2"]
    descriptor = context["target_column"]
    for test in test_type:
        subprocess.run([sys.executable, script_path, input_file, output_dir, test, descriptor])
    validate_contract(
        bind_output_path(ANALYZE_STATS_OUTPUT_CONTRACT, output_dir),
        warn_only=True,
    )


def run_node_analyze_eda(context: dict) -> None:
    validate_contract(
        bind_output_path(ANALYZE_EDA_INPUT_2CLASS_CONTRACT, context["paths"]["pic50_2class"]),
        warn_only=False,
    )
    validate_contract(
        bind_output_path(ANALYZE_EDA_INPUT_3CLASS_CONTRACT, context["paths"]["pic50_3class"]),
        warn_only=False,
    )
    script_path = os.path.join("utilities", "EDA.py")
    input_file_2class = context["paths"]["pic50_2class"]
    input_file_3class = context["paths"]["pic50_3class"]
    output_dir = context["base_dir"]
    subprocess.run(
        [sys.executable, script_path, input_file_2class, input_file_3class, output_dir]
    )
    validate_contract(
        bind_output_path(ANALYZE_EDA_OUTPUT_CONTRACT, output_dir),
        warn_only=True,
    )


def run_node_featurize_rdkit(context: dict) -> None:
    validate_contract(
        bind_output_path(
            FEATURIZE_RDKIT_INPUT_CONTRACT,
            context.get("curated_path", context["paths"]["curated"]),
        ),
        warn_only=False,
    )
    # Use labeled descriptors so features and target live in a single canonical file.
    script_path = os.path.join("GenDescriptors", "RDKit_descriptors_labeled.py")
    input_file = context.get("curated_path", context["paths"]["curated"])
    output_file = context["paths"]["rdkit_descriptors"]
    labeled_output_file = context["paths"]["rdkit_labeled"]
    subprocess.run(
        [
            sys.executable,
            script_path,
            input_file,
            output_file,
            "--labeled-output-file",
            labeled_output_file,
            "--property-columns",
            context["target_column"],
        ]
    )
    validate_contract(
        bind_output_path(FEATURIZE_RDKIT_OUTPUT_CONTRACT, output_file),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(FEATURIZE_RDKIT_LABELED_OUTPUT_LABELS_CONTRACT, labeled_output_file),
        warn_only=True,
    )
    context["feature_matrix"] = labeled_output_file
    context["labels_matrix"] = labeled_output_file


def run_node_featurize_rdkit_labeled(context: dict) -> None:
    # Backward-compatible alias: use the same implementation as featurize.rdkit.
    run_node_featurize_rdkit(context)


def run_node_featurize_morgan(context: dict) -> None:
    validate_contract(
        bind_output_path(
            FEATURIZE_MORGAN_INPUT_CONTRACT,
            context.get("curated_path", context["paths"]["curated"]),
        ),
        warn_only=False,
    )
    script_path = os.path.join("GenDescriptors", "Morgan_fingerprints.py")
    input_file = context.get("curated_path", context["paths"]["curated"])
    output_file = context["paths"]["morgan_fingerprints"]
    labeled_output = context["paths"]["morgan_labeled"]
    featurize_config = context.get("featurize_config", {})
    radius = featurize_config.get("radius", 2)
    n_bits = featurize_config.get("n_bits", 2048)
    cmd = [
        sys.executable,
        script_path,
        input_file,
        output_file,
        "--radius",
        str(radius),
        "--n_bits",
        str(n_bits),
        "--labeled-output-file",
        labeled_output,
        "--property-columns",
        context["target_column"],
    ]
    subprocess.run(cmd)
    with open(context["paths"]["morgan_meta"], "w", encoding="utf-8") as meta_out:
        json.dump(
            {"radius": radius, "n_bits": n_bits},
            meta_out,
            indent=2,
        )
    validate_contract(
        bind_output_path(FEATURIZE_MORGAN_OUTPUT_CONTRACT, output_file),
        warn_only=True,
    )
    validate_contract(
        make_target_column_contract(
            name="featurize_morgan_labeled_output",
            target_column=context["target_column"],
            output_path=labeled_output,
        ),
        warn_only=True,
    )
    context["feature_matrix"] = labeled_output
    context["labels_matrix"] = labeled_output


def run_node_label_normalize(context: dict) -> None:
    label_config = context.get("label_config", {})
    source_column = label_config.get("source_column")
    target_column = label_config.get("target_column", context["target_column"])
    positive = label_config.get("positive")
    negative = label_config.get("negative")
    drop_unmapped = label_config.get("drop_unmapped", True)

    def _ensure_list(value):
        if value is None:
            return None
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return [str(value)]

    positive_list = _ensure_list(positive)
    negative_list = _ensure_list(negative)

    if not source_column or not positive_list or not negative_list:
        raise ValueError("label.normalize requires source_column, positive, and negative label lists.")

    script_path = os.path.join("utilities", "label_normalize.py")
    input_file = context.get("curated_path", context["paths"]["curated"])
    output_file = context["paths"]["curated_labeled"]
    cmd = [
        sys.executable,
        script_path,
        input_file,
        output_file,
        "--source-column",
        source_column,
        "--target-column",
        target_column,
        "--positive",
        ",".join(positive_list),
        "--negative",
        ",".join(negative_list),
    ]
    if drop_unmapped:
        cmd.append("--drop-unmapped")
    subprocess.run(cmd, check=True)
    context["curated_path"] = output_file
    context["target_column"] = target_column


def run_node_split(context: dict) -> None:
    split_config = context.get("split_config", {})
    strategy = split_config.get("strategy", "random")
    test_size = split_config.get("test_size", 0.2)
    val_size = split_config.get("val_size", 0.1)
    random_state = split_config.get("random_state", 42)
    stratify = split_config.get("stratify", False)
    stratify_column = split_config.get("stratify_column")
    min_coverage = split_config.get("min_coverage")
    allow_missing_val = split_config.get("allow_missing_val", True)
    require_disjoint = split_config.get("require_disjoint", False)
    if stratify and not stratify_column:
        stratify_column = context.get("target_column")

    def _safe_token(value: str) -> str:
        out = []
        for ch in str(value):
            if ch.isalnum() or ch in {"_", "-"}:
                out.append(ch)
            else:
                out.append("_")
        return "".join(out)

    def _fmt_float(value) -> str:
        # 0.2 -> 0p2 for stable filenames
        try:
            return str(float(value)).replace(".", "p")
        except Exception:
            return _safe_token(str(value))

    curated_path = context.get("curated_path", context["paths"]["curated"])
    curated_df = pd.read_csv(curated_path)
    if stratify:
        if not stratify_column:
            raise ValueError("split.stratify=true requires split.stratify_column to be set.")
        if stratify_column not in curated_df.columns:
            raise ValueError(
                f"split.stratify_column={stratify_column!r} not found in curated data."
            )
        unique = curated_df[stratify_column].dropna().nunique()
        # sklearn stratify expects discrete labels; guard against accidental regression stratification.
        if unique > 20:
            raise ValueError(
                f"split.stratify_column={stratify_column!r} has {unique} unique values; "
                "stratification requires a low-cardinality label column (e.g., binary class)."
            )
    tdc_group = context.get("source", {}).get("group")
    tdc_name = context.get("source", {}).get("name")
    split_indices = splitters.build_split_indices(
        strategy=strategy,
        curated_df=curated_df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify_column=stratify_column,
        tdc_group=tdc_group,
        tdc_name=tdc_name,
    )
    normalized = {}
    for key, value in split_indices.items():
        lowered = key.lower()
        if lowered in {"valid", "val", "validation"}:
            normalized["val"] = value
        elif lowered in {"test"}:
            normalized["test"] = value
        elif lowered in {"train"}:
            normalized["train"] = value
        else:
            normalized[key] = value
    normalized.setdefault("val", [])
    os.makedirs(context["paths"]["split_dir"], exist_ok=True)
    split_filename_parts = [
        _safe_token(strategy),
        f"test{_fmt_float(test_size)}",
        f"val{_fmt_float(val_size)}",
        f"seed{_safe_token(random_state)}",
    ]
    if stratify and stratify_column:
        split_filename_parts.append(f"strat_{_safe_token(stratify_column)}")
    if strategy.startswith("tdc") and tdc_group and tdc_name:
        split_filename_parts.append(f"tdc_{_safe_token(tdc_group)}_{_safe_token(tdc_name)}")
    dataset_split_path = os.path.join(
        context["paths"]["split_dir"],
        "_".join(split_filename_parts) + ".json",
    )
    run_split_path = os.path.join(context["run_dir"], "split_indices.json")

    splitters.save_split_indices(normalized, dataset_split_path)
    splitters.save_split_indices(normalized, run_split_path)
    curated_count = len(curated_df)
    train_set = set(normalized.get("train", []))
    val_set = set(normalized.get("val", []))
    test_set = set(normalized.get("test", []))
    all_indices = set(train_set | val_set | test_set)
    if not all_indices:
        raise ValueError("Split mapping produced zero indices.")
    if max(all_indices) >= curated_count:
        raise ValueError("Split indices exceed curated dataset size.")
    if min_coverage is not None:
        coverage = len(all_indices) / max(1, curated_count)
        if coverage < min_coverage:
            raise ValueError(
                f"Split coverage {coverage:.2%} below min_coverage={min_coverage:.2%}. "
                "Check split mapping or strategy."
            )
    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlap:
        message = f"Split overlap detected across train/val/test: {len(overlap)} shared indices."
        if require_disjoint:
            raise ValueError(message)
        logging.warning(message)
    if not normalized.get("train"):
        raise ValueError("Split mapping produced empty train split.")
    if not normalized.get("test"):
        raise ValueError("Split mapping produced empty test split.")
    if val_size > 0 and not normalized.get("val"):
        if strategy.startswith("tdc") and allow_missing_val:
            pass
        else:
            raise ValueError("Split mapping produced empty validation split.")
    context["split_indices"] = normalized
    context["split_path"] = run_split_path
    context["dataset_split_path"] = dataset_split_path


def _resolve_feature_inputs(context: dict) -> tuple[str, str]:
    feature_matrix = context.get("feature_matrix")
    labels_matrix = context.get("labels_matrix")
    if feature_matrix and labels_matrix:
        return feature_matrix, labels_matrix
    if feature_matrix and not labels_matrix:
        return feature_matrix, context["paths"]["curated"]
    pipeline_type = context["pipeline_type"]
    if pipeline_type == "qm9":
        return context["paths"]["rdkit_labeled"], context["paths"]["rdkit_labeled"]
    return context["paths"]["rdkit_descriptors"], context["paths"]["pic50_3class"]


def _preprocess_params(context: dict) -> tuple[float, float, tuple[float, float], int, int]:
    preprocess_config = context.get("preprocess_config", {})
    variance_threshold = preprocess_config.get("variance_threshold", 0.8 * (1 - 0.8))
    corr_threshold = preprocess_config.get("corr_threshold", 0.95)
    clip_range = preprocess_config.get("clip_range", (-1e10, 1e10))
    if isinstance(clip_range, list):
        clip_range = tuple(clip_range)
    stable_k = preprocess_config.get("stable_features_k", 50)
    random_state = preprocess_config.get("random_state", 42)
    return variance_threshold, corr_threshold, clip_range, stable_k, random_state


def _resolve_split_partitions(
    context: dict,
    index: pd.Index,
) -> tuple[list[int], list[int], list[int]] | None:
    """
    Resolve train/val/test indices against a cleaned feature/label index.

    If split_indices exist in the context, we filter them down to rows that are
    still present after cleaning (NaN/duplicate dropping). If a split becomes
    empty after filtering, we raise instead of silently re-splitting elsewhere.
    """
    split_indices = context.get("split_indices")
    if not split_indices:
        return None

    train_idx = list(split_indices.get("train", []) or [])
    val_idx = list(split_indices.get("val", []) or [])
    test_idx = list(split_indices.get("test", []) or [])
    if not test_idx and val_idx:
        # Some sources only provide train/val; treat val as test downstream.
        test_idx, val_idx = val_idx, []

    orig_counts = (len(train_idx), len(val_idx), len(test_idx))
    available = set(index.tolist())
    train_idx = [i for i in train_idx if i in available]
    val_idx = [i for i in val_idx if i in available]
    test_idx = [i for i in test_idx if i in available]

    if not train_idx or not test_idx:
        raise ValueError(
            "split_indices did not align with cleaned features/labels (train/test empty after filtering). "
            "Do not re-split downstream: run the 'split' node after any row-filtering/cleaning, or ensure "
            f"your feature/label matrices preserve {data_preprocessing.ROW_INDEX_COL}. "
            f"(available_rows={len(available)} split_counts(train,val,test)={orig_counts} "
            f"filtered_counts(train,val,test)=({len(train_idx)},{len(val_idx)},{len(test_idx)}))"
        )
    return train_idx, val_idx, test_idx


def run_node_preprocess_features(context: dict) -> None:
    split_indices = context.get("split_indices")
    if not split_indices:
        raise ValueError(
            "preprocess.features requires split_indices from the split node. "
            "Add 'split' before 'preprocess.features' in pipeline.nodes."
        )

    features_file, labels_file = _resolve_feature_inputs(context)
    validate_contract(
        bind_output_path(PREPROCESS_FEATURES_INPUT_CONTRACT, features_file),
        warn_only=False,
    )
    validate_contract(
        bind_output_path(
            make_target_column_contract(
                name="preprocess_input_labels_dynamic",
                target_column=context["target_column"],
            ),
            labels_file,
        ),
        warn_only=False,
    )

    X_clean, y_clean = data_preprocessing.load_features_labels(
        features_file,
        labels_file,
        context["target_column"],
        context.get("categorical_features"),
    )
    data_preprocessing.verify_data_quality(X_clean, y_clean)

    partitions = _resolve_split_partitions(context, X_clean.index)
    assert partitions is not None  # split_indices was validated above
    train_idx, _, _ = partitions
    X_train = X_clean.loc[train_idx]

    variance_threshold, corr_threshold, clip_range, _, _ = _preprocess_params(context)
    preprocessor = data_preprocessing.fit_preprocessor(
        X_train,
        variance_threshold=variance_threshold,
        corr_threshold=corr_threshold,
        clip_range=clip_range,
    )
    X_preprocessed = data_preprocessing.transform_preprocessor(X_clean, preprocessor)
    X_preprocessed.index = X_clean.index
    y_aligned = y_clean

    preprocessed_features = context["paths"]["preprocessed_features"]
    preprocessed_labels = context["paths"]["preprocessed_labels"]
    X_preprocessed = X_preprocessed.copy()
    X_preprocessed[data_preprocessing.ROW_INDEX_COL] = X_preprocessed.index
    X_preprocessed.to_csv(preprocessed_features, index=False)
    y_values = y_aligned.values
    if hasattr(y_values, "ndim") and y_values.ndim > 1:
        y_values = y_values[:, 0]
    pd.DataFrame(
        {
            context["target_column"]: y_values,
            data_preprocessing.ROW_INDEX_COL: y_aligned.index,
        }
    ).to_csv(
        preprocessed_labels, index=False
    )
    joblib.dump(preprocessor, context["paths"]["preprocess_artifacts"])
    data_preprocessing.check_data_leakage(X_train, X_test)

    validate_contract(
        bind_output_path(PREPROCESS_FEATURES_OUTPUT_CONTRACT, preprocessed_features),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(
            make_target_column_contract(
                name="preprocess_labels_output_dynamic",
                target_column=context["target_column"],
            ),
            preprocessed_labels,
        ),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(PREPROCESS_ARTIFACTS_CONTRACT, context["paths"]["preprocess_artifacts"]),
        warn_only=True,
    )
    context["preprocessed_ready"] = True


def run_node_select_features(context: dict) -> None:
    split_indices = context.get("split_indices")
    if not split_indices:
        raise ValueError(
            "select.features requires split_indices from the split node. "
            "Add 'split' before 'select.features' in pipeline.nodes."
        )

    features_file = context["paths"]["preprocessed_features"]
    labels_file = context["paths"]["preprocessed_labels"]

    validate_contract(
        bind_output_path(SELECT_FEATURES_INPUT_FEATURES_CONTRACT, features_file),
        warn_only=False,
    )
    validate_contract(
        bind_output_path(
            make_target_column_contract(
                name="select_input_labels_dynamic",
                target_column=context["target_column"],
            ),
            labels_file,
        ),
        warn_only=False,
    )

    target_column = context["target_column"]
    X, y = data_preprocessing.load_features_labels(
        features_file,
        labels_file,
        target_column,
        context.get("categorical_features"),
    )

    partitions = _resolve_split_partitions(context, X.index)
    assert partitions is not None  # split_indices was validated above
    train_idx, _, test_idx = partitions
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]

    _, _, _, stable_k, random_state = _preprocess_params(context)
    X_selected_train = data_preprocessing.select_stable_features(
        X_train,
        y_train,
        random_state=random_state,
        k=stable_k,
        out_path=context["paths"]["selected_features_list"],
    )

    X_selected = X[X_selected_train.columns]
    data_preprocessing.check_data_leakage(X_train, X_test)

    selected_features = context["paths"]["selected_features"]
    X_selected_out = X_selected.copy()
    X_selected_out[data_preprocessing.ROW_INDEX_COL] = X_selected_out.index
    X_selected_out.to_csv(selected_features, index=False)

    validate_contract(
        bind_output_path(SELECT_FEATURES_OUTPUT_CONTRACT, selected_features),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(SELECT_FEATURES_LIST_CONTRACT, context["paths"]["selected_features_list"]),
        warn_only=True,
    )
    context["selected_ready"] = True

def run_node_train(context: dict) -> None:
    pipeline_type = context["pipeline_type"]
    output_dir = context["run_dir"]
    model_type = context["model_type"]
    target_column = context["target_column"]
    paths = context["paths"]
    task_type = context.get("task_type", "regression")
    model_config = context.get("model_config", {})
    split_indices = context.get("split_indices")
    if not split_indices:
        raise ValueError(
            "train requires split_indices from the split node. "
            "Add 'split' before 'train' in pipeline.nodes."
        )

    # Chemprop is a SMILES-native model; it does not use tabular descriptors.
    # For apples-to-apples benchmarking, we still rely on CheMLFlow's split_indices.
    if model_type == "chemprop":
        curated_path = context.get("curated_path", paths["curated"])
        curated_df = pd.read_csv(curated_path)
        if not split_indices.get("val"):
            raise ValueError(
                "chemprop training requires an explicit validation split from the split node. "
                "Set split.val_size > 0."
            )

        _, train_result = train_models.train_chemprop_model(
            curated_df=curated_df,
            target_column=target_column,
            split_indices=split_indices,
            output_dir=output_dir,
            random_state=int(context.get("split_config", {}).get("random_state", 42)),
            task_type=task_type,
            model_config=model_config,
        )
        context["trained_model_path"] = train_result.model_path

        validate_contract(
            bind_output_path(TRAIN_OUTPUT_CONTRACT, output_dir),
            warn_only=True,
        )
        return

    use_selected = context.get("selected_ready", False)
    use_preprocessed = context.get("preprocessed_ready", False)
    skip_preprocess = use_preprocessed
    skip_feature_selection = use_selected
    skip_quality_checks = use_selected

    features_file = paths["rdkit_descriptors"]
    labels_file = paths["pic50_3class"]
    if pipeline_type == "qm9":
        features_file = paths["rdkit_labeled"]
        labels_file = paths["rdkit_labeled"]
    if context.get("feature_matrix"):
        features_file = context["feature_matrix"]
    if context.get("labels_matrix"):
        labels_file = context["labels_matrix"]
    if use_selected:
        features_file = paths["selected_features"]
    elif use_preprocessed:
        features_file = paths["preprocessed_features"]
    if use_preprocessed and os.path.exists(paths["preprocessed_labels"]):
        labels_file = paths["preprocessed_labels"]

    validate_contract(
        bind_output_path(TRAIN_INPUT_FEATURES_CONTRACT, features_file),
        warn_only=False,
    )
    validate_contract(
        bind_output_path(TRAIN_INPUT_LABELS_CONTRACT, labels_file),
        warn_only=False,
    )
    validate_contract(
        make_target_column_contract(
            name="train_input_labels_dynamic",
            target_column=target_column,
            output_path=labels_file,
        ),
        warn_only=False,
    )

    X, y = data_preprocessing.load_features_labels(
        features_file,
        labels_file,
        target_column,
        context.get("categorical_features"),
    )
    if isinstance(y, pd.DataFrame):
        y = data_preprocessing.select_target_series(y, target_column)
    if not skip_quality_checks:
        data_preprocessing.verify_data_quality(X, y)

    random_state = context.get("preprocess_config", {}).get("random_state", 42)
    cv_folds = context.get("model_config", {}).get("cv_folds", 5)
    search_iters = context.get("model_config", {}).get("search_iters", 100)
    X_val = None
    y_val = None
    partitions = _resolve_split_partitions(context, X.index)
    assert partitions is not None  # split_indices was validated above
    train_idx, val_idx, test_idx = partitions

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    if val_idx:
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    if isinstance(y_val, pd.DataFrame):
        y_val = y_val.iloc[:, 0]
    if not skip_quality_checks:
        data_preprocessing.check_data_leakage(X_train, X_test)

    estimator, train_result = train_models.train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type,
        output_dir,
        random_state=random_state,
        cv_folds=cv_folds,
        search_iters=search_iters,
        use_hpo=model_config.get("use_hpo", False),
        hpo_trials=model_config.get("hpo_trials", 30),
        patience=model_config.get("patience", 20),
        task_type=task_type,
        model_config=model_config,
        X_val=X_val,
        y_val=y_val,
    )
    context["trained_model_path"] = train_result.model_path

    validate_contract(
        bind_output_path(TRAIN_OUTPUT_CONTRACT, output_dir),
        warn_only=True,
    )


def run_node_explain(context: dict) -> None:
    output_dir = context["run_dir"]
    model_type = context["model_type"]
    target_column = context["target_column"]
    paths = context["paths"]
    is_dl = model_type.startswith("dl_")
    if model_type == "chemprop":
        logging.warning("Explainability is not implemented for chemprop; skipping explain node.")
        return
    split_indices = context.get("split_indices")
    if not split_indices:
        raise ValueError(
            "explain requires split_indices from the split node. "
            "Add 'split' before 'explain' in pipeline.nodes."
        )

    model_path = context.get("trained_model_path")
    if not model_path:
        if is_dl:
            model_path = os.path.join(output_dir, f"{model_type}_best_model.pth")
        elif model_type == "catboost_classifier":
            model_path = os.path.join(output_dir, f"{model_type}_best_model.cbm")
        else:
            model_path = os.path.join(output_dir, f"{model_type}_best_model.pkl")

    validate_contract(
        bind_output_path(EXPLAIN_INPUT_MODEL_CONTRACT, model_path),
        warn_only=False,
    )

    features_file = context.get("feature_matrix", paths["rdkit_descriptors"])
    labels_file = context.get("labels_matrix", paths["pic50_3class"])
    if context["pipeline_type"] == "qm9":
        features_file = paths["rdkit_labeled"]
        labels_file = paths["rdkit_labeled"]
    if context.get("selected_ready", False):
        features_file = paths["selected_features"]
    elif context.get("preprocessed_ready", False):
        features_file = paths["preprocessed_features"]
    if context.get("preprocessed_ready", False) and os.path.exists(paths["preprocessed_labels"]):
        labels_file = paths["preprocessed_labels"]

    X, y = data_preprocessing.load_features_labels(
        features_file,
        labels_file,
        target_column,
        context.get("categorical_features"),
    )
    partitions = _resolve_split_partitions(context, X.index)
    assert partitions is not None  # split_indices was validated above
    train_idx, _, test_idx = partitions

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    estimator = train_models.load_model(
        model_path, 
        model_type, 
        input_dim=X_train.shape[1]
    )
    train_models.run_explainability(estimator, X_train, X_test, y_test, model_type, output_dir)

    validate_contract(
        bind_output_path(EXPLAIN_OUTPUT_CONTRACT, output_dir),
        warn_only=True,
    )


NODE_REGISTRY = {
    "get_data": run_node_get_data,
    "curate": run_node_curate,
    "use.curated_features": run_node_use_curated_features,
    "label.normalize": run_node_label_normalize,
    "split": run_node_split,
    "featurize.lipinski": run_node_featurize_lipinski,
    "label.ic50": run_node_label_ic50,
    "analyze.stats": run_node_analyze_stats,
    "analyze.eda": run_node_analyze_eda,
    "featurize.rdkit": run_node_featurize_rdkit,
    "featurize.rdkit_labeled": run_node_featurize_rdkit_labeled,
    "featurize.morgan": run_node_featurize_morgan,
    "preprocess.features": run_node_preprocess_features,
    "select.features": run_node_select_features,
    "train": run_node_train,
    "explain": run_node_explain,
}

_SPLIT_REQUIRED_FOR = {
    "preprocess.features",
    "select.features",
    "train",
    "explain",
}

_SPLIT_MUST_FOLLOW = {
    "curate",
    "label.normalize",
    "label.ic50",
}


def validate_pipeline_nodes(nodes: list[str]) -> None:
    """Validate pipeline node order and required dependencies.

    This pipeline enforces a single source of truth for dataset splits: the split node.
    """
    uses_split_dependents = any(node in nodes for node in _SPLIT_REQUIRED_FOR)
    split_positions = [i for i, node in enumerate(nodes) if node == "split"]
    if len(split_positions) > 1:
        raise ValueError("Pipeline must include at most one 'split' node.")

    if uses_split_dependents and not split_positions:
        raise ValueError(
            "Pipeline includes nodes that require train/val/test membership, but is missing 'split'. "
            "Add 'split' before preprocess.features/select.features/train/explain."
        )

    if not split_positions:
        return

    split_pos = split_positions[0]

    for prerequisite in _SPLIT_MUST_FOLLOW:
        if prerequisite in nodes and nodes.index(prerequisite) > split_pos:
            raise ValueError(f"'split' must come after '{prerequisite}'.")

    if uses_split_dependents:
        for dep in _SPLIT_REQUIRED_FOR:
            if dep in nodes and split_pos > nodes.index(dep):
                raise ValueError(f"'split' must come before '{dep}'.")


def run_configured_pipeline_nodes(config: dict, config_path: str) -> bool:
    pipeline = config.get("pipeline", {})
    nodes = pipeline.get("nodes")
    if not nodes:
        return False
    validate_pipeline_nodes(nodes)

    global_config = config.get("global")
    if not global_config:
        raise ValueError("global section is required in config")
    pipeline_type = global_config.get("pipeline_type", "chembl")
    base_dir = global_config["base_dir"]
    os.makedirs(base_dir, exist_ok=True)
    run_dir = resolve_run_dir(config)
    os.makedirs(run_dir, exist_ok=True)
    _configure_logging(run_dir)

    # If the pipeline uses curated tabular descriptors directly, keep all columns during curate
    # so the downstream model has feature columns to train on.
    keep_all_columns = config.get("preprocess", {}).get(
        "keep_all_columns", config.get("keep_all_columns", False)
    )
    if "use.curated_features" in nodes:
        keep_all_columns = True

    context = {
        "config_path": config_path,
        "base_dir": base_dir,
        "paths": build_paths(base_dir),
        "pipeline_type": pipeline_type,
        "task_type": global_config.get("task_type", "regression"),
        "active_threshold": global_config["thresholds"]["active"],
        "inactive_threshold": global_config["thresholds"]["inactive"],
        "target_column": global_config.get("target_column", "pIC50"),
        "model_type": config["model"]["type"],
        "get_data_config": config.get("get_data", {}),
        "curate_config": config.get("curate", {}),
        "preprocess_config": config.get("preprocess", {}),
        "split_config": config.get("split", {}),
        "featurize_config": config.get("featurize", {}),
        "label_config": config.get("label", {}),
        "model_config": config.get("model", {}),
        "categorical_features": config.get("preprocess", {}).get("categorical_features", []),
        "keep_all_columns": keep_all_columns,
        "source": config.get("get_data", {}).get("source", {}),
        "run_dir": run_dir,
    }

    with open(os.path.join(run_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    for node_name in nodes:
        node_fn = NODE_REGISTRY.get(node_name)
        if not node_fn:
            raise ValueError(f"Unknown pipeline node: {node_name}")
        node_fn(context)

    return True


def main() -> int:
    config_path = os.environ.get("CHEMLFLOW_CONFIG", "config/config.chembl.yaml")
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in config: {config_path}: {exc}", file=sys.stderr)
        return 1

    if run_configured_pipeline_nodes(config, config_path):
        return 0

    print(
        "Missing required pipeline definition. Add pipeline.nodes to the config and run the node-based pipeline. "
        "Splitting is performed only by the 'split' node.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
