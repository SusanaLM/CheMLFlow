# an example workflow that uses ChemBL bioactivty data
import csv
import json
import os
import subprocess
import sys

import yaml
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

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
    PREPROCESS_FEATURES_INPUT_CONTRACT,
    PREPROCESS_FEATURES_OUTPUT_CONTRACT,
    PREPROCESS_LABELS_OUTPUT_CONTRACT,
    PREPROCESS_ARTIFACTS_CONTRACT,
    SELECT_FEATURES_INPUT_FEATURES_CONTRACT,
    SELECT_FEATURES_INPUT_LABELS_CONTRACT,
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

def build_paths(base_dir: str) -> dict[str, str]:
    return {
        "raw": os.path.join(base_dir, "raw.csv"),
        "raw_sample": os.path.join(base_dir, "raw_sample.csv"),
        "preprocessed": os.path.join(base_dir, "preprocessed.csv"),
        "curated": os.path.join(base_dir, "curated.csv"),
        "curated_smiles": os.path.join(base_dir, "curated_smiles.csv"),
        "lipinski": os.path.join(base_dir, "lipinski_results.csv"),
        "pic50_3class": os.path.join(base_dir, "bioactivity_3class_pIC50.csv"),
        "pic50_2class": os.path.join(base_dir, "bioactivity_2class_pIC50.csv"),
        "rdkit_descriptors": os.path.join(base_dir, "rdkit_descriptors.csv"),
        "rdkit_labeled": os.path.join(base_dir, "rdkit_descriptors_labeled.csv"),
        "preprocessed_features": os.path.join(base_dir, "preprocessed_features.csv"),
        "preprocessed_labels": os.path.join(base_dir, "preprocessed_labels.csv"),
        "selected_features": os.path.join(base_dir, "selected_features.csv"),
        "selected_features_list": os.path.join(base_dir, "selected_features.txt"),
        "preprocess_artifacts": os.path.join(base_dir, "preprocess_artifacts.joblib"),
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
    qm9_config = context["qm9_config"]
    if pipeline_type == "qm9":
        max_rows = qm9_config.get("max_rows")
        if max_rows:
            sampled_path = context["paths"]["raw_sample"]
            sample_csv(raw_data_file, sampled_path, int(max_rows))
            raw_data_file = sampled_path

    curate_config = context["curate_config"]
    target_column = context["target_column"]
    properties = curate_config.get("properties")
    if not properties:
        properties = target_column if pipeline_type == "qm9" else "standard_value"

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
    if context["keep_all_columns"]:
        cmd.append("--keep_all_columns")
    subprocess.run(cmd)
    validate_contract(bind_output_path(PREPROCESSED_CONTRACT, preprocessed_file), warn_only=True)
    validate_contract(bind_output_path(CURATE_OUTPUT_CONTRACT, curated_file), warn_only=True)


def run_node_featurize_lipinski(context: dict) -> None:
    validate_contract(
        bind_output_path(FEATURIZE_LIPINSKI_INPUT_CONTRACT, context["paths"]["curated"]),
        warn_only=False,
    )
    script_path = os.path.join("utilities", "Lipinski_rules.py")
    smiles_file = context["paths"]["curated"]
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
        bind_output_path(FEATURIZE_RDKIT_INPUT_CONTRACT, context["paths"]["pic50_3class"]),
        warn_only=False,
    )
    script_path = os.path.join("GenDescriptors", "RDKit_descriptors.py")
    input_file = context["paths"]["pic50_3class"]
    output_file = context["paths"]["rdkit_descriptors"]
    subprocess.run([sys.executable, script_path, input_file, output_file])
    validate_contract(
        bind_output_path(FEATURIZE_RDKIT_OUTPUT_CONTRACT, output_file),
        warn_only=True,
    )


def run_node_featurize_rdkit_labeled(context: dict) -> None:
    validate_contract(
        bind_output_path(FEATURIZE_RDKIT_LABELED_INPUT_CONTRACT, context["paths"]["curated"]),
        warn_only=False,
    )
    script_path = os.path.join("GenDescriptors", "RDKit_descriptors_labeled.py")
    input_file = context["paths"]["curated"]
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
        bind_output_path(FEATURIZE_RDKIT_LABELED_OUTPUT_LABELS_CONTRACT, labeled_output_file),
        warn_only=True,
    )


def _resolve_feature_inputs(context: dict) -> tuple[str, str]:
    pipeline_type = context["pipeline_type"]
    if pipeline_type == "qm9":
        return context["paths"]["rdkit_labeled"], context["paths"]["rdkit_labeled"]
    return context["paths"]["rdkit_descriptors"], context["paths"]["pic50_3class"]


def _preprocess_params(context: dict) -> tuple[float, float, tuple[float, float], int, float, int]:
    preprocess_config = context.get("preprocess_config", {})
    variance_threshold = preprocess_config.get("variance_threshold", 0.8 * (1 - 0.8))
    corr_threshold = preprocess_config.get("corr_threshold", 0.95)
    clip_range = preprocess_config.get("clip_range", (-1e10, 1e10))
    if isinstance(clip_range, list):
        clip_range = tuple(clip_range)
    stable_k = preprocess_config.get("stable_features_k", 50)
    test_size = preprocess_config.get("test_size", 0.2)
    random_state = preprocess_config.get("random_state", 42)
    return variance_threshold, corr_threshold, clip_range, stable_k, test_size, random_state


def run_node_preprocess_features(context: dict) -> None:
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
    )
    X_clean = X_clean.reset_index(drop=True)
    y_clean = y_clean.reset_index(drop=True)
    data_preprocessing.verify_data_quality(X_clean, y_clean)

    variance_threshold, corr_threshold, clip_range, _, test_size, random_state = _preprocess_params(context)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )
    preprocessor = data_preprocessing.fit_preprocessor(
        X_train,
        variance_threshold=variance_threshold,
        corr_threshold=corr_threshold,
        clip_range=clip_range,
    )
    X_train_pre = data_preprocessing.transform_preprocessor(X_train, preprocessor)
    X_test_pre = data_preprocessing.transform_preprocessor(X_test, preprocessor)
    X_preprocessed = (
        pd.concat([X_train_pre, X_test_pre])
        .sort_index()
        .reset_index(drop=True)
    )
    y_aligned = (
        pd.concat([y_train, y_test])
        .sort_index()
        .reset_index(drop=True)
    )

    preprocessed_features = context["paths"]["preprocessed_features"]
    preprocessed_labels = context["paths"]["preprocessed_labels"]
    X_preprocessed.to_csv(preprocessed_features, index=False)
    y_values = y_aligned.values
    if hasattr(y_values, "ndim") and y_values.ndim > 1:
        y_values = y_values[:, 0]
    pd.DataFrame({context["target_column"]: y_values}).to_csv(
        preprocessed_labels, index=False
    )
    joblib.dump(preprocessor, context["paths"]["preprocess_artifacts"])

    validate_contract(
        bind_output_path(PREPROCESS_FEATURES_OUTPUT_CONTRACT, preprocessed_features),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(PREPROCESS_LABELS_OUTPUT_CONTRACT, preprocessed_labels),
        warn_only=True,
    )
    validate_contract(
        bind_output_path(PREPROCESS_ARTIFACTS_CONTRACT, context["paths"]["preprocess_artifacts"]),
        warn_only=True,
    )
    context["preprocessed_ready"] = True


def run_node_select_features(context: dict) -> None:
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

    X = pd.read_csv(features_file)
    y_df = pd.read_csv(labels_file)
    target_column = context["target_column"]
    y = data_preprocessing.select_target_series(y_df, target_column)

    _, _, _, stable_k, test_size, random_state = _preprocess_params(context)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
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
    X_selected.to_csv(selected_features, index=False)

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
    output_dir = "results"
    model_type = context["model_type"]
    target_column = context["target_column"]
    paths = context["paths"]

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
        features_file, labels_file, target_column
    )
    if not skip_quality_checks:
        data_preprocessing.verify_data_quality(X, y)

    test_size = context.get("preprocess_config", {}).get("test_size", 0.2)
    random_state = context.get("preprocess_config", {}).get("random_state", 42)
    cv_folds = context.get("model_config", {}).get("cv_folds", 5)
    search_iters = context.get("model_config", {}).get("search_iters", 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if not skip_quality_checks:
        data_preprocessing.check_data_leakage(X_train, X_test)

    model_config = context.get("model_config", {})

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
    )
    context["trained_model_path"] = train_result.model_path

    validate_contract(
        bind_output_path(TRAIN_OUTPUT_CONTRACT, output_dir),
        warn_only=True,
    )


def run_node_explain(context: dict) -> None:
    output_dir = "results"
    model_type = context["model_type"]
    target_column = context["target_column"]
    paths = context["paths"]
    is_dl = model_type.startswith("dl_")

    model_path = context.get("trained_model_path")
    if not model_path:
        if is_dl:
            model_path = os.path.join(output_dir, f"{model_type}_best_model.pth")
        else:
            model_path = os.path.join(output_dir, f"{model_type}_best_model.pkl")

    validate_contract(
        bind_output_path(EXPLAIN_INPUT_MODEL_CONTRACT, model_path),
        warn_only=False,
    )

    features_file = paths["rdkit_descriptors"]
    labels_file = paths["pic50_3class"]
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
        features_file, labels_file, target_column
    )
    test_size = context.get("preprocess_config", {}).get("test_size", 0.2)
    random_state = context.get("preprocess_config", {}).get("random_state", 42)
    X_train, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
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
    "featurize.lipinski": run_node_featurize_lipinski,
    "label.ic50": run_node_label_ic50,
    "analyze.stats": run_node_analyze_stats,
    "analyze.eda": run_node_analyze_eda,
    "featurize.rdkit": run_node_featurize_rdkit,
    "featurize.rdkit_labeled": run_node_featurize_rdkit_labeled,
    "preprocess.features": run_node_preprocess_features,
    "select.features": run_node_select_features,
    "train": run_node_train,
    "explain": run_node_explain,
}


def run_configured_pipeline_nodes(config: dict, config_path: str) -> bool:
    pipeline = config.get("pipeline", {})
    nodes = pipeline.get("nodes")
    if not nodes:
        return False

    pipeline_type = config.get("pipeline_type", "chembl")
    base_dir = config["base_dir"]
    os.makedirs(base_dir, exist_ok=True)

    context = {
        "config_path": config_path,
        "base_dir": base_dir,
        "paths": build_paths(base_dir),
        "pipeline_type": pipeline_type,
        "active_threshold": config["thresholds"]["active"],
        "inactive_threshold": config["thresholds"]["inactive"],
        "target_column": config.get("target_column", "pIC50"),
        "model_type": config["model"]["type"],
        "qm9_config": config.get("qm9", {}),
        "curate_config": config.get("curate", {}),
        "preprocess_config": config.get("preprocess", {}),
        "model_config": config.get("model", {}),
        "keep_all_columns": config.get("preprocess", {}).get(
            "keep_all_columns", config.get("keep_all_columns", False)
        ),
    }

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

    pipeline_type = config.get("pipeline_type", "chembl")
    data_source = config["data_source"]
    source = config.get("source", {})
    base_dir = config["base_dir"]
    active_threshold = config["thresholds"]["active"]
    inactive_threshold = config["thresholds"]["inactive"]
    model_type = config["model"]["type"]
    qm9_config = config.get("qm9", {})
    preprocess_config = config.get("preprocess", {})
    keep_all_columns = preprocess_config.get(
        "keep_all_columns", config.get("keep_all_columns", False)
    )
    if pipeline_type == "qm9":
        target_column = qm9_config.get("target_column", config.get("target_column", "gap"))
    else:
        target_column = config.get("target_column", "pIC50")

    os.makedirs("data", exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)

    # step 1.
    # get raw data from a configured source.
    script_path = os.path.join("GetData", "get_data.py")
    output_file = os.path.join(base_dir, "raw.csv")
    subprocess.run([sys.executable, script_path, output_file, "--config", config_path])
    validate_contract(bind_output_path(GET_DATA_OUTPUT_CONTRACT, output_file), warn_only=True)

    if pipeline_type == "qm9":
        max_rows = qm9_config.get("max_rows")
        raw_data_file = output_file
        if max_rows:
            sampled_path = os.path.join(base_dir, "raw_sample.csv")
            sample_csv(output_file, sampled_path, int(max_rows))
            raw_data_file = sampled_path

        # preprocess and clean raw molecular data as required.
        script_path = os.path.join("utilities", "prepareActivityData.py")
        preprocessed_file = os.path.join(base_dir, "qm9_preprocessed.csv")
        curated_file = os.path.join(base_dir, "qm9_curated_data.csv")
        curated_smiles_output = os.path.join(base_dir, "qm9_curated_smiles.csv")
        qm9_cmd = [
            sys.executable,
            script_path,
            raw_data_file,
            preprocessed_file,
            curated_file,
            curated_smiles_output,
            "--active_threshold",
            str(active_threshold),
            "--inactive_threshold",
            str(inactive_threshold),
            "--properties",
            target_column,
        ]
        if keep_all_columns:
            qm9_cmd.append("--keep_all_columns")
        subprocess.run(qm9_cmd)

        # calculate RDKit descriptors (and include labels)
        script_path = os.path.join("GenDescriptors", "RDKit_descriptors_labeled.py")
        input_file = curated_file
        output_file = os.path.join(base_dir, "qm9_with_RDKit_descriptors.csv")
        labeled_output_file = os.path.join(base_dir, "qm9_RDKit_desc_labels.csv")
        subprocess.run(
            [
                sys.executable,
                script_path,
                input_file,
                output_file,
                "--labeled-output-file",
                labeled_output_file,
                "--property-columns",
                target_column,
            ]
        )
        validate_contract(
            ContractSpec(
                name="qm9_labeled_descriptors",
                required_columns=[target_column],
                output_path=labeled_output_file,
            ),
            warn_only=True,
        )

        # train model (legacy pipeline path)
        features_file = labeled_output_file
        labels_file = labeled_output_file
        output_dir = "results"
        X, y = data_preprocessing.load_features_labels(
            features_file, labels_file, target_column
        )
        test_size = preprocess_config.get("test_size", 0.2)
        random_state = preprocess_config.get("random_state", 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        train_models.train_model(
            X_train,
            y_train,
            X_test,
            y_test,
            model_type,
            output_dir,
            random_state=random_state,
        )
        return 0


    # step 2.
    # preprocess and clean raw bioactivity data as required.
    script_path = os.path.join("utilities", "prepareActivityData.py")
    raw_data_file = os.path.join(base_dir, "raw.csv")
    preprocessed_file = os.path.join(base_dir, "urease_preprocessed.csv")
    curated_file = os.path.join(base_dir, "urease_bio_curated_data.csv")
    curated_smiles_output = os.path.join(base_dir, "urease_bio_curated_smiles.csv")
    chembl_cmd = [
        sys.executable,
        script_path,
        raw_data_file,
        preprocessed_file,
        curated_file,
        curated_smiles_output,
        "--active_threshold",
        str(active_threshold),
        "--inactive_threshold",
        str(inactive_threshold),
        "--properties",
        "standard_value",
    ]
    if keep_all_columns:
        chembl_cmd.append("--keep_all_columns")
    subprocess.run(chembl_cmd)
    validate_contract(bind_output_path(PREPROCESSED_CONTRACT, preprocessed_file), warn_only=True)
    validate_contract(bind_output_path(CURATE_OUTPUT_CONTRACT, curated_file), warn_only=True)


    # step 3.
    # Lipinski rules (rule of five) application.
    script_path = os.path.join("utilities", "Lipinski_rules.py")
    smiles_file = os.path.join(base_dir, "urease_bio_curated_data.csv")
    output_file = os.path.join(base_dir, "lipinski_results.csv")
    subprocess.run([sys.executable, script_path, smiles_file, output_file])
    validate_contract(bind_output_path(LIPINSKI_CONTRACT, output_file), warn_only=True)
    validate_contract(bind_output_path(IC50_INPUT_CONTRACT, output_file), warn_only=True)


    # step 4
    # bioactivity data normalisation and curation.
    script_path = os.path.join("utilities", "IC50_pIC50.py")
    input_file = os.path.join(base_dir, "lipinski_results.csv")
    output_file_3class = os.path.join(base_dir, "bioactivity_3class_pIC50.csv")
    output_file_2class = os.path.join(base_dir, "bioactivity_2class_pIC50.csv")
    subprocess.run(
        [sys.executable, script_path, input_file, output_file_3class, output_file_2class]
    )
    validate_contract(bind_output_path(PIC50_3CLASS_CONTRACT, output_file_3class), warn_only=True)
    validate_contract(bind_output_path(PIC50_2CLASS_CONTRACT, output_file_2class), warn_only=True)


    # step 5
    # some statistical data analysis.
    script_path = os.path.join("utilities", "stat_tests.py")
    input_file = os.path.join(base_dir, "bioactivity_2class_pIC50.csv")
    output_dir = base_dir
    test_type = ["mannwhitney", "ttest", "chi2"]
    descriptor = "pIC50"
    for test in test_type:
        subprocess.run(
            [sys.executable, script_path, input_file, output_dir, test, descriptor]
        )


    # step 6
    # some exploratory data analysis.
    script_path = os.path.join("utilities", "EDA.py")
    input_file_2class = os.path.join(base_dir, "bioactivity_2class_pIC50.csv")
    input_file_3class = os.path.join(base_dir, "bioactivity_3class_pIC50.csv")
    output_dir = base_dir
    subprocess.run(
        [sys.executable, script_path, input_file_2class, input_file_3class, output_dir]
    )


    # step 7
    # calculate descriptors
    # (a) RDKit descriptors
    script_path = os.path.join("GenDescriptors", "RDKit_descriptors.py")
    input_file = os.path.join(base_dir, "bioactivity_3class_pIC50.csv")
    output_file = os.path.join(base_dir, "bioactivity_3class_with_RDKit_descriptors.csv")
    subprocess.run([sys.executable, script_path, input_file, output_file])
    validate_contract(bind_output_path(DESCRIPTORS_CONTRACT, output_file), warn_only=True)


# # (b) PaDDEL descriptors
# script_path = os.path.join('GenDescriptors', 'PaDEL_descriptors_only.py')
# input_file = 'data/bioactivity_3class_pIC50.csv'
# output_file = 'data/bioactivity_3class_with_PaDDEL_descriptors.csv'
# chunk_size = 10
# threads = 10
# delay = 1
# cpulimit = 90
# subprocess.run(
#     [
#         'python', script_path, input_file, output_file,
        
#     ]
# )
# subprocess.run(['python', script_path, input_file, output_file,
#                 '--chunk_size', str(chunk_size),
#                 '--threads', str(threads),
#                 '--delay', str(1),
#                 '--cpulimit', str(cpulimit)])

# # (c) mordred descriptors
# # to implement


# # step 8 
# # Supervised ML and AI models
# # Lazy predictors: iter0 performance of general ML models: queit lazy!
# script_path = os.path.join('MLModels', 'Lazy_predictor.py')
# features_file = 'data/COVID-19/bioactivity_3class_with_RDKit_descriptors.csv'
# labels_file = 'data/COVID-19/bioactivity_3class_pIC50.csv'
# test_size = 0.2
# output_dir = 'results'
# subprocess.run(['python', script_path, features_file, labels_file,
#                 '--test_size', str(test_size),
#                 '--output_dir', output_dir])

# # (b) SVM, XGboost, random forest models with cross-validation
# script_path = os.path.join('MLModels', 'MlModels_explainable_v3.py')
# features_file = 'data/COVID-19/bioactivity_3class_with_RDKit_descriptors.csv'
# labels_file = 'data/COVID-19/bioactivity_3class_pIC50.csv'
# # ML_model = 'random_forest'
# # ML_model = 'svm'
# # ML_model = 'decision_tree'
# ML_model = 'xgboost'
# subprocess.run([
#     'python', script_path, 
#     features_file, labels_file, 
#     ML_model
# ])

# # (b) SVM, XGboost, random forest models with cross-validation
# script_path = os.path.join('MLModels', 'MlModels_explainable_v5.py')
# features_file = 'data/COVID-19/bioactivity_3class_with_RDKit_descriptors.csv'
# labels_file = 'data/COVID-19/bioactivity_3class_pIC50.csv'
# output_dir = 'results'
# # ML_model = 'random_forest'
# # ML_model = 'svm'
# # ML_model = 'decision_tree'
# ML_model = 'xgboost'
# subprocess.run([
#     'python', script_path, 
#     features_file, labels_file, 
#     ML_model, output_dir
# ])



# (b) SVM, XGboost, random forest models with cross-validation
    script_path = os.path.join("MLModels", "MlModels_explainable.py")
    features_file = os.path.join(base_dir, "bioactivity_3class_with_RDKit_descriptors.csv")
    labels_file = os.path.join(base_dir, "bioactivity_3class_pIC50.csv")
    output_dir = "results"
    ML_model = model_type
# ML_model = 'svm'
# ML_model = 'decision_tree'
# ML_model = 'xgboost'
    subprocess.run(
        [
            sys.executable,
            script_path,
            features_file,
            labels_file,
            ML_model,
            output_dir,
        ]
    )
    validate_contract(
        ContractSpec(
            name="model_labels",
            required_columns=[target_column],
            output_path=labels_file,
        ),
        warn_only=True,
    )



# # (c) Neural network models 



    return 0


if __name__ == "__main__":
    raise SystemExit(main())
