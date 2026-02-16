import pandas as pd

from MLModels import data_preprocessing


def test_load_features_labels_drops_excluded_columns(tmp_path):
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"

    pd.DataFrame(
        {
            data_preprocessing.ROW_INDEX_COL: [0, 1, 2],
            "feature_a": [1.0, 2.0, 3.0],
            "FP Calc.": [100.0, 101.0, 102.0],
            "SMILES": ["C", "CC", "CCC"],
        }
    ).to_csv(features_path, index=False)

    pd.DataFrame(
        {
            data_preprocessing.ROW_INDEX_COL: [0, 1, 2],
            "FP Exp.": [150.0, 151.0, 152.0],
        }
    ).to_csv(labels_path, index=False)

    X, y = data_preprocessing.load_features_labels(
        str(features_path),
        str(labels_path),
        target_column="FP Exp.",
        exclude_columns=["FP Calc."],
    )

    assert "FP Calc." not in X.columns
    assert "SMILES" not in X.columns
    assert list(X.columns) == ["feature_a"]
    assert y.tolist() == [150.0, 151.0, 152.0]
