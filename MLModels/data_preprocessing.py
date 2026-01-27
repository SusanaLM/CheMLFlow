# data_preprocessing.py
import os
import sys
import logging
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import pdb

# ── 1) Load data ───────────────────────────────────────────────────────────────
def load_data(features_file: str, labels_file: str):
    """Load the RDKit descriptors as features (X) and the pIC50 values as target (y)."""
    try:
        logging.info(f"Loading features from {features_file}")
        X = pd.read_csv(features_file)
        logging.info(f"Loading labels (pIC50) from {labels_file}")
        y_df = pd.read_csv(labels_file)

        if X.shape[0] != y_df.shape[0]:
            raise ValueError("The number of samples in the features file and the labels file do not match.")
        if 'pIC50' not in y_df.columns:
            raise ValueError("The labels file must contain a 'pIC50' column.")

        y = y_df['pIC50']
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_mask = ~X.isna().any(axis=1) & ~y.isna()

        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # Remove duplicates across X|y join
        combined = pd.concat([X_clean, y_clean], axis=1)
        before = combined.shape[0]
        combined = combined.drop_duplicates()
        after = combined.shape[0]
        logging.info(f"Removed {before - after} duplicate entries.")

        X_clean = combined.drop('pIC50', axis=1)
        y_clean = combined['pIC50']

        logging.info(f"Data shape after initial cleaning: {X_clean.shape}")
        return X_clean, y_clean
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

# ── 2) Preprocess ──────────────────────────────────────────────────────────────
def preprocess_data(X: pd.DataFrame,
    variance_threshold: float = 0.8 * (1 - 0.8),
    corr_threshold: float = 0.95,
    clip_range: tuple = (-1e10, 1e10),
    ):
    """Preprocess the data by scaling and removing low-variance and highly correlated features."""
    try:

        # Scaling features using RobustScaler
        logging.info("Scaling features using RobustScaler.")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X) #doing this to the whole data is considered data leakage
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Removing low-variance features (on both training dataset and test) - OR DO BEFORE SCALING
        logging.info("Removing low-variance features.")
        selector = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
        X_reduced = selector.fit_transform(X_scaled_df)
        features_after_variance = X_scaled_df.columns[selector.get_support(indices=True)]
        X_reduced_df = pd.DataFrame(X_reduced, columns=features_after_variance)
        logging.info(f"Data reduced to {X_reduced_df.shape[1]} features after removing low-variance columns.")

        # Removing highly correlated features
        logging.info("Removing highly correlated features.")
        corr_matrix = X_reduced_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Adjust the threshold to 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        X_final = X_reduced_df.drop(columns=to_drop)
        logging.info(f"Data reduced to {X_final.shape[1]} features after removing highly correlated features.")
        logging.info(f"Data shape after preprocessing: {X_final.shape}")
        
        # Clip extreme values to a safe range for float32 conversion.
        # This ensures that when the model or pandas casts the data to float32,
        # the values do not exceed the representable limits.
        min_threshold, max_threshold = -1e10, 1e10  # Adjust these thresholds based on your data's expected range.
        X_final = X_final.clip(lower=min_threshold, upper=max_threshold)
        logging.info(f"Data clipped to range [{min_threshold}, {max_threshold}].")
        logging.info(f"Final data range: min={X_final.min().min()}, max={X_final.max().max()}")

        return X_final
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        sys.exit(1)

# ── 3) Stable feature selection ────────────────────────────────────────────────
def select_stable_features(X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    k: int = 50,
    out_path: str | None = None,):
    """Select features based on cross-validation stability."""
    try:
        logging.info("Selecting features based on cross-validation stability (RandomForest).")
        importances_accum = np.zeros(X.shape[1])
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

        for tr, _ in cv.split(X):
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            rf = RandomForestRegressor(random_state=random_state)
            rf.fit(X_tr, y_tr)
            importances_accum += rf.feature_importances_

        importances_accum /= cv.get_n_splits()
        s = pd.Series(importances_accum, index=X.columns)
        top_features = s.nlargest(k).index
        X_sel = X[top_features]
        logging.info(f"Selected top {k} stable features. Shape: {X_sel.shape}")

        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                for name in top_features:
                    f.write(f"{name}\n")
            logging.info(f"Selected feature list written to {out_path}")

        return X_sel
    except Exception as e:
        logging.error(f"Error during stable feature selection: {e}")
        sys.exit(1)

# ── 4) Data quality ───────────────────────────────────────────────────────────
def verify_data_quality(X: pd.DataFrame, y: pd.Series):
    """Check for data quality issues."""
    try:
        logging.info("Verifying data quality.")
        # Check for duplicates
        if X.duplicated().any():
            logging.warning("Duplicates found in feature data.")
        # Check for constant features
        if (X.nunique() == 1).any():
            logging.warning("Constant features found in data.")
        # Additional checks can be added as needed
    except Exception as e:
        logging.error(f"Error verifying data quality: {e}")

# ── 5) Leakage check ──────────────────────────────────────────────────────────
def check_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Check for data leakage between training and test sets."""
    
    try:
        logging.info("Checking for data leakage between training and test sets.")
        intersection = pd.merge(X_train, X_test, how='inner')
        if not intersection.empty:
            logging.warning("Potential data leakage detected between training and test sets.")
        else:
            logging.info("No data leakage detected.")
    except Exception as e:
        logging.error(f"Error checking data leakage: {e}")
