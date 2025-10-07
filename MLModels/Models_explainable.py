import os
import sys
import numpy as np
import pandas as pd
import argparse
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_score,
    KFold
)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import shap
import joblib

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader, random_split

# from DLModels.dlregressor import DLRegressor
# from DLModels.simpleregressionnn import SimpleRegressionNN
# from DLModels.simpleregressiontnn import SimpleRegressionTNN
# from DLModels.deepregressionnn import DeepRegressionNN
# from DLModels.deepregressiontnn import DeepRegressionTNN
# from DLModels.gruregressor import GRURegressor
# from DLModels.resmlp import ResMLP
# from DLModels.tabtransformer import TabTransformer
# from DLModels.autoencoder import Autoencoder
# from DLModels.aeregressor import AERegressor

from dlregressor import DLRegressor
from simpleregressionnn import SimpleRegressionNN
from simpleregressiontnn import SimpleRegressionTNN
from deepregressionnn import DeepRegressionNN
from deepregressiontnn import DeepRegressionTNN
from gruregressor import GRURegressor
from resmlp import ResMLP
from tabtransformer import TabTransformer
from autoencoder import Autoencoder
from aeregressor import AERegressor


# ─── Reproducibility ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# make CUDNN deterministic, at some performance cost
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ─────────────────────────────────────────────────────────────────────────────────


class MLModel:
    """Train, evaluate, and explain classical ML and DL models on pIC50 from RDKit descriptors."""

    def __init__(self, features_file, labels_file, model_type, output_dir):
        self.features_file = features_file
        self.labels_file = labels_file
        self.model_type = model_type
        self.test_size = 0.2
        self.random_state = 42
        self.model = None
        # Mac (MPS) → CUDA → CPU
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = output_dir
        self.best_params_file = os.path.join(self.results_dir, f"{self.model_type}_best_params.pkl")
        self.best_model_file = os.path.join(self.results_dir, f"{self.model_type}_best_model.pkl")
        self.selected_features_file = os.path.join(self.results_dir, f"{self.model_type}_selected_features.txt")

        os.makedirs(self.results_dir, exist_ok=True)
        logging.info(f"Initialized MLModel with features: {self.features_file}, labels: {self.labels_file}, model: {self.model_type}")

    # ─── Data IO & Preprocessing ────────────────────────────────────────────────
    def load_data(self):
        """Load the RDKit descriptors as features (X) and the pIC50 values as target (y)."""
        try:
            logging.info(f"Loading features from {self.features_file}")
            X = pd.read_csv(self.features_file)
            logging.info(f"Loading labels (pIC50) from {self.labels_file}")
            y_df = pd.read_csv(self.labels_file)

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

            # Remove duplicate entries
            combined = pd.concat([X_clean, y_clean], axis=1)
            combined_before = combined.shape[0]
            combined = combined.drop_duplicates()
            combined_after = combined.shape[0]
            logging.info(f"Removed {combined_before - combined_after} duplicate entries.")

            X_clean = combined.drop('pIC50', axis=1)
            y_clean = combined['pIC50']

            logging.info(f"Data shape after initial cleaning: {X_clean.shape}")
            return X_clean, y_clean
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            sys.exit(1)

    def preprocess_data(self, X):
            """Preprocess the data by scaling and removing low-variance and highly correlated features."""
            try:
                # Scaling features using RobustScaler
                logging.info("Scaling features using RobustScaler.")
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

                # Removing low-variance features
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


    def select_stable_features(self, X, y):
        """Select features based on cross-validation stability."""
        try:
            logging.info("Selecting features based on cross-validation stability.")
        
            feature_importances = np.zeros(X.shape[1])
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            for train_idx, test_idx in cv.split(X):
                X_train_cv, y_train_cv = X.iloc[train_idx], y.iloc[train_idx]
                model = RandomForestRegressor(random_state=self.random_state)
                model.fit(X_train_cv, y_train_cv)
                feature_importances += model.feature_importances_
            feature_importances /= cv.get_n_splits()
            importances = pd.Series(feature_importances, index=X.columns)
            top_features = importances.nlargest(50).index
            X_selected = X[top_features]
            logging.info(f"Selected top 50 stable features.")
            logging.info(f"Data shape after stable feature selection: {X_selected.shape}")

            # Write the selected feature columns to a txt file
            with open(self.selected_features_file, 'w') as f:
                for feature in top_features:
                    f.write(f"{feature}\n")
            logging.info(f"Selected stable feature columns saved to {self.selected_features_file}")

            return X_selected
        except Exception as e:
            logging.error(f"Error during stable feature selection: {e}")
            sys.exit(1)

    def verify_data_quality(self, X, y):
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

    def check_data_leakage(self, X_train, X_test):
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


    # ─── DL helpers ─────────────────────────────────────────────────────────────
    def _mk_dl(self, model_class, model_kwargs, learning_rate, epochs, batch_size, verbose):
        return DLRegressor(
            model_class=model_class,
            model_kwargs=model_kwargs,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

    def _get_encoder(self, X_train, bottleneck=64, epochs=150, lr=1e-4):
        return self._load_or_train_autoencoder(X_train, bottleneck=bottleneck, epochs=epochs, lr=lr)

    def _load_or_train_autoencoder(self, X_train, bottleneck=64, epochs=150, lr=1e-4):
        """Return a trained AE encoder (load if exists, else train)."""
        SEED = getattr(self, "random_state", 42)
        rng = torch.Generator().manual_seed(SEED)
        X_np = np.asarray(X_train)
        input_dim = X_np.shape[1]

        ae = Autoencoder(input_dim=input_dim, bottleneck=bottleneck).to(self.device)
        os.makedirs(getattr(self, "results_dir", "."), exist_ok=True)
        ae_path = os.path.join(self.results_dir, f'autoencoder_b{bottleneck}.pth')

        if os.path.exists(ae_path):
            logging.info(f"Loading pretrained autoencoder from {ae_path}")
            ae.load_state_dict(torch.load(ae_path, map_location=self.device))
            ae.eval()
            return ae.encoder

        logging.info("Training autoencoder from scratch...")
        scaler_ae = StandardScaler()
        X_scaled = scaler_ae.fit_transform(X_np)

        full_ds = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        n_val = max(1, int(0.1 * len(full_ds)))
        n_tr = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_tr, n_val], generator=rng)
        loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        loader_val = DataLoader(val_ds, batch_size=32, shuffle=False)

        optimizer_ae = torch.optim.Adam(ae.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer_ae, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        criterion_ae = nn.MSELoss()

        best_val = float('inf')
        patience = 0
        max_epochs = max(epochs, 300)

        for ep in range(1, max_epochs + 1):
            # train
            ae.train()
            total_train, seen_train = 0.0, 0
            for (xb,) in loader:
                xb = xb.to(self.device)
                optimizer_ae.zero_grad()
                recon = ae(xb)
                loss = criterion_ae(recon, xb)
                loss.backward()
                optimizer_ae.step()
                bs = xb.size(0)
                total_train += loss.item() * bs
                seen_train += bs
            avg_train = total_train / max(1, seen_train)

            # val
            ae.eval()
            total_val, seen_val = 0.0, 0
            with torch.no_grad():
                for (xb,) in loader_val:
                    xb = xb.to(self.device)
                    recon = ae(xb)
                    l = criterion_ae(recon, xb).item()
                    bs = xb.size(0)
                    total_val += l * bs
                    seen_val += bs
            avg_val = total_val / max(1, seen_val)

            scheduler.step(avg_val)

            improved = avg_val + 1e-6 < best_val
            if improved:
                best_val = avg_val
                patience = 0
                torch.save(ae.state_dict(), ae_path)
                logging.info(f"[AE Epoch {ep}] New best val MSE: {avg_val:.6f} — saved.")
            else:
                patience += 1

            if ep % 10 == 0 or patience >= 20:
                logging.info(f"[AE Epoch {ep}] Train MSE: {avg_train:.6f}, Val MSE: {avg_val:.6f}")

            if patience >= 20:
                logging.info(f"AE early stopping at epoch {ep} (no improvement in 20 epochs).")
                break

        if not os.path.exists(ae_path):
            torch.save(ae.state_dict(), ae_path)
        logging.info(f"Saved autoencoder weights to {ae_path}")

        return ae.encoder

    def sweep_bottlenecks(self, X_train, y_train, X_val, y_val, bottlenecks=[8, 16, 32]):
        """Train AERegressor for each bottleneck and report validation R²."""
        results = {}
        for b in bottlenecks:
            logging.info(f"\n=== Sweeping bottleneck={b} ===")
            encoder = self._load_or_train_autoencoder(X_train, bottleneck=b, epochs=150, lr=1e-4)
            m_kwargs = {'pretrained_encoder': encoder, 'bottleneck': b, 'dropout': 0.1}
            self.model = DLRegressor(
                model_class=AERegressor,
                model_kwargs=m_kwargs,
                learning_rate=1e-4,
                epochs=150,
                batch_size=64,
                verbose=False
            )
            # Changes by chatgp #
            self.model.model = AERegressor(**m_kwargs).to(self.device)
            self.model.device = self.device
            # until here #
            self.train_model(X_train, y_train, dl_val_fraction=0.1)
            _, _, r2 = self.evaluate_model(X_val, y_val, use_nested_cv=False)
            results[b] = r2
            logging.info(f"bottleneck={b} ➜ R2={r2:.3f}")
        return results

    # ─── Initialize model (DL + ML branches) ────────────────────────────────────
    def initialize_model(self, X_train):
        input_dim = X_train.shape[1]
        try:
            # DL branches
            if self.model_type == 'ae_regressor':
                encoder = self._load_or_train_autoencoder(X_train, bottleneck=64, epochs=150, lr=1e-4)
                m_kwargs = {'pretrained_encoder': encoder, 'bottleneck': 64, 'dropout': 0.1}
                self.model = self._mk_dl(AERegressor, m_kwargs, learning_rate=1e-4, epochs=150, batch_size=64, verbose=False)
                self.model.model = AERegressor(**m_kwargs).to(self.device)
                self.model.device = self.device
                logging.info("Initialized AERegressor (DLRegressor wrapper).")

            elif self.model_type == 'dl_regressor_simple':
                m_kwargs = {'input_dim': input_dim, 'hidden_dim': 512}
                self.model = self._mk_dl(SimpleRegressionNN, m_kwargs, learning_rate=1e-3, epochs=200, batch_size=32, verbose=False)
                self.model.model = SimpleRegressionNN(**m_kwargs).to(self.device)
                self.model.device = self.device
                logging.info("Initialized DLRegressor(SimpleRegressionNN).")

            elif self.model_type == 'dl_regressor_deep':
                m_kwargs = {'input_dim': input_dim, 'hidden_dims': [512, 256, 128, 64, 32], 'dropout_rate': 0.2}
                self.model = self._mk_dl(DeepRegressionNN, m_kwargs, learning_rate=1e-3, epochs=300, batch_size=32, verbose=False)
                self.model.model = DeepRegressionNN(**m_kwargs).to(self.device)
                self.model.device = self.device
                logging.info("Initialized DLRegressor(DeepRegressionNN).")

            elif self.model_type == 'dl_t_regressor_simple':
                m_kwargs = {'input_dim': input_dim, 'hidden_dim': 512}
                self.model = self._mk_dl(SimpleRegressionTNN, m_kwargs, learning_rate=1e-5, epochs=500, batch_size=32, verbose=False)
                self.model.model = SimpleRegressionTNN(**m_kwargs).to(self.device)
                self.model.device = self.device
                logging.info("Initialized DLRegressor(SimpleRegressionTNN).")

            elif self.model_type == 'dl_t_regressor_deep':
                m_kwargs = {'input_dim': input_dim, 'hidden_dims': [512, 256, 128, 64, 32], 'dropout_rate': 0.2}
                self.model = self._mk_dl(DeepRegressionTNN, m_kwargs, learning_rate=1e-4, epochs=500, batch_size=32, verbose=False)
                self.model.model = DeepRegressionTNN(**m_kwargs).to(self.device)
                self.model.device = self.device
                logging.info("Initialized DLRegressor(DeepRegressionTNN).")

            elif self.model_type == 'dl_regressor_gru':
                m_kwargs = {'seq_len': input_dim, 'input_size': 1, 'hidden_size': 512, 'num_layers': 2, 'bidirectional': True, 'dropout': 0.3}
                self.model = self._mk_dl(GRURegressor, m_kwargs, learning_rate=5e-4, epochs=200, batch_size=16, verbose=True)
                self.model.model = GRURegressor(**m_kwargs).to(self.device)
                self.model.device = self.device
                logging.info("Initialized DLRegressor(GRURegressor).")

            elif self.model_type == 'res_mlp':
                m_kwargs = {'input_dim': input_dim, 'hidden_dim': 512, 'n_blocks': 8, 'dropout': 0.1}
                self.model = self._mk_dl(ResMLP, m_kwargs, learning_rate=1e-3, epochs=200, batch_size=32, verbose=False)
                self.model.model = ResMLP(**m_kwargs).to(self.device)
                self.model.device = self.device
                logging.info("Initialized DLRegressor(ResMLP).")

            elif self.model_type == 'tab_transformer':
                m_kwargs = {'input_dim': input_dim, 'embed_dim': 512, 'n_heads': 8, 'n_layers': 2, 'dropout': 0.1}
                self.model = self._mk_dl(TabTransformer, m_kwargs, learning_rate=1e-3, epochs=200, batch_size=32, verbose=False)
                self.model.model = TabTransformer(**m_kwargs).to(self.device)
                self.model.device = self.device
                logging.info("Initialized DLRegressor(TabTransformer).")

            # ML branches
            elif self.model_type == 'random_forest':
                param_dist = {
                    'n_estimators': [int(x) for x in np.linspace(100, 1000, 10)],
                    'max_depth': [int(x) for x in np.linspace(10, 110, 11)] + [None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
                base_rf = RandomForestRegressor(random_state=self.random_state)
                self.model = RandomizedSearchCV(
                    estimator=base_rf, param_distributions=param_dist,
                    n_iter=100, cv=5, scoring='r2', n_jobs=-1, random_state=self.random_state
                )
                logging.info("Random Forest with RandomizedSearchCV initialized.")

            elif self.model_type == 'svm':
                param_grid_svm = {
                    'svr__C': [0.1, 1, 10, 100],
                    'svr__gamma': ['scale', 'auto', 0.1, 0.01],
                    'svr__epsilon': [0.1, 0.2, 0.5]
                }
                pipeline = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf'))])
                self.model = GridSearchCV(estimator=pipeline, param_grid=param_grid_svm, cv=5, scoring='r2', n_jobs=-1)
                logging.info("SVR with scaling + GridSearchCV initialized.")

            elif self.model_type == 'decision_tree':
                param_dist_dt = {
                    'max_depth': [int(x) for x in np.linspace(5, 50, 10)] + [None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['auto', 'sqrt', 'log2', None]
                }
                base_dt = DecisionTreeRegressor(random_state=self.random_state)
                self.model = RandomizedSearchCV(
                    estimator=base_dt, param_distributions=param_dist_dt,
                    n_iter=100, cv=5, scoring='r2', n_jobs=-1, random_state=self.random_state
                )
                logging.info("Decision Tree with RandomizedSearchCV initialized.")

            elif self.model_type == 'xgboost':
                param_dist_xgb = {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'reg_alpha': [0, 0.01, 0.1, 1],
                    'reg_lambda': [0.1, 1, 10, 100]
                }
                base_xgb = XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
                self.model = RandomizedSearchCV(
                    estimator=base_xgb, param_distributions=param_dist_xgb,
                    n_iter=100, cv=5, scoring='r2', n_jobs=-1, random_state=self.random_state
                )
                logging.info("XGBoost with RandomizedSearchCV initialized.")

            elif self.model_type == 'ensemble':
                rf = RandomForestRegressor(
                    n_estimators=300, max_depth=30, max_features='sqrt', bootstrap=False,
                    min_samples_leaf=1, min_samples_split=2, random_state=self.random_state
                )
                xgb = XGBRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0, reg_lambda=1, random_state=self.random_state
                )
                # Remove n_jobs if your sklearn version doesn't support it
                self.model = VotingRegressor([('rf', rf), ('xgb', xgb)], n_jobs=-1)
                logging.info("Ensemble (VotingRegressor[rf, xgb]) initialized.")

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            logging.info(f"Model '{self.model_type}' initialized.")
        except Exception as e:
            logging.error(f"Error initializing model: {e}", exc_info=True)
            sys.exit(1)

    # ─── Training ───────────────────────────────────────────────────────
    def train_model(self, X_train, y_train, dl_val_fraction=0.1):
        """
        - DLRegressor: internal train/val split + best checkpoint saving.
        - Classical ML: fit (with hyperparam search if present) and persist best params/model.
        """
        estimator = self.model

        # DL path
        if isinstance(estimator, DLRegressor):
            logging.info("Training DLRegressor…")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            estimator.scaler.fit(X_train)
            Xs = estimator.scaler.transform(X_train)
            Ys = np.asarray(y_train).reshape(-1, 1)

            Xt = torch.tensor(Xs, dtype=torch.float32).to(self.device)
            Yt = torch.tensor(Ys, dtype=torch.float32).to(self.device)

            Xtr, Xval, Ytr, Yval = train_test_split(Xt, Yt, test_size=dl_val_fraction, random_state=self.random_state)
            train_ld = DataLoader(TensorDataset(Xtr, Ytr), batch_size=estimator.batch_size, shuffle=True)
            val_ld = DataLoader(TensorDataset(Xval, Yval), batch_size=estimator.batch_size, shuffle=False)

            # AE special param groups
            if self.model_type == 'ae_regressor' and hasattr(estimator.model, "encoder") and hasattr(estimator.model, "head"):
                enc_params = list(estimator.model.encoder.parameters())
                head_params = list(estimator.model.head.parameters())
                optimizer = torch.optim.Adam([
                    {'params': enc_params, 'lr': 1e-4},
                    {'params': head_params, 'lr': estimator.learning_rate},
                ])
            else:
                optimizer = torch.optim.Adam(estimator.model.parameters(), lr=estimator.learning_rate)

            criterion = nn.MSELoss()
            best_loss = float('inf')
            best_path = os.path.join(self.results_dir, f"best_{self.model_type}.pth")

            try:
                for epoch in range(1, estimator.epochs + 1):
                    # train
                    estimator.model.train()
                    total_train = 0.0
                    for bx, by in train_ld:
                        optimizer.zero_grad()
                        preds = estimator.model(bx).view(-1)
                        loss = criterion(preds, by.view(-1))
                        loss.backward()
                        optimizer.step()
                        total_train += loss.item() * bx.size(0)
                    train_loss = total_train / len(train_ld.dataset)

                    # validate
                    estimator.model.eval()
                    total_val = 0.0
                    with torch.no_grad():
                        for vx, vy in val_ld:
                            vpred = estimator.model(vx).view(-1)
                            vloss = criterion(vpred, vy.view(-1))
                            total_val += vloss.item() * vx.size(0)
                    avg_val = total_val / len(val_ld.dataset)

                    # checkpoint
                    if avg_val < best_loss:
                        best_loss = avg_val
                        torch.save(estimator.model.state_dict(), best_path)

                    # log
                    if epoch % 20 == 0:
                        if avg_val == best_loss:
                            logging.info(f"[Epoch {epoch}] New best val_loss={avg_val:.4f}, saved.")
                        else:
                            logging.info(f"[Epoch {epoch}] val_loss={avg_val:.4f} | train_loss={train_loss:.4f}")

                logging.info(f"DL training complete. Best val_loss={best_loss:.4f}")
            except Exception as e:
                logging.error(f"Error during DL training: {e}", exc_info=True)
                sys.exit(1)
            return

        # Classical ML path
        logging.info("Training classical ML estimator…")
        try:
            if os.path.exists(self.best_params_file):
                logging.info("Loading best hyperparameters from file.")
                best_params = joblib.load(self.best_params_file)
                if isinstance(estimator, (RandomizedSearchCV, GridSearchCV)):
                    estimator.estimator.set_params(**best_params)
                else:
                    estimator.set_params(**best_params)
                estimator.fit(X_train, y_train)
            else:
                estimator.fit(X_train, y_train)
                if isinstance(estimator, (RandomizedSearchCV, GridSearchCV)):
                    best_params = estimator.best_estimator_.get_params()
                    joblib.dump(best_params, self.best_params_file)
                    logging.info(f"Best parameters saved at {self.best_params_file}.")

            joblib.dump(estimator, self.best_model_file)
            logging.info(f"Best model saved at {self.best_model_file}.")
            if isinstance(estimator, (RandomizedSearchCV, GridSearchCV)):
                logging.info(f"Best parameters for {self.model_type}: {estimator.best_params_}")
        except Exception as e:
            logging.error(f"Error during ML training: {e}", exc_info=True)
            sys.exit(1)

    # ─── Unified evaluation ─────────────────────────────────────────────────────
    def evaluate_model(self, X_test, y_test, use_nested_cv=True):
        """
        - DLRegressor: load best checkpoint (if exists) and run batched inference.
        - Classical ML: optional nested CV on test set, then single-shot test metrics.
        """
        # Resolve estimator
        if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
            estimator = self.model.best_estimator_
        else:
            estimator = self.model

        results_path = os.path.join(self.results_dir, f"{self.model_type}_evaluation_results.txt")

        # DL path
        if isinstance(estimator, DLRegressor):
            best_path = os.path.join(self.results_dir, 'best_dl_model.pth')
            if os.path.exists(best_path):
                try:
                    estimator.model.load_state_dict(torch.load(best_path, map_location=self.device))
                    logging.info(f"Loaded best checkpoint from {best_path}")
                except Exception as e:
                    logging.warning(f"Could not load checkpoint '{best_path}': {e}. Using in-memory weights.")
            else:
                logging.info("No DL checkpoint found; evaluating current in-memory weights.")

            Xs = estimator.scaler.transform(X_test)
            Xt = torch.tensor(Xs, dtype=torch.float32).to(self.device)

            estimator.model.eval()
            preds, bs = [], getattr(estimator, "batch_size", 64)
            with torch.no_grad():
                for i in range(0, len(Xt), bs):
                    out = estimator.model(Xt[i:i+bs]).detach().cpu().numpy().flatten()
                    preds.append(out)
            y_pred = np.concatenate(preds)
            y_true = np.asarray(y_test)

            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f"[DL] Test R2={r2:.3f}, MAE={mae:.3f}")

            with open(results_path, 'w') as f:
                f.write(f"Test Set R2 Score: {r2:.4f}\n")
                f.write(f"Mean Absolute Error: {mae:.4f}\n")
            return y_true, y_pred, r2

        # Classical ML path
        try:
            if use_nested_cv:
                cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(estimator, X_test, y_test, cv=cv, scoring='r2', n_jobs=-1)
                r2_mean, r2_std = cv_scores.mean(), cv_scores.std()
                logging.info(f"[ML] Nested CV R2: {r2_mean:.3f} ± {r2_std:.3f}")
            else:
                r2_mean = r2_std = None

            y_pred = estimator.predict(X_test)
            y_true = np.asarray(y_test)
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f"[ML] Test R2={r2:.3f}, MAE={mae:.3f}")

            with open(results_path, 'w') as f:
                if use_nested_cv and (r2_mean is not None):
                    f.write(f"Nested CV R2 Score: {r2_mean:.3f} ± {r2_std:.3f}\n")
                f.write(f"Test Set R2 Score: {r2:.3f}\n")
                f.write(f"Mean Absolute Error: {mae:.3f}\n")
                if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
                    f.write(f"Best Parameters: {self.model.best_params_}\n")

            return y_true, y_pred, r2
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}", exc_info=True)
            sys.exit(1)

    # ─── Plots & Explainability ────────────────────────────────────────────────
    def plot_predicted_vs_actual(self, y_test, y_pred):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor=None)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{self.model_type.capitalize()} Predicted vs Actual")
        plot_path = os.path.join(self.results_dir, f"{self.model_type}_predicted_vs_actual.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Predicted vs Actual plot saved at {plot_path}")

    def plot_residuals(self, y_test, y_pred):
        try:
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title(f"{self.model_type.capitalize()} Residuals Plot")
            plot_path = os.path.join(self.results_dir, f"{self.model_type}_residuals_plot.png")
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Residuals plot saved at {plot_path}")
        except Exception as e:
            logging.error(f"Error plotting residuals: {e}", exc_info=True)

    def plot_permutation_importance(self, X_test, y_test):
        """Permutation importance (skip by default for DL to avoid cost)."""
        try:
            logging.info("Computing permutation importances.")
            if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
                estimator = self.model.best_estimator_
            else:
                estimator = self.model

            if isinstance(estimator, DLRegressor):
                logging.info("Skipping permutation importance for DLRegressor (too slow).")
                return

            result = permutation_importance(estimator, X_test, y_test, n_repeats=10, random_state=self.random_state, n_jobs=-1)
            importances = pd.Series(result.importances_mean, index=X_test.columns)
            top_features = importances.nlargest(20)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_features.values, y=top_features.index)
            plt.xlabel("Permutation Importance")
            plt.ylabel("Feature")
            plt.title(f"Top 20 Permutation Importances for {self.model_type.capitalize()}")
            plt.tight_layout()
            plot_path = os.path.join(self.results_dir, f"{self.model_type}_permutation_importances.png")
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Permutation importances plot saved at {plot_path}")
        except Exception as e:
            logging.error(f"Error plotting permutation importances: {e}", exc_info=True)


    def plot_shap_summary(self, X_train, X_test):
        """Plot SHAP summary plot for the model (ML unchanged; DL via gradient-based SHAP)."""
        try:
            logging.info("Computing SHAP values for model explainability.")
            if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
                estimator = self.model.best_estimator_
            else:
                estimator = self.model

            # ========== DL branch (DLRegressor): GradientExplainer -> DeepExplainer fallback ==========
            if isinstance(estimator, DLRegressor):
                logging.info("DL model detected → using gradient-based SHAP.")
                # tiny inline wrapper to apply the fitted scaler inside forward()
                class _ScaledModelWrapper(torch.nn.Module):
                    def __init__(self, base_model, scaler, device, is_gru=False, eps=1e-12):
                        super().__init__()
                        self.base = base_model
                        self.device = device
                        self.is_gru = is_gru
                        # StandardScaler: mean_/scale_, RobustScaler: center_/scale_
                        if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                            center = scaler.mean_
                            scale  = scaler.scale_
                        elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
                            center = scaler.center_
                            scale  = scaler.scale_
                        else:
                            raise ValueError("Unsupported scaler type; need (mean_,scale_) or (center_,scale_).")
                        center = torch.tensor(center, dtype=torch.float32, device=device)
                        scale  = torch.tensor(scale,  dtype=torch.float32, device=device)
                        self.register_buffer("center", center)
                        self.register_buffer("scale",  torch.clamp(scale, min=eps))
                    def forward(self, x):
                        # x expected in RAW feature space
                        x = (x - self.center) / self.scale
                        if self.is_gru:
                            x = x.unsqueeze(-1)  # (B, D, 1)
                        y = self.base(x)
                        return y.view(-1)

                # load best checkpoint if present (non-fatal if missing)
                best_path = os.path.join(self.results_dir, 'best_dl_model.pth')
                try:
                    estimator.model.load_state_dict(torch.load(best_path, map_location=self.device))
                    logging.info(f"Loaded DL checkpoint: {best_path}")
                except Exception as e:
                    logging.info(f"No/failed DL checkpoint load ({e}); using in-memory weights.")

                estimator.model.to(self.device).eval()
                is_gru = (self.model_type == 'dl_regressor_gru')

                # small, representative subsets (uniform random is fine here)
                rng = np.random.default_rng(self.random_state)
                bg_size   = min(200, len(X_train))
                eval_size = min(512, len(X_test))
                bg_idx    = rng.choice(np.arange(len(X_train)), size=bg_size, replace=False) if len(X_train) > bg_size else np.arange(len(X_train))
                ex_idx    = rng.choice(np.arange(len(X_test)),  size=eval_size, replace=False) if len(X_test)  > eval_size else np.arange(len(X_test))

                X_bg_t  = torch.tensor(X_train.iloc[bg_idx].values, dtype=torch.float32, device=self.device)
                X_ex_t  = torch.tensor(X_test.iloc[ex_idx].values,  dtype=torch.float32, device=self.device)
                # If GRU: expect (batch, seq_len, input_size=1)
                if is_gru:
                    X_bg_t = X_bg_t.unsqueeze(-1)
                    X_ex_t = X_ex_t.unsqueeze(-1)

                # GradientExplainer needs grads w.r.t. inputs
                X_bg_t.requires_grad_(True)
                X_ex_t.requires_grad_(True)
                X_ex_df = X_test.iloc[ex_idx]  # for summary_plot

                wrapped = _ScaledModelWrapper(estimator.model, estimator.scaler, self.device, is_gru=is_gru).to(self.device).eval()
                # Ensure SHAP sees a 2D output (N, 1) not a 1D (N,)
                class _Ensure2D(nn.Module):
                    def __init__(self, base):
                        super().__init__()
                        self.base = base
                    def forward(self, x):
                        out = self.base(x)
                        if isinstance(out, torch.Tensor) and out.dim() == 1:
                            out = out.unsqueeze(-1)
                        return out

                wrapped = _Ensure2D(wrapped).to(self.device).eval()

                # Prefer GradientExplainer; fall back to DeepExplainer if needed
                try:
                    explainer = shap.GradientExplainer(wrapped, X_bg_t)
                    shap_values = explainer.shap_values(X_ex_t)
                except Exception as e_grad:
                    logging.warning(f"GradientExplainer failed ({e_grad}); trying DeepExplainer.")
                    explainer = shap.DeepExplainer(wrapped, X_bg_t)
                    shap_values = explainer.shap_values(X_ex_t)

                if isinstance(shap_values, list):  # single-output regression → take first
                    shap_values = shap_values[0]

                if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
                    shap_values = shap_values.squeeze(-1)  # -> (N, D)

                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_ex_df, feature_names=X_test.columns, plot_type="bar", show=False)
                plot_path = os.path.join(self.results_dir, f"{self.model_type}_shap_summary.png")
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                logging.info(f"SHAP summary plot saved at {plot_path}")
                return  # DL branch done

            # ========== ML branch (unchanged from your version) ==========
            # For XGBoost models and ensemble, use generic Explainer; otherwise use TreeExplainer.
            if self.model_type in ['xgboost', 'ensemble']:
                explainer = shap.Explainer(estimator)
            else:
                explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_test)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            plot_path = os.path.join(self.results_dir, f"{self.model_type}_shap_summary.png")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            logging.info(f"SHAP summary plot saved at {plot_path}")

        except Exception as e:
            logging.error(f"Error plotting SHAP summary: {e}")


# ─── Main  ─────────────────────────────────────────────────────────────────
def main(features_file, labels_file, model_type, output_dir, sweep=False) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    ml_model = MLModel(features_file, labels_file, model_type, output_dir)
    X, y = ml_model.load_data()
    ml_model.verify_data_quality(X, y)
    X_preprocessed = ml_model.preprocess_data(X)
    X_selected = ml_model.select_stable_features(X_preprocessed, y)

    # Optional AE sweep (quick val split)
    if sweep:
        if model_type != 'ae_regressor':
            print("Sweep is only available for ae_regressor. Nothing to do.")
            return 1
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_selected, y, test_size=0.2, random_state=ml_model.random_state
        )
        results = ml_model.sweep_bottlenecks(X_tr, y_tr, X_val, y_val)
        print("Bottleneck sweep results (R2):", results)
        return 0

    # Standard train/test workflow
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=ml_model.test_size, random_state=ml_model.random_state
    )
    ml_model.check_data_leakage(X_train, X_test)
    ml_model.initialize_model(X_train)
    ml_model.train_model(X_train, y_train)
    y_true, y_pred, r2 = ml_model.evaluate_model(X_test, y_test)

    # Plots
    ml_model.plot_predicted_vs_actual(y_true, y_pred)
    ml_model.plot_residuals(y_true, y_pred)
    ml_model.plot_permutation_importance(X_test, y_test)
    ml_model.plot_shap_summary(X_train, X_test)

    print(f"Final Test R2: {r2:.3f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ML and DL models on bioactivity data.")
    parser.add_argument('features_file', type=str, help="Path to the CSV file with selected descriptors.")
    parser.add_argument('labels_file', type=str, help="Path to the CSV file with pIC50 labels.")
    parser.add_argument('model_type', type=str, choices=[
        'random_forest', 'svm', 'decision_tree', 'xgboost', 'ensemble',
        'dl_regressor_simple', 'dl_regressor_deep',
        'dl_t_regressor_simple', 'dl_t_regressor_deep',
        'dl_regressor_gru', 'res_mlp', 'tab_transformer', 'ae_regressor'
    ], help="Model to use for training.")
    parser.add_argument('output_dir', type=str, help="Path to the output directory where results will be saved.")
    parser.add_argument('--sweep', action='store_true', help="For ae_regressor only: sweep bottlenecks then exit.")
    args = parser.parse_args()
    code = main(args.features_file, args.labels_file, args.model_type, args.output_dir, sweep=args.sweep)
    sys.exit(code)
