import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from DLModels import (
    SimpleRegressionNN,
    SimpleRegressionTNN,
    DeepRegressionNN,
    DeepRegressionTNN,
    GRURegressor,
    ResMLP,
    TabTransformer,
    Autoencoder, AERegressor
)


class DLModel:
    """Class to handle training, testing, and evaluation of ML models using raw PyTorch (no skorch)."""

    def __init__(self, features_file, labels_file, model_type, output_dir):
        self.features_file = features_file
        self.labels_file = labels_file
        self.model_type = model_type
        self.test_size = 0.2
        self.random_state = 42
        self.results_dir = output_dir
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except OSError as e:
            logging.warning(f"Could not create `{self.results_dir}` ({e}); using `./results` instead.")
            self.results_dir = os.path.join(os.getcwd(), "results_susana_DL")
            os.makedirs(self.results_dir, exist_ok=True)
        logging.info(f"Initialized DLModel with model: {self.model_type}")

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
            X = X.select_dtypes(include=[np.number])
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
            min_threshold, max_threshold = -1e10, 1e10
            X_final = X_final.clip(lower=min_threshold, upper=max_threshold)
            logging.info(f"Data clipped to range [{min_threshold}, {max_threshold}].")
            logging.info(f"Final data range: min={X_final.min().min()}, max={X_final.max().max()}")

            return X_final
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            sys.exit(1)

    def select_stable_features(self, X, y, top_k=50):
        import numpy as np
        import pandas as pd
        import logging
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import KFold

        importances = np.zeros(X.shape[1])
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for tr, te in cv.split(X):
            model = RandomForestRegressor(random_state=self.random_state)
            model.fit(X.iloc[tr], y.iloc[tr])
            importances += model.feature_importances_
        importances /= cv.get_n_splits()
        idx = np.argsort(importances)[::-1][:top_k]
        selected = X.columns[idx]
        with open(os.path.join(self.results_dir, f"{self.model_type}_selected_features.txt"), 'w') as f:
            for feat in selected:
                f.write(feat + "\n")
        logging.info(f"Selected top {top_k} features.")
        return X[selected]

    def initialize_model(self):
        # 1) If it's the GRU variant, handle it separately:
        if self.model_type == 'dl_regressor_gru':
            # treat each feature as a time‐step, scalar per step
            seq_len = self.X_train.shape[1]
            # hidden = min(512, max(16, seq_len * 4))
            hidden = 512

            # instantiate with the right args & keywords
            self.model = GRURegressor(
                seq_len,            # positional
                input_size=1,       # scalar per timestep
                hidden_size=hidden,
                num_layers=3,
                bidirectional=False,
                dropout=0.3
            )

            # standard device + optimizer setup
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            logging.info(f"Initialized GRURegressor on {self.device}.")
            return


        # 2) Otherwise pick from the other four networks
        if self.model_type == 'dl_regressor_simple':
            net_cls = SimpleRegressionNN
        elif self.model_type == 'dl_regressor_deep':
            net_cls = DeepRegressionNN
        elif self.model_type == 'dl_t_regressor_simple':
            net_cls = SimpleRegressionTNN
        elif self.model_type == 'dl_t_regressor_deep':
            net_cls = DeepRegressionTNN
        elif self.model_type == 'res_mlp':
            net_cls = ResMLP
            args = {
                'input_dim': self.X_train.shape[1],
                'hidden_dim': 256,
                'n_blocks': 8,
                'dropout': 0.1
            }
        elif self.model_type == 'tab_transformer':
            net_cls = TabTransformer
            args = {
                'input_dim': self.X_train.shape[1],
                'embed_dim': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.1
            }
        elif self.model_type == 'ae_regressor':
            seq_len = self.X_train.shape[1]
            ae_path = os.path.join(self.results_dir, 'autoencoder.pth')
            autoenc = Autoencoder(input_dim=seq_len, bottleneck=32)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            autoenc.to(self.device)

            # 1) If we have saved weights, load them
            if os.path.exists(ae_path):
                logging.info(f"Loading pretrained autoencoder from {ae_path}")
                autoenc.load_state_dict(torch.load(ae_path, map_location=self.device))

            # 2) Otherwise, train it now
            else:
                logging.info("Training autoencoder from scratch...")
                optimizer_ae = torch.optim.Adam(autoenc.parameters(), lr=1e-3)
                criterion_ae = nn.MSELoss()
                ae_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(self.X_train.values, dtype=torch.float32)
                    ),
                    batch_size=64, shuffle=True
                )
                autoenc.train()
                for epoch in range(1, 51):  # 50 epochs
                    total_loss = 0.0
                    for (xb,) in ae_loader:
                        xb = xb.to(self.device)
                        optimizer_ae.zero_grad()
                        recon = autoenc(xb)
                        loss_ae = criterion_ae(recon, xb)
                        loss_ae.backward()
                        optimizer_ae.step()
                        total_loss += loss_ae.item() * xb.size(0)
                    if epoch % 10 == 0:
                        logging.info(f"[AE Epoch {epoch}/50] Recon Loss: {total_loss / len(self.X_train):.4f}")
                # save for next time
                torch.save(autoenc.state_dict(), ae_path)
                logging.info(f"Saved autoencoder weights to {ae_path}")

            # 3) Build the regressor on top of the encoder
            self.model = AERegressor(
                pretrained_encoder=autoenc.encoder,
                bottleneck=32,
                dropout=0.2
            )
            self._setup_training()
            return


        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.input_dim = self.X_train.shape[1]
        hidden = 512

        # hidden = min(512, max(16, self.input_dim * 4))
        # for simple nets we pass hidden_dim, for deep nets we rely on their default list
        if 'deep' in self.model_type:
            # Use the class’s default hidden_dims=[128,64,32]
            kwargs = {
                'input_dim': self.input_dim,
                'dropout_rate': 0.2
            }
        else:
            kwargs = {
                'input_dim': self.input_dim,
                'hidden_dim': hidden
            }

        self.model = net_cls(**kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        logging.info(f"Initialized PyTorch model ({self.model_type}) on {self.device}.")


    def get_dataloader(self, X, y, batch_size=64, shuffle=True):
        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def train_model(self, X_train, y_train, epochs=200):
        train_loader = self.get_dataloader(X_train, y_train, batch_size=32, shuffle=True)
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(Xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * Xb.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
            if epoch % 20 == 0:
                logging.info(f"[Epoch {epoch}/{epochs}] Loss: {avg_loss:.4f}")
        torch.save(self.model.state_dict(), os.path.join(self.results_dir, f"{self.model_type}_model.pt"))
        logging.info("Training complete and model saved.")

    def evaluate_model(self, X_test, y_test):
        test_loader = self.get_dataloader(X_test, y_test, batch_size=64, shuffle=False)
        self.model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(self.device)
                out = self.model(Xb).cpu().squeeze().numpy()
                preds.append(out)
                truths.append(yb.numpy().squeeze())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(truths)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        logging.info(f"Evaluation — R2: {r2:.3f}, MAE: {mae:.3f}")
        with open(os.path.join(self.results_dir, f"{self.model_type}_evaluation.txt"), 'w') as f:
            f.write(f"R2 = {r2:.4f}\nMAE = {mae:.4f}\n")
        return y_true, y_pred, r2

    def run(self):
        # full pipeline
        X, y = self.load_data()
        self.verify_data_quality(X, y)
        X_prep = self.preprocess_data(X)
        X_sel = self.select_stable_features(X_prep, y)
        self.X_train, X_test, self.y_train, y_test = train_test_split(
            X_sel, y, test_size=self.test_size, random_state=self.random_state
        )
        self.initialize_model()
        self.train_model(self.X_train, self.y_train)
        return self.evaluate_model(X_test, y_test)

    def verify_data_quality(self, X, y):
        if X.duplicated().any():
            logging.warning("Duplicates found.")
        if (X.nunique() == 1).any():
            logging.warning("Constant features found.")


def main(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    model = DLModel(
        features_file=args.features_file,
        labels_file=args.labels_file,
        model_type=args.model_type,
        output_dir=args.output_dir
    )
    y_test, y_pred, r2 = model.run()
    print(f"Final Test R2: {r2:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch models (no skorch).")
    parser.add_argument("features_file", help="CSV of RDKit descriptors")
    parser.add_argument("labels_file", help="CSV of pIC50 labels")
    parser.add_argument("model_type", choices=[
        "dl_regressor_simple",
        "dl_t_regressor_simple",
        "dl_regressor_deep",
        "dl_t_regressor_deep",
        "dl_regressor_gru",
        "res_mlp",
        "tab_transformer",
        "ae_regressor"
    ], help="Which network to train")
    parser.add_argument("output_dir", help="Directory for outputs")
    args = parser.parse_args()
    main(args)
