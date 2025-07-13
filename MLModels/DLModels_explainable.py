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
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader

from DLModels import (
    DLRegressor,
    SimpleRegressionNN,
    SimpleRegressionTNN,
    DeepRegressionNN,
    DeepRegressionTNN,
    GRURegressor,
    ResMLP,
    TabTransformer,
    Autoencoder,
    AERegressor
)

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

class DLModel:
    """Class to handle training, testing, and evaluation of DL models using PyTorch."""

    def __init__(self, features_file, labels_file, model_type, output_dir):
        self.features_file = features_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels_file = labels_file
        self.model_type = model_type
        self.test_size = 0.2
        self.random_state = 42
        self.results_dir = output_dir
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except OSError as e:
            logging.warning(f"Could not create `{self.results_dir}` ({e}); using `./results_DL` instead.")
            self.results_dir = os.path.join(os.getcwd(), "results_DL")
            os.makedirs(self.results_dir, exist_ok=True)
        logging.info(f"Initialized DLModel with model: {self.model_type}")
        # file to save selected features
        self.selected_features_file = os.path.join(self.results_dir, 'selected_features.txt')
        self.selected_features_file = os.path.join(self.results_dir, 'selected_features.txt')

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
            logging.error(f"Error loading data: {e}", exc_info=True)
            sys.exit(1)

    def verify_data_quality(self, X, y):
        """Check for data quality issues."""
        try:
            logging.info("Verifying data quality.")
            if X.duplicated().any():
                logging.warning("Duplicates found in feature data.")
            if (X.nunique() == 1).any():
                logging.warning("Constant features found in data.")
        except Exception as e:
            logging.error(f"Error verifying data quality: {e}", exc_info=True)

    def preprocess_data(self, X):
        """Preprocess the data by scaling and removing low-variance and highly correlated features."""
        try:
            logging.info("Scaling features using RobustScaler.")
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

            logging.info("Removing low-variance features.")
            selector = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
            X_reduced = selector.fit_transform(X_scaled_df)
            features_after_variance = X_scaled_df.columns[selector.get_support(indices=True)]
            X_reduced_df = pd.DataFrame(X_reduced, columns=features_after_variance)
            logging.info(f"Data reduced to {X_reduced_df.shape[1]} features after removing low-variance columns.")

            logging.info("Removing highly correlated features.")
            corr_matrix = X_reduced_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            X_final = X_reduced_df.drop(columns=to_drop)
            logging.info(f"Data reduced to {X_final.shape[1]} features after removing highly correlated features.")
            logging.info(f"Data shape after preprocessing: {X_final.shape}")
            
            min_threshold, max_threshold = -1e10, 1e10
            X_final = X_final.clip(lower=min_threshold, upper=max_threshold)
            logging.info(f"Data clipped to range [{min_threshold}, {max_threshold}].")
            logging.info(f"Final data range: min={X_final.min().min()}, max={X_final.max().max()}")

            return X_final
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}", exc_info=True)
            sys.exit(1)

    def select_stable_features(self, X, y):
        """Select features based on cross-validation stability."""
        try:
            logging.info("Selecting features based on cross-validation stability.")
            feature_importances = np.zeros(X.shape[1])
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            for train_idx, test_idx in cv.split(X):
                model = RandomForestRegressor(random_state=self.random_state)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                feature_importances += model.feature_importances_
            feature_importances /= cv.get_n_splits()
            importances = pd.Series(feature_importances, index=X.columns)
            top_features = importances.nlargest(50).index
            X_selected = X[top_features]
            with open(self.selected_features_file, 'w') as f:
                for feature in top_features:
                    f.write(f"{feature}\n")
            logging.info(f"Selected stable feature columns saved to {self.selected_features_file}")
            return X_selected
        except Exception as e:
            logging.error(f"Error during stable feature selection: {e}", exc_info=True)
            sys.exit(1)

    def _load_or_train_autoencoder(self, X_train, bottleneck=64, epochs=150, lr=1e-4):
        """Load a pretrained Autoencoder or train a new one, returning its encoder."""
        input_dim = X_train.shape[1]
        # data_t = torch.tensor(X_train.values, dtype=torch.float32)
        ae = Autoencoder(input_dim=input_dim, bottleneck=bottleneck)
        ae.to(self.device)
        ae_path = os.path.join(self.results_dir, 'autoencoder.pth')
        if os.path.exists(ae_path):
            logging.info(f"Loading pretrained autoencoder from {ae_path}")
            ae.load_state_dict(torch.load(ae_path, map_location=self.device))
        else:
            logging.info("Training autoencoder from scratch...")
            optimizer_ae = torch.optim.Adam(ae.parameters(), lr=lr)
            scheduler = ReduceLROnPlateau(optimizer_ae, mode='min', factor=0.5, patience=10, min_lr=1e-6)
            best_val = float('inf')
            patience_counter = 0
            max_epochs = 300 
            criterion_ae = nn.MSELoss()
            scaler_ae = StandardScaler()
            X_scaled = scaler_ae.fit_transform(X_train.values)
            dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=32, shuffle=True)


            full_ds = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
            n_val  = int(0.1 * len(full_ds))
            n_tr   = len(full_ds) - n_val
            train_ds, val_ds = random_split(full_ds, [n_tr, n_val],
                                            generator=torch.Generator().manual_seed(SEED))
            loader     = DataLoader(train_ds, batch_size=32, shuffle=True)
            loader_val = DataLoader(val_ds,   batch_size=32, shuffle=False)


            ae.train()

            for ep in range(1, max_epochs+1):
                # — training pass (no change) —
                total_train = 0.0
                ae.train()
                for (xb,) in loader:
                    xb = xb.to(self.device)
                    optimizer_ae.zero_grad()
                    recon = ae(xb)
                    loss = criterion_ae(recon, xb)
                    loss.backward()
                    optimizer_ae.step()
                    total_train += loss.item() * xb.size(0)
                avg_train = total_train / len(loader.dataset)

                # — validation pass —
                total_val = 0.0
                ae.eval()
                with torch.no_grad():
                    for (xb,) in loader_val:   # you’ll need to split off a small val loader before
                        xb = xb.to(self.device)
                        recon = ae(xb)
                        total_val += criterion_ae(recon, xb).item()
                avg_val = total_val / len(loader_val.dataset)

                # step the scheduler
                scheduler.step(avg_val)

                # early stopping logic
                if avg_val + 1e-4 < best_val:
                    best_val = avg_val
                    patience_counter = 0
                    torch.save(ae.state_dict(), ae_path)
                    logging.info(f"[AE Epoch {ep}] New best val MSE: {avg_val:.4f}, saved.")
                else:
                    patience_counter += 1

                # logging every 10 or if early stop
                if ep % 10 == 0 or patience_counter >= 20:
                    logging.info(f"[AE Epoch {ep}] Train MSE: {avg_train:.4f}, Val MSE: {avg_val:.4f}")

                if patience_counter >= 20:
                    logging.info(f"Early stopping at epoch {ep} (no improvement in 20 epochs).")
                    break

            # for ep in range(1, epochs+1):
            #     total_loss = 0.0
            #     for (xb,) in loader:
            #         xb = xb.to(self.device)
            #         optimizer_ae.zero_grad()
            #         recon = ae(xb)
            #         loss_ae = criterion_ae(recon, xb)
            #         loss_ae.backward()
            #         optimizer_ae.step()
            #         total_loss += loss_ae.item() * xb.size(0)
            #     if ep % 10 == 0:
            #         logging.info(f"[AE Epoch {ep}/{epochs}] Loss: {total_loss/len(loader.dataset):.4f}")
            torch.save(ae.state_dict(), ae_path)
            logging.info(f"Saved autoencoder weights to {ae_path}")
        return ae.encoder

    def initialize_model(self, X_train):
        input_dim = X_train.shape[1]
        if self.model_type == 'ae_regressor':
            encoder = self._load_or_train_autoencoder(
                X_train,
                bottleneck=64,
                epochs=150,
                lr=1e-4
            )
            self.model = DLRegressor(
                model_class    = AERegressor,
                model_kwargs   = {
                    'pretrained_encoder': encoder,
                    'bottleneck': 64,
                    'dropout': 0.1
                },
                learning_rate  = 1e-4,
                epochs         = 150,
                batch_size     = 64,
                verbose        = False
            )
            self.model.model = AERegressor(
                pretrained_encoder=encoder,
                bottleneck=64,
                dropout=0.1
            ).to(self.device)

            self.model.device = self.device
            logging.info("Initialized AERegressor with pretrained encoder.")
            return
        else:
            specs = {
                'dl_regressor_simple': (
                    SimpleRegressionNN,
                    {'input_dim': input_dim, 'hidden_dim': 512},
                    {'learning_rate': 1e-3, 'epochs': 200, 'batch_size':32, 'verbose':False}
                ),
                'dl_regressor_deep': (
                    DeepRegressionNN,
                    {'input_dim': input_dim, 'hidden_dims': [512, 256, 128, 64, 32], 'dropout_rate':0.2},
                    {'learning_rate':1e-3, 'epochs':300, 'batch_size':32, 'verbose':False}
                ),
                'dl_t_regressor_simple': (
                    SimpleRegressionTNN,
                    {'input_dim': input_dim, 'hidden_dim': 512},
                    {'learning_rate': 1e-5, 'epochs': 500, 'batch_size':32, 'verbose':False}
                ),
                'dl_t_regressor_deep': (
                    DeepRegressionTNN,
                    {'input_dim': input_dim, 'hidden_dims': [512, 256, 128, 64, 32], 'dropout_rate':0.2},
                    {'learning_rate':1e-4, 'epochs':500, 'batch_size':32, 'verbose':False}
                ),
                'dl_regressor_gru': (
                    GRURegressor,
                    {'seq_len': input_dim, 'input_size':1, 'hidden_size':512,'num_layers':2,'bidirectional':True,'dropout':0.3},
                    {'learning_rate':5e-4,'epochs':200,'batch_size':16,'verbose':True}
                ),
                'res_mlp': (
                    ResMLP,
                    {'input_dim': input_dim, 'hidden_dim':512, 'n_blocks':8, 'dropout':0.1},
                    {'learning_rate':1e-3,'epochs':200,'batch_size':32,'verbose':False}
                ),
                'tab_transformer': (
                    TabTransformer,
                    {'input_dim': input_dim, 'embed_dim':512, 'n_heads':8, 'n_layers':2, 'dropout':0.1},
                    {'learning_rate':1e-3,'epochs':200,'batch_size':32,'verbose':False}
                ),
            }
        if self.model_type not in specs:
            logging.error(f"Unsupported model type: {self.model_type}")
            sys.exit(1)
        cls, m_kwargs, fit_kwargs = specs[self.model_type]
        self.model = DLRegressor(
            model_class=cls,
            model_kwargs=m_kwargs,
            **fit_kwargs
        )
        # instantiate underlying PyTorch model
        self.model.model = cls(**m_kwargs).to(self.device)
        self.model.device = self.device
        logging.info(f"Initialized DLRegressor with {cls.__name__}")

    def train_model(self, X_train, y_train, val_fraction=0.1):
        """
        Train the model with an internal validation split, 
        track the best‐performing weights, and log every 20 epochs.
        """
        torch.cuda.empty_cache()
        # 1) Prep data & loader
        self.model.scaler.fit(X_train)
        Xs = self.model.scaler.transform(X_train)
        Ys = np.array(y_train).reshape(-1, 1)
        Xt = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        Yt = torch.tensor(Ys, dtype=torch.float32).to(self.device)
        Xtr, Xval, Ytr, Yval = train_test_split(
            Xt, Yt, test_size=val_fraction, random_state=self.random_state
        )
        train_ld = DataLoader(TensorDataset(Xtr, Ytr),
                              batch_size=self.model.batch_size,
                              shuffle=True)
        val_ld   = DataLoader(TensorDataset(Xval, Yval),
                              batch_size=self.model.batch_size,
                              shuffle=False)

        # for AERegressor: fine-tune encoder at 1e-5 and head at the usual rate
        if self.model_type == 'ae_regressor':
            enc_params  = list(self.model.model.encoder.parameters())
            head_params = list(self.model.model.head.parameters())
            optimizer = torch.optim.Adam([
                {'params': enc_params,  'lr': 1e-4},               
                {'params': head_params, 'lr': self.model.learning_rate},
            ])
        else:
            optimizer = torch.optim.Adam(self.model.model.parameters(),
                                         lr=self.model.learning_rate)
        criterion = nn.MSELoss()
        best_loss = float('inf')
        best_path = os.path.join(self.results_dir, 'best_dl_model.pth')

        try:
            for epoch in range(1, self.model.epochs + 1):
                # — training —
                self.model.model.train()
                total_train = 0.0
                for bx, by in train_ld:
                    optimizer.zero_grad()
                    preds = self.model.model(bx).view(-1)
                    target = by.view(-1)
                    loss = criterion(preds, target)
                    loss.backward()
                    optimizer.step()
                    total_train += loss.item() * bx.size(0)
                train_loss = total_train / len(train_ld.dataset)

                # — validation —
                self.model.model.eval()
                total_val = 0.0
                with torch.no_grad():
                    for vx, vy in val_ld:
                        vpred = self.model.model(vx).view(-1)
                        vtarget = vy.view(-1)
                        vloss = criterion(vpred, vtarget)
                        total_val += vloss.item() * vx.size(0)
                avg_val = total_val / len(val_ld.dataset)

                # track best weights (no log here)
                if avg_val < best_loss:
                    best_loss = avg_val
                    torch.save(self.model.model.state_dict(), best_path)

                # log every 20 epochs
                if epoch % 20 == 0:
                    if avg_val == best_loss:
                        logging.info(f"[Epoch {epoch}] New best val_loss={avg_val:.4f}, saved.")
                    else:
                        logging.info(f"[Epoch {epoch}] val_loss={avg_val:.4f}")
                        logging.info(f"[Epoch {epoch}] training MSE: {train_loss:.4f}")

            logging.info(f"Training complete. Best val_loss={best_loss:.4f}")
        except Exception as e:
            logging.error(f"Error during training: {e}", exc_info=True)
            sys.exit(1)

    def evaluate_model(self, X_test, y_test):
        # load best
        best_path = os.path.join(self.results_dir, 'best_dl_model.pth')
        self.model.model.load_state_dict(torch.load(best_path, map_location=self.device))
        # data
        Xs = self.model.scaler.transform(X_test)
        Xt = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        # predict
        self.model.model.eval()
        preds=[]
        with torch.no_grad():
            for i in range(0, len(Xt), self.model.batch_size):
                batch = Xt[i:i+self.model.batch_size]
                out = self.model.model(batch).cpu().numpy().flatten()
                preds.append(out)
        y_pred = np.concatenate(preds)
        y_true= np.array(y_test)
        r2 = r2_score(y_true, y_pred)
        mae= mean_absolute_error(y_true, y_pred)
        logging.info(f"Evaluation R2={r2:.3f}, MAE={mae:.3f}")
        eval_file = os.path.join(self.results_dir, 'evaluation.txt')
        with open(eval_file, 'w') as f:
            f.write(f"R2={r2:.4f}\nMAE={mae:.4f}\n")
        return y_true, y_pred, r2

    def run(self):
        X, y = self.load_data()
        self.verify_data_quality(X, y)
        X_p = self.preprocess_data(X)
        X_s = self.select_stable_features(X_p, y)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y, test_size=self.test_size, random_state=self.random_state
        )
        self.initialize_model(X_tr)
        self.train_model(X_tr, y_tr)
        return self.evaluate_model(X_te, y_te)


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    model = DLModel(
        features_file=args.features_file,
        labels_file=args.labels_file,
        model_type=args.model_type,
        output_dir=args.output_dir
    )
    y_test, y_pred, r2 = model.run()
    print(f"Final Test R2: {r2:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch DL models.")
    parser.add_argument("features_file", help="CSV of RDKit descriptors")
    parser.add_argument("labels_file", help="CSV of pIC50 labels")
    parser.add_argument("model_type", choices=[
        "dl_regressor_simple",
        "dl_regressor_deep",
        "dl_t_regressor_simple",
        "dl_t_regressor_deep",
        "dl_regressor_gru",
        "res_mlp",
        "tab_transformer",
        "ae_regressor"
    ], help="Which model to train")
    parser.add_argument("output_dir", help="Directory for outputs")
    args = parser.parse_args()
    sys.exit(main(args))
