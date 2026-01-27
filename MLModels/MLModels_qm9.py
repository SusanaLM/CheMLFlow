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
import shap
import joblib

class MLModel:
    """Class to handle the training, testing, and evaluation of machine learning models."""

    def __init__(self, features_file, labels_file, model_type, output_dir, target_column='pIC50'):
        self.features_file = features_file
        self.labels_file = labels_file
        self.model_type = model_type
        self.test_size = 0.2
        self.random_state = 42
        self.model = None
        self.results_dir = output_dir  # Use the output directory provided via argparse
        self.best_params_file = os.path.join(self.results_dir, f"{self.model_type}_best_params.pkl")
        self.best_model_file = os.path.join(self.results_dir, f"{self.model_type}_best_model.pkl")
        self.selected_features_file = os.path.join(self.results_dir, f"{self.model_type}_selected_features.txt")

        # Ensure the results directory exists
        self.target_column = target_column
        os.makedirs(self.results_dir, exist_ok=True)

        logging.info(f"Initialized MLModel with features file: {self.features_file}, labels file: {self.labels_file}, model: {self.model_type}")

    def load_data(self):
        """Load the RDKit descriptors as features (X) and the user-specified target values (y)."""
        try:
            logging.info(f"Loading features from {self.features_file}")
            X = pd.read_csv(self.features_file)
            logging.info(f"Loading labels ({self.target_column}) from {self.labels_file}")
            y_df = pd.read_csv(self.labels_file)

            if X.shape[0] != y_df.shape[0]:
                raise ValueError("The number of samples in the features file and the labels file do not match.")
            target_candidates = [c for c in y_df.columns if c.lower() == self.target_column.lower()]
            if not target_candidates:
                raise ValueError(f"The labels file must contain a '{self.target_column}' column.")
            target_col = target_candidates[0]

            y = y_df[target_col]
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

            cols_to_drop = [target_col] + [c for c in combined.columns if c.lower() in {'smiles','canonical_smiles'}]
            X_clean = combined.drop(columns=[c for c in cols_to_drop if c in combined.columns])
            y_clean = combined[target_col]

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

    def initialize_model(self):
        """Initialize the machine learning model based on the model_type."""
        try:
            if self.model_type == 'random_forest':
                param_dist = {
                    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
                    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
                base_rf = RandomForestRegressor(random_state=self.random_state)
                self.model = RandomizedSearchCV(
                    estimator=base_rf,
                    param_distributions=param_dist,
                    n_iter=100,
                    cv=5,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=self.random_state
                )
                logging.info("Random Forest model with RandomizedSearchCV initialized.")

            elif self.model_type == 'svm':
                param_grid_svm = {
                    'svr__C': [0.1, 1, 10, 100],
                    'svr__gamma': ['scale', 'auto', 0.1, 0.01],
                    'svr__epsilon': [0.1, 0.2, 0.5]
                }
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svr', SVR(kernel='rbf'))
                ])
                self.model = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid_svm,
                    cv=5,
                    scoring='r2',
                    n_jobs=-1
                )
                logging.info("SVR model with scaling and GridSearchCV initialized.")

            elif self.model_type == 'decision_tree':
                param_dist_dt = {
                    'max_depth': [int(x) for x in np.linspace(5, 50, num=10)] + [None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['auto', 'sqrt', 'log2', None]
                }
                base_dt = DecisionTreeRegressor(random_state=self.random_state)
                self.model = RandomizedSearchCV(
                    estimator=base_dt,
                    param_distributions=param_dist_dt,
                    n_iter=100,
                    cv=5,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=self.random_state
                )
                logging.info("Decision Tree model with RandomizedSearchCV initialized.")

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
                    estimator=base_xgb,
                    param_distributions=param_dist_xgb,
                    n_iter=100,
                    cv=5,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=self.random_state
                )
                logging.info("XGBoost model with RandomizedSearchCV initialized.")

            elif self.model_type == 'ensemble':
                rf = RandomForestRegressor(
                    n_estimators=300, max_depth=30, max_features='sqrt', bootstrap=False,
                    min_samples_leaf=1, min_samples_split=2, random_state=self.random_state
                )
                xgb = XGBRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0, reg_lambda=1, random_state=self.random_state
                )
                self.model = VotingRegressor([('rf', rf), ('xgb', xgb)], n_jobs=-1)
                logging.info("Ensemble model (VotingRegressor) initialized.")

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            logging.info(f"Model {self.model_type} initialized.")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            sys.exit(1)

    def train_model(self, X_train, y_train):
        """Train the model on the training data."""
        try:
            logging.info("Training the model.")

            # Check if best hyperparameters are saved
            if os.path.exists(self.best_params_file):
                logging.info("Loading best hyperparameters from file.")
                best_params = joblib.load(self.best_params_file)
                if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
                    self.model.estimator.set_params(**best_params)
                else:
                    self.model.set_params(**best_params)
                self.model.fit(X_train, y_train)
            else:
                self.model.fit(X_train, y_train)
                if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
                    best_params = self.model.best_estimator_.get_params()
                    # Save best hyperparameters
                    joblib.dump(best_params, self.best_params_file)
                    logging.info(f"Best parameters saved at {self.best_params_file}.")

            # Save the best model
            joblib.dump(self.model, self.best_model_file)
            logging.info(f"Best model saved at {self.best_model_file}.")

            if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
                best_params = self.model.best_params_
                logging.info(f"Best parameters for {self.model_type}: {best_params}")
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            sys.exit(1)

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model using nested cross-validation."""
        try:
            logging.info("Evaluating the model with nested cross-validation.")
            if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
                estimator = self.model.best_estimator_
            else:
                estimator = self.model

            # Outer cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(estimator, X_test, y_test, cv=cv, scoring='r2', n_jobs=-1)
            r2_mean = cv_scores.mean()
            r2_std = cv_scores.std()

            logging.info(f"Nested CV R2 Score: {r2_mean:.3f} ± {r2_std:.3f}")

            # Proceed with standard evaluation
            y_pred = estimator.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Save evaluation metrics to a text file
            results_path = os.path.join(self.results_dir, f"{self.model_type}_evaluation_results.txt")
            with open(results_path, 'w') as f:
                f.write(f"Nested CV R2 Score: {r2_mean:.3f} ± {r2_std:.3f}\n")
                f.write(f"Test Set R2 Score: {r2:.3f}\n")
                f.write(f"Mean Absolute Error: {mae:.3f}\n")
                if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
                    f.write(f"Best Parameters: {self.model.best_params_}\n")
                    logging.info(f"Best parameters for {self.model_type}: {self.model.best_params_}")

            logging.info(f"Model evaluation completed with Nested CV R2 score: {r2_mean:.3f} ± {r2_std:.3f}. Results saved at {results_path}.")

            return y_test, y_pred, r2
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            sys.exit(1)

    def plot_predicted_vs_actual(self, y_test, y_pred):
        """Plot the predicted vs actual values for regression tasks."""
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
        """Plot residuals to analyze model performance."""
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
            logging.error(f"Error plotting residuals: {e}")

    def plot_permutation_importance(self, X_test, y_test):
        """Plot permutation importances for the model."""
        try:
            logging.info("Computing permutation importances.")
            if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
                estimator = self.model.best_estimator_
            else:
                estimator = self.model

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
            logging.error(f"Error plotting permutation importances: {e}")

    def plot_shap_summary(self, X_train, X_test):
        """Plot SHAP summary plot for the model."""
        try:
            logging.info("Computing SHAP values for model explainability.")
            if isinstance(self.model, (RandomizedSearchCV, GridSearchCV)):
                estimator = self.model.best_estimator_
            else:
                estimator = self.model

            # For XGBoost models, use built-in TreeExplainer
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

def main(features_file, labels_file, model_type, output_dir):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    ml_model = MLModel(features_file, labels_file, model_type, output_dir, target_column=args.target_column)
    X, y = ml_model.load_data()
    ml_model.verify_data_quality(X, y)
    X_preprocessed = ml_model.preprocess_data(X)
    X_selected = ml_model.select_stable_features(X_preprocessed, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=ml_model.test_size, random_state=ml_model.random_state)
    ml_model.check_data_leakage(X_train, X_test)
    ml_model.initialize_model()
    ml_model.train_model(X_train, y_train)
    y_test, y_pred, r2 = ml_model.evaluate_model(X_test, y_test)

    # Generate plots for model evaluation
    ml_model.plot_predicted_vs_actual(y_test, y_pred)
    ml_model.plot_residuals(y_test, y_pred)
    ml_model.plot_permutation_importance(X_test, y_test)
    ml_model.plot_shap_summary(X_train, X_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ML models on bioactivity data.")
    parser.add_argument('features_file', type=str, help="Path to the CSV file with selected descriptors.")
    parser.add_argument('labels_file', type=str, help="Path to the CSV file with target labels.")
    parser.add_argument('model_type', type=str, choices=['random_forest', 'svm', 'decision_tree', 'xgboost', 'ensemble'], help="Model to use for training.")
    parser.add_argument('output_dir', type=str, help="Path to the output directory where results will be saved.")
    parser.add_argument('--target-column', type=str, default='pIC50', help='Name of the target column (default: pIC50).')

    args = parser.parse_args()
    main(args.features_file, args.labels_file, args.model_type, args.output_dir)



