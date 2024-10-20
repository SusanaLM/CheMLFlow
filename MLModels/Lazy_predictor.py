import pandas as pd
import argparse
import logging
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class LazyModelEvaluator:
    """Class to evaluate multiple regression models using LazyPredict and visualize the results."""
    
    def __init__(self, features_file, labels_file, test_size=0.2, random_state=42):
        self.features_file = features_file
        self.labels_file = labels_file
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models_train = None
        self.predictions_train = None
        self.models_test = None
        self.predictions_test = None
        logging.info(f"Initialized LazyModelEvaluator with features file: {self.features_file} and labels file: {self.labels_file}")
    
    def load_data(self):
        """Load the features and labels datasets."""
        try:
            logging.info(f"Loading features from {self.features_file}")
            X = pd.read_csv(self.features_file)

            logging.info(f"Loading labels from {self.labels_file}")
            y_df = pd.read_csv(self.labels_file)

            if 'pIC50' not in y_df.columns:
                raise ValueError(f"The labels file must contain a 'pIC50' column.")
            
            y = y_df['pIC50']
            logging.info(f"Successfully loaded features and labels data.")
            return X, y
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, X, y):
        """Clean and preprocess the features (X) and labels (y), ensuring consistency."""
        try:
            # Replace infinity values with NaN
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            y.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Create a mask for valid rows in both X and y
            valid_mask = ~X.isna().any(axis=1) & ~y.isna()

            # Apply the mask to both X and y to remove invalid rows
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            logging.info(f"Preprocessing completed. Retained {X_clean.shape[0]} valid samples.")
            return X_clean, y_clean
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def split_data(self, X, y):
        """Split the data into training and testing sets."""
        logging.info(f"Splitting data with test size {self.test_size}.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        logging.info(f"Data split into {self.X_train.shape[0]} training samples and {self.X_test.shape[0]} testing samples.")

    def fit_models(self):
        """Fit multiple regression models using LazyPredict and evaluate their performance."""
        try:
            logging.info(f"Evaluating models using LazyRegressor.")
            clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            
            # Fit models on training data
            self.models_train, self.predictions_train = clf.fit(self.X_train, self.X_train, self.y_train, self.y_train)

            # Fit models on test data
            self.models_test, self.predictions_test = clf.fit(self.X_train, self.X_test, self.y_train, self.y_test)
            
            logging.info(f"Model evaluation completed.")
        except Exception as e:
            logging.error(f"Error during model fitting: {e}")
            raise

    def plot_metrics(self, metric, title, xlabel, output_file):
        """Generate and save a bar plot for a given metric."""
        try:
            logging.info(f"Generating bar plot for {metric}.")
            plt.figure(figsize=(5, 10))
            sns.set_theme(style="whitegrid")
            colors = sns.color_palette("husl", len(self.predictions_train))
            ax = sns.barplot(y=self.predictions_train.index, x=metric, data=self.predictions_train, palette=colors)
            ax.set(xlim=(0, 1) if metric == "R-Squared" else (0, 10))
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Model")
            plt.tight_layout()

            # Save the plot
            plt.savefig(output_file)
            logging.info(f"Plot saved to {output_file}")
            plt.close()
        except Exception as e:
            logging.error(f"Error generating bar plot: {e}")
            raise

def main(features_file, labels_file, test_size, output_dir):
    logging.basicConfig(level=logging.INFO)

    # Initialize the LazyModelEvaluator class
    evaluator = LazyModelEvaluator(features_file, labels_file, test_size)

    # Load and preprocess data
    X, y = evaluator.load_data()
    X_clean, y_clean = evaluator.preprocess_data(X, y)

    # Split data into training and testing sets
    evaluator.split_data(X_clean, y_clean)

    # Fit models and evaluate them
    evaluator.fit_models()

    # Generate and save plots for R-Squared, RMSE, and Time Taken
    evaluator.plot_metrics("R-Squared", "R-Squared for Different Models", "R-Squared", f"{output_dir}/r_squared_plot.png")
    evaluator.plot_metrics("RMSE", "RMSE for Different Models", "RMSE", f"{output_dir}/rmse_plot.png")
    evaluator.plot_metrics("Time Taken", "Time Taken for Different Models", "Time (s)", f"{output_dir}/time_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple machine learning models using LazyPredict and visualize the results.")
    parser.add_argument('features_file', type=str, help="Input CSV file containing the features (X).")
    parser.add_argument('labels_file', type=str, help="Input CSV file containing the labels (y).")
    parser.add_argument('--test_size', type=float, default=0.2, help="Test size for splitting the dataset.")
    parser.add_argument('--output_dir', type=str, default='.', help="Directory to save the plots.")
    
    args = parser.parse_args()
    
    main(args.features_file, args.labels_file, args.test_size, args.output_dir)
