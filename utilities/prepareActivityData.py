import pandas as pd
import argparse
import logging

class DataPreparer:
    def __init__(self, raw_data_file, preprocessed_file, curated_file):
        self.raw_data_file = raw_data_file
        self.preprocessed_file = preprocessed_file
        self.curated_file = curated_file
        logging.info(f"Initialized DataPreparer with raw data file: {self.raw_data_file}")

    def handle_missing_data_and_duplicates(self):
        """Load data, handle missing values, and remove duplicates."""
        try:
            logging.info(f"Loading data from {self.raw_data_file}")
            df = pd.read_csv(self.raw_data_file)

            # Filter out rows with missing 'standard_value' or 'canonical_smiles'
            df_clean = df[df.standard_value.notna() & df.canonical_smiles.notna()]

            # Remove duplicates based on 'canonical_smiles'
            df_clean = df_clean.drop_duplicates(subset=['canonical_smiles'])

            # Select relevant columns
            selected_columns = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
            df_clean = df_clean[selected_columns]

            # Save preprocessed data
            df_clean.to_csv(self.preprocessed_file, index=False)
            logging.info(f"Preprocessed data saved to {self.preprocessed_file}")
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise

    def label_bioactivity(self, active_threshold=1000, inactive_threshold=10000):
        """Label compounds based on bioactivity thresholds."""
        try:
            logging.info(f"Loading preprocessed data from {self.preprocessed_file}")
            df = pd.read_csv(self.preprocessed_file)

            bioactivity_labels = []
            for value in df['standard_value']:
                if float(value) >= inactive_threshold:
                    bioactivity_labels.append("inactive")
                elif float(value) <= active_threshold:
                    bioactivity_labels.append("active")
                else:
                    bioactivity_labels.append("intermediate")

            # Add the 'class' column to the dataframe
            df['class'] = pd.Series(bioactivity_labels, name='class')

            # Save curated data with bioactivity labels
            df.to_csv(self.curated_file, index=False)
            logging.info(f"Bioactivity-labeled data saved to {self.curated_file}")
        except Exception as e:
            logging.error(f"Error during bioactivity labeling: {e}")
            raise

    def clean_smiles_column(self, curated_smiles_output):
        """Clean the 'canonical_smiles' column by keeping the longest fragment."""
        try:
            logging.info(f"Loading curated data from {self.curated_file}")
            df = pd.read_csv(self.curated_file)

            # Remove the 'canonical_smiles' column to clean it
            df_no_smiles = df.drop(columns='canonical_smiles')

            # Process the 'canonical_smiles' column by keeping the longest fragment
            smiles = []
            for smile in df['canonical_smiles'].tolist():
                fragments = str(smile).split('.')
                longest_fragment = max(fragments, key=len)
                smiles.append(longest_fragment)

            # Add the cleaned 'canonical_smiles' back to the dataframe
            smiles_series = pd.Series(smiles, name='canonical_smiles')
            df_cleaned = pd.concat([df_no_smiles, smiles_series], axis=1)

            # Save the cleaned data
            df_cleaned.to_csv(curated_smiles_output, index=False)
            logging.info(f"Cleaned SMILES data saved to {curated_smiles_output}")
        except Exception as e:
            logging.error(f"Error during SMILES cleaning: {e}")
            raise


def main(raw_data_file, preprocessed_file, curated_file, curated_smiles_output, active_threshold, inactive_threshold):
    """Main function to preprocess, label, and clean SMILES data."""
    logging.basicConfig(level=logging.INFO)

    # Initialize the DataPreparer class
    data_preparer = DataPreparer(raw_data_file, preprocessed_file, curated_file)

    try:
        # Step 1: Handle missing data and remove duplicates
        data_preparer.handle_missing_data_and_duplicates()

        # Step 2: Label bioactivity based on thresholds
        data_preparer.label_bioactivity(active_threshold, inactive_threshold)

        # Step 3: Clean SMILES column
        data_preparer.clean_smiles_column(curated_smiles_output)

    except Exception as e:
        logging.error(f"An error occurred during the data preparation process: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and process bioactivity data.")
    parser.add_argument('raw_data_file', type=str, help="Input CSV file with raw bioactivity data.")
    parser.add_argument('preprocessed_file', type=str, help="Output CSV file for preprocessed data.")
    parser.add_argument('curated_file', type=str, help="Output CSV file for curated bioactivity data with labels.")
    parser.add_argument('curated_smiles_output', type=str, help="Output CSV file for curated SMILES data.")
    parser.add_argument('--active_threshold', type=float, default=1000, help="Threshold for labeling active compounds.")
    parser.add_argument('--inactive_threshold', type=float, default=10000, help="Threshold for labeling inactive compounds.")

    args = parser.parse_args()

    main(args.raw_data_file, args.preprocessed_file, args.curated_file, args.curated_smiles_output, 
         args.active_threshold, args.inactive_threshold)
