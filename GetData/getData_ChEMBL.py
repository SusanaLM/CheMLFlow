import numpy as np
import pandas as pd
import argparse
import logging
from chembl_webresource_client.new_client import new_client


class ChemblDataRetriever:
    def __init__(self, target_name):
        """Initialize with the target name."""
        self.target_name = target_name
        self.target = new_client.target
        self.activity = new_client.activity
        self.selected_target = None
        logging.info(f"Initialized ChemblDataRetriever for target: {self.target_name}")

    def search_target(self):
        """Search the target by name."""
        try:
            logging.info(f"Searching for target: {self.target_name}")
            target_query = self.target.search(self.target_name)
            if not target_query:
                logging.error(f"Target {self.target_name} not found.")
                raise ValueError(f"Target {self.target_name} not found.")
            
            targets_df = pd.DataFrame.from_dict(target_query)
            self.selected_target = targets_df.target_chembl_id[0]
            logging.info(f"Selected target ChEMBL ID: {self.selected_target}")
            return self.selected_target
        except Exception as e:
            logging.error(f"Error in target search: {e}")
            raise

    def retrieve_bioactivity_data(self):
        """Retrieve bioactivity data for the selected target."""
        if not self.selected_target:
            raise ValueError("Target not selected. Run search_target first.")
        
        try:
            logging.info(f"Retrieving bioactivity data for target ChEMBL ID: {self.selected_target}")
            res = self.activity.filter(target_chembl_id=self.selected_target).filter(standard_type="IC50")
            df = pd.DataFrame.from_dict(res)
            if df.empty:
                logging.warning(f"No bioactivity data found for target {self.selected_target}.")
                return None
            logging.info(f"Retrieved {len(df)} bioactivity records.")
            return df
        except Exception as e:
            logging.error(f"Error retrieving bioactivity data: {e}")
            raise

    def save_data(self, df, filename):
        """Save bioactivity data to a CSV file."""
        try:
            logging.info(f"Saving data to {filename}")
            df.to_csv(filename, index=False)
            logging.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise


def main(target_name, output_file):
    """Main function to retrieve and save bioactivity data."""
    logging.basicConfig(level=logging.INFO)
    retriever = ChemblDataRetriever(target_name)
    
    try:
        retriever.search_target()
        bioactivity_data = retriever.retrieve_bioactivity_data()
        
        if bioactivity_data is not None:
            retriever.save_data(bioactivity_data, output_file)
        else:
            logging.warning(f"No data to save for target: {target_name}")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and save bioactivity data from ChEMBL.")
    parser.add_argument('target_name', type=str, help="Name of the target to search for (e.g., acetylcholinesterase)")
    parser.add_argument('output_file', type=str, help="Output CSV file to save the bioactivity data")
    
    args = parser.parse_args()
    
    main(args.target_name, args.output_file)
