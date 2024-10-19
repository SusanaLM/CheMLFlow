# import libraries
import subprocess
import os

# preprocess and clean raw bioactivity data as required.
script_path = os.path.join('utilities', 'prepareActivityData.py')
raw_data_file = 'data/acetylchol_bio_raw.csv'
preprocessed_file = 'data/acetylchol_bio_preprocessed.csv'
curated_file = 'data/acetaylchol_bio_curated_data.csv'
curated_smiles_output = 'data/acetylchol_bio_curated_smiles.csv'
active_threshold = 1000
inactive_threshold = 10000
subprocess.run(
    ['python', script_path, raw_data_file, preprocessed_file, curated_file, curated_smiles_output,
     '--active_threshold', str(active_threshold),
     '--inactive_threshold', str(inactive_threshold)]
)
