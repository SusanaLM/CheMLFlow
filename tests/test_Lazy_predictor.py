# import libraries
import subprocess
import os

# Supervised ML and AI models
# Lazy predictors: iter0 performance of general ML models: queit lazy!
script_path = os.path.join('MLModels', 'Lazy_predictor.py')
features_file = 'data/bioactivity_3class_with_RDKit_descriptors.csv'
labels_file = 'data/bioactivity_3class_pIC50.csv'
test_size = 0.2
output_dir = 'results'
subprocess.run(['python', script_path, features_file, labels_file,
                '--test_size', str(test_size),
                '--output_dir', output_dir])