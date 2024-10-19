# import libraries
import subprocess
import os

# bioactivity data normalisation and curation.
script_path = os.path.join('utilities', 'IC50_pIC50.py')
input_file = 'data/lipinski_results.csv'
output_file_3class = 'data/bioactivity_3class_pIC50.csv'
output_file_2class = 'data/bioactivity_2class_pIC50.csv'
subprocess.run(['python', script_path, input_file, output_file_3class, output_file_2class])