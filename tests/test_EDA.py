# import libraries
import subprocess
import os

# some exploratory data analysis.
script_path = os.path.join('utilities', 'EDA.py')
input_file_2class = 'data/bioactivity_2class_pIC50.csv'
input_file_3class = 'data/bioactivity_3class_pIC50.csv'
output_dir = 'data'
subprocess.run(['python', script_path, input_file_2class, input_file_3class, output_dir])