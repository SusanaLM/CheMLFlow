# import libraries
import subprocess
import os

# calculate RDKit descriptors
script_path = os.path.join('GenDescriptors', 'RDKit_descriptors.py')
input_file = 'data/bioactivity_3class_pIC50.csv'
output_file = 'data/bioactivity_3class_with_RDKit_descriptors.csv'
subprocess.run(['python', script_path, input_file, output_file])