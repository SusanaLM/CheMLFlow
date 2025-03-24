# import libraries
import subprocess
import os
 
# get ChemBL bioactivity data.
script_path = os.path.join('GetData', 'get_ChEMBL_target_full.py')
target_name = 'SARS-CoV-2'
output_file = 'data/covid19_bio_raw.csv'
subprocess.run(['python', script_path, target_name, output_file])
