# import libraries
import subprocess
import os
 
# get ChemBL bioactivity data.
script_path = os.path.join('GetData', 'get_ChEMBL_target_full.py')
target_name = 'acetylcholinesterase'
output_file = 'data/acetylchol_bio_raw.csv'
subprocess.run(['python', script_path, target_name, output_file])
