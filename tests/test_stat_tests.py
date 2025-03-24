# import libraries
import subprocess
import os

# some statistical data analysis.
script_path = os.path.join('utilities', 'stat_tests.py')
input_file = 'data/bioactivity_2class_pIC50.csv'
output_dir = 'data'
test_type = ['mannwhitney', 'ttest', 'chi2']
descriptor = 'pIC50'
for test in test_type:
    subprocess.run(['python', script_path, input_file, output_dir, test, descriptor])
