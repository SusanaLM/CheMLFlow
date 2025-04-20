import subprocess
import os

script_path = os.path.join('GenDescriptors', 'PaDEL_descriptors_only.py')
input_file = 'data/bioactivity_3class_pIC50.csv'
output_file = 'data/bioactivity_3class_with_PaDDEL_descriptors.csv'
chunk_size = 10
threads = 10
delay = 1
cpulimit = 90
subprocess.run(
    [
        'python', script_path, input_file, output_file,
        
    ]
)
subprocess.run(['python', script_path, input_file, output_file,
                '--chunk_size', str(chunk_size),
                '--threads', str(threads),
                '--delay', str(1),
                '--cpulimit', str(cpulimit)])