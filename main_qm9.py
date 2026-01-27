import subprocess
import os

# preprocess and clean raw bioactivity data as required.
script_path = os.path.join('utilities', 'prepareMolecularData_general.py')
raw_data_file = 'data/qm9.csv'
preprocessed_file = 'data/qm9_preprocessed.csv'
curated_file = 'data/qm9_curated_data.csv'
curated_smiles_output = 'data/qm9_curated_smiles.csv'

# active threshold for bioactivity datasets
# by defaults, checks for standard_value columns and skips if not found
active_threshold = 1000
inactive_threshold = 10000

# by defaults, properties column is skipped if not specified as below or similar
property = 'gap'

subprocess.run(
    ['python', script_path, raw_data_file, preprocessed_file, curated_file, curated_smiles_output,
     '--active_threshold', str(active_threshold),
     '--inactive_threshold', str(inactive_threshold),
     '--properties', property]
)

# calculate RDKit descriptors
script_path = os.path.join('GenDescriptors', 'RDKit_descriptors_labeled.py')
input_file = 'data/qm9_curated_data.csv'
output_file = 'data/qm9_with_RDKit_descriptors.csv'
labeled_output_file = 'data/qm9_RDKit_desc_labels.csv'
prop_cols = 'gap'
subprocess.run(['python', script_path, input_file, output_file,
'--labeled-output-file', labeled_output_file,
'--property-columns', prop_cols])

# SVM, XGboost, random forest models with cross-validation
script_path = os.path.join('MLModels', 'MLModels_qm9.py')
target_column = 'gap'
features_file = 'data/qm9_RDKit_desc_labels.csv'
labels_file = 'data/qm9_RDKit_desc_labels.csv'
output_dir = 'results'
# ML_model = 'random_forest'
# ML_model = 'svm'
ML_model = 'decision_tree'
# ML_model = 'xgboost'
subprocess.run([
    'python', script_path,
    features_file, labels_file,
    ML_model, output_dir,
'--target-column', target_column
])

