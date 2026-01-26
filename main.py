# an example workflow that uses ChemBL bioactivty data
import subprocess
import os




# step 1. 
# get ChemBL bioactivity data.

os.makedirs('data', exist_ok=True)
os.makedirs('data/urease', exist_ok=True)

script_path = os.path.join('GetData', 'get_ChEMBL_target_full.py')
target_name = 'urease'
output_file = 'data/urease/urease_raw.csv'
subprocess.run(['python', script_path, target_name, output_file])


# step 2.
# preprocess and clean raw bioactivity data as required.
script_path = os.path.join('utilities', 'prepareActivityData.py')
raw_data_file = 'data/urease/urease_raw.csv'
preprocessed_file = 'data/urease/urease_preprocessed.csv'
curated_file = 'data/urease/urease_bio_curated_data.csv'
curated_smiles_output = 'data/urease/urease_bio_curated_smiles.csv'
active_threshold = 1000
inactive_threshold = 10000
subprocess.run(
    ['python', script_path, raw_data_file, preprocessed_file, curated_file, curated_smiles_output,
     '--active_threshold', str(active_threshold),
     '--inactive_threshold', str(inactive_threshold),
     '--properties', 'standard_value']
)


# step 3.
# Lipinski rules (rule of five) application.
script_path = os.path.join('utilities', 'Lipinski_rules.py')
smiles_file = 'data/urease/urease_bio_curated_data.csv'
output_file = 'data/urease/lipinski_results.csv'
subprocess.run(['python', script_path, smiles_file, output_file])


# step 4
# bioactivity data normalisation and curation.
script_path = os.path.join('utilities', 'IC50_pIC50.py')
input_file = 'data/urease/lipinski_results.csv'
output_file_3class = 'data/urease/bioactivity_3class_pIC50.csv'
output_file_2class = 'data/urease/bioactivity_2class_pIC50.csv'
subprocess.run(['python', script_path, input_file, output_file_3class, output_file_2class])


# step 5
# some statistical data analysis.
script_path = os.path.join('utilities', 'stat_tests.py')
input_file = 'data/urease/bioactivity_2class_pIC50.csv'
output_dir = 'data/urease'
test_type = ['mannwhitney', 'ttest', 'chi2']
descriptor = 'pIC50'
for test in test_type:
    subprocess.run(['python', script_path, input_file, output_dir, test, descriptor])


# step 6
# some exploratory data analysis.
script_path = os.path.join('utilities', 'EDA.py')
input_file_2class = 'data/urease/bioactivity_2class_pIC50.csv'
input_file_3class = 'data/urease/bioactivity_3class_pIC50.csv'
output_dir = 'data/urease'
subprocess.run(['python', script_path, input_file_2class, input_file_3class, output_dir])


# step 7
# calculate descriptors
# (a) RDKit descriptors
script_path = os.path.join('GenDescriptors', 'RDKit_descriptors.py')
input_file = 'data/urease/bioactivity_3class_pIC50.csv'
output_file = 'data/urease/bioactivity_3class_with_RDKit_descriptors.csv'
subprocess.run(['python', script_path, input_file, output_file])


# # (b) PaDDEL descriptors
# script_path = os.path.join('GenDescriptors', 'PaDEL_descriptors_only.py')
# input_file = 'data/bioactivity_3class_pIC50.csv'
# output_file = 'data/bioactivity_3class_with_PaDDEL_descriptors.csv'
# chunk_size = 10
# threads = 10
# delay = 1
# cpulimit = 90
# subprocess.run(
#     [
#         'python', script_path, input_file, output_file,
        
#     ]
# )
# subprocess.run(['python', script_path, input_file, output_file,
#                 '--chunk_size', str(chunk_size),
#                 '--threads', str(threads),
#                 '--delay', str(1),
#                 '--cpulimit', str(cpulimit)])

# # (c) mordred descriptors
# # to implement


# # step 8 
# # Supervised ML and AI models
# # Lazy predictors: iter0 performance of general ML models: queit lazy!
# script_path = os.path.join('MLModels', 'Lazy_predictor.py')
# features_file = 'data/COVID-19/bioactivity_3class_with_RDKit_descriptors.csv'
# labels_file = 'data/COVID-19/bioactivity_3class_pIC50.csv'
# test_size = 0.2
# output_dir = 'results'
# subprocess.run(['python', script_path, features_file, labels_file,
#                 '--test_size', str(test_size),
#                 '--output_dir', output_dir])

# # (b) SVM, XGboost, random forest models with cross-validation
# script_path = os.path.join('MLModels', 'MlModels_explainable_v3.py')
# features_file = 'data/COVID-19/bioactivity_3class_with_RDKit_descriptors.csv'
# labels_file = 'data/COVID-19/bioactivity_3class_pIC50.csv'
# # ML_model = 'random_forest'
# # ML_model = 'svm'
# # ML_model = 'decision_tree'
# ML_model = 'xgboost'
# subprocess.run([
#     'python', script_path, 
#     features_file, labels_file, 
#     ML_model
# ])

# # (b) SVM, XGboost, random forest models with cross-validation
# script_path = os.path.join('MLModels', 'MlModels_explainable_v5.py')
# features_file = 'data/COVID-19/bioactivity_3class_with_RDKit_descriptors.csv'
# labels_file = 'data/COVID-19/bioactivity_3class_pIC50.csv'
# output_dir = 'results'
# # ML_model = 'random_forest'
# # ML_model = 'svm'
# # ML_model = 'decision_tree'
# ML_model = 'xgboost'
# subprocess.run([
#     'python', script_path, 
#     features_file, labels_file, 
#     ML_model, output_dir
# ])



# (b) SVM, XGboost, random forest models with cross-validation
script_path = os.path.join('MLModels', 'MlModels_explainable.py')
features_file = 'data/urease/bioactivity_3class_with_RDKit_descriptors.csv'
labels_file = 'data/urease/bioactivity_3class_pIC50.csv'
output_dir = 'results'
ML_model = 'random_forest'
# ML_model = 'svm'
# ML_model = 'decision_tree'
# ML_model = 'xgboost'
subprocess.run([
    'python', script_path, 
    features_file, labels_file, 
    ML_model, output_dir
])



# # (c) Neural network models 



# # step 9 
# # unsupervised ML models
# # clustering, TSNE, UMAP


 
