# CheMLFlow
CheMLFlow is an open source software to develop, implement and apply modern cheminformatics workflows.

## Installation

1. Clone the repository

git clone https://github.com/nijamudheen/CheMLFlow.git

2. Create conda environment 

cd CheMLFlow
conda create -n chemlflow_env python=3.13
conda activate chemlflow_env

3. Install dependencies
pip install -e .

4. Install RDKit
conda install -c conda-forge rdkit

5. Remove install files
make clean

## Running tests
Scripts to run tests in CLI formats are in tests directory



