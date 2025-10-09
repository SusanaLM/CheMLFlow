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

4. Install RDKit from via conda or pip install

# Conda installation (recommended)
conda install -c conda-forge rdkit

# pip installation (works for most applications)
pip install rdkit 


5. Install PyTorch and PyTorch Lightning with GPU support

# For Linux/Windows (check for cuda version available. for instance, cu121 here)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning

# For AMD GPU (ROCm, Linux only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install pytorch-lightning

# For Apple Silicon (M1/M2/M3), no CUDA/GPU available 
# Can use the MPS backend (built into the default macOS wheels) when using mps as the device
 
pip install torch torchvision torchaudio
pip install pytorch-lightning 

6. Remove additional install files

make clean

## Running tests

Scripts to run tests in CLI formats are in tests directory


