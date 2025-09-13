# CC-GmNet: Class-Conditioned Gaussian Mixture Modeling for Imbalanced Time-Series Quantification

## Overview
CC-GmNet is a PyTorch-based framework for quantifying class prevalence in imbalanced time-series datasets. It combines LSTM-based feature extraction with class-conditioned Gaussian mixture modeling, offering robust quantification for biomedical signals (e.g., EEG, EMG).

## Project Structure
- `dataset/` — Contains time-series data files and data loading scripts
- `dlquantification/` — Core source code: models and quantification utilities
- `experiments/` — Experiment configuration and parameter files (JSON)
- `environment.yml` — Conda environment for dependency management
- `requirements.txt` — Required Python packages for pip installs
- `train_lequa.py` — Main training script

## Installation
```bash
git clone https://github.com/mithila442/CC-GMNet.git
cd CC-GmNet
conda env create -f environment.yml # Or: pip install -r requirements.txt
conda activate cc-gmnet
```


## Usage
Train the model on EEG data:
```bash
python train_lequa.py
--train_name eeg_test
--network gmnet
--network_parameters experiments/parameters/common_parameters_EEG.json
--feature_extraction timeseriescnn
--bag_generator UnlabeledMixerBagGenerator
--standarize
--dataset EEG
--cuda_device cuda:0
```

**Arguments:**
- `--train_name` - Name for the training run
- `--network` - Model choice (`gmnet`)
- `--network_parameters` - Path to experiment hyperparameters JSON
- `--feature_extraction` - Backbone feature extractor
- `--bag_generator` - Bag sampling method
- `--standarize` - Apply feature standardization
- `--dataset` - Dataset to use (EEG, EMG, etc.)
- `--cuda_device` - Specify CUDA device for GPU training

## Contact
Questions or issues?  
Email: elx12@txstate.edu