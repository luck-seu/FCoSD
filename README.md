<h2 align="center">FCoSD-KDD'26</h2>

# FCoSD: Eliciting Frequency-Conditioned Spatial Dynamics for Long-Term Spatio-Temporal Forecasting

<p align="center">
  <img src="assets/fcosd-logo.svg" alt="FCoSD logo" width="760">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-384/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

Official implementation of the paper: **"Eliciting Frequency-Conditioned Spatial Dynamics for Long-Term Spatio-Temporal Forecasting"**.

## Overview

FCoSD is a novel deep learning architecture designed for long-term spatio-temporal forecasting tasks. The model leverages adaptive frequency-domain decomposition to capture complex long-term temporal patterns and combines them with spatial dynamics learning through memory-enhanced Mamba2.

The implementation supports long-term spatio-temporal forecasting with configurable input/output horizons, adaptive frequency bands, memory-enhanced spatial modeling, and Mamba/Mamba2-based temporal-spatial encoding.

## Project Structure

```text
FCoSD/
+-- train.py                         # Main training and testing entry
+-- config/                          # Dataset-specific experiment configs
|   +-- AIR/
|   +-- ENERGY/
|   +-- G56/
|   +-- PEMSTREAM/
|   +-- UrbanEV/
+-- data/
|   +-- dataloader.py                # STF/LTSF data loaders
+-- model/
|   +-- FCoSDNet.py                  # FCoSD model definition
|   +-- FreqDec.py                   # Frequency decomposition modules
|   +-- MultiPeriodFusion.py         # Multi-period/frequency fusion
|   +-- MambaEnc.py                  # Encoder layers
|   +-- Embed.py                     # Temporal and flow embeddings
+-- runners/
|   +-- FCoSDLTSFRunner.py           # Training, validation, and testing runner
+-- utils/
|   +-- metrics.py                   # MAE, MSE, RMSE, MAPE
|   +-- log.py                       # Logging utilities
|   +-- StandardScaler.py            # Data normalization
+-- checkpoints/                     # Saved checkpoints
```

## Requirements

### Environment

Recommended environment:

- Python 3.8+
- PyTorch 2.0+
- CUDA-enabled GPU
- Linux environment is recommended for `mamba-ssm` and Triton kernels

### Dependencies

Install PyTorch following the official instructions for your CUDA version:

```bash
pip install torch torchvision torchaudio
```

Install the remaining dependencies:

```bash
pip install numpy pyyaml einops torchinfo packaging triton mamba-ssm
```

## Dataset Preparation

### Supported Datasets

Currently supported datasets include:

- **UrbanEV**: https://github.com/IntelligentSystemsLab/UrbanEV
- **ENERGY, PEMSTREAM, AIR**: https://github.com/Onedean/EAC

Configuration files are provided for `AIR`, `ENERGY`, `G56`, `PEMSTREAM`, and `UrbanEV`.

## Usage

### Training

Basic training command:

```bash
python train.py \
    --dataset_name UrbanEV \
    --config_path ./config/UrbanEV/UrbanEV_Seq96.yaml
```

You can train on other configured datasets by changing both `--dataset_name` and `--config_path`, for example:

```bash
python train.py \
    --dataset_name ENERGY \
    --config_path ./config/ENERGY/ENERGY_Seq96.yaml
```

### Testing

To evaluate a saved checkpoint, set `GENERAL.mode` to `test` in the corresponding YAML config and provide the checkpoint path:

```bash
python train.py \
    --dataset_name ENERGY \
    --config_path ./config/ENERGY/ENERGY_Seq96.yaml \
    --checkpoint ./checkpoints/ENERGY/ENERGY-2026-01-22-17-46-59-best1.pt
```

## Configuration

Experiment settings are controlled by YAML files under `config/`.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{fcosd2026,
  title     = {Eliciting Frequency-Conditioned Spatial Dynamics for Long-Term Spatio-Temporal Forecasting},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026}
}
```

The complete citation will be updated after publication.

## Acknowledgements

We gratefully acknowledge the datasets provided by **EXPAND AND COMPRESS: EXPLORING TUNING PRINCIPLES FOR CONTINUAL SPATIO-TEMPORAL GRAPH FORECASTING** and **UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction**.

## License

This project is released under the MIT License.
