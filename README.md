# ST-MoE-RMQRN

**Spatiotemporal Mixture-of-Experts with Recurrent Multi-Quantile Regression Network**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper:

> **"Uncertainty quantification for joint demand prediction of multi-modal ride-sourcing services using spatiotemporal Mixture-of-Expert neural network"**
>
> *Transportation Research Part C: Emerging Technologies*, 2026

## Overview

ST-MoE-RMQRN is a deep learning model for **multi-task demand forecasting** with **probabilistic uncertainty quantification**. Key features:

- 🎯 **Multi-task learning**: Joint prediction for multiple transportation modes (ride-sourcing, taxi, bike-sharing)
- 🔮 **Uncertainty quantification**: Recurrent multi-quantile regression for prediction intervals
- 🧠 **Mixture-of-Experts**: Environment-aware routing with spatiotemporal experts
- 📊 **Spatiotemporal modeling**: Bidirectional TCN and Chebyshev Graph Convolution

## Project Structure

```
ST-MoE-RMQRN/
├── config/
│   └── default.yaml           # Configuration file
├── data/
│   ├── __init__.py
│   └── dataset.py             # Data loading utilities
├── models/
│   ├── __init__.py
│   └── st_moe_rmqrn.py        # Main model implementation
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # Loss functions and evaluation metrics
│   └── session.py             # Training session management
├── scripts/
│   └── train.py               # Training script
├── notebooks/                 # Jupyter notebooks for analysis
├── Datasets/                  # Dataset directory
│   ├── Manhattan/
│   │   ├── rs_dataset_df_15min.parquet
│   │   ├── kc_dataset_df_15min.parquet
│   │   ├── weather_normalized.csv
│   │   └── GraphAdj.npy
│   └── Beijing/
│       └── ...
├── checkpoints/               # Model checkpoints
├── outputs/                   # Evaluation results
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ST-MoE-RMQRN.git
cd ST-MoE-RMQRN

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train on Manhattan dataset
python scripts/train.py --city Manhattan --epochs 300

# Train on Beijing dataset with custom settings
python scripts/train.py --city Beijing --epochs 500 --batch_size 32 --num_experts 8

# Quick test with synthetic data (recommended for first-time users)
python scripts/train.py --synthetic --quick_test
```

### Quick Test Mode

For quick testing and verification, use the synthetic dataset with simplified model:

```bash
# Generate synthetic dataset and run quick test
python scripts/train.py --synthetic

# Or equivalently
python scripts/train.py --city SyntheticCity --quick_test

# Generate synthetic data manually (optional)
python scripts/generate_synthetic_data.py --num_zones 200 --num_days 30
```

Quick test mode automatically:
- Uses batch_size=2 (instead of 16)
- Uses hidden_dim_t=16, hidden_dim_s=32 (instead of 64, 128)
- Uses num_experts=2 (instead of 6)
- Runs only 5 epochs (instead of 300)

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--city` | Manhattan | City name (Manhattan/Beijing/SyntheticCity) |
| `--synthetic` | False | Use synthetic dataset for quick testing |
| `--quick_test` | False | Enable simplified model for fast testing |
| `--epochs` | 300 | Number of training epochs |
| `--batch_size` | 16 | Training batch size |
| `--num_experts` | 6 | Number of shared experts |
| `--t_in` | 24 | Input timesteps |
| `--t_out` | 3 | Output timesteps |
| `--hidden_dim_t` | 64 | Temporal hidden dimension |
| `--hidden_dim_s` | 128 | Spatial hidden dimension |
| `--lr` | 0.001 | Learning rate |
| `--early_stopping` | 100 | Early stopping patience |

### Using the Model Programmatically

```python
import torch
from models import STMoERMQRN
from data import create_dataloaders

# Define quantile levels
quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 
             0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

# Initialize model
model = STMoERMQRN(
    space_dim=67,           # Number of spatial zones
    contextual_dim=13,      # Weather features + time embeddings
    t_in=24,                # Input timesteps
    t_out=3,                # Output timesteps
    task_num=2,             # Number of transportation modes
    quantile_list=quantiles,
    num_experts=6
)

# Forward pass
# x: (batch, num_tasks, space_dim + contextual_dim, t_in)
predictions = model(x)
# Output: (batch, num_tasks, num_quantiles, space_dim, t_out)
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ST-MoE-RMQRN                             │
├─────────────────────────────────────────────────────────────────┤
│  Input: Multi-source demand data + Contextual features         │
│         (batch, tasks, nodes + context, timesteps)              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Task-Specific   │    │ Shared Experts  │                     │
│  │ Experts         │    │ (with routing)  │                     │
│  │                 │    │                 │                     │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │  ┌───────────────┐ │
│  │ │ ST Backbone │ │    │ │ ST Backbone │ │  │ Env-Aware     │ │
│  │ │ (TCN + GCN) │ │    │ │ (TCN + GCN) │ │←─│ Router        │ │
│  │ └─────────────┘ │    │ └─────────────┘ │  │ (contextual)  │ │
│  └────────┬────────┘    └────────┬────────┘  └───────────────┘ │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      ▼                                          │
│              ┌───────────────┐                                  │
│              │ Expert Gate   │                                  │
│              │ (per task)    │                                  │
│              └───────┬───────┘                                  │
│                      ▼                                          │
│              ┌───────────────┐                                  │
│              │ RMQR Head     │                                  │
│              │ (quantiles)   │                                  │
│              └───────────────┘                                  │
├─────────────────────────────────────────────────────────────────┤
│  Output: Quantile predictions with uncertainty estimates        │
│          (batch, tasks, quantiles, nodes, timesteps)            │
└─────────────────────────────────────────────────────────────────┘
```

## Evaluation Metrics

### Deterministic Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **Accuracy**: Round-based accuracy for count data
- **F1 (zero-demand)**: F1 score for zero-demand prediction

### Probabilistic Metrics
- **PICP**: Prediction Interval Coverage Probability
- **MPIW**: Mean Prediction Interval Width
- **CWC**: Coverage Width Criterion

## Data Format

### Required Files

For each city directory (`Datasets/{city}/`):

1. **Demand data** (parquet format):
   - `rs_dataset_df_{granularity}min.parquet`: Ride-sourcing demand
   - `kc_dataset_df_{granularity}min.parquet`: Taxi demand
   - `zc_dataset_df_{granularity}min.parquet`: Bike-sharing demand (Beijing only)

2. **Contextual data**:
   - `weather_normalized.csv`: Weather features with datetime index

3. **Graph structure**:
   - `GraphAdj.npy`: Adjacency matrix for spatial graph

## Citation

If you find this work useful, please cite:

```bibtex
@article{LIU2026105507,
  title = {Uncertainty quantification for joint demand prediction of multi-modal 
           ride-sourcing services using spatiotemporal Mixture-of-Expert neural network},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume = {184},
  pages = {105507},
  year = {2026},
  issn = {0968-090X},
  doi = {https://doi.org/10.1016/j.trc.2025.105507},
  url = {https://www.sciencedirect.com/science/article/pii/S0968090X2500511X},
  author = {Xiaobing Liu and Yu Duan and Yangli-ao Geng and Yun Wang and 
            Qingyong Li and Xuedong Yan and Ziyou Gao}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Chebyshev Graph Convolution implementation inspired by [ChebNet](https://arxiv.org/abs/1606.09375)
- Vision Transformer components from [timm](https://github.com/rwightman/pytorch-image-models)
