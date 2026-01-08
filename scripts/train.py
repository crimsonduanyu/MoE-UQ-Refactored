"""Training script for ST-MoE-RMQRN model.

This script handles data loading, model initialization, training,
and evaluation for the ST-MoE-RMQRN model.
"""

import os
import sys
import time
import argparse
from typing import List

# Add project root to path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch

from data.dataset import create_dataloaders
from models.st_moe_rmqrn import STMoERMQRN
from utils.metrics import joint_pinball_loss
from utils.session import QuantileTrainingSession


# Default quantile levels for uncertainty quantification
DEFAULT_QUANTILES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train ST-MoE-RMQRN model for demand forecasting'
    )
    
    # Data arguments
    parser.add_argument(
        '--city', type=str, default='Manhattan',
        choices=['Manhattan', 'Beijing', 'SyntheticCity'],
        help='City name for dataset'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic dataset for quick testing (equivalent to --city SyntheticCity)'
    )
    parser.add_argument(
        '--data_root', type=str, default='Datasets/',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--time_granularity', type=int, default=15,
        help='Time granularity in minutes'
    )
    
    # Model arguments
    parser.add_argument(
        '--num_experts', type=int, default=6,
        help='Number of shared experts'
    )
    parser.add_argument(
        '--t_in', type=int, default=24,
        help='Number of input timesteps'
    )
    parser.add_argument(
        '--t_out', type=int, default=3,
        help='Number of output timesteps'
    )
    parser.add_argument(
        '--hidden_dim_t', type=int, default=64,
        help='Temporal hidden dimension'
    )
    parser.add_argument(
        '--hidden_dim_s', type=int, default=128,
        help='Spatial hidden dimension'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs', type=int, default=300,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--early_stopping', type=int, default=100,
        help='Early stopping patience'
    )
    
    # Mode arguments
    parser.add_argument(
        '--train', action='store_true', default=True,
        help='Run training'
    )
    parser.add_argument(
        '--test', action='store_true', default=True,
        help='Run testing'
    )
    parser.add_argument(
        '--pretrained', type=str, default=None,
        help='Path to pretrained model'
    )
    
    # Output arguments
    parser.add_argument(
        '--checkpoint_root', type=str, default='checkpoints/',
        help='Root directory for checkpoints'
    )
    parser.add_argument(
        '--output_root', type=str, default='outputs/',
        help='Root directory for outputs'
    )
    
    # Quick test mode
    parser.add_argument(
        '--quick_test', action='store_true',
        help='Enable quick test mode with simplified model (small hidden dims, fewer experts)'
    )
    
    return parser.parse_args()


def load_data(args) -> tuple:
    """Load dataset based on city configuration.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Tuple of (data_list, context_info, space_dim, task_num).
    """
    data_path = os.path.join(args.data_root, args.city)
    
    # Define data sources by city
    if args.city == 'Manhattan':
        modes = ['rs', 'kc']
    elif args.city == 'SyntheticCity':
        modes = ['rs', 'kc']
    else:  # Beijing
        modes = ['rs', 'kc', 'zc']
    
    # Load datasets
    dataset_paths = []
    for mode in modes:
        # Try parquet first, then CSV
        parquet_path = os.path.join(data_path, f'{mode}_dataset_df_{args.time_granularity}min.parquet')
        csv_path = os.path.join(data_path, f'{mode}_dataset_df_{args.time_granularity}min.csv')
        if os.path.exists(parquet_path):
            dataset_paths.append(parquet_path)
        elif os.path.exists(csv_path):
            dataset_paths.append(csv_path)
        else:
            raise FileNotFoundError(f"Dataset not found: {parquet_path} or {csv_path}")
    
    data_list = []
    for path in dataset_paths:
        if path.endswith('.csv'):
            data = pd.read_csv(path, index_col='departure_time').values
        elif path.endswith('.parquet'):
            data = pd.read_parquet(path).values
        elif path.endswith('.npy'):
            data = np.load(path)
        else:
            raise ValueError(f"Unknown data format: {path}")
        data_list.append(data)
    
    # Load contextual information (weather data)
    context_path = os.path.join(data_path, 'weather_normalized.csv')
    context_info = pd.read_csv(context_path, index_col='time')
    context_info.index = pd.to_datetime(context_info.index)
    
    space_dim = data_list[0].shape[1]
    task_num = len(data_list)
    
    return data_list, context_info, space_dim, task_num


def main():
    """Main training function."""
    args = parse_args()
    
    # Handle synthetic dataset shortcut
    if args.synthetic:
        args.city = 'SyntheticCity'
    
    # Apply quick test mode settings
    if args.quick_test or args.city == 'SyntheticCity':
        print("=" * 50)
        print("QUICK TEST MODE ENABLED")
        print("Using simplified model configuration")
        print("=" * 50)
        
        # Override with simplified settings
        if args.batch_size == 16:  # Only override if default
            args.batch_size = 2
        if args.hidden_dim_t == 64:
            args.hidden_dim_t = 16
        if args.hidden_dim_s == 128:
            args.hidden_dim_s = 32
        if args.num_experts == 6:
            args.num_experts = 2
        if args.epochs == 300:
            args.epochs = 5
        if args.early_stopping == 100:
            args.early_stopping = 30
    
    # Check if synthetic data exists, generate if not
    if args.city == 'SyntheticCity':
        synthetic_path = os.path.join(args.data_root, 'SyntheticCity')
        if not os.path.exists(synthetic_path):
            print("Synthetic dataset not found, generating...")
            from scripts.generate_synthetic_data import generate_synthetic_dataset
            generate_synthetic_dataset(output_dir=synthetic_path)
    
    # Setup paths
    checkpoint_path = os.path.join(args.checkpoint_root, args.city)
    output_path = args.output_root
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # Environment configuration
    env_config = {
        'city_name': args.city,
        'time_granularity': args.time_granularity,
        'data_root': args.data_root,
        'checkpoint_root': args.checkpoint_root,
        'data_path': os.path.join(args.data_root, args.city),
        'checkpoint_path': checkpoint_path
    }
    
    # Load data
    print(f"Loading data for {args.city}...")
    data_list, context_info, space_dim, task_num = load_data(args)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_list,
        num_timesteps_input=args.t_in,
        num_timesteps_output=args.t_out,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.8,
        mode='long',
        contextual_data=context_info
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Compute contextual dimension (Fourier embedding + weather features)
    contextual_dim = context_info.values.shape[1] + 8  # 8 for Fourier embedding
    
    # Initialize model
    print("Initializing model...")
    model = STMoERMQRN(
        space_dim=space_dim,
        contextual_dim=contextual_dim,
        hidden_dim_t=args.hidden_dim_t,
        hidden_dim_s=args.hidden_dim_s,
        t_in=args.t_in,
        t_out=args.t_out,
        task_num=task_num,
        quantile_list=DEFAULT_QUANTILES,
        num_experts=args.num_experts,
        prior_graph_path=f'Datasets/{args.city}/GraphAdj.npy'
    )
    
    # Load pretrained weights if specified
    if args.pretrained:
        try:
            model = torch.load(args.pretrained, weights_only=False)
            print(f'Loaded pretrained model from {args.pretrained}')
        except FileNotFoundError:
            print('Pretrained weights not found, training from scratch.')
    
    # Model checkpoint name
    model_name = f'{args.city}_{args.time_granularity}_ST_MoE_RMQRN_{args.num_experts}expert_{args.epochs}epoch.pth'
    save_path = os.path.join(checkpoint_path, model_name)
    
    # Create training session
    session = QuantileTrainingSession(
        model=model,
        loss_fn=joint_pinball_loss,
        quantile_list=DEFAULT_QUANTILES,
        device=device,
        early_stopping_patience=args.early_stopping,
        env_config=env_config
    )
    
    # Training
    if args.train:
        time_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Training starts at {time_start}")
        
        session.fit(
            train_loader, 
            val_loader, 
            epochs=args.epochs,
            save_path=save_path
        )
        
        train_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Training ends at {train_end}")
    
    # Testing
    if args.test:
        result_file = f'{args.city}_ST_MoE_RMQRN_{args.time_granularity}min.json'
        result_path = os.path.join(output_path, result_file)
        
        results = session.test(
            test_loader,
            checkpoint_path=save_path,
            result_path=result_path,
            evaluation_mode='probabilistic'
        )
        
        test_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Testing ends at {test_end}")
        print(f"Results saved to {result_path}")
        
        # Print key metrics
        print("\n=== Test Results ===")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
