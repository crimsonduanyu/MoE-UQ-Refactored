"""Generate synthetic dataset for quick testing.

This script creates a synthetic city dataset with random demand data
and contextual features for testing the ST-MoE-RMQRN model.
"""

import os
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_synthetic_dataset(
    output_dir: str = 'Datasets/SyntheticCity',
    num_zones: int = 200,
    num_days: int = 30,
    time_granularity: int = 15,
    num_sources: int = 2,
    random_seed: int = 42
):
    """Generate synthetic dataset for testing.
    
    Args:
        output_dir: Output directory for the dataset.
        num_zones: Number of spatial zones (OD pairs).
        num_days: Number of days to generate.
        time_granularity: Time granularity in minutes.
        num_sources: Number of data sources (transportation modes).
        random_seed: Random seed for reproducibility.
    """
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate dimensions
    slots_per_day = 24 * 60 // time_granularity  # 96 for 15-min
    total_timesteps = num_days * slots_per_day
    
    print(f"Generating synthetic dataset:")
    print(f"  - Zones: {num_zones}")
    print(f"  - Days: {num_days}")
    print(f"  - Timesteps: {total_timesteps}")
    print(f"  - Sources: {num_sources}")
    
    # Generate time index
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    time_index = pd.date_range(
        start=start_time,
        periods=total_timesteps,
        freq=f'{time_granularity}min'
    )
    
    # Generate demand data for each source
    source_names = ['rs', 'kc', 'zc'][:num_sources]
    
    for i, source_name in enumerate(source_names):
        print(f"  Generating {source_name} demand data...")
        
        # Generate realistic-looking demand with patterns
        # Base demand with some structure
        base_demand = np.random.exponential(scale=5.0, size=(total_timesteps, num_zones))
        
        # Add daily pattern (higher during day, lower at night)
        hour_of_day = np.array([t.hour for t in time_index])
        daily_pattern = 0.5 + 0.5 * np.sin((hour_of_day - 6) * np.pi / 12)
        daily_pattern = np.clip(daily_pattern, 0.2, 1.0)
        base_demand = base_demand * daily_pattern[:, np.newaxis]
        
        # Add weekly pattern (lower on weekends)
        day_of_week = np.array([t.weekday() for t in time_index])
        weekly_pattern = np.where(day_of_week < 5, 1.0, 0.7)
        base_demand = base_demand * weekly_pattern[:, np.newaxis]
        
        # Add some zero-inflation (common in demand data)
        zero_mask = np.random.random(size=base_demand.shape) < 0.3
        base_demand[zero_mask] = 0
        
        # Round to integers (demand counts)
        demand_data = np.round(base_demand).astype(np.float32)
        
        # Create DataFrame
        columns = [f'zone_{j}' for j in range(num_zones)]
        df = pd.DataFrame(demand_data, index=time_index, columns=columns)
        df.index.name = 'departure_time'
        
        # Save as CSV (more portable than parquet)
        output_path = os.path.join(output_dir, f'{source_name}_dataset_df_{time_granularity}min.csv')
        df.to_csv(output_path)
        print(f"    Saved: {output_path}")
    
    # Generate weather/contextual data
    print("  Generating contextual (weather) data...")
    
    weather_features = {
        'temperature': np.random.normal(15, 10, total_timesteps),  # Celsius
        'humidity': np.random.uniform(30, 90, total_timesteps),     # Percentage
        'wind_speed': np.random.exponential(5, total_timesteps),    # m/s
        'precipitation': np.random.exponential(0.5, total_timesteps),  # mm
        'visibility': np.random.uniform(5, 15, total_timesteps),    # km
    }
    
    # Normalize weather features to [0, 1]
    weather_df = pd.DataFrame(weather_features, index=time_index)
    for col in weather_df.columns:
        min_val = weather_df[col].min()
        max_val = weather_df[col].max()
        weather_df[col] = (weather_df[col] - min_val) / (max_val - min_val + 1e-8)
    
    weather_df.index.name = 'time'
    weather_path = os.path.join(output_dir, 'weather_normalized.csv')
    weather_df.to_csv(weather_path)
    print(f"    Saved: {weather_path}")
    
    # Generate adjacency matrix (identity matrix for simplicity)
    print("  Generating adjacency matrix (identity)...")
    adj_matrix = np.eye(num_zones, dtype=np.float32)
    
    # Optionally add some random connections for more realistic graph
    # Add 5% random connections
    random_connections = np.random.random((num_zones, num_zones)) < 0.05
    adj_matrix = np.maximum(adj_matrix, random_connections.astype(np.float32))
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(adj_matrix, 1.0)
    
    adj_path = os.path.join(output_dir, 'GraphAdj.npy')
    np.save(adj_path, adj_matrix)
    print(f"    Saved: {adj_path}")
    
    # Create dataset info file
    info = {
        'name': 'SyntheticCity',
        'num_zones': num_zones,
        'num_days': num_days,
        'time_granularity': time_granularity,
        'num_sources': num_sources,
        'source_names': source_names,
        'total_timesteps': total_timesteps,
        'random_seed': random_seed,
        'generated_at': datetime.now().isoformat()
    }
    
    info_path = os.path.join(output_dir, 'dataset_info.txt')
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f'{key}: {value}\n')
    print(f"    Saved: {info_path}")
    
    print("\nSynthetic dataset generation complete!")
    print(f"Output directory: {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset for ST-MoE-RMQRN testing'
    )
    parser.add_argument(
        '--output_dir', type=str, default='Datasets/SyntheticCity',
        help='Output directory for the dataset'
    )
    parser.add_argument(
        '--num_zones', type=int, default=200,
        help='Number of spatial zones'
    )
    parser.add_argument(
        '--num_days', type=int, default=30,
        help='Number of days to generate'
    )
    parser.add_argument(
        '--time_granularity', type=int, default=15,
        help='Time granularity in minutes'
    )
    parser.add_argument(
        '--num_sources', type=int, default=2,
        help='Number of data sources'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        output_dir=args.output_dir,
        num_zones=args.num_zones,
        num_days=args.num_days,
        time_granularity=args.time_granularity,
        num_sources=args.num_sources,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
