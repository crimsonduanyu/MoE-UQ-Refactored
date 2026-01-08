"""Data loading utilities for ST-MoE-RMQRN.

This module provides dataset classes and data loaders for multi-source
taxi demand forecasting with temporal and contextual features.
"""

from __future__ import division

from datetime import datetime
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


def get_normalized_adj(adj: torch.Tensor) -> torch.Tensor:
    """Compute degree-normalized adjacency matrix for GCN.
    
    Args:
        adj: Adjacency matrix tensor of shape (batch, num_nodes, num_nodes).
        
    Returns:
        Normalized adjacency matrix.
    """
    if adj[0, 0].min() <= 1e-5:
        adj = adj + torch.diag_embed(
            torch.ones(adj.size(1), dtype=torch.float32)
            .unsqueeze(0).expand(adj.size(0), -1)
        ).to(adj.device)
        
    degree = torch.sum(adj, dim=1)
    degree[degree <= 1e-5] = 1e-5  # Prevent division by zero
    diag = torch.reciprocal(torch.sqrt(degree))
    normalized = diag.unsqueeze(0) * adj * diag.unsqueeze(1)
    return normalized


def get_normalized_adj_numpy(adj: np.ndarray) -> np.ndarray:
    """Compute degree-normalized adjacency matrix for GCN (NumPy version).
    
    Args:
        adj: Adjacency matrix of shape (num_nodes, num_nodes).
        
    Returns:
        Normalized adjacency matrix.
    """
    if adj[0, 0] == 0:
        adj = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))
        
    degree = np.array(np.sum(adj, axis=1)).reshape(-1)
    degree[degree <= 1e-5] = 1e-5
    diag = np.reciprocal(np.sqrt(degree))
    normalized = np.multiply(np.multiply(diag.reshape(-1, 1), adj), diag.reshape(1, -1))
    return normalized


def compute_random_walk_matrix(adj: torch.Tensor) -> torch.Tensor:
    """Compute random walk transition matrix for D-GCN.
    
    Args:
        adj: Adjacency matrix tensor.
        
    Returns:
        Random walk matrix.
    """
    degree = torch.sum(adj, dim=1)
    degree_inv = torch.pow(degree, -1)
    degree_inv[degree_inv == float('inf')] = 0.
    degree_mat_inv = torch.diag_embed(degree_inv)
    return torch.matmul(degree_mat_inv, adj)


def compute_random_walk_matrix_numpy(adj: np.ndarray) -> np.ndarray:
    """Compute random walk transition matrix (NumPy version).
    
    Args:
        adj: Adjacency matrix.
        
    Returns:
        Random walk matrix.
    """
    adj_sparse = sp.coo_matrix(adj)
    degree = np.array(adj_sparse.sum(1))
    np.seterr(divide='ignore', invalid='ignore')
    degree_inv = np.power(degree, -1).flatten()
    degree_inv[np.isinf(degree_inv)] = 0.
    degree_mat_inv = sp.diags(degree_inv)
    return degree_mat_inv.dot(adj_sparse).tocoo().toarray()


def fourier_time_embedding(timestamps: np.ndarray, num_freqs: int = 4) -> np.ndarray:
    """Create Fourier embeddings for temporal features.
    
    Encodes timestamps using sinusoidal functions at multiple frequencies
    to capture periodic patterns.
    
    Args:
        timestamps: Array of datetime64 timestamps.
        num_freqs: Number of frequency components.
        
    Returns:
        Fourier embeddings of shape (num_samples, num_freqs * 2).
    """
    # Convert to seconds
    t_seconds = np.array([
        t.astype('datetime64[s]').astype('int64') / 1e9 
        for t in timestamps
    ])
    
    # Normalize to [0, 1] within a day
    seconds_in_day = 24 * 3600
    t_norm = (t_seconds % seconds_in_day) / seconds_in_day
    
    embeddings = []
    for k in range(num_freqs):
        freq = 2.0 ** k
        embeddings.append(np.sin(2 * np.pi * freq * t_norm))
        embeddings.append(np.cos(2 * np.pi * freq * t_norm))
        
    return np.stack(embeddings, axis=-1)


class MultiSourceDatasetLong(Dataset):
    """Dataset for long-term multi-source demand prediction.
    
    Uses historical patterns from last week, yesterday, and today
    for long-term demand forecasting with BQN mode.
    
    Args:
        data_list: List of numpy arrays, each of shape (num_nodes, 1, num_timesteps).
        num_timesteps_input: Number of input timesteps.
        num_timesteps_output: Number of output timesteps.
        data_range: [start, end] indices for data slicing.
        contextual_data: Optional contextual features (DataFrame or array).
    """
    
    def __init__(self, data_list: List[np.ndarray], num_timesteps_input: int,
                 num_timesteps_output: int, data_range: List[int],
                 contextual_data: Optional[Union[pd.DataFrame, np.ndarray]] = None):
        assert num_timesteps_input % 3 == 0, "num_timesteps_input must be a multiple of 3"
        assert isinstance(data_range, list) and len(data_range) == 2
        assert isinstance(data_list, list)

        self.data_list = data_list
        self.num_sources = len(data_list)
        self.num_nodes = data_list[0].shape[0]
        
        # Process contextual data
        if contextual_data is not None:
            if isinstance(contextual_data, pd.DataFrame):
                time_info = contextual_data.index.to_numpy()
                values = contextual_data.values
                time_embedding = fourier_time_embedding(time_info)
                contextual_data = np.concatenate([time_embedding, values], axis=1)
            contextual_data = np.expand_dims(contextual_data.transpose(1, 0), 1)
            self.data_list = [
                np.concatenate([x, contextual_data], axis=0) 
                for x in self.data_list
            ]
            
        self.day_length = 72  # 72 timesteps per day (15-min intervals)
        slot = num_timesteps_input // 3
        
        dataset_len = data_list[0].shape[2] - (7 * self.day_length + num_timesteps_output) + 1
        
        # Build indices for last week, yesterday, and today
        x_indices = [
            [lw for lw in range(i, i + slot)] +  # Last week
            [yd for yd in range(i + 6 * self.day_length, i + 6 * self.day_length + slot)] +  # Yesterday
            [td for td in range(i + 7 * self.day_length - slot, i + 7 * self.day_length)]  # Today
            for i in range(dataset_len)
        ]
        y_indices = [
            [t for t in range(i + 7 * self.day_length, i + 7 * self.day_length + num_timesteps_output)]
            for i in range(dataset_len)
        ]

        # Apply data range
        start_idx, end_idx = int(data_range[0]), int(data_range[1])
        x_indices = x_indices[start_idx:end_idx]
        y_indices = y_indices[start_idx:end_idx]

        # Build feature and target lists
        self.features_list = [[] for _ in range(self.num_sources)]
        self.targets_list = [[] for _ in range(self.num_sources)]

        for xi, yi in zip(x_indices, y_indices):
            for idx in range(self.num_sources):
                self.features_list[idx].append(self.data_list[idx][:, 0, xi])
                self.targets_list[idx].append(data_list[idx][:, 0, yi])

        # Convert to tensors
        self.features_list = [
            torch.from_numpy(np.array(features)) 
            for features in self.features_list
        ]
        self.targets_list = [
            torch.from_numpy(np.array(targets)) 
            for targets in self.targets_list
        ]

        self.dataset_len = end_idx - start_idx

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample.
        
        Returns:
            Tuple of (features, targets) with shapes:
            - features: (num_sources, num_nodes, num_timesteps_input)
            - targets: (num_sources, num_nodes, num_timesteps_output)
        """
        features = torch.stack([f[idx] for f in self.features_list], dim=0)
        targets = torch.stack([t[idx] for t in self.targets_list], dim=0)
        return features, targets


class MultiSourceDatasetShort(Dataset):
    """Dataset for short-term multi-source demand prediction.
    
    Uses consecutive timesteps for short-term forecasting.
    
    Args:
        data_list: List of numpy arrays.
        num_timesteps_input: Number of input timesteps.
        num_timesteps_output: Number of output timesteps.
        data_range: [start, end] indices for data slicing.
        contextual_data: Optional contextual features.
    """
    
    def __init__(self, data_list: List[np.ndarray], num_timesteps_input: int,
                 num_timesteps_output: int, data_range: List[int],
                 contextual_data: Optional[Union[pd.DataFrame, np.ndarray]] = None):
        assert num_timesteps_input % 3 == 0
        assert isinstance(data_range, list) and len(data_range) == 2
        assert isinstance(data_list, list)

        self.data_list = data_list
        self.num_sources = len(data_list)

        dataset_len = data_list[0].shape[2] - num_timesteps_input - num_timesteps_output + 1

        x_indices = [
            list(range(i, i + num_timesteps_input)) 
            for i in range(dataset_len)
        ]
        y_indices = [
            list(range(i + num_timesteps_input, i + num_timesteps_input + num_timesteps_output))
            for i in range(dataset_len)
        ]
        
        start_idx, end_idx = int(data_range[0]), int(data_range[1])
        x_indices = x_indices[start_idx:end_idx]
        y_indices = y_indices[start_idx:end_idx]

        self.features_list = [[] for _ in range(self.num_sources)]
        self.targets_list = [[] for _ in range(self.num_sources)]

        for xi, yi in zip(x_indices, y_indices):
            for idx, x in enumerate(data_list):
                self.features_list[idx].append(x[:, :, xi].transpose((0, 2, 1)))
                self.targets_list[idx].append(x[:, 0, yi])

        self.features_list = [
            torch.from_numpy(np.array(features)) 
            for features in self.features_list
        ]
        self.targets_list = [
            torch.from_numpy(np.array(targets)) 
            for targets in self.targets_list
        ]

        self.dataset_len = end_idx - start_idx

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = [f[idx] for f in self.features_list]
        targets = [t[idx] for t in self.targets_list]
        return features, targets


def create_dataloaders(
    data_list: List[np.ndarray],
    num_timesteps_input: int,
    num_timesteps_output: int,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.8,
    mode: str = 'long',
    contextual_data: Optional[Union[pd.DataFrame, np.ndarray]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test DataLoaders for multi-source data.
    
    Args:
        data_list: List of numpy arrays, each of shape (num_timesteps, num_nodes).
        num_timesteps_input: Number of input timesteps.
        num_timesteps_output: Number of output timesteps.
        batch_size: Batch size for DataLoaders.
        train_ratio: Ratio of data for training.
        val_ratio: Cumulative ratio for validation (train + val).
        mode: 'long' for BQN mode, 'short' for consecutive timesteps.
        contextual_data: Optional contextual features.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    assert isinstance(data_list, list)

    # Transpose and reshape data
    data_list = [x.T.astype(np.float32) for x in data_list]
    data_list = [x.reshape((x.shape[0], 1, x.shape[1])) for x in data_list]

    # Clip outliers at 99.9th percentile
    for i, x in enumerate(data_list):
        threshold = np.percentile(x, 99.9)
        data_list[i][data_list[i] > threshold] = threshold

    day_length = 72

    if mode == 'long':
        assert num_timesteps_input % 3 == 0
        dataset_len = data_list[0].shape[2] - (7 * day_length + num_timesteps_input // 3 + num_timesteps_output) + 1
        DatasetClass = MultiSourceDatasetLong
    else:
        dataset_len = data_list[0].shape[2] - num_timesteps_input - num_timesteps_output + 1
        DatasetClass = MultiSourceDatasetShort

    # Calculate split points
    train_end = int(dataset_len * train_ratio)
    val_end = int(dataset_len * val_ratio)

    # Create datasets
    train_dataset = DatasetClass(
        data_list,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output,
        data_range=[0, train_end],
        contextual_data=contextual_data
    )
    
    test_dataset = DatasetClass(
        data_list,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output,
        data_range=[val_end, dataset_len],
        contextual_data=contextual_data
    )

    # Create validation dataset
    if train_end < val_end:
        val_dataset = DatasetClass(
            data_list,
            num_timesteps_input=num_timesteps_input,
            num_timesteps_output=num_timesteps_output,
            data_range=[train_end, val_end],
            contextual_data=contextual_data
        )
    else:
        import warnings
        warnings.warn("Validation set is empty, using test set instead.")
        val_dataset = test_dataset

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )
