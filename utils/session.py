"""Training session utilities for ST-MoE-RMQRN.

This module provides training and evaluation session classes for
managing the training loop, validation, and testing of models.
"""

import json
import math
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.metrics import (
    mae_loss,
    compute_r_squared,
    compute_accuracy,
    compute_mae,
    compute_rmse,
    compute_zero_demand_f1,
    compute_interval_metrics,
)


class TrainingSession:
    """Base training session for managing model training and evaluation.
    
    Args:
        model: PyTorch model to train.
        loss_fn: Loss function.
        optimizer: Optional optimizer (default: AdamW).
        device: Computation device.
        early_stopping_patience: Patience for early stopping.
        env_config: Environment configuration dictionary.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        early_stopping_patience: int = 100,
        env_config: Optional[Dict] = None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.device = device
        self.model = model.to(device)
        self.current_epoch = 0
        
        if optimizer is None:
            optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.optimizer = optimizer
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.88
        )
        
        self.loss_fn = loss_fn
        self.early_stopping_patience = max(early_stopping_patience, 30)
        
        self.train_losses = []
        self.val_losses = []
        
        # Environment configuration
        env_config = env_config or {}
        self.city_name = env_config.get('city_name', 'unknown')
        self.time_granularity = env_config.get('time_granularity', 15)
        self.data_root = env_config.get('data_root', 'Datasets/')
        self.checkpoint_root = env_config.get('checkpoint_root', 'checkpoints/')
        self.data_path = env_config.get('data_path', '')
        self.checkpoint_path = env_config.get('checkpoint_path', '')

    def _forward_batch(
        self, 
        x_batch: torch.Tensor, 
        y_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple]:
        """Process a single batch through the model.
        
        Args:
            x_batch: Input batch.
            y_batch: Target batch.
            
        Returns:
            Tuple of (loss, (x_batch, y_batch, predictions)).
        """
        raise NotImplementedError

    def _train_epoch(self, train_loader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Average training loss.
        """
        epoch_losses = []
        for x_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            loss, _ = self._forward_batch(x_batch, y_batch)
            
            # L2 regularization
            loss = loss + 1e-4 * sum(
                torch.norm(param) for param in self.model.parameters()
            )
            
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.detach().cpu().item())
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.train_losses.append(avg_loss)
        return avg_loss

    def _validate(self, val_loader) -> Tuple[float, float, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Tuple of (loss, MAE, R2).
        """
        val_losses = []
        mae_values = []
        r2_values = []
        picp_values = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                loss, (_, y_true, y_pred) = self._forward_batch(x_batch, y_batch)
                val_losses.append(loss.detach().cpu().item())
                
                y_hat = self._select_point_prediction(y_pred)
                mae_values.append(
                    mae_loss(y_true, y_hat).detach().cpu().item()
                )
                r2_values.append(
                    compute_r_squared(
                        y_true.detach().cpu().numpy(),
                        y_hat.detach().cpu().numpy()
                    )
                )
                
                # Compute PICP for outermost quantiles
                y_pred_np = y_pred.detach().cpu().numpy()
                y_true_np = y_true.detach().cpu().numpy()
                picp = np.sum(
                    np.logical_and(
                        np.less_equal(y_pred_np[:, :, 0, ...], y_true_np),
                        np.greater_equal(y_pred_np[:, :, -1, ...], y_true_np)
                    )
                ) / np.size(y_true_np)
                picp_values.append(picp)
                
        print(f"PICP: {sum(picp_values) / len(picp_values):.4f}")
        
        avg_loss = sum(val_losses) / len(val_losses)
        avg_mae = sum(mae_values) / len(mae_values)
        avg_r2 = sum(r2_values) / len(r2_values)
        
        self.val_losses.append(avg_loss)
        return avg_loss, avg_mae, avg_r2

    def _select_point_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Select point prediction from quantile predictions.
        
        Args:
            y_pred: Quantile predictions.
            
        Returns:
            Point prediction (median).
        """
        return y_pred

    def save_checkpoint(self, path: str):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
        """
        torch.save(self.model, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint.
        """
        self.model = torch.load(path, weights_only=False)
        self.model.to(self.device)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 300,
        save_path: Optional[str] = None
    ):
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            save_path: Path to save best model.
        """
        self.train_losses = []
        self.val_losses = []
        best_val_loss = math.inf
        patience_counter = 0
        
        if save_path is None:
            save_path = os.path.join(
                self.checkpoint_path,
                f'{self.city_name}_{self.time_granularity}_{epochs}epoch.pth'
            )
            warnings.warn(f"Save path not specified, model saved to {save_path}")
            
        for epoch in range(epochs):
            self.current_epoch = epoch
            patience_counter += 1
            
            # Training
            self.model.train()
            train_loss = self._train_epoch(train_loader)
            
            # Validation
            self.model.eval()
            val_loss, mae, r2 = self._validate(val_loader)
            
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(save_path)
                patience_counter = 0
                
            # Early stopping
            if patience_counter > self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            print(
                f'Epoch: {epoch}; TrainLoss {train_loss:.2f}; ValLoss {val_loss:.2f}; '
                f'lr {self.optimizer.param_groups[0]["lr"]:.6f}; '
                f'MAE {mae:.5f}; R2 {r2:.5f}'
            )

    def _compute_deterministic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        results: Dict
    ) -> Dict:
        """Compute deterministic evaluation metrics.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            results: Dictionary to store results.
            
        Returns:
            Updated results dictionary.
        """
        results['R2'].append(compute_r_squared(y_true, y_pred))
        results['Accuracy'].append(compute_accuracy(y_true, y_pred))
        
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        results['MAE'].append(compute_mae(y_true_flat, y_pred_flat))
        results['RMSE'].append(compute_rmse(y_true_flat, y_pred_flat))
        results['F1(on 0-demand)'].append(
            compute_zero_demand_f1(y_true_flat, y_pred_flat)
        )
        
        return results

    def _compute_probabilistic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantile_list: List[float],
        results: Dict
    ) -> Dict:
        """Compute probabilistic evaluation metrics.
        
        Args:
            y_true: Ground truth values.
            y_pred: Quantile predictions.
            quantile_list: List of quantile levels.
            results: Dictionary to store results.
            
        Returns:
            Updated results dictionary.
        """
        assert len(quantile_list) % 2 == 1, "Quantile list should have odd length"
        
        quantile_labels = [f'{int(q * 100)}' for q in quantile_list]
        y_true_flat = y_true.ravel()
        
        for i in range(len(quantile_list) // 2):
            coverage = quantile_list[-i - 1] - quantile_list[i]
            mpiw, picp, cwc = compute_interval_metrics(
                y_true_flat,
                y_pred[:, :, i, ...].ravel(),
                y_pred[:, :, -i - 1, ...].ravel(),
                coverage
            )
            
            label = f'{quantile_labels[i]}-{quantile_labels[-i - 1]}%'
            results[f'MPIW@{label}'] = mpiw
            results[f'PICP@{label}'] = picp
            results[f'CWC@{label}'] = cwc
            
        return results

    def test(
        self,
        test_loader,
        checkpoint_path: str,
        result_path: str,
        evaluation_mode: str = 'deterministic'
    ) -> Dict:
        """Test the model and save results.
        
        Args:
            test_loader: Test data loader.
            checkpoint_path: Path to model checkpoint.
            result_path: Path to save results (JSON).
            evaluation_mode: 'deterministic' or 'probabilistic'.
            
        Returns:
            Results dictionary.
        """
        self.load_checkpoint(checkpoint_path)
        
        assert evaluation_mode in ['deterministic', 'probabilistic'], \
            "evaluation_mode should be 'deterministic' or 'probabilistic'"

        results = {
            'MAE': [],
            'RMSE': [],
            'R2': [],
            'Accuracy': [],
            'F1(on 0-demand)': []
        }

        with torch.no_grad():
            all_y, all_y_pred = [], []
            
            for x_batch, y_batch in test_loader:
                _, (_, y_batch_out, y_pred) = self._forward_batch(x_batch, y_batch)
                
                y_batch_np = [y.cpu().detach().numpy() for y in y_batch_out]
                if isinstance(y_pred[0], torch.Tensor):
                    y_pred_np = [y.cpu().detach().numpy() for y in y_pred]
                else:
                    y_pred_np = [
                        [y.cpu().detach().numpy() for y in yp] 
                        for yp in y_pred
                    ]
                    
                all_y.append(y_batch_np)
                all_y_pred.append(y_pred_np)
                
            all_y = np.concatenate(all_y, axis=0)
            all_y_pred = np.concatenate(all_y_pred, axis=0)
            
            y_point = self._select_point_prediction(all_y_pred)
            
            results = self._compute_deterministic_metrics(all_y, y_point, results)
            
            if evaluation_mode == 'probabilistic':
                quantiles = [
                    0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                    0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99
                ]
                results = self._compute_probabilistic_metrics(
                    all_y, all_y_pred, quantiles, results
                )

            # Save predictions
            np.savez_compressed(
                result_path.replace('.json', '.npz'),
                y_true=all_y,
                y_pred=all_y_pred
            )

        # Average results
        results = {key: float(np.mean(val)) for key, val in results.items()}

        # Save to JSON
        if result_path.endswith('.json'):
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            raise ValueError("Result path should be a JSON file path.")
            
        return results


class QuantileTrainingSession(TrainingSession):
    """Training session for quantile regression models.
    
    Args:
        model: PyTorch model.
        loss_fn: Loss function (joint pinball loss).
        quantile_list: List of quantile levels.
        optimizer: Optional optimizer.
        device: Computation device.
        early_stopping_patience: Patience for early stopping.
        env_config: Environment configuration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        quantile_list: List[float],
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        early_stopping_patience: int = 100,
        env_config: Optional[Dict] = None
    ):
        super().__init__(
            model, loss_fn, optimizer, device, 
            early_stopping_patience, env_config
        )
        self.quantile_list = quantile_list

    def _forward_batch(
        self, 
        x_batch: torch.Tensor, 
        y_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple]:
        """Process a batch through the model.
        
        Args:
            x_batch: Input batch.
            y_batch: Target batch.
            
        Returns:
            Tuple of (loss, (x_batch, y_batch, predictions)).
        """
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        y_pred = self.model(x_batch)
        
        # Compute loss
        loss = self.loss_fn(y_batch, y_pred, self.quantile_list, 2, False)
        
        # L1 regularization
        loss = loss + 0.001 * sum(
            torch.norm(param, p=1) for param in self.model.parameters()
        )
        
        return loss, (x_batch, y_batch, y_pred)

    def _select_point_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Select median as point prediction.
        
        Args:
            y_pred: Quantile predictions.
            
        Returns:
            Median predictions as torch tensor.
        """
        # Select the middle quantile (median)
        return y_pred[:, :, y_pred.shape[2] // 2, ...]
