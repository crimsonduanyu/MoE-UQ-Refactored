"""Metrics and loss functions for ST-MoE-RMQRN.

This module provides loss functions for quantile regression and
evaluation metrics for both deterministic and probabilistic predictions.
"""

from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, f1_score


def pinball_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float,
    huber_delta: float = 0.2,
    zero_inflated: bool = True
) -> torch.Tensor:
    """Compute pinball (quantile) loss with optional Huber smoothing.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        alpha: Quantile level in (0, 1).
        huber_delta: Delta for Huber loss smoothing. None for L1 loss.
        zero_inflated: Whether to apply zero-inflation weighting.
        
    Returns:
        Pinball loss value.
    """
    support = torch.ones_like(y_true)
    
    if zero_inflated:
        zero_mask = y_true == 0
        zero_ratio = torch.sum(zero_mask) / y_true.nelement() + 1e-4
        nonzero_ratio = 1 - zero_ratio + 2e-4
        support[zero_mask] = zero_ratio
        support[~zero_mask] = nonzero_ratio

    if huber_delta is None:
        loss_fn = nn.L1Loss(reduction='none')
    else:
        loss_fn = nn.HuberLoss(reduction='none', delta=huber_delta)
        
    pred_upper = y_pred >= y_true
    pred_lower = ~pred_upper
    
    loss = (
        torch.sum(alpha * loss_fn(y_true[pred_lower], y_pred[pred_lower]) / support[pred_lower]) +
        torch.sum((1 - alpha) * loss_fn(y_true[pred_upper], y_pred[pred_upper]) / support[pred_upper])
    )
    
    return loss


def joint_pinball_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantile_list: list,
    huber_delta: float = None,
    zero_inflated: bool = False
) -> torch.Tensor:
    """Compute joint pinball loss for multiple quantiles.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted quantiles of shape (..., num_quantiles, ...).
        quantile_list: List of quantile levels.
        huber_delta: Delta for Huber loss smoothing.
        zero_inflated: Whether to apply zero-inflation weighting.
        
    Returns:
        Joint pinball loss value.
    """
    total_loss = torch.tensor(0.0, device=y_true.device)
    
    for idx, alpha in enumerate(quantile_list):
        y_hat = y_pred[:, :, idx, ...]
        total_loss = total_loss + pinball_loss(y_true, y_hat, alpha, huber_delta, zero_inflated)
    
    return total_loss


def compute_coverage_width_criterion(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.90,
    penalty_weight: float = 1.0
) -> float:
    """Compute Coverage Width Criterion (CWC).
    
    CWC balances prediction interval width with coverage, penalizing
    intervals that fail to meet the target coverage.
    
    Args:
        y_true: Ground truth values.
        lower: Lower bound predictions.
        upper: Upper bound predictions.
        target_coverage: Target coverage rate (PICP).
        penalty_weight: Weight for coverage penalty.
        
    Returns:
        CWC value.
    """
    y_true = np.asarray(y_true).ravel()
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()
    
    # Compute PICP (Prediction Interval Coverage Probability)
    covered1 = np.logical_and(y_true >= lower, y_true <= upper)
    covered2 = np.logical_and(y_true <= lower, y_true >= upper)
    covered = np.logical_or(covered1, covered2)
    picp = np.mean(covered)
    
    # Compute PINAW (Prediction Interval Normalized Average Width)
    widths = np.abs(upper - lower)
    mean_width = np.mean(widths)
    y_range = np.max(y_true) - np.min(y_true)
    pinaw = mean_width / y_range if y_range != 0 else mean_width
    
    # Compute CWC
    gamma = 0 if picp >= target_coverage else 1
    exp_term = np.exp(-penalty_weight * (picp - target_coverage))
    cwc = pinaw * (1 + gamma * exp_term)
    
    return cwc


def compute_interval_metrics(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    target_coverage: float = 0.9,
    penalty_weight: float = 10.0
) -> tuple:
    """Compute prediction interval metrics: MPIW, PICP, and CWC.
    
    Args:
        y_true: Ground truth values.
        y_pred_lower: Lower bound predictions.
        y_pred_upper: Upper bound predictions.
        target_coverage: Target coverage rate.
        penalty_weight: Weight for CWC penalty.
        
    Returns:
        Tuple of (MPIW, PICP, CWC).
    """
    y_true = np.array(y_true).ravel()
    y_pred_upper = np.array(y_pred_upper).ravel()
    y_pred_lower = np.array(y_pred_lower).ravel()

    # PICP (Prediction Interval Coverage Probability)
    hits = (
        (np.greater_equal(y_true, y_pred_lower) & np.less_equal(y_true, y_pred_upper)) |
        (np.greater_equal(y_pred_lower, y_true) & np.less_equal(y_pred_upper, y_true))
    )
    picp = np.mean(hits.astype(np.float32))
    
    # MPIW (Mean Prediction Interval Width)
    mpiw = np.mean(np.abs(y_pred_upper - y_pred_lower))
    
    # CWC (Coverage Width Criterion)
    cwc = compute_coverage_width_criterion(
        y_true, y_pred_lower, y_pred_upper, 
        target_coverage, penalty_weight
    )
    
    return mpiw, picp, cwc


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared score.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        
    Returns:
        R-squared score.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return r2_score(y_true, y_pred, force_finite=True)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute prediction accuracy (for count data).
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        
    Returns:
        Accuracy score.
    """
    y_true = np.round(y_true.ravel())
    y_pred = np.round(y_pred.ravel())
    return np.mean(np.equal(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        
    Returns:
        MAE value.
    """
    y_pred = y_pred.copy()
    y_pred[y_pred < 0.5] = 0
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        
    Returns:
        RMSE value.
    """
    y_pred = y_pred.copy()
    y_pred[y_pred < 0.5] = 0
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def compute_zero_demand_f1(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    threshold: float = 0.5
) -> float:
    """Compute F1 score for zero-demand prediction.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        threshold: Threshold for zero classification.
        
    Returns:
        Weighted F1 score.
    """
    true_zeros = y_true == 0
    pred_zeros = y_pred < threshold
    return f1_score(true_zeros, pred_zeros, average='weighted')


# PyTorch loss for training
mae_loss = nn.L1Loss(reduction='mean')


