"""Evaluation metrics for anomaly detection."""

from typing import Tuple, Dict, List, Any
import logging

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc

from src.config import logger as config_logger

logger = config_logger


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Compute confusion matrix elements.
    
    Args:
        y_true: Ground truth labels (1 for anomaly, 0 for normal)
        y_pred: Predicted labels (1 for anomaly, 0 for normal)
        
    Returns:
        Tuple: (TP, FP, TN, FN)
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return int(tp), int(fp), int(tn), int(fn)


def precision(tp: int, fp: int) -> float:
    """Compute precision metric.
    
    Precision = TP / (TP + FP)
    
    Answers: Of the predicted positives, how many were correct?
    
    Args:
        tp: True positives
        fp: False positives
        
    Returns:
        float: Precision (0 to 1)
    """
    denominator = tp + fp
    if denominator == 0:
        return 0.0
    
    return float(tp) / float(denominator)


def recall(tp: int, fn: int) -> float:
    """Compute recall metric.
    
    Recall = TP / (TP + FN)
    
    Answers: Of the actual positives, how many were detected?
    
    Args:
        tp: True positives
        fn: False negatives
        
    Returns:
        float: Recall (0 to 1)
    """
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    
    return float(tp) / float(denominator)


def f1_score(precision_val: float, recall_val: float) -> float:
    """Compute F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Harmonic mean of precision and recall.
    
    Args:
        precision_val: Precision value
        recall_val: Recall value
        
    Returns:
        float: F1 score (0 to 1)
    """
    denominator = precision_val + recall_val
    if denominator == 0:
        return 0.0
    
    return 2.0 * (precision_val * recall_val) / denominator


def false_positive_rate(fp: int, tn: int) -> float:
    """Compute false positive rate.
    
    FPR = FP / (FP + TN)
    
    What fraction of negatives were incorrectly classified as positive?
    
    Args:
        fp: False positives
        tn: True negatives
        
    Returns:
        float: FPR (0 to 1)
    """
    denominator = fp + tn
    if denominator == 0:
        return 0.0
    
    return float(fp) / float(denominator)


def true_positive_rate(tp: int, fn: int) -> float:
    """Compute true positive rate (same as recall).
    
    TPR = TP / (TP + FN)
    
    Args:
        tp: True positives
        fn: False negatives
        
    Returns:
        float: TPR (0 to 1)
    """
    return recall(tp, fn)


def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC AUC score.
    
    Uses sklearn's roc_auc_score.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Confidence scores (0 to 1)
        
    Returns:
        float: ROC AUC score (0 to 1)
    """
    try:
        y_true = np.asarray(y_true, dtype=int)
        y_scores = np.asarray(y_scores, dtype=float)
        
        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            logger.warning("ROC AUC: Only one class in y_true")
            return 0.5
        
        score = roc_auc_score(y_true, y_scores)
        return float(score)
        
    except Exception as e:
        logger.error(f"Error computing ROC AUC: {e}")
        return 0.5


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels (1 for anomaly, 0 for normal)
        y_pred: Predicted labels (1 or 0)
        y_scores: Confidence scores (0 to 1)
        
    Returns:
        Dict: All metrics
            {
                'precision': float,
                'recall': float,
                'f1': float,
                'fpr': float,
                'tpr': float,
                'roc_auc': float,
                'tp': int,
                'fp': int,
                'tn': int,
                'fn': int,
            }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_scores = np.asarray(y_scores)
    
    # Compute confusion matrix
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    
    # Compute metrics
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)
    fpr = false_positive_rate(fp, tn)
    tpr = true_positive_rate(tp, fn)
    auc_score = roc_auc(y_true, y_scores)
    
    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': auc_score,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


def aggregate_metrics(
    list_of_metric_dicts: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Aggregate metrics across multiple folds.
    
    Computes mean and standard deviation for each metric.
    
    Args:
        list_of_metric_dicts: List of metric dicts (one per fold)
        
    Returns:
        Dict: {
            metric_name: {
                'mean': float,
                'std': float,
            }
        }
    """
    if not list_of_metric_dicts:
        return {}
    
    # Get all metric keys
    all_keys = set()
    for metric_dict in list_of_metric_dicts:
        all_keys.update(metric_dict.keys())
    
    aggregated = {}
    
    for key in all_keys:
        values = []
        for metric_dict in list_of_metric_dicts:
            if key in metric_dict:
                val = metric_dict[key]
                # Skip count metrics (tp, fp, tn, fn)
                if key not in ['tp', 'fp', 'tn', 'fn']:
                    values.append(val)
        
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
    
    # Aggregate count metrics (sum them)
    for count_key in ['tp', 'fp', 'tn', 'fn']:
        total = sum(
            metric_dict.get(count_key, 0)
            for metric_dict in list_of_metric_dicts
        )
        aggregated[count_key] = {
            'total': total,
        }
    
    return aggregated


def format_results(results_dict: Dict[str, Any]) -> str:
    """Format results for console output.
    
    Args:
        results_dict: Results dictionary
        
    Returns:
        str: Formatted string
    """
    output = []
    output.append("=" * 70)
    output.append("EVALUATION RESULTS")
    output.append("=" * 70)
    
    for key, value in results_dict.items():
        if isinstance(value, dict):
            output.append(f"{key}:")
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, float):
                    output.append(f"  {sub_key}: {sub_val:.4f}")
                else:
                    output.append(f"  {sub_key}: {sub_val}")
        else:
            if isinstance(value, float):
                output.append(f"{key}: {value:.4f}")
            else:
                output.append(f"{key}: {value}")
    
    output.append("=" * 70)
    return "\n".join(output)
