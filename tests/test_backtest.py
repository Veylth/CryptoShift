"""Tests for backtesting framework."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.backtesting.backtest import BacktestFramework
from src.backtesting.metrics import (
    confusion_matrix,
    precision,
    recall,
    f1_score,
    false_positive_rate,
    roc_auc,
)


@pytest.fixture
def backtest_framework():
    """Create backtest framework."""
    return BacktestFramework(train_months=3, test_months=1)


@pytest.fixture
def time_series_data():
    """Create multi-month time series data."""
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='1h')
    
    return pd.DataFrame({
        'timestamp': dates,
        'price': np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + 100 + np.random.normal(0, 0.5, len(dates)),
        'volume': np.random.uniform(1e6, 5e6, len(dates)),
        'price_zscore': np.random.normal(0, 1, len(dates)),
        'volume_zscore': np.random.normal(0, 1, len(dates)),
        'volatility_1h': np.random.uniform(0, 0.05, len(dates)),
        'momentum_1h': np.random.uniform(-0.1, 0.1, len(dates)),
    })


def test_walk_forward_split(backtest_framework, time_series_data):
    """Test walk-forward splitting logic."""
    folds = list(backtest_framework.walk_forward_split(time_series_data))
    
    # Should have multiple folds
    assert len(folds) > 0
    
    for train_df, test_df, fold_num in folds:
        # Check no overlap
        train_max = train_df['timestamp'].max()
        test_min = test_df['timestamp'].min()
        assert train_max < test_min, "Training and test data overlap!"
        
        # Check both have data
        assert len(train_df) > 0
        assert len(test_df) > 0


def test_walk_forward_order(backtest_framework, time_series_data):
    """Test that test period comes after training period."""
    for train_df, test_df, fold_num in backtest_framework.walk_forward_split(time_series_data):
        # Test period should be later than training
        assert test_df['timestamp'].min() > train_df['timestamp'].max()


def test_insufficient_data_error(backtest_framework):
    """Test error handling for insufficient data."""
    small_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10),
        'price': np.random.normal(100, 5, 10),
    })
    
    with pytest.raises(ValueError):
        list(backtest_framework.walk_forward_split(small_df))


def test_missing_timestamp_error(backtest_framework):
    """Test error handling for missing timestamp column."""
    bad_df = pd.DataFrame({'price': [100, 101, 102]})
    
    with pytest.raises(ValueError):
        list(backtest_framework.walk_forward_split(bad_df))


def test_label_anomalies_by_percentile(backtest_framework):
    """Test anomaly labeling."""
    df = pd.DataFrame({
        'anomaly_score': np.linspace(0, 100, 100),
    })
    
    labels = backtest_framework.label_anomalies_by_percentile(df, percentile=0.95)
    
    # Should label top 5%
    assert labels.sum() == 5
    assert labels[-1] == 1  # Highest score is anomaly


# Metrics tests

def test_confusion_matrix():
    """Test confusion matrix calculation."""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    
    # Manual verification
    assert tp == 2  # Correct positives
    assert fp == 1  # Wrong positives
    assert tn == 2  # Correct negatives
    assert fn == 1  # Wrong negatives


def test_precision_metric():
    """Test precision calculation."""
    assert precision(8, 2) == 0.8  # 8 / (8 + 2) = 0.8
    assert precision(0, 5) == 0.0
    assert precision(5, 0) == 1.0


def test_recall_metric():
    """Test recall calculation."""
    assert recall(8, 2) == 0.8  # 8 / (8 + 2) = 0.8
    assert recall(0, 5) == 0.0
    assert recall(5, 0) == 1.0


def test_f1_score_metric():
    """Test F1 score calculation."""
    # Perfect: F1 = 1.0
    assert f1_score(1.0, 1.0) == 1.0
    
    # Both zero: F1 = 0.0
    assert f1_score(0.0, 0.0) == 0.0
    
    # 0.8, 0.8: F1 = 0.8
    assert f1_score(0.8, 0.8) == 0.8


def test_false_positive_rate():
    """Test FPR calculation."""
    assert false_positive_rate(5, 95) == 0.05  # 5 / (5 + 95) = 0.05
    assert false_positive_rate(0, 100) == 0.0


def test_roc_auc_single_class():
    """Test ROC AUC with single class handling."""
    y_true = np.array([0, 0, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4])
    
    # Should handle gracefully
    score = roc_auc(y_true, y_scores)
    assert 0.0 <= score <= 1.0
