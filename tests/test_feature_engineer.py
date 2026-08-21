"""Tests for feature engineering module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.feature_engineer import FeatureEngineer


@pytest.fixture
def feature_engineer():
    """Create feature engineer instance."""
    return FeatureEngineer(
        volatility_window=24,
        momentum_window=24,
        ewma_alpha=0.3,
    )


@pytest.fixture
def sample_data():
    """Create sample price data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    prices = np.linspace(100, 110, 100) + np.random.normal(0, 0.5, 100)
    volumes = np.random.uniform(1e6, 5e6, 100)
    
    return pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': volumes,
    })


def test_compute_rolling_zscore(feature_engineer, sample_data):
    """Test Z-score computation."""
    zscore = feature_engineer.compute_rolling_zscore(sample_data['price'], window=24)
    
    # Check shape
    assert len(zscore) == len(sample_data)
    
    # First few should be NaN (before window)
    assert zscore.isna().sum() > 0
    
    # Later values should not be NaN
    assert not zscore.iloc[-1:].isna().all()


def test_compute_volatility(feature_engineer, sample_data):
    """Test volatility computation."""
    volatility = feature_engineer.compute_volatility(sample_data['price'], window=24)
    
    # Check shape
    assert len(volatility) == len(sample_data)
    
    # Volatility should be positive
    assert (volatility.dropna() >= 0).all()


def test_compute_momentum(feature_engineer, sample_data):
    """Test momentum computation."""
    momentum = feature_engineer.compute_momentum(sample_data['price'], window=24)
    
    # Check shape
    assert len(momentum) == len(sample_data)
    
    # Should be between -1 and 1 approximately
    assert momentum.dropna().min() > -1
    assert momentum.dropna().max() < 1


def test_compute_ewma(feature_engineer, sample_data):
    """Test EWMA computation."""
    ewma = feature_engineer.compute_ewma(sample_data['price'], alpha=0.3)
    
    # Check shape
    assert len(ewma) == len(sample_data)
    
    # EWMA values should be in data range
    assert ewma.min() >= sample_data['price'].min()
    assert ewma.max() <= sample_data['price'].max()


def test_compute_all_features(feature_engineer, sample_data):
    """Test full feature engineering pipeline."""
    result = feature_engineer.compute_all_features(sample_data)
    
    # Check required columns
    required_cols = ['price_zscore', 'volume_zscore', 'volatility_1h', 'momentum_1h']
    for col in required_cols:
        assert col in result.columns
    
    # Check shape
    assert len(result) == len(sample_data)
    
    # Check some values are computed
    assert not result['volatility_1h'].isna().all()
    assert not result['momentum_1h'].isna().all()


def test_missing_columns(feature_engineer):
    """Test error handling for missing columns."""
    bad_data = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=10)})
    
    with pytest.raises(ValueError):
        feature_engineer.compute_all_features(bad_data)


def test_insufficient_data(feature_engineer):
    """Test error handling for small datasets."""
    small_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5),
        'price': [100, 101, 102, 103, 104],
        'volume': [1e6] * 5,
    })
    
    with pytest.raises(ValueError):
        feature_engineer.compute_all_features(small_data)


def test_nan_handling(feature_engineer):
    """Test NaN handling in features."""
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=50),
        'price': [100] * 50,  # Constant price
        'volume': [1e6] * 50,
    })
    
    result = feature_engineer.compute_all_features(data)
    
    # Should handle constant price (zero volatility)
    assert len(result) > 0
