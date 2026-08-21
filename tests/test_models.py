"""Tests for anomaly detection models."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.models.isolation_forest import IsolationForestDetector
from src.models.statistical import ZScoreDetector, EWMADetector
from src.models.ensemble import EnsembleDetector


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    np.random.seed(42)
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100),
        'price': np.random.normal(100, 5, 100),
        'volume': np.random.normal(1e6, 1e5, 100),
        'price_zscore': np.random.normal(0, 1, 100),
        'volume_zscore': np.random.normal(0, 1, 100),
        'volatility_1h': np.random.uniform(0, 0.05, 100),
        'momentum_1h': np.random.uniform(-0.1, 0.1, 100),
    })


def test_isolation_forest_fit_predict(sample_features):
    """Test Isolation Forest training and prediction."""
    detector = IsolationForestDetector()
    
    # Fit on data
    detector.fit(sample_features)
    assert detector.is_fitted
    
    # Predict single row
    pred = detector.predict(sample_features.iloc[:1])
    assert isinstance(pred, dict)
    assert 'is_anomaly' in pred
    assert 'anomaly_score' in pred
    assert 'confidence' in pred
    assert 0.0 <= pred['confidence'] <= 1.0


def test_isolation_forest_batch_predict(sample_features):
    """Test batch predictions."""
    detector = IsolationForestDetector()
    detector.fit(sample_features)
    
    preds = detector.predict_batch(sample_features)
    assert len(preds) == len(sample_features)
    
    for pred in preds:
        assert isinstance(pred, dict)
        assert 0.0 <= pred['confidence'] <= 1.0


def test_zscore_detector(sample_features):
    """Test Z-score detector."""
    detector = ZScoreDetector(threshold=2.0)
    detector.fit(sample_features)
    
    # Predict
    pred = detector.predict(sample_features.iloc[:1])
    
    assert 'is_anomaly' in pred
    assert 'zscore' in pred
    assert 'confidence' in pred


def test_ewma_detector(sample_features):
    """Test EWMA detector."""
    detector = EWMADetector(alpha=0.3, std_threshold=2.0)
    detector.fit(sample_features)
    
    # Predict
    pred = detector.predict(sample_features.iloc[:1])
    
    assert 'is_anomaly' in pred
    assert 'deviation' in pred
    assert 'confidence' in pred


def test_ensemble_detector(sample_features):
    """Test ensemble detector."""
    detectors_list = [
        IsolationForestDetector(),
        ZScoreDetector(),
        EWMADetector(),
    ]
    
    ensemble = EnsembleDetector(detectors_list=detectors_list, voting_threshold=2)
    ensemble.fit(sample_features)
    
    # Predict
    pred = ensemble.predict(sample_features.iloc[:1])
    
    assert 'is_anomaly' in pred
    assert 'ensemble_confidence' in pred
    assert 'votes' in pred
    assert 'agreement' in pred


def test_ensemble_voting_logic(sample_features):
    """Test ensemble voting logic."""
    detectors_list = [
        IsolationForestDetector(),
        ZScoreDetector(),
        EWMADetector(),
    ]
    
    ensemble = EnsembleDetector(detectors_list=detectors_list, voting_threshold=2)
    ensemble.fit(sample_features)
    
    pred = ensemble.predict(sample_features.iloc[:1])
    
    # Check voting
    votes = pred['votes']
    agreement = pred['agreement']
    
    # Agreement should match vote count
    expected_agreement = sum(1 for v in votes.values() if v)
    assert agreement == expected_agreement


def test_unfitted_model_error(sample_features):
    """Test error handling for unfitted models."""
    detector = IsolationForestDetector()
    
    with pytest.raises(ValueError):
        detector.predict(sample_features.iloc[:1])


def test_model_persistence(sample_features, tmp_path):
    """Test model save/load."""
    # Train and save
    detector = IsolationForestDetector()
    detector.fit(sample_features)
    
    model_path = tmp_path / "model.pkl"
    detector.save_model(str(model_path))
    
    # Load and predict
    detector2 = IsolationForestDetector()
    detector2.load_model(str(model_path))
    
    assert detector2.is_fitted
    
    # Predictions should work
    pred = detector2.predict(sample_features.iloc[:1])
    assert 'is_anomaly' in pred


def test_empty_data_handling():
    """Test handling of empty DataFrames."""
    detector = IsolationForestDetector()
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        detector.fit(empty_df)


def test_all_positive_confidence(sample_features):
    """Test that all confidence scores are positive."""
    detector = EnsembleDetector(
        detectors_list=[
            IsolationForestDetector(),
            ZScoreDetector(),
            EWMADetector(),
        ]
    )
    detector.fit(sample_features)
    
    preds = detector.predict_batch(sample_features)
    
    for pred in preds:
        assert 0.0 <= pred['ensemble_confidence'] <= 1.0
