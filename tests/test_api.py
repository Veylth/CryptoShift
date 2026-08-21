"""Tests for API endpoints."""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from src.api import app
from src.data import database


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def setup_db():
    """Initialize test database."""
    database.init_db()
    
    # Add sample data
    database.add_price_data(
        asset="bitcoin",
        timestamp=datetime.utcnow(),
        price=50000.0,
        volume=1e9,
    )
    
    database.add_alert(
        asset="bitcoin",
        timestamp=datetime.utcnow(),
        price=50000.0,
        volume=1e9,
        detector_name="ensemble",
        confidence=0.85,
    )
    
    yield
    
    # Cleanup (optional)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_get_predictions_valid(client, setup_db):
    """Test getting predictions for valid asset."""
    response = client.get("/api/predictions?asset=bitcoin&lookback=24h")
    
    assert response.status_code == 200
    data = response.json()
    assert data["asset"] == "bitcoin"
    assert "predictions" in data


def test_get_predictions_invalid_asset(client, setup_db):
    """Test invalid asset parameter."""
    response = client.get("/api/predictions?asset=invalid&lookback=24h")
    
    assert response.status_code == 400


def test_get_alerts_valid(client, setup_db):
    """Test getting alerts."""
    response = client.get("/api/alerts?asset=bitcoin&hours=24")
    
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert "count" in data


def test_get_alerts_invalid_hours(client):
    """Test invalid hours parameter."""
    response = client.get("/api/alerts?asset=bitcoin&hours=-1")
    
    assert response.status_code == 400


def test_get_alerts_invalid_confidence(client):
    """Test invalid confidence threshold."""
    response = client.get("/api/alerts?asset=bitcoin&min_confidence=2.0")
    
    assert response.status_code == 400


def test_get_performance_valid(client, setup_db):
    """Test getting performance metrics."""
    # First add some backtest results
    database.add_backtest_result(
        model_name="EnsembleDetector",
        asset="bitcoin",
        fold=1,
        fold_start_date=datetime.utcnow(),
        fold_end_date=datetime.utcnow(),
        precision=0.90,
        recall=0.85,
        f1_score=0.87,
        roc_auc=0.92,
        false_positive_rate=0.05,
        true_positive_rate=0.85,
        n_anomalies_detected=50,
        n_true_positives=42,
        n_false_positives=8,
        n_true_negatives=942,
        n_false_negatives=8,
    )
    
    response = client.get("/api/performance?asset=bitcoin")
    
    assert response.status_code == 200
    data = response.json()
    assert data["asset"] == "bitcoin"
    assert "metrics" in data
    assert "folds" in data


def test_get_performance_invalid_asset(client):
    """Test performance with invalid asset."""
    response = client.get("/api/performance?asset=invalid")
    
    assert response.status_code == 400


def test_backtest_comparison(client, setup_db):
    """Test backtest comparison endpoint."""
    # Add sample results
    for model in ["IsolationForestDetector", "EnsembleDetector"]:
        database.add_backtest_result(
            model_name=model,
            asset="bitcoin",
            fold=1,
            fold_start_date=datetime.utcnow(),
            fold_end_date=datetime.utcnow(),
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            roc_auc=0.89,
            false_positive_rate=0.08,
            true_positive_rate=0.80,
            n_anomalies_detected=45,
            n_true_positives=36,
            n_false_positives=9,
            n_true_negatives=945,
            n_false_negatives=10,
        )
    
    response = client.get("/api/backtest/comparison")
    
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "best_model" in data


def test_verify_alert_valid(client, setup_db):
    """Test verifying an alert."""
    # Get an alert ID first
    alerts = database.get_alerts("bitcoin", hours=24)
    if alerts:
        alert_id = alerts[0].id
        
        response = client.post(f"/api/alerts/{alert_id}/verify?is_real=true")
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data


def test_verify_alert_nonexistent(client):
    """Test verifying non-existent alert."""
    response = client.post("/api/alerts/99999/verify?is_real=true")
    
    assert response.status_code == 404


def test_api_error_handling(client):
    """Test error handling."""
    # Missing required parameter
    response = client.get("/api/predictions")
    
    assert response.status_code in [400, 422]  # Validation error
