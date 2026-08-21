"""Pydantic schemas for API responses."""

from datetime import datetime
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """Single prediction result."""
    
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    price: float = Field(..., description="Asset price at prediction time")
    volume: float = Field(..., description="Trading volume")
    is_anomaly: bool = Field(..., description="Whether anomaly detected")
    ensemble_confidence: float = Field(..., description="Confidence score (0-1)")
    detector_votes: Dict[str, bool] = Field(..., description="Individual detector votes")


class PredictionResponse(BaseModel):
    """Response for predictions endpoint."""
    
    asset: str = Field(..., description="Asset name (e.g., 'bitcoin')")
    lookback: str = Field(..., description="Lookback period (e.g., '24h')")
    predictions: List[PredictionItem] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions")


class AlertItem(BaseModel):
    """Single alert record."""
    
    id: int = Field(..., description="Alert ID")
    timestamp: str = Field(..., description="Alert timestamp (ISO format)")
    asset: str = Field(..., description="Asset name")
    price: float = Field(..., description="Price at alert time")
    volume: float = Field(..., description="Volume at alert time")
    detector_name: str = Field(..., description="Detector that triggered alert")
    confidence: float = Field(..., description="Confidence score (0-1)")
    is_real_anomaly: Optional[bool] = Field(
        None, description="User verification (True/False/None for unverified)"
    )


class AlertResponse(BaseModel):
    """Response for alerts endpoint."""
    
    alerts: List[AlertItem] = Field(..., description="List of alerts")
    count: int = Field(..., description="Total alert count")


class PerformanceMetrics(BaseModel):
    """Backtest performance metrics."""
    
    precision: float = Field(..., description="Precision metric")
    recall: float = Field(..., description="Recall metric")
    f1_score: float = Field(..., description="F1 score")
    roc_auc: float = Field(..., description="ROC AUC score")
    false_positive_rate: float = Field(..., description="False positive rate")


class BacktestFoldResult(BaseModel):
    """Results for single backtest fold."""
    
    fold: int = Field(..., description="Fold number")
    precision: float = Field(..., description="Precision")
    recall: float = Field(..., description="Recall")
    f1: float = Field(..., description="F1 score")
    roc_auc: float = Field(..., description="ROC AUC")
    fpr: float = Field(..., description="False positive rate")
    start_date: Optional[str] = Field(None, description="Fold start date")
    end_date: Optional[str] = Field(None, description="Fold end date")


class PerformanceResponse(BaseModel):
    """Response for performance endpoint."""
    
    asset: str = Field(..., description="Asset name")
    metrics: PerformanceMetrics = Field(..., description="Aggregate metrics")
    folds: List[BacktestFoldResult] = Field(..., description="Per-fold results")


class ModelComparisonMetrics(BaseModel):
    """Metrics for single model in comparison."""
    
    model_name: str = Field(..., description="Model name")
    precision: float = Field(..., description="Mean precision")
    recall: float = Field(..., description="Mean recall")
    f1: float = Field(..., description="Mean F1 score")
    roc_auc: float = Field(..., description="Mean ROC AUC")
    fpr: float = Field(..., description="Mean false positive rate")
    num_folds: int = Field(..., description="Number of folds tested")


class BacktestComparisonResponse(BaseModel):
    """Response for backtest comparison endpoint."""
    
    models: List[ModelComparisonMetrics] = Field(..., description="Models and their metrics")
    best_model: str = Field(..., description="Best performing model (by F1)")


class ErrorResponse(BaseModel):
    """Error response format."""
    
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error detail message")
    timestamp: Optional[str] = Field(None, description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="System status (ok/degraded/error)")
    timestamp: str = Field(..., description="Response timestamp")
    uptime_seconds: Optional[int] = Field(None, description="System uptime in seconds")
