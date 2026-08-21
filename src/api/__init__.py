"""FastAPI application for CryptoShift."""

from datetime import datetime, timedelta
from typing import Optional, List
import logging
import time

from fastapi import FastAPI, Query, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from src.config import (
    API_PORT,
    ASSETS,
    ASSET_SYMBOLS,
    logger as config_logger,
)
from src.data import database
from .schemas import (
    PredictionResponse,
    PredictionItem,
    AlertResponse,
    AlertItem,
    PerformanceResponse,
    PerformanceMetrics,
    BacktestFoldResult,
    BacktestComparisonResponse,
    ModelComparisonMetrics,
    ErrorResponse,
    HealthResponse,
)

logger = config_logger

# Initialize FastAPI app
app = FastAPI(
    title="CryptoShift API",
    description="Real-time cryptocurrency anomaly detection API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application state
app_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        logger.info("Starting CryptoShift API...")
        database.init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        HealthResponse: System status
    """
    uptime = int(time.time() - app_start_time)
    
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=uptime,
    )


@app.get("/api/predictions", response_model=PredictionResponse)
async def get_predictions(
    asset: str = Query(..., description="Asset name (e.g., 'bitcoin')"),
    lookback: str = Query("24h", description="Lookback period (1h, 24h, 7d)"),
) -> PredictionResponse:
    """Get recent predictions/anomaly detections.
    
    Args:
        asset: Asset name
        lookback: Time lookback period
        
    Returns:
        PredictionResponse: List of predictions
        
    Raises:
        HTTPException: 400 if invalid parameters, 404 if no data
    """
    # Validate asset
    if asset not in ASSETS:
        logger.warning(f"Invalid asset requested: {asset}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid asset. Must be one of: {ASSETS}",
        )
    
    # Parse lookback period
    lookback_hours = 24
    if lookback.endswith("h"):
        lookback_hours = int(lookback[:-1])
    elif lookback.endswith("d"):
        lookback_hours = int(lookback[:-1]) * 24
    
    try:
        # Fetch recent alerts (predictions)
        alerts = database.get_alerts(asset, hours=lookback_hours)
        
        if not alerts:
            logger.info(f"No predictions found for {asset} in last {lookback_hours}h")
            raise HTTPException(
                status_code=404,
                detail=f"No predictions available for {asset}",
            )
        
        # Convert to response format
        predictions = []
        for alert in alerts:
            predictions.append(
                PredictionItem(
                    timestamp=alert.timestamp.isoformat(),
                    price=alert.price,
                    volume=alert.volume,
                    is_anomaly=True,
                    ensemble_confidence=alert.confidence,
                    detector_votes={alert.detector_name: True},
                )
            )
        
        logger.info(f"Returned {len(predictions)} predictions for {asset}")
        
        return PredictionResponse(
            asset=asset,
            lookback=lookback,
            predictions=predictions,
            count=len(predictions),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/alerts", response_model=AlertResponse)
async def get_alerts(
    asset: Optional[str] = Query(None, description="Filter by asset"),
    hours: int = Query(24, description="Lookback hours"),
    limit: int = Query(100, description="Max alerts to return"),
    offset: int = Query(0, description="Pagination offset"),
    min_confidence: float = Query(0.0, description="Minimum confidence threshold"),
) -> AlertResponse:
    """Get recent anomaly alerts.
    
    Args:
        asset: Filter by asset (optional)
        hours: Lookback hours
        limit: Max results
        offset: Pagination offset
        min_confidence: Minimum confidence filter
        
    Returns:
        AlertResponse: Filtered alerts
        
    Raises:
        HTTPException: 400 if invalid params
    """
    if hours < 1 or hours > 365 * 24:
        raise HTTPException(status_code=400, detail="Invalid hours range")
    
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="Invalid limit range")
    
    if not 0.0 <= min_confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be 0-1")
    
    try:
        # Fetch alerts
        if asset:
            if asset not in ASSETS:
                raise HTTPException(status_code=400, detail=f"Invalid asset: {asset}")
            alerts = database.get_alerts(asset, hours=hours)
        else:
            # Get alerts for all assets
            all_alerts = []
            for a in ASSETS:
                all_alerts.extend(database.get_alerts(a, hours=hours))
            alerts = sorted(all_alerts, key=lambda x: x.timestamp, reverse=True)
        
        # Filter by confidence
        alerts = [a for a in alerts if a.confidence >= min_confidence]
        
        # Apply pagination
        total_count = len(alerts)
        alerts = alerts[offset:offset + limit]
        
        # Convert to response format
        alert_items = [
            AlertItem(
                id=a.id,
                timestamp=a.timestamp.isoformat(),
                asset=a.asset,
                price=a.price,
                volume=a.volume,
                detector_name=a.detector_name,
                confidence=a.confidence,
                is_real_anomaly=a.is_real_anomaly,
            )
            for a in alerts
        ]
        
        logger.info(f"Returned {len(alert_items)}/{total_count} alerts")
        
        return AlertResponse(
            alerts=alert_items,
            count=total_count,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/performance", response_model=PerformanceResponse)
async def get_performance(
    asset: str = Query(..., description="Asset name"),
) -> PerformanceResponse:
    """Get model performance metrics for asset.
    
    Args:
        asset: Asset name
        
    Returns:
        PerformanceResponse: Metrics
        
    Raises:
        HTTPException: 400 for invalid asset, 404 if no results
    """
    if asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"Invalid asset: {asset}")
    
    try:
        # Fetch backtest results for ensemble model on this asset
        results = database.get_backtest_results("EnsembleDetector", asset)
        
        if not results:
            logger.warning(f"No backtest results for {asset}")
            raise HTTPException(
                status_code=404,
                detail=f"No backtest results available for {asset}",
            )
        
        # Aggregate metrics
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1s = [r.f1_score for r in results]
        aucs = [r.roc_auc for r in results]
        fprs = [r.false_positive_rate for r in results]
        
        metrics = PerformanceMetrics(
            precision=float(np.mean(precisions)),
            recall=float(np.mean(recalls)),
            f1_score=float(np.mean(f1s)),
            roc_auc=float(np.mean(aucs)),
            false_positive_rate=float(np.mean(fprs)),
        )
        
        # Per-fold results
        folds = [
            BacktestFoldResult(
                fold=r.fold,
                precision=r.precision,
                recall=r.recall,
                f1=r.f1_score,
                roc_auc=r.roc_auc,
                fpr=r.false_positive_rate,
                start_date=r.fold_start_date.isoformat(),
                end_date=r.fold_end_date.isoformat(),
            )
            for r in results
        ]
        
        logger.info(f"Returned performance for {asset} ({len(results)} folds)")
        
        return PerformanceResponse(
            asset=asset,
            metrics=metrics,
            folds=folds,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching performance: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/backtest/comparison", response_model=BacktestComparisonResponse)
async def backtest_comparison() -> BacktestComparisonResponse:
    """Compare performance across all models.
    
    Returns:
        BacktestComparisonResponse: Model comparison
    """
    try:
        model_names = [
            "IsolationForestDetector",
            "ZScoreDetector",
            "EWMADetector",
            "EnsembleDetector",
        ]
        
        model_results = []
        best_f1 = 0.0
        best_model = ""
        
        for model_name in model_names:
            results = database.get_backtest_results(model_name)
            
            if results:
                # Aggregate across folds and assets
                precisions = [r.precision for r in results]
                recalls = [r.recall for r in results]
                f1s = [r.f1_score for r in results]
                aucs = [r.roc_auc for r in results]
                fprs = [r.false_positive_rate for r in results]
                
                mean_f1 = float(np.mean(f1s))
                
                model_results.append(
                    ModelComparisonMetrics(
                        model_name=model_name,
                        precision=float(np.mean(precisions)),
                        recall=float(np.mean(recalls)),
                        f1=mean_f1,
                        roc_auc=float(np.mean(aucs)),
                        fpr=float(np.mean(fprs)),
                        num_folds=len(results),
                    )
                )
                
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_model = model_name
        
        if not model_results:
            raise HTTPException(
                status_code=404,
                detail="No backtest results available",
            )
        
        logger.info(f"Model comparison: {len(model_results)} models, best={best_model}")
        
        return BacktestComparisonResponse(
            models=model_results,
            best_model=best_model,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in model comparison: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/alerts/{alert_id}/verify")
async def verify_alert(
    alert_id: int = Path(..., description="Alert ID"),
    is_real: bool = Query(..., description="Is this a real anomaly?"),
) -> AlertItem:
    """Mark an alert as verified (real anomaly or false positive).
    
    Args:
        alert_id: Alert ID to verify
        is_real: True if real anomaly, False if false positive
        
    Returns:
        AlertItem: Updated alert
        
    Raises:
        HTTPException: 404 if alert not found
    """
    try:
        alert = database.mark_alert_verified(alert_id, is_real)
        
        if not alert:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        logger.info(f"Alert {alert_id} verified as {is_real}")
        
        return AlertItem(
            id=alert.id,
            timestamp=alert.timestamp.isoformat(),
            asset=alert.asset,
            price=alert.price,
            volume=alert.volume,
            detector_name=alert.detector_name,
            confidence=alert.confidence,
            is_real_anomaly=alert.is_real_anomaly,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying alert: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    
    return {
        "error": f"HTTP {exc.status_code}",
        "detail": exc.detail,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return {
        "error": "Internal Server Error",
        "detail": str(exc),
        "timestamp": datetime.utcnow().isoformat(),
    }


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application
    """
    return app


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting API server on {API_PORT}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=False,
    )
