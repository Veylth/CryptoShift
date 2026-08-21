"""Start background data ingestion scheduler."""

import logging
import signal
import sys
from datetime import datetime

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from src.config import (
    ASSETS,
    POLLING_INTERVAL,
    logger as config_logger,
)
from src.data.database import init_db, add_price_data, add_alert, get_price_data
from src.data.fetcher import create_client
from src.data.feature_engineer import create_feature_engineer
from src.models.isolation_forest import IsolationForestDetector
from src.models.statistical import ZScoreDetector, EWMADetector
from src.models.ensemble import EnsembleDetector

logger = config_logger

scheduler = BackgroundScheduler()
running = True


def fetch_current_data():
    """Job 1: Fetch latest price data."""
    try:
        logger.debug("Fetching current data...")
        client = create_client()
        
        for asset in ASSETS:
            try:
                data = client.fetch_current_data(asset)
                add_price_data(
                    asset=asset,
                    timestamp=data["timestamp"],
                    price=data["price"],
                    volume=data["volume"],
                )
                logger.debug(f"Fetched {asset}: ${data['price']:.2f}")
            except Exception as e:
                logger.error(f"Error fetching {asset}: {e}")
        
    except Exception as e:
        logger.error(f"Fetch job error: {e}")


def compute_features():
    """Job 2: Engineer features from recent data."""
    try:
        logger.debug("Computing features...")
        feature_engineer = create_feature_engineer()
        
        for asset in ASSETS:
            try:
                prices = get_price_data(asset, hours=24)
                
                if prices and len(prices) > 10:
                    df = pd.DataFrame([
                        {
                            "timestamp": p.timestamp,
                            "price": p.price,
                            "volume": p.volume,
                        }
                        for p in prices
                    ])
                    
                    df = feature_engineer.compute_all_features(df)
                    logger.debug(f"Features computed for {asset}: {len(df)} rows")
                    
            except Exception as e:
                logger.error(f"Error computing features for {asset}: {e}")
        
    except Exception as e:
        logger.error(f"Feature job error: {e}")


def detect_anomalies():
    """Job 3: Run anomaly detection."""
    try:
        logger.debug("Running anomaly detection...")
        feature_engineer = create_feature_engineer()
        
        # Initialize detectors
        detectors = {
            "isolation_forest": IsolationForestDetector(),
            "zscore": ZScoreDetector(),
            "ewma": EWMADetector(),
            "ensemble": EnsembleDetector(
                detectors_list=[
                    IsolationForestDetector(),
                    ZScoreDetector(),
                    EWMADetector(),
                ],
            ),
        }
        
        # Train on recent data and predict
        for asset in ASSETS:
            try:
                prices = get_price_data(asset, hours=24)
                
                if prices and len(prices) > 50:
                    df = pd.DataFrame([
                        {
                            "timestamp": p.timestamp,
                            "price": p.price,
                            "volume": p.volume,
                        }
                        for p in prices
                    ])
                    
                    df = feature_engineer.compute_all_features(df)
                    
                    # Train detectors on historical data
                    for detector in detectors.values():
                        detector.fit(df)
                    
                    # Get latest row for prediction
                    latest = df.iloc[-1:] if len(df) > 0 else None
                    
                    if latest is not None:
                        # Run ensemble detector
                        ensemble_result = detectors["ensemble"].predict(latest)
                        
                        if ensemble_result.get("is_anomaly"):
                            add_alert(
                                asset=asset,
                                timestamp=latest["timestamp"].iloc[0],
                                price=latest["price"].iloc[0],
                                volume=latest["volume"].iloc[0],
                                detector_name="ensemble",
                                confidence=ensemble_result.get("ensemble_confidence", 0.0),
                                model_version="1.0",
                            )
                            logger.info(
                                f"Anomaly detected in {asset}: "
                                f"confidence={ensemble_result['ensemble_confidence']:.2f}"
                            )
                
            except Exception as e:
                logger.error(f"Error detecting anomalies in {asset}: {e}")
        
    except Exception as e:
        logger.error(f"Detection job error: {e}")


def health_check():
    """Job 4: Log system health."""
    try:
        logger.info("System health check - all jobs running normally")
        logger.info(f"Next scheduled jobs: {[job.name for job in scheduler.get_jobs()]}")
    except Exception as e:
        logger.error(f"Health check error: {e}")


def signal_handler(sig, frame):
    """Handle graceful shutdown."""
    global running
    logger.info("Shutdown signal received")
    running = False
    scheduler.shutdown()
    logger.info("Scheduler shut down successfully")
    sys.exit(0)


def main():
    """Initialize and start ingestion scheduler."""
    logger.info("Initializing CryptoShift Data Ingestion System...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized")
        
        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Add jobs to scheduler
        logger.info(f"Adding jobs with {POLLING_INTERVAL}s interval...")
        
        scheduler.add_job(
            fetch_current_data,
            "interval",
            seconds=POLLING_INTERVAL,
            id="fetch_data",
            name="Fetch Current Data",
            misfire_grace_time=60,
        )
        
        scheduler.add_job(
            compute_features,
            "interval",
            seconds=POLLING_INTERVAL,
            id="compute_features",
            name="Compute Features",
            misfire_grace_time=60,
        )
        
        scheduler.add_job(
            detect_anomalies,
            "interval",
            seconds=POLLING_INTERVAL,
            id="detect_anomalies",
            name="Detect Anomalies",
            misfire_grace_time=60,
        )
        
        scheduler.add_job(
            health_check,
            "interval",
            seconds=300,  # 5 minutes
            id="health_check",
            name="Health Check",
        )
        
        # Start scheduler
        scheduler.start()
        logger.info("Scheduler started successfully")
        logger.info("Data ingestion running. Press Ctrl+C to stop.")
        
        # Keep running
        while running:
            import time
            time.sleep(1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
