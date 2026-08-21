"""Train models with hyperparameter tuning."""

import logging
from itertools import product

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ASSETS, logger as config_logger
from src.data.database import init_db, get_price_data
from src.data.feature_engineer import create_feature_engineer
from src.models.isolation_forest import IsolationForestDetector

logger = config_logger


def main():
    """Train Isolation Forest with hyperparameter grid search."""
    logger.info("Starting model training...")
    
    try:
        # Initialize database
        init_db()
        
        # Get historical data
        prices = get_price_data("bitcoin", hours=6*30*24)
        
        if not prices or len(prices) < 100:
            logger.error("Insufficient training data")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": p.timestamp,
                "price": p.price,
                "volume": p.volume,
            }
            for p in prices
        ])
        
        # Engineer features
        feature_engineer = create_feature_engineer()
        df = feature_engineer.compute_all_features(df)
        
        # Hyperparameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "contamination": [0.03, 0.05, 0.07],
        }
        
        best_score = 0.0
        best_params = None
        best_model = None
        
        # Grid search
        for n_est, contam in product(
            param_grid["n_estimators"],
            param_grid["contamination"],
        ):
            logger.info(f"Testing: n_estimators={n_est}, contamination={contam}")
            
            try:
                # Split data
                train_idx = int(len(df) * 0.7)
                train_df = df.iloc[:train_idx]
                val_df = df.iloc[train_idx:]
                
                # Train model
                model = IsolationForestDetector(
                    n_estimators=n_est,
                    contamination=contam,
                )
                model.fit(train_df)
                
                # Evaluate on validation set
                predictions = model.predict_batch(val_df)
                scores = [p["confidence"] for p in predictions]
                
                # Simple score: average confidence
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                logger.info(f"  Score: {avg_score:.4f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {"n_estimators": n_est, "contamination": contam}
                    best_model = model
                
            except Exception as e:
                logger.warning(f"Error with parameters: {e}")
                continue
        
        if best_model:
            logger.info(f"\nBest parameters: {best_params}")
            logger.info(f"Best score: {best_score:.4f}")
            
            # Save best model
            model_path = "data/best_model.pkl"
            best_model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.error("No successful model training")
        
        logger.info("Training completed!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
