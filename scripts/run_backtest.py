"""Run backtesting on all models."""

import logging
from datetime import datetime

import pandas as pd

from src.config import ASSETS, logger as config_logger
from src.data.database import init_db, get_price_data, add_backtest_result
from src.data.feature_engineer import create_feature_engineer
from src.models.isolation_forest import IsolationForestDetector
from src.models.statistical import ZScoreDetector, EWMADetector
from src.models.ensemble import EnsembleDetector
from src.backtesting.backtest import BacktestFramework

logger = config_logger


def main():
    """Run walk-forward backtesting on all models."""
    logger.info("Starting backtesting...")
    
    try:
        # Initialize database
        init_db()
        
        # Create components
        feature_engineer = create_feature_engineer()
        backtest_framework = BacktestFramework()
        
        # Create detectors
        detectors = [
            ("IsolationForestDetector", IsolationForestDetector()),
            ("ZScoreDetector", ZScoreDetector()),
            ("EWMADetector", EWMADetector()),
            ("EnsembleDetector", EnsembleDetector(
                detectors_list=[
                    IsolationForestDetector(),
                    ZScoreDetector(),
                    EWMADetector(),
                ],
            )),
        ]
        
        # Run backtest for each asset and model
        for asset in ASSETS:
            logger.info(f"Backtesting on {asset}...")
            
            try:
                # Get historical data
                prices = get_price_data(asset, hours=6*30*24)  # 6 months
                
                if not prices or len(prices) < 100:
                    logger.warning(f"Insufficient data for {asset}")
                    continue
                
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
                df = feature_engineer.compute_all_features(df)
                
                # Run backtest for each model
                for model_name, detector in detectors:
                    logger.info(f"  Testing {model_name}...")
                    
                    try:
                        # Run backtest
                        result = backtest_framework.run_backtest(df, detector, asset)
                        
                        # Save results to database
                        for fold_result in result['folds']:
                            add_backtest_result(
                                model_name=model_name,
                                asset=asset,
                                fold=fold_result.get('fold', 0),
                                fold_start_date=fold_result.get('fold_start', datetime.utcnow()),
                                fold_end_date=fold_result.get('fold_end', datetime.utcnow()),
                                precision=fold_result.get('precision', 0.0),
                                recall=fold_result.get('recall', 0.0),
                                f1_score=fold_result.get('f1', 0.0),
                                roc_auc=fold_result.get('roc_auc', 0.0),
                                false_positive_rate=fold_result.get('fpr', 0.0),
                                true_positive_rate=fold_result.get('tpr', 0.0),
                                n_anomalies_detected=fold_result.get('tp', 0) + fold_result.get('fp', 0),
                                n_true_positives=fold_result.get('tp', 0),
                                n_false_positives=fold_result.get('fp', 0),
                                n_true_negatives=fold_result.get('tn', 0),
                                n_false_negatives=fold_result.get('fn', 0),
                            )
                        
                        # Print results
                        backtest_framework.print_results(result)
                        
                    except Exception as e:
                        logger.error(f"Error testing {model_name}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Error backtesting {asset}: {e}")
                continue
        
        logger.info("Backtesting completed!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
