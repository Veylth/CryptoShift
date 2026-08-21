"""Walk-forward backtesting framework."""

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Generator, Optional
import logging

import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_TRAIN_MONTHS,
    BACKTEST_TEST_MONTHS,
    BACKTEST_ANOMALY_PERCENTILE,
    logger as config_logger,
)
from .metrics import compute_all_metrics, aggregate_metrics

logger = config_logger


class BacktestFramework:
    """Walk-forward backtesting framework for anomaly detection.
    
    Implements temporal cross-validation to prevent look-ahead bias.
    Trains on historical data and tests on future data.
    """
    
    def __init__(
        self,
        train_months: int = BACKTEST_TRAIN_MONTHS,
        test_months: int = BACKTEST_TEST_MONTHS,
        anomaly_percentile: float = BACKTEST_ANOMALY_PERCENTILE,
    ):
        """Initialize backtesting framework.
        
        Args:
            train_months: Months of data for training each fold
            test_months: Months of data for testing each fold
            anomaly_percentile: Percentile threshold for anomaly labeling (e.g., 0.95 = top 5%)
        """
        self.train_months = train_months
        self.test_months = test_months
        self.anomaly_percentile = anomaly_percentile
        
        logger.info(
            f"Initialized BacktestFramework: "
            f"train_months={train_months}, test_months={test_months}"
        )
    
    def walk_forward_split(
        self,
        df: pd.DataFrame,
        train_months: Optional[int] = None,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
        """Create walk-forward time series splits.
        
        Prevents look-ahead bias by ensuring test period comes after training period.
        
        Example with 6 months total, train_months=3:
            Fold 1: Train months 0-3, Test month 4
            Fold 2: Train months 1-4, Test month 5
            Fold 3: Train months 2-5, Test month 6
        
        Args:
            df: DataFrame with timestamp column, sorted ascending
            train_months: Override training months
            
        Yields:
            Tuple: (train_df, test_df, fold_number)
            
        Raises:
            ValueError: If insufficient data
        """
        train_months = train_months or self.train_months
        
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Ensure sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        if len(df) < (train_months + self.test_months) * 30:  # Rough estimate
            raise ValueError(
                f"Insufficient data for {train_months} train + "
                f"{self.test_months} test months"
            )
        
        # Get date range
        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()
        
        logger.info(
            f"Walk-forward split: {min_date.date()} to {max_date.date()} "
            f"({(max_date - min_date).days} days)"
        )
        
        # Create monthly boundaries
        current_date = min_date
        fold = 0
        
        while True:
            # Training period: current_date to current_date + train_months
            train_start = current_date
            train_end = train_start + timedelta(days=30 * train_months)
            
            # Testing period: train_end to train_end + test_months
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.test_months)
            
            # Check if test period exceeds data range
            if test_end > max_date:
                break
            
            # Get data for this fold
            train_df = df[
                (df["timestamp"] >= train_start) &
                (df["timestamp"] < train_end)
            ].reset_index(drop=True)
            
            test_df = df[
                (df["timestamp"] >= test_start) &
                (df["timestamp"] < test_end)
            ].reset_index(drop=True)
            
            if len(train_df) > 0 and len(test_df) > 0:
                fold += 1
                logger.info(
                    f"Fold {fold}: Train {train_start.date()} to {train_end.date()} "
                    f"({len(train_df)} rows), "
                    f"Test {test_start.date()} to {test_end.date()} ({len(test_df)} rows)"
                )
                yield train_df, test_df, fold
            
            # Slide window forward by 1 month
            current_date += timedelta(days=30)
    
    def label_anomalies_by_percentile(
        self,
        df: pd.DataFrame,
        score_column: str = "anomaly_score",
        percentile: Optional[float] = None,
    ) -> np.ndarray:
        """Label anomalies as top X percentile of scores.
        
        Args:
            df: DataFrame with score column
            score_column: Name of score column
            percentile: Percentile threshold (default: self.anomaly_percentile)
            
        Returns:
            np.ndarray: Binary labels (1 for anomaly, 0 for normal)
        """
        percentile = percentile or self.anomaly_percentile
        
        if score_column not in df.columns:
            raise ValueError(f"Column '{score_column}' not found")
        
        scores = df[score_column].values
        threshold = np.percentile(scores, percentile * 100)
        
        labels = (scores >= threshold).astype(int)
        
        logger.debug(
            f"Labeled anomalies: threshold={threshold:.4f}, "
            f"count={labels.sum()}/{len(labels)}"
        )
        
        return labels
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        detector_instance,
        asset: str = "unknown",
    ) -> Dict[str, Any]:
        """Run walk-forward backtest on detector.
        
        Args:
            df: DataFrame with price, volume, timestamp
            detector_instance: Fitted detector (or will be fitted on each fold)
            asset: Asset name (for logging)
            
        Returns:
            Dict: {
                'folds': [fold results],
                'aggregate': aggregated metrics,
                'detector_name': str,
            }
        """
        logger.info(f"Starting backtest for {detector_instance.__class__.__name__} on {asset}")
        
        fold_results = []
        
        # Use feature-engineered data if available, else work with raw data
        # Assume df already has engineered features
        
        for train_df, test_df, fold_num in self.walk_forward_split(df):
            try:
                # Train detector on this fold's training data
                detector_instance.fit(train_df)
                
                # Predict on test data
                predictions_test = detector_instance.predict_batch(test_df)
                
                # Extract predictions
                y_pred = np.array([p["is_anomaly"] for p in predictions_test]).astype(int)
                y_scores = np.array([p.get("confidence", p.get("anomaly_score", 0)) 
                                    for p in predictions_test])
                
                # Label ground truth as top anomaly_percentile
                y_true = self.label_anomalies_by_percentile(test_df)
                
                # Compute metrics
                metrics = compute_all_metrics(y_true, y_pred, y_scores)
                metrics['fold'] = fold_num
                metrics['fold_start'] = train_df["timestamp"].min()
                metrics['fold_end'] = test_df["timestamp"].max()
                metrics['n_test_samples'] = len(test_df)
                
                fold_results.append(metrics)
                
                logger.info(
                    f"Fold {fold_num} results: "
                    f"precision={metrics['precision']:.3f}, "
                    f"recall={metrics['recall']:.3f}, "
                    f"f1={metrics['f1']:.3f}, "
                    f"roc_auc={metrics['roc_auc']:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Error in fold {fold_num}: {e}")
                continue
        
        if not fold_results:
            logger.error("No successful folds")
            return {
                'folds': [],
                'aggregate': {},
                'detector_name': detector_instance.__class__.__name__,
            }
        
        # Aggregate results across folds
        aggregate = aggregate_metrics(fold_results)
        
        result = {
            'folds': fold_results,
            'aggregate': aggregate,
            'detector_name': detector_instance.__class__.__name__,
            'asset': asset,
        }
        
        logger.info(
            f"Backtest completed: {len(fold_results)} folds, "
            f"avg_f1={aggregate.get('f1', {}).get('mean', 0):.3f}"
        )
        
        return result
    
    def print_results(self, backtest_result: Dict[str, Any]) -> None:
        """Print backtest results in formatted table.
        
        Args:
            backtest_result: Result from run_backtest()
        """
        print("\n" + "=" * 100)
        print(f"BACKTEST RESULTS: {backtest_result['detector_name']}")
        print("=" * 100)
        
        # Fold-level results
        print("\nPer-Fold Results:")
        print("-" * 100)
        print(f"{'Fold':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12} {'FPR':<12} {'TP':<8}")
        print("-" * 100)
        
        for fold_result in backtest_result['folds']:
            print(
                f"{fold_result.get('fold', 0):<8} "
                f"{fold_result.get('precision', 0):<12.4f} "
                f"{fold_result.get('recall', 0):<12.4f} "
                f"{fold_result.get('f1', 0):<12.4f} "
                f"{fold_result.get('roc_auc', 0):<12.4f} "
                f"{fold_result.get('fpr', 0):<12.4f} "
                f"{fold_result.get('tp', 0):<8}"
            )
        
        # Aggregate results
        print("\n" + "-" * 100)
        print("Aggregate Results (Mean ± Std):")
        print("-" * 100)
        
        aggregate = backtest_result['aggregate']
        for metric_name in ['precision', 'recall', 'f1', 'roc_auc', 'fpr']:
            if metric_name in aggregate:
                metric = aggregate[metric_name]
                mean = metric.get('mean', 0)
                std = metric.get('std', 0)
                print(f"{metric_name:<20}: {mean:.4f} ± {std:.4f}")
        
        print("=" * 100 + "\n")
