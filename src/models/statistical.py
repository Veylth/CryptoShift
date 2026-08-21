"""Statistical anomaly detectors (Z-score and EWMA)."""

from typing import Dict, Any, List, Optional
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    ZSCORE_THRESHOLD,
    EWMA_STD_THRESHOLD,
    EWMA_ALPHA,
    logger as config_logger,
)

logger = config_logger


class ZScoreDetector:
    """Anomaly detector using Z-score statistical method.
    
    Detects anomalies where price or volume deviates significantly from
    the rolling mean (typically >3 standard deviations is anomalous).
    """
    
    def __init__(self, threshold: float = ZSCORE_THRESHOLD):
        """Initialize Z-score detector.
        
        Args:
            threshold: Z-score threshold for anomaly (typically 2-4)
        """
        self.threshold = threshold
        self.is_fitted = True  # Z-score doesn't need fitting
        
        logger.info(f"Initialized ZScoreDetector with threshold={threshold}")
    
    def fit(self, features_df: pd.DataFrame) -> "ZScoreDetector":
        """Fit detector (Z-score requires no training).
        
        Args:
            features_df: Feature data (ignored)
            
        Returns:
            self
        """
        return self
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict anomaly for single row.
        
        Uses price_zscore and volume_zscore columns.
        
        Returns:
            Dict: {
                "is_anomaly": bool,
                "zscore": float,
                "confidence": float
            }
        """
        if features_df.empty:
            raise ValueError("Cannot predict on empty DataFrame")
        
        row = features_df.iloc[0]
        
        # Get z-scores from features
        price_zscore = row.get("price_zscore", np.nan)
        volume_zscore = row.get("volume_zscore", np.nan)
        
        # Handle NaN
        if np.isnan(price_zscore):
            price_zscore = 0.0
        if np.isnan(volume_zscore):
            volume_zscore = 0.0
        
        # Take maximum absolute z-score
        max_zscore = max(abs(price_zscore), abs(volume_zscore))
        
        # Anomaly if |zscore| > threshold
        is_anomaly = max_zscore > self.threshold
        
        # Confidence: higher z-score = higher confidence
        # Map |zscore| to 0-1 range
        confidence = min(1.0, max_zscore / (self.threshold * 2))
        confidence = max(0.0, confidence)
        
        return {
            "is_anomaly": bool(is_anomaly),
            "zscore": float(max_zscore),
            "confidence": float(confidence),
        }
    
    def predict_batch(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict anomalies for multiple rows.
        
        Args:
            features_df: DataFrame with price_zscore, volume_zscore columns
            
        Returns:
            List[Dict]: One prediction per row
        """
        if features_df.empty:
            return []
        
        results = []
        
        for idx, row in features_df.iterrows():
            price_zscore = row.get("price_zscore", 0.0)
            volume_zscore = row.get("volume_zscore", 0.0)
            
            if np.isnan(price_zscore):
                price_zscore = 0.0
            if np.isnan(volume_zscore):
                volume_zscore = 0.0
            
            max_zscore = max(abs(price_zscore), abs(volume_zscore))
            is_anomaly = max_zscore > self.threshold
            confidence = min(1.0, max(0.0, max_zscore / (self.threshold * 2)))
            
            results.append({
                "is_anomaly": bool(is_anomaly),
                "zscore": float(max_zscore),
                "confidence": float(confidence),
            })
        
        return results


class EWMADetector:
    """Exponential Weighted Moving Average anomaly detector.
    
    Detects deviations from the EWMA trend. Points that deviate significantly
    from the smoothed trend are considered anomalous.
    """
    
    def __init__(
        self,
        alpha: float = EWMA_ALPHA,
        std_threshold: float = EWMA_STD_THRESHOLD,
    ):
        """Initialize EWMA detector.
        
        Args:
            alpha: Smoothing factor (0 to 1, lower = more smooth)
            std_threshold: Standard deviations for anomaly threshold
        """
        self.alpha = alpha
        self.std_threshold = std_threshold
        self.baseline_mean = None
        self.baseline_std = None
        self.is_fitted = False
        
        logger.info(
            f"Initialized EWMADetector: alpha={alpha}, std_threshold={std_threshold}"
        )
    
    def fit(self, features_df: pd.DataFrame) -> "EWMADetector":
        """Fit EWMA baseline on historical data.
        
        Args:
            features_df: DataFrame with price data
            
        Returns:
            self
        """
        if features_df.empty:
            raise ValueError("Cannot fit on empty DataFrame")
        
        if "price" not in features_df.columns:
            raise ValueError("DataFrame must contain 'price' column")
        
        try:
            prices = features_df["price"].dropna()
            
            if prices.empty:
                raise ValueError("No valid price data")
            
            # Compute EWMA of prices
            ewma = prices.ewm(span=int(1/self.alpha), adjust=False).mean()
            
            # Compute deviation from EWMA
            deviation = prices - ewma
            
            # Store baseline statistics
            self.baseline_mean = ewma.mean()
            self.baseline_std = deviation.std()
            
            if self.baseline_std == 0:
                self.baseline_std = 1.0  # Avoid division by zero
            
            self.is_fitted = True
            logger.debug(
                f"EWMA baseline fitted: mean={self.baseline_mean:.2f}, "
                f"std={self.baseline_std:.4f}"
            )
            return self
            
        except Exception as e:
            logger.error(f"Error fitting EWMA: {e}")
            raise
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict anomaly for single row.
        
        Args:
            features_df: DataFrame with price column
            
        Returns:
            Dict: {
                "is_anomaly": bool,
                "deviation": float,
                "confidence": float
            }
        """
        if not self.is_fitted:
            # If not fitted, initialize with benign baseline
            self.baseline_mean = 0.0
            self.baseline_std = 1.0
        
        if features_df.empty:
            raise ValueError("Cannot predict on empty DataFrame")
        
        row = features_df.iloc[0]
        price = row.get("price", np.nan)
        
        if np.isnan(price):
            price = self.baseline_mean
        
        # Compute deviation from baseline
        deviation = price - self.baseline_mean
        
        # Anomaly if deviation > threshold * std
        is_anomaly = abs(deviation) > (self.std_threshold * self.baseline_std)
        
        # Confidence: higher deviation = higher confidence
        normalized_deviation = abs(deviation) / max(self.baseline_std, 0.001)
        confidence = min(1.0, normalized_deviation / (self.std_threshold * 2))
        confidence = max(0.0, confidence)
        
        return {
            "is_anomaly": bool(is_anomaly),
            "deviation": float(deviation),
            "confidence": float(confidence),
        }
    
    def predict_batch(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict anomalies for multiple rows.
        
        Args:
            features_df: DataFrame with price column
            
        Returns:
            List[Dict]: One prediction per row
        """
        if not self.is_fitted:
            self.baseline_mean = 0.0
            self.baseline_std = 1.0
        
        if features_df.empty:
            return []
        
        results = []
        
        for idx, row in features_df.iterrows():
            price = row.get("price", self.baseline_mean)
            
            if np.isnan(price):
                price = self.baseline_mean
            
            deviation = price - self.baseline_mean
            is_anomaly = abs(deviation) > (self.std_threshold * self.baseline_std)
            
            normalized_deviation = abs(deviation) / max(self.baseline_std, 0.001)
            confidence = min(1.0, max(0.0, normalized_deviation / (self.std_threshold * 2)))
            
            results.append({
                "is_anomaly": bool(is_anomaly),
                "deviation": float(deviation),
                "confidence": float(confidence),
            })
        
        return results
