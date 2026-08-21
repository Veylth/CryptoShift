"""Isolation Forest anomaly detector."""

from typing import Dict, List, Optional, Any
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import (
    ISOLATION_FOREST_N_ESTIMATORS,
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_RANDOM_STATE,
    logger as config_logger,
)

logger = config_logger


class IsolationForestDetector:
    """Anomaly detector using Isolation Forest algorithm.
    
    Isolation Forest works by randomly selecting features and split values,
    isolating observations by randomly "cutting" the feature space.
    Anomalies require fewer cuts to be isolated, resulting in shorter path lengths.
    """
    
    def __init__(
        self,
        n_estimators: int = ISOLATION_FOREST_N_ESTIMATORS,
        contamination: float = ISOLATION_FOREST_CONTAMINATION,
        random_state: int = ISOLATION_FOREST_RANDOM_STATE,
    ):
        """Initialize Isolation Forest detector.
        
        Args:
            n_estimators: Number of trees in the forest
            contamination: Fraction of outliers in dataset (0 to 0.5)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
        logger.info(
            f"Initialized IsolationForestDetector: "
            f"n_estimators={n_estimators}, "
            f"contamination={contamination}"
        )
    
    def fit(self, features_df: pd.DataFrame) -> "IsolationForestDetector":
        """Train Isolation Forest on features.
        
        Args:
            features_df: DataFrame with numeric features
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If insufficient data or invalid features
        """
        logger.debug(f"Training IsolationForest on {len(features_df)} samples")
        
        # Handle edge cases
        if features_df.empty:
            raise ValueError("Cannot fit on empty DataFrame")
        
        if len(features_df) < 10:
            logger.warning(f"Very few samples for training: {len(features_df)}")
        
        # Remove non-numeric columns and handle NaN
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            raise ValueError("No numeric features in input data")
        
        # Drop rows with any NaN values
        clean_data = numeric_features.dropna()
        
        if clean_data.empty:
            raise ValueError("All feature rows contain NaN values")
        
        self.feature_names = clean_data.columns.tolist()
        
        # Train model
        try:
            self.model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model.fit(clean_data)
            self.is_fitted = True
            logger.info(f"IsolationForest trained on {len(clean_data)} samples")
            
        except Exception as e:
            logger.error(f"Error training IsolationForest: {e}")
            raise
        
        return self
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict anomaly for a single row (or take first row if multiple).
        
        Returns:
            Dict: {
                "is_anomaly": bool,
                "anomaly_score": float (-1 to 1),
                "confidence": float (0 to 1)
            }
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if features_df.empty:
            raise ValueError("Cannot predict on empty DataFrame")
        
        # Use first row
        row = features_df.iloc[[0]]
        
        # Select only the features used during training
        if self.feature_names:
            row = row[self.feature_names]
        
        # Handle NaN values
        if row.isna().any().any():
            logger.warning("Input contains NaN values, imputing with 0")
            row = row.fillna(0)
        
        # Get prediction and anomaly score
        prediction = self.model.predict(row)[0]  # -1 for anomaly, 1 for normal
        score = self.model.score_samples(row)[0]  # Lower score = more anomalous
        
        # Convert to standard format
        is_anomaly = prediction == -1
        
        # Normalize score to 0-1 range
        # Score typically ranges from ~-0.5 to 0.5, map to 0-1
        anomaly_score_normalized = (score + 0.5) * 2  # Rough normalization
        anomaly_score_normalized = np.clip(anomaly_score_normalized, -1, 1)
        
        # Confidence: higher absolute score (more anomalous) = higher confidence
        confidence = 1.0 - (anomaly_score_normalized + 1.0) / 2.0
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score_normalized),
            "confidence": float(confidence),
        }
    
    def predict_batch(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict anomalies for multiple rows (vectorized).
        
        Args:
            features_df: DataFrame with multiple rows
            
        Returns:
            List[Dict]: One prediction dict per row
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if features_df.empty:
            return []
        
        # Select only training features
        if self.feature_names:
            features_df = features_df[self.feature_names]
        
        # Handle NaN
        features_df = features_df.fillna(0)
        
        predictions = self.model.predict(features_df)  # -1 or 1 for each row
        scores = self.model.score_samples(features_df)
        
        results = []
        for pred, score in zip(predictions, scores):
            is_anomaly = pred == -1
            
            # Normalize score
            anomaly_score_normalized = (score + 0.5) * 2
            anomaly_score_normalized = np.clip(anomaly_score_normalized, -1, 1)
            
            # Confidence
            confidence = 1.0 - (anomaly_score_normalized + 1.0) / 2.0
            confidence = np.clip(confidence, 0.0, 1.0)
            
            results.append({
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(anomaly_score_normalized),
                "confidence": float(confidence),
            })
        
        return results
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk.
        
        Args:
            path: File path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, path: str) -> "IsolationForestDetector":
        """Load trained model from disk.
        
        Args:
            path: File path to load model
            
        Returns:
            self for method chaining
        """
        try:
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            
            self.is_fitted = True
            logger.info(f"Model loaded from {path}")
            return self
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores.
        
        Returns:
            Dict: feature_name → importance_score, or None if not available
        """
        if not self.is_fitted or not self.model:
            return None
        
        # Isolation Forest doesn't have traditional feature importance
        # Return placeholder or estimated importance based on model structure
        if self.feature_names:
            return {name: 0.0 for name in self.feature_names}
        
        return None
