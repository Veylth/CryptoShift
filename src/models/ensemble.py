"""Ensemble anomaly detector combining multiple detectors."""

from typing import Dict, List, Any, Optional
import logging

import pandas as pd
import numpy as np

from src.config import ENSEMBLE_VOTING_THRESHOLD, logger as config_logger

logger = config_logger


class EnsembleDetector:
    """Ensemble anomaly detector using voting from multiple detectors.
    
    Combines predictions from Isolation Forest, Z-score, and EWMA detectors.
    An anomaly is flagged if at least N detectors agree (voting_threshold).
    """
    
    def __init__(
        self,
        detectors_list: Optional[List] = None,
        voting_threshold: int = ENSEMBLE_VOTING_THRESHOLD,
    ):
        """Initialize ensemble detector.
        
        Args:
            detectors_list: List of detector instances to ensemble
            voting_threshold: Min number of detectors that must agree (e.g., 2 of 3)
        """
        self.detectors_list = detectors_list or []
        self.voting_threshold = voting_threshold
        self.is_fitted = False
        
        logger.info(
            f"Initialized EnsembleDetector with {len(self.detectors_list)} "
            f"detectors, voting_threshold={voting_threshold}"
        )
    
    def set_detectors(self, detectors_list: List) -> "EnsembleDetector":
        """Set detector instances.
        
        Args:
            detectors_list: List of detector instances
            
        Returns:
            self
        """
        self.detectors_list = detectors_list
        logger.info(f"Ensemble detectors set: {len(detectors_list)} detectors")
        return self
    
    def fit(self, features_df: pd.DataFrame) -> "EnsembleDetector":
        """Fit all ensemble detectors.
        
        Args:
            features_df: Training features
            
        Returns:
            self
        """
        if not self.detectors_list:
            raise ValueError("No detectors configured")
        
        logger.debug(f"Fitting {len(self.detectors_list)} ensemble detectors")
        
        for i, detector in enumerate(self.detectors_list):
            try:
                detector.fit(features_df)
                logger.debug(f"Detector {i} fitted successfully")
            except Exception as e:
                logger.error(f"Error fitting detector {i}: {e}")
                raise
        
        self.is_fitted = True
        logger.info("Ensemble detectors fitted successfully")
        return self
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict anomaly using ensemble voting.
        
        Returns:
            Dict: {
                "is_anomaly": bool,
                "ensemble_confidence": float,
                "votes": {
                    detector_name: bool (is_anomaly)
                },
                "confidences": {
                    detector_name: float (confidence)
                },
                "agreement": int (number of votes for anomaly)
            }
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        if features_df.empty:
            raise ValueError("Cannot predict on empty DataFrame")
        
        if not self.detectors_list:
            raise ValueError("No detectors configured")
        
        votes = {}
        confidences = {}
        vote_count = 0
        
        # Collect predictions from all detectors
        for detector in self.detectors_list:
            try:
                prediction = detector.predict(features_df)
                
                # Get detector name
                detector_name = type(detector).__name__
                
                # Extract key fields
                is_anomaly = prediction.get("is_anomaly", False)
                confidence = prediction.get("confidence", 0.0)
                
                votes[detector_name] = bool(is_anomaly)
                confidences[detector_name] = float(confidence)
                
                if is_anomaly:
                    vote_count += 1
                
            except Exception as e:
                logger.warning(f"Error in detector {type(detector).__name__}: {e}")
                votes[type(detector).__name__] = False
                confidences[type(detector).__name__] = 0.0
        
        # Ensemble decision: anomaly if votes >= threshold
        is_anomaly_ensemble = vote_count >= self.voting_threshold
        
        # Ensemble confidence: average of all detectors' confidences
        avg_confidence = np.mean(list(confidences.values())) if confidences else 0.0
        
        # Boost confidence if detectors agree on anomaly
        if is_anomaly_ensemble and vote_count > 0:
            agreement_boost = vote_count / len(self.detectors_list)
            ensemble_confidence = avg_confidence * (0.5 + 0.5 * agreement_boost)
        else:
            ensemble_confidence = avg_confidence * (vote_count / max(len(self.detectors_list), 1))
        
        ensemble_confidence = np.clip(ensemble_confidence, 0.0, 1.0)
        
        return {
            "is_anomaly": bool(is_anomaly_ensemble),
            "ensemble_confidence": float(ensemble_confidence),
            "votes": votes,
            "confidences": confidences,
            "agreement": int(vote_count),
        }
    
    def predict_batch(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict anomalies for multiple rows.
        
        Args:
            features_df: DataFrame with multiple rows
            
        Returns:
            List[Dict]: One ensemble prediction per row
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        if features_df.empty:
            return []
        
        if not self.detectors_list:
            raise ValueError("No detectors configured")
        
        results = []
        
        for idx, row in features_df.iterrows():
            # Create single-row dataframe for this sample
            row_df = pd.DataFrame([row])
            
            # Get ensemble prediction
            prediction = self.predict(row_df)
            results.append(prediction)
        
        return results
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get statistics about detectors in ensemble.
        
        Returns:
            Dict: Information about each detector
        """
        stats = {
            "num_detectors": len(self.detectors_list),
            "voting_threshold": self.voting_threshold,
            "detectors": [],
        }
        
        for detector in self.detectors_list:
            detector_name = type(detector).__name__
            detector_info = {
                "name": detector_name,
                "fitted": getattr(detector, "is_fitted", False),
            }
            stats["detectors"].append(detector_info)
        
        return stats
