"""Feature engineering for cryptocurrency price data."""

from typing import Optional, Dict, List, Any
import logging
import numpy as np
import pandas as pd

from src.config import (
    VOLATILITY_WINDOW,
    MOMENTUM_WINDOW,
    EWMA_ALPHA,
    MIN_DATA_POINTS,
    logger as config_logger,
)

logger = config_logger


class FeatureEngineer:
    """Engineer features from raw price/volume data."""
    
    def __init__(
        self,
        volatility_window: int = VOLATILITY_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        ewma_alpha: float = EWMA_ALPHA,
    ):
        """Initialize feature engineer.
        
        Args:
            volatility_window: Window for rolling volatility (hours/candles)
            momentum_window: Window for momentum calculation
            ewma_alpha: Exponential weight for EWMA (0 to 1)
        """
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
        self.ewma_alpha = ewma_alpha
        
        logger.info(
            f"Initialized FeatureEngineer: "
            f"vol_window={volatility_window}, "
            f"mom_window={momentum_window}, "
            f"ewma_alpha={ewma_alpha}"
        )
    
    def compute_rolling_zscore(
        self,
        data: pd.Series,
        window: int = 24,
    ) -> pd.Series:
        """Compute rolling Z-score normalization.
        
        Formula: (x - rolling_mean) / rolling_std
        
        Args:
            data: Series of values
            window: Window size for rolling stats
            
        Returns:
            pd.Series: Z-scores (NaN for first window-1 rows)
        """
        if data.empty or len(data) < window:
            logger.warning(f"Insufficient data for z-score (need {window}, got {len(data)})")
            return pd.Series(np.nan, index=data.index)
        
        rolling_mean = data.rolling(window=window, min_periods=1).mean()
        rolling_std = data.rolling(window=window, min_periods=1).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (data - rolling_mean) / rolling_std
        
        return zscore
    
    def compute_volatility(
        self,
        prices: pd.Series,
        window: int = 24,
    ) -> pd.Series:
        """Compute rolling volatility as standard deviation of log returns.
        
        Formula:
            log_return[t] = log(price[t] / price[t-1])
            volatility[t] = std(log_return[t-window:t])
        
        Args:
            prices: Series of asset prices
            window: Window size
            
        Returns:
            pd.Series: Rolling volatility (NaN for first window rows)
        """
        if prices.empty or len(prices) < 2:
            logger.warning(f"Insufficient price data for volatility (got {len(prices)})")
            return pd.Series(np.nan, index=prices.index)
        
        # Compute log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Compute rolling standard deviation
        volatility = log_returns.rolling(window=window, min_periods=1).std()
        
        return volatility
    
    def compute_momentum(
        self,
        prices: pd.Series,
        window: int = 24,
    ) -> pd.Series:
        """Compute momentum as percentage change over window.
        
        Formula: momentum[t] = (price[t] - price[t-window]) / price[t-window]
        
        Args:
            prices: Series of prices
            window: Window size
            
        Returns:
            pd.Series: Momentum (percentage change)
        """
        if prices.empty or len(prices) < window:
            logger.warning(f"Insufficient data for momentum (need {window}, got {len(prices)})")
            return pd.Series(np.nan, index=prices.index)
        
        momentum = (prices - prices.shift(window)) / prices.shift(window)
        
        return momentum
    
    def compute_ewma(
        self,
        data: pd.Series,
        alpha: Optional[float] = None,
    ) -> pd.Series:
        """Compute exponential weighted moving average.
        
        Formula: ewma[t] = alpha * data[t] + (1 - alpha) * ewma[t-1]
        
        Args:
            data: Series of values
            alpha: Smoothing factor (0 to 1). If None, uses self.ewma_alpha
            
        Returns:
            pd.Series: EWMA
        """
        if data.empty:
            logger.warning("Empty data for EWMA computation")
            return pd.Series(np.nan, index=data.index)
        
        alpha = alpha or self.ewma_alpha
        ewma = data.ewm(span=int(1/alpha), adjust=False).mean()
        
        return ewma
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features from raw data.
        
        Input DataFrame must have columns: timestamp, price, volume
        
        Output adds columns:
            - price_zscore: Z-score of price
            - volume_zscore: Z-score of volume
            - volatility_1h: Rolling volatility
            - momentum_1h: Momentum
        
        Args:
            df: DataFrame with price and volume data (sorted by timestamp ascending)
            
        Returns:
            pd.DataFrame: DataFrame with engineered features added
            
        Raises:
            ValueError: If required columns missing or insufficient data
        """
        logger.debug(f"Engineering features for {len(df)} rows")
        
        # Validate input
        required_cols = ["timestamp", "price", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(df) < MIN_DATA_POINTS:
            raise ValueError(
                f"Insufficient data for feature engineering (need {MIN_DATA_POINTS}, got {len(df)})"
            )
        
        # Create copy to avoid modifying original
        result = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(result["timestamp"]):
            result["timestamp"] = pd.to_datetime(result["timestamp"])
        
        # Sort by timestamp to ensure correct rolling calculations
        result = result.sort_values("timestamp").reset_index(drop=True)
        
        # Engineer features using vectorized operations
        try:
            # Price Z-score
            result["price_zscore"] = self.compute_rolling_zscore(
                result["price"],
                window=self.volatility_window,
            )
            
            # Volume Z-score
            result["volume_zscore"] = self.compute_rolling_zscore(
                result["volume"],
                window=self.volatility_window,
            )
            
            # Volatility
            result["volatility_1h"] = self.compute_volatility(
                result["price"],
                window=self.volatility_window,
            )
            
            # Momentum
            result["momentum_1h"] = self.compute_momentum(
                result["price"],
                window=self.momentum_window,
            )
            
            # Drop rows with NaN features from rolling calculations
            # Keep first few NaNs to align with the window
            valid_rows = result.notna().all(axis=1).sum()
            logger.debug(f"Generated features for {valid_rows} rows out of {len(result)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise
    
    def get_latest_features(
        self,
        df: pd.DataFrame,
        n_rows: int = 1,
    ) -> pd.DataFrame:
        """Get the latest N rows with complete features.
        
        Skips rows with NaN values.
        
        Args:
            df: DataFrame with engineered features
            n_rows: Number of latest rows to return
            
        Returns:
            pd.DataFrame: Latest N valid rows
        """
        # Get rows with complete features (no NaN)
        valid_df = df.dropna(subset=[
            "price_zscore",
            "volume_zscore",
            "volatility_1h",
            "momentum_1h",
        ])
        
        if valid_df.empty:
            logger.warning("No valid feature rows available")
            return valid_df
        
        # Return latest n_rows
        return valid_df.tail(n_rows).reset_index(drop=True)


def create_feature_engineer() -> FeatureEngineer:
    """Factory function to create feature engineer.
    
    Returns:
        FeatureEngineer: Configured instance
    """
    return FeatureEngineer(
        volatility_window=VOLATILITY_WINDOW,
        momentum_window=MOMENTUM_WINDOW,
        ewma_alpha=EWMA_ALPHA,
    )
