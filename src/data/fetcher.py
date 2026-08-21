"""CoinGecko API data fetcher for CryptoShift."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import logging
from functools import wraps

import requests
import pandas as pd
from tqdm import tqdm

from src.config import (
    COIN_GECKO_API,
    API_REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    ASSETS,
    MIN_PRICE_VALUE,
    logger as config_logger,
)

logger = config_logger


def rate_limit(min_interval: float):
    """Decorator to enforce minimum interval between function calls.
    
    Args:
        min_interval: Minimum seconds between calls
    """
    def decorator(func):
        func.last_called = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - func.last_called
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            result = func(*args, **kwargs)
            func.last_called = time.time()
            return result
        
        return wrapper
    return decorator


class CoinGeckoClient:
    """Client for fetching cryptocurrency data from CoinGecko API."""
    
    def __init__(self, min_request_interval: float = 1.0):
        """Initialize CoinGecko client.
        
        Args:
            min_request_interval: Minimum seconds between API requests (rate limiting)
        """
        self.base_url = COIN_GECKO_API
        self.session = requests.Session()
        self.min_request_interval = min_request_interval
        self.last_request_time = 0
        
        logger.info(f"Initialized CoinGeckoClient with rate limit: {min_request_interval}s")
    
    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _request_with_retry(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with exponential backoff retry logic.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            Dict: JSON response
            
        Raises:
            requests.RequestException: If all retries failed
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.session.get(
                    url,
                    params=params,
                    timeout=API_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                logger.debug(f"API request successful: {endpoint}")
                return response.json()
                
            except requests.exceptions.RequestException as e:
                wait_time = RETRY_BACKOFF_FACTOR ** attempt
                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {MAX_RETRIES} API requests failed for {endpoint}")
                    raise
    
    def fetch_current_data(self, asset: str) -> Dict[str, Any]:
        """Fetch current price and volume data for an asset.
        
        Args:
            asset: Asset name (e.g., 'bitcoin', 'ethereum')
            
        Returns:
            Dict: {
                'timestamp': datetime,
                'price': float (USD),
                'volume': float (USD)
            }
            
        Raises:
            ValueError: If asset not found
            requests.RequestException: If API call fails
        """
        logger.debug(f"Fetching current data for {asset}")
        
        try:
            data = self._request_with_retry(
                "simple/price",
                params={
                    "ids": asset,
                    "vs_currencies": "usd",
                    "include_market_cap": True,
                    "include_24hr_vol": True,
                },
            )
            
            if asset not in data:
                raise ValueError(f"Asset {asset} not found in response")
            
            asset_data = data[asset]
            timestamp = datetime.utcnow()
            price = asset_data.get("usd", 0.0)
            volume = asset_data.get("usd_24h_vol", 0.0)
            
            if price <= 0:
                raise ValueError(f"Invalid price for {asset}: {price}")
            
            result = {
                "timestamp": timestamp,
                "price": float(price),
                "volume": float(volume),
            }
            
            logger.debug(f"Current data for {asset}: price=${price}, volume=${volume}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching current data for {asset}: {e}")
            raise
    
    def fetch_historical_data(
        self,
        asset: str,
        days: int = 180,
    ) -> List[Dict[str, Any]]:
        """Fetch historical daily price and volume data.
        
        Args:
            asset: Asset name (e.g., 'bitcoin')
            days: Number of days of historical data (max ~365)
            
        Returns:
            List[Dict]: List of {
                'timestamp': datetime,
                'price': float,
                'volume': float
            }
            
        Raises:
            ValueError: If data validation fails
            requests.RequestException: If API call fails
        """
        logger.info(f"Fetching {days} days historical data for {asset}")
        
        try:
            # CoinGecko returns daily OHLCV data
            data = self._request_with_retry(
                f"coins/{asset}/market_chart",
                params={
                    "vs_currency": "usd",
                    "days": days,
                    "interval": "daily",
                },
            )
            
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            if not prices:
                raise ValueError(f"No price data returned for {asset}")
            
            # Convert timestamps (milliseconds) to datetime
            result = []
            for i, (timestamp_ms, price) in enumerate(prices):
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                volume = volumes[i][1] if i < len(volumes) else 0.0
                
                # Validate data
                if price <= 0:
                    logger.warning(f"Invalid price for {asset} @ {timestamp}: {price}")
                    continue
                
                result.append({
                    "timestamp": timestamp,
                    "price": float(price),
                    "volume": float(volume),
                })
            
            logger.info(f"Retrieved {len(result)} historical data points for {asset}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {asset}: {e}")
            raise
    
    def validate_data(
        self,
        data: List[Dict[str, Any]],
        asset: str,
        max_gap_hours: int = 24,
    ) -> bool:
        """Validate data integrity.
        
        Args:
            data: List of data records
            asset: Asset name (for logging)
            max_gap_hours: Maximum allowed gap between records (hours)
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        logger.debug(f"Validating {len(data)} records for {asset}")
        
        if not data:
            raise ValueError(f"No data to validate for {asset}")
        
        if len(data) < 10:
            logger.warning(f"Very few data points for {asset}: {len(data)}")
        
        # Check timestamps in order and for gaps
        for i, record in enumerate(data):
            timestamp = record.get("timestamp")
            price = record.get("price")
            volume = record.get("volume")
            
            # Validate types and values
            if not isinstance(timestamp, datetime):
                raise ValueError(f"Invalid timestamp type at index {i}")
            
            if not isinstance(price, (int, float)) or price <= 0:
                raise ValueError(f"Invalid price at index {i}: {price}")
            
            if not isinstance(volume, (int, float)) or volume < 0:
                raise ValueError(f"Invalid volume at index {i}: {volume}")
            
            # Check gap from previous record
            if i > 0:
                prev_timestamp = data[i - 1]["timestamp"]
                gap = (timestamp - prev_timestamp).total_seconds() / 3600
                
                if gap > max_gap_hours:
                    logger.warning(f"Large gap in {asset} data: {gap:.1f}h at {timestamp}")
        
        logger.debug(f"Data validation passed for {asset}")
        return True
    
    def fetch_batch(
        self,
        assets: Optional[List[str]] = None,
        days: int = 180,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch historical data for multiple assets.
        
        Args:
            assets: List of asset names (default: ASSETS config)
            days: Days of historical data
            
        Returns:
            Dict: asset name → list of historical data
        """
        assets = assets or ASSETS
        logger.info(f"Fetching historical data for {len(assets)} assets: {assets}")
        
        result = {}
        for asset in tqdm(assets, desc="Fetching historical data"):
            try:
                data = self.fetch_historical_data(asset, days=days)
                self.validate_data(data, asset)
                result[asset] = data
                logger.info(f"Successfully fetched {len(data)} records for {asset}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {asset}: {e}")
                result[asset] = []  # Return empty list on failure
        
        return result


def create_client() -> CoinGeckoClient:
    """Factory function to create and configure a CoinGecko client.
    
    Returns:
        CoinGeckoClient: Configured client instance
    """
    return CoinGeckoClient(min_request_interval=1.0)
