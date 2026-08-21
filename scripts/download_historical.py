"""Download historical cryptocurrency data from CoinGecko."""

import logging
from datetime import datetime

from src.config import ASSETS, DATA_DIR, logger as config_logger
from src.data.fetcher import create_client
from src.data.database import init_db, add_price_data

logger = config_logger


def main():
    """Download 6 months of historical data for all assets."""
    logger.info("Starting historical data download...")
    
    try:
        # Initialize database
        init_db()
        
        # Create CoinGecko client
        client = create_client()
        
        # Download data for each asset
        for asset in ASSETS:
            logger.info(f"Downloading {asset} (180 days)...")
            
            try:
                # Fetch historical data
                data = client.fetch_historical_data(asset, days=180)
                
                # Validate
                client.validate_data(data, asset)
                
                # Store in database
                added_count = 0
                for record in data:
                    result = add_price_data(
                        asset=asset,
                        timestamp=record["timestamp"],
                        price=record["price"],
                        volume=record["volume"],
                    )
                    if result:
                        added_count += 1
                
                logger.info(f"Added {added_count} records for {asset}")
                
            except Exception as e:
                logger.error(f"Failed to download {asset}: {e}")
                continue
        
        logger.info("Historical data download completed!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
