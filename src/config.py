"""Configuration settings for CryptoShift."""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# BASIC SETTINGS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# CRYPTOCURRENCY ASSETS
# ============================================================================

ASSETS = ["bitcoin", "ethereum", "solana"]
ASSET_SYMBOLS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
}

# ============================================================================
# API CONFIGURATION
# ============================================================================

COIN_GECKO_API = "https://api.coingecko.com/api/v3"
POLLING_INTERVAL = 60  # seconds
API_REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2  # exponential backoff

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_PATH = DATA_DIR / "cryptoshift.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

VOLATILITY_WINDOW = 24  # hours
MOMENTUM_WINDOW = 24  # hours
EWMA_ALPHA = 0.3

# ============================================================================
# ANOMALY DETECTION CONFIGURATION
# ============================================================================

# Z-Score Detector
ZSCORE_THRESHOLD = 3.0

# Isolation Forest
ISOLATION_FOREST_N_ESTIMATORS = 100
ISOLATION_FOREST_CONTAMINATION = 0.05
ISOLATION_FOREST_RANDOM_STATE = 42

# EWMA Detector
EWMA_STD_THRESHOLD = 2.0

# Ensemble voting
ENSEMBLE_VOTING_THRESHOLD = 2  # 2 out of 3 detectors must agree

# ============================================================================
# BACKTESTING CONFIGURATION
# ============================================================================

BACKTEST_TRAIN_MONTHS = 3
BACKTEST_TEST_MONTHS = 1
BACKTEST_RANDOM_STATE = 42
BACKTEST_ANOMALY_PERCENTILE = 0.95  # Top 5% marked as anomalies

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

DASHBOARD_THEME = "dark"
DASHBOARD_DEFAULT_LOOKBACK = 7  # days
DASHBOARD_REFRESH_INTERVAL = 60  # seconds

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "cryptoshift.log"

# Configure root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# DATA VALIDATION SETTINGS
# ============================================================================

MIN_DATA_POINTS = 10  # minimum rows for feature engineering
MIN_PRICE_VALUE = 0.00001  # minimum valid price
MAX_VOLUME_OUTLIER_STD = 10  # mark as outlier if > 10 std

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

BATCH_SIZE = 100  # rows per batch
CACHE_TIMEOUT = 300  # seconds

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG_MODE = ENVIRONMENT == "development"

if DEBUG_MODE:
    logger.info(f"Running in {ENVIRONMENT} mode")
