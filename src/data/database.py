"""Database models and session management for CryptoShift."""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    UniqueConstraint,
    Index,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from src.config import DATABASE_URL, logger as config_logger

logger = config_logger

Base = declarative_base()


class PriceData(Base):
    """SQLAlchemy model for raw cryptocurrency price data."""
    
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True)
    asset = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        UniqueConstraint("asset", "timestamp", name="uq_price_asset_timestamp"),
        Index("idx_price_asset_timestamp", "asset", "timestamp"),
    )
    
    def __repr__(self) -> str:
        """String representation of PriceData."""
        return f"<PriceData(asset={self.asset}, timestamp={self.timestamp}, price={self.price})>"


class Features(Base):
    """SQLAlchemy model for engineered features."""
    
    __tablename__ = "features"
    
    id = Column(Integer, primary_key=True)
    asset = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price_zscore = Column(Float, nullable=True)
    volume_zscore = Column(Float, nullable=True)
    volatility_1h = Column(Float, nullable=True)
    momentum_1h = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        UniqueConstraint("asset", "timestamp", name="uq_features_asset_timestamp"),
        Index("idx_features_asset_timestamp", "asset", "timestamp"),
    )
    
    def __repr__(self) -> str:
        """String representation of Features."""
        return f"<Features(asset={self.asset}, timestamp={self.timestamp})>"


class Alert(Base):
    """SQLAlchemy model for anomaly alerts."""
    
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True)
    asset = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    detector_name = Column(String(100), nullable=False)  # isolation_forest, zscore, ewma, ensemble
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    is_real_anomaly = Column(Boolean, nullable=True)  # User verification
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_alerts_asset_timestamp", "asset", "timestamp"),
        Index("idx_alerts_detector", "detector_name"),
        Index("idx_alerts_confidence", "confidence"),
    )
    
    def __repr__(self) -> str:
        """String representation of Alert."""
        return f"<Alert(asset={self.asset}, detector={self.detector_name}, confidence={self.confidence})>"


class BacktestResult(Base):
    """SQLAlchemy model for backtesting results."""
    
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False, index=True)
    asset = Column(String(50), nullable=False, index=True)
    fold = Column(Integer, nullable=False)
    fold_start_date = Column(DateTime, nullable=False)
    fold_end_date = Column(DateTime, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    false_positive_rate = Column(Float, nullable=False)
    true_positive_rate = Column(Float, nullable=False)
    n_anomalies_detected = Column(Integer, nullable=False)
    n_true_positives = Column(Integer, nullable=False)
    n_false_positives = Column(Integer, nullable=False)
    n_true_negatives = Column(Integer, nullable=False)
    n_false_negatives = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_backtest_model_asset", "model_name", "asset"),
        Index("idx_backtest_dates", "fold_start_date", "fold_end_date"),
    )
    
    def __repr__(self) -> str:
        """String representation of BacktestResult."""
        return f"<BacktestResult(model={self.model_name}, fold={self.fold}, f1={self.f1_score})>"


# Global session factory
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create database engine.
    
    Returns:
        sqlalchemy.Engine: Database engine
    """
    global _engine
    if _engine is None:
        logger.info(f"Creating database engine: {DATABASE_URL}")
        _engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False,
        )
    return _engine


def get_session_factory():
    """Get or create session factory.
    
    Returns:
        sqlalchemy.orm.sessionmaker: Session factory
    """
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def get_session() -> Session:
    """Get a new database session.
    
    Returns:
        Session: SQLAlchemy session
        
    Yields:
        Session: Database session
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        return session
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise
    finally:
        pass  # Session is returned, caller must close


def init_db() -> None:
    """Initialize database: create all tables.
    
    Creates all tables defined in SQLAlchemy models.
    Safe to call multiple times (idempotent).
    """
    try:
        engine = get_engine()
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def add_price_data(asset: str, timestamp: datetime, price: float, volume: float) -> Optional[PriceData]:
    """Add price data to database.
    
    Args:
        asset: Asset name (e.g., 'bitcoin', 'ethereum')
        timestamp: Data timestamp
        price: Asset price in USD
        volume: 24h trading volume in USD
        
    Returns:
        PriceData: Created record, or None if duplicate
    """
    session = get_session()
    try:
        # Check if already exists
        existing = session.query(PriceData).filter(
            PriceData.asset == asset,
            PriceData.timestamp == timestamp,
        ).first()
        
        if existing:
            logger.debug(f"Price data already exists: {asset} @ {timestamp}")
            return None
        
        # Create new record
        record = PriceData(
            asset=asset,
            timestamp=timestamp,
            price=price,
            volume=volume,
        )
        session.add(record)
        session.commit()
        logger.debug(f"Added price data: {asset} @ {timestamp} = ${price}")
        return record
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding price data: {e}")
        raise
    finally:
        session.close()


def add_feature(asset: str, timestamp: datetime, features_dict: Dict[str, float]) -> Optional[Features]:
    """Add engineered features to database.
    
    Args:
        asset: Asset name
        timestamp: Feature timestamp
        features_dict: Dictionary with keys: price_zscore, volume_zscore, volatility_1h, momentum_1h
        
    Returns:
        Features: Created record, or None if duplicate
    """
    session = get_session()
    try:
        # Check if already exists
        existing = session.query(Features).filter(
            Features.asset == asset,
            Features.timestamp == timestamp,
        ).first()
        
        if existing:
            logger.debug(f"Features already exist: {asset} @ {timestamp}")
            return None
        
        # Create new record
        record = Features(
            asset=asset,
            timestamp=timestamp,
            price_zscore=features_dict.get("price_zscore"),
            volume_zscore=features_dict.get("volume_zscore"),
            volatility_1h=features_dict.get("volatility_1h"),
            momentum_1h=features_dict.get("momentum_1h"),
        )
        session.add(record)
        session.commit()
        logger.debug(f"Added features: {asset} @ {timestamp}")
        return record
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding features: {e}")
        raise
    finally:
        session.close()


def add_alert(
    asset: str,
    timestamp: datetime,
    price: float,
    volume: float,
    detector_name: str,
    confidence: float,
    model_version: Optional[str] = None,
) -> Alert:
    """Add anomaly alert to database.
    
    Args:
        asset: Asset name
        timestamp: Alert timestamp
        price: Asset price at alert time
        volume: Volume at alert time
        detector_name: Name of detector that generated alert
        confidence: Confidence score (0.0 to 1.0)
        model_version: Optional model version string
        
    Returns:
        Alert: Created record
    """
    session = get_session()
    try:
        record = Alert(
            asset=asset,
            timestamp=timestamp,
            price=price,
            volume=volume,
            detector_name=detector_name,
            confidence=confidence,
            model_version=model_version,
        )
        session.add(record)
        session.commit()
        logger.info(f"Alert created: {asset} {detector_name} confidence={confidence:.2f}")
        return record
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding alert: {e}")
        raise
    finally:
        session.close()


def mark_alert_verified(alert_id: int, is_real: bool) -> Optional[Alert]:
    """Mark an alert as verified (real or false positive).
    
    Args:
        alert_id: Alert ID
        is_real: True if real anomaly, False if false positive
        
    Returns:
        Alert: Updated record, or None if not found
    """
    session = get_session()
    try:
        alert = session.query(Alert).filter(Alert.id == alert_id).first()
        
        if not alert:
            logger.warning(f"Alert not found: {alert_id}")
            return None
        
        alert.is_real_anomaly = is_real
        session.commit()
        logger.info(f"Alert {alert_id} marked as {'real' if is_real else 'false positive'}")
        return alert
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error marking alert verified: {e}")
        raise
    finally:
        session.close()


def get_price_data(asset: str, hours: int = 24) -> List[PriceData]:
    """Get recent price data.
    
    Args:
        asset: Asset name
        hours: Look-back hours (default 24)
        
    Returns:
        List[PriceData]: Recent price records ordered by timestamp ascending
    """
    session = get_session()
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        records = session.query(PriceData).filter(
            PriceData.asset == asset,
            PriceData.timestamp >= cutoff_time,
        ).order_by(PriceData.timestamp.asc()).all()
        
        logger.debug(f"Retrieved {len(records)} price records for {asset} (last {hours}h)")
        return records
        
    except Exception as e:
        logger.error(f"Error retrieving price data: {e}")
        raise
    finally:
        session.close()


def get_alerts(asset: str, hours: int = 24) -> List[Alert]:
    """Get recent alerts.
    
    Args:
        asset: Asset name
        hours: Look-back hours (default 24)
        
    Returns:
        List[Alert]: Recent alerts ordered by timestamp descending
    """
    session = get_session()
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        records = session.query(Alert).filter(
            Alert.asset == asset,
            Alert.timestamp >= cutoff_time,
        ).order_by(Alert.timestamp.desc()).all()
        
        logger.debug(f"Retrieved {len(records)} alerts for {asset} (last {hours}h)")
        return records
        
    except Exception as e:
        logger.error(f"Error retrieving alerts: {e}")
        raise
    finally:
        session.close()


def get_backtest_results(model_name: str, asset: Optional[str] = None) -> List[BacktestResult]:
    """Get backtest results for a model.
    
    Args:
        model_name: Model name (e.g., 'isolation_forest', 'ensemble')
        asset: Optional asset filter
        
    Returns:
        List[BacktestResult]: Backtest results ordered by fold ascending
    """
    session = get_session()
    try:
        query = session.query(BacktestResult).filter(BacktestResult.model_name == model_name)
        
        if asset:
            query = query.filter(BacktestResult.asset == asset)
        
        records = query.order_by(BacktestResult.fold.asc()).all()
        
        logger.debug(f"Retrieved {len(records)} backtest results for {model_name}")
        return records
        
    except Exception as e:
        logger.error(f"Error retrieving backtest results: {e}")
        raise
    finally:
        session.close()


def add_backtest_result(
    model_name: str,
    asset: str,
    fold: int,
    fold_start_date: datetime,
    fold_end_date: datetime,
    precision: float,
    recall: float,
    f1_score: float,
    roc_auc: float,
    false_positive_rate: float,
    true_positive_rate: float,
    n_anomalies_detected: int,
    n_true_positives: int,
    n_false_positives: int,
    n_true_negatives: int,
    n_false_negatives: int,
) -> BacktestResult:
    """Add backtest result to database.
    
    Args:
        model_name: Model name
        asset: Asset name
        fold: Fold number
        fold_start_date: Fold start date
        fold_end_date: Fold end date
        precision: Precision metric
        recall: Recall metric
        f1_score: F1 score
        roc_auc: ROC AUC score
        false_positive_rate: False positive rate
        true_positive_rate: True positive rate
        n_anomalies_detected: Number of anomalies detected
        n_true_positives: Number of true positives
        n_false_positives: Number of false positives
        n_true_negatives: Number of true negatives
        n_false_negatives: Number of false negatives
        
    Returns:
        BacktestResult: Created record
    """
    session = get_session()
    try:
        record = BacktestResult(
            model_name=model_name,
            asset=asset,
            fold=fold,
            fold_start_date=fold_start_date,
            fold_end_date=fold_end_date,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            roc_auc=roc_auc,
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            n_anomalies_detected=n_anomalies_detected,
            n_true_positives=n_true_positives,
            n_false_positives=n_false_positives,
            n_true_negatives=n_true_negatives,
            n_false_negatives=n_false_negatives,
        )
        session.add(record)
        session.commit()
        logger.info(f"Backtest result saved: {model_name} fold={fold} f1={f1_score:.3f}")
        return record
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding backtest result: {e}")
        raise
    finally:
        session.close()


def clear_old_data(days: int = 90) -> int:
    """Delete old data from database (older than N days).
    
    Args:
        days: Delete data older than this many days
        
    Returns:
        int: Number of records deleted
    """
    session = get_session()
    try:
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Delete old price data
        deleted_price = session.query(PriceData).filter(
            PriceData.created_at < cutoff_time
        ).delete()
        
        # Delete old alerts
        deleted_alerts = session.query(Alert).filter(
            Alert.created_at < cutoff_time
        ).delete()
        
        session.commit()
        total_deleted = deleted_price + deleted_alerts
        logger.info(f"Deleted {total_deleted} records older than {days} days")
        return total_deleted
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error clearing old data: {e}")
        raise
    finally:
        session.close()
