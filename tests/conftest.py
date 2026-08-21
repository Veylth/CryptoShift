"""Pytest configuration and fixtures for CryptoShift tests."""

import pytest
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import init_db, get_session, engine


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Initialize database before running tests."""
    # Create tables
    try:
        init_db()
        yield
        # Cleanup after tests
        from src.data.database import Base
        Base.metadata.drop_all(bind=engine)
    except Exception as e:
        print(f"Error setting up database: {e}")
        raise


@pytest.fixture
def db_session():
    """Provide a database session for tests."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()
