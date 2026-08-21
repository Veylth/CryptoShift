"""CryptoShift Streamlit Dashboard."""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

# Configure Streamlit
st.set_page_config(
    page_title="CryptoShift Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import components
from dashboard.components.overview import render_overview_tab
from dashboard.components.anomaly_explorer import render_anomaly_explorer_tab
from dashboard.components.model_performance import render_performance_tab
from dashboard.components.alerts_table import render_alerts_table_tab
from dashboard.components.feature_analysis import render_feature_analysis_tab

from src.data import database
from src.config import ASSETS


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.title("⚙️ CryptoShift Settings")

# Asset selection
selected_assets = st.sidebar.multiselect(
    "Select Assets",
    options=ASSETS,
    default=["bitcoin", "ethereum"],
    help="Choose which cryptocurrencies to analyze"
)

# Date range
date_range = st.sidebar.slider(
    "Lookback Period (days)",
    min_value=1,
    max_value=180,
    value=7,
    help="How many days of historical data to display"
)

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum confidence score for anomaly alerts"
)

# Detector selection
detectors = st.sidebar.multiselect(
    "Detectors to Show",
    options=["isolation_forest", "zscore", "ewma", "ensemble"],
    default=["ensemble"],
    help="Which detectors' results to display"
)

# Refresh button
if st.sidebar.button("🔄 Refresh Data", key="refresh_btn"):
    st.rerun()

# Display settings info
st.sidebar.divider()
st.sidebar.info(
    f"**Active Filters:**\n"
    f"- Assets: {len(selected_assets)} selected\n"
    f"- Lookback: {date_range} days\n"
    f"- Min Confidence: {confidence_threshold:.2f}\n"
    f"- Detectors: {len(detectors)} selected"
)

# Last update timestamp
st.sidebar.divider()
last_update = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
st.sidebar.caption(f"Last updated: {last_update}")


# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================

st.title("📈 CryptoShift - Cryptocurrency Anomaly Detection")
st.markdown(
    "Real-time anomaly detection system for Bitcoin, Ethereum, and Solana "
    "using ensemble machine learning models."
)

# Create tabs
tabs = st.tabs(
    [
        "📊 Overview",
        "🔍 Anomaly Explorer",
        "📈 Model Performance",
        "📋 Feature Analysis",
        "🚨 Alerts Table",
    ]
)

# Tab 1: Overview
with tabs[0]:
    try:
        render_overview_tab(selected_assets, date_range, database)
    except Exception as e:
        st.error(f"Error loading overview tab: {e}")
        logger.error(f"Overview tab error: {e}", exc_info=True)

# Tab 2: Anomaly Explorer
with tabs[1]:
    try:
        render_anomaly_explorer_tab(
            selected_assets,
            date_range,
            confidence_threshold,
            detectors,
            database,
        )
    except Exception as e:
        st.error(f"Error loading anomaly explorer: {e}")
        logger.error(f"Anomaly explorer error: {e}", exc_info=True)

# Tab 3: Model Performance
with tabs[2]:
    try:
        render_performance_tab(database)
    except Exception as e:
        st.error(f"Error loading performance tab: {e}")
        logger.error(f"Performance tab error: {e}", exc_info=True)

# Tab 4: Feature Analysis
with tabs[3]:
    try:
        render_feature_analysis_tab(selected_assets, date_range, database)
    except Exception as e:
        st.error(f"Error loading feature analysis: {e}")
        logger.error(f"Feature analysis error: {e}", exc_info=True)

# Tab 5: Alerts Table
with tabs[4]:
    try:
        render_alerts_table_tab(selected_assets, date_range, database)
    except Exception as e:
        st.error(f"Error loading alerts table: {e}")
        logger.error(f"Alerts table error: {e}", exc_info=True)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(
    """
    **CryptoShift v1.0** | Real-time Cryptocurrency Anomaly Detection  
    Built with FastAPI, Streamlit, and scikit-learn  
    [GitHub](https://github.com/cryptoshift) • [Docs](https://docs.cryptoshift.io)
    """
)
