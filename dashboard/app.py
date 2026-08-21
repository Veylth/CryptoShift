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

# ============================================================================
# CUSTOM STYLING & ANIMATIONS
# ============================================================================

st.markdown("""
<style>
/* Global theme */
:root {
    --primary-color: #00D9FF;
    --secondary-color: #FF006E;
    --accent-color: #FFBE0B;
    --bg-dark: #0A0E27;
    --bg-card: #16213E;
}

/* Main background gradient animation */
.main {
    background: linear-gradient(-45deg, #0A0E27, #1A1F3A, #16213E, #0F3460);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Title animations with gradient shimmer */
h1, h2, h3 {
    background: linear-gradient(120deg, #00D9FF, #FF006E, #FFBE0B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

/* Smooth transitions */
* {
    transition: all 0.3s ease;
}

/* Metric cards with glow effect */
.stMetric {
    background: rgba(22, 33, 62, 0.8) !important;
    border-radius: 15px !important;
    border: 1px solid rgba(0, 217, 255, 0.3) !important;
    padding: 20px !important;
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
}

.stMetric:hover {
    box-shadow: 0 0 30px rgba(0, 217, 255, 0.3) !important;
    transform: translateY(-5px);
    border-color: rgba(0, 217, 255, 0.6) !important;
}

/* Button animations - Bubble ripple effect */
.stButton > button {
    background: linear-gradient(135deg, #00D9FF, #FF006E) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 30px !important;
    font-weight: 600 !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(0, 217, 255, 0.4) !important;
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0 8px 25px rgba(0, 217, 255, 0.6) !important;
}

.stButton > button:active {
    transform: scale(0.95);
}

/* Tab animations */
.stTabs [data-baseweb="tab-list"] button {
    border-radius: 20px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #00D9FF, #FF006E) !important;
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.5) !important;
    transform: scale(1.05);
}

.stTabs [data-baseweb="tab-list"] button:hover {
    background: rgba(0, 217, 255, 0.2) !important;
    transform: translateY(-2px);
}

/* Input field styling */
.stTextInput > div > div > input,
.stSelectbox > div > div > select,
.stNumberInput > div > div > input {
    background: rgba(22, 33, 62, 0.8) !important;
    border: 1px solid rgba(0, 217, 255, 0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus,
.stNumberInput > div > div > input:focus {
    border-color: #00D9FF !important;
    box-shadow: 0 0 15px rgba(0, 217, 255, 0.4) !important;
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 10px !important;
    border: 1px solid rgba(0, 217, 255, 0.2) !important;
}

/* Expander animations */
.stExpander {
    border-radius: 10px !important;
    border: 1px solid rgba(0, 217, 255, 0.2) !important;
}

.stExpander > div > button:hover {
    background: rgba(0, 217, 255, 0.1) !important;
}

/* Sidebar styling */
.stSidebar {
    background: linear-gradient(180deg, rgba(16, 33, 60, 0.9) 0%, rgba(22, 33, 62, 0.9) 100%) !important;
    border-right: 1px solid rgba(0, 217, 255, 0.2) !important;
}

/* Smooth fade-in for content */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.stMarkdown, .stMetric {
    animation: fadeInUp 0.6s ease-out;
}

/* Chart styling */
.plotly-graph-div {
    border-radius: 15px !important;
    box-shadow: 0 8px 32px rgba(0, 217, 255, 0.1) !important;
}

/* Message styling */
.stSuccess {
    border-radius: 10px !important;
    background-color: rgba(0, 217, 255, 0.15) !important;
    border: 1px solid #00D9FF !important;
}

.stWarning {
    border-radius: 10px !important;
    background-color: rgba(255, 190, 11, 0.15) !important;
    border: 1px solid #FFBE0B !important;
}

.stError {
    border-radius: 10px !important;
    background-color: rgba(255, 0, 110, 0.15) !important;
    border: 1px solid #FF006E !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 217, 255, 0.05);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00D9FF, #FF006E);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    box-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
}

</style>
""", unsafe_allow_html=True)

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

# Enhanced title with custom styling
st.markdown("""
<div style="text-align: center; margin: 30px 0;">
    <h1 style="font-size: 3em; margin: 0; letter-spacing: 2px;">
        📈 CryptoShift
    </h1>
    <p style="font-size: 1.3em; color: #00D9FF; margin: 10px 0; font-weight: 500;">
        Real-Time Cryptocurrency Anomaly Detection
    </p>
    <p style="font-size: 0.95em; color: #AAAAAA; margin: 5px 0;">
        Ensemble ML • Walk-Forward Backtesting • Interactive Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# Create tabs with enhanced styling
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
