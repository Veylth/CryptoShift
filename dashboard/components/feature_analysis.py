"""Feature analysis component."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List


def render_feature_analysis_tab(selected_assets: List[str], days: int, db):
    """Render feature analysis tab.
    
    Args:
        selected_assets: Assets to analyze
        days: Lookback days
        db: Database module
    """
    st.header("📊 Feature Analysis")
    
    # Collect feature data
    all_features = []
    for asset in selected_assets:
        try:
            prices = db.get_price_data(asset, hours=days*24)
            for price_rec in prices:
                all_features.append({
                    "asset": asset,
                    "timestamp": price_rec.timestamp,
                    "price": price_rec.price,
                    "volume": price_rec.volume,
                })
        except Exception as e:
            st.warning(f"Error loading features for {asset}: {e}")
    
    if not all_features:
        st.info("No feature data available.")
        return
    
    df_features = pd.DataFrame(all_features)
    
    # Feature statistics
    st.subheader("📈 Feature Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Price", f"${df_features['price'].mean():.2f}")
    with col2:
        st.metric("Avg Volume", f"${df_features['volume'].mean():.2e}")
    with col3:
        st.metric("Total Records", len(df_features))
    
    st.divider()
    
    # Price distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price Distribution")
        fig = go.Figure(data=[
            go.Histogram(x=df_features['price'], nbinsx=50, marker_color='skyblue')
        ])
        fig.update_layout(height=400, xaxis_title="Price (USD)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Volume Distribution")
        fig = go.Figure(data=[
            go.Histogram(x=df_features['volume'], nbinsx=50, marker_color='lightgreen')
        ])
        fig.update_layout(height=400, xaxis_title="Volume (USD)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Asset-specific analysis
    st.subheader("🔍 Asset Breakdown")
    
    for asset in selected_assets:
        asset_data = df_features[df_features["asset"] == asset]
        
        with st.expander(f"{asset.upper()} Details", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Min Price", f"${asset_data['price'].min():.2f}")
            with col2:
                st.metric("Max Price", f"${asset_data['price'].max():.2f}")
            with col3:
                st.metric("Mean Volume", f"${asset_data['volume'].mean():.2e}")
            with col4:
                st.metric("Data Points", len(asset_data))
    
    st.info(
        "**Feature Engineering**: Uses rolling 24-hour windows for "
        "volatility and momentum calculations."
    )
