"""Anomaly explorer component."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any


def render_anomaly_explorer_tab(
    selected_assets: List[str],
    days: int,
    confidence_threshold: float,
    detectors: List[str],
    db,
):
    """Render anomaly explorer with filterable table.
    
    Args:
        selected_assets: Assets to display
        days: Lookback days
        confidence_threshold: Minimum confidence
        detectors: Selected detectors
        db: Database module
    """
    st.header("🔍 Anomaly Explorer")
    
    # Fetch anomalies
    anomalies = []
    for asset in selected_assets:
        try:
            alerts = db.get_alerts(asset, hours=days*24)
            for alert in alerts:
                if alert.confidence >= confidence_threshold:
                    anomalies.append({
                        "Timestamp": alert.timestamp.isoformat(),
                        "Asset": asset.upper(),
                        "Price": f"${alert.price:.2f}",
                        "Volume": f"${alert.volume:.2e}",
                        "Detector": alert.detector_name,
                        "Confidence": f"{alert.confidence:.2%}",
                        "Real Anomaly": str(alert.is_real_anomaly) if alert.is_real_anomaly is not None else "Unverified",
                        "ID": alert.id,
                    })
        except Exception as e:
            st.warning(f"Error loading {asset}: {e}")
    
    if not anomalies:
        st.info("No anomalies found with selected filters.")
        return
    
    df_anomalies = pd.DataFrame(anomalies)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anomalies", len(df_anomalies))
    with col2:
        real_count = len(df_anomalies[df_anomalies["Real Anomaly"] == "True"])
        st.metric("Verified Real", real_count)
    with col3:
        false_count = len(df_anomalies[df_anomalies["Real Anomaly"] == "False"])
        st.metric("False Positives", false_count)
    
    st.divider()
    
    # Sortable table
    st.subheader("📋 Anomalies (sorted by timestamp)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔎 Search by asset or timestamp", "")
    with col2:
        sort_by = st.selectbox("Sort by", ["Timestamp", "Confidence", "Asset"])
    
    # Filter and sort
    if search_term:
        mask = df_anomalies["Timestamp"].str.contains(search_term, case=False) | \
               df_anomalies["Asset"].str.contains(search_term, case=False)
        df_display = df_anomalies[mask]
    else:
        df_display = df_anomalies.copy()
    
    # Sort
    if sort_by == "Confidence":
        df_display = df_display.sort_values("Confidence", ascending=False)
    elif sort_by == "Asset":
        df_display = df_display.sort_values("Asset")
    
    # Display table
    st.dataframe(
        df_display[["Timestamp", "Asset", "Price", "Detector", "Confidence", "Real Anomaly"]],
        use_container_width=True,
        height=400,
    )
    
    st.divider()
    
    # Export button
    col1, col2 = st.columns(2)
    with col1:
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"anomalies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    
    with col2:
        st.info(f"Showing {len(df_display)} of {len(df_anomalies)} anomalies")
