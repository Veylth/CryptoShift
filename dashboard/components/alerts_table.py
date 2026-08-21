"""Alerts table component."""

import streamlit as st
import pandas as pd
from typing import List


def render_alerts_table_tab(selected_assets: List[str], days: int, db):
    """Render alerts table tab with sorting and export.
    
    Args:
        selected_assets: Assets to display
        days: Lookback days
        db: Database module
    """
    st.header("🚨 Alerts Table")
    
    # Fetch all alerts
    all_alerts = []
    for asset in selected_assets:
        try:
            alerts = db.get_alerts(asset, hours=days*24)
            for alert in alerts:
                all_alerts.append({
                    "ID": alert.id,
                    "Timestamp": alert.timestamp.isoformat(),
                    "Asset": asset.upper(),
                    "Price": f"${alert.price:.2f}",
                    "Volume": f"${alert.volume:.2e}",
                    "Detector": alert.detector_name,
                    "Confidence": f"{alert.confidence:.2%}",
                    "Is Real": alert.is_real_anomaly,
                })
        except Exception as e:
            st.warning(f"Error loading alerts for {asset}: {e}")
    
    if not all_alerts:
        st.info("No alerts found for selected period.")
        return
    
    df_alerts = pd.DataFrame(all_alerts)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(df_alerts))
    with col2:
        real = len(df_alerts[df_alerts["Is Real"] == True])
        st.metric("Verified Real", real)
    with col3:
        false_pos = len(df_alerts[df_alerts["Is Real"] == False])
        st.metric("False Positives", false_pos)
    with col4:
        unverified = len(df_alerts[df_alerts["Is Real"].isna()])
        st.metric("Unverified", unverified)
    
    st.divider()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_detector = st.multiselect(
            "Filter by Detector",
            options=df_alerts["Detector"].unique(),
            default=df_alerts["Detector"].unique(),
        )
    
    with col2:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0)
    
    with col3:
        verification_status = st.multiselect(
            "Verification Status",
            options=["Verified Real", "False Positive", "Unverified"],
            default=["Verified Real", "False Positive", "Unverified"],
        )
    
    # Apply filters
    df_filtered = df_alerts.copy()
    
    # Detector filter
    df_filtered = df_filtered[df_filtered["Detector"].isin(selected_detector)]
    
    # Confidence filter
    confidence_values = df_filtered["Confidence"].str.rstrip('%').astype(float) / 100
    df_filtered = df_filtered[confidence_values >= min_confidence]
    
    # Status filter
    status_mask = []
    for idx, row in df_filtered.iterrows():
        if row["Is Real"] == True and "Verified Real" in verification_status:
            status_mask.append(True)
        elif row["Is Real"] == False and "False Positive" in verification_status:
            status_mask.append(True)
        elif pd.isna(row["Is Real"]) and "Unverified" in verification_status:
            status_mask.append(True)
        else:
            status_mask.append(False)
    
    df_filtered = df_filtered[status_mask]
    
    st.divider()
    
    # Display table
    st.subheader(f"📋 Showing {len(df_filtered)} alerts")
    
    st.dataframe(
        df_filtered[["Timestamp", "Asset", "Price", "Detector", "Confidence", "Is Real"]],
        use_container_width=True,
        height=500,
    )
    
    st.divider()
    
    # Export
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Alerts",
            data=csv,
            file_name=f"alerts_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    
    with col2:
        st.metric("Filtered Results", len(df_filtered))
