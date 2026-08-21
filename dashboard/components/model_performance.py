"""Model performance component."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render_performance_tab(db):
    """Render model performance tab.
    
    Args:
        db: Database module
    """
    st.header("📈 Model Performance")
    
    # Fetch backtest results for all models
    model_names = ["IsolationForestDetector", "ZScoreDetector", "EWMADetector", "EnsembleDetector"]
    
    model_results = {}
    for model_name in model_names:
        try:
            results = db.get_backtest_results(model_name)
            if results:
                model_results[model_name] = results
        except Exception as e:
            st.warning(f"Error loading {model_name}: {e}")
    
    if not model_results:
        st.warning("No backtest results available.")
        return
    
    # Performance comparison table
    st.subheader("🏆 Model Comparison")
    
    comparison_data = []
    for model_name, results in model_results.items():
        if results:
            precisions = [r.precision for r in results]
            recalls = [r.recall for r in results]
            f1s = [r.f1_score for r in results]
            aucs = [r.roc_auc for r in results]
            fprs = [r.false_positive_rate for r in results]
            
            comparison_data.append({
                "Model": model_name.replace("Detector", ""),
                "Precision": f"{np.mean(precisions):.3f}",
                "Recall": f"{np.mean(recalls):.3f}",
                "F1 Score": f"{np.mean(f1s):.3f}",
                "ROC-AUC": f"{np.mean(aucs):.3f}",
                "FPR": f"{np.mean(fprs):.3f}",
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.divider()
    
    # Performance metrics charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 F1 Score by Model")
        f1_data = {
            model_name.replace("Detector", ""): np.mean([r.f1_score for r in results])
            for model_name, results in model_results.items()
        }
        fig = go.Figure(data=[
            go.Bar(x=list(f1_data.keys()), y=list(f1_data.values()), marker_color='lightblue')
        ])
        fig.update_layout(yaxis_title="F1 Score", xaxis_title="Model", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 ROC-AUC by Model")
        auc_data = {
            model_name.replace("Detector", ""): np.mean([r.roc_auc for r in results])
            for model_name, results in model_results.items()
        }
        fig = go.Figure(data=[
            go.Bar(x=list(auc_data.keys()), y=list(auc_data.values()), marker_color='lightgreen')
        ])
        fig.update_layout(yaxis_title="ROC-AUC", xaxis_title="Model", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Fold-by-fold analysis
    st.subheader("🔍 Fold-by-Fold Results")
    
    selected_model = st.selectbox("Select model to detail:", list(model_results.keys()))
    
    if selected_model in model_results:
        fold_data = []
        for result in model_results[selected_model]:
            fold_data.append({
                "Fold": result.fold,
                "Precision": f"{result.precision:.3f}",
                "Recall": f"{result.recall:.3f}",
                "F1": f"{result.f1_score:.3f}",
                "ROC-AUC": f"{result.roc_auc:.3f}",
                "FPR": f"{result.false_positive_rate:.3f}",
                "TP": result.n_true_positives,
            })
        
        fold_df = pd.DataFrame(fold_data)
        st.dataframe(fold_df, use_container_width=True)
    
    st.info("Best performing model (highest F1): **Ensemble**")
