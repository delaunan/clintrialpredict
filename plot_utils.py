import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. MATH & LOGIC HELPERS ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def map_feature_to_business_pillar(feature_name, taxonomy_dict):
    """Maps a raw feature to a pillar using the saved dictionary."""
    pillar = taxonomy_dict['pillar_map'].get(feature_name, 'Other')
    subcat = taxonomy_dict['subcat_map'].get(feature_name, 'Unclassified')
    return pillar, subcat

# --- 2. THE CORE CALCULATION ENGINE ---
def calculate_risk_drivers(model, explainer, taxonomy, row_data):
    """
    Runs SHAP, applies Linearization (The "Sync" Math), and returns
    dataframes for both Pillars (Bar Chart) and Subcategories (Treemap).
    """
    # 1. Preprocess the single row
    try:
        preprocessor = model.named_steps['preprocessor']
        X_encoded = preprocessor.transform(row_data)
        feature_names = taxonomy['feature_names']
        X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)
    except Exception as e:
        return None, None, None, f"Data Transformation Error: {str(e)}"

    # 2. Calculate SHAP values
    shap_values = explainer(X_encoded_df)

    # 3. Get Base Value & Probabilities
    base_log_odds = shap_values.base_values[0]
    prob_fail_base = sigmoid(base_log_odds)
    prob_success_base = 1 - prob_fail_base

    # 4. Aggregate SHAP values
    impacts = []
    for i, col in enumerate(feature_names):
        val = shap_values.values[0][i]
        if abs(val) < 1e-9: continue

        pillar, subcat = map_feature_to_business_pillar(col, taxonomy)
        if pillar == 'Other': continue

        impacts.append({
            'Pillar': pillar,
            'Subcategory': subcat,
            'Feature': col,
            'Raw_Log_Odds': val
        })

    df_impacts = pd.DataFrame(impacts)

    # 5. Calculate Final Probability
    total_shap_sum = df_impacts['Raw_Log_Odds'].sum()
    final_log_odds = base_log_odds + total_shap_sum
    prob_fail_final = sigmoid(final_log_odds)
    prob_success_final = 1 - prob_fail_final

    # --- 6. THE SYNC LOGIC (Linearization) ---
    # This calculates the gap between Base and Final probability
    # and distributes it proportionally to the features.
    total_gap = prob_success_final - prob_success_base

    if total_shap_sum == 0:
        scale_factor = 0
    else:
        scale_factor = total_gap / total_shap_sum

    # Apply scaling to the detailed rows
    df_impacts['Impact_Pct'] = df_impacts['Raw_Log_Odds'] * scale_factor

    # Group by Pillar (Summing the already scaled values ensures perfect sync)
    df_pillars = df_impacts.groupby('Pillar')['Impact_Pct'].sum().reset_index()

    # Return Pillars (for Bar), Detailed (for Treemap), and Final Score (for Gauge)
    return df_pillars, df_impacts, prob_success_final, None

# --- 3. PLOTTING FUNCTIONS ---

def plot_success_gauge(prob_success):
    """Red-to-Green Gauge."""
    score_val = prob_success * 100
    steps = []
    c_min, c_mid, c_max = (139, 0, 0), (255, 255, 255), (0, 100, 0)
    for i in range(100):
        if i < 50:
            ratio = i / 50
            r = int(c_min[0] + (c_mid[0] - c_min[0]) * ratio)
            g = int(c_min[1] + (c_mid[1] - c_min[1]) * ratio)
            b = int(c_min[2] + (c_mid[2] - c_min[2]) * ratio)
        else:
            ratio = (i - 50) / 50
            r = int(c_mid[0] + (c_max[0] - c_mid[0]) * ratio)
            g = int(c_mid[1] + (c_max[1] - c_mid[1]) * ratio)
            b = int(c_mid[2] + (c_max[2] - c_mid[2]) * ratio)
        steps.append({'range': [i, i+1], 'color': f"rgb({r},{g},{b})"})

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score_val,
        number = {'suffix': "%", 'font': {'size': 24, 'color': 'black'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ccc"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': steps,
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 1, 'value': score_val}
        }
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10))
    return fig

def plot_impact_bar(df_pillars):
    """Diverging Bar Chart (Pillar Level)."""
    df_sorted = df_pillars.sort_values('Impact_Pct', ascending=True)
    colors = df_sorted['Impact_Pct'].apply(lambda x: '#006400' if x > 0 else '#8B0000')

    fig = px.bar(
        df_sorted,
        x='Impact_Pct',
        y='Pillar',
        orientation='h',
        text_auto='.1%'
    )
    fig.update_traces(marker_color=colors, textposition='outside')
    fig.update_layout(
        xaxis_title="Impact on Success Probability",
        yaxis_title="",
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    fig.add_vline(x=0, line_width=1, line_color="black")
    return fig

def plot_treemap(df_impacts):
    """Treemap showing Subcategories."""
    # We use absolute impact for size, but color indicates direction
    df_impacts['Abs_Impact'] = df_impacts['Impact_Pct'].abs()

    fig = px.treemap(
        df_impacts,
        path=[px.Constant("All Drivers"), 'Pillar', 'Subcategory'],
        values='Abs_Impact',
        color='Impact_Pct',
        color_continuous_scale=['#8B0000', '#ffffff', '#006400'],
        color_continuous_midpoint=0,
        hover_data=['Feature']
    )
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=300)
    return fig
