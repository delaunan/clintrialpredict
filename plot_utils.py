import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. MATH & LOGIC HELPERS ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def map_feature_to_business_pillar(feature_name, taxonomy_dict):
    """Maps a raw feature to a pillar using the saved dictionary."""
    # Try exact match first
    if feature_name in taxonomy_dict['pillar_map']:
        return taxonomy_dict['pillar_map'][feature_name], taxonomy_dict['subcat_map'][feature_name]

    # Fallback logic if feature wasn't in training set (Safety net)
    name = str(feature_name).lower()
    if 'emb_' in name or 'pca' in name: return 'Patient & Criteria', 'Criteria Complexity (AI)'
    if 'sponsor' in name: return 'Sponsor & Operations', 'Sponsor Capability'
    return 'Other', 'Unclassified'

def get_business_hypothesis(subcategory, impact_val):
    """
    Returns a concise, hypothetical explanation based on the Subcategory and direction.
    """
    is_positive = impact_val > 0 # Positive here means SUCCESS (Green)

    hypotheses = {
        # --- PATIENT & CRITERIA ---
        'Criteria Complexity (AI)': (
            "Complex/atypical inclusion criteria may limit enrollment.",
            "Standardized criteria likely facilitate easier recruitment."
        ),
        'Inclusion Constraints': (
            "Strict eligibility rules may reduce the patient pool.",
            "Broad eligibility likely expands the addressable patient population."
        ),
        'Patient Demographics': (
            "Target demographic (Age/Gender) may appear historically challenging.",
            "Target demographic aligns with historically higher success rates."
        ),
        'Patient Condition': (
            "Condition severity or volunteer status may add recruitment friction.",
            "Patient condition/volunteer status suggests easier recruitment."
        ),

        # --- THERAPEUTIC LANDSCAPE ---
        'Intervention Profile': (
            "Molecule class or regulatory status carries higher historical risk.",
            "Intervention type (e.g., Biologic/Vaccine) has strong historical precedence."
        ),
        'Disease Area': (
            "This therapeutic area historically has higher attrition rates.",
            "This therapeutic area historically has higher approval rates."
        ),
        'Competitive Intensity': (
            "High market competition may impact enrollment speed.",
            "Lower competition may allow for faster site activation."
        ),

        # --- PROTOCOL DESIGN ---
        'Scientific Rigor': (
            "Study design (Masking/Allocation) may lack robustness compared to peers.",
            "Robust design (Randomized/Double-Blind) signals high evidence quality."
        ),
        'Study Configuration': (
            "Phase or Model configuration carries higher statistical risk.",
            "Phase/Model configuration aligns with successful precedents."
        ),
        'Complexity & Safety': (
            "Complex endpoints or arm structure may increase operational risk.",
            "Streamlined endpoints/arms likely reduce operational complexity."
        ),

        # --- SPONSOR & OPERATIONS ---
        'Sponsor Capability': (
            "Sponsor track record or type may be less established.<br>Could be also prone to more frequent strategy changes.",
            "Sponsor experience likely mitigates operational risks."
        ),
        'Geography & Context': (
            "Location or timing factors may introduce regional/temporal risk.",
            "US involvement or recent start year correlates with higher data quality."
        )
    }

    default = ("Unusual pattern detected.", "Favorable pattern detected.")
    texts = hypotheses.get(subcategory, default)
    # Index 0 = Risk (Red/Negative), Index 1 = Success (Green/Positive)
    return texts[1] if is_positive else texts[0]

# --- 2. THE CORE CALCULATION ENGINE ---

def calculate_risk_drivers(model, explainer, taxonomy, row_data):
    """
    Runs SHAP, applies Linearization, and returns dataframes for plotting.
    """
    try:
        # 1. Transform Data
        preprocessor = model.named_steps['preprocessor']
        X_encoded = preprocessor.transform(row_data)

        # Get feature names from taxonomy to ensure alignment
        feature_names = taxonomy['feature_names']
        X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

        # 2. Calculate SHAP
        shap_values = explainer(X_encoded_df)
        base_log_odds = shap_values.base_values[0]

        # 3. Calculate Probabilities
        prob_fail_base = sigmoid(base_log_odds)
        prob_success_base = 1 - prob_fail_base

        # 4. Aggregate SHAP values
        impacts = []
        for i, col in enumerate(feature_names):
            val = shap_values.values[0][i]
            if abs(val) < 1e-9: continue # Skip zero impact

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

        # 6. Linearization (Sync Logic)
        # Calculate the gap between Base and Final probability
        total_gap = prob_success_final - prob_success_base

        if abs(total_shap_sum) < 1e-9:
            scale_factor = 0
        else:
            scale_factor = total_gap / total_shap_sum

        # Apply Scaling: Raw SHAP * Scale Factor = % Contribution
        # Note: scale_factor naturally handles the sign inversion (Risk -> Success)
        df_impacts['Impact_Pct'] = df_impacts['Raw_Log_Odds'] * scale_factor

        # Group by Pillar
        df_pillars = df_impacts.groupby('Pillar')['Impact_Pct'].sum().reset_index()

        return df_pillars, df_impacts, prob_success_final, None

    except Exception as e:
        return None, None, None, f"Calculation Error: {str(e)}"

# --- 3. PLOTTING FUNCTIONS ---

def plot_success_gauge(prob_success):
    """Micro-Gauge with Marker Line."""
    score_val = prob_success * 100

    # Gradient Steps
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
        number = {'suffix': "%", 'font': {'size': 20, 'color': 'black', 'family': 'Arial'}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "<b>Success Scoring</b>", 'font': {'size': 14, 'color': 'black'}},
        gauge = {
            'axis': {
                'range': [0, 100],
                'tickmode': 'array',
                'tickvals': [0, 25, 50, 75, 100],
                'ticktext': ['0', '25', '50', '75', '100'],
                'tickcolor': "#cccccc",
                'tickwidth': 1,
                'tickfont': {'size': 9, 'color': '#888'}
            },
            'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0}, # Hide Needle
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': steps,
            'threshold': {'line': {'color': "#333333", 'width': 3}, 'thickness': 1.0, 'value': score_val}
        }
    ))
    fig.update_layout(
        width=240, height=140,
        margin=dict(l=35, r=35, t=35, b=10),
        paper_bgcolor='white', font={'family': "Arial"}
    )
    return fig

def plot_impact_bar(df_pillars):
    """Compact Bar Chart."""
    df_sorted = df_pillars.sort_values('Impact_Pct', ascending=True)

    # Custom Scale
    custom_scale = [
        (0.00, "#8B0000"), (0.49, "#EF9A9A"), (0.50, "#FFFFFF"), (0.51, "#A5D6A7"), (1.00, "#006400")
    ]

    max_abs = max(abs(df_sorted['Impact_Pct'].min()), abs(df_sorted['Impact_Pct'].max()))

    fig = px.bar(
        df_sorted,
        x='Impact_Pct',
        y='Pillar',
        color='Impact_Pct',
        orientation='h',
        color_continuous_scale=custom_scale,
        range_color=[-max_abs, max_abs]
    )

    fig.update_traces(
        texttemplate='%{x:+.1%}', textposition='outside',
        textfont_color='black', textfont_weight='bold', textfont_size=12,
        cliponaxis=False, width=0.8
    )

    # Calculate Range for Zoom
    x_min = df_sorted['Impact_Pct'].min()
    x_max = df_sorted['Impact_Pct'].max()
    if x_min == 0 and x_max == 0: x_range = [-0.1, 0.1]
    else: x_range = [x_min * 1.35, x_max * 1.35]

    fig.update_layout(
        title=None, xaxis_title="", yaxis_title="", showlegend=False, coloraxis_showscale=False,
        width=400, height=180,
        margin=dict(l=150, r=45, t=10, b=10),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(size=11, color='black')
    )
    fig.add_vline(x=0, line_width=1.5, line_color="#333333", opacity=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=x_range)
    fig.update_yaxes(tickfont=dict(size=11, color='black'), ticksuffix="      ", automargin=True)

    return fig

def plot_treemap(df_impacts, df_pillars):
    """Refined Treemap with Hypotheses."""
    # Group by Subcategory first
    df_sub = df_impacts.groupby(['Pillar', 'Subcategory'])['Impact_Pct'].sum().reset_index()

    # Create Labels
    pillar_totals = df_pillars.set_index('Pillar')['Impact_Pct'].to_dict()
    df_sub['Pillar_Label'] = df_sub['Pillar'].apply(
        lambda x: f"<b>{x.upper()}</b> ({pillar_totals.get(x, 0):+.1%})"
    )

    # Generate Hypotheses
    df_sub['Hypothesis'] = df_sub.apply(
        lambda x: get_business_hypothesis(x['Subcategory'], x['Impact_Pct']),
        axis=1
    )

    # Add Size & Filter
    df_sub['Importance'] = df_sub['Impact_Pct'].abs()
    df_sub = df_sub[df_sub['Importance'] > 0.0005]

    # Color Bounds
    max_abs = max(abs(df_sub['Impact_Pct'].min()), abs(df_sub['Impact_Pct'].max()))
    custom_scale = [
        (0.00, "#8B0000"), (0.45, "#E57373"), (0.50, "#F5F5F5"), (0.55, "#81C784"), (1.00, "#006400")
    ]

    fig = px.treemap(
        df_sub,
        path=[px.Constant("<b>ALL DRIVERS</b>"), 'Pillar_Label', 'Subcategory'],
        values='Importance',
        color='Impact_Pct',
        color_continuous_scale=custom_scale,
        range_color=[-max_abs, max_abs],
        custom_data=['Impact_Pct', 'Hypothesis']
    )

    fig.update_traces(
        textinfo="label+value",
        texttemplate=(
            "<span style='font-size:16px; font-weight:bold;'>%{label}</span><br>"
            "<span style='font-size:14px;'>%{customdata[0]:+.1%}</span><br><br>"
            "<span style='font-size:12px; font-style:italic;'>%{customdata[1]}</span>"
        ),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Impact: <b>%{customdata[0]:+.2%}</b><br><br>"
            "<i>Analysis:</i><br>%{customdata[1]}<extra></extra>"
        ),
        marker=dict(line=dict(width=1, color='white'), pad=dict(t=60, l=10, r=10, b=10)),
        pathbar=dict(visible=True, thickness=25, textfont=dict(size=14, family="Arial")),
        textfont=dict(size=14, family="Arial")
    )

    fig.update_layout(
        title=None,
        margin=dict(t=10, l=10, r=10, b=10),
        coloraxis_showscale=False,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig
