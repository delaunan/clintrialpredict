import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Clinical Trial Completion Predictor",
    page_icon="🧪",
    layout="wide",
)

# ============================================================
# 1. Locate the Project Root
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# 2. Add "src" to Python Path (if needed)
# ============================================================
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# ============================================================
# 3. Define All Important Paths
# ============================================================

MODEL_PATH      = PROJECT_ROOT / "models" / "ctp_model.joblib"
EXPLAINER_PATH  = PROJECT_ROOT / "models" / "shap_explainer.joblib"        # <- change
TAXONOMY_PATH   = PROJECT_ROOT / "models" / "feature_taxonomy.joblib"      # <- change

DATA_PATH = PROJECT_ROOT / "data" / "data_predict.csv"
HISTORICAL_PATH = PROJECT_ROOT / "data" / "project_data.csv"

PHASE_COL = "phase"
TA_COL = "therapeutic_area"
OUTCOME_COL = "completed"
ID_COL = "nct_id"

# Risk thresholds (based on p_fail)
LOW_RISK_MAX = 0.25      # p_fail <= 25%  -> Low risk
MEDIUM_RISK_MAX = 0.50   # 25–50%        -> Medium risk
# p_fail > 50%           -> High risk

# --------------------------------------------------
# GLOBAL STYLES
# --------------------------------------------------
st.markdown(
    """
    <style>
        .main > div {
            max-width: 1200px;
            margin: 0 auto;
        }

        .stButton > button {
            background: #3a3a3a;
            color: white;
            padding: 0.55rem 1.2rem;
            border-radius: 8px;
            font-size: 0.9rem;
            border: none;
            font-weight: 500;
        }
        .stButton > button:hover {
            background: #2f2f2f;
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.2);
        }

        .card {
            padding: 1.5rem 1.75rem;
            border-radius: 1rem;
            background: #FFFFFF;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
            border: 1px solid #E5E7EB;
        }

        h1 {
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
        }

        /* Shrink expanders to half width */
        div.streamlit-expander {
            max-width: 50% !important;
            margin-left: 0 !important;
            margin-right: auto !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# LOADING DATA / MODELS
# ==========================

@st.cache_data
def load_historical_data(path: Path = HISTORICAL_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # failed = 1  -> did NOT complete
    # failed = 0  -> completed
    df[OUTCOME_COL] = 1 - df["scientific_success"]
    return df

@st.cache_data
def load_predict_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_explainer():
    return joblib.load(EXPLAINER_PATH)

@st.cache_resource
def load_taxonomy():
    return joblib.load(TAXONOMY_PATH)

historical_df = load_historical_data()
X = load_predict_data()
model = load_model()
explainer = load_explainer()
taxonomy = load_taxonomy()

# ==========================
# HELPER FUNCTIONS (YOUR ORIGINAL)
# ==========================

def get_risk_tier(p_fail: float):
    """Return (tier, description) based on failure probability."""
    if p_fail <= LOW_RISK_MAX:
        return (
            "Low",
            "Model sees few patterns associated with withdrawn/suspended trials."
        )
    elif p_fail <= MEDIUM_RISK_MAX:
        return (
            "Medium",
            "Mixed signals; trial resembles both successful and failed studies."
        )
    else:
        return (
            "High",
            "Trial shares strong characteristics with previously halted trials."
        )

def compute_benchmarks(
    historical_df: pd.DataFrame,
    current_row: pd.Series,
    p_comp: float,
) -> dict:
    """Compare predicted completion probability to historical data."""
    overall_rate = historical_df[OUTCOME_COL].mean()

    mask_similar = (
        (historical_df[PHASE_COL] == current_row[PHASE_COL]) &
        (historical_df[TA_COL] == current_row[TA_COL])
    )
    similar_df = historical_df[mask_similar]

    if not similar_df.empty:
        similar_rate = similar_df[OUTCOME_COL].mean()
        n_similar = len(similar_df)
    else:
        similar_rate = np.nan
        n_similar = 0

    return {
        "overall_rate": overall_rate,
        "similar_rate": similar_rate,
        "n_similar": n_similar,
    }

def build_summary(row: pd.Series, p_comp: float, tier: str, bench: dict) -> str:
    phase_val = str(row.get(PHASE_COL, "unknown phase"))
    ta_val = str(row.get(TA_COL, "this therapeutic area"))

    base = (
        f"This Phase {phase_val} trial in {ta_val} has an estimated "
        f"completion probability of {p_comp:.1%}"
    )

    if bench is not None and not np.isnan(bench.get("similar_rate", np.nan)):
        diff = p_comp - bench["similar_rate"]
        direction = "higher" if diff >= 0 else "lower"
        base += (
            f", which is {abs(diff):.1%} {direction} than the historical "
            f"completion rate for {bench['n_similar']} similar trials"
        )

    if tier:
        base += f" and is classified as {tier.lower()} risk."

    return base

# ==========================
# COLLEAGUE'S HELPERS (SHAP + PLOTS)
# ==========================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def map_feature_to_business_pillar(feature_name, taxonomy_dict):
    """Maps a raw feature to a pillar using the saved dictionary."""
    # Exact match
    if feature_name in taxonomy_dict['pillar_map']:
        return taxonomy_dict['pillar_map'][feature_name], taxonomy_dict['subcat_map'][feature_name]

    # Fallback logic for unseen features
    name = str(feature_name).lower()
    if 'emb_' in name or 'pca' in name:
        return 'Patient & Criteria', 'Criteria Complexity (AI)'
    if 'sponsor' in name:
        return 'Sponsor & Operations', 'Sponsor Capability'
    return 'Other', 'Unclassified'

def get_business_hypothesis(subcategory, impact_val):
    """Returns a concise, hypothetical explanation based on subcategory and direction."""
    is_positive = impact_val > 0  # Positive here means SUCCESS (Green)

    hypotheses = {
        # --- PATIENT & CRITERIA ---
        'Criteria Complexity': (
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
    return texts[1] if is_positive else texts[0]

def calculate_risk_drivers(model, explainer, taxonomy, row_data):
    """
    Runs SHAP, applies Linearization, and returns dataframes for plotting.
    """
    try:
        # 1. Transform Data
        preprocessor = model.named_steps['preprocessor']
        X_encoded = preprocessor.transform(row_data)

        feature_names = taxonomy['feature_names']
        X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

        # 2. Calculate SHAP
        shap_values = explainer(X_encoded_df)
        base_log_odds = shap_values.base_values[0]

        # 3. Base probabilities
        prob_fail_base = sigmoid(base_log_odds)
        prob_success_base = 1 - prob_fail_base

        # 4. Aggregate SHAP values
        impacts = []
        for i, col in enumerate(feature_names):
            val = shap_values.values[0][i]
            if abs(val) < 1e-9:
                continue

            pillar, subcat = map_feature_to_business_pillar(col, taxonomy)
            if pillar == 'Other':
                continue

            impacts.append({
                'Pillar': pillar,
                'Subcategory': subcat,
                'Feature': col,
                'Raw_Log_Odds': val
            })

        df_impacts = pd.DataFrame(impacts)

        # 5. Final probability
        total_shap_sum = df_impacts['Raw_Log_Odds'].sum()
        final_log_odds = base_log_odds + total_shap_sum
        prob_fail_final = sigmoid(final_log_odds)
        prob_success_final = 1 - prob_fail_final

        # 6. Linearization to % contributions
        total_gap = prob_success_final - prob_success_base

        if abs(total_shap_sum) < 1e-9:
            scale_factor = 0
        else:
            scale_factor = total_gap / total_shap_sum

        df_impacts['Impact_Pct'] = df_impacts['Raw_Log_Odds'] * scale_factor

        # Group by Pillar
        df_pillars = df_impacts.groupby('Pillar')['Impact_Pct'].sum().reset_index()

        return df_pillars, df_impacts, prob_success_final, None

    except Exception as e:
        return None, None, None, f"Calculation Error: {str(e)}"

def plot_success_gauge(prob_success):
    """Micro-Gauge with Marker Line."""
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
        mode="gauge+number",
        value=score_val,
        number={'suffix': "%", 'font': {'size': 20, 'color': 'black', 'family': 'Arial'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>Success Scoring</b>", 'font': {'size': 14, 'color': 'black'}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickmode': 'array',
                'tickvals': [0, 25, 50, 75, 100],
                'ticktext': ['0', '25', '50', '75', '100'],
                'tickcolor': "#cccccc",
                'tickwidth': 1,
                'tickfont': {'size': 9, 'color': '#888'}
            },
            'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': steps,
            'threshold': {'line': {'color': "#333333", 'width': 3},
                          'thickness': 1.0, 'value': score_val}
        }
    ))
    fig.update_layout(
        width=240, height=140,
        margin=dict(l=35, r=35, t=35, b=10),
        paper_bgcolor='white', font={'family': "Arial"}
    )
    return fig

def plot_impact_bar(df_pillars):
    """Compact Bar Chart of pillar impacts."""
    df_sorted = df_pillars.sort_values('Impact_Pct', ascending=True)

    custom_scale = [
        (0.00, "#8B0000"), (0.49, "#EF9A9A"),
        (0.50, "#FFFFFF"),
        (0.51, "#A5D6A7"), (1.00, "#006400")
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
        textfont_color='black', textfont_size=12,
        cliponaxis=False, width=0.8
    )

    x_min = df_sorted['Impact_Pct'].min()
    x_max = df_sorted['Impact_Pct'].max()
    x_range = [x_min * 1.35, x_max * 1.35] if not (x_min == 0 and x_max == 0) else [-0.1, 0.1]

    fig.update_layout(
        title=None, xaxis_title="", yaxis_title="", showlegend=False,
        coloraxis_showscale=False,
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
    df_sub = df_impacts.groupby(['Pillar', 'Subcategory'])['Impact_Pct'].sum().reset_index()

    pillar_totals = df_pillars.set_index('Pillar')['Impact_Pct'].to_dict()
    df_sub['Pillar_Label'] = df_sub['Pillar'].apply(
        lambda x: f"<b>{x.upper()}</b> ({pillar_totals.get(x, 0):+.1%})"
    )

    df_sub['Hypothesis'] = df_sub.apply(
        lambda x: get_business_hypothesis(x['Subcategory'], x['Impact_Pct']),
        axis=1
    )

    df_sub['Importance'] = df_sub['Impact_Pct'].abs()
    df_sub = df_sub[df_sub['Importance'] > 0.0005]

    max_abs = max(abs(df_sub['Impact_Pct'].min()), abs(df_sub['Impact_Pct'].max()))
    custom_scale = [
        (0.00, "#8B0000"), (0.45, "#E57373"),
        (0.50, "#F5F5F5"),
        (0.55, "#81C784"), (1.00, "#006400")
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
        marker=dict(line=dict(width=1, color='white'),
                    pad=dict(t=60, l=10, r=10, b=10)),
        pathbar=dict(visible=True, thickness=25,
                     textfont=dict(size=14, family="Arial")),
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

# ==========================
# UI: TITLE & TRIAL SELECTION
# ==========================

st.markdown("### 🧪 Clinical Trial Completion Predictor")
st.write("Select a clinical trial to generate a prediction on its completion and see SHAP-based explanations.")

# Build nice readable label: "NCTID — Brief Title"
X["short_label"] = X[ID_COL].astype(str) + " — " + X["brief_title"].astype(str)
label_to_nct = dict(zip(X["short_label"], X[ID_COL]))

selected_label = st.selectbox(
    "Select a trial",
    X["short_label"].tolist()
)

trial_id = label_to_nct[selected_label]
selected_trial = X[X[ID_COL] == trial_id].iloc[[0]]
row = selected_trial.iloc[0]

# ==========================
# PREDICTION + SHAP ON BUTTON CLICK
# ==========================

if st.button("Make prediction"):
    # --- SHAP-based driver calculation ---
    df_pillars, df_impacts, prob_success_final, error_msg = calculate_risk_drivers(
        model=model,
        explainer=explainer,
        taxonomy=taxonomy,
        row_data=selected_trial
    )

    if error_msg is not None or df_pillars is None:
        st.error(f"SHAP explanation failed: {error_msg}")
        # Fallback to plain model probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(selected_trial)[0]
            idx_fail = list(model.classes_).index(1)
            p_fail = float(proba[idx_fail])
        else:
            pred = int(model.predict(selected_trial)[0])
            p_fail = 1.0 if pred == 1 else 0.0
        p_comp = 1.0 - p_fail
    else:
        # Use SHAP-derived probability as main success probability
        p_comp = float(prob_success_final)
        p_fail = 1.0 - p_comp

    # ---- OUTCOME SUMMARY ----
    st.subheader("1. Outcome Summary")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Completion Probability")
        st.write("Predicted probability that the trial will be **completed**:")
        st.write(f"**{p_comp:.1%}**")
        st.progress(int(p_comp * 100))

        # Risk level
        st.markdown("#### Risk Level (Low / Medium / High)")
        tier, desc = get_risk_tier(p_fail)
        if tier == "Low":
            box = st.success
        elif tier == "Medium":
            box = st.warning
        else:
            box = st.error
        box(f"**{tier} risk** – {desc}")

    # ---- SHAP GAUGE + PILLAR BAR ----
    with col_right:
        if df_pillars is not None:
            st.markdown("#### SHAP Success Gauge")
            gauge_fig = plot_success_gauge(p_comp)
            st.plotly_chart(gauge_fig, use_container_width=False)

            st.markdown("#### Pillar Impact Overview")
            bar_fig = plot_impact_bar(df_pillars)
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("SHAP-based visual explanations not available for this run.")

    # =========================
    # BENCHMARKS & COMPARISON
    # =========================
    bench = compute_benchmarks(historical_df, row, p_comp)

    st.subheader("2. How does this trial compare?")

    c1, c2 = st.columns(2)
    c1.metric(
        "Overall completion rate (historical)",
        f"{bench['overall_rate']:.1%}"
    )

    if not np.isnan(bench["similar_rate"]):
        label = f"Similar trials (Phase {row[PHASE_COL]}, {row[TA_COL]})"
        c2.metric(
            label,
            f"{bench['similar_rate']:.1%}",
            help=f"Based on n = {bench['n_similar']} historical trials."
        )
    else:
        c2.info("No similar historical trials found for this phase/therapeutic area.")

    summary_text = build_summary(row, p_comp, get_risk_tier(p_fail)[0], bench)
    st.markdown("#### Summary")
    st.markdown(summary_text)

    # =========================
    # TREEMAP OF DRIVERS
    # =========================
    if df_pillars is not None and df_impacts is not None:
        st.subheader("3. Key Drivers of Predicted Outcome")
        st.write(
            "Each block below represents a group of model features. "
            "Green blocks contribute positively to completion probability; "
            "red blocks reduce it."
        )
        treemap_fig = plot_treemap(df_impacts, df_pillars)
        st.plotly_chart(treemap_fig, use_container_width=True)

# -----------------------------------
# TRIAL IDENTITY CARD
# -----------------------------------
st.subheader("Selected Trial Overview")

st.markdown(f"""
**{row['official_title']}**

- **NCT ID:** {row['nct_id']}
- **Phase:** {row['phase']}
- **Therapeutic Area:** {row['therapeutic_area']}
- **Reason for stopping:** {row['why_stopped']}
""")

col1, col2 = st.columns([1, 1])

with col1:
    with st.expander("Protocol Design", expanded=False):
        st.write(f"**Primary Purpose:** {row['primary_purpose']}")
        st.write(f"**Intervention Model:** {row['intervention_model']}")
        st.write(f"**Allocation:** {row['allocation']}")
        st.write(f"**Masking:** {row['masking']}")
        st.write(f"**Number of Arms:** {row['number_of_arms']}")
        st.write(f"**# Primary Endpoints:** {row['num_primary_endpoints']}")
        st.write(f"**Data Monitoring Committee (DMC):** {row['has_dmc']}")

    with st.expander("Eligibility & Population", expanded=False):
        st.write(f"**Eligibility Strictness Score:** {row['eligibility_strictness_score']}")
        st.write(f"**Criteria Length (log):** {row['criteria_len_log']}")
        st.write(f"**Gender Restriction:** {row['gender']} (restricted: {row['is_gender_restricted']})")
        st.write(f"**Healthy Volunteers:** {row['healthy_volunteers']}")
        st.write(
            "**Population Flags:** "
            f"Child={row['child']}, Adult={row['adult']}, Older Adult={row['older_adult']}"
        )
        with st.expander("Full Criteria Text", expanded=False):
            st.write(row["txt_criteria"])

    with st.expander("Sponsor & Operational Factors", expanded=False):
        st.write(f"**Lead Sponsor:** {row['lead_sponsor']}")
        st.write(f"**Sponsor Tier:** {row['sponsor_tier']}")
        st.write(f"**Sponsor Class:** {row['agency_class']}")
        st.write(f"**Includes U.S. Sites:** {row['includes_us']}")
        st.write(f"**Study Start Year:** {row['start_year']}")
        st.write(f"**COVID Exposure:** {row['covid_exposure']}")
