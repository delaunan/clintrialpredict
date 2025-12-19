import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def get_business_hypothesis(subcategory, impact_val):
    """Generates tooltip text for the Treemap based on business logic."""
    is_positive = impact_val > 0

    hypotheses = {
        'Criteria Complexity': ("Complex/atypical inclusion criteria may limit enrollment.", "Standardized criteria likely facilitate easier recruitment."),
        'Inclusion Constraints': ("Strict eligibility rules may reduce the patient pool.", "Broad eligibility likely expands the addressable patient population."),
        'Patient Demographics': ("Target demographic (Age/Gender) may appear historically challenging.", "Target demographic aligns with historically higher success rates."),
        'Patient Condition': ("Condition severity or volunteer status may add recruitment friction.", "Patient condition/volunteer status suggests easier recruitment."),
        'Intervention Profile': ("Molecule class or regulatory status carries higher historical risk.", "Intervention type (e.g., Biologic/Vaccine) has strong historical precedence."),
        'Disease Area': ("This therapeutic area historically has higher attrition rates.", "This therapeutic area historically has higher approval rates."),
        'Competitive Intensity': ("High market competition may impact enrollment speed.", "Lower competition may allow for faster site activation."),
        'Scientific Rigor': ("Study design (Masking/Allocation) may lack robustness compared to peers.", "Robust design (Randomized/Double-Blind) signals high evidence quality."),
        'Study Configuration': ("Phase or Model configuration carries higher statistical risk.", "Phase/Model configuration aligns with successful precedents."),
        'Complexity & Safety': ("Complex endpoints or arm structure may increase operational risk.", "Streamlined endpoints/arms likely reduce operational complexity."),
        'Sponsor Capability': ("Sponsor track record or type may be less established.", "Sponsor experience likely mitigates operational risks."),
        'Geography & Context': ("Location or timing factors may introduce regional/temporal risk.", "US involvement or recent start year correlates with higher data quality.")
    }

    default = ("Unusual pattern detected.", "Favorable pattern detected.")
    texts = hypotheses.get(subcategory, default)
    return texts[1] if is_positive else texts[0]

def plot_success_gauge(prob_success):
    """Draws the gauge chart for success probability."""
    score_val = prob_success * 100
    steps = []

    # Red -> White -> Green Gradient
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
        mode="gauge+number", value=score_val,
        number={"suffix": "%", "font": {"size": 34, "color": "black", "family": "Arial"}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickmode": "array", "tickvals": [0, 25, 50, 75, 100], "tickcolor": "#cccccc"},
            "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
            "bgcolor": "white", "borderwidth": 0, "steps": steps,
            "threshold": {"line": {"color": "#333333", "width": 3}, "thickness": 1.0, "value": score_val},
        },
    ))
    fig.update_layout(margin=dict(l=15, r=15, t=10, b=10), paper_bgcolor="white", font={"family": "Arial"}, height=220)
    return fig

def plot_impact_bar(df_pillars):
    """Draws the bar chart for pillar impacts."""
    df_sorted = df_pillars.dropna(subset=["Impact_Pct"]).sort_values("Impact_Pct", ascending=True)
    if df_sorted.empty: return go.Figure()

    custom_scale = [(0.00, "#8B0000"), (0.49, "#EF9A9A"), (0.50, "#FFFFFF"), (0.51, "#A5D6A7"), (1.00, "#006400")]
    max_abs = max(abs(df_sorted["Impact_Pct"].min()), abs(df_sorted["Impact_Pct"].max()))

    fig = px.bar(
        df_sorted, x="Impact_Pct", y="Pillar", color="Impact_Pct", orientation="h",
        color_continuous_scale=custom_scale, range_color=[-max_abs, max_abs],
    )
    fig.update_traces(
        # --- DISABLE HOVER ---
        hoverinfo="none",      # This turns off the hover box
        hovertemplate=None,    # This ensures no template overrides it
        # ---------------------
        texttemplate="%{x:+.1%}", textposition="outside", textfont_color="black", textfont_size=16, cliponaxis=False)
    fig.update_layout(
        title_text="", xaxis_title="", yaxis_title="", showlegend=False, coloraxis_showscale=False,
        margin=dict(l=180, r=60, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=16, color="black"), height=220,
    )
    fig.add_vline(x=0, line_width=1.0, line_color="#333333", opacity=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(tickfont=dict(size=16, color="black"), ticksuffix="         ", automargin=True)
    return fig



def plot_treemap(df_impacts, df_pillars):
    """
    Draws a Treemap with HIGH CONTRAST colors and HIDDEN TOOLTIPS.
    Reverted to standard line width.
    """
    # 1. Prepare Data
    df_sub = df_impacts.groupby(['Pillar', 'Subcategory'])['Impact_Pct'].sum().reset_index()
    pillar_totals = df_pillars.set_index('Pillar')['Impact_Pct'].to_dict()

    df_sub['Pillar_Label'] = df_sub['Pillar'].apply(lambda x: f"<b>{x.upper()}</b> ({pillar_totals.get(x, 0):+.1%})")
    df_sub['Hypothesis'] = df_sub.apply(lambda x: get_business_hypothesis(x['Subcategory'], x['Impact_Pct']), axis=1)
    df_sub['Importance'] = df_sub['Impact_Pct'].abs()

    # Filter noise
    df_sub = df_sub[df_sub['Importance'] > 0.0005]
    if df_sub.empty: return go.Figure()

    # --- COLOR STRATEGY ---
    true_net_impact = df_sub['Impact_Pct'].sum()
    saturation_limit = 0.08
    max_abs = saturation_limit
    custom_scale = [(0.00, "#B71C1C"), (0.50, "#FFFFFF"), (1.00, "#1B5E20")]

    fig = px.treemap(
        df_sub,
        path=[px.Constant("<b>ALL DRIVERS</b>"), 'Pillar_Label', 'Subcategory'],
        values='Importance',
        color='Impact_Pct',
        color_continuous_scale=custom_scale,
        range_color=[-max_abs, max_abs],
        custom_data=['Impact_Pct', 'Hypothesis']
    )

    # --- PATCH COLORS ONLY ---
    try:
        trace = fig.data[0]
        new_colors = list(trace.marker.colors) if trace.marker.colors else [None] * len(trace.labels)

        for i, label in enumerate(trace.labels):
            val_to_map = None

            # Identify Value
            if label == "<b>ALL DRIVERS</b>":
                val_to_map = true_net_impact
            elif any(p in label for p in pillar_totals.keys()):
                for p_name, p_val in pillar_totals.items():
                    if f"<b>{p_name.upper()}</b>" in label:
                        val_to_map = p_val
                        break

            # Apply Color
            if val_to_map is not None:
                val_clamped = max(-max_abs, min(max_abs, val_to_map))
                norm = (val_clamped + max_abs) / (2 * max_abs)

                if norm < 0.5: # Red
                    r_f = (norm * 2)
                    r = 183 + (255 - 183) * r_f
                    g = 28 + (255 - 28) * r_f
                    b = 28 + (255 - 28) * r_f
                else: # Green
                    g_f = (norm - 0.5) * 2
                    r = 255 + (27 - 255) * g_f
                    g = 255 + (94 - 255) * g_f
                    b = 255 + (32 - 255) * g_f

                new_colors[i] = f"rgb({int(r)},{int(g)},{int(b)})"

        trace.marker.colors = tuple(new_colors)

    except Exception as e:
        print(f"Viz Warning: Color patching failed: {e}")

    fig.update_traces(
        # --- HIDE TOOLTIPS ---
        hoverinfo="none",
        hovertemplate=None,
        # ---------------------

        textinfo="label+value",
        texttemplate="<span style='font-size:16px; font-weight:bold;'>%{label}</span><br><span style='font-size:14px;'>%{customdata[0]:+.1%}</span><br><br><span style='font-size:12px; font-style:italic;'>%{customdata[1]}</span>",

        # Standard thin white lines
        marker=dict(line=dict(width=1, color='white'), pad=dict(t=60, l=10, r=10, b=10)),

        pathbar=dict(visible=False), textfont=dict(size=14, family="Arial")
    )

    fig.update_layout(
        margin=dict(t=0, l=10, r=10, b=10),
        coloraxis_showscale=False, paper_bgcolor='white', plot_bgcolor='white', height=520
    )

    return fig
