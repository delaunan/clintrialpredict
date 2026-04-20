import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re
import textwrap

# ==========================
# 0. STYLE CONFIG (NEUTRAL PRO)
# ==========================
STYLE_CONFIG = {
    "font_family": "Helvetica, Arial, sans-serif",
    "font_color": "#1f2a38",
    "colors": {
        "red_deep":   (168, 50, 50),
        "red_soft":   (240, 163, 163),
        "grey_warm":  (242, 244, 248),
        "blue_soft":  (154, 203, 232),
        "blue_deep":  (28, 86, 153),
        "therapeutic_grey": "#CFD8DC",
        # Pastel Zones
        "pastel_red": "#fde8e8",
        "pastel_orange": "#fff7ed",
        "pastel_blue": "#eff6ff",
        "pastel_green": "#f0fdf4"
    }
}

def get_rgb_str(rgb_tuple):
    return f"rgb({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]})"

def interpolate_color(c1, c2, ratio):
    r = int(c1[0] + (c2[0] - c1[0]) * ratio)
    g = int(c1[1] + (c2[1] - c1[1]) * ratio)
    b = int(c1[2] + (c2[2] - c1[2]) * ratio)
    return (r, g, b)

def mix_white(rgb, factor=0.7):
    r, g, b = rgb
    new_r = int(r + (255 - r) * factor)
    new_g = int(g + (255 - g) * factor)
    new_b = int(b + (255 - b) * factor)
    return (new_r, new_g, new_b)

def fmt_header(text, value, txt_color="black", split=False):
    label = f"{text}<br>({value:+.1f} pts)" if split else f"{text} ({value:+.1f} pts)"
    return f"<span style='font-size:14px; font-weight:bold; color:{txt_color}'>{label}</span>"

# ==========================
# 1. GAUGE CHART
# ==========================
def plot_success_gauge(score_val):
    steps = []
    c = STYLE_CONFIG["colors"]
    for i in range(100):
        if i < 25: color = interpolate_color(c["red_deep"], c["red_soft"], i / 25.0)
        elif i < 50: color = interpolate_color(c["red_soft"], c["grey_warm"], (i - 25) / 25.0)
        elif i < 75: color = interpolate_color(c["grey_warm"], c["blue_soft"], (i - 50) / 25.0)
        else: color = interpolate_color(c["blue_soft"], c["blue_deep"], (i - 75) / 25.0)
        steps.append({'range': [i, i + 1], 'color': get_rgb_str(color)})
    for sep in [25, 50, 75]: steps.append({'range': [sep - 0.5, sep + 0.5], 'color': 'white'})

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score_val,
        number={"valueformat": ".1f", "font": {"size": 52, "color": STYLE_CONFIG["font_color"], "family": "Helvetica, Arial, sans-serif", "weight": "bold"}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickmode": "array", "tickvals": [0, 25, 50, 75, 100], "tickfont": {"size": 14, "color": "#555555", "family": "Helvetica, Arial, sans-serif", "weight": "bold"}},
            "bar": {"color": "rgba(0,0,0,0)"}, "bgcolor": "white", "borderwidth": 0, "steps": steps,
            "threshold": {"line": {"color": STYLE_CONFIG["font_color"], "width": 5}, "thickness": 0.75, "value": score_val},
        },
    ))
    fig.update_layout(margin=dict(l=35, r=35, t=40, b=10), height=220, paper_bgcolor="white", hovermode=False)
    return fig

# ==========================
# 2. IMPACT BAR CHART
# ==========================
def plot_impact_bar(df_pillars):
    df_plot = df_pillars.copy()
    
    df_plot['Pillar_Clean'] = df_plot['Pillar'].apply(lambda x: re.sub(r'^\d+\.\s*', '', x))
    df_plot = df_plot.sort_values(by='Impact', ascending=True)

    max_val = df_plot['Impact'].abs().max()
    limit, BAR_WIDTH, SLICE_STEP, GLOBAL_MAX_SCALE = max(max_val * 1.35, 6.0), 0.75, 0.08, 4.0
    c = STYLE_CONFIG["colors"]
    fig = go.Figure()

    n_slices = int(np.ceil(max_val / SLICE_STEP)) if max_val > 0 else 1
    
    for i in range(n_slices):
        pos = i * SLICE_STEP
        widths, colors, texts = [], [], []
        for _, row in df_plot.iterrows():
            val = row['Impact']
            abs_v = abs(val)
            if pos >= abs_v: w, is_tip = 0, False
            elif pos + SLICE_STEP > abs_v: w, is_tip = abs_v - pos, True
            else: w, is_tip = SLICE_STEP, False
            
            widths.append(-w if val < 0 else w)
            texts.append(f"{val:+.1f} pts" if is_tip else None)
            
            ratio = min((pos + abs(w)/2) / GLOBAL_MAX_SCALE, 1.0) ** 0.5
            if val >= 0:
                final_rgb = interpolate_color(c["grey_warm"], c["blue_soft"], ratio*2) if ratio < 0.5 else interpolate_color(c["blue_soft"], c["blue_deep"], (ratio-0.5)*2)
                colors.append(get_rgb_str(final_rgb))
            else:
                final_rgb = interpolate_color(c["grey_warm"], c["red_soft"], ratio*2) if ratio < 0.5 else interpolate_color(c["red_soft"], c["red_deep"], (ratio-0.5)*2)
                colors.append(get_rgb_str(final_rgb))
        
        fig.add_trace(go.Bar(
            y=df_plot['Pillar_Clean'],
            x=widths,
            orientation='h',
            width=BAR_WIDTH,
            marker=dict(color=colors, line=dict(width=0)),
            text=texts,
            textposition='outside',
            textfont=dict(size=12, color="black"),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.add_vline(x=0, line_width=1, line_color="#333333")
    fig.update_layout(
        barmode='relative',
        xaxis=dict(showticklabels=False, range=[-limit, limit], zeroline=False, showgrid=False),
        yaxis=dict(automargin=True, tickfont=dict(size=12, color="#1f2a38")),
        margin=dict(l=10, r=10, t=10, b=10),
        height=240,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False
    )
    return fig

# ==========================
# 3. TREEMAP
# ==========================
def plot_treemap(subcat_impacts, pillar_impacts, show_values=True):
    nodes = {}
    node_sums = {"ALL_DRIVERS": sum(p['Impact'] for p in pillar_impacts)}
    c = STYLE_CONFIG["colors"]
    leaf_data = []

    nodes["ALL_DRIVERS"] = {"parent": "", "label": fmt_header("ALL DRIVERS", node_sums["ALL_DRIVERS"]), "color": "#FFFFFF", "value": 0}

    for p_item in pillar_impacts:
        p_name, p_imp = p_item['Pillar'], p_item['Impact']
        p_clean = re.sub(r'^\d+\.\s*', '', p_name)
        p_id = f"PILLAR_{p_clean}"
        
        ratio = min(abs(p_imp) / 8.0, 1.0)
        base_color = interpolate_color(c["blue_soft" if p_imp >= 0 else "red_soft"], c["blue_deep" if p_imp >= 0 else "red_deep"], ratio)
        bg_color = get_rgb_str(mix_white(base_color))
        
        nodes[p_id] = {
            "parent": "ALL_DRIVERS",
            "label": fmt_header(p_clean.upper(), p_imp),
            "color": bg_color,
            "value": 0
        }

    for item in subcat_impacts:
        p_raw = item['Pillar']
        subtopic = item['Subcategory']
        impact = item['Impact']
        feat_details = item.get('FeatureDetails', [])

        if not show_values:
            # Strip the value part (": <b>...</b>") using regex or splitting
            feat_details = [re.sub(r":\s*<b>.*</b>", "", f) for f in feat_details]
        
        p_clean = re.sub(r'^\d+\.\s*', '', p_raw)
        parent_id = f"PILLAR_{p_clean}"
        
        ratio = max(0.15, min(abs(impact) / 8.0, 1.0))
        color = interpolate_color(c["blue_soft"], c["blue_deep"], ratio) if impact >= 0 else interpolate_color(c["red_soft"], c["red_deep"], ratio)

        feat_html = "• " + "<br>• ".join(feat_details) if feat_details else ""
        
        label_html = (
            f"<span style='font-size:15px; font-weight:bold; color:white'>{subtopic}</span>"
            f"<br><span style='font-size:14px; color:white'>{impact:+.1f} pts</span>"
            f"<br><br><span style='font-size:11px; color:white'>{feat_html}</span>"
        )
        
        leaf_id = f"{parent_id}_{subtopic}"
        leaf_data.append({
            "id": leaf_id,
            "parent": parent_id,
            "label": label_html,
            "color": get_rgb_str(color),
            "value": max(0.5, abs(impact))
        })

    for item in leaf_data:
        nodes[item['id']] = {"parent": item['parent'], "label": item['label'], "color": item['color'], "value": item['value']}

    ids = list(nodes.keys())
    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=[nodes[k]['label'] for k in ids],
        parents=[nodes[k]['parent'] for k in ids],
        values=[nodes[k]['value'] for k in ids],
        marker=dict(colors=[nodes[k]['color'] for k in ids], line=dict(width=1, color='white')),
        textinfo="label",
        textposition="top left",
        tiling=dict(packing="squarify"),
        hoverinfo="skip"
    ))
    
    fig.update_layout(
        margin=dict(t=34, l=15, r=15, b=15),
        height=600,
        font=dict(family="Helvetica, Arial, sans-serif"),
        paper_bgcolor='white',
        hovermode=False
    )
    return fig
