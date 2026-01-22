import plotly.graph_objects as go
import pandas as pd
import numpy as np
import textwrap

# ==============================================================================
# 0. UNIFIED STYLE CONFIGURATION (VIBRANT PALETTE - v01 PROD)
# ==============================================================================
STYLE_CONFIG = {
    "font_family": "Arial",
    "font_size_header": 16,
    "font_size_body": 14,
    "font_color": "#1f2a38", # Dark Navy/Black
    "colors": {
        "red_deep":   (168, 50, 50),
        "red_soft":   (240, 163, 163),
        "grey_warm":  (242, 244, 248),
        "blue_soft":  (154, 203, 232),
        "blue_deep":  (28, 86, 153),
        "therapeutic_grey": "#CFD8DC",
        "separator_white": "white"
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

def fmt_header(text, value, txt_color="black"):
    return (
        f"<span style='font-size:14px; font-weight:bold; color:{txt_color}'>"
        f"{text} ({value:+.1f} pts)"
        f"</span>"
    )

# ==============================================================================
# 1. GAUGE CHART
# ==============================================================================
def plot_success_gauge(score_val):
    steps = []
    n_slices = 100
    c = STYLE_CONFIG["colors"]

    for i in range(n_slices):
        v_start, v_end = i, i + 1
        if i < 25:
            color = interpolate_color(c["red_deep"], c["red_soft"], i / 25.0)
        elif i < 50:
            color = interpolate_color(c["red_soft"], c["grey_warm"], (i - 25) / 25.0)
        elif i < 75:
            color = interpolate_color(c["grey_warm"], c["blue_soft"], (i - 50) / 25.0)
        else:
            color = interpolate_color(c["blue_soft"], c["blue_deep"], (i - 75) / 25.0)
        steps.append({'range': [v_start, v_end], 'color': get_rgb_str(color)})

    for sep in [25, 50, 75]:
        steps.append({'range': [sep - 0.5, sep + 0.5], 'color': 'white'})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_val,
        number={
            "valueformat": ".1f",
            "font": {"size": 52, "color": STYLE_CONFIG["font_color"],
                     "family": STYLE_CONFIG["font_family"], "weight": "bold"}
        },
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickmode": "array",
                "tickvals": [0, 25, 50, 75, 100],
                "ticktext": ["0", "25", "50", "75", "100"],
                "ticklen": 0, "tickwidth": 0, "tickcolor": "white",
                "tickfont": {"size": STYLE_CONFIG["font_size_body"],
                             "color": "#555555",
                             "family": STYLE_CONFIG["font_family"], "weight": "bold"},
            },
            "bar": {"color": "rgba(0,0,0,0)"},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": steps,
            "threshold": {
                "line": {"color": STYLE_CONFIG["font_color"], "width": 5},
                "thickness": 0.75,
                "value": score_val
            },
        },
    ))
    fig.update_layout(
        margin=dict(l=35, r=35, t=40, b=10),
        paper_bgcolor="white",
        font={"family": STYLE_CONFIG["font_family"], "color": "#333333"},
        height=220,
        hovermode=False
    )
    return fig

# ==============================================================================
# 2. IMPACT BAR CHART
# ==============================================================================
def plot_impact_bar(df_pillars):
    """
    df_pillars expects columns: ['Pillar', 'Impact']
    """
    df_plot = df_pillars.copy()
    df_plot['Pillar_Clean'] = df_plot['Pillar'].apply(lambda x: x.split('. ', 1)[1] if '. ' in x else x)
    df_plot = df_plot.sort_values(by='Impact', ascending=True)

    fig = go.Figure()

    BAR_WIDTH = 0.6
    GLOBAL_MAX_SCALE = 15.0
    SLICE_STEP = 0.2
    c = STYLE_CONFIG["colors"]
    center_grey = c["grey_warm"]

    for idx, row_data in df_plot.iterrows():
        pillar_name = row_data['Pillar']
        val = row_data['Impact']
        abs_val = abs(val)

        slice_widths = []
        slice_colors = []
        slice_texts = []

        n_steps = int(np.ceil(abs_val / SLICE_STEP))
        for step in range(n_steps):
            current_abs_pos = step * SLICE_STEP
            
            if current_abs_pos + SLICE_STEP > abs_val:
                w = abs_val - current_abs_pos
                is_tip = True
            else:
                w = SLICE_STEP
                is_tip = False

            if val < 0: w = -w
            slice_widths.append(w)
            
            if is_tip:
                slice_texts.append(f"{val:+.1f}")
            else:
                slice_texts.append(None)

            color_pos = current_abs_pos + (abs(w) / 2)
            
            if "Therapeutic" in pillar_name:
                slice_colors.append(STYLE_CONFIG["colors"]["therapeutic_grey"])
                continue

            ratio = min(color_pos / GLOBAL_MAX_SCALE, 1.0)
            ratio = ratio ** 0.5

            if val >= 0:
                if ratio < 0.5:
                    final_rgb = interpolate_color(center_grey, c["blue_soft"], ratio * 2)
                else:
                    final_rgb = interpolate_color(c["blue_soft"], c["blue_deep"], (ratio - 0.5) * 2)
            else:
                if ratio < 0.5:
                    final_rgb = interpolate_color(center_grey, c["red_soft"], ratio * 2)
                else:
                    final_rgb = interpolate_color(c["red_soft"], c["red_deep"], (ratio - 0.5) * 2)

            slice_colors.append(get_rgb_str(final_rgb))

        fig.add_trace(go.Bar(
            y=[row_data['Pillar_Clean']],
            x=slice_widths,
            orientation='h',
            width=BAR_WIDTH,
            marker=dict(color=slice_colors, line=dict(width=0)),
            text=slice_texts,
            textposition='outside',
            textfont=dict(size=STYLE_CONFIG["font_size_body"], color="black", family=STYLE_CONFIG["font_family"]),
            hoverinfo='skip',
            showlegend=False
        ))

    limit = max(df_plot['Impact'].abs().max() * 1.2, 8.0)
    fig.add_vline(x=0, line_width=1, line_color="#333333")
    fig.update_layout(
        barmode='relative',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-limit, limit]),
        yaxis=dict(showticklabels=True, tickfont=dict(size=STYLE_CONFIG["font_size_body"], color="black", family=STYLE_CONFIG["font_family"]), automargin=True),
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white", paper_bgcolor="white", height=240, showlegend=False
    )
    return fig

# ==============================================================================
# 3. TREEMAP
# ==============================================================================
def plot_treemap(subcat_impacts, pillar_impacts):
    """
    subcat_impacts: list of {Pillar, Subcategory, Impact, Narrative}
    pillar_impacts: list of {Pillar, Impact}
    """
    nodes = {}
    node_sums = {"ALL_DRIVERS": sum(p['Impact'] for p in pillar_impacts)}
    c = STYLE_CONFIG["colors"]
    leaf_data = []

    for item in subcat_impacts:
        pillar_raw = item['Pillar']
        subtopic = item['Subcategory']
        impact = item['Impact']
        narrative = item['Narrative']
        
        pillar_clean = pillar_raw.split('. ', 1)[1] if '. ' in pillar_raw else pillar_raw
        is_therapeutic = "Therapeutic" in pillar_raw
        parent_id = f"PILLAR_{pillar_clean}"
        
        ratio = min(abs(impact) / 10.0, 1.0)
        ratio = max(0.2, ratio)

        if is_therapeutic:
            r, g, b = (207, 216, 220)
            txt_color = "black"
        elif impact >= 0:
            r, g, b = interpolate_color(c["blue_soft"], c["blue_deep"], ratio)
            txt_color = "white"
        else:
            r, g, b = interpolate_color(c["red_soft"], c["red_deep"], ratio)
            txt_color = "white"

        bg_hex = get_rgb_str((r,g,b))
        wrapped_narrative = '<br>'.join(textwrap.wrap(narrative, width=30))
        label_html = (
            f"<span style='font-size:15px; font-weight:bold; color:{txt_color}'>{subtopic}</span>"
            f"<br><span style='font-size:14px; color:{txt_color}'>{impact:+.1f} pts</span>"
            f"<br><br><span style='font-size:13px; font-style:normal; color:{txt_color}'>{wrapped_narrative}</span>"
        )

        leaf_id = f"{parent_id}_{subtopic}"
        leaf_data.append({
            "id": leaf_id, 
            "parent": parent_id, 
            "label": label_html, 
            "color": bg_hex, 
            "value": max(abs(impact), 0.1)
        })

    # Add Pillar Nodes
    for p_item in pillar_impacts:
        p_name = p_item['Pillar']
        p_imp = p_item['Impact']
        p_clean = p_name.split('. ', 1)[1] if '. ' in p_name else p_name
        p_id = f"PILLAR_{p_clean}"
        
        is_therapeutic = "Therapeutic" in p_name
        
        nodes[p_id] = {
            "parent": "ALL_DRIVERS",
            "label": fmt_header(p_clean.upper(), p_imp, txt_color="black"),
            "color": "#ECEFF1" if is_therapeutic else "#F5F5F5",
            "value": 0
        }

    nodes["ALL_DRIVERS"] = {
        "parent": "", 
        "label": fmt_header("ALL DRIVERS", node_sums["ALL_DRIVERS"], txt_color="black"), 
        "color": "#FFFFFF", 
        "value": 0
    }

    for item in leaf_data:
        nodes[item['id']] = {"parent": item['parent'], "label": item['label'], "color": item['color'], "value": item['value']}

    ids = list(nodes.keys())
    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=[nodes[k]['label'] for k in ids],
        parents=[nodes[k]['parent'] for k in ids],
        values=[nodes[k]['value'] for k in ids],
        marker=dict(colors=[nodes[k]['color'] for k in ids], line=dict(width=1, color='white')),
        textinfo="label", textposition="top left", tiling=dict(packing="squarify"), hoverinfo="skip",
        pathbar=dict(visible=True, thickness=30, textfont=dict(family="Arial", size=11))
    ))
    fig.update_layout(margin=dict(t=34, l=15, r=15, b=15), height=550, font=dict(family="Arial"), paper_bgcolor='white', hovermode=False)
    return fig
