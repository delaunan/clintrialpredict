import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re
import textwrap

# ==========================
# 0. STYLE CONFIG (NEUTRAL PRO)
# ==========================
STYLE_CONFIG = {
    "font_family": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    "font_color": "#334155",
    "colors": {
        "red_deep":   (176, 63, 63),
        "red_soft":   (236, 162, 162),
        "grey_warm":  (242, 244, 248),
        "blue_soft":  (162, 198, 228),
        "blue_deep":  (47, 98, 166),
        "therapeutic_grey": "#CFD8DC",
        # Pastel Zones
        "pastel_red": "#fde8e8",
        "pastel_orange": "#fff7ed",
        "pastel_blue": "#eff6ff",
        "pastel_green": "#f0fdf4"
    }
}

GAUGE_MIN = 0.0
GAUGE_MAX = 100.0

# Shared segmentation control for both gauge bands and impact-bar slices.
# Increase for finer / more numerous segments.
# Decrease for chunkier / fewer segments.
SEGMENT_COUNT = 50
BAR_SEGMENT_COUNT_DIVISOR = 2

# Keep these separate from segmentation.
GAUGE_MARKER_LINE_WIDTH = 4
GAUGE_MARKER_THICKNESS = 0.7
BAR_WIDTH = 0.7




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

def fmt_header(text, value, txt_color="#334155", split=False):
    if split:
        return (
            f"<span style='font-family:{STYLE_CONFIG['font_family']}; "
            f"font-size:18px; color:{txt_color}; line-height:1.0;'>"
            f"<b>{text}</b><br><b>({value:+.1f} pts)</b>"
            f"</span>"
        )
    return (
        f"<span style='font-family:{STYLE_CONFIG['font_family']}; "
        f"font-size:18px; color:{txt_color}; line-height:1.0;'>"
        f"<b>{text}</b> <b>({value:+.1f} pts)</b>"
        f"</span>"
    )

# ==========================
# 1. GAUGE CHART
# ==========================

def plot_success_gauge(score_val, height=220):
    steps = []
    c = STYLE_CONFIG["colors"]

    gauge_segment_step = (GAUGE_MAX - GAUGE_MIN) / SEGMENT_COUNT
    gauge_separator_width = gauge_segment_step

    for start in np.arange(GAUGE_MIN, GAUGE_MAX, gauge_segment_step):
        end = min(start + gauge_segment_step, GAUGE_MAX)
        mid = (start + end) / 2

        if mid < 25:
            color = interpolate_color(c["red_deep"], c["red_soft"], mid / 25.0)
        elif mid < 50:
            color = interpolate_color(c["red_soft"], c["grey_warm"], (mid - 25) / 25.0)
        elif mid < 75:
            color = interpolate_color(c["grey_warm"], c["blue_soft"], (mid - 50) / 25.0)
        else:
            color = interpolate_color(c["blue_soft"], c["blue_deep"], (mid - 75) / 25.0)

        steps.append({"range": [start, end], "color": get_rgb_str(color)})

    for sep in [25, 50, 75]:
        steps.append({
            "range": [
                max(GAUGE_MIN, sep - gauge_separator_width / 2),
                min(GAUGE_MAX, sep + gauge_separator_width / 2),
            ],
            "color": "white",
        })

    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=score_val,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {
                "range": [GAUGE_MIN, GAUGE_MAX],
                "tickmode": "array",
                "tickvals": [0, 25, 50, 75, 100],
                "tickfont": {
                    "size": 14,
                    "color": "#475569",
                    "family": STYLE_CONFIG["font_family"],
                    "weight": "bold",
                },
            },
            "bar": {"color": "rgba(0,0,0,0)"},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": steps,
            "threshold": {
                "line": {
                    "color": STYLE_CONFIG["font_color"],
                    "width": GAUGE_MARKER_LINE_WIDTH,
                },
                "thickness": GAUGE_MARKER_THICKNESS,
                "value": score_val,
            },
        },
    ))

    fig.add_annotation(
        x=0.5,
        y=0.35,
        xref="paper",
        yref="paper",
        text=f"<span style='font-weight:700'>{score_val:.1f}</span>",
        showarrow=False,
        font=dict(
            size=23,
            color=STYLE_CONFIG["font_color"],
            family=STYLE_CONFIG["font_family"],
        ),
        xanchor="center",
        yanchor="middle",
        align="center",
    )

    fig.update_layout(
        margin=dict(l=35, r=35, t=40, b=10),
        height=height,
        paper_bgcolor="white",
        hovermode=False,
    )
    return fig

# ==========================
# 2. IMPACT BAR CHART
# ==========================

def plot_impact_bar(df_pillars, height=240):
    df_plot = df_pillars.copy()

    df_plot['Pillar_Clean'] = df_plot['Pillar'].apply(
        lambda x: f"<b>{re.sub(r'^\\d+\\.\\s*', '', x)}</b>"
    )
    df_plot = df_plot.sort_values(by='Impact', ascending=True)

    max_val = df_plot['Impact'].abs().max()
    limit = max(max_val * 1.35, 6.0)
    effective_bar_segment_count = max(1, SEGMENT_COUNT / BAR_SEGMENT_COUNT_DIVISOR)
    bar_scale_reference = max_val if max_val > 0 else 1.0
    bar_segment_step = bar_scale_reference / effective_bar_segment_count

    c = STYLE_CONFIG["colors"]
    fig = go.Figure()

    n_slices = int(np.ceil(max_val / bar_segment_step)) if max_val > 0 else 1

    for i in range(n_slices):
        pos = i * bar_segment_step
        widths, colors = [], []

        for _, row in df_plot.iterrows():
            val = row['Impact']
            abs_v = abs(val)

            if pos >= abs_v:
                w = 0
            elif pos + bar_segment_step > abs_v:
                w = abs_v - pos
            else:
                w = bar_segment_step

            widths.append(-w if val < 0 else w)

            ratio = min((pos + abs(w) / 2) / bar_scale_reference, 1.0) ** 0.5
            if val >= 0:
                final_rgb = (
                    interpolate_color(c["grey_warm"], c["blue_soft"], ratio * 2)
                    if ratio < 0.5
                    else interpolate_color(c["blue_soft"], c["blue_deep"], (ratio - 0.5) * 2)
                )
                colors.append(get_rgb_str(final_rgb))
            else:
                final_rgb = (
                    interpolate_color(c["grey_warm"], c["red_soft"], ratio * 2)
                    if ratio < 0.5
                    else interpolate_color(c["red_soft"], c["red_deep"], (ratio - 0.5) * 2)
                )
                colors.append(get_rgb_str(final_rgb))

        fig.add_trace(go.Bar(
            y=df_plot['Pillar_Clean'],
            x=widths,
            orientation='h',
            width=BAR_WIDTH,
            marker=dict(color=colors, line=dict(width=0)),
            hoverinfo='skip',
            showlegend=False
        ))

    label_offset = max(limit * 0.05, 0.35)
    axis_limit = limit + label_offset + 0.35

    for _, row in df_plot.iterrows():
        val = row["Impact"]
        fig.add_annotation(
            x=val + label_offset if val >= 0 else val - label_offset,
            y=row["Pillar_Clean"],
            xref="x",
            yref="y",
            text=f"<b>{val:+.1f} pts</b>",
            showarrow=False,
            xanchor="left" if val >= 0 else "right",
            yanchor="middle",
            align="left" if val >= 0 else "right",
            font=dict(
                size=13,
                color="#334155",
                family=STYLE_CONFIG["font_family"]
            )
        )

    fig.add_vline(x=0, line_width=1, line_color="#333333")
    fig.update_layout(
        barmode='relative',
        xaxis=dict(
            showticklabels=False,
            range=[-axis_limit, axis_limit],
            zeroline=False,
            showgrid=False
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(
                size=13,
                color="#334155",
                family=STYLE_CONFIG["font_family"]
            )
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False
    )
    return fig


# ==========================
# 3. TREEMAP
# ==========================
def plot_treemap(subcat_impacts, pillar_impacts, show_values=True, height=600):
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
            f"<span style='font-family:{STYLE_CONFIG['font_family']}; color:white; line-height:1.05;'>"
            f"<b style='font-size:15px;'>{subtopic}</b>"
            f"<br><b style='font-size:14px;'>{impact:+.1f} pts</b>"
            f"<br><br><span style='font-size:11px; font-weight:500;'>{feat_html}</span>"
            f"</span>"
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
        height=height,
        font=dict(family=STYLE_CONFIG["font_family"], color=STYLE_CONFIG["font_color"]),
        paper_bgcolor='white',
        hovermode=False
    )
    return fig
