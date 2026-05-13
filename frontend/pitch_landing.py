import base64
from pathlib import Path
import streamlit as st

def render_pitch_page():
    """
    Renders a highly polished, responsive, professional landing page for CTPredict.
    Matches the exact text provided, uses the app.py header style, and implements
    a modern SaaS design with high-contrast CTA cards and perfect flexbox/grid alignment.
    """

    # --- ASSET LOADING ---
    assets_dir = Path(__file__).resolve().parent

    def get_img_b64(filename):
        path = assets_dir / filename
        if path.exists():
            with open(path, "rb") as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        return None

    def render_image_box(filename, alt_text, style_class=""):
        b64 = get_img_b64(filename)
        if b64:
            return f'<div class="pitch-img-box {style_class}"><img src="{b64}" alt="{alt_text}"></div>'
        else:
            return f'<div class="pitch-img-box pitch-placeholder {style_class}"><span>Visual: {alt_text}</span></div>'

    # --- CALLBACK ---
    def launch_demo():
        st.session_state["pitch_seen"] = True
        st.session_state["selected_nct_id"] = None
        st.rerun()

    # --- CSS INJECTION ---
    st.markdown("""
        <style>
            /* Base Variables */
            :root {
                --pitch-bg: #f1f5f9;
                --pitch-text-main: #334155;
                --pitch-text-sec: #64748b;
                --pitch-brand-dark: #52606d;
                --pitch-accent: #89A7C9; /* App.py button blue */
                --pitch-deep-blue: #2f62a6;
                --pitch-red: #b03f3f;
                --pitch-border: #e2e8f0;
                --pitch-card-bg: #ffffff;
                --pitch-radius: 16px;
                --pitch-shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
                --pitch-shadow-md: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
                --pitch-shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
            }

            /* Clean up Streamlit default padding */
            .block-container {
                padding-top: 3rem !important;
                padding-bottom: 4rem !important;
                max-width: 1100px !important;
            }

            /* Typography Hierarchy */
            .pitch-section-title {
                display: flex;
                align-items: center;
                justify-content: center;
                font-family: 'Inter', sans-serif;
                font-size: clamp(1.8rem, 3vw, 2.2rem);
                font-weight: 800;
                color: var(--pitch-text-sec);
                margin: 5rem 0 3rem 0;
                white-space: nowrap;
            }
            .pitch-section-title::before,
            .pitch-section-title::after {
                content: "";
                flex: 1;
                border-bottom: 2px solid var(--pitch-border);
            }
            .pitch-section-title::before { margin-right: 1.5rem; }
            .pitch-section-title::after { margin-left: 1.5rem; }

            .pitch-h2-full {
                font-family: 'Inter', sans-serif;
                font-size: clamp(1.75rem, 3vw, 2.25rem);
                font-weight: 800;
                color: var(--pitch-brand-dark);
                margin-bottom: 1.5rem;
                letter-spacing: -0.01em;
                line-height: 1.2;
                width: 100%;
            }
            .pitch-h3 {
                font-family: 'Inter', sans-serif;
                font-size: clamp(1.35rem, 2vw, 1.6rem);
                font-weight: 800;
                color: var(--pitch-brand-dark);
                margin-bottom: 1rem;
                line-height: 1.3;
            }
            .pitch-h4 {
                font-family: 'Inter', sans-serif;
                font-size: 1.15rem;
                font-weight: 800;
                color: var(--pitch-brand-dark);
                margin-bottom: 0.5rem;
            }
            .pitch-p {
                font-family: 'Inter', sans-serif;
                font-size: 1.1rem;
                color: var(--pitch-text-sec);
                line-height: 1.65;
                margin-bottom: 1.25rem;
            }
            .pitch-p-strong {
                font-weight: 700;
                color: var(--pitch-text-main);
            }

            /* Hero Section */
            .pitch-hero-headline {
                font-size: clamp(2.2rem, 5vw, 3.5rem);
                font-weight: 900;
                color: var(--pitch-brand-dark);
                text-align: center;
                line-height: 1.15;
                letter-spacing: -0.02em;
                margin-bottom: 2.5rem;
                max-width: 900px;
                margin-left: auto;
                margin-right: auto;
            }
            .pitch-hero-body {
                font-size: 1.2rem;
                color: var(--pitch-text-main);
                line-height: 1.7;
                text-align: center;
                max-width: 850px;
                margin: 0 auto 4rem auto;
            }

            /* CTA Card (Narrower, Taller, Centered Content) */
            .st-key-cta_card_top, .st-key-cta_card_bottom {
                background: var(--pitch-card-bg);
                padding: 2rem;
                border-radius: var(--pitch-radius);
                box-shadow: var(--pitch-shadow-lg);
                border: 1px solid var(--pitch-border);
                max-width: 420px;
                margin: 0 auto 5rem auto;
                min-height: 320px;
            }
            /* Force Streamlit's inner container to use flexbox for perfect centering */
            .st-key-cta_card_top > div > div[data-testid="stVerticalBlock"],
            .st-key-cta_card_bottom > div > div[data-testid="stVerticalBlock"] {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100%;
                text-align: center;
            }
            .pitch-cta-title {
                font-size: 1.35rem;
                font-weight: 800;
                color: var(--pitch-brand-dark);
                margin-bottom: 0.75rem;
            }
            .pitch-cta-text {
                font-size: 1.05rem;
                color: var(--pitch-text-sec);
                margin-bottom: 2rem;
                line-height: 1.5;
            }

            /* Dark Card (Why it matters) */
            .pitch-dark-card {
                background: var(--pitch-brand-dark);
                border-radius: var(--pitch-radius);
                padding: clamp(2.5rem, 5vw, 4rem);
                text-align: center;
                box-shadow: var(--pitch-shadow-md);
                margin-bottom: 6rem;
            }
            .pitch-dark-card .pitch-h3 { color: #ffffff; font-size: clamp(1.5rem, 3vw, 2rem); margin-bottom: 1.5rem; }
            .pitch-dark-card .pitch-p { color: #f8fafc; max-width: 900px; margin: 0 auto; font-size: 1.15rem; }

            /* Flexbox Layout for "How it works" (Perfect Height Matching) */
            .pitch-flex-row {
                display: flex;
                gap: 3rem;
                align-items: stretch; /* Forces both columns to be the same height */
                margin-bottom: 1rem;
            }
            .pitch-flex-col-text {
                flex: 1;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }
            .pitch-flex-col-img {
                flex: 1.2; /* Slightly wider for the images */
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                padding: 0.5rem; /* Reduced padding to make image look larger */
                box-shadow: var(--pitch-shadow-md);
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .pitch-flex-col-img img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            .pitch-full-width-text {
                width: 100%;
                margin-bottom: 4rem;
            }
            @media (max-width: 768px) {
                .pitch-flex-row {
                    flex-direction: column;
                }
                .pitch-flex-col-img {
                    min-height: 300px;
                }
            }

            /* Score Section Specifics (Grid Layout for perfect alignment) */
            .score-master-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                align-items: stretch; /* Ensures items align to top/bottom heights */
            }
            @media (max-width: 768px) {
                .score-master-grid { grid-template-columns: 1fr; }
            }
            .pitch-h2-score {
                font-family: 'Inter', sans-serif;
                font-size: clamp(1.75rem, 3vw, 2.25rem);
                font-weight: 800;
                color: var(--pitch-brand-dark);
                margin-top: 0;
                margin-bottom: 1.25rem;
                letter-spacing: -0.01em;
                line-height: 1.2;
            }
            .color-box {
                width: 12px;
                height: 12px;
                border-radius: 3px;
                display: inline-block;
                margin-right: 8px;
                vertical-align: middle;
            }
            .cb-low { background: linear-gradient(90deg, rgb(162,198,228) 0%, rgb(47,98,166) 100%); }
            .cb-fav { background: linear-gradient(90deg, rgb(242,244,248) 0%, rgb(162,198,228) 100%); }
            .cb-watch { background: linear-gradient(90deg, rgb(236,162,162) 0%, rgb(242,244,248) 100%); }
            .cb-high { background: linear-gradient(90deg, rgb(176,63,63) 0%, rgb(236,162,162) 100%); }

            .pitch-img-box.gauge-box {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 240px;
                padding: 1rem 1.5rem;
                overflow: hidden;
                width: 100%;
                height: 100%;
                border-radius: 12px;
                box-shadow: var(--pitch-shadow-sm);
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
            }
            .pitch-img-box.gauge-box img {
                transform: scale(1.35);
                transform-origin: center center;
            }

            .pitch-score-text {
                font-size: clamp(1.15rem, 2vw, 1.35rem);
                font-weight: 800;
                color: var(--pitch-brand-dark);
                line-height: 1.6;
                margin: 0;
                display: inline-block;
            }
            .hl-blue {
                color: var(--pitch-deep-blue);
                background: rgba(47,98,166,0.1);
                padding: 2px 8px;
                border-radius: 6px;
            }
            .hl-red {
                color: var(--pitch-red);
                background: rgba(176,63,63,0.1);
                padding: 2px 8px;
                border-radius: 6px;
            }

            .interpret-card {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: var(--pitch-shadow-sm);
                height: 100%;
            }
            .interpret-card-title {
                font-size: 1.1rem;
                font-weight: 800;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
            }

            /* Dimension Items */
            .dim-item {
                margin-bottom: 1.5rem;
            }
            .dim-title {
                font-weight: 700;
                font-size: 1.1rem;
                color: var(--pitch-text-sec); /* Matched to "High Risk" label color */
                margin-bottom: 0.2rem;
            }
            .dim-desc {
                font-size: 1.05rem;
                color: var(--pitch-text-sec); /* Matched to description text size/color */
            }

            /* Foundation Banner */
            .pitch-foundation-banner {
                background: linear-gradient(135deg, var(--pitch-brand-dark), #334155);
                border-radius: var(--pitch-radius);
                padding: 4rem 3rem;
                text-align: center;
                box-shadow: var(--pitch-shadow-md);
                margin-bottom: 6rem;
                color: white;
            }
            .pitch-foundation-banner .pitch-h3 {
                color: white;
                font-size: 2rem;
                margin-bottom: 1.5rem;
            }
            .pitch-foundation-banner .pitch-p {
                color: #e2e8f0;
                max-width: 800px;
                margin: 0 auto 1.5rem auto;
                font-size: 1.15rem;
            }

            /* Metrics Grid */
            .pitch-metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 2rem;
                margin-bottom: 6rem;
            }
            .pitch-metric-card {
                background: var(--pitch-card-bg);
                border-radius: var(--pitch-radius);
                padding: 2.5rem;
                box-shadow: var(--pitch-shadow-sm);
                border: 1px solid var(--pitch-border);
                border-top: 5px solid var(--pitch-accent);
            }
            .pitch-metric-title {
                font-size: 1.5rem;
                font-weight: 900;
                color: var(--pitch-brand-dark);
                margin-bottom: 1rem;
            }

            /* Value Grid */
            .pitch-value-subtitle {
                text-align: center;
                font-size: 1.35rem;
                font-weight: 700;
                color: var(--pitch-text-sec);
                margin-bottom: 4rem;
                margin-top: -1.5rem;
            }
            .pitch-value-grid {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 1.5rem;
                margin-bottom: 6rem;
            }
            .pitch-value-card {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                padding: 2.5rem 2rem;
                box-shadow: var(--pitch-shadow-sm);
                width: calc(33.333% - 1rem);
                min-width: 280px;
                max-width: 340px;
                text-align: center;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .pitch-value-icon {
                width: 100px;
                height: 100px;
                object-fit: contain;
                margin-bottom: 1.5rem;
                background: #f8fafc;
                padding: 16px;
                border-radius: 20px;
                border: 1px solid #e2e8f0;
            }

            /* Footer */
            .pitch-footer {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                padding: 3rem;
                text-align: center;
                box-shadow: var(--pitch-shadow-sm);
                margin-bottom: 2rem;
            }

            /* Streamlit Primary Button Override */
            .stButton > button[kind="primary"] {
                border: none !important;
                background: var(--pitch-accent) !important;
                color: #ffffff !important;
                border-radius: 10px !important;
                padding: 1.5rem 0 !important;
                font-size: 1.15rem !important;
                font-weight: 800 !important;
                box-shadow: 0 4px 12px rgba(137, 167, 201, 0.3) !important;
                transition: all 0.2s ease !important;
                width: 100% !important;
                max-width: 250px !important;
                margin: 0 auto !important;
            }
            .stButton > button[kind="primary"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 16px rgba(137, 167, 201, 0.4) !important;
                background: #7a96b5 !important;
            }
            .stButton > button[kind="primary"] p {
                font-size: 1.15rem !important;
                font-weight: 800 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- 1. HEADER (Replicating app.py exactly) ---
    logo_b64 = get_img_b64("logo_grey_title.png")
    brand_filter = "contrast(1.5) brightness(0.9) grayscale(100%) sepia(100%) hue-rotate(180deg) saturate(0.8) brightness(0.85) contrast(1.2)"

    st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 12px; justify-content: center; margin-bottom: 4rem;'>
            <div style='background-color: white; border: 4px solid #52606d; padding: 2px; border-radius: 18px; display: flex; align-items: center; justify-content: center; height: 72px; width: 72px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
                <img src='{logo_b64}' style='height: 68px; filter: {brand_filter};'>
            </div>
            <div style='display: block; align-items: stretch; gap: 0px; text-align: left;'>
                <div style='font-size: 2.8rem; font-weight: 800; color: #52606d; line-height: 1; margin-top: 0px;'>CTPredict</div>
                <div style='color: #52606d; font-size: 1.5rem; font-weight: 800; display: flex; align-items: baseline; gap: 15px; margin-top: 0px;'>
                    <span style='line-height: 1;'>Late-Stage Clinical Trial Predictive Engine</span>
                    <span style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;'>demo version</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 2. HERO SECTION ---
    st.markdown("""
        <div class="pitch-hero-headline">
            Identify late-stage clinical trials at risk<br>
            <span style="color: var(--pitch-accent);">before execution begins.</span>
        </div>
        <div class="pitch-hero-body">
            <span class="pitch-p-strong">CTPredict</span><br>
            Predicts full completion or early termination in Phase II/III trials from early design-stage information.<br><br>
            A predictive pre-screening engine built on publicly available data from 30,000+ late-stage clinical trials to identify trials at risk of early termination, classify risk tiers, and reveal trial-specific risk drivers.<br><br>
            Simulation mode also enables testing the impact of trial-design modifications before trial initiation.
        </div>
    """, unsafe_allow_html=True)

    # --- 3. TOP CTA CARD ---
    with st.container(key="cta_card_top"):
        st.markdown("""
            <div class="pitch-cta-title">See the prediction in action.</div>
            <div class="pitch-cta-text">Start from a real Phase II/III trial and move from risk tier to score drivers in a few clicks.</div>
        """, unsafe_allow_html=True)
        st.button("Launch Demo", key="cta_btn_top", on_click=launch_demo, type="primary")

    # --- 4. WHY IT MATTERS ---
    st.markdown('<div class="pitch-section-title">Why it matters</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="pitch-dark-card">
            <div class="pitch-h3">Focus attention early, where clinical, financial, and strategic stakes are highest.</div>
            <div class="pitch-p">
                Phase II/III is where scientific ambition meets major financial and strategic stakes. CTPredict helps identify trials most exposed to early-termination risk, helping focus support, review, and key protocol or strategic decisions where they matter most.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 5. HOW IT WORKS ---
    st.markdown('<div class="pitch-section-title" style="color: var(--pitch-accent);">How it works</div>', unsafe_allow_html=True)

    # Row 1: The score
    gauge_b64 = get_img_b64("gauge.png")
    gauge_img_html = f'<img src="{gauge_b64}" alt="Completion Score Gauge">' if gauge_b64 else '<div class="pitch-placeholder"><span>Visual: Completion Score Gauge</span></div>'

    st.markdown(f"""
        <div class="score-master-grid">
            <div class="score-cell">
                <div class="pitch-h2-score">The Score</div>
                <div class="pitch-p" style="margin-bottom: 0;">
                    <span style="font-weight: 700; font-size: 1.05rem; color: var(--pitch-text-sec);">
                        Reflects how closely a trial resembles historical patterns of <span class="hl-red">early termination</span> or <span class="hl-blue">full completion</span>,<br>
                        and assigns it to one of four completion-risk tiers:
                    </span>
                    <br><br>
                    <div style="margin-bottom: 8px;"><span class="color-box cb-high"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">High Risk:</span> 0-25 points</div>
                    <div style="margin-bottom: 8px;"><span class="color-box cb-watch"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">Watchlist:</span> 25-50 points</div>
                    <div style="margin-bottom: 8px;"><span class="color-box cb-fav"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">Favorable:</span> 50-75 points</div>
                    <div style="margin-bottom: 8px;"><span class="color-box cb-low"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">Low Risk:</span> 75-100 points</div>
                </div>
            </div>
            <div class="score-cell">r
                <div class="pitch-img-box gauge-box">{gauge_img_html}</div>
            </div>
            <div class="score-cell">
                <div class="interpret-card">
                    <div class="interpret-card-title" style="color: var(--pitch-red);"><span class="color-box cb-high"></span> Red reflects a riskier trial profile.</div>
                    <div class="pitch-p" style="margin-bottom: 0; font-size: 1.05rem;">May flag design choices worth challenging - or capture ambition: novel science, rigorous endpoints, flexible design, or higher complexity. Higher risk can also point to higher potential value.</div>
                </div>
            </div>
            <div class="score-cell">
                <div class="interpret-card">
                    <div class="interpret-card-title" style="color: var(--pitch-deep-blue);"><span class="color-box cb-low"></span> Blue signals higher likelihood of full completion.</div>
                    <div class="pitch-p" style="margin-bottom: 0; font-size: 1.05rem;">May signal strong execution conditions, high-quality design, or a more established scientific profile with lower complexity. Full completion should still be read alongside scientific and asset value.</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 4rem;'></div>", unsafe_allow_html=True)

    # --- 6. FOUR RISK DIMENSIONS ---
    st.markdown(f"""
        <div class="pitch-flex-row">
            <div class="pitch-flex-col-img">
                {render_image_box("barchart.png", "Impact Bar Chart")}
            </div>
            <div class="pitch-flex-col-text">
                <div class="pitch-h2-score" style="margin-top: 0; margin-bottom: 2rem;">Four Risk Dimensions</div>
                <div class="dim-item">
                    <div class="dim-title">Scientific Challenge</div>
                    <div class="dim-desc">Biological Complexity · Protocol Design</div>
                </div>
                <div class="dim-item">
                    <div class="dim-title">Execution Framework</div>
                    <div class="dim-desc">Operational Setup · Methodological Rigor</div>
                </div>
                <div class="dim-item">
                    <div class="dim-title">Therapeutic Context</div>
                    <div class="dim-desc">Disease Profile · Development Stage</div>
                </div>
                <div class="dim-item">
                    <div class="dim-title">Patient Profile</div>
                    <div class="dim-desc">Population Scope · Clinical Severity</div>
                </div>
            </div>
        </div>
        <div class="pitch-full-width-text"></div>
    """, unsafe_allow_html=True)

    # Row 3: Interactive drivers map
    st.markdown('<div class="pitch-h2-full">Interactive drivers map</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="pitch-flex-row">
            <div class="pitch-flex-col-text">
                <div class="pitch-p">
                    The interactive drivers map breaks each prediction into 27 core features, showing which factors contribute most to the final risk signal.<br><br>
                    Larger blocks indicate stronger impact.<br><br>
                    <span style="color: var(--pitch-red); font-weight: 700;">Red drivers push toward early termination.</span><br>
                    <span style="color: var(--pitch-accent); font-weight: 700;">Blue drivers support full completion.</span><br><br>
                    Some of these features are actionable design drivers, helping identify trial-design elements that may deserve closer review.
                </div>
            </div>
            <div class="pitch-flex-col-img">
                {render_image_box("treemap.png", "Interactive Treemap")}
            </div>
        </div>
        <div class="pitch-full-width-text"></div>
    """, unsafe_allow_html=True)

    # --- 7. MODEL AND DATA FOUNDATION ---
    st.markdown('<div class="pitch-section-title">Model and data foundation</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="pitch-foundation-banner">
            <div class="pitch-p">
                CTPredict is built on publicly available AACT data from industry-led Phase II/III trials initiated since 2009. Its XGBoost supervised machine-learning model uses 27 design-stage variables to learn patterns of full completion vs. early termination.
            </div>
            <div class="pitch-p" style="margin-bottom: 0; font-size: 0.95rem; opacity: 0.85;">
                <strong>AACT:</strong> Aggregate Analysis of ClinicalTrials.gov - an analysis-ready public database derived from ClinicalTrials.gov records.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 8. HOW STRONG IS THE PREDICTION ---
    st.markdown('<div class="pitch-section-title">How strong is the prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="pitch-h3" style="text-align: center; margin-bottom: 3rem;">Model performance and early-risk signal.</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="pitch-metrics-grid">
            <div class="pitch-metric-card">
                <div class="pitch-metric-title">30,000+ trials</div>
                <div class="pitch-p" style="margin-bottom:0;">Industry-led Phase II/III trials initiated since 2009, labelled for supervised learning as full completion vs. early termination.</div>
            </div>
            <div class="pitch-metric-card">
                <div class="pitch-metric-title">78% AUC</div>
                <div class="pitch-p" style="margin-bottom:0;">CTPredict ranks early-terminating trials as higher risk 78% of the time - an outstanding result, well above the 50% baseline, using only publicly available Phase II/III trial data.</div>
            </div>
            <div class="pitch-metric-card">
                <div class="pitch-metric-title">3 in 4 early terminations flagged early</div>
                <div class="pitch-p" style="margin-bottom:0;">Around 75% of trials that later terminated early fall into the High Risk or Watchlist categories - below 50 points.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 9. WHERE IT ADDS VALUE ---
    st.markdown('<div class="pitch-section-title">Where it adds value</div>', unsafe_allow_html=True)
    st.markdown('<div class="pitch-value-subtitle">One predictive engine. Multiple decision perspectives.</div>', unsafe_allow_html=True)

    def get_icon_html(filename):
        b64 = get_img_b64(filename)
        if b64: return f'<img src="{b64}" class="pitch-value-icon">'
        return '<div class="pitch-value-icon" style="background:#e2e8f0; border-radius:8px;"></div>'

    st.markdown(f"""
        <div class="pitch-value-grid">
            <div class="pitch-value-card">
                {get_icon_html("icon_portfolio_mgt.png")}
                <div class="pitch-h4">Portfolio Management</div>
                <div class="pitch-p" style="margin-bottom:0;">Highlight late-stage assets that may require closer scrutiny.</div>
            </div>
            <div class="pitch-value-card">
                {get_icon_html("icon_ta_lead.png")}
                <div class="pitch-h4">Therapeutic Area Leadership</div>
                <div class="pitch-p" style="margin-bottom:0;">Benchmark completion-risk patterns across indications, modalities, phases, and trial designs within a therapeutic area.</div>
            </div>
            <div class="pitch-value-card">
                {get_icon_html("icon_clin_lead.png")}
                <div class="pitch-h4">Clinical Development Leads</div>
                <div class="pitch-p" style="margin-bottom:0;">Explore in simulation mode how design choices may shift the completion-risk profile before trial initiation.</div>
            </div>
            <div class="pitch-value-card">
                {get_icon_html("icon_investor.png")}
                <div class="pitch-h4">Investors & Analysts</div>
                <div class="pitch-p" style="margin-bottom:0;">Compare completion-risk profiles across late-stage assets in the industry using public data.</div>
            </div>
            <div class="pitch-value-card">
                {get_icon_html("icon_training.png")}
                <div class="pitch-h4">Training & Capability Building</div>
                <div class="pitch-p" style="margin-bottom:0;">Use real trials and simulation mode to support learning and strengthen risk-based decision-making.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 10. BOTTOM CTA CARD ---
    with st.container(key="cta_card_bottom"):
        st.markdown("""
            <div class="pitch-cta-title">See the prediction in action.</div>
            <div class="pitch-cta-text">Start from a real Phase II/III trial and move from risk tier to score drivers in a few clicks.</div>
        """, unsafe_allow_html=True)
        st.button("Launch Demo", key="cta_btn_bottom", on_click=launch_demo, type="primary")

    # --- 11. FOOTER ---
    st.markdown("""
        <div class="pitch-footer">
            <div class="pitch-h3">Pilot version. Welcoming your ideas.</div>
            <div class="pitch-p" style="max-width: 800px; margin: 0 auto 1.5rem auto;">
                This demo focuses on single-trial exploration. Broader views are available or currently being developed, including sponsor full-portfolio screening, therapeutic-area benchmarking, and simulation-based use cases.<br><br>
                I would be happy to hear your feedback, questions, or ideas for future development.
            </div>
            <div class="pitch-p-strong">Contact: Nicolas Delaunay</div>
            <div class="pitch-p">Email: <a href="mailto:delaunay80@gmail.com" style="color: var(--pitch-brand-dark); text-decoration: none; font-weight: 700;">delaunay80@gmail.com</a></div>
        </div>
    """, unsafe_allow_html=True)
