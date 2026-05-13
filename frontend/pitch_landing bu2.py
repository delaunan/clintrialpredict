import base64
from pathlib import Path
import streamlit as st

def render_pitch_page():
    """
    Renders a highly polished, responsive, professional landing page for CTPredict.
    Matches the exact text provided, uses a left-aligned header style, and implements
    a modern SaaS design with high-contrast CTA cards and perfect flexbox/grid alignment.
    """

    # --- ASSET LOADING ---
    assets_dir = Path(__file__).resolve().parent

    def get_img_b64(filename):
        # Checks root, then frontend subfolder for the specific screenshot
        paths_to_check = [
            assets_dir / filename,
            assets_dir / "frontend" / filename,
            Path("./frontend") / filename
        ]
        for path in paths_to_check:
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
                --pitch-accent: #2f62a6; /* Professional SaaS Blue */
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
                padding-top: 2rem !important;
                padding-bottom: 4rem !important;
                max-width: 1200px !important;
            }

            /* Header Alignment (Left) */
            .header-container {
                display: flex;
                align-items: center;
                gap: 18px;
                justify-content: flex-start;
                margin-bottom: 5rem;
                text-align: left;
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
                margin: 6rem 0 3rem 0;
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
                font-size: 1.15rem;
                color: var(--pitch-text-sec);
                line-height: 1.65;
                margin-bottom: 1.25rem;
            }

            /* Hero Section Styling */
            .hero-h1 {
                font-size: clamp(2.8rem, 5vw, 4rem);
                font-weight: 900;
                color: var(--pitch-brand-dark);
                line-height: 1.05;
                letter-spacing: -0.03em;
                margin-bottom: 2rem;
            }
            .hero-p-bullet {
                font-size: 1.2rem;
                color: var(--pitch-text-sec);
                line-height: 1.6;
                margin-bottom: 1.5rem;
                padding-left: 0;
            }

            /* Cards */
            .pitch-card {
                background: var(--pitch-card-bg);
                padding: 2rem;
                border-radius: var(--pitch-radius);
                box-shadow: var(--pitch-shadow-sm);
                border: 1px solid var(--pitch-border);
                height: 100%;
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
            .pitch-dark-card .pitch-p { color: #f8fafc; max-width: 900px; margin: 0 auto; font-size: 1.2rem; }

            /* Flexbox Layout for "How it works" */
            .pitch-flex-row {
                display: flex;
                gap: 3rem;
                align-items: stretch;
                margin-bottom: 1rem;
            }
            .pitch-flex-col-text {
                flex: 1;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }
            .pitch-flex-col-img {
                flex: 1.7;
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                padding: 0.5rem;
                box-shadow: var(--pitch-shadow-md);
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .pitch-flex-col-img img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                border-radius: 10px;
            }

            /* Score Section Specifics */
            .score-master-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                align-items: stretch;
            }
            @media (max-width: 768px) {
                .score-master-grid { grid-template-columns: 1fr; }
                .pitch-flex-row { flex-direction: column; }
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

            .hl-blue { color: var(--pitch-deep-blue); background: rgba(47,98,166,0.1); padding: 2px 8px; border-radius: 6px; font-weight: 700; }
            .hl-red { color: var(--pitch-red); background: rgba(176,63,63,0.1); padding: 2px 8px; border-radius: 6px; font-weight: 700; }

            /* Dimension Items */
            .dim-item { margin-bottom: 1.2rem; }
            .dim-title { font-weight: 800; font-size: 1.25rem; color: var(--pitch-text-sec); margin-bottom: 0.1rem; }
            .dim-desc { font-size: 1.15rem; color: var(--pitch-text-sec); }

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
                text-align: center;
            }
            .pitch-metric-title { font-size: 2rem; font-weight: 900; color: var(--pitch-brand-dark); margin-bottom: 0.5rem; }

            /* Value Grid */
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
                padding: 2rem;
                box-shadow: var(--pitch-shadow-sm);
                width: calc(33.333% - 1rem);
                min-width: 300px;
                display: flex;
                flex-direction: column;
            }

            /* Streamlit Primary Button Override */
            .stButton > button[kind="primary"] {
                border: none !important;
                background: var(--pitch-accent) !important;
                color: #ffffff !important;
                border-radius: 10px !important;
                padding: 0.8rem 2.5rem !important;
                font-size: 1.2rem !important;
                font-weight: 800 !important;
                box-shadow: 0 4px 12px rgba(47, 98, 166, 0.2) !important;
                transition: all 0.2s ease !important;
            }
            .stButton > button[kind="primary"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 16px rgba(47, 98, 166, 0.3) !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- 1. HEADER (Left-Aligned) ---
    logo_b64 = get_img_b64("logo_grey_title.png")
    st.markdown(f"""
        <div class="header-container">
            <div style='background-color: white; border: 4px solid #52606d; padding: 2px; border-radius: 16px; display: flex; align-items: center; justify-content: center; height: 72px; width: 72px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
                <img src='{logo_b64}' style='height: 55px; filter: contrast(1.2) brightness(0.9);'>
            </div>
            <div style='display: block;'>
                <div style='font-size: 2.8rem; font-weight: 800; color: #52606d; line-height: 1;'>CTPredict</div>
                <div style='color: #52606d; font-size: 1.4rem; font-weight: 800; display: flex; align-items: baseline; gap: 15px;'>
                    <span>Late-Stage Clinical Trial Predictive Engine</span>
                    <span style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; border: 1px solid #e2e8f0; padding: 2px 8px; border-radius: 4px;'>demo version</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 2. HERO SECTION (Split Layout) ---
    hero_col1, hero_col2 = st.columns([1.2, 1])

    with hero_col1:
        st.markdown(f"""
            <div class="hero-h1">
                Identify clinical trials at risk<br>
                <span style="color: var(--pitch-accent);">before execution begins.</span>
            </div>
            <div class="hero-p-bullet">
                Predict full completion or early termination in Phase II/III trials from early design-stage information.
            </div>
            <div class="hero-p-bullet">
                Built on publicly available data from 30,000+ late-stage clinical trials, classifying a trial by risk tiers, and reveal trial-specific risk drivers.
            </div>
            <div class="hero-p-bullet" style="margin-bottom: 3rem;">
                Helps test the impact of trial-design modifications before trial initiation through simulation mode.
            </div>
        """, unsafe_allow_html=True)
        st.button("Launch Demo", key="hero_cta_btn", on_click=launch_demo, type="primary")

    with hero_col2:
        screenshot_b64 = get_img_b64("screenshot.png")
        if screenshot_b64:
            st.markdown(f"""
                <div class="pitch-flex-col-img" style="padding: 0.5rem; border-radius: 20px;">
                    <img src="{screenshot_b64}" alt="App Screenshot" style="width: 100%; border-radius: 15px;">
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="pitch-flex-col-img" style="min-height: 350px; color: #94a3b8;">[Visual: screenshot.png]</div>', unsafe_allow_html=True)

    # --- 3. WHY IT MATTERS ---
    st.markdown('<div class="pitch-section-title">Why it matters</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="pitch-dark-card">
            <div class="pitch-h3">Focus attention early, where clinical, financial, and strategic stakes are highest.</div>
            <div class="pitch-p">
                Phase II/III is where scientific ambition meets major financial and strategic stakes. CTPredict helps identify trials most exposed to early-termination risk, helping focus support, review, and key protocol or strategic decisions where they matter most.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 4. HOW IT WORKS ---
    st.markdown('<div class="pitch-section-title" style="color: var(--pitch-accent);">How it works</div>', unsafe_allow_html=True)

    # Row 1: The score
    gauge_b64 = get_img_b64("gauge.png")
    gauge_img_html = f'<img src="{gauge_b64}" alt="Completion Score Gauge" style="width:100%;">' if gauge_b64 else '<div style="color:#94a3b8;">[Visual: Gauge]</div>'

    st.markdown(f"""
        <div class="score-master-grid">
            <div class="score-cell">
                <div class="pitch-card">
                    <div class="pitch-h2-score">The Score</div>
                    <div class="pitch-p" style="margin-bottom: 1.5rem;">
                        Reflects how closely a trial resembles historical patterns of <span class="hl-red">early termination</span> or <span class="hl-blue">full completion</span>, and assigns it to one of four completion-risk tiers:
                    </div>
                    <div style="line-height: 2.2; font-size: 1.1rem;">
                        <div><span class="color-box cb-high"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">High Risk:</span> 0-25 points</div>
                        <div><span class="color-box cb-watch"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">Watchlist:</span> 25-50 points</div>
                        <div><span class="color-box cb-fav"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">Favorable:</span> 50-75 points</div>
                        <div><span class="color-box cb-low"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">Low Risk:</span> 75-100 points</div>
                    </div>
                </div>
            </div>
            <div class="score-cell">
                <div class="pitch-img-box gauge-box" style="border: 1px solid var(--pitch-border); background: white;">
                    {gauge_img_html}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 4rem;'></div>", unsafe_allow_html=True)

    # --- 5. FOUR RISK DIMENSIONS ---
    st.markdown(f"""
        <div class="pitch-flex-row">
            <div class="pitch-flex-col-img">
                {render_image_box("barchart.png", "Impact Bar Chart")}
            </div>
            <div class="pitch-flex-col-text" style="padding-left: 1rem;">
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
    """, unsafe_allow_html=True)

    # --- 6. DRIVERS MAP ---
    st.markdown("<div style='height: 4rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="pitch-flex-row">
            <div class="pitch-flex-col-text">
                <div class="pitch-h2-score" style="margin-top: 0; margin-bottom: 2rem;">Drivers Map</div>
                <div class="pitch-p">
                    Breaks each prediction into <span class="hl-blue" style="background:#e2e8f0; color:var(--pitch-brand-dark);">27 core trial features</span>, distributed across the four main risk dimensions, showing which factors contribute most to the final risk signal.<br><br>
                    Larger blocks indicate stronger impact.
                </div>
            </div>
            <div class="pitch-flex-col-img" style="flex: 2.5;">
                {render_image_box("treemap.png", "Interactive Treemap")}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 7. MODEL AND DATA FOUNDATION ---
    st.markdown('<div class="pitch-section-title">Model and data foundation</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="background: linear-gradient(135deg, var(--pitch-brand-dark), #334155); border-radius: 16px; padding: 4rem 3rem; text-align: center; color: white; margin-bottom: 4rem;">
            <div class="pitch-p" style="color: #e2e8f0; max-width: 850px; margin: 0 auto; font-size: 1.2rem;">
                CTPredict is built on publicly available AACT data from industry-led Phase II/III trials initiated since 2009. Its XGBoost supervised machine-learning model uses 27 design-stage variables to learn patterns of full completion vs. early termination.
            </div>
            <div style="margin-top: 2rem; font-size: 0.95rem; opacity: 0.7;">
                <strong>AACT:</strong> Aggregate Analysis of ClinicalTrials.gov - an analysis-ready public database derived from ClinicalTrials.gov records.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 8. PERFORMANCE METRICS ---
    st.markdown("""
        <div class="pitch-metrics-grid">
            <div class="pitch-metric-card">
                <div class="pitch-metric-title">30,000+</div>
                <div class="pitch-p" style="margin-bottom:0;">Industry-led Phase II/III trials initiated since 2009, labelled for supervised learning.</div>
            </div>
            <div class="pitch-metric-card">
                <div class="pitch-metric-title">78% AUC</div>
                <div class="pitch-p" style="margin-bottom:0;">CTPredict ranks early-terminating trials as higher risk 78% of the time—well above the 50% baseline.</div>
            </div>
            <div class="pitch-metric-card">
                <div class="pitch-metric-title">3 in 4</div>
                <div class="pitch-p" style="margin-bottom:0;">Around 75% of trials that later terminated early fall into the High Risk or Watchlist categories.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 9. WHERE IT ADDS VALUE ---
    st.markdown('<div class="pitch-section-title">Where it adds value</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="pitch-value-grid">
            <div class="pitch-value-card">
                <div class="pitch-h4">Portfolio Management</div>
                <div class="pitch-p" style="margin-bottom:0; font-size: 1.05rem;">Highlight late-stage assets that may require closer scrutiny.</div>
            </div>
            <div class="pitch-value-card">
                <div class="pitch-h4">Therapeutic Area Leadership</div>
                <div class="pitch-p" style="margin-bottom:0; font-size: 1.05rem;">Benchmark completion-risk patterns across indications, modalities, and designs.</div>
            </div>
            <div class="pitch-value-card">
                <div class="pitch-h4">Clinical Development Leads</div>
                <div class="pitch-p" style="margin-bottom:0; font-size: 1.05rem;">Explore in simulation mode how design choices shift the risk profile before initiation.</div>
            </div>
            <div class="pitch-value-card">
                <div class="pitch-h4">Investors & Analysts</div>
                <div class="pitch-p" style="margin-bottom:0; font-size: 1.05rem;">Compare completion-risk profiles across late-stage assets using public data.</div>
            </div>
            <div class="pitch-value-card">
                <div class="pitch-h4">Training & Capability Building</div>
                <div class="pitch-p" style="margin-bottom:0; font-size: 1.05rem;">Use real trials and simulation to support learning and risk-based decision-making.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 10. BOTTOM CTA ---
    st.markdown("<div style='height: 4rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <div class="pitch-h2-full">See the prediction in action.</div>
            <div class="pitch-p" style="max-width: 600px; margin: 0 auto;">Start from a real Phase II/III trial and move from risk tier to score drivers in a few clicks.</div>
        </div>
    """, unsafe_allow_html=True)

    # Centering the button
    c_left, c_mid, c_right = st.columns([1, 1, 1])
    with c_mid:
        st.button("Launch Demo", key="bottom_cta_btn", on_click=launch_demo, type="primary")

    # --- 11. FOOTER ---
    st.markdown(f"""
        <div style="text-align: center; padding: 5rem 0; border-top: 1px solid #e2e8f0; margin-top: 6rem; color: var(--pitch-text-sec);">
            <div class="pitch-h4" style="margin-bottom: 1rem;">Pilot version. Welcoming your ideas.</div>
            <div class="pitch-p" style="max-width: 800px; margin: 0 auto 2rem auto; font-size: 1.05rem;">
                This demo focuses on single-trial exploration. Broader views are available or currently being developed, including sponsor full-portfolio screening and therapeutic-area benchmarking.
            </div>
            <div style="font-weight: 800; color: var(--pitch-brand-dark);">Contact: Nicolas Delaunay</div>
            <div>Email: <a href="mailto:delaunay80@gmail.com" style="color: var(--pitch-accent); text-decoration: none; font-weight: 700;">delaunay80@gmail.com</a></div>
        </div>
    """, unsafe_allow_html=True)
