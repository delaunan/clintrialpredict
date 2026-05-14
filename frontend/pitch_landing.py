import base64
from pathlib import Path
import streamlit as st


# =========================================================================
# Inline SVG used in place of Why_it_matters.png when that file is absent.
# Minimalist, brand-color only, scales perfectly at any size.
# Drop a Why_it_matters.png in the same folder (or ./frontend) and it will
# override this SVG automatically.
# =========================================================================
_WHY_SVG = """
<svg viewBox="0 0 480 320" xmlns="http://www.w3.org/2000/svg"
     preserveAspectRatio="xMidYMid meet" role="img"
     aria-label="Phase II/III is the high-stakes phase; risk is revealed pre-execution">
  <defs>
    <linearGradient id="phaseG" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#89A7C9"/>
      <stop offset="100%" stop-color="#2f62a6"/>
    </linearGradient>
  </defs>

  <text x="240" y="28" text-anchor="middle" fill="#94a3b8"
        font-family="Inter, system-ui, sans-serif" font-size="10"
        font-weight="800" letter-spacing="2.5">TRIAL LIFECYCLE</text>

  <line x1="55" y1="68" x2="425" y2="68" stroke="#e2e8f0" stroke-width="2"/>

  <circle cx="85" cy="68" r="4" fill="#cbd5e1"/>
  <text x="85" y="88" text-anchor="middle" fill="#94a3b8"
        font-family="Inter, system-ui, sans-serif" font-size="9"
        font-weight="700">Preclinical</text>

  <circle cx="160" cy="68" r="4" fill="#cbd5e1"/>
  <text x="160" y="88" text-anchor="middle" fill="#94a3b8"
        font-family="Inter, system-ui, sans-serif" font-size="9"
        font-weight="700">Phase I</text>

  <rect x="215" y="54" width="130" height="28" rx="14" fill="url(#phaseG)"/>
  <text x="280" y="72" text-anchor="middle" fill="white"
        font-family="Inter, system-ui, sans-serif" font-size="11"
        font-weight="800" letter-spacing="1.5">PHASE II / III</text>

  <circle cx="395" cy="68" r="4" fill="#cbd5e1"/>
  <text x="395" y="88" text-anchor="middle" fill="#94a3b8"
        font-family="Inter, system-ui, sans-serif" font-size="9"
        font-weight="700">Approval</text>

  <line x1="280" y1="84" x2="240" y2="118" stroke="#cbd5e1"
        stroke-width="1" stroke-dasharray="3,3"/>

  <rect x="125" y="120" width="230" height="125" rx="14"
        fill="white" stroke="#e2e8f0" stroke-width="1.5"/>

  <g transform="translate(240, 200)">
    <path d="M -55 0 A 55 55 0 0 1 55 0" fill="none"
          stroke="#f1f5f9" stroke-width="13" stroke-linecap="round"/>
    <path d="M -55 0 A 55 55 0 0 1 -24 -49" fill="none"
          stroke="#b03f3f" stroke-width="13" stroke-linecap="round"/>
    <path d="M -24 -49 A 55 55 0 0 1 0 -55" fill="none"
          stroke="#ecb4b4" stroke-width="13" stroke-linecap="round"/>
    <path d="M 0 -55 A 55 55 0 0 1 24 -49" fill="none"
          stroke="#cdd9ec" stroke-width="13" stroke-linecap="round"/>
    <path d="M 24 -49 A 55 55 0 0 1 55 0" fill="none"
          stroke="#2f62a6" stroke-width="13" stroke-linecap="round"/>
    <line x1="0" y1="0" x2="-36" y2="-22"
          stroke="#334155" stroke-width="3" stroke-linecap="round"/>
    <circle cx="0" cy="0" r="4" fill="#334155"/>
  </g>

  <text x="240" y="232" text-anchor="middle" fill="#52606d"
        font-family="Inter, system-ui, sans-serif" font-size="11"
        font-weight="800">Risk revealed pre-execution</text>

  <text x="240" y="285" text-anchor="middle" fill="#94a3b8"
        font-family="Inter, system-ui, sans-serif" font-size="10"
        font-weight="800" letter-spacing="2.5">HIGH STAKES</text>

  <g font-family="Inter, system-ui, sans-serif" font-size="11"
     font-weight="700" fill="#64748b">
    <text x="140" y="307" text-anchor="middle">Clinical</text>
    <text x="240" y="307" text-anchor="middle">Financial</text>
    <text x="340" y="307" text-anchor="middle">Strategic</text>
  </g>
  <circle cx="187" cy="303" r="1.5" fill="#cbd5e1"/>
  <circle cx="290" cy="303" r="1.5" fill="#cbd5e1"/>
</svg>
"""


def render_pitch_page():
    """
    Renders a polished, responsive landing page for CTPredict.

    Section flow:
      1. Header
      2. Hero — full-width headline, then 3 short sentences (left) facing
                the screenshot (right)
      3. CTA card — see it score a real trial
      4. What's at stake     (was "Why it matters")
      5. Inside the engine   (was "How it works") — Score / Four Risk
         Dimensions / Drivers Map
      6. Does it hold up?    (was "How strong is the prediction?")
         - What powers the engine?     (foundation banner)
         - How well does it predict?   (metric cards)
      7. Where it pays off   (was "Where it adds value")
      8. CTA card — curious how a real trial would score
      9. Footer
    """

    # ---------- ASSET LOADING ----------
    assets_dir = Path(__file__).resolve().parent

    def get_img_b64(filename):
        paths_to_check = [
            assets_dir / filename,
            assets_dir / "frontend" / filename,
            Path("./frontend") / filename,
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

    # ---------- CALLBACK ----------
    def launch_demo():
        st.session_state["pitch_seen"] = True
        st.session_state["selected_nct_id"] = None
        st.rerun()

    # ---------- CSS ----------
    st.markdown("""
        <style>
            /* ====== BASE ====== */
            :root {
                --pitch-bg: #f1f5f9;
                --pitch-text-main: #334155;
                --pitch-text-sec: #64748b;
                --pitch-text-soft: #94a3b8;
                --pitch-brand-dark: #52606d;
                --pitch-brand-deep: #334155;
                --pitch-accent: #89A7C9;
                --pitch-accent-hover: #7a96b5;
                --pitch-accent-soft: #e8eef7;
                --pitch-deep-blue: #2f62a6;
                --pitch-red: #b03f3f;
                --pitch-border: #e2e8f0;
                --pitch-card-bg: #ffffff;
                --pitch-radius: 16px;
                --pitch-radius-sm: 12px;
                --pitch-shadow-sm: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
                --pitch-shadow-md: 0 10px 25px -5px rgba(0,0,0,0.08), 0 8px 10px -6px rgba(0,0,0,0.04);
                --pitch-shadow-lg: 0 25px 50px -12px rgba(0,0,0,0.15);
            }

            .block-container {
                padding-top: 2rem !important;
                padding-bottom: 4rem !important;
                max-width: 1200px !important;
            }

            /* ====== HEADER (smaller, more compact) ====== */
            .header-container {
                display: flex;
                align-items: center;
                gap: 14px;
                justify-content: flex-start;
                margin-bottom: 4rem;
                text-align: left;
            }
            .header-logo-box {
                background-color: white;
                border: 3px solid var(--pitch-brand-dark);
                padding: 2px;
                border-radius: 14px;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 60px;
                width: 60px;
                flex-shrink: 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }
            .header-logo-box img { height: 50px; }
            .header-title {
                font-size: 2.2rem;
                font-weight: 800;
                color: var(--pitch-brand-dark);
                line-height: 1;
                letter-spacing: -0.01em;
            }
            .header-subtitle {
                color: var(--pitch-brand-dark);
                font-size: 1.15rem;
                font-weight: 800;
                display: flex;
                align-items: baseline;
                gap: 12px;
                margin-top: 4px;
            }
            .header-badge {
                font-size: 0.65rem;
                color: var(--pitch-text-soft);
                text-transform: uppercase;
                border: 1px solid var(--pitch-border);
                padding: 2px 8px;
                border-radius: 4px;
                letter-spacing: 0.08em;
                font-weight: 700;
            }

            /* ====== TYPOGRAPHY ====== */
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
                letter-spacing: -0.005em;
            }
            .pitch-section-title::before,
            .pitch-section-title::after {
                content: "";
                flex: 1;
                border-bottom: 2px solid var(--pitch-border);
            }
            .pitch-section-title::before { margin-right: 1.5rem; }
            .pitch-section-title::after  { margin-left: 1.5rem; }

            .pitch-sub-question {
                font-family: 'Inter', sans-serif;
                font-size: clamp(1.25rem, 1.8vw, 1.5rem);
                font-weight: 700;
                color: var(--pitch-brand-dark);
                text-align: center;
                margin: 3rem 0 1.75rem 0;
                letter-spacing: -0.005em;
            }
            .pitch-sub-question::before {
                content: "";
                display: block;
                width: 40px;
                height: 3px;
                background: var(--pitch-accent);
                border-radius: 2px;
                margin: 0 auto 1rem auto;
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
                line-height: 1.3;
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

            /* ====== HERO ====== */
            .hero-headline-wrap {
                margin-bottom: 2.5rem;
            }
            /* Reduced size + no trailing period (more modern, less imposing) */
            .hero-h1 {
                font-family: 'Inter', sans-serif;
                font-size: clamp(2.4rem, 4.5vw, 3.5rem);
                font-weight: 900;
                color: var(--pitch-brand-dark);
                line-height: 1.08;
                letter-spacing: -0.025em;
                margin: 0;
            }
            .hero-h1 .accent-line {
                color: var(--pitch-accent);
                display: block;
            }

            /* Stretch + center-align inside columns → text and screenshot
               share the same top and bottom edges */
            .hero-body {
                display: flex;
                gap: 3rem;
                align-items: stretch;
                margin-bottom: 0;
            }
            .hero-body-text {
                flex: 1;
                min-width: 0;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .hero-body-img {
                flex: 1.4;
                min-width: 0;
                display: flex;
                align-items: center;
            }
            .hero-p {
                font-family: 'Inter', sans-serif;
                font-size: 1.1rem;
                color: var(--pitch-text-sec);
                line-height: 1.6;
                margin-bottom: 1.2rem;
            }
            .hero-p:last-child { margin-bottom: 0; }

            /* The hero highlight pills inherit hl-blue / hl-red but use a
               slightly tighter pad so they sit naturally inside the sentence */
            .hero-p .hl-blue,
            .hero-p .hl-red {
                padding: 1px 7px;
                font-weight: 700;
            }

            .hero-screenshot-frame {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: 22px;
                padding: 0.7rem;
                box-shadow: var(--pitch-shadow-lg);
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                width: 100%;
            }
            .hero-screenshot-frame img {
                width: 100%;
                height: auto;
                display: block;
                border-radius: 14px;
                object-fit: contain;
            }
            .hero-screenshot-frame.placeholder {
                min-height: 320px;
                color: var(--pitch-text-soft);
                font-weight: 600;
            }

            @media (max-width: 900px) {
                .hero-body { flex-direction: column; gap: 2.5rem; align-items: stretch; }
            }

            /* ====== WIDE CTA CARDS ====== */
            .st-key-cta_wide_top, .st-key-cta_wide_bottom {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                box-shadow: var(--pitch-shadow-lg);
                padding: 2rem 2.75rem !important;
                margin: 3.5rem 0 5rem 0 !important;
                position: relative;
                overflow: hidden;
            }
            .st-key-cta_wide_top::before, .st-key-cta_wide_bottom::before {
                content: "";
                position: absolute;
                top: 0; left: 0; right: 0;
                height: 4px;
                background: linear-gradient(90deg, var(--pitch-accent) 0%, var(--pitch-deep-blue) 100%);
            }

            .cta-wide-title {
                font-family: 'Inter', sans-serif;
                font-size: clamp(1.35rem, 2.1vw, 1.65rem);
                font-weight: 800;
                color: var(--pitch-brand-dark);
                margin-bottom: 0.35rem;
                line-height: 1.25;
                letter-spacing: -0.01em;
            }
            .cta-wide-subtitle {
                font-size: 1.02rem;
                color: var(--pitch-text-sec);
                line-height: 1.55;
                margin: 0;
            }

            .st-key-cta_wide_top  div[data-testid="stColumn"]:last-child > div,
            .st-key-cta_wide_bottom div[data-testid="stColumn"]:last-child > div {
                display: flex;
                align-items: center;
                justify-content: flex-end;
                height: 100%;
            }
            .st-key-cta_wide_top .stButton > button[kind="primary"],
            .st-key-cta_wide_bottom .stButton > button[kind="primary"] {
                padding: 1rem 2.75rem !important;
                font-size: 1.1rem !important;
                width: auto !important;
                max-width: none !important;
                min-width: 200px !important;
                margin: 0 !important;
            }
            .st-key-cta_wide_top .stButton > button[kind="primary"] p,
            .st-key-cta_wide_bottom .stButton > button[kind="primary"] p {
                font-size: 1.1rem !important;
                font-weight: 800 !important;
                white-space: nowrap !important;
            }
            @media (max-width: 768px) {
                .st-key-cta_wide_top, .st-key-cta_wide_bottom {
                    padding: 2rem !important;
                    text-align: center;
                }
                .st-key-cta_wide_top  div[data-testid="stColumn"]:last-child > div,
                .st-key-cta_wide_bottom div[data-testid="stColumn"]:last-child > div {
                    justify-content: center;
                    margin-top: 1.25rem;
                }
            }

            /* ====== WHAT'S AT STAKE — image + dark message card ====== */
            .why-grid {
                display: flex;
                gap: 2.5rem;
                align-items: stretch;
                margin-bottom: 4rem;
            }
            .why-image-col {
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .why-image-frame {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                padding: 1.25rem;
                box-shadow: var(--pitch-shadow-md);
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }
            .why-image-frame img,
            .why-image-frame svg {
                width: 100%;
                height: auto;
                display: block;
                max-height: 360px;
            }
            .why-image-frame img { border-radius: 10px; object-fit: contain; }
            .why-text-col {
                flex: 1.15;
                display: flex;
                align-items: stretch;
            }
            .why-dark-card {
                background: var(--pitch-brand-dark);
                border-radius: var(--pitch-radius);
                padding: 2.75rem 2.5rem;
                box-shadow: var(--pitch-shadow-md);
                display: flex;
                flex-direction: column;
                justify-content: center;
                width: 100%;
            }
            .why-dark-card .pitch-h3 {
                color: #ffffff;
                font-size: clamp(1.4rem, 2.3vw, 1.75rem);
                margin-bottom: 1.25rem;
                line-height: 1.25;
            }
            .why-dark-card .pitch-p {
                color: #f8fafc;
                font-size: 1.1rem;
                margin-bottom: 0;
            }
            @media (max-width: 900px) {
                .why-grid { flex-direction: column; }
                .why-image-frame { min-height: 260px; }
            }

            /* ====== INSIDE THE ENGINE rows ====== */
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
            }
            .pitch-full-width-text {
                width: 100%;
                margin-bottom: 4rem;
            }
            @media (max-width: 768px) {
                .pitch-flex-row { flex-direction: column; }
                .pitch-flex-col-img { min-height: 300px; }
            }

            /* ---- Score subsection ---- */
            .score-master-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                align-items: stretch;
            }
            @media (max-width: 768px) {
                .score-master-grid { grid-template-columns: 1fr; }
            }
            .pitch-h2-score {
                font-family: 'Inter', sans-serif;
                font-size: clamp(1.6rem, 2.6vw, 2rem);
                font-weight: 800;
                color: var(--pitch-brand-dark);
                margin-top: 0;
                margin-bottom: 1.25rem;
                letter-spacing: -0.01em;
                line-height: 1.2;
            }
            .color-box {
                width: 12px; height: 12px;
                border-radius: 3px;
                display: inline-block;
                margin-right: 8px;
                vertical-align: middle;
            }
            .cb-low   { background: linear-gradient(90deg, rgb(162,198,228) 0%, rgb(47,98,166)  100%); }
            .cb-fav   { background: linear-gradient(90deg, rgb(242,244,248) 0%, rgb(162,198,228) 100%); }
            .cb-watch { background: linear-gradient(90deg, rgb(236,162,162) 0%, rgb(242,244,248) 100%); }
            .cb-high  { background: linear-gradient(90deg, rgb(176,63,63)   0%, rgb(236,162,162) 100%); }

            .pitch-img-box.gauge-box {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 240px;
                padding: 1rem 1.5rem;
                overflow: hidden;
                width: 100%;
                height: 100%;
                border-radius: var(--pitch-radius-sm);
                box-shadow: var(--pitch-shadow-sm);
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
            }
            .pitch-img-box.gauge-box img {
                transform: scale(1.35);
                transform-origin: center center;
            }

            .hl-blue { color: var(--pitch-deep-blue); background: rgba(47,98,166,0.1); padding: 2px 8px; border-radius: 6px; }
            .hl-red  { color: var(--pitch-red);       background: rgba(176,63,63,0.1); padding: 2px 8px; border-radius: 6px; }
            .hl-grey { color: var(--pitch-brand-dark); background: #e2e8f0;            padding: 2px 8px; border-radius: 6px; font-weight: 700; }

            .interpret-card {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius-sm);
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

            /* Dimension items */
            .dim-item   { margin-bottom: 1.2rem; }
            .dim-title  { font-weight: 800; font-size: 1.25rem; color: var(--pitch-text-sec); margin-bottom: 0.1rem; }
            .dim-desc   { font-size: 1.15rem; color: var(--pitch-text-sec); }

            /* ====== DOES IT HOLD UP? — foundation + metrics ====== */
            .pitch-foundation-banner {
                background: linear-gradient(135deg, var(--pitch-brand-dark), var(--pitch-brand-deep));
                border-radius: var(--pitch-radius);
                padding: 2.75rem 3rem;
                text-align: center;
                box-shadow: var(--pitch-shadow-md);
                margin-bottom: 4rem;
                color: white;
            }
            .pitch-foundation-banner .pitch-p {
                color: #e2e8f0;
                max-width: 850px;
                margin: 0 auto 1.25rem auto;
                font-size: 1.1rem;
            }
            .pitch-foundation-banner .pitch-p:last-child { margin-bottom: 0; }

            .pitch-metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.5rem;
                margin-bottom: 4rem;
            }
            .pitch-metric-card {
                background: var(--pitch-card-bg);
                border-radius: var(--pitch-radius);
                padding: 2.25rem 2rem;
                box-shadow: var(--pitch-shadow-sm);
                border: 1px solid var(--pitch-border);
                border-top: 4px solid var(--pitch-accent);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .pitch-metric-card:hover {
                transform: translateY(-3px);
                box-shadow: var(--pitch-shadow-md);
            }
            .pitch-metric-title {
                font-size: 1.4rem;
                font-weight: 900;
                color: var(--pitch-brand-dark);
                margin-bottom: 0.75rem;
                letter-spacing: -0.01em;
            }

            /* ====== WHERE IT PAYS OFF — horizontal cards ====== */
            .pitch-value-grid-v2 {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1.25rem;
                margin-bottom: 2rem;
            }
            .pitch-value-grid-v2 > .pitch-value-card-h:nth-child(5) {
                grid-column: 1 / -1;
                max-width: calc(50% - 0.625rem);
                margin: 0 auto;
                width: 100%;
            }
            @media (max-width: 900px) {
                .pitch-value-grid-v2 { grid-template-columns: 1fr; }
                .pitch-value-grid-v2 > .pitch-value-card-h:nth-child(5) {
                    max-width: 100%;
                    grid-column: auto;
                }
            }

            .pitch-value-card-h {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                padding: 1.5rem 1.75rem;
                box-shadow: var(--pitch-shadow-sm);
                display: flex;
                flex-direction: row;
                align-items: flex-start;
                gap: 1.5rem;
                transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            }
            .pitch-value-card-h:hover {
                transform: translateY(-3px);
                box-shadow: var(--pitch-shadow-md);
                border-color: var(--pitch-accent);
            }
            .pitch-value-icon-wrap {
                flex-shrink: 0;
                width: 76px;
                height: 76px;
                background: linear-gradient(135deg, var(--pitch-accent-soft) 0%, #d4e0f0 100%);
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 12px rgba(137,167,201,0.18);
            }
            .pitch-value-icon-wrap img {
                width: 54px;
                height: 54px;
                object-fit: contain;
                display: block;
            }
            .pitch-value-icon-wrap.placeholder { background: #e2e8f0; }
            .pitch-value-content {
                flex: 1;
                min-width: 0;
            }
            .pitch-value-content .pitch-h4 {
                margin-bottom: 0.4rem;
                font-size: 1.15rem;
            }
            .pitch-value-content .pitch-p {
                font-size: 1rem;
                margin-bottom: 0;
                line-height: 1.55;
            }

            /* ====== FOOTER ====== */
            .pitch-footer {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                padding: 3rem;
                text-align: center;
                box-shadow: var(--pitch-shadow-sm);
                margin-bottom: 2rem;
            }

            /* ====== PRIMARY BUTTON ====== */
            .stButton > button[kind="primary"] {
                border: none !important;
                background: var(--pitch-accent) !important;
                color: #ffffff !important;
                border-radius: 10px !important;
                padding: 1rem 2.5rem !important;
                font-size: 1.1rem !important;
                font-weight: 800 !important;
                box-shadow: 0 4px 12px rgba(137,167,201,0.3) !important;
                transition: all 0.2s ease !important;
            }
            .stButton > button[kind="primary"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 16px rgba(137,167,201,0.4) !important;
                background: var(--pitch-accent-hover) !important;
            }
            .stButton > button[kind="primary"] p {
                font-size: 1.1rem !important;
                font-weight: 800 !important;
                white-space: nowrap !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # ====================================================================
    # 1. HEADER
    # ====================================================================
    logo_b64 = get_img_b64("logo_grey_title.png")
    brand_filter = "contrast(1.5) brightness(0.9) grayscale(100%) sepia(100%) hue-rotate(180deg) saturate(0.8) brightness(0.85) contrast(1.2)"

    st.markdown(f"""
        <div class="header-container">
            <div class="header-logo-box">
                <img src='{logo_b64}' style='filter: {brand_filter};'>
            </div>
            <div>
                <div class="header-title">CTPredict</div>
                <div class="header-subtitle">
                    <span style='line-height: 1;'>Late-Stage Clinical Trial Predictive Engine</span>
                    <span class="header-badge">demo version</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ====================================================================
    # 2. HERO — full-width headline + (sentences left, screenshot right)
    # ====================================================================
    screenshot_b64 = get_img_b64("screenshot.png")
    screenshot_html = (
        f'<div class="hero-screenshot-frame"><img src="{screenshot_b64}" alt="CTPredict application screenshot"></div>'
        if screenshot_b64
        else '<div class="hero-screenshot-frame placeholder"><span>Visual: screenshot.png</span></div>'
    )

    st.markdown(f"""
        <div class="hero-headline-wrap">
            <div class="hero-h1">
                Identify clinical trials at risk
                <span class="accent-line">before execution begins</span>
            </div>
        </div>
        <div class="hero-body">
            <div class="hero-body-text">
                <div class="hero-p">
                    Predicts <span class="hl-blue">full completion</span> or <span class="hl-red">early termination</span> in Phase II/III trials from early design-stage information.
                </div>
                <div class="hero-p">
                    Built on publicly available data from 30,000+ past trials, classifying trials by risk tier and revealing trial-specific risk drivers.
                </div>
                <div class="hero-p">
                    Helps assess the impact of trial-design modifications through simulation mode.
                </div>
            </div>
            <div class="hero-body-img">
                {screenshot_html}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ====================================================================
    # 3. WIDE CTA CARD #1
    # ====================================================================
    with st.container(key="cta_wide_top"):
        c1, c2 = st.columns([2.2, 1], gap="large")
        with c1:
            st.markdown("""
                <div class="cta-wide-title">See it score a real trial.</div>
                <div class="cta-wide-subtitle">Pick any Phase II/III trial — from risk tier to score drivers in a few clicks.</div>
            """, unsafe_allow_html=True)
        with c2:
            st.button("Launch Demo", key="cta_btn_top", on_click=launch_demo, type="primary")

    # ====================================================================
    # 4. WHAT'S AT STAKE  (was "Why it matters")
    # ====================================================================
    why_b64 = get_img_b64("Why_it_matters.png")
    if why_b64:
        why_visual_html = f'<div class="why-image-frame"><img src="{why_b64}" alt="What is at stake"></div>'
    else:
        # Fallback to the inline brand-colored SVG
        why_visual_html = f'<div class="why-image-frame">{_WHY_SVG}</div>'

    st.markdown('<div class="pitch-section-title">What\'s at stake</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="why-grid">
            <div class="why-image-col">
                {why_visual_html}
            </div>
            <div class="why-text-col">
                <div class="why-dark-card">
                    <div class="pitch-h3">Focus attention early, where clinical, financial, and strategic stakes are highest.</div>
                    <div class="pitch-p">
                        Phase II/III is where scientific ambition meets major financial and strategic stakes. CTPredict helps identify trials most exposed to early-termination risk, helping focus support, review, and key protocol or strategic decisions where they matter most.
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ====================================================================
    # 5. INSIDE THE ENGINE  (was "How it works")
    # ====================================================================
    st.markdown('<div class="pitch-section-title">Inside the engine</div>', unsafe_allow_html=True)

    gauge_b64 = get_img_b64("gauge.png")
    gauge_img_html = f'<img src="{gauge_b64}" alt="Completion Score Gauge">' if gauge_b64 else '<div class="pitch-placeholder"><span>Visual: Completion Score Gauge</span></div>'

    st.markdown(f"""
        <div class="score-master-grid">
            <div class="score-cell">
                <div class="pitch-h2-score">The Score</div>
                <div class="pitch-p" style="margin-bottom: 0;">
                    <span style="color: var(--pitch-text-sec);">
                        Reflects how closely a trial resembles historical patterns of <span class="hl-red">early termination</span> or <span class="hl-blue">full completion</span>, and assigns it to one of four completion-risk tiers:
                    </span>
                    <br><br>
                    <div style="margin-bottom: 8px;"><span class="color-box cb-high"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">High Risk:</span> 0-25 points</div>
                    <div style="margin-bottom: 8px;"><span class="color-box cb-watch"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">Watchlist:</span> 25-50 points</div>
                    <div style="margin-bottom: 8px;"><span class="color-box cb-fav"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">Favorable:</span> 50-75 points</div>
                    <div style="margin-bottom: 8px;"><span class="color-box cb-low"></span><span style="font-weight: 700; color: var(--pitch-text-sec);">Low Risk:</span> 75-100 points</div>
                </div>
            </div>
            <div class="score-cell">
                <div class="pitch-img-box gauge-box">{gauge_img_html}</div>
            </div>
            <div class="score-cell">
                <div class="interpret-card">
                    <div class="interpret-card-title" style="color: var(--pitch-red);"><span class="color-box cb-high"></span> Red reflects a riskier trial profile.</div>
                    <div class="pitch-p" style="margin-bottom: 0;">May flag design choices worth challenging - or capture ambition: novel science, rigorous endpoints, flexible design, or higher complexity. Higher risk can also point to higher potential value.</div>
                </div>
            </div>
            <div class="score-cell">
                <div class="interpret-card">
                    <div class="interpret-card-title" style="color: var(--pitch-deep-blue);"><span class="color-box cb-low"></span> Blue signals higher likelihood of full completion.</div>
                    <div class="pitch-p" style="margin-bottom: 0;">May signal strong execution conditions, high-quality design, or a more established scientific profile with lower complexity. Full completion should still be read alongside scientific and asset value.</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 4rem;'></div>", unsafe_allow_html=True)

    # Four Risk Dimensions
    st.markdown(f"""
        <div class="pitch-flex-row">
            <div class="pitch-flex-col-img" style="flex: 1.7;">
                {render_image_box("barchart.png", "Impact Bar Chart")}
            </div>
            <div class="pitch-flex-col-text" style="flex: 1.3;">
                <div class="pitch-h2-score" style="margin-top: 0; margin-bottom: 1rem;">Four Risk Dimensions</div>
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

    # Drivers Map
    st.markdown(f"""
        <div class="pitch-flex-row">
            <div class="pitch-flex-col-text">
                <div class="pitch-h2-score" style="margin-top: 0; margin-bottom: 2rem;">Drivers Map</div>
                <div class="pitch-p">
                    Breaks each prediction into <span class="hl-grey">27 core trial features</span>, distributed across the four main risk dimensions, showing which factors contribute most to the final risk signal.<br><br>
                    Larger blocks indicate stronger impact.
                </div>
            </div>
            <div class="pitch-flex-col-img" style="flex: 3; padding: 0.5rem;">
                {render_image_box("treemap.png", "Interactive Treemap")}
            </div>
        </div>
        <div class="pitch-full-width-text"></div>
    """, unsafe_allow_html=True)

    # ====================================================================
    # 6. DOES IT HOLD UP?  (was "How strong is the prediction?")
    # ====================================================================
    st.markdown('<div class="pitch-section-title">Does it hold up?</div>', unsafe_allow_html=True)

    # Sub-question 1
    st.markdown('<div class="pitch-sub-question">What powers the engine?</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="pitch-foundation-banner">
            <div class="pitch-p">
                CTPredict is built on publicly available AACT data from industry-led Phase II/III trials initiated since 2009. Its XGBoost supervised machine-learning model uses 27 design-stage variables to learn patterns of full completion vs. early termination.
            </div>
            <div class="pitch-p" style="font-size: 0.95rem; opacity: 0.85;">
                <strong>AACT:</strong> Aggregate Analysis of ClinicalTrials.gov - an analysis-ready public database derived from ClinicalTrials.gov records.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sub-question 2 — AUC card description updated for reversed dashboard logic
    st.markdown('<div class="pitch-sub-question">How well does it predict?</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="pitch-metrics-grid">
            <div class="pitch-metric-card">
                <div class="pitch-metric-title">30,000+ trials</div>
                <div class="pitch-p" style="margin-bottom:0;">Industry-led Phase II/III trials initiated since 2009, labelled for supervised learning as full completion vs. early termination.</div>
            </div>
            <div class="pitch-metric-card">
                <div class="pitch-metric-title">78% AUC</div>
                <div class="pitch-p" style="margin-bottom:0;">78% of the time, CTPredict assigns a lower completion score to trials that later terminate early - well above the 50% chance baseline, using only publicly available Phase II/III trial data.</div>
            </div>
            <div class="pitch-metric-card">
                <div class="pitch-metric-title">3 in 4 early terminations flagged early</div>
                <div class="pitch-p" style="margin-bottom:0;">Around 75% of trials that later terminated early fall into the High Risk or Watchlist categories - below 50 points.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ====================================================================
    # 7. WHERE IT PAYS OFF  (was "Where it adds value")
    # ====================================================================
    st.markdown('<div class="pitch-section-title">Where it pays off</div>', unsafe_allow_html=True)
    st.markdown('<div class="pitch-h3" style="text-align: center; margin: -1.5rem 0 3rem 0; color: var(--pitch-text-sec); font-weight: 700;">One predictive engine. Multiple decision perspectives.</div>', unsafe_allow_html=True)

    def get_value_icon_html(filename):
        b64 = get_img_b64(filename)
        if b64:
            return f'<div class="pitch-value-icon-wrap"><img src="{b64}" alt=""></div>'
        return '<div class="pitch-value-icon-wrap placeholder"></div>'

    st.markdown(f"""
        <div class="pitch-value-grid-v2">
            <div class="pitch-value-card-h">
                {get_value_icon_html("icon_portfolio_mgt.png")}
                <div class="pitch-value-content">
                    <div class="pitch-h4">Portfolio Management</div>
                    <div class="pitch-p">Highlight late-stage assets that may require closer scrutiny.</div>
                </div>
            </div>
            <div class="pitch-value-card-h">
                {get_value_icon_html("icon_ta_lead.png")}
                <div class="pitch-value-content">
                    <div class="pitch-h4">Therapeutic Area Leadership</div>
                    <div class="pitch-p">Benchmark completion-risk patterns across indications, modalities, phases, and trial designs within a therapeutic area.</div>
                </div>
            </div>
            <div class="pitch-value-card-h">
                {get_value_icon_html("icon_clin_lead.png")}
                <div class="pitch-value-content">
                    <div class="pitch-h4">Clinical Development Leads</div>
                    <div class="pitch-p">Explore in simulation mode how design choices may shift the completion-risk profile before trial initiation.</div>
                </div>
            </div>
            <div class="pitch-value-card-h">
                {get_value_icon_html("icon_investor.png")}
                <div class="pitch-value-content">
                    <div class="pitch-h4">Investors & Analysts</div>
                    <div class="pitch-p">Compare completion-risk profiles across late-stage assets in the industry using public data.</div>
                </div>
            </div>
            <div class="pitch-value-card-h">
                {get_value_icon_html("icon_training.png")}
                <div class="pitch-value-content">
                    <div class="pitch-h4">Training & Capability Building</div>
                    <div class="pitch-p">Use real trials and simulation mode to support learning and strengthen risk-based decision-making.</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ====================================================================
    # 8. WIDE CTA CARD #2  (different wording from #1)
    # ====================================================================
    with st.container(key="cta_wide_bottom"):
        c1, c2 = st.columns([2.2, 1], gap="large")
        with c1:
            st.markdown("""
                <div class="cta-wide-title">Curious how a real trial would score?</div>
                <div class="cta-wide-subtitle">Pick any Phase II/III trial and explore its risk tier, score, and drivers.</div>
            """, unsafe_allow_html=True)
        with c2:
            st.button("Launch Demo", key="cta_btn_bottom", on_click=launch_demo, type="primary")

    # ====================================================================
    # 9. FOOTER
    # ====================================================================
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
