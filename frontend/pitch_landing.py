import base64
import time
from contextlib import contextmanager
from pathlib import Path
import streamlit as st


# =========================================================================
# Inline SVG: "What's at stake" visual (Venn — Clinical / Financial /
# Strategic converging on Phase II/III). Used when Why_it_matters.png is
# absent. Drop a Why_it_matters.png in the folder (or ./frontend) to override.
# =========================================================================
_STAKE_SVG = """
<svg viewBox="0 0 420 340" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Clinical, financial and strategic stakes converge at Phase II/III">
  <circle cx="210" cy="135" r="90" fill="#89A7C9" fill-opacity="0.14" stroke="#89A7C9" stroke-width="2"/>
  <circle cx="160" cy="218" r="90" fill="#2f62a6" fill-opacity="0.11" stroke="#2f62a6" stroke-width="2"/>
  <circle cx="260" cy="218" r="90" fill="#52606d" fill-opacity="0.11" stroke="#52606d" stroke-width="2"/>
  <text x="210" y="102" text-anchor="middle" fill="#2f62a6"
        font-family="Inter, system-ui, sans-serif" font-size="14" font-weight="800">Clinical</text>
  <text x="134" y="250" text-anchor="middle" fill="#2f62a6"
        font-family="Inter, system-ui, sans-serif" font-size="14" font-weight="800">Financial</text>
  <text x="286" y="250" text-anchor="middle" fill="#52606d"
        font-family="Inter, system-ui, sans-serif" font-size="14" font-weight="800">Strategic</text>
  <rect x="164" y="174" width="92" height="28" rx="14" fill="#334155"/>
  <text x="210" y="192" text-anchor="middle" fill="white"
        font-family="Inter, system-ui, sans-serif" font-size="10.5"
        font-weight="800" letter-spacing="1">PHASE II / III</text>
</svg>
"""

# =========================================================================
# Inline SVG line icons for the "Where it pays off" cards.
# Minimalist, single consistent stroke weight, inherit color via currentColor.
# Each can be overridden by a PNG of the matching name in the assets folder.
# =========================================================================
_VALUE_ICONS = {
    # Portfolio Management -> stacked layers (a portfolio of assets)
    "portfolio": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2.5 2.5 7 12 11.5 21.5 7z"/>
        <path d="M2.5 12 12 16.5 21.5 12"/>
        <path d="M2.5 17 12 21.5 21.5 17"/></svg>""",
    # Therapeutic Area Leadership -> compass (navigating a whole area)
    "ta": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="9.3"/>
        <polygon points="12 6 14.2 12 12 18 9.8 12"/></svg>""",
    # Clinical Development Leads -> conical flask (clinical / lab)
    "clinical": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 2.5h6"/>
        <path d="M10 2.5v6.3L4.6 17.4A1.6 1.6 0 0 0 6 20h12a1.6 1.6 0 0 0 1.4-2.6L14 8.8V2.5"/>
        <path d="M7.4 14.6h9.2"/></svg>""",
    # Investors & Analysts -> trending chart
    "investor": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 3v17a1 1 0 0 0 1 1h17"/>
        <path d="M7 14l3.6-3.6 3 3 5.4-6"/>
        <path d="M15.6 7.4h3.8v3.8"/></svg>""",
    # Training & Capability Building -> graduation cap
    "training": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 4 2 8.5 12 13l10-4.5z"/>
        <path d="M6 10.6V16c0 1.4 2.7 2.8 6 2.8s6-1.4 6-2.8v-5.4"/>
        <path d="M22 8.5v5"/></svg>""",
}


def render_pitch_page(audit_log=None):
    """
    Polished, responsive landing page for CTPredict.

    Parameters
    ----------
    audit_log : callable, optional
        The audit_log(event, **fields) function from app.py. Passed in by
        the caller to avoid a circular import (app.py imports this module).
        When provided, the landing page logs two events:
          * landing_page_shown  — once, when the page first renders
          * landing_demo_click  — when the "Access Demo" CTA is clicked,
            including seconds_on_landing (dwell time) and which button.
        When None, all audit calls are no-ops, so this file still runs
        standalone.

    Section flow:
      Header
      Hero            — compact headline + 3 sentences facing the screenshot
      CTA card        — Access Demo (with shine)
      What's at stake / Inside the engine / How CTPredict performs /
      Where it brings value
      CTA card        — Access Demo (with shine)
      Footer
    """

    # ---------- AUDIT ----------
    # Reuse app.py's audit_log (passed in) so events share the same
    # visitor_id / session_id / JSON format and land in Cloud Logging.
    # A no-op fallback keeps the module importable and runnable on its own.
    def _audit(event, **fields):
        if callable(audit_log):
            try:
                audit_log(event, **fields)
            except Exception:
                # Audit must never break the page render.
                pass

    # Record when the landing page was first shown this session, and emit a
    # one-time "shown" event. Streamlit reruns this function on every
    # interaction, so both are guarded to fire only once per session.
    if "_landing_shown_ts" not in st.session_state:
        st.session_state["_landing_shown_ts"] = time.time()
    if not st.session_state.get("_landing_shown_logged", False):
        _audit("landing_page_shown")
        st.session_state["_landing_shown_logged"] = True

    # ---------- ASSETS ----------
    assets_dir = Path(__file__).resolve().parent

    def get_img_b64(filename):
        for path in (assets_dir / filename,
                     assets_dir / "frontend" / filename,
                     Path("./frontend") / filename):
            if path.exists():
                with open(path, "rb") as f:
                    return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        return None

    def media_img(filename, alt_text):
        """Image for a split-card media panel, with a graceful placeholder."""
        b64 = get_img_b64(filename)
        if b64:
            return f'<img src="{b64}" alt="{alt_text}">'
        return f'<div class="media-placeholder"><span>Visual: {alt_text}</span></div>'

    def value_icon(key, png_name):
        """PNG override if present, otherwise the inline SVG line icon."""
        b64 = get_img_b64(png_name)
        if b64:
            return f'<img src="{b64}" alt="">'
        return _VALUE_ICONS[key]

    # ---------- CALLBACK ----------
    def launch_demo(source="unknown"):
        # on_click callback: just set state. Streamlit automatically reruns
        # after a callback returns, so calling st.rerun() here would be a
        # no-op (and logs a warning). app.py's main() re-checks pitch_seen
        # on that automatic rerun and moves past the landing page.
        #
        # Audit: record how long the visitor spent on the landing page
        # before clicking, and which CTA button they used.
        shown_ts = st.session_state.get("_landing_shown_ts")
        seconds_on_landing = (
            round(time.time() - shown_ts, 1) if shown_ts else None
        )
        _audit(
            "landing_demo_click",
            source=source,
            seconds_on_landing=seconds_on_landing,
        )

        st.session_state["pitch_seen"] = True
        st.session_state["selected_nct_id"] = None

    # ---------- HELPERS ----------
    def section_head(title, subtitle=None):
        # Header = title flanked by fading side rules (+ optional subtitle).
        # The flanking rules are the only frame element — the old underline
        # was removed to keep the side rules as the single accent.
        sub = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''
        st.markdown(f"""
            <div class="section-head">
                <div class="section-title-row">
                    <span class="section-rule left"></span>
                    <div class="section-title">{title}</div>
                    <span class="section-rule right"></span>
                </div>
                {sub}
            </div>
        """, unsafe_allow_html=True)

    def sub_question(text):
        st.markdown(f'<div class="sub-question">{text}</div>', unsafe_allow_html=True)

    @contextmanager
    def hover_lift(key):
        """
        Wrap a visual in a container that gets a subtle hover lift (pure CSS,
        see .st-key-lift_ rules). No button, no click behaviour — just the
        hover affordance. (An earlier invisible-button overlay was removed at
        the user's request — they preferred only the CTA buttons to be
        clickable.)
        """
        with st.container(key=f"lift_{key}"):
            yield

    # ---------- CSS ----------
    st.markdown("""
        <style>
            /* ============ BASE ============ */
            :root {
                --pitch-bg: #f1f5f9;
                --pitch-text-main: #334155;
                --pitch-text-sec: #64748b;
                --pitch-text-soft: #94a3b8;
                --pitch-brand-dark: #52606d;
                --pitch-brand-deep: #3b4654;
                --pitch-accent: #89A7C9;
                --pitch-accent-hover: #7a96b5;
                --pitch-accent-soft: #eef3f9;
                --pitch-deep-blue: #2f62a6;
                --pitch-red: #b03f3f;
                --pitch-border: #e2e8f0;
                --pitch-card-bg: #ffffff;
                --pitch-media-bg: #f8fafc;
                --pitch-radius: 16px;
                --pitch-radius-lg: 20px;
                --pitch-radius-sm: 12px;
                --pitch-shadow-sm: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
                --pitch-shadow-md: 0 10px 25px -5px rgba(0,0,0,0.08), 0 8px 10px -6px rgba(0,0,0,0.04);
                --pitch-shadow-lg: 0 22px 45px -12px rgba(0,0,0,0.15);
            }

            /* tightened top/bottom padding to lift the fold */
            .block-container {
                padding-top: 0.5rem !important;
                padding-bottom: 4rem !important;
                max-width: 1200px !important;
            }

            /* ============ HEADER — APP LANDING SIZE ============ */
            .header-container {
                display: flex;
                align-items: flex-start;
                gap: 12px;
                justify-content: flex-start;
                margin-bottom: 4rem;
            }
            .header-logo-box {
                background-color: #ffffff;
                border: 4px solid var(--pitch-brand-dark);
                padding: 2px;
                border-radius: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 72px;
                width: 72px;
                flex-shrink: 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }
            .header-logo-box img { height: 70px; }
            .header-title {
                font-size: 2.8rem;
                font-weight: 800;
                color: var(--pitch-brand-dark);
                line-height: 1;
                letter-spacing: -0.01em;
            }
            .header-subtitle {
                color: var(--pitch-brand-dark);
                font-size: 1.5rem;
                font-weight: 800;
                display: flex;
                align-items: baseline;
                gap: 15px;
                margin-top: 0px;
            }
            .header-badge {
                font-size: 0.7rem;
                color: var(--pitch-text-soft);
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-weight: 700;
            }

            /* ============ UNIFIED SECTION HEADER ============ */
            /* Kept deliberately flat: eyebrow + title + one short underline.
               No second decorative line before sub-questions — the section
               already has its underline; stacking another reads as clutter. */
            .section-head {
                text-align: center;
                margin: 4.25rem 0 0 0;
            }
            /* title flanked by soft rules that fade toward the text —
               frames each section without hard side borders */
            .section-title-row {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 1.5rem;
                max-width: 760px;
                margin: 0 auto;
            }
            .section-rule {
                height: 2px;
                flex: 1;
                max-width: 200px;
            }
            /* slower fade — the accent stays solid for most of the rule's
               length and only eases out near the far end */
            .section-rule.left {
                background: linear-gradient(90deg,
                    rgba(137,167,201,0) 0%,
                    rgba(137,167,201,0.55) 35%,
                    var(--pitch-accent) 100%);
            }
            .section-rule.right {
                background: linear-gradient(90deg,
                    var(--pitch-accent) 0%,
                    rgba(137,167,201,0.55) 65%,
                    rgba(137,167,201,0) 100%);
            }
            .section-eyebrow {
                font-family: 'Inter', sans-serif;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.17em;
                color: var(--pitch-accent);
                text-transform: uppercase;
                margin-bottom: 0.4rem;
            }
            .section-title {
                font-family: 'Inter', sans-serif;
                font-size: clamp(1.7rem, 2.5vw, 2.1rem);
                font-weight: 800;
                color: var(--pitch-brand-dark);
                letter-spacing: -0.015em;
                line-height: 1.15;
                white-space: nowrap;
            }
            @media (max-width: 560px) {
                .section-rule { display: none; }
                .section-title { white-space: normal; }
            }
            .section-subtitle {
                font-family: 'Inter', sans-serif;
                font-size: 1.08rem;
                font-weight: 600;
                color: var(--pitch-text-sec);
                margin-top: 0.85rem;
            }

            /* ============ UNIFIED SUB-QUESTION ============ */
            /* Just a clean bold line — no underline tick. The section header
               carries the only accent mark; this keeps the rhythm calm. */
            .sub-question {
                font-family: 'Inter', sans-serif;
                font-size: clamp(1.12rem, 1.5vw, 1.3rem);
                font-weight: 700;
                color: var(--pitch-brand-dark);
                text-align: center;
                margin: 2.4rem 0 1.4rem 0;
                letter-spacing: -0.005em;
            }

            /* ============ SHARED TEXT ============ */
            .pitch-p {
                font-family: 'Inter', sans-serif;
                font-size: 1.05rem;
                color: var(--pitch-text-sec);
                line-height: 1.62;
                margin-bottom: 1.1rem;
            }
            .pitch-p:last-child { margin-bottom: 0; }
            .pitch-p-strong { font-weight: 700; color: var(--pitch-text-main); }
            .hl-blue { color: var(--pitch-deep-blue); background: rgba(47,98,166,0.10); padding: 1px 7px; border-radius: 5px; font-weight: 700; }
            .hl-red  { color: var(--pitch-red);       background: rgba(176,63,63,0.10); padding: 1px 7px; border-radius: 5px; font-weight: 700; }
            .hl-grey { color: var(--pitch-brand-dark); background: #e2e8f0;            padding: 1px 7px; border-radius: 5px; font-weight: 700; }
            /* light highlight for use on dark backgrounds (foundation banner) */
            .hl-light { color: #ffffff; background: rgba(255,255,255,0.16); padding: 1px 7px; border-radius: 5px; font-weight: 800; }
            /* keeps a multi-word phrase from breaking across lines */
            .nowrap { white-space: nowrap; }

            /* ============ HERO ============ */
            /* the gap above the headline (header margin-bottom) and the gap
               below it are kept equal — the headline sits evenly framed */
            .hero-headline-wrap {
                margin-top: 0;
                margin-bottom: 4rem;
            }
            .hero-h1 {
                font-family: 'Inter', sans-serif;
                font-weight: 900;
                color: var(--pitch-brand-dark);
                line-height: 1.1;
                letter-spacing: -0.025em;
                margin: 0;
            }
            /* line 1 — slightly larger */
            .hero-h1 .hero-line-1 {
                display: block;
                font-size: clamp(2.1rem, 3.5vw, 2.95rem);
            }
            /* line 2 — keeps the previous (smaller) size, with a touch of
               space above it so the two headline lines aren't cramped */
            .hero-h1 .accent-line {
                color: var(--pitch-accent);
                display: block;
                font-size: clamp(1.85rem, 3vw, 2.5rem);
                margin-top: 0.35rem;
            }

            /* The screenshot frame sizes to its image (equal padding all
               round). The text card stretches to match it — since the body
               copy is short, the screenshot is the taller element and
               drives the row height. */
            .hero-body {
                display: flex;
                gap: 2rem;
                align-items: stretch;
            }
            .hero-body-text {
                flex: 1;
                min-width: 0;
                display: flex;
                flex-direction: column;
                justify-content: center;
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: 18px;
                box-shadow: var(--pitch-shadow-md);
                padding: 1.5rem 2rem;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .hero-body-text .pitch-p {
                font-size: 1.02rem;
                line-height: 1.55;
                font-weight: 500;
                margin-bottom: 0.85rem;
            }
            .hero-body-img {
                flex: 1.3;
                min-width: 0;
                display: flex;
                align-items: stretch;
                justify-content: center;
            }
            .hero-screenshot-frame {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: 18px;
                padding: 0.6rem;             /* equal margin all round */
                box-shadow: var(--pitch-shadow-lg);
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .hero-screenshot-frame img {
                width: 100%;
                max-height: 330px;
                height: auto;
                display: block;
                border-radius: 12px;
                object-fit: contain;
            }
            .hero-screenshot-frame.placeholder {
                width: 100%;
                min-height: 260px;
                color: var(--pitch-text-soft);
                font-weight: 600;
            }
            @media (max-width: 900px) {
                .hero-body {
                    flex-direction: column;
                    gap: 1.5rem;
                    align-items: stretch;
                }
                .hero-screenshot-frame img { max-height: none; }
            }

            /* ============ WIDE CTA CARDS ============ */
            .st-key-cta_wide_top, .st-key-cta_wide_bottom {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                box-shadow: var(--pitch-shadow-lg);
                padding: 1.9rem 2.75rem !important;
                margin: 2.5rem 0 1.5rem 0 !important;
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
                font-size: clamp(1.3rem, 2vw, 1.6rem);
                font-weight: 800;
                color: var(--pitch-brand-dark);
                margin-bottom: 0.3rem;
                line-height: 1.25;
                letter-spacing: -0.01em;
            }
            .cta-wide-subtitle {
                font-size: 1rem;
                color: var(--pitch-text-sec);
                line-height: 1.5;
                margin: 0;
            }
            .st-key-cta_wide_top  div[data-testid="stColumn"]:last-child > div,
            .st-key-cta_wide_bottom div[data-testid="stColumn"]:last-child > div {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100%;
                width: max-content;
                margin-left: auto;
                transform: translate(-120px, 5px) !important;
            }
            .st-key-cta_wide_top .stButton,
            .st-key-cta_wide_bottom .stButton {
                transform: none !important;
                width: 100%;
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
            /* reassurance microline directly under the CTA button.
               width:100% + right-aligned text makes its right edge line up
               cleanly with the button's right edge (the button hugs the
               column's right side via align-items: flex-end above). */
            /* CTA reassurance text position
               transform: translateX(...) controls left/right.
               margin-top controls up/down.
               More negative translateX = more left.
               Bigger margin-top = more down. */

            .cta-reassure {
                font-family: 'Inter', sans-serif;
                font-size: 0.8rem;
                font-weight: 600;
                color: var(--pitch-text-soft);
                text-align: center;
                margin-top: 0.0rem;
                white-space: nowrap;
                letter-spacing: 0.01em;
                width: 100%;
                padding-right: 0;
                transform: translateY(-8px) !important;
            }

            @media (max-width: 768px) {
                .st-key-cta_wide_top, .st-key-cta_wide_bottom {
                    padding: 1.75rem !important;
                    text-align: center;
                }
                .st-key-cta_wide_top  div[data-testid="stColumn"]:last-child > div,
                .st-key-cta_wide_bottom div[data-testid="stColumn"]:last-child > div {
                    justify-content: center;
                    align-items: center;
                    margin-top: 1.1rem;
                    width: 100%;
                    margin-left: 0;
                    transform: none !important;
                }
                .st-key-cta_wide_top .stButton,
                .st-key-cta_wide_bottom .stButton {
                    transform: none !important;
                }
                .cta-reassure {
                    text-align: center;
                    padding-right: 0;
                    transform: none !important;
                    width: 100%;
                }
            }

            /* ============ SPLIT CARD (shared component) ============ */
            .split-card {
                display: flex;
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius-lg);
                box-shadow: var(--pitch-shadow-md);
                overflow: hidden;
                margin-bottom: 1.5rem;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .split-card-media {
                flex: 1.35;
                background: #ffffff;          /* pure white — PNG edges blend invisibly */
                border-right: 1px solid var(--pitch-border);
                padding: 1.25rem;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 0;
            }
            /* alternating layout — media on the right instead of the left.
               Applied to every other split-card so the page zig-zags
               (best practice: breaks the monotonous single-gutter scan). */
            .split-card.reverse { flex-direction: row-reverse; }
            .split-card.reverse .split-card-media {
                border-right: none;
                border-left: 1px solid var(--pitch-border);
            }
            .split-card-media img,
            .split-card-media svg {
                width: 100%;
                max-width: 100%;
                max-height: 340px;
                height: auto;
                display: block;
                object-fit: contain;
            }
            /* the chart PNGs read better filling their panel — slightly
               tighter padding + a larger height ceiling */
            .split-card-media.wide { padding: 0.85rem; }
            .split-card-media.wide img { max-height: 380px; }
            .split-card-media.gauge { overflow: hidden; }
            .split-card-media.gauge img,
            .split-card-media.gauge svg { width: auto; }
            .split-card-media.gauge .gauge-zoom {
                transform: scale(1.04);
                transform-origin: center;
            }
            .media-placeholder {
                color: var(--pitch-text-soft);
                font-weight: 600;
                min-height: 220px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .split-card-body {
                flex: 1;
                padding: 2.4rem 2.4rem;
                background: var(--pitch-media-bg); /* slight grey — calm ground for text */
                display: flex;
                flex-direction: column;
                justify-content: center;
                min-width: 0;
            }
            /* dark-body variant — used for the opening "What's at stake" card */
            .split-card-body.dark {
                background: var(--pitch-brand-dark);
                border: none;
            }
            .split-card-body.dark .split-card-h,
            .split-card-body.dark .pitch-p { color: #f1f5f9; }
            .split-card-body.dark .split-card-h { color: #ffffff; }
            .split-card-h {
                font-family: 'Inter', sans-serif;
                font-size: clamp(1.25rem, 1.9vw, 1.5rem);
                font-weight: 800;
                color: var(--pitch-brand-dark);
                line-height: 1.28;
                margin: 0 0 0.9rem 0;
                letter-spacing: -0.01em;
            }
            @media (max-width: 850px) {
                .split-card,
                .split-card.reverse { flex-direction: column; }
                .split-card-media,
                .split-card.reverse .split-card-media {
                    border-right: none;
                    border-left: none;
                    border-bottom: 1px solid var(--pitch-border);
                }
            }

            /* dimension list inside a split-card body */
            .dim-item { margin-bottom: 0.95rem; }
            .dim-item:last-child { margin-bottom: 0; }
            .dim-title { font-weight: 800; font-size: 1.08rem; color: var(--pitch-brand-dark); margin-bottom: 0.05rem; }
            .dim-desc  { font-size: 1rem; color: var(--pitch-text-sec); }

            /* tier rows inside a split-card body */
            .tier-row { margin-bottom: 7px; font-size: 1rem; color: var(--pitch-text-sec); }
            .tier-row:last-child { margin-bottom: 0; }
            .color-box {
                width: 12px; height: 12px;
                border-radius: 3px;
                display: inline-block;
                margin-right: 9px;
                vertical-align: middle;
            }
            .cb-low   { background: linear-gradient(90deg, rgb(162,198,228) 0%, rgb(47,98,166)  100%); }
            .cb-fav   { background: linear-gradient(90deg, rgb(242,244,248) 0%, rgb(162,198,228) 100%); }
            .cb-watch { background: linear-gradient(90deg, rgb(236,162,162) 0%, rgb(242,244,248) 100%); }
            .cb-high  { background: linear-gradient(90deg, rgb(176,63,63)   0%, rgb(236,162,162) 100%); }

            /* ============ INTERPRET CARDS (red / blue) ============ */
            .interpret-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.25rem;
                margin-bottom: 1rem;
            }
            @media (max-width: 768px) { .interpret-grid { grid-template-columns: 1fr; } }
            .interpret-card {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                padding: 1.6rem 1.75rem;
                box-shadow: var(--pitch-shadow-sm);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .interpret-card:hover { transform: translateY(-3px); box-shadow: var(--pitch-shadow-md); }
            .interpret-card-title {
                font-size: 1.05rem;
                font-weight: 800;
                margin-bottom: 0.6rem;
                display: flex;
                align-items: center;
            }

            /* ============ FOUNDATION BANNER (dark statement) ============ */
            .foundation-banner {
                background: var(--pitch-brand-dark);
                border-radius: var(--pitch-radius-lg);
                padding: 2.6rem 3rem;
                text-align: center;
                box-shadow: var(--pitch-shadow-md);
                margin-bottom: 1rem;
            }
            .foundation-banner .pitch-p {
                color: #e8edf2;
                max-width: 820px;
                margin: 0 auto 1rem auto;
                font-size: 1.08rem;
            }
            .foundation-banner .pitch-p:last-child { margin-bottom: 0; }
            .foundation-banner .fn-note {
                font-size: 0.9rem;
                color: #b5c0cc;
            }

            /* ============ METRIC CARDS ============ */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 1.25rem;
                margin-bottom: 1rem;
                align-items: stretch;
            }
            .metric-card {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-top: 4px solid var(--pitch-accent);
                border-radius: var(--pitch-radius);
                padding: 2rem 1.85rem;
                box-shadow: var(--pitch-shadow-sm);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .metric-card:hover { transform: translateY(-3px); box-shadow: var(--pitch-shadow-md); }
            .metric-title {
                font-size: 1.3rem;
                font-weight: 900;
                color: var(--pitch-brand-dark);
                margin-bottom: 0.6rem;
                letter-spacing: -0.01em;
                line-height: 1.25;
                min-height: 2.5em;       /* 1- and 2-line titles align */
                display: flex;
                align-items: center;
            }
            .metric-card .pitch-p { font-size: 0.98rem; margin-bottom: 0; }
            @media (max-width: 640px) {
                .metric-title { min-height: 0; }
            }

            /* ============ WHERE IT PAYS OFF — value cards ============ */
            /* Cards live inside st.columns (3 + 2). A fixed min-height keeps
               every card the same size regardless of copy length — far more
               reliable than chaining height:100% through Streamlit's DOM.
               vertical margin gives the two rows breathing room. */
            .value-card {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                padding: 2rem 1.6rem;
                box-shadow: var(--pitch-shadow-sm);
                width: 100%;
                min-height: 288px;
                margin: 0.7rem 0;
                text-align: center;
                display: flex;
                flex-direction: column;
                align-items: center;
                transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            }
            .value-icon {
                width: 56px;
                height: 56px;
                border-radius: 14px;
                background: var(--pitch-accent-soft);
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1.1rem;
                color: var(--pitch-deep-blue);
                flex-shrink: 0;
            }
            .value-icon svg { width: 28px; height: 28px; display: block; }
            .value-icon img { width: 30px; height: 30px; object-fit: contain; display: block; }
            .value-card h4 {
                font-family: 'Inter', sans-serif;
                font-size: 1.1rem;
                font-weight: 800;
                color: var(--pitch-brand-dark);
                margin: 0 0 0.6rem 0;
                line-height: 1.3;
                min-height: 2.6em;            /* 1 or 2-line titles align */
                display: flex;
                align-items: center;
            }
            .value-card .pitch-p {
                font-size: 0.96rem;
                line-height: 1.55;
                margin: 0;
            }
            @media (max-width: 640px) {
                .value-card { min-height: 0; }
            }

            /* ============ FOOTER ============ */
            .pitch-footer {
                background: var(--pitch-card-bg);
                border: 1px solid var(--pitch-border);
                border-radius: var(--pitch-radius);
                padding: 2.75rem 3rem;
                text-align: center;
                box-shadow: var(--pitch-shadow-sm);
                margin: 3rem 0 2rem 0;
            }
            .pitch-footer .footer-h {
                font-family: 'Inter', sans-serif;
                font-size: 1.4rem;
                font-weight: 800;
                color: var(--pitch-brand-dark);
                margin-bottom: 0.9rem;
            }
            /* data & responsible-use note — plain quiet text at the very
               bottom of the page, no card, no panel */
            .page-disclaimer {
                max-width: 900px;
                margin: 2.25rem auto 0.5rem auto;
                font-family: 'Inter', sans-serif;
                font-size: 0.82rem;
                line-height: 1.65;
                color: var(--pitch-text-soft);
                text-align: center;
            }
            .page-disclaimer-title {
                font-weight: 700;
                color: var(--pitch-text-sec);
            }

            /* ============ PRIMARY BUTTON + SHINE ============ */
            .stButton > button[kind="primary"] {
                position: relative !important;
                overflow: hidden !important;
                border: none !important;
                background: var(--pitch-accent) !important;
                color: #ffffff !important;
                border-radius: 10px !important;
                padding: 1rem 2.5rem !important;
                font-size: 1.1rem !important;
                font-weight: 800 !important;
                box-shadow: 0 4px 12px rgba(137,167,201,0.30) !important;
                transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
            }
            .stButton > button[kind="primary"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 18px rgba(137,167,201,0.42) !important;
                background: var(--pitch-accent-hover) !important;
            }
            .stButton > button[kind="primary"] p {
                font-size: 1.1rem !important;
                font-weight: 800 !important;
                white-space: nowrap !important;
                position: relative;
                z-index: 1;
            }
            /* the moving shine band */
            .stButton > button[kind="primary"]::after {
                content: "";
                position: absolute;
                top: 0;
                left: -160%;
                width: 55%;
                height: 100%;
                background: linear-gradient(120deg,
                    transparent 0%, rgba(255,255,255,0.55) 50%, transparent 100%);
                transform: skewX(-22deg);
                pointer-events: none;
                animation: btnShine 1.25s ease-in-out infinite;
            }
            @keyframes btnShine {
                0%   { left: -160%; }
                70%  { left: 160%; }
                100% { left: 160%; }
            }
            @media (prefers-reduced-motion: reduce) {
                .stButton > button[kind="primary"]::after { animation: none; display: none; }
            }

            /* ============ HOVER LIFT ============ */
            /* Visuals wrapped in st.container(key="lift_<x>") get a subtle
               lift on hover. Pure CSS affordance — no click behaviour. */
            div[class*="st-key-lift_"]:hover .split-card,
            div[class*="st-key-lift_"]:hover .value-card,
            div[class*="st-key-lift_"]:hover .hero-screenshot-frame,
            div[class*="st-key-lift_"]:hover .hero-body-text {
                transform: translateY(-4px);
                box-shadow: var(--pitch-shadow-lg);
            }
            div[class*="st-key-lift_"]:hover .value-card {
                border-color: var(--pitch-accent);
            }
            @media (prefers-reduced-motion: reduce) {
                div[class*="st-key-lift_"]:hover .split-card,
                div[class*="st-key-lift_"]:hover .value-card,
                div[class*="st-key-lift_"]:hover .hero-screenshot-frame,
                div[class*="st-key-lift_"]:hover .hero-body-text {
                    transform: none;
                }
            }
            /* ============ APP-ALIGNED LARGE-SCREEN SCALE ============ */
            /* Matches the visual scale philosophy of app.py:
               wide page shell, controlled horizontal padding, capped growth. */
            @media (min-width: 1500px) {

                .block-container {
                    width: 100% !important;
                    max-width: 1760px !important;
                    padding-left: clamp(2.5rem, 5vw, 6.5rem) !important;
                    padding-right: clamp(2.5rem, 5vw, 6.5rem) !important;
                    padding-top: 2rem !important;
                    padding-bottom: 4rem !important;
                    margin-left: auto !important;
                    margin-right: auto !important;
                }

                /* Header — match app.py landing scale */
                .header-container {
                    margin-bottom: 4.2rem;
                    gap: 13px;
                }

                .header-logo-box {
                    height: 78px;
                    width: 78px;
                    border-radius: 19px;
                    border-width: 4px;
                }

                .header-logo-box img {
                    height: 76px;
                }

                .header-title {
                    font-size: 3rem;
                }

                .header-subtitle {
                    font-size: 1.6rem;
                }

                .header-badge {
                    font-size: 0.73rem;
                }

                /* Hero headline */
                .hero-headline-wrap {
                    margin-bottom: 4.1rem;
                }

                .hero-h1 .hero-line-1 {
                    font-size: clamp(3.35rem, 2.75vw, 4rem);
                }

                .hero-h1 .accent-line {
                    font-size: clamp(2.85rem, 2.25vw, 3.35rem);
                }

                /* Hero row */
                .hero-body {
                    gap: clamp(2.7rem, 2.8vw, 3.6rem);
                }

                .hero-body-text {
                    flex: 1.05;
                    padding: clamp(2.1rem, 1.9vw, 2.8rem)
                             clamp(2.5rem, 2.4vw, 3.3rem);
                    border-radius: 20px;
                }

                .hero-body-text .pitch-p {
                    font-size: clamp(1.18rem, 0.95vw, 1.3rem);
                    line-height: 1.64;
                    margin-bottom: 1rem;
                }

                .hero-body-img {
                    flex: 1.5;
                }

                .hero-screenshot-frame {
                    padding: 0.8rem;
                    border-radius: 20px;
                }

                .hero-screenshot-frame img {
                    max-height: clamp(460px, 26vw, 560px);
                }

                /* CTA cards */
                .st-key-cta_wide_top,
                .st-key-cta_wide_bottom {
                    padding: 2.4rem 3.4rem !important;
                    margin: 3rem 0 2rem 0 !important;
                }

                .cta-wide-title {
                    font-size: clamp(1.7rem, 1.45vw, 2rem);
                }

                .cta-wide-subtitle {
                    font-size: 1.16rem;
                }

                .st-key-cta_wide_top .stButton > button[kind="primary"],
                .st-key-cta_wide_bottom .stButton > button[kind="primary"] {
                    padding: 1.15rem 3.2rem !important;
                    font-size: 1.2rem !important;
                    min-width: 235px !important;
                }

                .stButton > button[kind="primary"] p {
                    font-size: 1.2rem !important;
                }

                .cta-reassure {
                    font-size: 0.9rem;
                }

                /* Section headers */
                .section-head {
                    margin: 5.2rem 0 0 0;
                }

                .section-title-row {
                    max-width: 960px;
                    gap: 2rem;
                }

                .section-rule {
                    max-width: 270px;
                }

                .section-title {
                    font-size: clamp(2.35rem, 1.9vw, 2.8rem);
                }

                .sub-question {
                    font-size: clamp(1.42rem, 1.15vw, 1.65rem);
                    margin: 2.85rem 0 1.8rem 0;
                }

                /* Shared text */
                .pitch-p {
                    font-size: clamp(1.12rem, 0.9vw, 1.22rem);
                    line-height: 1.66;
                }

                /* Split cards */
                .split-card {
                    margin-bottom: 2rem;
                    border-radius: 22px;
                }

                .split-card-body {
                    padding: clamp(3rem, 2.6vw, 3.9rem);
                }

                .split-card-h {
                    font-size: clamp(1.75rem, 1.45vw, 2.05rem);
                    margin-bottom: 1.1rem;
                }

                .split-card-media {
                    padding: 1.6rem;
                }

                .split-card-media img,
                .split-card-media svg {
                    max-height: clamp(460px, 26vw, 580px);
                }

                .split-card-media.wide {
                    padding: 1rem;
                }

                .split-card-media.wide img {
                    max-height: clamp(500px, 28vw, 620px);
                }

                .dim-title {
                    font-size: 1.2rem;
                }

                .dim-desc {
                    font-size: 1.1rem;
                }

                .tier-row {
                    font-size: 1.1rem;
                }

                /* Interpretation cards */
                .interpret-grid {
                    gap: 1.5rem;
                }

                .interpret-card {
                    padding: 2rem 2.25rem;
                }

                .interpret-card-title {
                    font-size: 1.18rem;
                }

                /* Foundation banner */
                .foundation-banner {
                    padding: 3.2rem 4rem;
                }

                .foundation-banner .pitch-p {
                    max-width: 1040px;
                    font-size: 1.18rem;
                }

                .foundation-banner .fn-note {
                    font-size: 0.98rem;
                }

                /* Metric cards */
                .metrics-grid {
                    gap: 1.5rem;
                }

                .metric-card {
                    padding: 2.45rem 2.25rem;
                }

                .metric-title {
                    font-size: 1.52rem;
                }

                .metric-card .pitch-p {
                    font-size: 1.08rem;
                }

                /* Value cards */
                .value-card {
                    min-height: clamp(320px, 18vw, 375px);
                    padding: clamp(2.35rem, 1.9vw, 2.85rem)
                             clamp(1.9rem, 1.5vw, 2.3rem);
                }

                .value-icon {
                    width: 66px;
                    height: 66px;
                    border-radius: 17px;
                    margin-bottom: 1.25rem;
                }

                .value-icon svg {
                    width: 33px;
                    height: 33px;
                }

                .value-icon img {
                    width: 35px;
                    height: 35px;
                }

                .value-card h4 {
                    font-size: 1.24rem;
                }

                .value-card .pitch-p {
                    font-size: 1.05rem;
                }

                /* Footer */
                .pitch-footer {
                    padding: 3.25rem 3.8rem;
                }

                .pitch-footer .footer-h {
                    font-size: 1.6rem;
                }

                .page-disclaimer {
                    max-width: 1080px;
                    font-size: 0.9rem;
                }
            }

            /* Same philosophy as app.py large desktop breakpoint */
            @media (min-width: 1800px) and (min-height: 950px) {

                .block-container {
                    max-width: 2060px !important;
                    padding-left: clamp(3rem, 5vw, 7.5rem) !important;
                    padding-right: clamp(3rem, 5vw, 7.5rem) !important;
                    padding-top: 2.3rem !important;
                }

                .hero-screenshot-frame img {
                    max-height: clamp(500px, 27vw, 620px);
                }
            }

            /* 2560px+ typography bump.
               Keeps 1920px close to current, but avoids the miniature look
               on 2560x1440 and 2880px screens. */
            @media (min-width: 2300px) and (min-height: 1200px) {

                .header-logo-box {
                    height: 88px;
                    width: 88px;
                    border-radius: 22px;
                    border-width: 4px;
                }

                .header-logo-box img {
                    height: 86px;
                }

                .header-title {
                    font-size: 3.35rem;
                }

                .header-subtitle {
                    font-size: 1.78rem;
                }

                .header-badge {
                    font-size: 0.78rem;
                }

                .hero-h1 .hero-line-1 {
                    font-size: clamp(3.75rem, 2.45vw, 4.35rem);
                }

                .hero-h1 .accent-line {
                    font-size: clamp(3.15rem, 2.05vw, 3.65rem);
                }

                .hero-body-text .pitch-p {
                    font-size: clamp(1.3rem, 0.95vw, 1.42rem);
                    line-height: 1.64;
                }

                .pitch-p {
                    font-size: clamp(1.22rem, 0.88vw, 1.34rem);
                    line-height: 1.66;
                }

                .cta-wide-title {
                    font-size: clamp(1.9rem, 1.35vw, 2.2rem);
                }

                .cta-wide-subtitle {
                    font-size: 1.28rem;
                }

                .cta-reassure {
                    font-size: 0.98rem;
                }

                .st-key-cta_wide_top .stButton > button[kind="primary"],
                .st-key-cta_wide_bottom .stButton > button[kind="primary"] {
                    font-size: 1.28rem !important;
                    min-width: 255px !important;
                }

                .stButton > button[kind="primary"] p {
                    font-size: 1.28rem !important;
                }

                .section-title {
                    font-size: clamp(2.65rem, 1.8vw, 3.05rem);
                }

                .sub-question {
                    font-size: clamp(1.6rem, 1.08vw, 1.85rem);
                }

                .split-card-h {
                    font-size: clamp(1.95rem, 1.35vw, 2.25rem);
                }

                .dim-title {
                    font-size: 1.3rem;
                }

                .dim-desc {
                    font-size: 1.2rem;
                }

                .tier-row {
                    font-size: 1.2rem;
                }

                .interpret-card-title {
                    font-size: 1.28rem;
                }

                .metric-title {
                    font-size: 1.7rem;
                }

                .metric-card .pitch-p {
                    font-size: 1.18rem;
                }

                .value-card h4 {
                    font-size: 1.36rem;
                }

                .value-card .pitch-p {
                    font-size: 1.14rem;
                }

                .pitch-footer .footer-h {
                    font-size: 1.75rem;
                }

                .page-disclaimer {
                    font-size: 0.98rem;
                }
            }
        </style>

    """, unsafe_allow_html=True)

    # ====================================================================
    # HEADER
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
    # HERO  — compact headline + (sentences left, screenshot right)
    # ====================================================================
    screenshot_b64 = get_img_b64("screenshot.png")
    screenshot_html = (
        f'<div class="hero-screenshot-frame"><img src="{screenshot_b64}" alt="CTPredict application screenshot"></div>'
        if screenshot_b64
        else '<div class="hero-screenshot-frame placeholder"><span>Visual: screenshot.png</span></div>'
    )

    st.markdown("""
        <div class="hero-headline-wrap">
            <div class="hero-h1">
                <span class="hero-line-1">Identify clinical trials at risk</span>
                <span class="accent-line">before execution begins</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with hover_lift("hero"):
        st.markdown(f"""
            <div class="hero-body">
                <div class="hero-body-text">
                    <div class="pitch-p">
                        Predict whether <span class="nowrap">Phase II/III trials</span> are most likely to <span class="hl-blue nowrap">complete</span> or <span class="hl-red nowrap">terminate early</span>, using early design-stage information.
                    </div>
                    <div class="pitch-p">
                        CTPredict is a <span class="hl-grey nowrap">machine-learning model</span> trained on 24,000 past trial outcomes to classify new trials by risk tier and reveal trial-specific risk drivers.
                    </div>
                    <div class="pitch-p">
                        Test the impact of trial-design changes through <span class="hl-grey nowrap">simulation mode</span>.
                    </div>
                </div>
                <div class="hero-body-img">
                    {screenshot_html}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ====================================================================
    # CTA CARD #1
    # ====================================================================
    with st.container(key="cta_wide_top"):
        c1, c2 = st.columns([2.2, 1], gap="large")
        with c1:
            st.markdown("""
                <div class="cta-wide-title">Choose a trial to score.</div>
                <div class="cta-wide-subtitle">Review its completion score, risk tier, and key risk drivers.<br><span style="font-size:0.88rem;">(Demo version: reach out for more trials, additional trials available on request.)</span></div>
            """, unsafe_allow_html=True)
        with c2:
            st.button("Access Demo  →", key="cta_btn_top", on_click=launch_demo,
                      args=("cta_top",), type="primary")
            st.markdown('<div class="cta-reassure">No sign-up · Opens instantly</div>',
                        unsafe_allow_html=True)

    # ====================================================================
    # THE PROBLEM — What's at stake
    # ====================================================================
    section_head("What's at stake")
    sub_question("Why focus on Phase II/III?")

    why_b64 = get_img_b64("Why_it_matters.png")
    stake_visual = (
        f'<img src="{why_b64}" alt="What is at stake">'
        if why_b64 else _STAKE_SVG
    )
    with hover_lift("stake"):
        st.markdown(f"""
            <div class="split-card">
                <div class="split-card-media">{stake_visual}</div>
                <div class="split-card-body dark">
                    <div class="split-card-h">Focus attention early, where clinical, financial, and strategic stakes are highest.</div>
                    <div class="pitch-p">
                        Phase II/III is where scientific ambition meets major financial and strategic stakes. CTPredict helps identify trials most exposed to early-termination risk, helping focus support, review, and key protocol or strategic decisions where they matter most.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ====================================================================
    # THE METHOD — Inside the engine
    # ====================================================================
    section_head("Inside the engine")

    # --- How do you read the score? ---
    sub_question("How do you read the score?")
    with hover_lift("gauge"):
        st.markdown(f"""
            <div class="split-card reverse">
                <div class="split-card-media gauge">
                    <div class="gauge-zoom">{media_img("gauge.png", "Completion Score Gauge")}</div>
                </div>
                <div class="split-card-body">
                    <div class="split-card-h">A 0-100 completion score, sorted into four risk tiers.</div>
                    <div class="pitch-p">
                        Reflects how closely a trial resembles historical patterns of <span class="hl-red nowrap">early termination</span> or <span class="hl-blue nowrap">full completion</span>. Four risk tiers:
                    </div>
                    <div style="margin-top: 0.4rem;">
                        <div class="tier-row"><span class="color-box cb-high"></span><span class="pitch-p-strong">High Risk</span> &nbsp;0-25 points</div>
                        <div class="tier-row"><span class="color-box cb-watch"></span><span class="pitch-p-strong">Watchlist</span> &nbsp;25-50 points</div>
                        <div class="tier-row"><span class="color-box cb-fav"></span><span class="pitch-p-strong">Favorable</span> &nbsp;50-75 points</div>
                        <div class="tier-row"><span class="color-box cb-low"></span><span class="pitch-p-strong">Low Risk</span> &nbsp;75-100 points</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="interpret-grid">
            <div class="interpret-card">
                <div class="interpret-card-title" style="color: var(--pitch-red);"><span class="color-box cb-high"></span> Red reflects a riskier trial profile.</div>
                <div class="pitch-p">May flag design choices worth challenging - or capture ambition: novel science, rigorous endpoints, flexible design, or higher complexity. Higher risk can also point to higher potential value.</div>
            </div>
            <div class="interpret-card">
                <div class="interpret-card-title" style="color: var(--pitch-deep-blue);"><span class="color-box cb-low"></span> Blue signals higher likelihood of full completion.</div>
                <div class="pitch-p">May signal strong execution conditions, high-quality design, or a more established scientific profile with lower complexity. Full completion should still be read alongside scientific and asset value.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- What does the model assess? ---
    sub_question("What does the model assess?")
    with hover_lift("barchart"):
        st.markdown(f"""
            <div class="split-card">
                <div class="split-card-media wide" style="flex: 1.55;">{media_img("barchart.png", "Impact Bar Chart")}</div>
                <div class="split-card-body">
                    <div class="split-card-h">Four risk dimensions.</div>
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

    # --- Which factors drive a prediction? ---
    sub_question("Which factors drive a prediction?")
    with hover_lift("treemap"):
        st.markdown(f"""
            <div class="split-card reverse">
                <div class="split-card-media wide" style="flex: 1.95;">{media_img("treemap.png", "Interactive Treemap")}</div>
                <div class="split-card-body">
                    <div class="split-card-h">A transparent view of the score.</div>
                    <div class="pitch-p">
                        Breaks each prediction into <span class="hl-grey">27 core trial features</span>, distributed across the four main risk dimensions, showing which factors contribute most to the final risk signal.
                    </div>
                    <div class="pitch-p">Larger blocks indicate stronger impact.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ====================================================================
    # THE EVIDENCE — How CTPredict performs
    # ====================================================================
    section_head("How CTPredict performs")

    # --- How well does it predict? ---
    # Metric cards — verified against the test-set evaluation
    # (Phase 2/3 industry trials, 2022-2023 start dates):
    #   recall  = 512 / (512+172) = 74.9%   -> "3 in 4 caught"
    #   ROC AUC = 78.0%
    #   top-20% audit captures 38.3% of all failures -> "nearly 40%"
    # Order is by impact: recall, then AUC, then the prioritization framing.
    sub_question("How well does it predict?")
    st.markdown("""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">3 in 4 at-risk trials caught</div>
                <div class="pitch-p">Around 75% of trials that later terminated early were placed in the High Risk or Watchlist tiers - scoring below 50 - from design-stage information alone.</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">78% ROC AUC</div>
                <div class="pitch-p">78% of the time, CTPredict assigns a lower completion score to trials that later fail - well above the 50% random baseline, an exceptional result while using only publicly available trial data.</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Audit 20%, catch nearly 40% of failures</div>
                <div class="pitch-p">Ranked by CTPredict score, reviewing just the riskiest 20% of a portfolio surfaces close to 40% of all trials that later terminate - roughly double a random review, so attention goes where it matters first.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- What powers the engine? ---
    sub_question("What powers the engine?")
    st.markdown("""
        <div class="foundation-banner">
            <div class="pitch-p">
                CTPredict is built on publicly available AACT* data from <span class="hl-light">24,000</span> industry-led Phase II/III trials initiated since 2009. At its core is a <span class="hl-light">supervised XGBoost machine-learning model</span>, embedded in a <span class="hl-light">multi-stage predictive pipeline</span> where supporting models and feature-engineering layers convert raw trial information into trial-level risk drivers and an overall completion-risk score.
            </div>
            <div class="pitch-p fn-note">
                &#42;AACT: Aggregate Analysis of ClinicalTrials.gov - a public database derived from ClinicalTrials.gov records.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ====================================================================
    # THE PAYOFF — Where it brings value
    # ====================================================================
    section_head("Where it brings value")
    sub_question("Which decisions does it support?")

    # Value cards laid out with st.columns (3 on top, 2 centered below) so
    # each sits in its own hover-lift container and rows stay height-matched.
    def value_card_html(icon_key, png_name, title, desc):
        return f"""
            <div class="value-card">
                <div class="value-icon">{value_icon(icon_key, png_name)}</div>
                <h4>{title}</h4>
                <div class="pitch-p">{desc}</div>
            </div>
        """

    _value_cards = [
        ("portfolio", "icon_portfolio_mgt.png", "Portfolio Management",
         "Highlight late-stage assets that may require closer scrutiny."),
        ("ta", "icon_ta_lead.png", "Therapeutic Area Leadership",
         "Benchmark completion-risk patterns across indications, modalities, and trial designs."),
        ("clinical", "icon_clin_lead.png", "Clinical Development Leads",
         "Explore in simulation mode how design choices may shift the completion-risk profile before trial initiation."),
        ("investor", "icon_investor.png", "Investors & Analysts",
         "Compare completion-risk profiles across late-stage assets in the industry using public data."),
        ("training", "icon_training.png", "Training & Capability Building",
         "Use real trials and simulation mode to support learning and strengthen risk-based decision-making."),
    ]

    # Row 1 — three cards
    row1 = st.columns(3, gap="medium")
    for col, (k, png, title, desc) in zip(row1, _value_cards[:3]):
        with col:
            with hover_lift(f"val_{k}"):
                st.markdown(value_card_html(k, png, title, desc),
                            unsafe_allow_html=True)

    # Row 2 — two cards, centered to match the 3-wide row's card width
    row2 = st.columns([1, 2, 2, 1], gap="medium")
    for col, (k, png, title, desc) in zip(row2[1:3], _value_cards[3:]):
        with col:
            with hover_lift(f"val_{k}"):
                st.markdown(value_card_html(k, png, title, desc),
                            unsafe_allow_html=True)

    # ====================================================================
    # CTA CARD #2
    # ====================================================================
    with st.container(key="cta_wide_bottom"):
        c1, c2 = st.columns([2.2, 1], gap="large")
        with c1:
            st.markdown("""
                <div class="cta-wide-title">Test CTPredict with a trial of your choice.</div>
                <div class="cta-wide-subtitle">Explore its completion score, risk tier, and key risk drivers.<br><span style="font-size:0.88rem;">(Demo version: reach out for more trials, additional trials available on request.)</span></div>
            """, unsafe_allow_html=True)
        with c2:
            st.button("Access Demo  →", key="cta_btn_bottom", on_click=launch_demo,
                      args=("cta_bottom",), type="primary")
            st.markdown('<div class="cta-reassure">No sign-up · Opens instantly</div>',
                        unsafe_allow_html=True)

    # ====================================================================
    # FOOTER
    # ====================================================================
    st.markdown("""
        <div class="pitch-footer">
            <div class="footer-h">Pilot version. Welcoming your ideas.</div>
            <div class="pitch-p" style="max-width: 820px; margin: 0 auto 1.5rem auto;">
                This demo focuses on single-trial exploration and includes a limited set of trials - additional trials can be made available on request. Broader capabilities are available or in development, including sponsor full-portfolio screening, therapeutic-area benchmarking, and simulation-based use cases. Feedback, questions, and ideas for future development are very welcome.
            </div>
            <div class="pitch-p-strong">Contact: Nicolas Delaunay</div>
            <div class="pitch-p">Email: <a href="mailto:delaunay80@gmail.com" style="color: var(--pitch-brand-dark); text-decoration: none; font-weight: 700;">delaunay80@gmail.com</a></div>
        </div>
    """, unsafe_allow_html=True)

    # ---- Data & responsible-use note: plain text, very bottom, no card ----
    st.markdown("""
        <div class="page-disclaimer">
            <span class="page-disclaimer-title">Data &amp; responsible-use note.</span>
            CTPredict uses only public, aggregated registry data (AACT / ClinicalTrials.gov) - no proprietary, patient-level, or personal data. It is a decision-support tool, not a regulated medical device, and its probabilistic estimates are meant to inform expert judgment, not replace it.
        </div>
    """, unsafe_allow_html=True)
