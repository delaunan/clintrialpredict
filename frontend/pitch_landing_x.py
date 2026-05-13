"""
CTPredict — Pitch Landing Page  v6
Place in  frontend/pitch_landing.py
"""

import base64
import streamlit as st
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
LOGO_WHITE  = CURRENT_DIR / "logo_white.png"
LOGO_GREY   = CURRENT_DIR / "logo_grey_title.png"
HERO_IMG    = CURRENT_DIR / "static" / "linkedin-preview-v5.png"


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.pr *, .pr *::before, .pr *::after { box-sizing: border-box; margin: 0; padding: 0; }
.pr { font-family: 'Inter', -apple-system, sans-serif; color: #334155; }
.pr .sec  { margin-bottom: 3rem; }
.pr .rule { display: flex; align-items: center; gap: 16px; margin-bottom: 1.6rem; }
.pr .rtxt { font-size: 0.92rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.12em; white-space: nowrap; }
.pr .rline{ flex: 1; height: 1px; background: #e2e8f0; }

.pr .hero-kicker{ font-size: 0.8rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase; display: block; margin-bottom: 0.8rem; }
.pr .hero-title { font-size: 2.1rem; font-weight: 800; color: #52606d; letter-spacing: -0.03em; line-height: 1.12; margin-bottom: 1rem; }
.pr .hero-title em { color: #89A7C9; font-style: normal; }
.pr .hero-lead  { font-size: 1.05rem; color: #64748b; line-height: 1.78; max-width: 680px; }

.pr .two-col  { display: grid; grid-template-columns: 1fr 1fr; gap: 2.6rem; }
.pr .col-head { font-size: 1.05rem; font-weight: 800; color: #52606d; letter-spacing: -0.02em; margin-bottom: 0.6rem; line-height: 1.25; }
.pr .col-body { font-size: 0.97rem; color: #64748b; line-height: 1.78; }
.pr .col-body strong { font-weight: 700; color: #334155; }

.pr .demo-box  { background: #52606d; border-radius: 14px; padding: 2.4rem 2.8rem; }
.pr .demo-eye  { font-size: 0.68rem; font-weight: 800; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.18em; margin-bottom: 0.5rem; }
.pr .demo-title{ font-size: 1.35rem; font-weight: 800; color: #ffffff; letter-spacing: -0.025em; margin-bottom: 0.5rem; line-height: 1.2; }
.pr .demo-body { font-size: 0.95rem; color: rgba(255,255,255,0.62); line-height: 1.62; }

.pr .stats    { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.pr .stat     { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 22px 24px; }
.pr .stat-lbl { font-size: 0.68rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 10px; }
.pr .stat-num { font-size: 2.2rem; font-weight: 800; color: #52606d; letter-spacing: -0.04em; line-height: 1; margin-bottom: 10px; }
.pr .stat-num b { color: #89A7C9; }
.pr .stat-desc  { font-size: 0.88rem; color: #94a3b8; line-height: 1.62; }
.pr .stat-desc strong { color: #64748b; font-weight: 600; }

.pr .tiers  { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.pr .tier   { border-radius: 10px; padding: 18px; border: 1px solid; }
.pr .t-lbl  { font-size: 0.62rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.13em; margin-bottom: 4px; }
.pr .t-name { font-size: 1.1rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 4px; line-height: 1.1; }
.pr .t-range{ font-size: 0.78rem; font-weight: 600; margin-bottom: 8px; }
.pr .t-desc { font-size: 0.85rem; color: #64748b; line-height: 1.52; }
.pr .thr { border-color: rgba(176,63,63,.28);  background: rgba(176,63,63,.05); }
.pr .thr .t-lbl, .pr .thr .t-name, .pr .thr .t-range { color: rgb(176,63,63); }
.pr .twl { border-color: rgba(160,100,20,.25); background: rgba(200,130,40,.05); }
.pr .twl .t-lbl, .pr .twl .t-name, .pr .twl .t-range { color: rgb(140,85,10); }
.pr .tfv { border-color: rgba(162,198,228,.45); background: rgba(162,198,228,.07); }
.pr .tfv .t-lbl, .pr .tfv .t-name, .pr .tfv .t-range { color: rgb(47,98,166); }
.pr .tlr { border-color: rgba(47,98,166,.35);  background: rgba(47,98,166,.06); }
.pr .tlr .t-lbl, .pr .tlr .t-name, .pr .tlr .t-range { color: rgb(25,65,120); }

.pr .nrow { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 10px; }
.pr .ndot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 7px; }
.pr .nr   { background: rgb(176,63,63); }
.pr .nb   { background: rgb(47,98,166); }
.pr .ntxt { font-size: 0.95rem; color: #64748b; line-height: 1.68; }
.pr .ntxt strong { color: #334155; font-weight: 700; }

.pr .pillars { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.pr .pillar  { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 24px 26px; border-top-width: 4px; }
.pr .pl-num  { font-size: 0.65rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px; }
.pr .pl-name { font-size: 1.15rem; font-weight: 800; color: #52606d; letter-spacing: -0.025em; margin-bottom: 5px; line-height: 1.15; }
.pr .pl-feat { font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 14px; }
.pr .pl-body { font-size: 0.9rem; color: #64748b; line-height: 1.72; }

.pr .steps  { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2.4rem; }
.pr .sn     { font-size: 0.68rem; font-weight: 800; color: #89A7C9; text-transform: uppercase; letter-spacing: 0.13em; margin-bottom: 8px; }
.pr .sbar   { width: 28px; height: 3px; background: #89A7C9; border-radius: 2px; margin-bottom: 14px; }
.pr .stitle { font-size: 1.05rem; font-weight: 800; color: #52606d; letter-spacing: -0.02em; margin-bottom: 8px; }
.pr .sbody  { font-size: 0.9rem; color: #64748b; line-height: 1.68; }

.pr .personas { display: flex; flex-wrap: wrap; justify-content: center; gap: 12px; }
.pr .persona  { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px 22px; flex: 0 0 calc(33.333% - 8px); min-width: 180px; }
.pr .picon    { width: 32px; height: 32px; background: #717d8b; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px; }
.pr .picon svg{ width: 16px; height: 16px; stroke: #ffffff; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.pr .pname  { font-size: 0.92rem; font-weight: 800; color: #52606d; letter-spacing: -0.02em; margin-bottom: 7px; line-height: 1.2; }
.pr .pbody  { font-size: 0.86rem; color: #64748b; line-height: 1.65; }

.pr .cta       { background: #717d8b; border-radius: 14px; padding: 3.4rem 2.8rem; text-align: center; margin-top: 2.8rem; }
.pr .cta-eye   { font-size: 0.68rem; font-weight: 800; color: rgba(255,255,255,.4); text-transform: uppercase; letter-spacing: 0.18em; margin-bottom: 0.7rem; }
.pr .cta-title { font-size: 1.65rem; font-weight: 800; color: #ffffff; letter-spacing: -0.03em; margin-bottom: 0.75rem; line-height: 1.15; }
.pr .cta-body  { font-size: 0.95rem; color: rgba(255,255,255,.6); line-height: 1.68; }
.pr .cta-body a{ color: rgba(255,255,255,.85); }

.pr .footer { text-align: center; margin-top: 1.8rem; font-size: 0.72rem; color: #94a3b8; line-height: 1.8; }
.pr .footer a { color: #89A7C9; text-decoration: none; }
</style>
"""


_PART1 = """
<div class="pr">
  <div class="sec" style="margin-bottom:1.6rem;">
    <span class="hero-kicker">Phase II &amp; III &middot; Built on AACT public data</span>
    <h1 class="hero-title">Know which trials are at risk &mdash;<br><em>at the design stage.</em></h1>
    <p class="hero-lead">CTPredict quantifies operational completion risk before trial initiation &mdash; so development teams can focus attention and resources on the trials that need it most, reduce financial exposure, and support earlier reallocation decisions when needed.</p>
  </div>

  <div class="sec">
    <div class="rule"><span class="rtxt">Why it matters</span><div class="rline"></div></div>
    <div class="two-col">
      <div>
        <div class="col-head">Late-stage trial terminations are critical</div>
        <p class="col-body">Phase II and III terminations carry some of the highest financial and strategic costs in drug development. Design-stage risk signals exist in the data. They are rarely quantified systematically &mdash; until now.</p>
      </div>
      <div>
        <div class="col-head">Machine learning on 30,000+ trials</div>
        <p class="col-body">Supervised learning &mdash; XGBoost &mdash; trained on the public <strong>AACT registry</strong>. Binary outcome: full completion vs. early termination. <strong>27 trial features</strong> evaluated per trial, using only information available at initiation.</p>
      </div>
    </div>
  </div>

  <div class="demo-box">
    <div class="demo-eye">Live tool</div>
    <div class="demo-title">Explore any trial from 30,000+ indexed studies</div>
    <div class="demo-body">Search Phase II &amp; III trials. Get a full risk report with score, driver map, and simulation mode in seconds.</div>
  </div>
</div>
"""


_PART2 = """
<div class="pr">
  <div class="sec">
    <div class="rule"><span class="rtxt">Prediction quality</span><div class="rline"></div></div>
    <div class="stats">
      <div class="stat">
        <div class="stat-lbl">ROC AUC</div>
        <div class="stat-num">0.<b>78</b></div>
        <div class="stat-desc">When comparing a completed trial with one that terminated early, the model assigns the higher risk score to the failed trial in <strong>78% of cases</strong> &mdash; vs. 50% by chance alone.</div>
      </div>
      <div class="stat">
        <div class="stat-lbl">Recall on early terminations</div>
        <div class="stat-num">3<b>/4</b></div>
        <div class="stat-desc">3 out of 4 trials that ultimately terminated early are correctly identified and flagged at design stage, before the trial starts.</div>
      </div>
      <div class="stat">
        <div class="stat-lbl">Trial features evaluated</div>
        <div class="stat-num">27</div>
        <div class="stat-desc">Across 4 pillars: therapeutic context, scientific challenge, execution framework, and patient profile &mdash; all from public sources.</div>
      </div>
    </div>
  </div>

  <div class="sec">
    <div class="rule"><span class="rtxt">Four risk tiers</span><div class="rline"></div></div>
    <div class="tiers">
      <div class="tier thr"><div class="t-lbl">Tier I</div><div class="t-name">High Risk</div><div class="t-range">Score 0&ndash;25</div><div class="t-desc">Design profile closely resembles prior failures. First priority for team scrutiny.</div></div>
      <div class="tier twl"><div class="t-lbl">Tier II</div><div class="t-name">Watchlist</div><div class="t-range">Score 25&ndash;50</div><div class="t-desc">Elevated risk &mdash; can also reflect increased innovation and complexity rather than poor design.</div></div>
      <div class="tier tfv"><div class="t-lbl">Tier III</div><div class="t-name">Favorable</div><div class="t-range">Score 50&ndash;75</div><div class="t-desc">Strong execution profile. Completion trajectory well-supported by precedent trials.</div></div>
      <div class="tier tlr"><div class="t-lbl">Tier IV</div><div class="t-name">Low Risk</div><div class="t-range">Score 75&ndash;100</div><div class="t-desc">Highest-confidence completion signal across all 27 trial features.</div></div>
    </div>
    <div style="margin-top:1.4rem;">
      <div class="nrow"><div class="ndot nr"></div><div class="ntxt"><strong>High risk can mean high scientific ambition.</strong> A first-in-class asset with a novel target, adaptive design, or hard clinical endpoint may carry more operational risk &mdash; and potentially more value. The score measures execution risk, not scientific worth.</div></div>
      <div class="nrow"><div class="ndot nb"></div><div class="ntxt"><strong>Low risk does not imply high value.</strong> A conventional trial may score well operationally while offering limited clinical differentiation. Always interpret alongside scientific context.</div></div>
    </div>
  </div>

  <div class="sec">
    <div class="rule"><span class="rtxt">27 trial features across 4 pillars</span><div class="rline"></div></div>
    <p style="font-size:0.97rem;color:#64748b;line-height:1.75;margin-bottom:1.6rem;">Each trial is evaluated across 27 features spanning its scientific profile, therapeutic context, execution setup, and patient population. The model learns which combinations of features historically distinguish completions from early terminations &mdash; and by how much each one contributes.</p>
    <div class="pillars">
      <div class="pillar" style="border-top-color:rgb(176,63,63);">
        <div class="pl-num" style="color:rgb(176,63,63);">01 &mdash; Therapeutic Context</div>
        <div class="pl-name">Disease &amp; Development Profile</div>
        <div class="pl-feat" style="color:rgb(176,63,63);">5 features</div>
        <div class="pl-body">Therapeutic area &middot; Indication &middot; Rare disease status &middot; Clinical phase &middot; Regulatory intent</div>
      </div>
      <div class="pillar" style="border-top-color:rgb(200,130,40);">
        <div class="pl-num" style="color:rgb(200,130,40);">02 &mdash; Scientific Challenge</div>
        <div class="pl-name">Biological &amp; Protocol Complexity</div>
        <div class="pl-feat" style="color:rgb(200,130,40);">9 features</div>
        <div class="pl-body">Target precedent &middot; Pathway profile &middot; Therapeutic modality &middot; Innovation rank &middot; Endpoint type &middot; Design flexibility &middot; Biomarker selection</div>
      </div>
      <div class="pillar" style="border-top-color:rgb(47,98,166);">
        <div class="pl-num" style="color:rgb(47,98,166);">03 &mdash; Execution Framework</div>
        <div class="pl-name">Operational &amp; Methodological Setup</div>
        <div class="pl-feat" style="color:rgb(47,98,166);">8 features</div>
        <div class="pl-body">Sponsor tier &middot; Allocation method &middot; DMC involvement &middot; Placebo control &middot; Benchmark comparator &middot; Delivery profile &middot; Number of arms &middot; Endpoint duration</div>
      </div>
      <div class="pillar" style="border-top-color:rgb(89,148,196);">
        <div class="pl-num" style="color:rgb(89,148,196);">04 &mdash; Patient Profile</div>
        <div class="pl-name">Population Scope &amp; Clinical Severity</div>
        <div class="pl-feat" style="color:rgb(89,148,196);">5 features</div>
        <div class="pl-body">Patient severity &middot; Line of therapy &middot; Population type &middot; Age eligibility &middot; Gender eligibility</div>
      </div>
    </div>
  </div>

  <div class="sec">
    <div class="rule"><span class="rtxt">How it works</span><div class="rline"></div></div>
    <div class="steps">
      <div>
        <div class="sn">01 &mdash; Score</div><div class="sbar"></div>
        <div class="stitle">One number. Full context.</div>
        <div class="sbody">Every trial receives a 0&ndash;100 completion score, benchmarked against 20 years of Phase II/III execution patterns from the AACT registry.</div>
      </div>
      <div>
        <div class="sn">02 &mdash; Diagnose</div><div class="sbar"></div>
        <div class="stitle">Interactive risk driver map</div>
        <div class="sbody">Score decomposed across all 4 pillars and sub-categories. An interactive treemap shows which trial features are driving risk &mdash; and by how much.</div>
      </div>
      <div>
        <div class="sn">03 &mdash; Simulate</div><div class="sbar"></div>
        <div class="stitle">What-if at design stage</div>
        <div class="sbody">Edit protocol parameters and re-score in real time. Explore how changes to key design assumptions shift the risk tier before trial initiation.</div>
      </div>
    </div>
  </div>

  <div class="sec">
    <div class="rule"><span class="rtxt">Who uses it &mdash; and how</span><div class="rline"></div></div>
    <div class="personas">
      <div class="persona">
        <div class="picon"><svg viewBox="0 0 24 24"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg></div>
        <div class="pname">Portfolio Management</div>
        <div class="pbody">Screen entire late-stage portfolios. Rank assets by operational risk. Focus bandwidth where it matters and support earlier reallocation decisions.</div>
      </div>
      <div class="persona">
        <div class="picon"><svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg></div>
        <div class="pname">Therapeutic Area Leadership</div>
        <div class="pbody">Benchmark your TA&rsquo;s risk profile across the industry. Understand where your programmes stand relative to 20 years of late-stage precedent.</div>
      </div>
      <div class="persona">
        <div class="picon"><svg viewBox="0 0 24 24"><path d="M9 12h6M9 16h6M9 8h6M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg></div>
        <div class="pname">Clinical Development Leads</div>
        <div class="pbody">Use simulation mode at design stage. Test how protocol choices affect the risk profile. Build evidence for discussions on the most sensitive design decisions.</div>
      </div>
      <div class="persona">
        <div class="picon"><svg viewBox="0 0 24 24"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg></div>
        <div class="pname">Investors &amp; Analysts</div>
        <div class="pbody">Compare operational risk profiles across assets, companies, or indications. A systematic lens to complement scientific due diligence on late-stage pipelines.</div>
      </div>
      <div class="persona">
        <div class="picon"><svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5M2 12l10 5 10-5"/></svg></div>
        <div class="pname">Trial Design &amp; Capability Building</div>
        <div class="pbody">Translate protocol choices into measurable risk signals. A structured environment for teams developing judgment in late-stage trial design.</div>
      </div>
    </div>
  </div>

  <div class="cta">
    <div class="cta-eye">Ready to start</div>
    <div class="cta-title">Screen a trial in seconds.</div>
    <div class="cta-body">
      Search 30,000+ indexed Phase II and III trials. Select one.<br>
      Get a full risk report with score, driver map, and simulation mode.<br><br>
      Simulation access &amp; custom views: <a href="mailto:delaunay80@gmail.com">delaunay80@gmail.com</a> &nbsp;&middot;&nbsp; +33 7 86 72 21 43
    </div>
  </div>

  <div class="footer">
    Built exclusively on public AACT registry data &nbsp;&middot;&nbsp; No patient-level data &nbsp;&middot;&nbsp; Privacy-first server-side logging<br>
    XGBoost &nbsp;&middot;&nbsp; Phase II &amp; III &nbsp;&middot;&nbsp; 2005&ndash;present &nbsp;&middot;&nbsp; <a href="mailto:delaunay80@gmail.com">delaunay80@gmail.com</a>
  </div>

</div>
"""


def render_pitch_page():
    # 1. Styles
    st.markdown(_CSS, unsafe_allow_html=True)

    # 2. Header: logo in rounded container + app name/tagline
    logo_path = LOGO_WHITE if LOGO_WHITE.exists() else LOGO_GREY
    use_dark  = LOGO_WHITE.exists()
    bg_css    = "background:#717d8b;" if use_dark else "background:#f1f5f9;border:1px solid #e2e8f0;"
    logo_b64  = base64.b64encode(logo_path.read_bytes()).decode() if logo_path.exists() else ""

    st.markdown(
        f'''<div style="display:flex;align-items:center;gap:14px;padding:1.2rem 0 1.6rem;">
          <div style="width:60px;height:60px;border-radius:16px;{bg_css}
                      display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            <img src="data:image/png;base64,{logo_b64}"
                 style="width:40px;height:40px;object-fit:contain;" alt="CTPredict">
          </div>
          <div>
            <div style="font-size:1.3rem;font-weight:800;color:#52606d;letter-spacing:-0.025em;line-height:1.1;">
              CTPredict
            </div>
            <div style="font-size:0.75rem;font-weight:600;color:#94a3b8;letter-spacing:0.02em;margin-top:3px;">
              Late-Stage Clinical Trial Predictive Engine
            </div>
          </div>
        </div>''',
        unsafe_allow_html=True,
    )

    # 3. LinkedIn preview banner
    if HERO_IMG.exists():
        st.image(str(HERO_IMG), use_container_width=True)
    st.markdown("<div style='height:1.4rem;'></div>", unsafe_allow_html=True)

    # 4. Hero + Why + Demo box
    st.markdown(_PART1, unsafe_allow_html=True)

    # 5. Demo access button — centered, directly below demo box
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button(
            "Access the demo →",
            key="pitch_demo_btn",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["pitch_seen"] = True
            st.rerun()

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

    # 6. Remaining sections
    st.markdown(_PART2, unsafe_allow_html=True)

    # 7. Bottom CTA button — centered under the dark panel
    _, col_cta, _ = st.columns([2, 3, 2])
    with col_cta:
        if st.button(
            "Open the engine →",
            key="pitch_cta_btn",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["pitch_seen"] = True
            st.rerun()
