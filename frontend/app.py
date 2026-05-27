import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================
# 1. GLOBAL PAGE CONFIG
# ==========================
# This must be the first Streamlit command and only called once.
st.set_page_config(
    page_title="ClinTrialPredict | Predictive Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# 2. VARIANT ROUTING
# ==========================
variant = os.getenv("APP_VARIANT", "trial_audit").lower()

if variant == "trial_audit":
    import views.trial_audit as audit
    audit.init_session_state()
    audit.keep_filter_state_alive()
    audit.inject_custom_styles()
    audit.render_transition_overlay_hook()
    audit.route_app()

elif variant == "simulator":
    # Placeholder for Simulator variant
    st.title("Simulator Mode")
    st.info("Simulation variant is currently under development.")
    if st.button("Back to Audit"):
        os.environ["APP_VARIANT"] = "trial_audit"
        st.rerun()

elif variant == "serious_game":
    # Placeholder for Serious Game variant
    st.title("Serious Game Mode")
    st.info("Serious Game variant (Portfolio, Costs, Market) is currently under development.")
    if st.button("Back to Audit"):
        os.environ["APP_VARIANT"] = "trial_audit"
        st.rerun()

else:
    st.error(f"Unknown APP_VARIANT: {variant}")
    st.info("Defaulting to Trial Audit mode...")
    if st.button("Launch Audit"):
        os.environ["APP_VARIANT"] = "trial_audit"
        st.rerun()
