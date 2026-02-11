# File: streamlit_app.py
# Last Updated: Tuesday, February 10, 2026
# Description: Main entry point and navigation controller with hot-reloading.

import streamlit as st
import importlib

# Set config MUST be the first Streamlit command
st.set_page_config(page_title="SAXS Simulator", layout="wide", page_icon="⚛️")

# --- Module Management ---
# We import and force reload these modules to ensure updates in single_mode.py 
# and batch_mode.py are reflected immediately without restarting the server.
import single_mode
import batch_mode
import analysis_utils
import sim_utils

try:
    importlib.reload(sim_utils)
    importlib.reload(analysis_utils)
    importlib.reload(single_mode)
    importlib.reload(batch_mode)
except Exception as e:
    st.error(f"Module reload failed: {e}")

# --- Global Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if 'init' not in st.session_state:
    st.session_state['init'] = True
    # Default parameters
    st.session_state['mean_rg'] = 4.0
    st.session_state['p_val'] = 0.3
    st.session_state['q_max'] = 2.5
    st.session_state['n_bins'] = 512
    st.session_state['nnls_max_rg'] = 30.0
    st.session_state['mode_key'] = 'Sphere' 
    st.session_state['dist_type'] = 'Gaussian'

# --- Navigation Logic ---
if st.session_state.page == 'home':
    st.title("SAXS Simulator & Analysis Tool")
    st.markdown("### Choose your workflow")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Single Run & Interactive Analysis**")
        st.markdown("Simulate one dataset, adjust parameters in real-time, and visualize 1D/2D results interactively.")
        
        # Updated to width='stretch' to fix deprecation warning
        if st.button("Start Single Mode", key="btn_single", width='stretch'):
            st.session_state.page = 'single'
            st.rerun()
            
    with col2:
        st.success("**Batch Processing**")
        st.markdown("Run multiple simulations automatically by defining parameter sweeps in a table.")
        
        # Updated to width='stretch' to fix deprecation warning
        if st.button("Start Batch Mode", key="btn_batch", width='stretch'):
            st.session_state.page = 'batch'
            st.rerun()

    st.markdown("---")
    st.markdown("""
    **Modules:**
    * **Single Mode:** Real-time feedback, detailed plots, 2D/1D toggles.
    * **Batch Mode:** Upload CSV or use grid editor to run permutations of parameters. Downloads results as ZIP.
    """)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("© Roy Beck Barkai - 2026")

elif st.session_state.page == 'single':
    single_mode.run()

elif st.session_state.page == 'batch':
    batch_mode.run()