# File: single_mode.py
# Last Updated: Wednesday, February 11, 2026
# Description: Single run with PDI vs PDI2 consistency checks and Rg distribution plotting.

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sim_utils import run_simulation_core, get_distribution
from analysis_utils import perform_saxs_analysis, get_header_string, create_intensity_csv, create_distribution_csv, parse_saxs_file

def update_rg_from_r():
    if 'mean_r_input' in st.session_state:
        st.session_state.mean_rg = st.session_state.mean_r_input * np.sqrt(0.6)
        st.session_state.mean_rg_input = st.session_state.mean_rg
        auto_scale_q(st.session_state.mean_rg)

def update_r_from_rg():
    if 'mean_rg_input' in st.session_state:
        st.session_state.mean_rg = st.session_state.mean_rg_input
        st.session_state.mean_r_input = st.session_state.mean_rg / np.sqrt(0.6)
        auto_scale_q(st.session_state.mean_rg)

def auto_scale_q(rg):
    if rg > 0:
        new_q_min = max(0.001, 0.3/rg)
        new_q_max = max(2.5, 10.0/rg)
        st.session_state.q_min_input = float(f"{new_q_min:.4f}")
        st.session_state.q_max_input = float(f"{new_q_max:.2f}")

def run():
    st.sidebar.title("Configuration")
    if st.sidebar.button("🏠 Return to Home", width='stretch'):
        st.session_state.page = 'home'
        st.rerun()

    # Defaults
    if 'mean_rg' not in st.session_state: st.session_state.mean_rg = 4.0
    if 'mean_rg_input' not in st.session_state: st.session_state.mean_rg_input = 4.0
    if 'mean_r_input' not in st.session_state: st.session_state.mean_r_input = 4.0 / np.sqrt(0.6)
    if 'q_min_input' not in st.session_state: st.session_state.q_min_input = 0.01
    if 'q_max_input' not in st.session_state: st.session_state.q_max_input = 2.5

    # Sidebar
    st.sidebar.markdown("### Simulation Mode")
    sim_mode = st.sidebar.radio("Particle Type", ["Polydisperse Spheres", "Fixed-Length Polymers (IDP)"], label_visibility="collapsed")
    mode_key = 'Sphere' if 'Sphere' in sim_mode else 'IDP'

    analysis_method = 'NNLS'
    if mode_key == 'Sphere':
        st.sidebar.markdown("### Analysis Method")
        analysis_method_sel = st.sidebar.selectbox("Method", ["Tomchuk (PDI)", "Tomchuk (PDI2)", "NNLS"], label_visibility="collapsed")
        if "PDI)" in analysis_method_sel: analysis_method = "Tomchuk_PDI"
        elif "PDI2" in analysis_method_sel: analysis_method = "Tomchuk_PDI2"
        else: analysis_method = "NNLS"

    st.sidebar.markdown("---")
    st.sidebar.header("Parameters")

    if mode_key == 'Sphere':
        c1, c2 = st.sidebar.columns(2)
        with c1:
            st.number_input("Mean Radius R (nm)", 0.1, 1000.0, step=1.0, key='mean_r_input', on_change=update_rg_from_r, format="%.2f")
        with c2:
            st.number_input("Mean Rg (nm)", 0.1, 1000.0, step=1.0, key='mean_rg_input', on_change=update_r_from_rg, format="%.2f")
        p_val = st.sidebar.number_input("Polydispersity (p)", 0.01, 10.0, value=0.1, step=0.01)
        dist_type = st.sidebar.selectbox("Distribution Type", ['Gaussian', 'Lognormal', 'Schulz', 'Boltzmann', 'Triangular', 'Uniform'], index=0)
    else:
        st.session_state.mean_rg = st.sidebar.number_input("Mean Rg (nm)", 0.1, 1000.0, value=4.0, step=1.0)
        p_val = st.sidebar.number_input("Polydispersity (p)", 0.01, 10.0, value=0.1)
        dist_type = 'Gaussian'

    with st.sidebar.expander("Detector & Physics Settings", expanded=False):
        q_min = st.number_input("q_min (1/nm)", 0.0001, 1.0, key='q_min_input', format="%.4f")
        q_max = st.number_input("q_max (1/nm)", 0.1, 100.0, key='q_max_input', format="%.2f")
        n_bins = st.number_input("1D Bins", 50, 5000, 256)
        pixels = st.number_input("Detector Pixels (NxN)", 101, 4096, 1024, step=2)
        smearing = st.number_input("PSF Smearing (Pixels)", 0.0, 10.0, 2.0, step=0.1)
        flux = st.number_input("Flux (I0 Peak)", 1e3, 1e15, 1e5, format="%.1e")
        add_noise = st.checkbox("Add Poisson Noise", value=True)
        noise_level = 1.0 if add_noise else 0.0
        
        default_max = float(st.session_state.mean_rg * 10.0)
        max_rg_analysis = st.number_input("Max Analysis Rg (nm)", 1.0, 5000.0, default_max)
        binning_mode = st.radio("Binning", ["Log", "Linear"], index=0)

    uploaded_file = st.sidebar.file_uploader("Load 1D Profile", type=['dat', 'out', 'txt', 'csv'])
    use_experimental = False
    q_meas, i_meas = None, None
    if uploaded_file:
        q_meas, i_meas, err = parse_saxs_file(uploaded_file)
        if q_meas is not None: use_experimental = True

    # --- SIMULATION ---
    if mode_key == 'Sphere':
        sim_mean_r = st.session_state.mean_rg / np.sqrt(0.6)
    else:
        sim_mean_r = st.session_state.mean_rg

    sim_params = {
        'mean_rg': st.session_state.mean_rg, 
        'mean_r': sim_mean_r, 
        'p': p_val, 'dist_type': dist_type, 'mode_key': mode_key,
        'q_min': q_min, 'q_max': q_max, 'n_bins': int(n_bins),
        'pixels': int(pixels), 'smearing_sigma': smearing, 
        'flux': flux, 'noise_level': noise_level, 'binning_mode': binning_mode,
        'max_rg_analysis': max_rg_analysis, 'add_noise': add_noise
    }

    try:
        res = run_simulation_core(sim_params)
        q_sim, i_sim = res['q'], res['i_obs']
        i_2d, r_dist, pdf_dist = res['i_2d'], res['r_dist'], res['pdf_dist']
        final_max_rg = res.get('max_rg_analysis', max_rg_analysis)
        qa_metrics = res.get('qa_metrics', {})
    except Exception as e:
        st.error(f"Sim Failed: {e}")
        return

    # Analysis
    target_q, target_i = (q_meas, i_meas) if use_experimental else (q_sim, i_sim)
    ana_res = perform_saxs_analysis(target_q, target_i, analysis_method, mode_key, final_max_rg, dist_type=dist_type)

    # Report
    st.markdown("""
    <style>
    .report-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .report-table td, .report-table th { border: 1px solid #ddd; padding: 4px; text-align: center; }
    .report-table th { background-color: #f0f2f6; font-weight: 600; color: #31333F; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("**Analysis Report**")
    v_rg_g = f"{ana_res.get('rg_guinier', 0):.2f}"
    v_Q = f"{ana_res.get('Q', 0):.2e}"
    v_lc = f"{ana_res.get('l_c', 0):.2f}"
    v_B = f"{ana_res.get('B', 0):.2e}"
    
    v_rg_pdi = f"{ana_res.get('rg_rec', 0):.2f}"
    v_p_pdi = f"{ana_res.get('p_rec', 0):.3f}"
    v_chi_pdi = f"{ana_res.get('chi2_pdi', 0):.4f}"
    
    v_rg_pdi2 = f"{ana_res.get('rg_rec_pdi2', 0):.2f}"
    v_p_pdi2 = f"{ana_res.get('p_rec_pdi2', 0):.3f}"
    v_chi_pdi2 = f"{ana_res.get('chi2_pdi2', 0):.4f}"
    
    v_rg_rec = f"{ana_res.get('rg_rec', 0):.2f}"
    v_p_rec = f"{ana_res.get('p_rec', 0):.3f}"
    v_chi2 = f"{ana_res.get('chi2', 0):.4f}"

    if "Tomchuk" in analysis_method:
        html = f"""
        <table class="report-table">
            <tr><th>Metric</th><th>Guinier Rg</th><th>Invariant Q</th><th>Corr. lc</th><th>Porod B</th><th>G (I0)</th></tr>
            <tr><td>Value</td><td>{v_rg_g} nm</td><td>{v_Q}</td><td>{v_lc} nm</td><td>{v_B}</td><td>{ana_res.get('G',0):.2e}</td></tr>
        </table>
        <table class="report-table" style="margin-top: 5px;">
            <tr><th>Method</th><th>Rec. Scat Rg</th><th>Rec. p</th><th>Chi2</th></tr>
            <tr><td><b>PDI (Vol)</b></td><td>{v_rg_pdi} nm</td><td>{v_p_pdi}</td><td>{v_chi_pdi}</td></tr>
            <tr><td><b>PDI2 (Cor)</b></td><td>{v_rg_pdi2} nm</td><td>{v_p_pdi2}</td><td>{v_chi_pdi2}</td></tr>
        </table><br>
        """
    else:
        html = f"""
        <table class="report-table">
            <tr><th>Guinier Rg</th><th>Inv. Q</th><th>lc</th><th>Porod B</th><th>Rec. Rg</th><th>Rec. p</th><th>Chi2</th></tr>
            <tr><td>{v_rg_g}</td><td>{v_Q}</td><td>{v_lc}</td><td>{v_B}</td><td><b>{v_rg_rec}</b></td><td><b>{v_p_rec}</b></td><td>{v_chi2}</td></tr>
        </table><br>
        """
    st.markdown(html, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["1D Profile", "2D Detector", "Distributions", "QA & Diagnostics"])

    with tab1:
        c1, c2 = st.columns([1, 4])
        with c1:
            ptype = st.radio("Plot Type", ["Log-Log", "Lin-Lin", "Guinier", "Porod", "Kratky"])
            show_fit = st.checkbox("Show Fit", True)
        
        with c2:
            fig = go.Figure()
            def transform(q, i, type):
                if type == "Log-Log": return q, i, "log", "log", "q", "I"
                if type == "Lin-Lin": return q, i, "linear", "linear", "q", "I"
                if type == "Guinier": 
                    ref_rg = st.session_state.mean_rg
                    cutoff = 1.3 / ref_rg if ref_rg > 0 else 999
                    mask = (q > 0) & (q < cutoff)
                    return q[mask]**2, np.log(i[mask]+1e-9), "linear", "linear", "q^2", "ln(I)"
                if type == "Porod": return q, i*q**4, "linear", "linear", "q", "I*q^4"
                if type == "Kratky": return q, i*q**2, "linear", "linear", "q", "I*q^2"
                return q, i, "log", "log", "q", "I"

            xq, yq, xs, ys, xl, yl = transform(q_sim, i_sim, ptype)
            fig.add_trace(go.Scatter(x=xq, y=yq, mode='markers', name='Simulated', marker=dict(size=4, opacity=0.5)))

            if show_fit:
                if 'I_fit_pdi' in ana_res:
                    fx, fy, _, _, _, _ = transform(q_sim, ana_res['I_fit_pdi'], ptype)
                    fig.add_trace(go.Scatter(x=fx, y=fy, mode='lines', name='Fit (PDI)', line=dict(color='red')))
                if 'I_fit_pdi2' in ana_res:
                    fx, fy, _, _, _, _ = transform(q_sim, ana_res['I_fit_pdi2'], ptype)
                    fig.add_trace(go.Scatter(x=fx, y=fy, mode='lines', name='Fit (PDI2)', line=dict(color='purple')))
                if 'I_fit' in ana_res:
                    fx, fy, _, _, _, _ = transform(q_sim, ana_res['I_fit'], ptype)
                    fig.add_trace(go.Scatter(x=fx, y=fy, mode='lines', name='Fit (NNLS)', line=dict(color='orange')))

            if ptype == "Porod":
                B = ana_res.get('B', 0)
                fig.add_hline(y=B, line_dash="dash", line_color="green", annotation_text=f"B={B:.2e}")

            if ptype == "Guinier":
                rg_g = ana_res.get('rg_guinier', 0)
                G0 = ana_res.get('G', 0)
                if rg_g > 0:
                    y_g = np.log(G0) - (rg_g**2/3)*xq
                    fig.add_trace(go.Scatter(x=xq, y=y_g, mode='lines', name=f'Guinier Fit', line=dict(color='blue', dash='dash')))

            fig.update_layout(xaxis_title=xl, yaxis_title=yl, xaxis_type=xs, yaxis_type=ys, height=500)
            st.plotly_chart(fig, width='stretch')
            
            h = get_header_string(st.session_state.mean_rg, p_val, dist_type, ana_res)
            csv = create_intensity_csv(h, q_sim, i_sim, ana_res, analysis_method)
            st.download_button("Download Data", csv, "saxs_data.csv", "text/csv")

    with tab2:
        if i_2d is not None:
            fig2d = go.Figure(go.Heatmap(z=np.log10(i_2d+1), colorscale='Viridis'))
            fig2d.update_layout(height=600, width=600, yaxis=dict(scaleanchor="x"))
            st.plotly_chart(fig2d, width='stretch')

    with tab3:
        figd = go.Figure()
        
        # Scale input distribution to Rg axis for display
        # r_dist is Geometric R. R_g = R * sqrt(0.6).
        # We want to plot P(Rg).
        # x_new = x_old * 0.775
        # y_new = y_old / 0.775 (to conserve area)
        factor = np.sqrt(0.6)
        
        rg_dist_axis = r_dist * factor
        pdf_rg = pdf_dist / factor
        
        figd.add_trace(go.Scatter(x=rg_dist_axis, y=pdf_rg, name='Input PDF (Rg)', fill='tozeroy'))
        
        if 'r_geo_rec' in ana_res:
             rec_r = ana_res['r_geo_rec']
             rec_p = ana_res.get('p_rec', 0.1)
             # Generate geometric pdf, then scale to Rg
             pdf_rec = get_distribution(dist_type, r_dist, rec_r, rec_p)
             pdf_rec_scaled = pdf_rec / factor
             
             figd.add_trace(go.Scatter(x=rg_dist_axis, y=pdf_rec_scaled, name=f'Rec PDI (p={rec_p:.2f})', line=dict(dash='dot', color='red')))
        
        if 'r_geo_rec_pdi2' in ana_res:
             rec_r2 = ana_res['r_geo_rec_pdi2']
             rec_p2 = ana_res.get('p_rec_pdi2', 0.1)
             pdf_rec2 = get_distribution(dist_type, r_dist, rec_r2, rec_p2)
             pdf_rec2_scaled = pdf_rec2 / factor
             figd.add_trace(go.Scatter(x=rg_dist_axis, y=pdf_rec2_scaled, name=f'Rec PDI2 (p={rec_p2:.2f})', line=dict(dash='dot', color='purple')))

        figd.update_layout(xaxis_title="Radius of Gyration Rg (nm)", yaxis_title="Probability Density P(Rg)")
        st.plotly_chart(figd, width='stretch')

    with tab4:
        st.subheader("Diagnostics: PDI vs PDI2 Consistency")
        
        p1 = ana_res.get('p_rec', 0)
        p2 = ana_res.get('p_rec_pdi2', 0)
        diff = abs(p1 - p2)
        status = "✅ Consistent" if diff < 0.05 else "⚠️ Discrepancy"
        
        st.markdown(f"""
        **Test for Ideal Data (No Noise/Smearing)**
        - **PDI1 (Volume Method)**: p = {p1:.3f}
        - **PDI2 (Correlation Method)**: p = {p2:.3f}
        - **Difference**: {diff:.4f}  --  **{status}**
        
        *Note: If discrepancy exists with ideal data, it indicates specific moments of the chosen distribution (e.g. Gaussian) diverge from the theoretical ratios used in the solver.*
        """)
        
        # ... existing debug plots ...
        d_g = ana_res.get('debug_guinier', {})
        if d_g.get('x') is not None and len(d_g['x']) > 0:
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(x=d_g['x'], y=d_g['y'], mode='markers', name='Points'))
            fig_g.add_trace(go.Scatter(x=d_g['x'], y=d_g['fit_y'], mode='lines', name='Fit', line=dict(color='red')))
            fig_g.update_layout(title="Guinier Fit Check", xaxis_title="q^2", yaxis_title="ln(I)", height=300)
            st.plotly_chart(fig_g, width='stretch')

if __name__ == "__main__":
    run()