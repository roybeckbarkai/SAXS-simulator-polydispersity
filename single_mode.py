# File: single_mode.py
# Description: Logic for single simulation runs with interactive parameter adjustments and visualizations.
# more info just for fun 

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sim_utils import run_simulation_core, get_distribution
from analysis_utils import perform_saxs_analysis, get_header_string, create_intensity_csv, create_distribution_csv, parse_saxs_file

def run():
    # ... (Keep existing Sidebar and Run Logic identical to previous versions) ...
    # Re-pasting the core logic to ensure context but focusing on the Download Section change.

    st.sidebar.title("Configuration")
    if st.sidebar.button("🏠 Return to Home"):
        st.session_state.page = 'home'
        st.rerun()

    sim_mode = st.sidebar.radio("Simulation Mode", ["Polydisperse Spheres", "Fixed-Length Polymers (IDP)"])
    mode_key = 'Sphere' if 'Sphere' in sim_mode else 'IDP'

    analysis_method = 'NNLS'
    if mode_key == 'Sphere':
        analysis_method = st.sidebar.selectbox("Analysis Method", ["Tomchuk (Invariants)", "NNLS (Distribution Fit)"])
        analysis_method = "Tomchuk" if "Tomchuk" in analysis_method else "NNLS"

    # ... (Data Loading Logic Same as Before) ...
    st.sidebar.header("Experimental Data")
    uploaded_file = st.sidebar.file_uploader("Load 1D Profile", type=['dat', 'out', 'txt', 'csv'])
    use_experimental = False
    q_meas, i_meas = None, None

    if uploaded_file is not None:
        q_load, i_load, err = parse_saxs_file(uploaded_file)
        if err: st.sidebar.error(err)
        else:
            use_experimental = st.sidebar.checkbox("Use Loaded Data", value=True)
            if use_experimental:
                q_meas, i_meas = q_load, i_load
                if st.session_state['last_filename'] != uploaded_file.name:
                    st.session_state['last_filename'] = uploaded_file.name
                    st.session_state['q_min'] = float(np.min(q_meas))
                    st.session_state['q_max'] = float(np.max(q_meas))
                    st.session_state['n_bins'] = len(q_meas)
                    res = perform_saxs_analysis(q_meas, i_meas, 'Gaussian', 4.0, mode_key, analysis_method, 50.0)
                    if res['Rg'] > 0: 
                        st.session_state['mean_rg'] = float(res['Rg'])
                        if res['Rg'] > 0: st.session_state['nnls_max_rg'] = float(round(5 * res['Rg'], 1))
                    st.rerun()

    def update_q_max():
        if st.session_state.mean_rg > 0:
            st.session_state.q_max = round(10.0 / st.session_state.mean_rg, 2)
    def update_q_max_and_basis():
        if st.session_state.mean_rg > 0:
            st.session_state.q_max = round(10.0 / st.session_state.mean_rg, 2)
            p = st.session_state.get('p_val', 0.3)
            st.session_state.nnls_max_rg = float(round(st.session_state.mean_rg * (1 + 6 * p), 1))

    st.sidebar.header("Sample Parameters")
    mean_rg = st.sidebar.number_input("Mean Rg (nm)", min_value=0.5, max_value=50.0, step=0.5, key='mean_rg', on_change=update_q_max_and_basis)
    p_val = st.sidebar.number_input("Polydispersity (p)", min_value=0.01, max_value=6.0, step=0.01, key='p_val', on_change=update_q_max_and_basis)
    dist_label = "Distribution Type" if mode_key == 'Sphere' else "Conformational Distribution"
    dist_type = st.sidebar.selectbox(dist_label, ['Gaussian', 'Lognormal', 'Schulz', 'Boltzmann', 'Triangular', 'Uniform'], key='dist_type')

    st.sidebar.header("NNLS Settings")
    nnls_max_rg = st.sidebar.number_input("Max Rg (Basis Set)", min_value=1.0, max_value=500.0, step=1.0, key='nnls_max_rg')

    st.sidebar.header("Instrument / Binning")
    pixels = st.sidebar.number_input("Detector Size (NxN)", value=1024, step=64, key='pixels')
    c_q1, c_q2 = st.sidebar.columns(2)
    with c_q1: q_min = st.sidebar.number_input("Min q", min_value=0.0, step=0.01, key='q_min')
    with c_q2: q_max = st.sidebar.number_input("Max q", min_value=0.01, step=0.1, key='q_max')
    n_bins = st.sidebar.number_input("1D Bins", min_value=10, step=10, key='n_bins')
    binning_mode = st.sidebar.selectbox("Binning Mode", ["Logarithmic", "Linear"], key='binning_mode')
    smearing = st.sidebar.number_input("Smearing (px)", value=2.0, step=0.5, key='smearing')

    st.sidebar.header("Flux & Noise")
    c_f1, c_f2 = st.sidebar.columns(2)
    with c_f1: flux_pre = st.number_input("Flux Coeff", 0.1, 9.9, 1.0, 0.1, key='flux_pre')
    with c_f2: flux_exp = st.number_input("Flux Exp", 1, 15, 6, 1, key='flux_exp')
    optimal_flux = st.sidebar.checkbox("Optimal Flux (No Noise)", value=False, key='optimal_flux')
    add_noise = st.sidebar.checkbox("Simulate Poisson Noise", value=True, disabled=optimal_flux, key='add_noise')

    # --- Run Simulation ---
    params = {
        'mean_rg': mean_rg, 'p_val': p_val, 'dist_type': dist_type, 'mode': mode_key,
        'pixels': pixels, 'q_min': q_min, 'q_max': q_max, 'n_bins': n_bins,
        'smearing': smearing, 'flux': flux_pre * (10**flux_exp), 'noise': (not optimal_flux and add_noise),
        'binning_mode': binning_mode
    }
    
    q_sim, i_sim, i_2d_final, r_vals, pdf_vals = run_simulation_core(params)
    
    if use_experimental and q_meas is not None:
        mask = (q_meas >= q_min) & (q_meas <= q_max)
        q_target = q_meas[mask]
        i_target = i_meas[mask]
    else:
        q_target = q_sim
        i_target = i_sim

    analysis_res = perform_saxs_analysis(q_target, i_target, dist_type, mean_rg, mode_key, analysis_method, nnls_max_rg)

    # ... (Visualization Code Identical to before, omitting for brevity) ...
    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        if use_experimental:
            st.subheader("Experimental Data Active"); st.info("Analyzing uploaded 1D data. 2D view disabled.")
        else:
            st.subheader("2D Detector")
            fig_2d = go.Figure(data=go.Heatmap(z=np.log10(np.maximum(i_2d_final, 1)), x=np.linspace(-q_max, q_max, pixels), y=np.linspace(-q_max, q_max, pixels), colorscale='Jet', colorbar=dict(title='log10(I)')))
            fig_2d.update_layout(xaxis_title='qx', yaxis_title='qy', width=500, height=500, margin=dict(l=40,r=40,t=20,b=40), yaxis=dict(scaleanchor="x", scaleratio=1))
            st.plotly_chart(fig_2d)

    with col_viz2:
        st.subheader("1D Profile Analysis")
        plot_opts = ["Log-Log", "Lin-Lin", "Guinier", "Kratky"]
        if mode_key == 'Sphere': plot_opts.append("Porod")
        plot_type = st.selectbox("Plot Type", plot_opts)
        
        fig_1d = go.Figure()
        label_str = 'Experimental' if use_experimental else 'Simulated'
        color_str = 'green' if use_experimental else 'blue'

        plot_x, plot_y = q_target, i_target
        x_type, y_type = 'linear', 'linear'
        x_label, y_label = 'q (nm⁻¹)', 'I(q)'
        
        if plot_type == "Log-Log": x_type, y_type = 'log', 'log'
        elif plot_type == "Guinier": 
            plot_x = q_target**2; plot_y = np.log(np.maximum(i_target, 1e-9))
            x_label = 'q²'; y_label = 'ln(I)'
        elif plot_type == "Porod": plot_y = i_target * (q_target**4); y_label = 'I · q⁴'
        elif plot_type == "Kratky": plot_y = i_target * (q_target**2); y_label = 'I · q²'

        fig_1d.add_trace(go.Scatter(x=plot_x, y=plot_y, mode='markers', name=label_str, marker=dict(color=color_str, size=4)))

        # Plot Fit Lines
        if analysis_method == 'Tomchuk':
             if 'I_fit_pdi' in analysis_res:
                 fit_y_pdi = analysis_res['I_fit_pdi']
                 fit_x = q_target
                 if plot_type == "Guinier": fit_x = q_target**2; fit_y_pdi = np.log(np.maximum(fit_y_pdi, 1e-9))
                 elif plot_type == "Porod": fit_y_pdi = fit_y_pdi * (q_target**4)
                 elif plot_type == "Kratky": fit_y_pdi = fit_y_pdi * (q_target**2)
                 fig_1d.add_trace(go.Scatter(x=fit_x, y=fit_y_pdi, mode='lines', name='PDI Fit', line=dict(color='orange', dash='dash', width=2)))
             
        elif 'I_fit' in analysis_res:
            fit_y = analysis_res['I_fit']
            fit_x = q_target
            if plot_type == "Guinier": fit_x = q_target**2; fit_y = np.log(np.maximum(fit_y, 1e-9))
            elif plot_type == "Porod": fit_y = fit_y * (q_target**4)
            elif plot_type == "Kratky": fit_y = fit_y * (q_target**2)
            fig_1d.add_trace(go.Scatter(x=fit_x, y=fit_y, mode='lines', name='NNLS Fit', line=dict(color='orange', dash='dash', width=2)))
            
        fig_1d.update_layout(xaxis_title=x_label, yaxis_title=y_label, xaxis_type=x_type, yaxis_type=y_type, width=500, height=450, margin=dict(l=40,r=40,t=20,b=40))
        st.plotly_chart(fig_1d)

    # --- Results & Distribution Visuals (Same as before) ---
    st.markdown("---"); st.subheader("Analysis Results")
    c1, c2, c3 = st.columns(3)
    input_label = "Ref (Sidebar)" if use_experimental else "Input"
    with c1:
        st.markdown("**Parameters**")
        st.metric(input_label + " Rg", f"{mean_rg:.2f} nm")
        if analysis_method == 'Tomchuk':
            st.metric("Rec. p (PDI)", f"{analysis_res.get('p_rec_pdi', 0):.3f}")
            st.metric("Rec. Rg (PDI)", f"{analysis_res.get('rg_num_rec_pdi', 0):.2f} nm")
        else:
            st.metric("NNLS Mean Rg", f"{analysis_res.get('rg_num_rec', 0):.2f} nm")
            st.metric("NNLS Width (p)", f"{analysis_res.get('p_rec', 0):.3f}")

    with c2:
        st.markdown("**Fits**")
        if analysis_method == 'Tomchuk':
             st.write(f"Chi2 (PDI): {analysis_res.get('chi2_pdi', 0):.4f}")
             st.write(f"Chi2 (PDI2): {analysis_res.get('chi2_pdi2', 0):.4f}")
        else:
             st.write(f"Chi2: {analysis_res.get('chi2', 0):.4f}")

    with c3:
        st.markdown("**Distribution**")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(x=r_vals, y=pdf_vals, mode='lines', name=input_label, line=dict(color='gray', dash='dash')))
        
        rec_dists_dl = {}
        if analysis_method == 'Tomchuk':
            m_r_pdi = analysis_res.get('rg_num_rec_pdi', 0) * np.sqrt(5.0/3.0)
            if m_r_pdi > 0:
                pdf_rec_pdi = get_distribution(dist_type, r_vals, m_r_pdi, analysis_res.get('p_rec_pdi', 0))
                fig_dist.add_trace(go.Scatter(x=r_vals, y=pdf_rec_pdi, mode='lines', name='PDI Rec.', line=dict(color='blue')))
                rec_dists_dl['pdi'] = pdf_rec_pdi
            
            m_r_pdi2 = analysis_res.get('rg_num_rec_pdi2', 0) * np.sqrt(5.0/3.0)
            if m_r_pdi2 > 0:
                pdf_rec_pdi2 = get_distribution(dist_type, r_vals, m_r_pdi2, analysis_res.get('p_rec_pdi2', 0))
                fig_dist.add_trace(go.Scatter(x=r_vals, y=pdf_rec_pdi2, mode='lines', name='PDI2 Rec.', line=dict(color='purple')))
                rec_dists_dl['pdi2'] = pdf_rec_pdi2
        else:
            if 'nnls_r' in analysis_res:
                fig_dist.add_trace(go.Scatter(x=analysis_res['nnls_r'], y=analysis_res['nnls_w'], mode='none', fill='tozeroy', name='NNLS Rec.', fillcolor='rgba(255, 165, 0, 0.5)', line=dict(color='orange')))
                rec_dists_dl['nnls_r'] = analysis_res['nnls_r']
                rec_dists_dl['nnls_w'] = analysis_res['nnls_w']
        
        fig_dist.update_layout(xaxis_title="Radius (nm)", yaxis_title="Prob", width=400, height=350, margin=dict(l=40,r=40,t=20,b=40))
        st.plotly_chart(fig_dist)

    # --- New Download Section (Split Buttons) ---
    params_dict = {'mean_rg': mean_rg, 'p_val': p_val, 'dist_type': dist_type, 'mode': mode_key, 'method': analysis_method}
    header = get_header_string(params_dict, analysis_res)
    
    # 1. Intensity CSV
    csv_intensity = create_intensity_csv(header, q_target, i_target, analysis_res, analysis_method)
    
    # 2. Distribution CSV
    csv_dist = create_distribution_csv(header, r_vals, pdf_vals, rec_dists_dl, params_dict)
    
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.download_button("Download Intensity Data (.csv)", csv_intensity, "saxs_intensity.csv", "text/csv", use_container_width=True)
    with c_d2:
        st.download_button("Download Distribution Data (.csv)", csv_dist, "saxs_distribution.csv", "text/csv", use_container_width=True)