import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sim_utils import run_simulation_core, get_distribution
from analysis_utils import perform_saxs_analysis, create_download_data, parse_saxs_file

def run():
    # Sidebar: Single Mode Specifics
    # Note: Global mode selections (Sim/IDP) are handled in streamlit_app.py or can be here.
    # To keep it consistent with request, let's keep all controls local if possible, 
    # but some might be shared. Let's assume controls are rendered here.

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

    st.sidebar.header("Experimental Data")
    uploaded_file = st.sidebar.file_uploader("Load 1D Profile", type=['dat', 'out', 'txt', 'csv'])
    use_experimental = False
    q_meas, i_meas = None, None

    if uploaded_file is not None:
        q_load, i_load, err = parse_saxs_file(uploaded_file)
        if err: st.sidebar.error(err)
        else:
            st.sidebar.success(f"Loaded {len(q_load)} points.")
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

    # Callbacks
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
    
    # Active Data
    if use_experimental and q_meas is not None:
        mask = (q_meas >= q_min) & (q_meas <= q_max)
        q_target = q_meas[mask]
        i_target = i_meas[mask]
    else:
        q_target = q_sim
        i_target = i_sim

    # Run Analysis
    analysis_res = perform_saxs_analysis(q_target, i_target, dist_type, mean_rg, mode_key, analysis_method, nnls_max_rg)

    # --- Visualization ---
    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        if use_experimental:
            st.subheader("Experimental Data Active")
            st.info("Analyzing uploaded 1D data. 2D view disabled.")
        else:
            st.subheader("2D Detector")
            fig_2d = go.Figure(data=go.Heatmap(
                z=np.log10(np.maximum(i_2d_final, 1)),
                x=np.linspace(-q_max, q_max, pixels),
                y=np.linspace(-q_max, q_max, pixels),
                colorscale='Jet',
                colorbar=dict(title='log10(I)')
            ))
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

        if 'I_fit' in analysis_res:
            fit_y = analysis_res['I_fit']
            fit_x = q_target
            if plot_type == "Guinier": fit_x = q_target**2; fit_y = np.log(np.maximum(fit_y, 1e-9))
            elif plot_type == "Porod": fit_y = fit_y * (q_target**4)
            elif plot_type == "Kratky": fit_y = fit_y * (q_target**2)
            fig_1d.add_trace(go.Scatter(x=fit_x, y=fit_y, mode='lines', name='Global Fit', line=dict(color='orange', dash='dash', width=2)))

        if plot_type == "Guinier":
            rg_f, g_f = analysis_res['Rg'], analysis_res['G']
            if rg_f > 0:
                x_line = np.linspace(0, (1.2/rg_f)**2, 50)
                y_line = np.log(g_f) - (rg_f**2/3.0)*x_line
                fig_1d.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Guinier Linear', line=dict(color='red', dash='dot')))
                fig_1d.update_xaxes(range=[0, (2.0/rg_f)**2])
                fig_1d.update_yaxes(range=[np.log(g_f)-3, np.log(g_f)+0.5])
        elif plot_type == "Porod" and analysis_method == 'Tomchuk':
            if 'B' in analysis_res and analysis_res['B'] > 0:
                fig_1d.add_hline(y=analysis_res['B'], line_dash="dot", line_color="red", annotation_text="B (Fit)")

        fig_1d.update_layout(xaxis_title=x_label, yaxis_title=y_label, xaxis_type=x_type, yaxis_type=y_type, width=500, height=450, margin=dict(l=40,r=40,t=20,b=40))
        st.plotly_chart(fig_1d)

    # --- Results ---
    st.markdown("---")
    st.subheader("Analysis Results")
    c1, c2, c3 = st.columns(3)

    # Define input_label HERE to be safe
    input_label = "Ref (Sidebar)" if use_experimental else "Input"

    with c1:
        st.markdown("**Parameters**")
        st.metric(input_label + " Rg", f"{mean_rg:.2f} nm")
        if analysis_method == 'Tomchuk':
            st.metric("Rec. p (PDI)", f"{analysis_res.get('p_rec_pdi', 0):.3f}")
            st.metric("Rec. Rg (PDI)", f"{analysis_res.get('rg_num_rec_pdi', 0):.2f} nm")
            st.metric("Rec. p (PDI₂)", f"{analysis_res.get('p_rec_pdi2', 0):.3f}")
            st.metric("Rec. Rg (PDI₂)", f"{analysis_res.get('rg_num_rec_pdi2', 0):.2f} nm")
        else:
            st.metric("NNLS Mean Rg", f"{analysis_res.get('rg_num_rec', 0):.2f} nm", delta=f"{analysis_res.get('rg_num_rec', 0) - mean_rg:.2f}")
            st.metric("NNLS Width (p)", f"{analysis_res.get('p_rec', 0):.3f}")
        if 'chi2' in analysis_res:
            st.metric("Chi-Square (Red.)", f"{analysis_res.get('chi2', 0):.2f}")

    with c2:
        st.markdown("**Invariants & Fit**")
        val_list = [f"{analysis_res.get('Rg', 0):.2f} nm", f"{analysis_res.get('G', 0):.2e}"]
        param_list = ["Rg (Guinier)", "G"]
        if analysis_method == 'Tomchuk':
            val_list.extend([f"{analysis_res.get('Q', 0):.2e}", f"{analysis_res.get('lc', 0):.2f} nm", f"{analysis_res.get('B', 0):.2e}"])
            param_list.extend(["Q", "lc", "B"])
        res_df = pd.DataFrame({"Parameter": param_list, "Value": val_list})
        st.dataframe(res_df, hide_index=True)

    with c3:
        st.markdown("**Recovered Distribution**")
        fig_dist = go.Figure()
        # Auto zoom logic same as before...
        pdf_max = np.max(pdf_vals)
        valid_range_mask = pdf_vals > (pdf_max * 1e-3)
        if np.any(valid_range_mask):
             r_plot_min = np.min(r_vals[valid_range_mask])
             r_plot_max = np.max(r_vals[valid_range_mask])
             if analysis_method == 'NNLS' and 'nnls_r' in analysis_res:
                  nnls_w = analysis_res['nnls_w']
                  nnls_r = analysis_res['nnls_r']
                  valid_nnls = nnls_w > (np.max(nnls_w) * 1e-3)
                  if np.any(valid_nnls):
                       r_plot_min = min(r_plot_min, np.min(nnls_r[valid_nnls]))
                       r_plot_max = max(r_plot_max, np.max(nnls_r[valid_nnls]))
             pad = (r_plot_max - r_plot_min) * 0.2
             fig_dist.update_xaxes(range=[max(0, r_plot_min - pad), r_plot_max + pad])

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

        fig_dist.update_layout(
            xaxis_title="Radius/Rg (nm)", 
            yaxis=dict(title="Probability Density"),
            yaxis2=dict(title="NNLS Weight", overlaying='y', side='right', showgrid=False),
            width=400, height=350, margin=dict(l=40,r=40,t=20,b=40),
            legend=dict(x=0.6, y=1.1, bgcolor='rgba(255,255,255,0.5)')
        )
        st.plotly_chart(fig_dist)

    params_dict = {'mean_rg': mean_rg, 'p_val': p_val, 'dist_type': dist_type, 'mode': mode_key, 'method': analysis_method}
    download_data = create_download_data(r_vals, pdf_vals, rec_dists_dl, analysis_res, params_dict, input_label, q_target, analysis_res.get('I_fit', np.zeros_like(q_target)))
    st.download_button(label="Download Analysis (CSV)", data=download_data, file_name="saxs_analysis.csv", mime="text/csv")