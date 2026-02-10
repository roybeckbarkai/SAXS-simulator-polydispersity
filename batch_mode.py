# File: batch_mode.py
# Last Updated: Tuesday, February 10, 2026
# Description: Logic for running batch parameter sweeps with short-code inputs and error analysis plotting.

import streamlit as st
import pandas as pd
import io
import zipfile
import ast
import itertools
import numpy as np
import plotly.express as px
from datetime import datetime
from sim_utils import run_simulation_core, get_distribution
from analysis_utils import perform_saxs_analysis, get_header_string, create_intensity_csv, create_distribution_csv

# Mappings for Short Codes -> Full Parameter Names
MODE_MAP = {'S': 'Sphere', 'P': 'IDP'}
DIST_MAP = {'G': 'Gaussian', 'L': 'Lognormal', 'S': 'Schulz', 'B': 'Boltzmann', 'T': 'Triangular', 'U': 'Uniform'}
METHOD_MAP = {'T': 'Tomchuk', 'N': 'NNLS'}
BIN_MAP = {'Log': 'Logarithmic', 'Lin': 'Linear'}

# Reverse mappings for initialization
REV_MODE_MAP = {v: k for k, v in MODE_MAP.items()}
REV_DIST_MAP = {v: k for k, v in DIST_MAP.items()}
REV_METHOD_MAP = {v: k for k, v in METHOD_MAP.items()}
REV_BIN_MAP = {v: k for k, v in BIN_MAP.items()}

def expand_batch_parameters(df):
    all_jobs = []
    # Force conversion to string
    df = df.astype(str)
    
    for _, row in df.iterrows():
        base_params = row.to_dict()
        list_params = {}
        single_params = {}
        
        for k, v in base_params.items():
            val_str = str(v).strip()
            if val_str.startswith('[') and val_str.endswith(']'):
                try:
                    val_list = ast.literal_eval(val_str)
                    if isinstance(val_list, list):
                        list_params[k] = val_list
                        continue
                except:
                    pass
            single_params[k] = v
            
        if not list_params:
            all_jobs.append(single_params)
        else:
            keys = list(list_params.keys())
            vals = list(list_params.values())
            for combination in itertools.product(*vals):
                job = single_params.copy()
                for k, val in zip(keys, combination):
                    job[k] = val
                all_jobs.append(job)
    
    return pd.DataFrame(all_jobs)

def run():
    st.header("Batch Simulation Runner")
    if st.button("🏠 Return to Home"):
        st.session_state.page = 'home'
        st.rerun()

    # Determine default short codes based on current session
    curr_mode = st.session_state.get('mode_key', 'Sphere')
    curr_dist = st.session_state.get('dist_type', 'Gaussian')
    
    curr_method = "NNLS"
    if curr_mode == 'Sphere':
         curr_method = "NNLS" 

    default_row = {
        "mode (S/P)": str(REV_MODE_MAP.get(curr_mode, 'S')),
        "dist (G/L/S/B/T/U)": str(REV_DIST_MAP.get(curr_dist, 'G')),
        "mean_rg": str(st.session_state.get('mean_rg', 4.0)),
        "p_val": str(st.session_state.get('p_val', 0.3)),
        "pixels": str(1024),
        "q_min": str(0.0),
        "q_max": str(st.session_state.get('q_max', 2.5)),
        "n_bins": str(512),
        "binning (Log/Lin)": "Log",
        "smearing": str(2.0),
        "flux": str(1e6),
        "noise": "True",
        "method (T/N)": str(REV_METHOD_MAP.get(curr_method, 'N')),
        "nnls_max_rg": str(st.session_state.get('nnls_max_rg', 30.0))
    }
    
    if 'batch_df' not in st.session_state:
        st.session_state.batch_df = pd.DataFrame([default_row])

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        uploaded_batch = st.file_uploader("Upload Batch CSV", type=['csv'])
        if uploaded_batch:
            try: 
                st.session_state.batch_df = pd.read_csv(uploaded_batch, dtype=str)
                st.success("Batch Loaded")
            except: st.error("Invalid CSV")
    with c_b2:
        st.download_button("Download Template", 
                           pd.DataFrame([default_row]).to_csv(index=False), 
                           "batch_template.csv", "text/csv")
        
    st.info("Tip: Enter lists like `[0.1, 0.3]` in parameter cells to run combinations. Use codes: S=Sphere, P=Polymer | G=Gaussian... | T=Tomchuk, N=NNLS")
    
    df_to_edit = st.session_state.batch_df.astype(str)
    # Using 'width'='stretch' logic for data editor if version supports, otherwise just default
    edited_df = st.data_editor(df_to_edit, num_rows="dynamic")
    st.session_state.batch_df = edited_df

    if st.button("Execute Batch Queue"):
        expanded_df = expand_batch_parameters(edited_df)
        st.write(f"Queue size: {len(expanded_df)} simulations")
        
        progress_bar = st.progress(0)
        zip_buffer = io.BytesIO()
        summary_results = []
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("batch_summary_expanded.csv", expanded_df.to_csv(index=False))
            
            for i, row in expanded_df.iterrows():
                row_dict = row.to_dict()
                
                try:
                    params = {
                        'mode': MODE_MAP.get(row_dict.get('mode (S/P)', 'S'), 'Sphere'),
                        'dist_type': DIST_MAP.get(row_dict.get('dist (G/L/S/B/T/U)', 'G'), 'Gaussian'),
                        'mean_rg': float(row_dict['mean_rg']),
                        'p_val': float(row_dict['p_val']),
                        'pixels': int(float(row_dict['pixels'])),
                        'q_min': float(row_dict['q_min']),
                        'q_max': float(row_dict['q_max']),
                        'n_bins': int(float(row_dict['n_bins'])),
                        'binning_mode': BIN_MAP.get(row_dict.get('binning (Log/Lin)', 'Log'), 'Logarithmic'),
                        'smearing': float(row_dict['smearing']),
                        'flux': float(row_dict['flux']),
                        'noise': str(row_dict['noise']).lower() in ['true', '1', 't', 'yes'],
                        'method': METHOD_MAP.get(row_dict.get('method (T/N)', 'N'), 'NNLS'),
                        'nnls_max_rg': float(row_dict['nnls_max_rg'])
                    }

                    q_sim, i_sim, _, r_vals, pdf_vals = run_simulation_core(params)
                    
                    res = perform_saxs_analysis(q_sim, i_sim, 
                                                params['dist_type'], 
                                                params['mean_rg'], 
                                                params['mode'], 
                                                params['method'], 
                                                params['nnls_max_rg'])
                    
                    rec_dists = {}
                    if params['method'] == 'Tomchuk':
                         m_pdi = res.get('rg_num_rec_pdi', params['mean_rg']) * np.sqrt(5.0/3.0)
                         if m_pdi > 0:
                             rec_dists['pdi'] = get_distribution(params['dist_type'], r_vals, m_pdi, res.get('p_rec_pdi', 0))
                         
                         m_pdi2 = res.get('rg_num_rec_pdi2', params['mean_rg']) * np.sqrt(5.0/3.0)
                         if m_pdi2 > 0:
                             rec_dists['pdi2'] = get_distribution(params['dist_type'], r_vals, m_pdi2, res.get('p_rec_pdi2', 0))
                    elif 'nnls_r' in res:
                         rec_dists['nnls_r'] = res['nnls_r']
                         rec_dists['nnls_w'] = res['nnls_w']

                    header = get_header_string(params, res)
                    intensity_csv = create_intensity_csv(header, q_sim, i_sim, res, params['method'])
                    dist_csv = create_distribution_csv(header, r_vals, pdf_vals, rec_dists, params)
                    
                    zf.writestr(f"run_{i}_intensity.csv", intensity_csv)
                    zf.writestr(f"run_{i}_distribution.csv", dist_csv)
                    
                    summary_row = row_dict.copy()
                    
                    rec_p = res.get('p_rec', 0) if params['method'] == 'NNLS' else res.get('p_rec_pdi', 0)
                    rec_rg = res.get('rg_num_rec', 0) if params['method'] == 'NNLS' else res.get('rg_num_rec_pdi', 0)
                    
                    rec_p_2 = res.get('p_rec_pdi2', 0) if params['method'] == 'Tomchuk' else 0
                    rec_rg_2 = res.get('rg_num_rec_pdi2', 0) if params['method'] == 'Tomchuk' else 0

                    p_true = params['p_val']
                    rg_true = params['mean_rg']
                    
                    err_p = (rec_p - p_true)/p_true if p_true != 0 else 0
                    err_rg = (rec_rg - rg_true)/rg_true if rg_true != 0 else 0

                    summary_row.update({
                        'Recovered_p': rec_p,
                        'Recovered_Rg': rec_rg,
                        'Rel_Err_p': err_p,
                        'Rel_Err_Rg': err_rg,
                        'Chi2': res.get('chi2', 0)
                    })
                    
                    if params['method'] == 'Tomchuk':
                        err_p2 = (rec_p_2 - p_true)/p_true if p_true != 0 else 0
                        err_rg2 = (rec_rg_2 - rg_true)/rg_true if rg_true != 0 else 0
                        summary_row.update({
                             'Recovered_p_PDI2': rec_p_2,
                             'Recovered_Rg_PDI2': rec_rg_2,
                             'Rel_Err_p_PDI2': err_p2,
                             'Rel_Err_Rg_PDI2': err_rg2,
                        })

                    summary_results.append(summary_row)

                except Exception as e:
                    zf.writestr(f"run_{i}_error.txt", str(e))
                
                progress_bar.progress((i + 1) / len(expanded_df))
            
            if summary_results:
                summary_df = pd.DataFrame(summary_results)
                zf.writestr("batch_results_summary.csv", summary_df.to_csv(index=False))

        if summary_results:
            st.success("Batch Complete!")
            st.markdown("### Batch Analysis Visualization")
            
            res_df = pd.DataFrame(summary_results)
            for col in res_df.columns:
                res_df[col] = pd.to_numeric(res_df[col], errors='ignore')
            
            result_cols = ['Recovered_p', 'Recovered_Rg', 'Rel_Err_p', 'Rel_Err_Rg', 'Chi2', 
                           'Recovered_p_PDI2', 'Recovered_Rg_PDI2', 'Rel_Err_p_PDI2', 'Rel_Err_Rg_PDI2']
            varying_cols = [c for c in res_df.columns if res_df[c].nunique() > 1 and c not in result_cols]
            
            if varying_cols:
                x_axis = st.selectbox("Select X-Axis Variable", varying_cols, index=0)
            else:
                x_axis = res_df.columns[0]

            tab1, tab2 = st.tabs(["Rg Recovery Error", "Polydispersity (p) Error"])
            
            with tab1:
                try:
                    fig_rg = px.scatter(res_df, x=x_axis, y="Rel_Err_Rg", 
                                        color="method (T/N)", hover_data=["mean_rg", "p_val", "smearing"],
                                        title=f"Relative Error in Rg vs {x_axis}")
                    fig_rg.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_rg) 
                except Exception as e:
                    st.warning(f"Could not plot Rg error: {e}")
                
            with tab2:
                try:
                    fig_p = px.scatter(res_df, x=x_axis, y="Rel_Err_p", 
                                       color="method (T/N)", hover_data=["mean_rg", "p_val", "smearing"],
                                       title=f"Relative Error in p vs {x_axis}")
                    fig_p.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_p) 
                except Exception as e:
                     st.warning(f"Could not plot p error: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download Batch Results (ZIP)", zip_buffer.getvalue(), f"saxs_batch_{timestamp}.zip", "application/zip")