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

# Reverse mappings for initialization
REV_MODE_MAP = {v: k for k, v in MODE_MAP.items()}
REV_DIST_MAP = {v: k for k, v in DIST_MAP.items()}
REV_METHOD_MAP = {v: k for k, v in METHOD_MAP.items()}

def expand_batch_parameters(df):
    """
    Expands rows containing lists [val1, val2] into individual jobs.
    """
    all_jobs = []
    # Force conversion to string to handle potential mixed types
    temp_df = df.astype(str)
    
    for _, row in temp_df.iterrows():
        keys = row.index.tolist()
        vals = []
        for k in keys:
            val_str = row[k].strip()
            if val_str.startswith('[') and val_str.endswith(']'):
                try:
                    parsed_list = ast.literal_eval(val_str)
                    vals.append(parsed_list if isinstance(parsed_list, list) else [parsed_list])
                except:
                    vals.append([val_str])
            else:
                vals.append([val_str])
        
        for combo in itertools.product(*vals):
            all_jobs.append(dict(zip(keys, combo)))
    return all_jobs

def run():
    st.title("Batch Simulation & Recovery Analysis")
    
    if st.sidebar.button("🏠 Return to Home"):
        st.session_state.page = 'home'
        st.rerun()

    st.markdown("""
    Define a sweep of simulations. Enter single values or lists like `[4.0, 5.0, 6.0]`.
    - **Mode**: S (Sphere), P (IDP) | **Dist**: G, L, S, B, T, U | **Method**: T (Tomchuk), N (NNLS)
    """)

    if 'batch_df' not in st.session_state:
        st.session_state.batch_df = pd.DataFrame([{
            "mode": REV_MODE_MAP.get(st.session_state.get('mode_key', 'Sphere'), 'S'),
            "mean_rg": str(st.session_state.get('mean_rg', 4.0)),
            "p_val": "[0.1, 0.2, 0.3]",
            "dist": REV_DIST_MAP.get(st.session_state.get('dist_type', 'Gaussian'), 'G'),
            "method": "T",
            "smearing": "0.0",
            "q_max": str(st.session_state.get('q_max', 2.5)),
            "n_bins": str(st.session_state.get('n_bins', 512))
        }])

    # Fixed: Use width='stretch'
    edited_df = st.data_editor(st.session_state.batch_df, num_rows="dynamic", width="stretch")
    st.session_state.batch_df = edited_df

    # Fixed: Use width='stretch'
    if st.button("🚀 Execute Batch Queue", width='stretch'):
        jobs = expand_batch_parameters(edited_df)
        n_jobs = len(jobs)
        
        if n_jobs == 0:
            st.error("No jobs found in the queue.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for i, job in enumerate(jobs):
                status_text.text(f"Processing Job {i+1}/{n_jobs}...")
                
                # Extract and parse params
                try:
                    m_rg = float(job.get('mean_rg', 4.0))
                    p_v = float(job.get('p_val', 0.2))
                    smear = float(job.get('smearing', 0.0))
                    q_max = float(job.get('q_max', 2.5))
                    n_bins = int(float(job.get('n_bins', 512)))
                    mode_key = MODE_MAP.get(job.get('mode', 'S'), 'Sphere')
                    dist_type = DIST_MAP.get(job.get('dist', 'G'), 'Gaussian')
                    method = METHOD_MAP.get(job.get('method', 'T'), 'Tomchuk')

                    # CONSTRUCTION OF SIMULATION PARAMETERS DICTIONARY
                    sim_params = {
                        'mean_rg': m_rg,
                        'mean_r': m_rg,          
                        'p_val': p_v,
                        'p': p_v,                
                        'dist_type': dist_type,
                        'dist': dist_type,       
                        'q_min': 0.01,
                        'q_max': q_max,
                        'n_bins': n_bins,
                        'smearing_sigma': smear,
                        'smearing': smear,       
                        'flux': 1e8,
                        'mode_key': mode_key,
                        'mode': mode_key,        
                        'pixels': 401,           
                        'noise_level': 0.0,      
                        'noise': 0.0,            
                        'binning_mode': 'Log',   
                        'binning': 'Log'         
                    }

                    # Execute Core Simulation
                    raw_res = run_simulation_core(sim_params)

                    # ADAPTER: Convert tuple result to dictionary if necessary
                    # The error "tuple indices must be integers" happens because 
                    # run_simulation_core returns (q, i, r, pdf) instead of a dict.
                    if isinstance(raw_res, tuple):
                        if len(raw_res) >= 4:
                            sim_res = {
                                'q': raw_res[0],
                                'i_obs': raw_res[1],
                                'r_dist': raw_res[2],
                                'pdf_dist': raw_res[3]
                            }
                        elif len(raw_res) == 2:
                            sim_res = {
                                'q': raw_res[0],
                                'i_obs': raw_res[1],
                                'r_dist': None,
                                'pdf_dist': None
                            }
                        else:
                            raise ValueError(f"Unexpected tuple length from simulation: {len(raw_res)}")
                    else:
                        sim_res = raw_res

                except Exception as e:
                    st.error(f"Simulation failed for job {i+1}: {e}")
                    break

                # Execute Analysis
                try:
                    analysis_res = perform_saxs_analysis(
                        q=sim_res['q'], i_obs=sim_res['i_obs'], 
                        method=method, mode_key=mode_key,
                        nnls_max_rg=st.session_state.get('nnls_max_rg', 30.0)
                    )

                    rg_rec = analysis_res.get('rg_rec', np.nan)
                    p_rec = analysis_res.get('p_rec', np.nan)
                    err_rg = ((rg_rec - m_rg) / m_rg) if m_rg != 0 else np.nan
                    err_p = ((p_rec - p_v) / p_v) if p_v != 0 else np.nan

                    res_entry = {
                        "Job_ID": i + 1, "mode": mode_key, "dist": dist_type, "method": method,
                        "mean_rg": m_rg, "p_val": p_v, "smearing": smear,
                        "Rg_Rec": rg_rec, "p_Rec": p_rec,
                        "Rel_Err_Rg": err_rg, "Rel_Err_p": err_p, "Chi2": analysis_res.get('chi2', np.nan)
                    }
                    results_list.append(res_entry)

                    prefix = f"job_{i+1:03d}_{mode_key}_{dist_type}_{m_rg}nm"
                    header = get_header_string(m_rg, p_v, dist_type, analysis_res)
                    zip_file.writestr(f"{prefix}_int.csv", create_intensity_csv(header, sim_res['q'], sim_res['i_obs'], analysis_res, method))
                    zip_file.writestr(f"{prefix}_dist.csv", create_distribution_csv(header, sim_res['r_dist'], sim_res['pdf_dist'], analysis_res, {"method": method}))
                
                except Exception as e:
                    st.error(f"Analysis failed for job {i+1}: {e}")
                    break

                progress_bar.progress((i + 1) / n_jobs)

            if results_list:
                summary_df = pd.DataFrame(results_list)
                zip_file.writestr("summary_report.csv", summary_df.to_csv(index=False))
                st.session_state.batch_results_df = summary_df
                st.session_state.batch_zip = zip_buffer.getvalue()

        status_text.success(f"Batch completed!")

    if 'batch_results_df' in st.session_state:
        res_df = st.session_state.batch_results_df
        st.divider()
        st.subheader("Results Summary")
        
        # Fixed: Use width='stretch'
        st.dataframe(res_df, width="stretch")

        # Fixed: Use width='stretch'
        st.download_button(
            label="📥 Download Full Batch Results (ZIP)",
            data=st.session_state.batch_zip,
            file_name=f"saxs_batch_{datetime.now().strftime('%H%M%S')}.zip",
            mime="application/zip",
            width='stretch'
        )

        st.subheader("Error Analysis Visualization")
        ignore_cols = ["Job_ID", "Rg_Rec", "p_Rec", "Rel_Err_Rg", "Rel_Err_p", "Chi2"]
        varying_cols = [c for c in res_df.columns if c not in ignore_cols and res_df[c].nunique() > 1]
        
        x_axis = st.selectbox("Select X-Axis", varying_cols if varying_cols else ["Job_ID"], key="batch_x_selector")

        if x_axis in res_df.columns:
            t1, t2 = st.tabs(["Rg Error", "p Error"])
            with t1:
                fig = px.scatter(res_df, x=x_axis, y="Rel_Err_Rg", color="method", symbol="mode", 
                                 hover_data=["mean_rg", "p_val"], title="Relative Error in Rg")
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                # Fixed: Use width='stretch'
                st.plotly_chart(fig, width='stretch')
            with t2:
                fig = px.scatter(res_df, x=x_axis, y="Rel_Err_p", color="method", symbol="mode", 
                                 hover_data=["mean_rg", "p_val"], title="Relative Error in p")
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                # Fixed: Use width='stretch'
                st.plotly_chart(fig, width='stretch')

if __name__ == "__main__":
    run()