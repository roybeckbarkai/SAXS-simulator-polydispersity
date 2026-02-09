import streamlit as st
import pandas as pd
import io
import zipfile
import ast
import itertools
import numpy as np
from datetime import datetime
from sim_utils import run_simulation_core, get_distribution
from analysis_utils import perform_saxs_analysis, create_download_data

def expand_batch_parameters(df):
    all_jobs = []
    for _, row in df.iterrows():
        base_params = row.to_dict()
        list_params = {}
        single_params = {}
        
        for k, v in base_params.items():
            if isinstance(v, str) and v.strip().startswith('[') and v.strip().endswith(']'):
                try:
                    val_list = ast.literal_eval(v)
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

    # Default based on current session if available
    default_row = {
        "mode": st.session_state.get('mode_key', 'Sphere'),
        "dist_type": st.session_state.get('dist_type', 'Gaussian'),
        "mean_rg": st.session_state.get('mean_rg', 4.0),
        "p_val": st.session_state.get('p_val', 0.3),
        "pixels": 1024,
        "q_min": 0.0,
        "q_max": st.session_state.get('q_max', 2.5),
        "n_bins": 512,
        "binning_mode": "Logarithmic",
        "smearing": 2.0,
        "flux": 1e6,
        "noise": True,
        "method": "NNLS",
        "nnls_max_rg": 30.0
    }
    
    if 'batch_df' not in st.session_state:
        st.session_state.batch_df = pd.DataFrame([default_row])

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        uploaded_batch = st.file_uploader("Upload Batch CSV", type=['csv'])
        if uploaded_batch:
            try: 
                st.session_state.batch_df = pd.read_csv(uploaded_batch)
                st.success("Batch Loaded")
            except: st.error("Invalid CSV")
    with c_b2:
        st.download_button("Download Template", 
                           pd.DataFrame([default_row]).to_csv(index=False), 
                           "batch_template.csv", "text/csv")
        
    st.info("Tip: Enter lists like `[0.1, 0.3]` in parameter cells to run combinations.")
    
    edited_df = st.data_editor(st.session_state.batch_df, num_rows="dynamic", use_container_width=True)
    st.session_state.batch_df = edited_df

    if st.button("Execute Batch Queue"):
        expanded_df = expand_batch_parameters(edited_df)
        st.write(f"Queue size: {len(expanded_df)} simulations")
        
        progress_bar = st.progress(0)
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("batch_summary.csv", expanded_df.to_csv(index=False))
            
            for i, row in expanded_df.iterrows():
                params = row.to_dict()
                try:
                    q_sim, i_sim, _, r_vals, pdf_vals = run_simulation_core(params)
                    
                    res = perform_saxs_analysis(q_sim, i_sim, 
                                                params['dist_type'], 
                                                float(params['mean_rg']), 
                                                params['mode'], 
                                                params['method'], 
                                                float(params['nnls_max_rg']))
                    
                    rec_dists = {}
                    if params['method'] == 'Tomchuk':
                         m_pdi = res.get('rg_num_rec_pdi', float(params['mean_rg'])) * np.sqrt(5.0/3.0)
                         if m_pdi > 0:
                             rec_dists['pdi'] = get_distribution(params['dist_type'], r_vals, m_pdi, res.get('p_rec_pdi', 0))
                         
                         m_pdi2 = res.get('rg_num_rec_pdi2', float(params['mean_rg'])) * np.sqrt(5.0/3.0)
                         if m_pdi2 > 0:
                             rec_dists['pdi2'] = get_distribution(params['dist_type'], r_vals, m_pdi2, res.get('p_rec_pdi2', 0))
                    elif 'nnls_r' in res:
                         rec_dists['nnls_r'] = res['nnls_r']
                         rec_dists['nnls_w'] = res['nnls_w']
                    
                    csv_content = create_download_data(r_vals, pdf_vals, rec_dists, res, params, "Input", q_sim, res.get('I_fit', []))
                    zf.writestr(f"run_{i}_analysis.csv", csv_content)
                except Exception as e:
                    zf.writestr(f"run_{i}_error.txt", str(e))
                
                progress_bar.progress((i + 1) / len(expanded_df))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.success("Batch Complete!")
        st.download_button("Download ZIP", zip_buffer.getvalue(), f"saxs_batch_{timestamp}.zip", "application/zip")