import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import erf, gamma, factorial
from scipy.ndimage import gaussian_filter
from scipy.optimize import bisect, nnls
from scipy.integrate import trapezoid
import pandas as pd
import io
import zipfile
from datetime import datetime
import ast
import itertools

# --- 1. Math & Physics Utilities ---

def double_factorial(n):
    if n <= 0: return 1
    return n * double_factorial(n - 2)

def nCr(n, r):
    try:
        return factorial(n) / (factorial(r) * factorial(n - r))
    except:
        return 1

def parse_saxs_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        data = []
        for line in content.splitlines():
            line = line.split('#')[0].split('!')[0].strip()
            if not line: continue
            line = line.replace(',', ' ').replace(';', ' ')
            parts = line.split()
            try:
                nums = [float(p) for p in parts]
                if len(nums) >= 2:
                    data.append([nums[0], nums[1]])
            except ValueError:
                continue

        arr = np.array(data)
        if arr.shape[0] < 5:
            return None, None, "File content too short."
        
        mask = np.isfinite(arr).all(axis=1)
        arr = arr[mask]
        
        if arr.shape[0] < 5:
            return None, None, "File contains mostly invalid (NaN/Inf) data."

        arr = arr[arr[:, 0].argsort()]
        arr = arr[arr[:, 0] > 1e-6]
        
        return arr[:, 0], arr[:, 1], None
        
    except Exception as e:
        return None, None, f"Error parsing file: {str(e)}"

# --- Distributions ---
def get_distribution(dist_type, r, mean_r, p):
    mean_r = max(mean_r, 1e-6)
    p = max(p, 1e-6)
    
    if dist_type == 'Lognormal':
        s = np.sqrt(np.log(1 + p**2))
        scale = mean_r / np.exp(s**2 / 2.0)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            coef = 1.0 / (r * s * np.sqrt(2 * np.pi))
            arg = (np.log(r / scale)**2) / (2 * s**2)
            pdf = coef * np.exp(-arg)
        pdf = np.nan_to_num(pdf)
        return pdf

    elif dist_type == 'Gaussian':
        sigma = p * mean_r
        coef = 1.0 / (sigma * np.sqrt(2 * np.pi))
        arg = ((r - mean_r)**2) / (2 * sigma**2)
        return coef * np.exp(-arg)

    elif dist_type == 'Schulz':
        if p < 1e-4: return np.zeros_like(r)
        z = (1.0 / (p**2)) - 1.0
        if z <= 0: return np.zeros_like(r)
        a = z + 1
        b = a / mean_r
        try:
            norm = (b**a) / gamma(a)
        except:
            norm = 1.0
            
        with np.errstate(over='ignore', invalid='ignore'):
             pdf = norm * (r**z) * np.exp(-b * r)
        return np.nan_to_num(pdf)

    elif dist_type == 'Boltzmann':
        sigma = p * mean_r
        coef = 1.0 / (sigma * np.sqrt(2))
        arg = (np.sqrt(2) * np.abs(r - mean_r)) / sigma
        return coef * np.exp(-arg)

    elif dist_type == 'Triangular':
        sigma = p * mean_r
        w = sigma * np.sqrt(6)
        lower = mean_r - w
        upper = mean_r + w
        h = 1.0 / w
        pdf = np.zeros_like(r)
        mask_up = (r >= lower) & (r <= mean_r)
        mask_down = (r > mean_r) & (r <= upper)
        pdf[mask_up] = h * (r[mask_up] - lower) / w
        pdf[mask_down] = h * (upper - r[mask_down]) / w
        return pdf

    elif dist_type == 'Uniform':
        sigma = p * mean_r
        w = sigma * np.sqrt(3)
        lower = mean_r - w
        upper = mean_r + w
        pdf = np.zeros_like(r)
        mask = (r >= lower) & (r <= upper)
        pdf[mask] = 1.0 / (2 * w)
        return pdf
    
    return np.zeros_like(r)

# --- Form Factors ---

def sphere_form_factor(q, r):
    q_col = q[:, np.newaxis]
    r_row = r[np.newaxis, :]
    qr = q_col * r_row
    
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = 3 * (np.sin(qr) - qr * np.cos(qr)) / (qr**3)
    amp = np.nan_to_num(amp, nan=1.0) 
    
    vol = (4.0/3.0) * np.pi * (r_row**3)
    return (vol**2) * (amp**2)

def debye_form_factor(q, rg):
    q_col = q[:, np.newaxis]
    r_row = rg[np.newaxis, :]
    u = (q_col * r_row)**2
    with np.errstate(divide='ignore', invalid='ignore'):
        val = 2.0 * (np.exp(-u) + u - 1.0) / (u**2)
    val = np.nan_to_num(val, nan=1.0)
    return val

# --- Analysis / Recovery Logic ---

def get_normalized_moment(k, p, dist_type):
    p = max(p, 1e-5)
    
    if dist_type == 'Lognormal':
        return (1 + p**2)**(k*(k-1)/2.0)
    elif dist_type == 'Gaussian':
        total = 0.0
        limit = k // 2
        for j in range(limit + 1): total += nCr(k, 2*j) * double_factorial(2*j - 1) * (p**(2*j))
        return total
    elif dist_type == 'Schulz':
        z = (1.0 / p**2) - 1.0
        if z <= 0: return 1.0
        product = 1.0
        for i in range(1, k + 1): product *= (z + i)
        return product / ((z + 1.0)**k)
    return 1.0 

def calculate_indices_from_p(p, dist_type):
    m2 = get_normalized_moment(2, p, dist_type)
    m3 = get_normalized_moment(3, p, dist_type)
    m4 = get_normalized_moment(4, p, dist_type)
    m6 = get_normalized_moment(6, p, dist_type)
    m8 = get_normalized_moment(8, p, dist_type)
    PDI, PDI2 = 0, 0
    if m6 > 0: PDI = (m2 * m8**2) / (m6**3)
    if m3 > 0: PDI2 = (m2 * m4) / (m3**2)
    return PDI, PDI2

def solve_p_tomchuk(target_val, index_type, dist_type):
    if target_val is None or target_val < 1.0001: return 0.0
    def func(p_guess):
        pdi, pdi2 = calculate_indices_from_p(p_guess, dist_type)
        return (pdi if index_type == 'PDI' else pdi2) - target_val
    try:
        f0 = func(0.001)
        f6 = func(6.0)
        if f0 * f6 > 0: return 0.0 
        return bisect(func, 0.001, 6.0, xtol=1e-4)
    except: return 0.0

def recover_distribution_nnls(q_exp, i_exp, q_min, q_max, kernel_func, max_rg_basis):
    if q_min <= 0: q_min_eff = 1e-3
    else: q_min_eff = q_min
    
    r_min_basis = max(0.5, 0.5 / q_max)
    r_max_basis = max_rg_basis
    n_basis = 150
    r_basis = np.logspace(np.log10(r_min_basis), np.log10(r_max_basis), n_basis)
    
    A = kernel_func(q_exp, r_basis)
    weights, resid = nnls(A, i_exp)
    
    i_fit = A @ weights
    
    total_w = np.sum(weights)
    if total_w > 0:
        weights = weights / total_w
    
    mean_val = np.sum(weights * r_basis)
    var = np.sum(weights * (r_basis - mean_val)**2)
    std = np.sqrt(var)
    p_rec = std / mean_val if mean_val > 0 else 0
    
    weights_smooth = gaussian_filter(weights, sigma=1.0)
    pdf_nnls = weights_smooth.copy()
    area = trapezoid(pdf_nnls, r_basis)
    if area > 0: pdf_nnls /= area
    
    return r_basis, weights, pdf_nnls, mean_val, p_rec, i_fit

def perform_saxs_analysis(q_exp, i_exp, dist_type, initial_rg_guess, mode, method, max_rg_nnls):
    results = {
        'Rg': 0, 'G': 0, 'B': 0, 'Q': 0, 'lc': 0, 
        'PDI': 0, 'PDI2': 0, 
        'p_rec_pdi': 0, 'p_rec_pdi2': 0, 
        'rg_num_rec_pdi': 0, 'rg_num_rec_pdi2': 0,
        'p_rec': 0, 'rg_num_rec': 0,
        'I_fit': np.zeros_like(q_exp) if len(q_exp) > 0 else []
    }
    
    if len(q_exp) < 5: return results

    rg_init = initial_rg_guess
    try:
        valid_pts = (q_exp > 0) & (i_exp > 0)
        q_v = q_exp[valid_pts]
        i_v = i_exp[valid_pts]
        if len(q_v) > 5:
            n_init = min(15, len(q_v))
            x_init = q_v[:n_init]**2
            y_init = np.log(i_v[:n_init])
            slope_init, _ = np.polyfit(x_init, y_init, 1)
            if slope_init < 0:
                rg_est = np.sqrt(-3 * slope_init)
                if rg_est > 0: rg_init = rg_est
    except:
        pass

    rg_fit = rg_init
    g_fit = i_exp[0] if len(i_exp) > 0 else 1.0
    valid_fit = False

    for _ in range(5): 
        limit = 1.0 if mode == 'IDP' else 1.3
        mask = (q_exp * rg_fit) < limit
        mask = mask & (q_exp > 0) & (i_exp > 0)
        
        if np.sum(mask) < 4: break
        
        x_fit = q_exp[mask]**2
        y_fit = np.log(i_exp[mask])
        
        try:
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            if slope >= 0: break 
            rg_new = np.sqrt(np.abs(-3 * slope))
            g_new = np.exp(intercept)
            rg_fit = rg_new
            g_fit = g_new
            valid_fit = True
        except:
            break
            
    results['Rg'] = rg_fit
    results['G'] = g_fit

    i_fit_global = np.zeros_like(i_exp)

    if mode == 'Sphere':
        if method == 'Tomchuk':
            n_fit_b = min(20, len(q_exp)//4)
            B_est = 0
            if n_fit_b > 0:
                b_region_i = i_exp[-n_fit_b:]
                b_region_q = q_exp[-n_fit_b:]
                B_est = np.mean(b_region_i * (b_region_q**4))

            q_max_meas = q_exp[-1] if len(q_exp) > 0 else 0
            integrand_q = (q_exp**2) * i_exp
            Q_obs = trapezoid(integrand_q, q_exp)
            Q_tail = B_est / q_max_meas if q_max_meas > 0 else 0
            Q = Q_obs + Q_tail

            integrand_lc = q_exp * i_exp
            lin_obs = trapezoid(integrand_lc, q_exp)
            lin_tail = B_est / (2 * q_max_meas**2) if q_max_meas > 0 else 0
            lc = (np.pi / Q) * (lin_obs + lin_tail) if Q > 0 else 0
            
            pdi_val = 0
            if valid_fit and g_fit > 0:
                pdi_val = (50.0/81.0) * (B_est * (rg_fit**4)) / g_fit

            pdi2_val = 0
            if Q > 0:
                pdi2_val = (2 * np.pi / 9) * (B_est * lc) / Q
                
            p_rec_pdi = solve_p_tomchuk(pdi_val, 'PDI', dist_type)
            p_rec_pdi2 = solve_p_tomchuk(pdi2_val, 'PDI2', dist_type)
            
            m6 = get_normalized_moment(6, p_rec_pdi, dist_type)
            m8 = get_normalized_moment(8, p_rec_pdi, dist_type)
            ratio = np.sqrt(m8 / m6) if m6 > 0 else 1.0
            rg_num_rec = rg_fit / ratio
            
            m6_2 = get_normalized_moment(6, p_rec_pdi2, dist_type)
            m8_2 = get_normalized_moment(8, p_rec_pdi2, dist_type)
            ratio_2 = np.sqrt(m8_2 / m6_2) if m6_2 > 0 else 1.0
            rg_num_rec_2 = rg_fit / ratio_2

            results['Q'] = Q
            results['lc'] = lc
            results['B'] = B_est
            results['PDI'] = pdi_val
            results['PDI2'] = pdi2_val
            results['p_rec_pdi'] = p_rec_pdi
            results['p_rec_pdi2'] = p_rec_pdi2
            results['rg_num_rec_pdi'] = rg_num_rec
            results['rg_num_rec_pdi2'] = rg_num_rec_2
            results['method'] = 'Tomchuk'
            
            r_sim = np.linspace(max(0.1, rg_num_rec*0.1), rg_num_rec*5, 200)
            pdf_sim = get_distribution(dist_type, r_sim, rg_num_rec*np.sqrt(5/3), p_rec_pdi)
            a_sim = trapezoid(pdf_sim, r_sim)
            if a_sim > 0: pdf_sim /= a_sim
            i_mtx = sphere_form_factor(q_exp, r_sim)
            i_calc_shape = trapezoid(i_mtx * pdf_sim, r_sim, axis=1)
            num = np.sum(i_exp * i_calc_shape)
            den = np.sum(i_calc_shape**2)
            scale = num/den if den > 0 else 1.0
            i_fit_global = i_calc_shape * scale
            
        elif method == 'NNLS':
            r_nnls, w_nnls, pdf_nnls, mean_r_nnls, p_nnls, i_fit_global = recover_distribution_nnls(q_exp, i_exp, q_exp[0], q_exp[-1], sphere_form_factor, max_rg_basis=max_rg_nnls)
            results['rg_num_rec'] = mean_r_nnls * np.sqrt(3.0/5.0)
            results['p_rec'] = p_nnls
            results['nnls_r'] = r_nnls
            results['nnls_w'] = w_nnls
            results['method'] = 'NNLS'
            results['Q'] = np.nan
            results['B'] = np.nan

    elif mode == 'IDP':
        r_nnls, w_nnls, pdf_nnls, mean_rg_nnls, p_nnls, i_fit_global = recover_distribution_nnls(q_exp, i_exp, q_exp[0], q_exp[-1], debye_form_factor, max_rg_basis=max_rg_nnls)
        results['p_rec'] = p_nnls
        results['rg_num_rec'] = mean_rg_nnls
        results['nnls_r'] = r_nnls
        results['nnls_w'] = w_nnls
        results['method'] = 'NNLS'
        results['Q'] = np.nan
        results['lc'] = np.nan
        results['B'] = np.nan
    
    sigma = np.sqrt(np.abs(i_exp))
    sigma[sigma == 0] = 1.0 
    chi2 = np.sum(((i_exp - i_fit_global) / sigma)**2)
    dof = max(1, len(i_exp) - 3)
    chi2_red = chi2 / dof
    
    results['I_fit'] = i_fit_global
    results['chi2'] = chi2_red

    return results

def run_simulation_core(params):
    mean_rg = float(params['mean_rg'])
    p_val = float(params['p_val'])
    dist_type = params['dist_type']
    mode_key = params['mode']
    pixels = int(params['pixels'])
    q_max = float(params['q_max'])
    q_min = float(params['q_min'])
    n_bins = int(params['n_bins'])
    smearing = float(params['smearing'])
    flux = float(params['flux'])
    noise = bool(params['noise'])
    binning_mode = params['binning_mode']

    mean_r = mean_rg * np.sqrt(5.0/3.0) 
    sigma = p_val * mean_r

    r_min = max(0.1, mean_r - 5 * sigma)
    r_max = mean_r + 15 * sigma
    r_steps = 400
    r_vals = np.linspace(r_min, r_max, r_steps)
    
    if mode_key == 'IDP':
        r_min_idp = max(0.1, mean_rg * (1 - 5*p_val))
        r_max_idp = mean_rg * (1 + 15*p_val)
        r_vals = np.linspace(r_min_idp, r_max_idp, r_steps)
        pdf_vals = get_distribution(dist_type, r_vals, mean_rg, p_val)
    else:
        pdf_vals = get_distribution(dist_type, r_vals, mean_r, p_val)

    area = trapezoid(pdf_vals, r_vals)
    if area > 0: pdf_vals /= area

    q_steps = 200
    q_1d = np.logspace(np.log10(1e-3), np.log10(q_max * 1.5), q_steps)

    if mode_key == 'Sphere':
        i_matrix = sphere_form_factor(q_1d, r_vals) 
        i_1d_ideal = trapezoid(i_matrix * pdf_vals, r_vals, axis=1) 
    else:
        i_matrix = debye_form_factor(q_1d, r_vals)
        i_1d_ideal = trapezoid(i_matrix * pdf_vals, r_vals, axis=1)

    x = np.linspace(-q_max, q_max, pixels)
    y = np.linspace(-q_max, q_max, pixels)
    xv, yv = np.meshgrid(x, y)
    qv_r = np.sqrt(xv**2 + yv**2)

    i_2d_ideal = np.interp(qv_r.ravel(), q_1d, i_1d_ideal, left=i_1d_ideal[0], right=0)
    i_2d_ideal = i_2d_ideal.reshape(pixels, pixels)

    if smearing > 0:
        i_2d_smeared = gaussian_filter(i_2d_ideal, sigma=smearing)
    else:
        i_2d_smeared = i_2d_ideal

    total_int = np.sum(i_2d_smeared)
    scale_factor = flux / total_int if total_int > 0 else 1.0
    i_2d_scaled = i_2d_smeared * scale_factor

    if noise:
        i_2d_final = np.random.poisson(i_2d_scaled).astype(float)
    else:
        i_2d_final = i_2d_scaled

    if binning_mode == "Linear":
        sim_bin_width = (q_max - q_min) / n_bins
        if sim_bin_width <= 0: sim_bin_width = q_max / n_bins
        r_indices = ((qv_r - q_min) / sim_bin_width).astype(int).ravel()
        valid_mask = (r_indices >= 0) & (r_indices < n_bins) & (qv_r.ravel() <= q_max)
        tbin = np.bincount(r_indices[valid_mask], weights=i_2d_final.ravel()[valid_mask], minlength=n_bins)
        nr = np.bincount(r_indices[valid_mask], minlength=n_bins)
        radial_prof = np.zeros(n_bins)
        nonzero = nr > 0
        radial_prof[nonzero] = tbin[nonzero] / nr[nonzero]
        q_sim = q_min + (np.arange(n_bins) + 0.5) * sim_bin_width
    else:
        q_min_log = max(q_min, 1e-4)
        edges = np.logspace(np.log10(q_min_log), np.log10(q_max), n_bins + 1)
        r_vals_flat = qv_r.ravel()
        i_vals_flat = i_2d_final.ravel()
        inds = np.digitize(r_vals_flat, edges)
        valid_mask = (inds >= 1) & (inds <= n_bins)
        valid_inds = inds[valid_mask] - 1 
        tbin = np.bincount(valid_inds, weights=i_vals_flat[valid_mask], minlength=n_bins)
        nr = np.bincount(valid_inds, minlength=n_bins)
        radial_prof = np.zeros(n_bins)
        nonzero = nr > 0
        radial_prof[nonzero] = tbin[nonzero] / nr[nonzero]
        q_sim = np.sqrt(edges[:-1] * edges[1:])

    valid_sim = radial_prof > 0
    return q_sim[valid_sim], radial_prof[valid_sim], i_2d_final, r_vals, pdf_vals

def create_download_data(r, input_dist, recovered_dists, analysis_res, params, input_label, q_axis, i_fit):
    header_lines = [
        f"# SAXS Analysis Results ({params['mode']})",
        "# ==========================================",
        "# Input/Reference Parameters:",
        f"#   Mean Rg: {float(params['mean_rg']):.4f} nm",
        f"#   Polydispersity (p): {float(params['p_val']):.4f}",
        f"#   Distribution Type: {params['dist_type']}",
        "#",
        f"# Analysis Output ({params['method']}):",
        f"#   Rg (Guinier Fit): {analysis_res.get('Rg', 0):.6f} nm",
        f"#   G (Forward Scattering): {analysis_res.get('G', 0):.6e}",
        f"#   Chi-Squared (Reduced): {analysis_res.get('chi2', 0):.4f}"
    ]

    if params['method'] == 'Tomchuk':
        header_lines.extend([
            f"#   B (Porod Constant): {analysis_res.get('B', 0):.6e}",
            f"#   Porod Invariant (Q): {analysis_res.get('Q', 0):.6e}",
            f"#   PDI (Calculated): {analysis_res.get('PDI', 0):.6f}",
            f"#   PDI2 (Calculated): {analysis_res.get('PDI2', 0):.6f}",
            f"#   Recovered p (from PDI): {analysis_res.get('p_rec_pdi', 0):.6f}",
            f"#   Recovered Mean Rg (from PDI): {analysis_res.get('rg_num_rec_pdi', 0):.6f} nm",
            f"#   Recovered p (from PDI2): {analysis_res.get('p_rec_pdi2', 0):.6f}",
            f"#   Recovered Mean Rg (from PDI2): {analysis_res.get('rg_num_rec_pdi2', 0):.6f} nm",
        ])
    else: 
        header_lines.extend([
            f"#   Recovered Rg (Mean): {analysis_res.get('rg_num_rec', 0):.6f} nm",
            f"#   Recovered p (Width): {analysis_res.get('p_rec', 0):.6f}"
        ])
    
    header_lines.append("# ==========================================")
    header_lines.append(f"# Data Columns: Radius [nm], {input_label}_PDF, ...Recovered_PDFs, q [nm-1], I_Fit")
    header = "\n".join(header_lines) + "\n"

    df_dist = pd.DataFrame()
    df_dist['Radius'] = r
    df_dist[f'{input_label}_PDF'] = input_dist
    
    if params['method'] == 'Tomchuk':
        if 'pdi' in recovered_dists:
            df_dist['PDI_Recovered_PDF'] = recovered_dists['pdi']
        if 'pdi2' in recovered_dists:
            df_dist['PDI2_Recovered_PDF'] = recovered_dists['pdi2']
    else: 
        if 'nnls_r' in recovered_dists:
             df_dist['NNLS_Recovered_PDF'] = np.interp(r, recovered_dists['nnls_r'], recovered_dists['nnls_w'], left=0, right=0)
    
    df_fit = pd.DataFrame({'q': q_axis, 'I_fit': i_fit})
    df_final = pd.concat([df_dist, df_fit], axis=1)

    return header + df_final.to_csv(index=False)

def expand_batch_parameters(df):
    all_jobs = []
    base_cols = df.columns
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

# --- 2. Streamlit App Layout ---

st.set_page_config(page_title="SAXS Simulator", layout="wide", page_icon="⚛️")

# --- Initialize Session State ---
if 'init' not in st.session_state:
    st.session_state['init'] = True
    st.session_state['mean_rg'] = 4.0
    st.session_state['p_val'] = 0.3
    st.session_state['n_bins'] = 512
    st.session_state['q_min'] = 0.0
    st.session_state['q_max'] = 2.5
    st.session_state['nnls_max_rg'] = 30.0 
    st.session_state['last_filename'] = None
    st.session_state['batch_mode'] = False

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("SAXS Simulator")
with col_h2:
    st.markdown("*Spheres (Tomchuk/NNLS) vs IDP (NNLS)*")

# --- GLOBAL SETTINGS (Before Layout Split) ---
st.sidebar.header("Configuration")
batch_mode = st.sidebar.checkbox("Batch Mode Simulation", value=False)
sim_mode = st.sidebar.radio("Simulation Mode", ["Polydisperse Spheres", "Fixed-Length Polymers (IDP)"])
mode_key = 'Sphere' if 'Sphere' in sim_mode else 'IDP'

analysis_method = 'NNLS'
if mode_key == 'Sphere':
    analysis_method = st.sidebar.selectbox("Analysis Method", ["Tomchuk (Invariants)", "NNLS (Distribution Fit)"])
    analysis_method = "Tomchuk" if "Tomchuk" in analysis_method else "NNLS"

# Experimental Data Load (Common)
st.sidebar.header("Experimental Data")
uploaded_file = st.sidebar.file_uploader("Load 1D Profile", type=['dat', 'out', 'txt', 'csv'])

# Shared Callbacks
def update_q_max():
    if st.session_state.mean_rg > 0:
        st.session_state.q_max = round(10.0 / st.session_state.mean_rg, 2)
def update_q_max_and_basis():
    if st.session_state.mean_rg > 0:
        st.session_state.q_max = round(10.0 / st.session_state.mean_rg, 2)
        p = st.session_state.get('p_val', 0.3)
        st.session_state.nnls_max_rg = float(round(st.session_state.mean_rg * (1 + 6 * p), 1))
def auto_set_nnls_max_rg(rg):
    if rg > 0: st.session_state['nnls_max_rg'] = float(round(5 * rg, 1))

# Common Params (Rendered always to preserve state)
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

# Global logic for file loading (needs to be outside if/else to update params)
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
                    auto_set_nnls_max_rg(res['Rg'])
                st.rerun()

# --- BATCH MODE ---
if batch_mode:
    st.subheader("Batch Simulation Runner")
    # Convert numeric defaults to string to allow user to type "[1, 2]"
    default_row = {
        "mode": mode_key, "dist_type": dist_type, 
        "mean_rg": str(mean_rg), "p_val": str(p_val),
        "pixels": pixels, "q_min": q_min, "q_max": q_max, "n_bins": n_bins,
        "binning_mode": binning_mode, "smearing": smearing, 
        "flux": flux_pre * (10**flux_exp), "noise": (not optimal_flux and add_noise),
        "method": analysis_method, "nnls_max_rg": nnls_max_rg
    }
    
    if 'batch_df' not in st.session_state: st.session_state.batch_df = pd.DataFrame([default_row])

    c_b1, c_b2, c_b3 = st.columns(3)
    with c_b1:
        uploaded_batch = st.file_uploader("Upload Batch CSV", type=['csv'])
        if uploaded_batch:
            try: st.session_state.batch_df = pd.read_csv(uploaded_batch); st.success("Batch Loaded")
            except: st.error("Invalid CSV")
    with c_b2:
        st.download_button("Download Template", pd.DataFrame([default_row]).to_csv(index=False), "batch_template.csv", "text/csv")
        
    edited_df = st.data_editor(st.session_state.batch_df, num_rows="dynamic", use_container_width=True)
    st.session_state.batch_df = edited_df
    
    st.info("Tip: You can enter python lists like `[0.1, 0.3, 0.5]` in Mean Rg, p, or Smearing columns to run all combinations.")

    if st.button("Run Batch Simulations"):
        expanded_df = expand_batch_parameters(edited_df)
        st.write(f"Queue: {len(expanded_df)} Simulations")
        
        progress_bar = st.progress(0)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("batch_summary_expanded.csv", expanded_df.to_csv(index=False))
            
            for i, row in expanded_df.iterrows():
                params = row.to_dict()
                # Ensure types
                try:
                    q_sim, i_sim, _, r_vals, pdf_vals = run_simulation_core(params)
                    res = perform_saxs_analysis(q_sim, i_sim, params['dist_type'], float(params['mean_rg']), params['mode'], params['method'], float(params['nnls_max_rg']))
                    
                    rec_dists = {}
                    if params['method'] == 'Tomchuk':
                         m_pdi = res.get('rg_num_rec_pdi', float(params['mean_rg'])) * np.sqrt(5.0/3.0)
                         rec_dists['pdi'] = get_distribution(params['dist_type'], r_vals, m_pdi, res.get('p_rec_pdi', 0))
                         m_pdi2 = res.get('rg_num_rec_pdi2', float(params['mean_rg'])) * np.sqrt(5.0/3.0)
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
        st.download_button("Download Batch Results (ZIP)", zip_buffer.getvalue(), f"saxs_batch_{timestamp}.zip", "application/zip")

# --- SINGLE MODE UI ---
else:
    # Use current sidebar params
    params = {
        'mean_rg': mean_rg, 'p_val': p_val, 'dist_type': dist_type, 'mode': mode_key,
        'pixels': pixels, 'q_min': q_min, 'q_max': q_max, 'n_bins': n_bins,
        'smearing': smearing, 'flux': flux_pre * (10**flux_exp), 'noise': (not optimal_flux and add_noise),
        'binning_mode': binning_mode
    }
    
    # 1. Simulation
    q_sim, i_sim, i_2d_final, r_vals, pdf_vals = run_simulation_core(params)
    
    # 2. Analysis Active Data
    if use_experimental and q_meas is not None:
        mask = (q_meas >= q_min) & (q_meas <= q_max)
        q_target = q_meas[mask]
        i_target = i_meas[mask]
    else:
        q_target = q_sim
        i_target = i_sim

    analysis_res = perform_saxs_analysis(q_target, i_target, dist_type, mean_rg, mode_key, analysis_method, nnls_max_rg)

    # 3. Visualization
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

    # Analysis Results Display
    st.markdown("---")
    st.subheader("Analysis Results")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Parameters**")
        input_label = "Ref (Sidebar)" if use_experimental else "Input"
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

    # Download
    params_dict = {'mean_rg': mean_rg, 'p_val': p_val, 'dist_type': dist_type, 'mode': mode_key, 'method': analysis_method}
    download_data = create_download_data(r_vals, pdf_vals, rec_dists_dl, analysis_res, params_dict, input_label, q_target, analysis_res.get('I_fit', np.zeros_like(q_target)))
    st.download_button(label="Download Analysis (CSV)", data=download_data, file_name="saxs_analysis.csv", mime="text/csv")