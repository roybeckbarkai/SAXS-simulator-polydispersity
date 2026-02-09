# File: streamlit_app.py   
# Description: Main entry point and navigation controller. new info just for fun

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import erf, gamma, factorial
from scipy.ndimage import gaussian_filter
from scipy.optimize import bisect
from scipy.integrate import trapezoid
import pandas as pd
import io

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
            return None, None, "File content too short or format not recognized."
        
        mask = np.isfinite(arr).all(axis=1)
        arr = arr[mask]
        
        if arr.shape[0] < 5:
            return None, None, "File contains mostly invalid (NaN/Inf) data."

        arr = arr[arr[:, 0].argsort()]
        arr = arr[arr[:, 0] > 0]
        
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
        with np.errstate(divide='ignore', invalid='ignore'):
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
        if p == 0: return np.zeros_like(r)
        z = (1.0 / (p**2)) - 1.0
        if z <= 0: return np.zeros_like(r)
        a = z + 1
        b = a / mean_r
        norm = (b**a) / gamma(a)
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

# --- Form Factor ---
def sphere_form_factor(q, r):
    q_col = q[:, np.newaxis]
    r_row = r[np.newaxis, :]
    qr = q_col * r_row
    
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = 3 * (np.sin(qr) - qr * np.cos(qr)) / (qr**3)
    amp = np.nan_to_num(amp, nan=1.0) 
    
    vol = (4.0/3.0) * np.pi * (r_row**3)
    return (vol**2) * (amp**2)

# --- Analysis / Recovery Logic ---

def get_normalized_moment(k, p, dist_type):
    if p <= 0: return 1
    
    if dist_type == 'Lognormal':
        return (1 + p**2)**(k*(k-1)/2.0)
    elif dist_type == 'Gaussian':
        total = 0.0
        limit = k // 2
        for j in range(limit + 1):
            total += nCr(k, 2*j) * double_factorial(2*j - 1) * (p**(2*j))
        return total
    elif dist_type == 'Schulz':
        z = (1.0 / p**2) - 1.0
        if z <= 0: return 1
        product = 1.0
        for i in range(1, k + 1):
            product *= (z + i)
        return product / ((z + 1.0)**k)
    elif dist_type == 'Boltzmann':
        total = 0.0
        limit = k // 2
        for i in range(limit + 1):
            term = (1.0 / (2**i)) * (factorial(k) / factorial(k - 2*i)) * (p**(2*i))
            total += term
        return total
    elif dist_type == 'Triangular':
        sqrt6 = np.sqrt(6)
        num = (1 - p*sqrt6)**(k+2) + (1 + p*sqrt6)**(k+2) - 2
        den = 6 * p**2 * (k+1) * (k+2)
        return num / den
    elif dist_type == 'Uniform':
        sqrt3 = np.sqrt(3)
        num = (1 + p*sqrt3)**(k+1) - (1 - p*sqrt3)**(k+1)
        den = 2 * p * sqrt3 * (k+1)
        return num / den
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

def solve_p(target_val, index_type, dist_type):
    if target_val is None or target_val < 1.0: return 0.0
    
    def func(p_guess):
        pdi, pdi2 = calculate_indices_from_p(p_guess, dist_type)
        val = pdi if index_type == 'PDI' else pdi2
        return val - target_val

    try:
        if func(0) * func(6.0) > 0: return 0.0 
        return bisect(func, 0, 6.0, xtol=1e-4)
    except:
        return 0.0

def get_calculated_mean_rg_num(rg_scat, p, dist_type):
    if not rg_scat or rg_scat <= 0: return 0
    if p <= 0: return rg_scat
    
    m6 = get_normalized_moment(6, p, dist_type)
    m8 = get_normalized_moment(8, p, dist_type)
    if m6 <= 0: return 0
    
    ratio = np.sqrt(m8 / m6)
    return rg_scat / ratio

def perform_saxs_analysis(q_exp, i_exp, dist_type, initial_rg_guess):
    """
    Core analysis logic.
    """
    results = {}
    
    # 1. Invariants with Porod Tail Correction
    n_fit_b = min(20, len(q_exp)//4)
    B_est = 0
    if n_fit_b > 0 and len(q_exp) > n_fit_b:
        b_region_i = i_exp[-n_fit_b:]
        b_region_q = q_exp[-n_fit_b:]
        b_vals = b_region_i * (b_region_q**4)
        B_est = np.mean(b_vals)

    q_max_meas = q_exp[-1] if len(q_exp) > 0 else 0

    integrand_q = (q_exp**2) * i_exp
    Q_obs = trapezoid(integrand_q, q_exp)
    Q_tail = B_est / q_max_meas if q_max_meas > 0 else 0
    Q = Q_obs + Q_tail

    integrand_lc = q_exp * i_exp
    lin_obs = trapezoid(integrand_lc, q_exp)
    lin_tail = B_est / (2 * q_max_meas**2) if q_max_meas > 0 else 0
    lc = (np.pi / Q) * (lin_obs + lin_tail) if Q > 0 else 0

    # 2. Iterative Guinier Fit
    # Use provided guess to start (critical for robustness)
    rg_fit = initial_rg_guess
    g_fit = i_exp[0] if len(i_exp) > 0 else 1.0
    valid_fit = False

    for _ in range(5): 
        mask = (q_exp * rg_fit) < 1.0 # Strict Guinier limit
        mask = mask & (q_exp > 0) & (i_exp > 0)
        
        if np.sum(mask) < 3: break
        
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

    # 3. PDI Calc
    pdi_val = 0
    if valid_fit and g_fit > 0:
        pdi_val = (50.0/81.0) * (B_est * (rg_fit**4)) / g_fit

    pdi2_val = 0
    if Q > 0:
        pdi2_val = (2 * np.pi / 9) * (B_est * lc) / Q

    # 4. Recovery
    p_rec_pdi = solve_p(pdi_val, 'PDI', dist_type)
    p_rec_pdi2 = solve_p(pdi2_val, 'PDI2', dist_type)

    rg_num_rec_pdi = get_calculated_mean_rg_num(rg_fit, p_rec_pdi, dist_type)
    rg_num_rec_pdi2 = get_calculated_mean_rg_num(rg_fit, p_rec_pdi2, dist_type)

    results['Q'] = Q
    results['lc'] = lc
    results['Rg'] = rg_fit
    results['G'] = g_fit
    results['B'] = B_est
    results['PDI'] = pdi_val
    results['PDI2'] = pdi2_val
    results['p_from_PDI'] = p_rec_pdi
    results['p_from_PDI2'] = p_rec_pdi2
    results['rg_num_rec_PDI'] = rg_num_rec_pdi
    results['rg_num_rec_PDI2'] = rg_num_rec_pdi2
    
    return results

def create_download_data(r, input_dist, pdi_dist, pdi2_dist, analysis_res, params, input_label):
    header = f"""# SAXS Analysis Results
# ==========================================
# Input/Reference Parameters:
#   Mean Rg (Num Avg): {params['mean_rg']:.4f} nm
#   Polydispersity (p): {params['p_val']:.4f}
#   Distribution Type: {params['dist_type']}
#
# Analysis Output:
#   Porod Invariant (Q): {analysis_res['Q']:.6e}
#   Correlation Length (lc): {analysis_res['lc']:.6f} nm
#   Rg (Fit, Scattering Avg): {analysis_res['Rg']:.6f} nm
#   Rg (Num Avg, Recovered): {analysis_res['rg_num_rec_PDI']:.6f} nm
#   G (Forward Scattering): {analysis_res['G']:.6e}
#   B (Porod Constant): {analysis_res['B']:.6e}
#
# Recovery Findings:
#   PDI (Calculated): {analysis_res['PDI']:.6f}
#   PDI2 (Calculated): {analysis_res['PDI2']:.6f}
#   Recovered p (from PDI): {analysis_res['p_from_PDI']:.6f}
#   Recovered p (from PDI2): {analysis_res['p_from_PDI2']:.6f}
# ==========================================
# Data Columns:
# Radius [nm], {input_label}_PDF, PDI_Recovered_PDF, PDI2_Recovered_PDF
"""
    df = pd.DataFrame({
        'Radius': r,
        f'{input_label}_PDF': input_dist,
        'PDI_Recovered_PDF': pdi_dist,
        'PDI2_Recovered_PDF': pdi2_dist
    })
    
    return header + df.to_csv(index=False)

# --- 2. Streamlit App Layout ---

st.set_page_config(page_title="SAXS Simulator", layout="wide", page_icon="⚛️")

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("SAXS Simulator: Polydisperse Spheres")
with col_h2:
    st.markdown("*Tomchuk et al. PDI Analysis*")

# --- Initialize Session State for Params ---
if 'init' not in st.session_state:
    st.session_state['init'] = True
    # Default parameters needed for batch mode initialization if user goes straight there
    st.session_state['mean_rg'] = 4.0
    st.session_state['p_val'] = 0.3
    st.session_state['q_max'] = 2.5
    st.session_state['n_bins'] = 512
    st.session_state['q_min'] = 0.0
    st.session_state['q_max'] = 2.5
    st.session_state['last_filename'] = None

# --- Sidebar Controls ---

st.sidebar.header("Experimental Data")
uploaded_file = st.sidebar.file_uploader("Load 1D Profile (q, I)", type=['dat', 'out', 'txt', 'csv'], help="Supports ATSAS .dat files or text files with q and I columns.")
use_experimental = False
q_meas, i_meas = None, None

# Handle File Logic
if uploaded_file is not None:
    q_load, i_load, err = parse_saxs_file(uploaded_file)
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success(f"Loaded {len(q_load)} points.")
        use_experimental = st.sidebar.checkbox("Use Loaded Data", value=True)
        
        if use_experimental:
            q_meas, i_meas = q_load, i_load
            
            if st.session_state['last_filename'] != uploaded_file.name:
                st.session_state['last_filename'] = uploaded_file.name
                
                file_min_q = float(np.min(q_meas))
                file_max_q = float(np.max(q_meas))
                file_n_bins = len(q_meas)
                
                st.session_state['q_min'] = file_min_q
                st.session_state['q_max'] = file_max_q
                st.session_state['n_bins'] = file_n_bins
                
                current_dist_type = st.session_state.get('dist_type', 'Gaussian')
                # Use current Mean Rg as initial guess for fit
                current_mean_rg = st.session_state.get('mean_rg', 4.0)
                res = perform_saxs_analysis(q_meas, i_meas, current_dist_type, current_mean_rg)
                
                if res['rg_num_rec_PDI'] > 0:
                    st.session_state['mean_rg'] = float(res['rg_num_rec_PDI'])
                if res['p_from_PDI'] > 0:
                    st.session_state['p_val'] = float(res['p_from_PDI'])
                
                st.rerun()

# Callback to auto-update q_max when mean_rg changes
def update_q_max():
    if st.session_state.mean_rg > 0:
        st.session_state.q_max = round(10.0 / st.session_state.mean_rg, 2)

st.sidebar.header("Sample Parameters")
mean_rg = st.sidebar.number_input("Mean Rg (nm)", min_value=0.5, max_value=50.0, step=0.5, key='mean_rg', help="Number-Average Radius of Gyration", on_change=update_q_max)
p_val = st.sidebar.number_input("Polydispersity (p)", min_value=0.01, max_value=6.0, step=0.01, key='p_val', help="sigma / mean_R")
dist_type = st.sidebar.selectbox("Distribution Type", ['Gaussian', 'Lognormal', 'Schulz', 'Boltzmann', 'Triangular', 'Uniform'], key='dist_type', help="Assumed distribution type for PDI recovery analysis.")

st.sidebar.header("Instrument / Binning")
pixels = st.sidebar.number_input("Detector Size (NxN)", value=1024, step=64, help="The detector will be N x N pixels.")
col_q1, col_q2 = st.sidebar.columns(2)
with col_q1:
    q_min = st.sidebar.number_input("Min q (nm⁻¹)", min_value=0.0, step=0.01, key='q_min')
with col_q2:
    q_max = st.sidebar.number_input("Max q (nm⁻¹)", min_value=0.01, step=0.1, key='q_max')
    
n_bins = st.sidebar.number_input("1D Bins", min_value=10, step=10, key='n_bins', help="If different from input data size, data will be rebinned.")
binning_mode = st.sidebar.selectbox("Binning Mode", ["Logarithmic", "Linear"], help="Logarithmic binning improves high-q statistics (standard for SAXS). Linear is simple averaging.")
smearing = st.sidebar.number_input("Smearing (px)", value=2.0, step=0.5)

st.sidebar.header("Flux & Noise")
col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    flux_pre = st.number_input("Flux Coeff", 0.1, 9.9, 1.0, 0.1)
with col_f2:
    flux_exp = st.number_input("Flux Exp (10^x)", 1, 15, 6, 1)

optimal_flux = st.sidebar.checkbox("Optimal Flux (No Noise)", value=False)
add_noise = st.sidebar.checkbox("Simulate Poisson Noise", value=True, disabled=optimal_flux)

# --- Calculation Engine (Simulation) ---

mean_r = mean_rg * np.sqrt(5.0/3.0)
flux = flux_pre * (10**flux_exp)
sigma = p_val * mean_r

r_min = max(0.1, mean_r - 5 * sigma)
r_max = mean_r + 15 * sigma
r_steps = 400
r_vals = np.linspace(r_min, r_max, r_steps)
pdf_vals = get_distribution(dist_type, r_vals, mean_r, p_val)
area = trapezoid(pdf_vals, r_vals)
if area > 0: pdf_vals /= area

q_steps = 200
q_1d = np.logspace(np.log10(1e-3), np.log10(q_max * 1.5), q_steps)
i_matrix = sphere_form_factor(q_1d, r_vals) 
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

if not optimal_flux and add_noise:
    i_2d_final = np.random.poisson(i_2d_scaled).astype(float)
else:
    i_2d_final = i_2d_scaled

# Radial Averaging
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
    
else: # Logarithmic
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
q_sim = q_sim[valid_sim]
i_sim = radial_prof[valid_sim]


# --- Determine Active Data for Analysis ---
if use_experimental and q_meas is not None:
    mask = (q_meas >= q_min) & (q_meas <= q_max)
    q_target = q_meas[mask]
    i_target = i_meas[mask]
    
    if len(q_target) != n_bins:
        if binning_mode == "Linear":
            count, _ = np.histogram(q_target, bins=n_bins, range=(q_min, q_max))
            sum_I, edges = np.histogram(q_target, bins=n_bins, weights=i_target, range=(q_min, q_max))
            with np.errstate(divide='ignore', invalid='ignore'):
                new_I = sum_I / count
            new_q = 0.5 * (edges[1:] + edges[:-1])
        else: # Log
            q_min_log = max(q_min, 1e-4)
            edges = np.logspace(np.log10(q_min_log), np.log10(q_max), n_bins + 1)
            count, _ = np.histogram(q_target, bins=edges)
            sum_I, _ = np.histogram(q_target, bins=edges, weights=i_target)
            with np.errstate(divide='ignore', invalid='ignore'):
                new_I = sum_I / count
            new_q = np.sqrt(edges[:-1] * edges[1:])
            
        valid_bins = count > 0
        q_target = new_q[valid_bins]
        i_target = new_I[valid_bins]
        
else:
    q_target = q_sim
    i_target = i_sim

# --- Analysis Logic ---
analysis_res = perform_saxs_analysis(q_target, i_target, dist_type, mean_rg)

# Theoretical Rg from Simulation (Reference)
m6_arr = (r_vals**6) * pdf_vals
m8_arr = (r_vals**8) * pdf_vals
m6_int = trapezoid(m6_arr, r_vals)
m8_int = trapezoid(m8_arr, r_vals)
rg_theory_scat = 0
if m6_int > 0:
    rg_theory_scat = np.sqrt(0.6 * (m8_int / m6_int))

# Recovery Distributions for Plotting
mean_r_pdi = analysis_res['rg_num_rec_PDI'] * np.sqrt(5.0/3.0)
pdf_pdi = get_distribution(dist_type, r_vals, mean_r_pdi, analysis_res['p_from_PDI'])
a_pdi = trapezoid(pdf_pdi, r_vals)
if a_pdi > 0: pdf_pdi /= a_pdi

mean_r_pdi2 = analysis_res['rg_num_rec_PDI2'] * np.sqrt(5.0/3.0)
pdf_pdi2 = get_distribution(dist_type, r_vals, mean_r_pdi2, analysis_res['p_from_PDI2'])
a_pdi2 = trapezoid(pdf_pdi2, r_vals)
if a_pdi2 > 0: pdf_pdi2 /= a_pdi2


# --- 3. Visualization ---

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
        fig_2d.update_layout(
            xaxis_title='qx (nm⁻¹)',
            yaxis_title='qy (nm⁻¹)',
            width=500, height=500, # Square
            margin=dict(l=40, r=40, t=20, b=40),
            yaxis=dict(scaleanchor="x", scaleratio=1) # Lock Aspect Ratio 1:1
        )
        st.plotly_chart(fig_2d, use_container_width=True)

with col_viz2:
    st.subheader("1D Profile Analysis")
    plot_type = st.selectbox("Plot Type", ["Log-Log", "Lin-Lin", "Guinier", "Porod", "Kratky"])
    
    fig_1d = go.Figure()
    label_str = 'Experimental' if use_experimental else 'Simulated'
    color_str = 'green' if use_experimental else 'blue'

    plot_x = q_target
    plot_y = i_target
    x_type = 'linear'
    y_type = 'linear'
    x_label = 'q (nm⁻¹)'
    y_label = 'I(q)'
    
    if plot_type == "Log-Log":
        x_type = 'log'
        y_type = 'log'
    elif plot_type == "Guinier":
        plot_x = q_target**2
        plot_y = np.log(np.maximum(i_target, 1e-9))
        x_label = 'q² (nm⁻²)'
        y_label = 'ln(I)'
    elif plot_type == "Porod":
        plot_y = i_target * (q_target**4)
        y_label = 'I · q⁴'
    elif plot_type == "Kratky":
        plot_y = i_target * (q_target**2)
        y_label = 'I · q²'

    fig_1d.add_trace(go.Scatter(x=plot_x, y=plot_y, mode='markers', name=label_str, marker=dict(color=color_str, size=4)))

    if plot_type == "Guinier":
        rg_f = analysis_res['Rg']
        g_f = analysis_res['G']
        if rg_f > 0:
            x_line = np.linspace(0, (1.2/rg_f)**2, 50)
            y_line = np.log(g_f) - (rg_f**2/3.0)*x_line
            fig_1d.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Fit', line=dict(color='red', dash='dash')))
            fig_1d.update_xaxes(range=[0, (2.0/rg_f)**2])
            fig_1d.update_yaxes(range=[np.log(g_f)-3, np.log(g_f)+0.5])
    elif plot_type == "Porod":
        if analysis_res['B'] > 0:
            fig_1d.add_hline(y=analysis_res['B'], line_dash="dash", line_color="red", annotation_text="B (Fit)")

    fig_1d.update_layout(
        xaxis_title=x_label, yaxis_title=y_label,
        xaxis_type=x_type, yaxis_type=y_type,
        width=500, height=450,
        margin=dict(l=40, r=40, t=20, b=40)
    )
    st.plotly_chart(fig_1d, use_container_width=True)

# Analysis Panel
st.markdown("---")
st.subheader("Analysis Results")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Polydispersity (p)**")
    input_label = "Ref (Sidebar)" if use_experimental else "Input"
    st.metric(input_label, f"{p_val:.3f}")
    st.metric("Recovered (PDI)", f"{analysis_res['p_from_PDI']:.3f}")
    st.metric("Recovered (PDI₂)", f"{analysis_res['p_from_PDI2']:.3f}")
    
    st.markdown("**Mean Rg (Num Avg)**")
    st.metric(input_label, f"{mean_rg:.2f} nm")
    delta_val = analysis_res['rg_num_rec_PDI'] - mean_rg
    st.metric("Recovered", f"{analysis_res['rg_num_rec_PDI']:.2f} nm", delta=f"{delta_val:.2f}" if not use_experimental else None)

with c2:
    st.markdown("**Parameters**")
    res_df = pd.DataFrame({
        "Parameter": ["Q", "lc", "Rg (Fit, Scat Avg)", "Rg (Num Avg, Rec)", "G", "B"],
        "Value": [
            f"{analysis_res['Q']:.2e}", 
            f"{analysis_res['lc']:.2f} nm", 
            f"{analysis_res['Rg']:.2f} nm", 
            f"{analysis_res['rg_num_rec_PDI']:.2f} nm", 
            f"{analysis_res['G']:.2e}", 
            f"{analysis_res['B']:.2e}"
        ]
    })
    st.dataframe(res_df, hide_index=True)

with c3:
    st.markdown("**Distribution**")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(x=r_vals, y=pdf_vals, mode='lines', name=input_label, line=dict(color='gray', dash='dash')))
    fig_dist.add_trace(go.Scatter(x=r_vals, y=pdf_pdi, mode='lines', name='PDI Rec.', line=dict(color='blue')))
    fig_dist.add_trace(go.Scatter(x=r_vals, y=pdf_pdi2, mode='lines', name='PDI₂ Rec.', line=dict(color='purple')))
    
    fig_dist.update_layout(
        xaxis_title="Radius (nm)", yaxis_title="Prob",
        width=400, height=350,
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(x=0.7, y=1)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Download Section
params_dict = {'mean_rg': mean_rg, 'p_val': p_val, 'dist_type': dist_type}
download_data = create_download_data(r_vals, pdf_vals, pdf_pdi, pdf_pdi2, analysis_res, params_dict, input_label)

st.download_button(
    label="Download Distribution & Analysis (CSV)",
    data=download_data,
    file_name="saxs_analysis_results.csv",
    mime="text/csv"
)