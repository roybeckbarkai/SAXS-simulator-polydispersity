# File: sim_utils.py
# Last Updated: Tuesday, February 10, 2026
# Description: Physics models, distribution functions, and core simulation engine.

import numpy as np
from scipy.special import gamma, factorial
from scipy.ndimage import gaussian_filter
from scipy.integrate import trapezoid

# --- Math & Physics Utilities ---
def double_factorial(n):
    if n <= 0: return 1
    return n * double_factorial(n - 2)

def nCr(n, r):
    try:
        return factorial(n) / (factorial(r) * factorial(n - r))
    except:
        return 1

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

# --- Core Simulation Runner ---
def run_simulation_core(params):
    mean_rg = float(params['mean_rg'])
    p_val = float(params['p_val'])
    dist_type = params['dist_type']
    mode_key = params['mode']
    pixels = int(float(params['pixels']))
    q_max = float(params['q_max'])
    q_min = float(params['q_min'])
    n_bins = int(float(params['n_bins']))
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

    # Radial Averaging
    if binning_mode == "Linear" or binning_mode == "Lin":
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
    # Return non-zero points for analysis
    return q_sim[valid_sim], radial_prof[valid_sim], i_2d_final, r_vals, pdf_vals