# File: sim_utils.py
# Last Updated: Wednesday, February 11, 2026
# Description: Core simulation with Ground Truth moment calculations for QA.

import numpy as np
from scipy.special import gamma, factorial
from scipy.ndimage import gaussian_filter
from scipy.integrate import trapezoid

# --- Distributions ---
def get_distribution(dist_type, r, mean_r, p):
    mean_r = max(mean_r, 1e-6)
    p = max(p, 1e-6)
    pdf = np.zeros_like(r, dtype=float)
    mask = r > 0
    r_valid = r[mask]
    
    if dist_type == 'Lognormal':
        s = np.sqrt(np.log(1 + p**2))
        scale = mean_r / np.exp(s**2 / 2.0)
        with np.errstate(all='ignore'):
            coef = 1.0 / (r_valid * s * np.sqrt(2 * np.pi))
            arg = (np.log(r_valid / scale)**2) / (2 * s**2)
            pdf[mask] = coef * np.exp(-arg)
    elif dist_type == 'Gaussian':
        sigma = p * mean_r
        coef = 1.0 / (sigma * np.sqrt(2 * np.pi))
        pdf[mask] = coef * np.exp(-0.5 * ((r_valid - mean_r) / sigma)**2)
    elif dist_type == 'Schulz':
        z = (1.0 / (p**2)) - 1.0
        if z <= 0: z = 100.0
        beta = mean_r / (z + 1.0)
        with np.errstate(all='ignore'):
            norm = (beta**(z + 1)) * gamma(z + 1)
            pdf[mask] = (r_valid**z * np.exp(-r_valid / beta)) / norm
    elif dist_type == 'Boltzmann':
        a = mean_r / 3.0
        norm = 2 * a**3
        pdf[mask] = (r_valid**2 * np.exp(-r_valid / a)) / norm
    elif dist_type == 'Triangular':
        half_width = p * mean_r * np.sqrt(6)
        min_r, max_r = mean_r - half_width, mean_r + half_width
        mask1 = (r_valid >= min_r) & (r_valid < mean_r)
        mask2 = (r_valid >= mean_r) & (r_valid <= max_r)
        pdf[mask][mask1] = (r_valid[mask1] - min_r) / (half_width**2)
        pdf[mask][mask2] = (max_r - r_valid[mask2]) / (half_width**2)
    elif dist_type == 'Uniform':
        half_width = p * mean_r * np.sqrt(3)
        min_r, max_r = max(0, mean_r - half_width), mean_r + half_width
        mask_u = (r_valid >= min_r) & (r_valid <= max_r)
        if (max_r - min_r) > 0:
            pdf[mask][mask_u] = 1.0 / (max_r - min_r)

    area = trapezoid(pdf, r)
    if area > 0: pdf /= area
    return np.nan_to_num(pdf)

def sphere_form_factor(q, r):
    qr = q * r
    with np.errstate(all='ignore'):
        val = 3 * (np.sin(qr) - qr * np.cos(qr)) / (qr**3)
        val[qr == 0] = 1.0
    return val

def run_simulation_core(params):
    # Inputs
    mean_r = params.get('mean_r', 2.0)
    p = params.get('p', 0.3)
    dist_type = params.get('dist_type', 'Gaussian')
    mode_key = params.get('mode_key', 'Sphere')
    q_min_user = params.get('q_min', 0.01)
    q_max = params.get('q_max', 2.5)
    n_bins = int(params.get('n_bins', 256))
    pixels = int(params.get('pixels', 1024))
    psf_pixels = params.get('smearing_sigma', 2.0)
    flux = params.get('flux', 1e5)
    add_noise = params.get('add_noise', True)
    max_rg_analysis = params.get('max_rg_analysis', mean_r * 10.0)

    # 1. Distribution & Ground Truth
    r_max_calc = mean_r * (1 + 6*p)
    r_axis = np.linspace(0.1, max(r_max_calc, 20.0), 1000)
    pdf = get_distribution(dist_type, r_axis, mean_r, p)

    # Calculate True Moments
    m3 = trapezoid(r_axis**3 * pdf, r_axis)
    m6 = trapezoid(r_axis**6 * pdf, r_axis)
    m8 = trapezoid(r_axis**8 * pdf, r_axis)
    
    true_rg_scat = np.sqrt(0.6 * m8 / m6) if m6 > 0 else 0
    # Porod Volume Vp = 2pi^2 I(0)/Q = 4/3 pi <R^6>/<R^3>
    true_vp = (4/3 * np.pi) * (m6 / m3) if m3 > 0 else 0
    # Expected I(0) proportional to m6 (scaled by contrast/volume later)
    
    qa_metrics = {
        'true_rg_scat': true_rg_scat,
        'true_vp': true_vp,
        'true_m6_m3': m6/m3 if m3 > 0 else 0
    }

    # 2. Detector
    center = pixels // 2
    y, x = np.ogrid[-center:pixels-center, -center:pixels-center]
    r_px = np.sqrt(x**2 + y**2)
    q_per_pixel = q_max / center
    q_map = r_px * q_per_pixel
    
    # 3. 1D Calculation
    q_fine_max = q_max * 1.42
    q_fine = np.linspace(0, q_fine_max, 2000)
    i_fine = np.zeros_like(q_fine)
    
    if mode_key == 'Sphere':
        for r_val, prob in zip(r_axis, pdf):
            if prob < 1e-6: continue
            ff = sphere_form_factor(q_fine, r_val)
            vol = (4/3) * np.pi * r_val**3
            i_fine += prob * (vol**2) * (ff**2)
    else:
        # IDP (Gaussian Chain)
        u = (q_fine * mean_r)**2
        with np.errstate(all='ignore'):
            f = 2 * (np.exp(-u) + u - 1) / (u**2)
            f[u==0] = 1.0
        i_fine = f * (mean_r**4)

    # 4. Map & Normalize
    i_2d = np.interp(q_map, q_fine, i_fine, right=0.0)
    
    # Normalize peak to Flux
    center_val = i_2d[center, center]
    if center_val > 0:
        i_2d = i_2d * (flux / center_val)

    if psf_pixels > 0:
        i_2d = gaussian_filter(i_2d, sigma=psf_pixels)

    if add_noise:
        i_2d_noisy = np.random.poisson(i_2d)
        i_2d_final = np.maximum(0, i_2d_noisy).astype(int)
    else:
        i_2d_final = i_2d

    # 5. Binning
    binning = params.get('binning_mode', 'Log')
    # Prevent empty bins: q_min must be >= resolution
    q_res = q_per_pixel
    q_min = max(q_min_user, q_res)
    
    flat_q = r_px.ravel() * q_per_pixel
    flat_i = i_2d_final.ravel()
    
    mask = flat_q <= q_max
    flat_q = flat_q[mask]
    flat_i = flat_i[mask]
    
    if binning == 'Linear':
        bins = np.linspace(q_min, q_max, n_bins+1)
    else:
        q_min_log = max(q_min, 1e-4)
        bins = np.logspace(np.log10(q_min_log), np.log10(q_max), n_bins+1)
        
    counts, _ = np.histogram(flat_q, bins, weights=flat_i)
    norm, _ = np.histogram(flat_q, bins)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        i_1d = counts / norm
    
    i_1d = np.nan_to_num(i_1d)
    
    if binning == 'Linear':
        q_1d = (bins[:-1] + bins[1:]) / 2
    else:
        q_1d = np.sqrt(bins[:-1] * bins[1:])

    # Remove empty bins
    valid = norm > 0
    q_1d = q_1d[valid]
    i_1d = i_1d[valid]

    return {
        'q': q_1d,
        'i_obs': i_1d,
        'i_2d': i_2d_final,
        'r_dist': r_axis,
        'pdf_dist': pdf,
        'max_rg_analysis': max_rg_analysis,
        'qa_metrics': qa_metrics
    }