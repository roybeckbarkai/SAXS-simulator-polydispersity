# File: analysis_utils.py 
# Description: Utilities for parsing data, performing Tomchuk analysis, and NNLS distribution recovery.

import numpy as np
from scipy.optimize import bisect, nnls
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter
import pandas as pd
from sim_utils import nCr, double_factorial, sphere_form_factor, debye_form_factor, get_distribution

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
        # Filter negative q
        arr = arr[arr[:, 0] > 1e-6]
        
        return arr[:, 0], arr[:, 1], None
        
    except Exception as e:
        return None, None, f"Error parsing file: {str(e)}"

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

def get_calculated_mean_rg_num(rg_scat, p, dist_type):
    # Returns Number Average Rg based on Scattering Average Rg
    if not rg_scat or rg_scat <= 0: return 0
    if p <= 0: return rg_scat
    
    m6 = get_normalized_moment(6, p, dist_type)
    m8 = get_normalized_moment(8, p, dist_type)
    if m6 <= 0: return 0
    
    # ratio = Rg_scat / Rg_num_theory
    ratio = np.sqrt(m8 / m6)
    return rg_scat / ratio

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
            
            # Rg Recovery
            rg_num_rec = get_calculated_mean_rg_num(rg_fit, p_rec_pdi, dist_type)
            rg_num_rec_2 = get_calculated_mean_rg_num(rg_fit, p_rec_pdi2, dist_type)

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
            
            # Reconstruct Fit
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