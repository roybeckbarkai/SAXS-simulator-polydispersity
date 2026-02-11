# File: analysis_utils.py
# Last Updated: Wednesday, February 11, 2026
# Description: Analysis utilities including robust Forward-Modeling Tomchuk solver with corrected lc moments.

import numpy as np
import pandas as pd
from scipy.optimize import nnls, brentq
from scipy.integrate import trapezoid
from sim_utils import get_distribution, sphere_form_factor

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
                if len(nums) >= 2: data.append([nums[0], nums[1]])
            except: continue
        arr = np.array(data)
        if arr.shape[0] < 5: return None, None, "File too short"
        arr = arr[arr[:,0].argsort()]
        arr = arr[arr[:,0]>0]
        return arr[:,0], arr[:,1], None
    except Exception as e:
        return None, None, str(e)

def guinier_analysis(q, i):
    """
    Robust Guinier analysis.
    """
    debug = {'x':[], 'y':[], 'fit_y':[]}
    try:
        valid = (i > 0)
        q_v, i_v = q[valid], i[valid]
        if len(q_v) < 5: return 0, 0, debug
        
        best_rg = 0
        best_i0 = 0
        best_r2 = -1.0
        
        # Search for best linear region satisfying q*Rg < 1.3
        for start in [0, 2, 5]:
            limit = min(len(q_v), 30)
            sub_q = q_v[start:limit]
            sub_i = i_v[start:limit]
            if len(sub_q) < 5: continue
            
            p = np.polyfit(sub_q**2, np.log(sub_i), 1)
            rg_est = np.sqrt(-3*p[0]) if p[0]<0 else 0
            
            if rg_est > 0:
                limit_val = 1.3 / rg_est
                mask = q_v < limit_val
                if np.sum(mask) >= 5:
                    x = q_v[mask]**2
                    y = np.log(i_v[mask])
                    
                    p_final = np.polyfit(x, y, 1)
                    y_pred = p_final[1] + p_final[0]*x
                    
                    # Calculate R2
                    ss_res = np.sum((y - y_pred)**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_rg = np.sqrt(-3*p_final[0])
                        best_i0 = np.exp(p_final[1])
                        debug = {'x': x, 'y': y, 'fit_y': y_pred}
        
        # Fallback
        if best_rg == 0:
            p = np.polyfit(q_v[:5]**2, np.log(i_v[:5]), 1)
            best_i0 = np.exp(p[1])
            best_rg = np.sqrt(-3*p[0]) if p[0]<0 else 0
            
        return best_rg, best_i0, debug
    except:
        return 0, 0, debug

def calculate_invariants_porod(q, i):
    debug = {'porod_x': [], 'porod_y': [], 'B': 0}
    try:
        # Porod B: Mean of high-q plateau I*q^4 (Assuming Zero Background from Sim)
        # Use last 15% points
        n_tail = max(10, int(len(q)*0.15))
        q_t = q[-n_tail:]
        i_t = i[-n_tail:]
        
        porod_y = i_t * (q_t**4)
        B = np.mean(porod_y)
        C_bkg = 0.0 
        
        debug['porod_x'] = q_t**4
        debug['porod_y'] = porod_y
        debug['B'] = B
        
        integrand = i * (q**2)
        Q_data = trapezoid(integrand, q)
        q_max = q[-1]
        Q_tail = B / q_max if q_max > 0 else 0
        Q = Q_data + Q_tail
        
        integrand_lc = i * q
        num_data = trapezoid(integrand_lc, q)
        num_tail = B / (2 * q_max**2) if q_max > 0 else 0
        lc = (np.pi * (num_data + num_tail)) / Q if Q > 0 else 0
        
        return Q, B, C_bkg, lc, debug
    except:
        return 0, 0, 0, 0, debug

# --- Theoretical Ratios for Tomchuk ---

def get_theoretical_ratio_pdi(p, r_guess=10.0, dist_type='Gaussian'):
    """
    PDI1 (Volume) Ratio: V_p / Rg^3
    V_p = 4/3*pi * <R^6>/<R^3>
    Rg = sqrt(0.6 * <R^8>/<R^6>)
    """
    r_axis = np.linspace(0.1, r_guess*5, 2000)
    pdf = get_distribution(dist_type, r_axis, r_guess, p)
    
    m3 = trapezoid(r_axis**3 * pdf, r_axis)
    m6 = trapezoid(r_axis**6 * pdf, r_axis)
    m8 = trapezoid(r_axis**8 * pdf, r_axis)
    
    if m3 == 0 or m6 == 0: return 0, 0
    
    V_p = (4/3 * np.pi) * (m6 / m3)
    Rg = np.sqrt(0.6 * m8 / m6)
    
    return V_p / (Rg**3), Rg

def get_theoretical_ratio_pdi2(p, r_guess=10.0, dist_type='Gaussian'):
    """
    PDI2 (Correlation) Ratio: lc / Rg
    lc = 4V/S = 4/3 * <R^3>/<R^2>  (Corrected for polydisperse sphere)
    Rg = sqrt(0.6 * <R^8>/<R^6>)
    """
    r_axis = np.linspace(0.1, r_guess*5, 2000)
    pdf = get_distribution(dist_type, r_axis, r_guess, p)
    
    m2 = trapezoid(r_axis**2 * pdf, r_axis)
    m3 = trapezoid(r_axis**3 * pdf, r_axis)
    m6 = trapezoid(r_axis**6 * pdf, r_axis)
    m8 = trapezoid(r_axis**8 * pdf, r_axis)
    
    if m2 == 0 or m6 == 0: return 0, 0
    
    # Corrected l_c definition for polydisperse spheres
    lc = (4/3) * (m3 / m2)
    Rg = np.sqrt(0.6 * m8 / m6)
    
    return lc / Rg, Rg

def solve_tomchuk_forward(rg_obs, val_obs, mode='PDI', dist_type='Gaussian'):
    if rg_obs <= 0 or val_obs <= 0: return 0, 0, 0
    
    ratio_target = val_obs / (rg_obs**3) if mode == 'PDI' else val_obs / rg_obs
    
    def target_func(p):
        rat, _ = get_theoretical_ratio_pdi(p, 10.0, dist_type) if mode == 'PDI' else get_theoretical_ratio_pdi2(p, 10.0, dist_type)
        return rat - ratio_target
    
    try:
        p_res = brentq(target_func, 0.0, 5.0)
        
        # Recover Scaling
        _, rg_unit = get_theoretical_ratio_pdi(p_res, 1.0, dist_type) if mode == 'PDI' else get_theoretical_ratio_pdi2(p_res, 1.0, dist_type)
        
        scale = rg_obs / rg_unit
        
        # r_geo_rec is the 'mean_r' parameter of the distribution
        r_geo_rec = 1.0 * scale
        rg_scat_rec = rg_unit * scale 
        
        return r_geo_rec, p_res, rg_scat_rec
    except:
        return rg_obs/np.sqrt(0.6), 0.1, rg_obs

def generate_noise_free_fit(q, res, method_suffix, dist_type='Gaussian'):
    i_fit = np.zeros_like(q)
    r_key = 'r_geo_rec_pdi2' if method_suffix == 'PDI2' else 'r_geo_rec'
    p_key = 'p_rec_pdi2' if method_suffix == 'PDI2' else 'p_rec'
    
    if res.get(r_key, 0) > 0:
        r_geo = res[r_key]
        p = res.get(p_key, 0.1)
        r_axis = np.linspace(r_geo*0.1, r_geo*5, 100)
        pdf = get_distribution(dist_type, r_axis, r_geo, p)
        
        for r_val, prob in zip(r_axis, pdf):
            ff = sphere_form_factor(q, r_val)
            vol = (4/3)*np.pi*r_val**3
            i_fit += prob * (vol**2) * (ff**2)
            
        if i_fit[0] > 0 and 'G' in res:
            i_fit *= (res['G'] / i_fit[0])
            
    return i_fit

def perform_saxs_analysis(q, i_obs, method, mode_key, nnls_max_rg, dist_type='Gaussian'):
    results = {}
    q = np.array(q).ravel()
    i_obs = np.array(i_obs).ravel()
    
    # 1. Invariants
    rg_g, G, dbg_g = guinier_analysis(q, i_obs)
    Q, B, C, lc, dbg_p = calculate_invariants_porod(q, i_obs)
    results.update({'rg_guinier': rg_g, 'G': G, 'Q': Q, 'B': B, 'C_bkg': C, 'l_c': lc, 'debug_guinier': dbg_g, 'debug_porod': dbg_p})
    
    # 2. Tomchuk
    if "Tomchuk" in method and mode_key == 'Sphere':
        target_rg = rg_g if rg_g > 0 else 1.0
        
        # PDI1 (Volume Ratio)
        V_p = 2 * (np.pi**2) * G / Q if Q > 0 else 0
        r1, p1, rs1 = solve_tomchuk_forward(target_rg, V_p, 'PDI', dist_type)
        results['r_geo_rec'] = r1
        results['rg_rec'] = rs1
        results['p_rec'] = p1
        
        # PDI2 (Correlation Ratio)
        r2, p2, rs2 = solve_tomchuk_forward(target_rg, lc, 'PDI2', dist_type)
        results['r_geo_rec_pdi2'] = r2
        results['rg_rec_pdi2'] = rs2
        results['p_rec_pdi2'] = p2
        
        results['I_fit_pdi'] = generate_noise_free_fit(q, results, 'PDI', dist_type)
        results['I_fit_pdi2'] = generate_noise_free_fit(q, results, 'PDI2', dist_type)
        
    elif method == 'NNLS':
        try:
            r_basis = np.linspace(0.5, nnls_max_rg, 100)
            A = np.zeros((len(q), len(r_basis)))
            for j, r in enumerate(r_basis):
                ff = sphere_form_factor(q, r)
                vol = (4/3)*np.pi*r**3
                A[:, j] = (ff**2) * (vol**2)
            w, _ = nnls(A, i_obs)
            results['I_fit'] = A @ w
            results['nnls_r'] = r_basis
            results['nnls_w'] = w
            
            norm = w / ((4/3)*np.pi*r_basis**3)**2
            if np.sum(norm) > 0:
                m_r = np.average(r_basis, weights=norm)
                m6 = np.sum(norm * r_basis**6)
                m8 = np.sum(norm * r_basis**8)
                results['rg_rec'] = np.sqrt(0.6 * m8/m6) if m6>0 else 0
                
                var = np.average((r_basis-m_r)**2, weights=norm)
                results['p_rec'] = np.sqrt(var)/m_r
        except: pass

    # Chi2
    for key in ['I_fit_pdi', 'I_fit_pdi2', 'I_fit']:
        if key in results:
            f = results[key]
            m = f > 1e-9
            chi = np.sum((i_obs[m]-f[m])**2/f[m])/(np.sum(m)-1)
            results[f"chi2_{key.replace('I_fit_','')}"] = chi
            if method.lower() in key.lower() or (method=='NNLS' and key=='I_fit'): results['chi2'] = chi

    return results

def get_header_string(m_rg, p_val, dist_type, res):
    return "\n".join([
        f"# Simulation: Rg={m_rg:.2f}, p={p_val:.2f}, Dist={dist_type}",
        f"# Invariants: G={res.get('G',0):.2e}, Q={res.get('Q',0):.2e}, B={res.get('B',0):.2e}, lc={res.get('l_c',0):.2f}",
        f"# PDI1: Rg_scat={res.get('rg_rec',0):.2f}, p={res.get('p_rec',0):.3f}, Chi2={res.get('chi2_pdi',0):.2f}",
        f"# PDI2: Rg_scat={res.get('rg_rec_pdi2',0):.2f}, p={res.get('p_rec_pdi2',0):.3f}, Chi2={res.get('chi2_pdi2',0):.2f}"
    ]) + "\n"

def create_intensity_csv(header, q, i_input, res, method):
    df = pd.DataFrame({'q': q, 'I_obs': i_input})
    if 'I_fit_pdi' in res: df['I_Fit_PDI'] = res['I_fit_pdi']
    if 'I_fit_pdi2' in res: df['I_Fit_PDI2'] = res['I_fit_pdi2']
    if 'I_fit' in res: df['I_Fit_NNLS'] = res['I_fit']
    return header + df.to_csv(index=False)

def create_distribution_csv(header, r, input_dist, res, params):
    r = np.array(r).ravel() if r is not None else np.array([])
    df = pd.DataFrame({'Radius': r})
    if input_dist is not None: df['Input_PDF'] = np.array(input_dist).ravel()
    if 'nnls_w' in res and len(res['nnls_w']) == len(r): df['NNLS_Weight'] = res['nnls_w']
    return header + df.to_csv(index=False)