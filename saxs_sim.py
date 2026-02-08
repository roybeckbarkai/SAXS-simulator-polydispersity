import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, gamma, factorial
from scipy.ndimage import gaussian_filter
from scipy.optimize import bisect
from scipy.integrate import trapezoid
import pandas as pd

# --- 1. Math & Physics Utilities ---

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
    # Avoid numerical errors
    mean_r = max(mean_r, 1e-6)
    p = max(p, 1e-6)
    
    if dist_type == 'Lognormal':
        # s = sqrt(ln(1 + p^2))
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
        # z = 1/p^2 - 1
        if p == 0: return np.zeros_like(r)
        z = (1.0 / (p**2)) - 1.0
        if z <= 0: return np.zeros_like(r)
        a = z + 1
        b = a / mean_r
        # (b^a / Gamma(a)) * r^z * exp(-b*r)
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
    # Using broadcasting: q is (Nq,), r is (Nr,)
    # qr will be (Nq, Nr) via outer product logic if passed correctly, 
    # but here we usually call this inside a loop or with broadcasting in mind.
    # To support broadcasting: q needs to be shape (Nq, 1), r shape (1, Nr)
    q_col = q[:, np.newaxis]
    r_row = r[np.newaxis, :]
    qr = q_col * r_row
    
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = 3 * (np.sin(qr) - qr * np.cos(qr)) / (qr**3)
    amp = np.nan_to_num(amp, nan=1.0) # limit as qr->0 is 1
    
    vol = (4.0/3.0) * np.pi * (r_row**3)
    # Intensity ~ Vol^2 * amp^2
    return (vol**2) * (amp**2)

# --- Analysis / Recovery Logic (Analytical Moments) ---

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
    # Returns Number Average Rg based on Scattering Average Rg
    if not rg_scat or rg_scat <= 0: return 0
    if p <= 0: return rg_scat
    
    m6 = get_normalized_moment(6, p, dist_type)
    m8 = get_normalized_moment(8, p, dist_type)
    if m6 <= 0: return 0
    
    # Rg_scat_theory = sqrt(3/5 * M8/M6) * R_num
    # Therefore R_num = Rg_scat / sqrt(3/5 * M8/M6)
    # But we want Rg_num which is sqrt(3/5)*R_num
    # So Rg_num = sqrt(3/5) * [ Rg_scat / sqrt(3/5 * M8/M6) ]
    #           = Rg_scat / sqrt(M8/M6)
    ratio = np.sqrt(m8 / m6)
    return rg_scat / ratio

# --- 2. Streamlit App Layout ---

st.set_page_config(page_title="SAXS Simulator", layout="wide", page_icon="⚛️")

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("SAXS Simulator: Polydisperse Spheres")
with col_h2:
    st.markdown("*Tomchuk et al. PDI Analysis*")

# --- Sidebar Controls ---
st.sidebar.header("Sample Parameters")
# Using geometric mean relation: <R> = sqrt(5/3) * <Rg>
mean_rg = st.sidebar.number_input("Mean Rg (nm)", min_value=0.5, max_value=50.0, value=4.0, step=0.5, help="Number-Average Radius of Gyration")
p_val = st.sidebar.number_input("Polydispersity (p)", min_value=0.01, max_value=6.0, value=0.3, step=0.01, help="sigma / mean_R")
dist_type = st.sidebar.selectbox("Distribution Type", ['Gaussian', 'Lognormal', 'Schulz', 'Boltzmann', 'Triangular', 'Uniform'])

st.sidebar.header("Instrument")
pixels = st.sidebar.number_input("Pixels", value=512, step=64)
# Auto-calculate q_max hint or default? We let user control it but default logic exists in React.
q_max = st.sidebar.number_input("Max q (nm⁻¹)", value=2.5, step=0.1)
smearing = st.sidebar.number_input("Smearing (px)", value=2.0, step=0.5)

st.sidebar.header("Flux & Noise")
col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    flux_pre = st.number_input("Flux Coeff", 0.1, 9.9, 1.0, 0.1)
with col_f2:
    flux_exp = st.number_input("Flux Exp (10^x)", 1, 15, 6, 1)

optimal_flux = st.sidebar.checkbox("Optimal Flux (No Noise)", value=False)
add_noise = st.sidebar.checkbox("Simulate Poisson Noise", value=True, disabled=optimal_flux)

# --- Calculation Engine ---

# 1. Physics Constants
mean_r = mean_rg * np.sqrt(5.0/3.0)
flux = flux_pre * (10**flux_exp)
sigma = p_val * mean_r

# 2. Distribution Arrays
# Extended range for broad distributions (15 sigma)
r_min = max(0.1, mean_r - 5 * sigma)
r_max = mean_r + 15 * sigma
r_steps = 400
r_vals = np.linspace(r_min, r_max, r_steps)
pdf_vals = get_distribution(dist_type, r_vals, mean_r, p_val)

# Normalize PDF
area = trapezoid(pdf_vals, r_vals)
if area > 0:
    pdf_vals /= area

# 3. 1D Ideal Profile
q_steps = 200
# Log space q for better low-q resolution
q_1d = np.logspace(np.log10(1e-3), np.log10(q_max * 1.5), q_steps)

# Intensity ~ Integral [ P(r) * I_sphere(q,r) ] dr
# sphere_form_factor handles q (Nq, 1) and r (1, Nr) broadcasting
i_matrix = sphere_form_factor(q_1d, r_vals) # Shape (Nq, Nr)
# Integrate over r (axis 1)
i_1d_ideal = trapezoid(i_matrix * pdf_vals, r_vals, axis=1) # Shape (Nq,)

# 4. 2D Image Generation
x = np.linspace(-q_max, q_max, pixels)
y = np.linspace(-q_max, q_max, pixels)
xv, yv = np.meshgrid(x, y)
qv_r = np.sqrt(xv**2 + yv**2)

# Interpolate 1D Ideal to 2D Grid
i_2d_ideal = np.interp(qv_r.ravel(), q_1d, i_1d_ideal, left=i_1d_ideal[0], right=0)
i_2d_ideal = i_2d_ideal.reshape(pixels, pixels)

# 5. Smearing
if smearing > 0:
    i_2d_smeared = gaussian_filter(i_2d_ideal, sigma=smearing)
else:
    i_2d_smeared = i_2d_ideal

# 6. Flux Scaling
total_int = np.sum(i_2d_smeared)
scale_factor = flux / total_int if total_int > 0 else 1.0
i_2d_scaled = i_2d_smeared * scale_factor

# 7. Noise
if not optimal_flux and add_noise:
    # Gaussian approx for Poisson: N(I, sqrt(I))
    noise_vals = np.random.normal(loc=0.0, scale=np.sqrt(np.maximum(i_2d_scaled, 1e-9)))
    i_2d_final = np.maximum(0, i_2d_scaled + noise_vals)
else:
    i_2d_final = i_2d_scaled

# 8. Radial Averaging (Experimental 1D)
# Radial binning
bin_width = q_max / (pixels/2)
r_indices = (qv_r / bin_width).astype(int).ravel()
valid_mask = r_indices < (pixels // 2)

tbin = np.bincount(r_indices[valid_mask.ravel()], weights=i_2d_final.ravel())
nr = np.bincount(r_indices[valid_mask.ravel()])

radial_prof = np.zeros_like(tbin)
nonzero = nr > 0
radial_prof[nonzero] = tbin[nonzero] / nr[nonzero]

q_exp = np.arange(len(radial_prof)) * bin_width
i_exp = radial_prof

# --- Analysis ---

# 1. Invariants with Porod Tail Correction
# Need B estimate first (last 20 points)
n_fit_b = min(20, len(q_exp)//4)
if n_fit_b > 0 and len(q_exp) > n_fit_b:
    b_region_i = i_exp[-n_fit_b:]
    b_region_q = q_exp[-n_fit_b:]
    b_vals = b_region_i * (b_region_q**4)
    B_est = np.mean(b_vals)
else:
    B_est = 0

q_max_meas = q_exp[-1] if len(q_exp) > 0 else 0

# Q Integral
integrand_q = (q_exp**2) * i_exp
Q_obs = trapezoid(integrand_q, q_exp)
Q_tail = B_est / q_max_meas if q_max_meas > 0 else 0
Q = Q_obs + Q_tail

# lc Integral
integrand_lc = q_exp * i_exp
lin_obs = trapezoid(integrand_lc, q_exp)
lin_tail = B_est / (2 * q_max_meas**2) if q_max_meas > 0 else 0
lc = (np.pi / Q) * (lin_obs + lin_tail) if Q > 0 else 0

# 2. Iterative Guinier Fit
rg_fit = mean_rg
g_fit = i_exp[0] if len(i_exp) > 0 else 1.0
valid_fit = False

for _ in range(3): # Iterate
    # Strict limit q*Rg < 1.0
    mask = (q_exp * rg_fit) < 1.0
    mask = mask & (q_exp > 0) & (i_exp > 0)
    
    if np.sum(mask) < 3: break
    
    x_fit = q_exp[mask]**2
    y_fit = np.log(i_exp[mask])
    
    try:
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        if slope >= 0: break # unphysical
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

rg_num_rec = get_calculated_mean_rg_num(rg_fit, p_rec_pdi, dist_type)

# Theoretical Rg (Scattering) Benchmark from PDF
# Rg_scat_theory = sqrt(3/5 * M8/M6) * R_mean_geom (since moments normalized to R=1)
# Actually here we integrated absolute moments (r^6, r^8) so the mean_r factor is included
m6_arr = (r_vals**6) * pdf_vals
m8_arr = (r_vals**8) * pdf_vals
m6_int = trapezoid(m6_arr, r_vals)
m8_int = trapezoid(m8_arr, r_vals)
rg_theory_scat = 0
if m6_int > 0:
    rg_theory_scat = np.sqrt(0.6 * (m8_int / m6_int))

# Recovery Distributions
mean_r_pdi = rg_num_rec * np.sqrt(5.0/3.0)
pdf_pdi = get_distribution(dist_type, r_vals, mean_r_pdi, p_rec_pdi)
# Normalize
a_pdi = trapezoid(pdf_pdi, r_vals)
if a_pdi > 0: pdf_pdi /= a_pdi

# --- 3. Visualization ---

# Two Columns: 2D & 1D
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.subheader("2D Detector")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(np.log10(np.maximum(i_2d_final, 1)), extent=[-q_max, q_max, -q_max, q_max], origin='lower', cmap='jet')
    plt.colorbar(im, ax=ax2, label="log10(I)")
    ax2.set_xlabel("qx (nm⁻¹)")
    ax2.set_ylabel("qy (nm⁻¹)")
    st.pyplot(fig2)

with col_viz2:
    st.subheader("1D Profile")
    plot_type = st.selectbox("Plot Type", ["Log-Log", "Lin-Lin", "Guinier", "Porod", "Kratky"])
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    
    if plot_type == "Log-Log":
        ax1.loglog(q_exp, i_exp, label='Simulated')
        ax1.set_xlabel("q (nm⁻¹)")
        ax1.set_ylabel("I(q)")
    elif plot_type == "Lin-Lin":
        ax1.plot(q_exp, i_exp, label='Simulated')
    elif plot_type == "Guinier":
        # Plot data
        x_g = q_exp**2
        y_g = np.log(np.maximum(i_exp, 1e-9))
        ax1.plot(x_g, y_g, '.', label='Data', markersize=3)
        # Plot Fit
        if valid_fit:
            x_line = np.linspace(0, (1.2/rg_fit)**2, 50)
            y_line = np.log(g_fit) - (rg_fit**2/3)*x_line
            ax1.plot(x_line, y_line, 'r--', label='Fit')
        ax1.set_xlabel("q² (nm⁻²)")
        ax1.set_ylabel("ln(I)")
        ax1.set_xlim(0, (2.0/mean_rg)**2)
        ax1.set_ylim(np.log(g_fit)-3, np.log(g_fit)+0.5)
    elif plot_type == "Porod":
        y_p = i_exp * (q_exp**4)
        ax1.plot(q_exp, y_p, label='Data')
        if B_est > 0:
            ax1.axhline(B_est, color='r', linestyle='--', label='B (Fit)')
        ax1.set_xlabel("q")
        ax1.set_ylabel("I · q⁴")
    elif plot_type == "Kratky":
        y_k = i_exp * (q_exp**2)
        ax1.plot(q_exp, y_k, label='Data')
        ax1.set_xlabel("q")
        ax1.set_ylabel("I · q²")

    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

# Analysis Panel
st.markdown("---")
st.subheader("Analysis Results")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Polydispersity (p)**")
    st.metric("Input", f"{p_val:.3f}")
    st.metric("Recovered (PDI)", f"{p_rec_pdi:.3f}")
    st.metric("Recovered (PDI₂)", f"{p_rec_pdi2:.3f}")
    
    st.markdown("**Mean Rg (Num Avg)**")
    st.metric("Input", f"{mean_rg:.2f} nm")
    st.metric("Recovered", f"{rg_num_rec:.2f} nm", delta=f"{rg_num_rec - mean_rg:.2f}")

with c2:
    st.markdown("**Parameters**")
    res_df = pd.DataFrame({
        "Parameter": ["Q", "lc", "Rg (Fit)", "Rg (Theory)", "G", "B"],
        "Value": [
            f"{Q:.2e}", 
            f"{lc:.2f} nm", 
            f"{rg_fit:.2f} nm", 
            f"{rg_theory_scat:.2f} nm", 
            f"{g_fit:.2e}", 
            f"{B_est:.2e}"
        ]
    })
    st.dataframe(res_df, hide_index=True)

with c3:
    st.markdown("**Distribution**")
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.plot(r_vals, pdf_vals, 'k--', label='Input')
    ax3.plot(r_vals, pdf_pdi, 'b-', alpha=0.6, label='Recovered')
    ax3.set_xlabel("Radius (nm)")
    ax3.set_ylabel("Prob")
    ax3.legend()
    st.pyplot(fig3)

# Download
csv = pd.DataFrame({"q": q_exp, "I": i_exp}).to_csv(index=False)
st.download_button("Download 1D Data", csv, "saxs_1d.csv", "text/csv")