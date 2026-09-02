import numpy as np
from scipy.optimize import minimize
from .energy import (
    surface_fit,
    projective_kinematics,
    compute_projective_bending_energy,
    total_energy
)

def generate_uniform_rho_cloud(num_points=64):
    """Uniformly samples S^2_+ and maps to stereographic (u,v) with rho=1."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - indices/num_points) 
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    u = x / (1 + z)
    v = y / (1 + z)
    rho = np.ones_like(u)
    
    return np.column_stack([u, v, rho])

def run_gradient_descent(T_domain, curves, f_pixels):
    """Delegates the gradient descent to SciPy's optimized L-BFGS-B solver."""
    
    cloud = generate_uniform_rho_cloud(num_points=64)
    initial_rho = cloud[:, 2].copy()
    
    # Downsample the time domain strictly for the optimization loop to speed up evaluations
    T_fast = np.linspace(0, 1, 20) 
    
    # Precompute kinematics for all curves at T_fast once before the optimization loop
    u_list, ut_list, utt_list, v_list, vt_list, vtt_list = [], [], [], [], [], []
    lam_list, lamt_list, lamtt_list = [], [], []
    curve_data = []
    offset = 0

    for curve in curves:
        x, y = curve.point_at(T_fast)
        vx, vy = curve.velocity_at(T_fast)
        ax, ay = curve.acceleration_at(T_fast)
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        vx, vy = np.asarray(vx).ravel(), np.asarray(vy).ravel()
        ax, ay = np.asarray(ax).ravel(), np.asarray(ay).ravel()
        
        u, u_t, u_tt, v, v_t, v_tt, lam, lam_t, lam_tt = projective_kinematics(x, y, vx, vy, ax, ay, f_pixels)
        
        n_pts = len(x)
        curve_data.append((slice(offset, offset + n_pts), x, y, vx, vy, ax, ay))
        offset += n_pts
        
        u_list.append(u); ut_list.append(u_t); utt_list.append(u_tt)
        v_list.append(v); vt_list.append(v_t); vtt_list.append(v_tt)
        lam_list.append(lam); lamt_list.append(lam_t); lamtt_list.append(lam_tt)

    if not curve_data:
        print("No valid curves found for optimization.")
        return cloud

    all_u = np.concatenate(u_list)
    all_ut = np.concatenate(ut_list)
    all_utt = np.concatenate(utt_list)
    all_v = np.concatenate(v_list)
    all_vt = np.concatenate(vt_list)
    all_vtt = np.concatenate(vtt_list)
    all_lam = np.concatenate(lam_list)
    all_lamt = np.concatenate(lamt_list)
    all_lamtt = np.concatenate(lamtt_list)

    def objective(rho_params):
        cloud[:, 2] = rho_params
        rho, rho_u, rho_v, rho_uu, rho_vv, rho_uv = surface_fit(all_u, all_v, cloud, deg=2, bin_size=1)
        
        rho_t = rho_u * all_ut + rho_v * all_vt
        rho_tt = (rho_uu * all_ut**2 + 2 * rho_uv * all_ut * all_vt + rho_vv * all_vt**2) + rho_u * all_utt + rho_v * all_vtt
        
        w = rho * all_lam
        w_dt = rho_t * all_lam + rho * all_lamt
        w_ddt = rho_tt * all_lam + 2 * rho_t * all_lamt + rho * all_lamtt
        
        total_E = 0.0
        for sl, x, y, vx, vy, ax, ay in curve_data:
            energy = compute_projective_bending_energy(
                T_fast, x, y, vx, vy, ax, ay, 
                w[sl], w_dt[sl], w_ddt[sl], f_pixels
            )
            total_E += float(energy)
        return total_E
        
    print("Delegating optimization to SciPy (L-BFGS-B)...")
    
    result = minimize(
        fun=objective, 
        x0=initial_rho, 
        method='L-BFGS-B',
        jac='2-point',
        options={
            'disp': True,
            'maxiter': 50,
            'ftol': 1e-5
        }
    )
    
    cloud[:, 2] = result.x
    print(f"Optimization finished. Final Energy: {result.fun:.5f}")
    
    return cloud
