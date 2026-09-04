import numpy as np
from scipy.optimize import minimize
from .energy import (
    surface_fit,
    projective_kinematics,
    compute_projective_bending_energy,
    total_energy
)

# ==============================================================================
# TOGGLES FOR FIXES: Set any of these to False to revert them individually
# ==============================================================================
USE_CONE_SAMPLING = True    # Fix 2: Sample only within field of view cone instead of full S^2_+
USE_REGULARIZATION = True   # Fix 3: Add alpha * sum((rho - 1)^2) penalty to prevent inflation
USE_PARAMETER_BOUNDS = True # Fix 1: Constrain rho to [0.5, 2.0] via SciPy bounds

# Hyperparameter for Fix 3: Regularization strength
ALPHA_REG = 1.0


def generate_uniform_rho_cloud(num_points=64, max_phi=None):
    """
    Uniformly samples a spherical cap on S^2_+ and maps to stereographic (u, v) with rho=1.
    If max_phi is None, samples the full upper hemisphere [0, pi/2].
    """
    indices = np.arange(0, num_points, dtype=float) + 0.5
    
    # ==========================================================================
    # FIX 2: Sample Only the Image Cone (Logic)
    # ==========================================================================
    if USE_CONE_SAMPLING and max_phi is not None:
        # Uniform area sampling on a spherical cap of half-angle max_phi:
        # cos(phi) runs from 1 down to cos(max_phi)
        cos_max = np.cos(max_phi)
        phi = np.arccos(1.0 - (indices / num_points) * (1.0 - cos_max))
    else:
        # Revert: Sample full half-sphere [0, pi/2]
        phi = np.arccos(1.0 - indices / num_points)
    # ==========================================================================
    
    theta = np.pi * (1.0 + 5.0**0.5) * indices
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    u = x / (1.0 + z)
    v = y / (1.0 + z)
    rho = np.ones_like(u)
    
    return np.column_stack([u, v, rho])


def run_gradient_descent(T_domain, curves, f_pixels):
    """Delegates the gradient descent to SciPy's optimized L-BFGS-B solver."""
    
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
        return generate_uniform_rho_cloud(num_points=64)

    all_u = np.concatenate(u_list)
    all_ut = np.concatenate(ut_list)
    all_utt = np.concatenate(utt_list)
    all_v = np.concatenate(v_list)
    all_vt = np.concatenate(vt_list)
    all_vtt = np.concatenate(vtt_list)
    all_lam = np.concatenate(lam_list)
    all_lamt = np.concatenate(lamt_list)
    all_lamtt = np.concatenate(lamtt_list)

    # ==========================================================================
    # FIX 2: Sample Only the Image Cone (Calculation of Angular Bound)
    # ==========================================================================
    if USE_CONE_SAMPLING:
        # Determine maximum angular radius (FOV half-angle) spanned by curve data points
        # with a 15% safety margin to ensure proper boundary support for splines
        all_r = np.sqrt(np.concatenate([data[1]**2 + data[2]**2 for data in curve_data]))
        max_r = np.max(all_r) if len(all_r) > 0 else f_pixels
        max_phi = min(np.arctan2(max_r, f_pixels) * 1.15, np.pi / 2.0 * 0.95)
    else:
        max_phi = None
    
    cloud = generate_uniform_rho_cloud(num_points=64, max_phi=max_phi)
    # ==========================================================================
    
    initial_rho = cloud[:, 2].copy()

    def objective(rho_params):
        cloud[:, 2] = rho_params
        rho, rho_u, rho_v, rho_uu, rho_vv, rho_uv = surface_fit(all_u, all_v, cloud, deg=2, bin_size=1)
        
        rho_t = rho_u * all_ut + rho_v * all_vt
        rho_tt = (rho_uu * all_ut**2 + 2.0 * rho_uv * all_ut * all_vt + rho_vv * all_vt**2) + rho_u * all_utt + rho_v * all_vtt
        
        w = rho * all_lam
        w_dt = rho_t * all_lam + rho * all_lamt
        w_ddt = rho_tt * all_lam + 2.0 * rho_t * all_lamt + rho * all_lamtt
        
        total_E = 0.0
        for sl, x, y, vx, vy, ax, ay in curve_data:
            energy = compute_projective_bending_energy(
                T_fast, x, y, vx, vy, ax, ay, 
                w[sl], w_dt[sl], w_ddt[sl], f_pixels
            )
            total_E += float(energy)
            
        # ======================================================================
        # FIX 3: Add Area/Depth Regularization Penalty
        # ======================================================================
        if USE_REGULARIZATION:
            reg_penalty = ALPHA_REG * np.sum((rho_params - 1.0)**2)
            total_E += reg_penalty
        # ======================================================================
            
        return total_E

    # ==========================================================================
    # FIX 1: Constrain rho to [0.5, 2.0]
    # ==========================================================================
    if USE_PARAMETER_BOUNDS:
        bounds = [(0.5, 2.0) for _ in range(len(initial_rho))]
    else:
        bounds = None
    # ==========================================================================
        
    print("Delegating optimization to SciPy (L-BFGS-B)...")
    
    result = minimize(
        fun=objective, 
        x0=initial_rho, 
        method='L-BFGS-B',
        jac='2-point',
        bounds=bounds,  # Applies Fix 1 (or None if reverted)
        options={
            'disp': True,
            'maxiter': 50,
            'ftol': 1e-5
        }
    )
    
    cloud[:, 2] = result.x
    print(f"Optimization finished. Final Objective: {result.fun:.5f}")
    
    return cloud
