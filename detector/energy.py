import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math
from .fnfit import bump, step_function, Curve
import torch


# (time_array, x_array, y_array, vx_array, vy_array, ax_array, ay_array, rho_restricted_to_curve(t), derivative of w, second derivative of w, focal length of camera)
def compute_projective_bending_energy(t, x, y, vx, vy, ax, ay, w, w_dt, w_ddt, f):
    """
    Computes the parameterization-invariant bending energy of the curve embedded in 3D space
    """
    V1_x = -f * vy
    V1_y =  f * vx
    V1_z =  x * vy - y * vx
    
    V2_x = -f * ay
    V2_y =  f * ax
    V2_z =  x * ay - y * ax
    
    V3_z = vx * ay - vy * ax
    
    c1 = 2 * (w_dt**2) - (w * w_ddt)
    c2 = w * w_dt
    c3 = w**2
    
    K_x = c1 * V1_x + c2 * V2_x
    K_y = c1 * V1_y + c2 * V2_y
    K_z = c1 * V1_z + c2 * V2_z + c3 * V3_z
    
    K_norm_sq = K_x**2 + K_y**2 + K_z**2
    
    # Step 4.2: The Speed Squared S(t)
    term1 = (w_dt**2) * (x**2 + y**2 + f**2)
    term2 = 2 * w * w_dt * (x * vx + y * vy)
    term3 = (w**2) * (vx**2 + vy**2)
    
    S = term1 + term2 + term3
    
    # Add epsilon to prevent division by zero in perfectly static segments
    eps = 1e-12
    if torch.is_tensor(S):
        integrand = K_norm_sq / torch.clamp(S + eps, min=eps)**2.5 
        energy = torch.trapezoid(integrand, t)
        return energy
    else:
        integrand = K_norm_sq / (np.maximum(S + eps, eps)**2.5)
        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        energy = trapz_fn(integrand, t)
        return float(np.asarray(energy).item() if np.ndim(energy) > 0 else energy)

def bump_2d(u, v, du, dv, u0, v0, ju, jv, deg=2):
    """
    2D Tensor Product B-spline bump function.
    Constructs the 2D surface patches and exact spatial derivatives.
    """
    Bu, Bu_du, Bu_ddu = bump(u, du, u0, ju, deg)
    Bv, Bv_dv, Bv_ddv = bump(v, dv, v0, jv, deg)
    
    B_val = Bu * Bv
    B_u = Bu_du * Bv
    B_v = Bu * Bv_dv
    B_uu = Bu_ddu * Bv
    B_vv = Bu * Bv_ddv
    B_uv = Bu_du * Bv_dv
    
    return B_val, B_u, B_v, B_uu, B_vv, B_uv

def surface_fit(u, v, cloud, deg=2, bin_size=1):
    u = torch.as_tensor(u, dtype=cloud.dtype, device=cloud.device)
    v = torch.as_tensor(v, dtype=cloud.dtype, device=cloud.device)
    cloud_u, cloud_v, cloud_rho = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    
    # Define bounds based on the target evaluation grid to prevent boundary collapse
    u_min, u_max = float(torch.min(u)), float(torch.max(u))
    v_min, v_max = float(torch.min(v)), float(torch.max(v))
    
    num_bins_u = int((u_max - u_min) // bin_size) + 1
    num_bins_v = int((v_max - v_min) // bin_size) + 1
    
    local_fits = {}
    for i in range(num_bins_u):
        for j in range(num_bins_v):
            bu_start = u_min + i * bin_size
            bu_end = u_min + (i + 1) * bin_size
            bv_start = v_min + j * bin_size
            bv_end = v_min + (j + 1) * bin_size
            
            # Bin centers for well-conditioned local polynomial fits
            cu = u_min + (i + 0.5) * bin_size
            cv = v_min + (j + 0.5) * bin_size
            
            mask = (cloud_u >= bu_start) & (cloud_u < bu_end) & (cloud_v >= bv_start) & (cloud_v < bv_end)
            
            if torch.sum(mask) < 6: 
                local_fits[(i, j)] = None
                continue
                
            # Center coordinates to prevent explosive quadratic coefficients
            lu, lv, lr = cloud_u[mask] - cu, cloud_v[mask] - cv, cloud_rho[mask]
            A = torch.column_stack([torch.ones_like(lu), lu, lv, lu**2, lu*lv, lv**2])
            coeffs = torch.linalg.pinv(A) @ lr
            local_fits[(i, j)] = (coeffs, cu, cv)

    valid_fits = {k: v for k, v in local_fits.items() if v is not None}

    for i in range(num_bins_u):
        for j in range(num_bins_v):
            if local_fits[(i, j)] is None:
                cu = u_min + (i + 0.5) * bin_size
                cv = v_min + (j + 0.5) * bin_size
                if valid_fits:
                    nearest_k = min(valid_fits.keys(), key=lambda k: (k[0]-i)**2 + (k[1]-j)**2)
                    source_coeffs, _, _ = valid_fits[nearest_k]
                    local_fits[(i, j)] = (source_coeffs, cu, cv)
                else:
                    lu_all = cloud_u - cu
                    lv_all = cloud_v - cv
                    A_all = torch.column_stack([torch.ones_like(lu_all), lu_all, lv_all, lu_all**2, lu_all*lv_all, lv_all**2])
                    local_coeffs = torch.linalg.pinv(A_all) @ cloud_rho
                    local_fits[(i, j)] = (local_coeffs, cu, cv)

    g_val = torch.zeros_like(u, dtype=cloud_rho.dtype, device=cloud_rho.device)
    g_u   = torch.zeros_like(u, dtype=cloud_rho.dtype, device=cloud_rho.device)
    g_v   = torch.zeros_like(u, dtype=cloud_rho.dtype, device=cloud_rho.device)
    g_uu  = torch.zeros_like(u, dtype=cloud_rho.dtype, device=cloud_rho.device)
    g_vv  = torch.zeros_like(u, dtype=cloud_rho.dtype, device=cloud_rho.device)
    g_uv  = torch.zeros_like(u, dtype=cloud_rho.dtype, device=cloud_rho.device)

    for ju in range(-deg, num_bins_u):
        for jv in range(-deg, num_bins_v):
            idx_u = max(0, min(ju + (deg // 2), num_bins_u - 1))
            idx_v = max(0, min(jv + (deg // 2), num_bins_v - 1))
            
            coeffs, cu, cv = local_fits[(idx_u, idx_v)]
            c0, c1, c2, c3, c4, c5 = coeffs
            
            du_val = u - cu
            dv_val = v - cv
            
            P_val = c0 + c1*du_val + c2*dv_val + c3*du_val**2 + c4*du_val*dv_val + c5*dv_val**2
            P_u = c1 + 2*c3*du_val + c4*dv_val
            P_v = c2 + c4*du_val + 2*c5*dv_val
            P_uu = 2*c3 * torch.ones_like(u)
            P_vv = 2*c5 * torch.ones_like(u)
            P_uv = c4 * torch.ones_like(u)
            
            B_val_np, B_u_np, B_v_np, B_uu_np, B_vv_np, B_uv_np = bump_2d(u.detach().cpu().numpy(), v.detach().cpu().numpy(), bin_size, bin_size, u_min, v_min, ju, jv, deg)
            
            # Convert immediately to PyTorch tensors for multiplication
            B_val = torch.as_tensor(B_val_np, dtype=cloud_rho.dtype, device=cloud_rho.device)
            B_u   = torch.as_tensor(B_u_np, dtype=cloud_rho.dtype, device=cloud_rho.device)
            B_v   = torch.as_tensor(B_v_np, dtype=cloud_rho.dtype, device=cloud_rho.device)
            B_uu  = torch.as_tensor(B_uu_np, dtype=cloud_rho.dtype, device=cloud_rho.device)
            B_vv  = torch.as_tensor(B_vv_np, dtype=cloud_rho.dtype, device=cloud_rho.device)
            B_uv  = torch.as_tensor(B_uv_np, dtype=cloud_rho.dtype, device=cloud_rho.device)
            
            g_val += B_val * P_val
            g_u += B_u * P_val + B_val * P_u
            g_v += B_v * P_val + B_val * P_v
            g_uu += B_uu * P_val + 2 * B_u * P_u + B_val * P_uu
            g_vv += B_vv * P_val + 2 * B_v * P_v + B_val * P_vv
            g_uv += B_uv * P_val + B_u * P_v + B_v * P_u + B_val * P_uv

    return g_val, g_u, g_v, g_uu, g_vv, g_uv

def projective_kinematics(x, y, vx, vy, ax, ay, f):
    """
    Computes exact algebraic time derivatives for stereographic coordinates u(t), v(t) 
    and the spherical scaling factor lambda(t).
    """
    R = np.sqrt(x**2 + y**2 + f**2)
    R_t = (x*vx + y*vy) / R
    R_tt = ((vx**2 + x*ax + vy**2 + y*ay) - R_t**2) / R
    
    lam = 1.0 / R
    lam_t = -R_t / (R**2)
    lam_tt = (2*R_t**2 - R*R_tt) / (R**3)
    
    D = R + f
    D_t = R_t
    D_tt = R_tt
    
    u = x / D
    u_t = (vx*D - x*D_t) / (D**2)
    u_tt = (ax*D - x*D_tt) / (D**2) - (2*D_t*u_t) / D
    
    v = y / D
    v_t = (vy*D - y*D_t) / (D**2)
    v_tt = (ay*D - y*D_tt) / (D**2) - (2*D_t*v_t) / D
    
    return u, u_t, u_tt, v, v_t, v_tt, lam, lam_t, lam_tt

def total_energy(T, curves, cloud_coords, cloud_values, f = 50.0):
    total_E = torch.tensor(0.0, dtype=torch.float64)
    t = torch.as_tensor(T)

    # convert to torch and extract value list
    #deformation_cloud = torch.as_tensor(deformation_cloud, dtype=torch.float64)
    #cloud_coords = deformation_cloud[:, :2]
    #cloud_values = deformation_cloud[:, 2].detach().clone().requires_grad_(True)
    deformation_cloud = torch.column_stack([cloud_coords, cloud_values])

    for curve in curves:
        x,y = curve.point_at(t)
        vx, vy = curve.velocity_at(t)
        ax, ay = curve.acceleration_at(t)

        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        vx, vy = np.asarray(vx).ravel(), np.asarray(vy).ravel()
        ax, ay = np.asarray(ax).ravel(), np.asarray(ay).ravel()
        
        # 1. Map continuous x(t), y(t) to stereographic u(t), v(t)
        u, u_t, u_tt, v, v_t, v_tt, lam, lam_t, lam_tt = projective_kinematics(x, y, vx, vy, ax, ay, f)
        
        # 2. Evaluate rho surface and exact spatial derivatives at u(t), v(t)
        rho, rho_u, rho_v, rho_uu, rho_vv, rho_uv = surface_fit(u, v, deformation_cloud, deg=2, bin_size=1)

        # Convert NumPy kinematics to PyTorch tensors before mixing with rho
        u_t = torch.as_tensor(u_t, dtype=torch.float64)
        u_tt = torch.as_tensor(u_tt, dtype=torch.float64)
        v_t = torch.as_tensor(v_t, dtype=torch.float64)
        v_tt = torch.as_tensor(v_tt, dtype=torch.float64)
        
        lam = torch.as_tensor(lam, dtype=torch.float64)
        lam_t = torch.as_tensor(lam_t, dtype=torch.float64)
        lam_tt = torch.as_tensor(lam_tt, dtype=torch.float64)

        x = torch.as_tensor(x, dtype=torch.float64)
        y = torch.as_tensor(y, dtype=torch.float64)
        vx = torch.as_tensor(vx, dtype=torch.float64)
        vy = torch.as_tensor(vy, dtype=torch.float64)
        ax = torch.as_tensor(ax, dtype=torch.float64)
        ay = torch.as_tensor(ay, dtype=torch.float64)
        
        # 3. Multivariable chain rule for temporal derivatives
        rho_t = rho_u * u_t + rho_v * v_t
        rho_tt = (rho_uu * u_t**2 + 2 * rho_uv * u_t * v_t + rho_vv * v_t**2) + rho_u * u_tt + rho_v * v_tt
        
        # 4. Construct depth function w(t) and its exact temporal derivatives
        w = rho * lam
        w_dt = rho_t * lam + rho * lam_t
        w_ddt = rho_tt * lam + 2 * rho_t * lam_t + rho * lam_tt
        
        # 5. Compute structural bending energy
        energy = compute_projective_bending_energy(t, x, y, vx, vy, ax, ay, w, w_dt, w_ddt, f)
        total_E = total_E + energy

    return total_E
