import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math
from .fnfit import bump, step_function, Curve
import torch

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
    u = torch.as_tensor(u, dtype=torch.float64, device=cloud.device)
    v = torch.as_tensor(v, dtype=torch.float64, device=cloud.device)
    cloud = torch.as_tensor(cloud, dtype=torch.float64)
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


def total_energy(T, curves, cloud_coords, cloud_values, f=1.0):
    return evaluate_penalties(T, curves, cloud_coords, cloud_values, f=1.0)[0]
def evaluate_complexity(T, curves, cloud_coords, cloud_values, f=1.0):
    return evaluate_penalties(T, curves, cloud_coords, cloud_values, f=1.0)[2]

def evaluate_penalties(T, curves, cloud_coords, cloud_values, f=1.0, arcmultiply=False):
    total_E = torch.tensor(0.0, dtype=torch.float64, device=cloud_coords.device)
    t = torch.as_tensor(T)
    geo_E = torch.tensor(0.0, dtype=torch.float64, device=cloud_coords.device)
    mean_E = torch.tensor(0.0, dtype=torch.float64, device=cloud_coords.device)
    gauss_E = torch.tensor(0.0, dtype=torch.float64, device=cloud_coords.device)
    var_E = torch.tensor(0.0, dtype=torch.float64, device=cloud_coords.device)
    height_E = torch.tensor(0.0, dtype=torch.float64, device=cloud_coords.device)

    deformation_cloud = torch.column_stack([cloud_coords, cloud_values])
    arclens = []

    lambda_depth = 0.0
    lambda_h = 0.0
    lambda_k = 0.0
    lambda_var = 0.0

    for curve in curves:
        x, y = curve.point_at(t)
        vx, vy = curve.velocity_at(t)
        ax, ay = curve.acceleration_at(t)

        x = torch.as_tensor(x, dtype=torch.float64).ravel()
        y = torch.as_tensor(y, dtype=torch.float64).ravel()
        vx = torch.as_tensor(vx, dtype=torch.float64).ravel()
        vy = torch.as_tensor(vy, dtype=torch.float64).ravel()
        ax = torch.as_tensor(ax, dtype=torch.float64).ravel()
        ay = torch.as_tensor(ay, dtype=torch.float64).ravel()
        f_tensor = torch.full_like(x, f)

        w, w_x, w_y, w_xx, w_yy, w_xy = surface_fit(x, y, deformation_cloud, deg=2, bin_size=0.05)
        
        gamma_x = torch.stack([w_x * x + w, w_x * y, w_x * f_tensor], dim=1)
        gamma_y = torch.stack([w_y * x, w_y * y + w, w_y * f_tensor], dim=1)
        gamma_xx = torch.stack([w_xx * x + 2 * w_x, w_xx * y, w_xx * f_tensor], dim=1)
        gamma_yy = torch.stack([w_yy * x, w_yy * y + 2 * w_y, w_yy * f_tensor], dim=1)
        gamma_xy = torch.stack([w_xy * x + w_y, w_xy * y + w_x, w_xy * f_tensor], dim=1)

        N_cross = torch.linalg.cross(gamma_x, gamma_y, dim=1)
        N_norm = torch.linalg.norm(N_cross, dim=1, keepdim=True)
        N_vec = N_cross / (N_norm + 1e-12)

        w_t = w_x * vx + w_y * vy
        w_tt = (w_xx * vx**2 + 2 * w_xy * vx * vy + w_yy * vy**2) + w_x * ax + w_y * ay
        w_ = w.unsqueeze(1)
        w_t_ = w_t.unsqueeze(1)
        w_tt_ = w_tt.unsqueeze(1)

        P = torch.stack([x, y, f_tensor], dim=1)
        P_t = torch.stack([vx, vy, torch.zeros_like(x)], dim=1)
        P_tt = torch.stack([ax, ay, torch.zeros_like(x)], dim=1)

        gamma_t = w_t_ * P + w_ * P_t
        gamma_tt = w_tt_ * P + 2 * w_t_ * P_t + w_ * P_tt

        v_cross_a = torch.linalg.cross(gamma_t, gamma_tt, dim=1)
        a_T = (v_cross_a * N_vec).sum(dim=1)

        speed_sq = (gamma_t**2).sum(dim=1)
        speed_sq_clamped = torch.clamp(speed_sq, min=1e-12)

        integrand = (a_T**2) / (speed_sq_clamped**2.5)
        trim = max(1, len(t) // 10) if len(t) > 10 else 0
        if trim > 0:
            energy = torch.trapezoid(integrand[trim:-trim], t[trim:-trim])
            arclen = torch.trapezoid(torch.sqrt(speed_sq_clamped[trim:-trim]), t[trim:-trim])
        else:
            energy = torch.trapezoid(integrand, t)
            arclen = torch.trapezoid(torch.sqrt(speed_sq_clamped), t)

        arclens.append(arclen)

        E = (gamma_x * gamma_x).sum(dim=1)
        F = (gamma_x * gamma_y).sum(dim=1)
        G = (gamma_y * gamma_y).sum(dim=1)
        L = (gamma_xx * N_vec).sum(dim=1)
        M = (gamma_xy * N_vec).sum(dim=1)
        N = (gamma_yy * N_vec).sum(dim=1)
        H = (E * N - 2.0 * F * M + G * L) / (2.0 * (E * G - F**2) + 1e-12)
        K = (L * N - M**2) / (E * G - F**2 + 1e-12)


        h_penalty = torch.trapezoid(H**2, t)
        k_penalty = torch.trapezoid(K**2, t)

        depth_penalty = torch.trapezoid(torch.relu(w - 3.0)**2 + torch.relu(0.2 - w)**2, t)


        geo_E = geo_E + energy
        mean_E = mean_E + lambda_h * h_penalty 
        gauss_E = gauss_E + lambda_k * k_penalty 
        height_E = height_E + lambda_depth * depth_penalty 
        total_E = total_E + energy + (lambda_h * h_penalty) + (lambda_k * k_penalty) + (lambda_depth * depth_penalty)

    if len(arclens) > 1:
        arclens_tensor = torch.stack(arclens)
        var = torch.var(arclens_tensor)
    else:
        var = torch.tensor(0.0, dtype=torch.float64, device=total_E.device)

    print("Length variance ", lambda_var * var)

    var_E = lambda_var * var
    total_E = total_E + lambda_var * var

    return (total_E, geo_E, mean_E, gauss_E, height_E, var_E)
