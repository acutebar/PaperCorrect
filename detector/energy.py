import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math
#from fnfit import bump, step_function, Curve


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
    
    integrand = K_norm_sq / ((S + eps)**2.5)
    
    # Integrate over the parameter t using trapezoidal approximation
    energy = np.trapezoid(integrand, t)
    
    return energy


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

def surface_fit(u, v, cloud, deg=2, bin_size=0.5):
    """
    Fits a local 2D quadratic polynomial surface to the stereographic domain and blends 
    it globally using 2D bump functions.
    cloud is expected as an Nx3 array: [u_coords, v_coords, rho_values].
    """
    cloud_u, cloud_v, cloud_rho = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    
    u_min, u_max = np.min(cloud_u), np.max(cloud_u)
    v_min, v_max = np.min(cloud_v), np.max(cloud_v)
    
    num_bins_u = int((u_max - u_min) // bin_size) + 1
    num_bins_v = int((v_max - v_min) // bin_size) + 1
    
    local_fits = {}
    
    # Generate local 2D polynomial surfaces
    for i in range(num_bins_u):
        for j in range(num_bins_v):
            bu_start = u_min + i * bin_size
            bu_end = u_min + (i + 1) * bin_size
            bv_start = v_min + j * bin_size
            bv_end = v_min + (j + 1) * bin_size
            
            mask = (cloud_u >= bu_start) & (cloud_u < bu_end) & (cloud_v >= bv_start) & (cloud_v < bv_end)
            
            if np.sum(mask) < 6: # Need at least 6 points for a 2D quadratic fit
                local_fits[(i, j)] = None
                continue
                
            lu, lv, lr = cloud_u[mask], cloud_v[mask], cloud_rho[mask]
            
            # Construct least squares matrix for c0 + c1*u + c2*v + c3*u^2 + c4*uv + c5*v^2
            A = np.column_stack([np.ones_like(lu), lu, lv, lu**2, lu*lv, lv**2])
            coeffs, _, _, _ = np.linalg.lstsq(A, lr, rcond=None)
            local_fits[(i, j)] = coeffs

    # Global arrays for surface and its spatial derivatives
    g_val = np.zeros_like(u, dtype=float)
    g_u, g_v = np.zeros_like(u, dtype=float), np.zeros_like(u, dtype=float)
    g_uu, g_vv, g_uv = np.zeros_like(u, dtype=float), np.zeros_like(u, dtype=float), np.zeros_like(u, dtype=float)

    # Blend local fits using 2D bumps
    for ju in range(-deg, num_bins_u):
        for jv in range(-deg, num_bins_v):
            idx_u = max(0, min(ju + (deg // 2), num_bins_u - 1))
            idx_v = max(0, min(jv + (deg // 2), num_bins_v - 1))
            
            coeffs = local_fits.get((idx_u, idx_v))
            if coeffs is None:
                continue
                
            c0, c1, c2, c3, c4, c5 = coeffs
            
            # Evaluate the local polynomial and its analytical spatial derivatives
            P_val = c0 + c1*u + c2*v + c3*u**2 + c4*u*v + c5*v**2
            P_u = c1 + 2*c3*u + c4*v
            P_v = c2 + c4*u + 2*c5*v
            P_uu = 2*c3 * np.ones_like(u)
            P_vv = 2*c5 * np.ones_like(u)
            P_uv = c4 * np.ones_like(u)
            
            B_val, B_u, B_v, B_uu, B_vv, B_uv = bump_2d(u, v, bin_size, bin_size, u_min, v_min, ju, jv, deg)
            
            # Product rule for blending
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

def total_energy(T, curves, deformation_cloud, f=50.0):
    """
    Evaluates exact kinematics and algebraic chain rule across all curves,
    then sums the energy using the original projective bending functional.
    """
    total_E = 0.0
    t = np.array(T)
    
    for curve in curves:
        # Extract kinematics from the 2D curve trace
        x, y = curve.point_at(t)
        vx, vy = curve.velocity_at(t)
        ax, ay = curve.acceleration_at(t)
        
        # 1. Map continuous x(t), y(t) to stereographic u(t), v(t)
        u, u_t, u_tt, v, v_t, v_tt, lam, lam_t, lam_tt = projective_kinematics(x, y, vx, vy, ax, ay, f)
        
        # 2. Evaluate rho surface and exact spatial derivatives at u(t), v(t)
        rho, rho_u, rho_v, rho_uu, rho_vv, rho_uv = surface_fit(u, v, deformation_cloud, deg=2, bin_size=0.5)
        
        # 3. Multivariable chain rule for temporal derivatives
        rho_t = rho_u * u_t + rho_v * v_t
        rho_tt = (rho_uu * u_t**2 + 2 * rho_uv * u_t * v_t + rho_vv * v_t**2) + rho_u * u_tt + rho_v * v_tt
        
        # 4. Construct depth function w(t) and its exact temporal derivatives
        w = rho * lam
        w_dt = rho_t * lam + rho * lam_t
        w_ddt = rho_tt * lam + 2 * rho_t * lam_t + rho * lam_tt
        
        # 5. Compute structural bending energy[cite: 4]
        energy = compute_projective_bending_energy(t, x, y, vx, vy, ax, ay, w, w_dt, w_ddt, f)
        total_E += energy
        
    return total_E
