import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math
from .fnfit import bump, step_function, Curve
import torch

def quadratic_fit(cloud_coord):
    raw_u, raw_v = cloud_coord[:, 0], cloud_coord[:, 1]
    cu = raw_u.mean()
    cv = raw_v.mean()
    u = raw_u - cu
    v = raw_v - cv

    A = torch.column_stack([torch.ones_like(u), u, v, u**2, u*v, v**2])
    return torch.linalg.pinv(A), (cu, cv)

def surface_fit(x, y, coeffs):
    """Global quadratic w and its derivatives. x, y must be centered (x - cu, y - cv)."""
    c0, c1, c2, c3, c4, c5 = coeffs
    w    = c0 + c1*x + c2*y + c3*x**2 + c4*x*y + c5*y**2
    w_x  = c1 + 2*c3*x + c4*y
    w_y  = c2 + c4*x + 2*c5*y
    w_xx = torch.full_like(x, 1.0) * (2*c3)
    w_yy = torch.full_like(x, 1.0) * (2*c5)
    w_xy = torch.full_like(x, 1.0) * c4
    return w, w_x, w_y, w_xx, w_yy, w_xy

def total_energy(T, curves, cloud_values, pinvX, cu, cv, f=1.0):
    t = torch.as_tensor(T)

    coeffs = pinvX @ cloud_values
    total_E = coeffs.new_zeros(())

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

        w, w_x, w_y, w_xx, w_yy, w_xy = surface_fit(x-cu, y-cv, coeffs)
        
        gamma_x = torch.stack([w_x * x + w, w_x * y, w_x * f_tensor], dim=1)
        gamma_y = torch.stack([w_y * x, w_y * y + w, w_y * f_tensor], dim=1)

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
        trim = max(1, len(t) // 20) if len(t) > 10 else 0
        #trim=0
        if trim > 0:
            energy = torch.trapezoid(integrand[trim:-trim], t[trim:-trim])
        else:
            energy = torch.trapezoid(integrand, t)

        total_E = total_E + energy

    scale = cloud_values.mean()
    adjusted_E = scale * total_E
    print(f"\tEnergy breakdown: Geodesic = {total_E} | Scale = {scale} | adjusted = {adjusted_E}")

    return (total_E)
