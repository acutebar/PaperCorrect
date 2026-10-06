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
    # Second derivatives of a quadratic are constant, so they stay scalars
    return w, w_x, w_y, 2*c3, 2*c5, c4

def energy_precompute(T, curves, f=1.0):
    t = torch.as_tensor(T)
    bumps = max(curves, key=lambda c: c.num_bins).get_bumps(t) if curves else None

    xs = []
    ys = []
    vxs = []
    vys = []
    axs = []
    ays = []

    for curve in curves:
        res = curve.curve_at(t, bumps)
        x, y = res[0][0], res[0][1]
        vx, vy = res[1][0], res[1][1]
        ax, ay = res[2][0], res[2][1]


        x = torch.as_tensor(x, dtype=torch.float64).ravel()
        y = torch.as_tensor(y, dtype=torch.float64).ravel()
        vx = torch.as_tensor(vx, dtype=torch.float64).ravel()
        vy = torch.as_tensor(vy, dtype=torch.float64).ravel()
        ax = torch.as_tensor(ax, dtype=torch.float64).ravel()
        ay = torch.as_tensor(ay, dtype=torch.float64).ravel()

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        axs.append(ax)
        ays.append(ay)

    X = torch.stack(xs)
    Y = torch.stack(ys)
    VX = torch.stack(vxs)
    VY = torch.stack(vys)
    AX = torch.stack(axs)
    AY = torch.stack(ays)

    curve_properties = (X, Y, VX, VY, AX, AY)
    return curve_properties

def total_energy(T, curve_properties, cloud_values, pinvX, cu, cv, f=1.0):
    t = torch.as_tensor(T)

    coeffs = pinvX @ cloud_values

    x, y, vx, vy, ax, ay = curve_properties
    w, w_x, w_y, w_xx, w_yy, w_xy = surface_fit(x-cu, y-cv, coeffs)

    # Surface normal gamma_x x gamma_y, with gamma_x = (w_x x + w, w_x y, w_x f)
    # and gamma_y = (w_y x, w_y y + w, w_y f), expands to w * (-f w_x, -f w_y, w_x x + w_y y + w)
    N1 = -f * w * w_x
    N2 = -f * w * w_y
    N3 = w * (w_x * x + w_y * y + w)

    w_t = w_x * vx + w_y * vy
    w_tt = (w_xx * vx**2 + 2 * w_xy * vx * vy + w_yy * vy**2) + w_x * ax + w_y * ay

    # gamma_t = w_t P + w P_t and gamma_tt = w_tt P + 2 w_t P_t + w P_tt, with P = (x, y, f)
    g1 = w_t * x + w * vx
    g2 = w_t * y + w * vy
    g3 = w_t * f
    h1 = w_tt * x + 2 * w_t * vx + w * ax
    h2 = w_tt * y + 2 * w_t * vy + w * ay
    h3 = w_tt * f

    # a_T = (gamma_t x gamma_tt) . N / |N|
    a_T = ((g2*h3 - g3*h2) * N1 + (g3*h1 - g1*h3) * N2 + (g1*h2 - g2*h1) * N3) / (torch.sqrt(N1**2 + N2**2 + N3**2) + 1e-12)

    speed_sq = g1**2 + g2**2 + g3**2
    speed_sq_clamped = torch.clamp(speed_sq, min=1e-12)

    integrand = (a_T**2) / (speed_sq_clamped**2.5)
    trim = max(1, len(t) // 20) if len(t) > 10 else 0
    #trim=0
    if trim > 0:
        energy = torch.trapezoid(integrand[:, trim:-trim], t[trim:-trim], dim=1)
    else:
        energy = torch.trapezoid(integrand, t, dim=1)

    total_E = energy.sum()

    scale = cloud_values.mean()
    adjusted_E = scale * total_E
    print(f"\tEnergy breakdown: Geodesic = {total_E} | Scale = {scale} | adjusted = {adjusted_E}")

    return (total_E)
