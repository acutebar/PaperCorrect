import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math

class Curve:
    def __init__(self, cloud, deg=2, bin_size=0.1):
        self.cloud = np.asarray(cloud, dtype=float)
        self.deg = deg
        self.bin_size = bin_size

    def point_at(self, t):
        t_arr = np.asarray(t)
        res = curve_fit(t_arr, self.cloud, deg=self.deg, bin_size=self.bin_size)
        return res[0][0], res[0][1]

    def velocity_at(self, t):
        t_arr = np.asarray(t)
        res = curve_fit(t_arr, self.cloud, deg=self.deg, bin_size=self.bin_size)
        return res[1][0], res[1][1]

    def acceleration_at(self, t):
        t_arr = np.asarray(t)
        res = curve_fit(t_arr, self.cloud, deg=self.deg, bin_size=self.bin_size)
        return res[2][0], res[2][1]

# bump(variable, bin_size, starting_point, cur_bin, degree)
# https://personal.math.vt.edu/embree/math5466/lecture10.pdf
def step_function(x, start, end):
    return np.where((x >= start) & (x < end), 1.0, 0.0)

def bump(x, d, x0, j, deg=2):
    B_dict = {(j+i, 0): step_function(x, x0+(j+i)*d, x0+(j+i+1)*d) for i in range(-2*deg, deg+1)}
    
    k = 1
    while k <= deg:
        for i in range(-2*deg + k, deg+1-k):
            B_dict[(j+i, k)] = (x-(x0 + (j+i)*d))/(k*d)*B_dict[(j+i, k-1)] + ((x0 + (j+i)*d) + (k+1)*d - x)/(k*d) * B_dict[(j+i+1, k-1)]
        k += 1

    B_x = B_dict[(j, deg)]
    
    if deg >= 1:
        B_dx = (B_dict[(j, deg-1)] - B_dict[(j+1, deg-1)]) / d
    else:
        B_dx = np.zeros_like(x)
        
    if deg >= 2:
        B_ddx = (B_dict[(j, deg-2)] - 2*B_dict[(j+1, deg-2)] + B_dict[(j+2, deg-2)]) / (d**2)
    else:
        B_ddx = np.zeros_like(x)

    return B_x, B_dx, B_ddx

def fn_fit(x, cloud, deg=2, bin_size=10):
    cloud = np.asarray(cloud, dtype=float)
    x_eval = np.asarray(x, dtype=float)
    cloudx = cloud[:, 0]
    cloudy = cloud[:, 1]
    x_max = np.max(cloudx)
    x_min = np.min(cloudx)
    
    span = x_max - x_min
    if span < 1e-12:
        val = np.full_like(x_eval, cloudy[0] if len(cloudy) > 0 else 0.0)
        zeros = np.zeros_like(x_eval)
        return val, zeros, zeros
        
    num_bins = int(np.ceil(span / bin_size))
    if num_bins < 1:
        num_bins = 1
    local_fits = {}

    for i in range(num_bins):
        bin_start = x_min + i * bin_size
        bin_end = x_min + (i + 1) * bin_size
        
        mask = (cloudx >= bin_start) & (cloudx <= bin_end if i == num_bins - 1 else cloudx < bin_end)
        
        if np.sum(mask) < deg + 1:
            local_fits[i] = None
            continue
            
        local_cloudx = cloudx[mask]
        local_cloudy = cloudy[mask]
        
        local_poly = np.polynomial.Polynomial.fit(local_cloudx, local_cloudy, deg=deg)
        local_fits[i] = local_poly

    valid_fits = {k: v for k, v in local_fits.items() if v is not None}
    fit_deg = min(deg, max(0, len(cloudx) - 1))
    global_poly = np.polynomial.Polynomial.fit(cloudx, cloudy, deg=fit_deg)

    for i in range(num_bins):
        if local_fits[i] is None:
            if valid_fits:
                nearest_k = min(valid_fits.keys(), key=lambda k: abs(k - i))
                local_fits[i] = valid_fits[nearest_k]
            else:
                local_fits[i] = global_poly

    # Initialize zero arrays matching the size of x
    global_value = np.zeros_like(x_eval, dtype=float)
    global_d1 = np.zeros_like(x_eval, dtype=float)
    global_d2 = np.zeros_like(x_eval, dtype=float)
    
    for j in range(-deg, num_bins):
        i = max(0, min(j + (deg // 2), num_bins - 1))
        poly = local_fits[i]

        P_x = poly(x_eval)
        P_dx = poly.deriv(1)(x_eval)
        P_ddx = poly.deriv(2)(x_eval)
        
        B_x, B_dx, B_ddx = bump(x_eval, bin_size, x_min, j, deg)
        
        global_value += B_x * P_x
        global_d1 += (B_dx * P_x) + (B_x * P_dx)
        global_d2 += (B_ddx * P_x) + (2 * B_dx * P_dx) + (B_x * P_ddx)

    return global_value, global_d1, global_d2

def curve_fit(t_eval, cloud, deg=2, bin_size=0.1):
    cloud = np.asarray(cloud, dtype=float)
    if len(cloud) == 0:
        t_arr = np.asarray(t_eval)
        z = np.zeros_like(t_arr)
        return ((z, z), (z, z), (z, z))

    cloudx = cloud[:, 0]
    cloudy = cloud[:, 1]
    
    cloudx_shift = np.roll(cloudx, 1)
    cloudy_shift = np.roll(cloudy, 1)
    cloudx_shift[0] = cloudx[0]
    cloudy_shift[0] = cloudy[0]

    increments = np.sqrt(np.square(cloudx - cloudx_shift) + np.square(cloudy - cloudy_shift))
    times = np.cumsum(increments)
    
    total_len = times[-1]
    if total_len > 1e-12:
        times = times / total_len
    else:
        times = np.linspace(0.0, 1.0, len(cloud))
    
    # Pack the 1D arrays into 2D clouds for fn_fit
    cloud_t_x = np.column_stack((times, cloudx))
    cloud_t_y = np.column_stack((times, cloudy))

    xt, xt_dt, xt_ddt = fn_fit(t_eval, cloud_t_x, deg=deg, bin_size=bin_size)
    yt, yt_dt, yt_ddt = fn_fit(t_eval, cloud_t_y, deg=deg, bin_size=bin_size)

    return ((xt, yt), (xt_dt, yt_dt), (xt_ddt, yt_ddt))
