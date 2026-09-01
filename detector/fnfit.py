import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math

class Curve:
    def __init__(self, cloud):
        self.cloud = cloud

    def point_at(self, t):
        return curve_fit(np.array([t]), self.cloud)[0]

    def velocity_at(self, t):
        return curve_fit(np.array([t]), self.cloud)[1]

    def acceleration(self, t):
        return curve_fit(np.array([t]), self.cloud)[2]

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
    # Assuming cloud is a NumPy array, slice directly instead of list comprehension
    #print("Fitting global function. Received cloud: ", cloud)
    cloudx = cloud[:, 0]
    cloudy = cloud[:, 1]
    x_max = np.max(cloudx)
    x_min = np.min(cloudx)
    #print("Minimum parameter is", x_min)
    
    num_bins = int((x_max - x_min) // bin_size) + 1
    local_fits = {}

    for i in range(num_bins):
        bin_start = x_min + i * bin_size
        bin_end = x_min + (i + 1) * bin_size
        
        # Vectorized mask
        mask = (cloudx >= bin_start) & (cloudx < bin_end)
        
        if np.sum(mask) < deg + 1:
            local_fits[i] = None
            continue
            
        local_cloudx = cloudx[mask]
        local_cloudy = cloudy[mask]
        
        local_poly = np.polynomial.Polynomial.fit(local_cloudx, local_cloudy, deg=deg)
        local_fits[i] = local_poly

    # Initialize zero arrays matching the size of x
    global_value = np.zeros_like(x, dtype=float)
    global_d1 = np.zeros_like(x, dtype=float)
    global_d2 = np.zeros_like(x, dtype=float)
    
    for j in range(-deg, num_bins):
        i = max(0, min(j + (deg // 2), num_bins - 1))
        if local_fits[i] is None:
            print("WHAT KIND OF ERROR IS THIS")
            continue

        P_x = local_fits[i](x)
        P_dx = local_fits[i].deriv(1)(x)
        P_ddx = local_fits[i].deriv(2)(x)
        
        B_x, B_dx, B_ddx = bump(x, bin_size, x_min, j, deg)
        
        global_value += B_x * P_x
        global_d1 += (B_dx * P_x) + (B_x * P_dx)
        global_d2 += (B_ddx * P_x) + (2 * B_dx * P_dx) + (B_x * P_ddx)

    print(f"Computed global value as {global_value[0]}")
    return global_value, global_d1, global_d2

def curve_fit(t_eval, cloud, deg=2, bin_size=0.1):
    # 1. REMOVED the argsort line assuming your cloud is sequentially traced
    cloud = np.array(cloud)
    print(cloud)
    cloudx = cloud[:, 0]
    cloudy = cloud[:, 1]
    
    cloudx_shift = np.roll(cloudx, 1)
    cloudy_shift = np.roll(cloudy, 1)
    cloudx_shift[0] = cloudx[0]
    cloudy_shift[0] = cloudy[0]

    increments = np.sqrt(np.square(cloudx - cloudx_shift) + np.square(cloudy - cloudy_shift))
    times = np.cumsum(increments)
    
    # 2. ADDED Normalization: scale the timeline to exactly [0.0, 1.0]
    # (Add a tiny epsilon to prevent division by zero just in case)
    times = times / (times[-1] + 1e-12)
    
    # Pack the 1D arrays into 2D clouds for fn_fit
    cloud_t_x = np.column_stack((times, cloudx))
    cloud_t_y = np.column_stack((times, cloudy))

    xt, xt_dt, xt_ddt = fn_fit(t_eval, cloud_t_x, deg, bin_size)
    yt, yt_dt, yt_ddt = fn_fit(t_eval, cloud_t_y, deg, bin_size)

    return ((xt, yt), (xt_dt, yt_dt), (xt_ddt, yt_ddt))
