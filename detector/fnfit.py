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
        self.res = None
        self.bumps = None

        if len(self.cloud) == 0:
            self.times = np.empty(0)
            self.qmat = None
            self.quadratics = None
            self.num_bins = max(1, int(np.ceil(1.0 / self.bin_size)))
            return

        cloudx = self.cloud[:, 0]
        cloudy = self.cloud[:, 1]

        self.cloudx = cloudx
        self.cloudy = cloudy
        
        cloudx_shift = np.roll(cloudx, 1)
        cloudy_shift = np.roll(cloudy, 1)
        cloudx_shift[0] = cloudx[0]
        cloudy_shift[0] = cloudy[0]

        increments = np.sqrt(np.square(cloudx - cloudx_shift) + np.square(cloudy - cloudy_shift))
        self.times = np.cumsum(increments)

        total_len = self.times[-1]
        if total_len > 1e-12:
            self.times = self.times / total_len
        else:
            self.times = np.linspace(0.0, 1.0, len(self.cloud))

        self.qmat = None
        self.quadratic_mat()

        self.num_bins = max(1, int(np.ceil((self.times[-1]) / self.bin_size)))
        self.fit_quadratics()

    def quadratic_mat(self):
        t = self.times
        A = np.column_stack([np.ones_like(t), t, t**2])
        self.qmat = A

    def fit_quadratics(self):
        # Per-bin fits depend only on the cloud, so they are computed once here
        t_min = 0
        num_bins = self.num_bins

        # Row index where each bin starts; times are sorted, so each bin is a contiguous slice
        edges = np.searchsorted(self.times, t_min + self.bin_size * np.arange(num_bins + 1))
        edges[-1] = len(self.times)

        quadratics = [None] * num_bins
        for i in range(num_bins):
            left = edges[i]
            right = edges[i + 1]
            if right - left < self.deg + 1:
                continue

            local_mat = self.qmat[left:right]
            X = np.linalg.pinv(local_mat)
            wx = X @ self.cloudx[left:right]
            wy = X @ self.cloudy[left:right]
            quadratics[i] = (wx, wy)

        valid = [i for i in range(num_bins) if quadratics[i] is not None]
        if valid:
            for i in range(num_bins):
                if quadratics[i] is None:
                    quadratics[i] = quadratics[min(valid, key=lambda k: abs(k - i))]
        else:
            fit_deg = min(2, self.deg, len(self.times) - 1)
            X = np.zeros((3, len(self.times)))
            X[:fit_deg + 1] = np.linalg.pinv(self.qmat[:, :fit_deg + 1])
            quadratics = [(X @ self.cloudx, X @ self.cloudy)] * num_bins

        self.quadratics = quadratics

    def curve_at(self, t, bumps=None):
        t_arr = np.asarray(t, dtype=float)
        if len(self.cloud) == 0:
            z = np.zeros_like(t_arr)
            return ((z, z), (z, z), (z, z))

        if bumps is None:
            bumps = self.get_bumps(t_arr)

        num_bins = self.num_bins
        quadratics = self.quadratics

        xt, yt = np.zeros_like(t_arr), np.zeros_like(t_arr)
        x_dt, y_dt = np.zeros_like(t_arr), np.zeros_like(t_arr)
        x_ddt, y_ddt = np.zeros_like(t_arr), np.zeros_like(t_arr)

        for j in range(-self.deg, num_bins):
            wx, wy = quadratics[max(0, min(j + (self.deg // 2), num_bins - 1))]
            B, B_dt, B_ddt = bumps[j + self.deg]

            px = wx[0] + wx[1] * t_arr + wx[2] * t_arr**2
            py = wy[0] + wy[1] * t_arr + wy[2] * t_arr**2

            px_dt = wx[1] + 2*wx[2]*t_arr
            py_dt = wy[1] + 2*wy[2]*t_arr

            px_ddt = 2*wx[2]
            py_ddt = 2*wy[2]

            xt += B * px
            yt += B * py
            x_dt += (B_dt * px) + (B * px_dt)
            y_dt += (B_dt * py) + (B * py_dt)
            x_ddt += (B_ddt * px) + (2 * B_dt * px_dt) + (B * px_ddt)
            y_ddt += (B_ddt * py) + (2 * B_dt * py_dt) + (B * py_ddt)

        res = ((xt, yt), (x_dt, y_dt), (x_ddt, y_ddt))
        self.res = res
        return res

    def get_bumps(self, t):
        bumps = []
        t_arr = np.asarray(t, dtype=float)
        for j in range(-self.deg, self.num_bins):
            B, B_dt, B_ddt = bump(t_arr, self.bin_size, 0, j, deg=self.deg)
            bumps.append((B, B_dt, B_ddt))

        self.bumps = bumps
        return bumps

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
