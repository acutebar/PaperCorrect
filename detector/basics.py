import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math


class LineDetector:
    def __init__(self, width=2, height=5, step=5, sweep=30):
        self.width = width
        self.height = height
        self.step = step
        self.sweep = sweep
        self.angles = np.arange(0, 360, step)
        
        self.templates = {}
        self.tips = {}
        
        self.max_radius = int(np.ceil(height)) + 2
        center = (self.max_radius, self.max_radius)
        
        for angle in self.angles:
            tilt = math.radians(angle)
            tip_r = height * math.cos(tilt)
            tip_c = -height * math.sin(tilt)
            
            mask = np.zeros((2 * self.max_radius + 1, 2 * self.max_radius + 1), dtype=np.uint8)
            cv.line(mask, center, 
                     (int(round(center[1] + tip_c)), int(round(center[0] + tip_r))), 
                     255, thickness=width)
            
            dy, dx = np.where(mask > 0)
            self.templates[angle] = (dy - center[0], dx - center[1])
            self.tips[angle] = (int(round(tip_r)), int(round(tip_c)))

    def get_valid_paths(self, padded_img, pivot, visited_mask, thresh_map):
        pr, pc = pivot
        threshold = thresh_map[pr, pc]
        
        means = []
        for angle in self.angles:
            dy, dx = self.templates[angle]
            mean = np.mean(padded_img[pr + dy, pc + dx])
            
            tr, tc = self.tips[angle]
            tip_r, tip_c = pr + tr, pc + tc
            
            is_visited = visited_mask[tip_r, tip_c]
            means.append((mean, angle, tip_r, tip_c, is_visited))
            
        paths = []
        n = len(means)
        neighbor_count = max(1, self.sweep // self.step)
        
        for i in range(n):
            curr_m = means[i][0]
            is_valley = True
            
            for j in range(1, neighbor_count + 1):
                prev_m = means[(i - j) % n][0]
                next_m = means[(i + j) % n][0]
                
                if curr_m >= prev_m or curr_m > next_m:
                    is_valley = False
                    break
                    
            if is_valley and curr_m < threshold and not means[i][4]:
                paths.append(means[i])
                
        return paths

    def mark_visited(self, visited_mask, p1, p2):
        pt1 = (int(p1[1]), int(p1[0]))
        pt2 = (int(p2[1]), int(p2[0]))
        cv.line(visited_mask, pt1, pt2, 1, thickness=self.width + 2)

    def line_detect(self, padded_img, start, visited_mask, thresh_map):
        paths = self.get_valid_paths(padded_img, start, visited_mask, thresh_map)
        if len(paths) != 1:
            return None
            
        line = [start]
        cv.circle(visited_mask, (int(start[1]), int(start[0])), self.width + 1, 1, -1)
        curr = start
        
        while True:
            paths = self.get_valid_paths(padded_img, curr, visited_mask, thresh_map)
            if len(paths) == 0:
                break 
                
            best_path = min(paths, key=lambda x: x[0])
            next_move = (best_path[2], best_path[3])
            
            line.append(next_move)
            self.mark_visited(visited_mask, curr, next_move)
            curr = next_move
            
        return line

    def findall_lines(self, img):
        kernel = get_disc_kernel(self.height)
        img_float = img.astype(np.float32)
        
        # Precompute local means and stds globally
        local_mean = cv.filter2D(img_float, -1, kernel, borderType=cv.BORDER_REFLECT)
        local_mean_sq = cv.filter2D(img_float**2, -1, kernel, borderType=cv.BORDER_REFLECT)
        local_var = np.maximum(0, local_mean_sq - (local_mean**2))
        thresh_map = local_mean - (np.sqrt(local_var) * 0.5)

        # Pad variables to prevent bounds checking
        pad = self.max_radius
        padded_img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=255)
        padded_thresh = cv.copyMakeBorder(thresh_map, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)
        
        # Visited mask defaults to 1 (visited) outside the true image bounds
        visited_mask = np.ones(padded_img.shape, dtype=np.uint8)
        visited_mask[pad:-pad, pad:-pad] = 0
        
        lines = []
        dark_pixels = np.argwhere(img == 0)
        
        for r, c in dark_pixels:
            pr, pc = r + pad, c + pad
            if not visited_mask[pr, pc]:
                line_padded = self.line_detect(padded_img, (pr, pc), visited_mask, padded_thresh)
                if line_padded is not None and len(line_padded) > 1:
                    line = [(pr_val - pad, pc_val - pad) for (pr_val, pc_val) in line_padded]
                    lines.append(line)
                    
        return lines


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

# Energy minimizing algorithm

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
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    energy = trapz_fn(integrand, t)
    
    return float(np.asarray(energy).item() if np.ndim(energy) > 0 else energy)

# Fast AI generated version for testing
def get_disc_kernel(radius):
    """Generates a normalized circular 2D kernel for convolution."""
    # Create a grid from -radius to +radius
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(np.float32)
    # Normalize so the sum of all weights equals 1.0 (calculates the mean)
    return kernel / np.sum(kernel)

def paper_clean_fast(img):
    print("Cleaning paper...")
    # Convert to float32 to prevent overflow during squaring and convolution
    img_float = img.astype(np.float32)

    # 1. Generate the circular kernels for radius r and R
    kernel_r2 = get_disc_kernel(3)
    kernel_r5 = get_disc_kernel(30)

    print("Computing global and local statistics simultaneously...")
    # 2. Compute pixel_brightness (radius 2 mean) across the entire image at once
    pixel_brightness = cv.filter2D(
        img_float, -1, kernel_r2, borderType=cv.BORDER_REFLECT
    )

    # 3. Compute local_brightness (radius 5 mean) across the entire image at once
    local_brightness = cv.filter2D(
        img_float, -1, kernel_r5, borderType=cv.BORDER_REFLECT
    )

    # 4. Compute local standard deviation mathematically: sqrt(E[X^2] - E[X]^2)
    img_sq = img_float**2
    local_mean_sq = cv.filter2D(
        img_sq, -1, kernel_r5, borderType=cv.BORDER_REFLECT
    )

    # np.maximum(0, ...) prevents tiny negative numbers from floating-point inaccuracies
    local_variance = np.maximum(0, local_mean_sq - (local_brightness**2))
    local_std = np.sqrt(local_variance)

    print("Applying adaptive threshold...")
    # 5. Initialize background with overall image mean
    paper_brightness = np.mean(img_float)
    new_img = np.full(img.shape, 255, dtype=np.uint8)

    # 6. Apply your exact conditional logic vector-wide using boolean masks
    dark_mask = pixel_brightness < (local_brightness - local_std)
    bright_mask = pixel_brightness > (local_brightness + local_std)

    new_img[dark_mask] = 0
    new_img[bright_mask] = 255

    return new_img

def display_lines(img, lines, thickness=3):
    """
    Reconstructs the lines. OpenCV's cv.line natively handles the 
    'connecting_pixels' logic you requested, rendering an anti-aliased line 
    between coordinate pairs instantly.
    """
    new_img = np.full(img.shape, 255, dtype=np.uint8)
    
    for line in lines:
        for i in range(len(line) - 1):
            # OpenCV points are (x, y), which maps to (col, row)
            p1 = (int(line[i][1]), int(line[i][0]))
            p2 = (int(line[i+1][1]), int(line[i+1][0]))
            
            cv.line(new_img, p1, p2, 0, thickness=thickness)
            
    return new_img
