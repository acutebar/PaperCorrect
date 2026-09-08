import detector
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("antiprinter.jpeg")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

x_lim,y_lim = gray_img.shape

blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)
cleaned_img = cv.adaptiveThreshold(
    blurred_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 15
)

plt.imshow(cleaned_img, cmap='gray')
plt.show()



print("\nRunning line detection (this may take a moment)...")
detector = detector.LineDetector(width=2, height=10, step=5) 
lines = detector.findall_lines(cleaned_img)
lines.sort(key = lambda x: len(x))

line = lines[-4]
#print(line)

all_lines_img = detector.display_lines(cleaned_img, [line], thickness=2)
plt.imshow(all_lines_img, cmap='gray')
plt.show()



T = np.linspace(0, 1, 1000)
(x, y), (vx, vy), (ax, ay) = detector.curve_fit(T, line, bin_size=0.13, deg=2)

# Energy stuff
img_height, img_width = gray_img.shape
true_center_x = img_width / 2.0
true_center_y = img_height / 2.0

x_c = x - true_center_x
y_c = y - true_center_y

# 2. Compute depth base 'u'
f = 2912.0
u = x_c**2 + y_c**2 + f**2

# 3. Compute w, w_dt, w_ddt for rho = 1
A = x_c * vx + y_c * vy
B = vx**2 + vy**2 + x_c * ax + y_c * ay

w = u**(-0.5)
w_dt = -(u**(-1.5)) * A
w_ddt = 3 * (u**(-2.5)) * (A**2) - (u**(-1.5)) * B
f=2912.0
#energy = detector.compute_projective_bending_energy(T, x, y, vx, vy, ax, ay, w, w_dt, w_ddt, f)
#print(energy)
#print(x)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
#axes[0].imshow(cleaned_img, cmap='gray')
axes[1].imshow(cleaned_img, cmap='gray')
axes[0].plot(T, y, color='red', linewidth=2)
axes[0].plot(T, x, color='green', linewidth=2)
#axes[1].plot(T, y, color='red', linewidth=2)
#axes[1].plot(T, x, color='green', linewidth=2)
#plt.xlim(0, y_lim)
#plt.ylim(0, x_lim)




def bump_function(x, center, radius):
    """Standard C-infinity bump function supported on (center-radius, center+radius)."""
    t = (x - center) / radius
    # Use np.where to avoid division by zero or log of negative warnings
    mask = np.abs(t) < 1.0
    result = np.zeros_like(t)
    result[mask] = np.exp(-1.0 / (1.0 - t[mask]**2))
    return result

def partition_of_unity_fit(x_data, y_data, x_eval, num_windows=10, poly_degree=3, overlap=1.5):
    """
    Fits overlapping local polynomials and blends them with a partition of unity.
    
    overlap: Determines how much the bump functions overlap (must be > 1.0 for full coverage)
    """
    x_min, x_max = np.min(x_data), np.max(x_data)
    
    # Define centers of the windows
    centers = np.linspace(x_min, x_max, num_windows)
    spacing = centers[1] - centers[0]
    radius = spacing * overlap
    
    global_y = np.zeros_like(x_eval)
    weight_sum = np.zeros_like(x_eval)
    
    for c in centers:
        # 1. Isolate data within the support of this window
        mask = np.abs(x_data - c) < radius
        if np.sum(mask) < poly_degree + 1:
            continue # Not enough points to fit the polynomial here
            
        x_local = x_data[mask]
        y_local = y_data[mask]
        
        # 2. Fit local polynomial
        coeffs = np.polyfit(x_local, y_local, poly_degree)
        poly = np.poly1d(coeffs)
        
        # 3. Evaluate local polynomial on the target evaluation grid
        y_poly_eval = poly(x_eval)
        
        # 4. Evaluate bump function for this window
        weights = bump_function(x_eval, c, radius)
        
        # 5. Add to global blend
        global_y += weights * y_poly_eval
        weight_sum += weights
        
    # Normalize by the sum of weights (this makes it a true Partition of Unity)
    # Add a tiny epsilon to prevent division by zero outside the strict domain
    global_y = global_y / (weight_sum + 1e-12)
    
    return global_y

# --- Example Usage ---
# Generate noisy data
x_raw = np.array([coord[0] for coord in line])
y_raw = np.array([coord[1] for coord in line])


distances = np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2)
t_raw = np.insert(np.cumsum(distances), 0, 0)
t_raw = t_raw / t_raw[-1]  # Normalized [0, 1]

# 2. Define a fine evaluation grid for t
t_dense = np.linspace(0, 1.0, 1000)

# 3. RUN IT TWICE: Treat x and y as independent functions of t!
x_smooth = partition_of_unity_fit(t_raw, x_raw, t_dense, num_windows=10, poly_degree=2)
y_smooth = partition_of_unity_fit(t_raw, y_raw, t_dense, num_windows=10, poly_degree=2)

# Evaluate on a fine grid
# Plot
axes[1].plot(y, x, color='red', linewidth=2)
plt.show()

