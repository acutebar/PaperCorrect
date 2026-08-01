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
        
        max_radius = int(np.ceil(height)) + 2
        center = (max_radius, max_radius)
        
        for angle in self.angles:
            tilt = math.radians(angle)
            tip_r = height * math.cos(tilt)
            tip_c = -height * math.sin(tilt)
            
            mask = np.zeros((2 * max_radius + 1, 2 * max_radius + 1), dtype=np.uint8)
            cv.line(mask, center, 
                     (int(round(center[1] + tip_c)), int(round(center[0] + tip_r))), 
                     255, thickness=width)
            
            dy, dx = np.where(mask > 0)
            self.templates[angle] = (dy - center[0], dx - center[1])
            self.tips[angle] = (tip_r, tip_c)

    def mean_rect(self, img, pivot, angle):
        dy, dx = self.templates[angle]
        
        r_idx = int(pivot[0]) + dy
        c_idx = int(pivot[1]) + dx
        
        valid = (r_idx >= 0) & (r_idx < img.shape[0]) & (c_idx >= 0) & (c_idx < img.shape[1])
        r_idx, c_idx = r_idx[valid], c_idx[valid]
        
        if len(r_idx) == 0:
            return 255.0, 0.0
            
        vals = img[r_idx, c_idx]
        return float(np.mean(vals)), float(np.std(vals))

    def mean_disc(self, img, pivot):
        r = self.height
        r0 = max(0, int(pivot[0] - r))
        r1 = min(img.shape[0], int(pivot[0] + r + 1))
        c0 = max(0, int(pivot[1] - r))
        c1 = min(img.shape[1], int(pivot[1] + r + 1))
        
        region = img[r0:r1, c0:c1]
        if region.size == 0:
            return 255.0, 0.0
        return float(np.mean(region)), float(np.std(region))

    def get_valid_paths(self, img, pivot, visited_mask):
        local_mean, local_std = self.mean_disc(img, pivot)
        threshold = local_mean - (local_std * 0.5) 
        
        means = []
        for angle in self.angles:
            mean, _ = self.mean_rect(img, pivot, angle)
            
            tip_r = int(pivot[0] + self.tips[angle][0])
            tip_c = int(pivot[1] + self.tips[angle][1])
            
            if 0 <= tip_r < img.shape[0] and 0 <= tip_c < img.shape[1]:
                is_visited = visited_mask[tip_r, tip_c]
            else:
                is_visited = True 
                
            means.append((mean, angle, tip_r, tip_c, is_visited))
            
        paths = []
        n = len(means)
        
        # Calculate how many adjacent angle steps make up the sweep window
        neighbor_count = max(1, self.sweep // self.step)
        
        for i in range(n):
            curr_m = means[i][0]
            is_valley = True
            
            for j in range(1, neighbor_count + 1):
                prev_m = means[(i - j) % n][0]
                next_m = means[(i + j) % n][0]
                
                # Asymmetric check to handle flat plateaus gracefully.
                # If three adjacent angles all return exactly 0.0, this ensures 
                # only the first one is marked as the valley, preventing duplicates.
                if curr_m >= prev_m or curr_m > next_m:
                    is_valley = False
                    break
                    
            if is_valley and curr_m < threshold and not means[i][4]:
                paths.append(means[i])
                    
        return paths

    def mark_visited(self, visited_mask, p1, p2):
        """Draws a thick line on the visited mask to consume all pixels along the path."""
        pt1 = (int(p1[1]), int(p1[0]))
        pt2 = (int(p2[1]), int(p2[0]))
        # Use a thickness slightly larger than your line width to ensure all stray edge pixels are consumed
        cv.line(visited_mask, pt1, pt2, 1, thickness=self.width + 2)

    def line_detect(self, img, start, visited_mask):
        paths = self.get_valid_paths(img, start, visited_mask)
        if len(paths) != 1:
            return None
            
        line = [start]
        cv.circle(visited_mask, (int(start[1]), int(start[0])), self.width + 1, 1, -1)
        curr = start
        
        while True:
            paths = self.get_valid_paths(img, curr, visited_mask)
            if len(paths) == 0:
                break 
                
            best_path = min(paths, key=lambda x: x[0])
            next_move = (best_path[2], best_path[3])
            
            line.append(next_move)
            self.mark_visited(visited_mask, curr, next_move)
            curr = next_move
            
        return line

    def findall_lines(self, img):
        lines = []
        visited_mask = np.zeros(img.shape, dtype=np.uint8)
        
        dark_pixels = np.argwhere(img == 0) 
        
        for r, c in dark_pixels:
            if not visited_mask[r, c]:
                line = self.line_detect(img, (r, c), visited_mask)
                if line is not None and len(line) > 1:
                    lines.append(line)
                    
        return lines


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

#img = cv.imread("curve.jpeg")
#if img is not None:
#    #gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)[1500:2100, 400:1700]
#    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)[500:1000, 400:1700]
#    # 1. Apply a slight blur first to smooth out the paper texture
#    blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)
#    
#    # 2. Apply Gaussian Adaptive Thresholding
#    # - 255: Max pixel value (white)
#    # - ADAPTIVE_THRESH_GAUSSIAN_C: Uses a Gaussian weighted mean for the local area
#    # - THRESH_BINARY: Outputs black ink (0) on white paper (255)
#    # - 31: Block size (pixel neighborhood to look at). Must be odd.
#    # - 15: Constant 'C' subtracted from the mean to fine-tune the threshold
#    cleaned_img = cv.adaptiveThreshold(
#        blurred_img, 
#        255, 
#        cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
#        cv.THRESH_BINARY, 
#        31, 
#        15
#    )
#    my_cleaned_img = paper_clean_fast(gray_img)
#
#    detector = LineDetector(width=2, height=10, step=5) 
#    lines = detector.findall_lines(cleaned_img)
#
#    new_img = display_lines(cleaned_img, lines, thickness=2)
#
#    images = [gray_img, cleaned_img, my_cleaned_img, new_img]
#    titles = ["Original", "Gaussian clean", "Cleaning with my algorithm", "Line detection with Gaussian clean"]
#
#    figs, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
#    axes = axes.flatten()
#    
#    # 4. Loop through images and axes simultaneously
#    for i, ax in enumerate(axes):
#        ax.imshow(images[i], cmap='gray')          # Display the image
#        ax.set_title(titles[i])       # Set individual titles
#        ax.axis('off')                # Hide the X and Y pixel ticks
#    
#    # 5. Render the plot
#    plt.tight_layout()
#    plt.show()

