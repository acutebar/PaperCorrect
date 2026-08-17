import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math

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

