import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def paper_clean(img):
    print("Cleaning paper")
    size = img.shape
    paper_brightness = np.mean(img)
    new_img = np.full(size, paper_brightness)

    print("Initiated background")

    for index, val in np.ndenumerate(gray_img):
        pixel_brightness = mean_disc(img, index, 2)[0]
        local_brightness, local_std = mean_disc(img, index, 5)
        print("Computed local brightness at", index)
        if (pixel_brightness < local_brightness - local_std):
            new_img[index] = 0
        elif (pixel_brightness > local_brightness + local_std):
            new_img[index] = 255

    return new_img

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

    # 1. Generate the circular kernels for radius 2 and 5
    kernel_r2 = get_disc_kernel(3)
    kernel_r5 = get_disc_kernel(8)

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
    new_img = np.full(img.shape, paper_brightness, dtype=np.uint8)

    # 6. Apply your exact conditional logic vector-wide using boolean masks
    dark_mask = pixel_brightness < (local_brightness - local_std)
    bright_mask = pixel_brightness > (local_brightness + local_std)

    new_img[dark_mask] = 0
    new_img[bright_mask] = 255

    return new_img

def mean_disc(mat, center, radius):
    n, m = mat.shape
    r_c, c_c = center

    r_min = max(0, int(np.floor(r_c - radius)))
    r_max = min(n, int(np.ceil(r_c + radius)) + 1)
    c_min = max(0, int(np.floor(c_c - radius)))
    c_max = min(m, int(np.ceil(c_c + radius)) + 1)

    r_grid, c_grid = np.ogrid[r_min:r_max, c_min:c_max]

    disc = [
        [(r - r_c) ** 2 + (c - c_c) ** 2 <= radius**2 for c in range(c_min, c_max)]
        for r in range(r_min, r_max)
    ]
    disc_values = mat[r_min:r_max, c_min:c_max][disc]

    if disc_values.size == 0:
        return 0

    return (float(np.mean(disc_values)), float(np.std(disc_values)))

def mean_rect(mat, width, height, corner, tilt):
    n, m = mat.shape
    r0, c0 = corner

    cos_t = math.cos(tilt)
    sin_t = math.sin(tilt)

    r_corners = [
        r0,
        r0 + width * sin_t,
        r0 + height * cos_t,
        r0 + width * sin_t + height * cos_t,
    ]
    c_corners = [
        c0,
        c0 + width * cos_t,
        c0 - height * sin_t,
        c0 + width * cos_t - height * sin_t,
    ]

    r_min = max(0, int(math.floor(min(r_corners))))
    r_max = min(n, int(math.ceil(max(r_corners))) + 1)
    c_min = max(0, int(math.floor(min(c_corners))))
    c_max = min(m, int(math.ceil(max(c_corners))) + 1)

    rect_values = []
    for r in range(r_min, r_max):
        dy = r - r0

        row_w_proj = dy * sin_t
        row_h_proj = dy * cos_t

        for c in range(c_min, c_max):
            dx = c - c0

            w_proj = dx * cos_t + row_w_proj
            h_proj = -dx * sin_t + row_h_proj

            # Check if the cell center lies within the width and height boundaries
            if 0 <= w_proj <= width and 0 <= h_proj <= height:
                rect_values.append(mat[r, c])

    if not rect_values:
        return 0.0

    return (float(np.mean(rect_values)), float(np.std(rect_values)))



img = cv.imread("curvedsurface1.jpeg")
if img is not None:
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #print(gray_img)
    #np.savetxt("pixels.txt", gray_img, fmt="%d")
    plt.imshow(gray_img, cmap='gray')
    plt.show()

    new_img = paper_clean_fast(gray_img)


    plt.imshow(new_img, cmap='gray')
    plt.show()
