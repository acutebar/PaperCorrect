import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

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

    # 1. Generate the circular kernels for radius r and R
    kernel_r2 = get_disc_kernel(3)
    kernel_r5 = get_disc_kernel(10)

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

def paper_clean_circles(img, r_in=2, r_out=5):
    size = img.shape
    paper_brightness = np.mean(img)
    # Initialize background
    new_img = np.full(size, paper_brightness, dtype=np.uint8)
    
    img_float = img.astype(np.float32)
    
    # 1. Create masks for inner and outer circles
    y_in, x_in = np.ogrid[-r_in:r_in+1, -r_in:r_in+1]
    mask_in = (x_in**2 + y_in**2 <= r_in**2).astype(np.float32)
    kernel_in = mask_in / np.sum(mask_in)
    
    y_out, x_out = np.ogrid[-r_out:r_out+1, -r_out:r_out+1]
    mask_out = (x_out**2 + y_out**2 <= r_out**2).astype(np.float32)
    kernel_out = mask_out / np.sum(mask_out)
    
    # 2. Compute means and standard deviations globally (instant via convolution)
    mean_in = cv.filter2D(img_float, -1, kernel_in)
    mean_out = cv.filter2D(img_float, -1, kernel_out)
    mean_out_sq = cv.filter2D(img_float**2, -1, kernel_out)
    
    var_out = np.maximum(mean_out_sq - mean_out**2, 0)
    std_out = np.sqrt(var_out)
    
    # 3. Get relative coordinates for painting the inner circle
    y_paint_idx, x_paint_idx = np.where(mask_in > 0)
    y_paint_idx -= r_in
    x_paint_idx -= r_in
    
    # Stride of 3 ensures circles of radius 2 overlap enough to cover the grid
    stride = 3 
    
    # 4. Grid sample and paint
    for r in range(r_in, size[0], stride):
        for c in range(r_in, size[1], stride):
            val_in = mean_in[r, c]
            val_out = mean_out[r, c]
            val_std = std_out[r, c]
            
            if val_in < val_out - val_std:
                color = 0
            elif val_in > val_out + val_std:
                color = 255
            else:
                continue # Remains paper_brightness
                
            # Shift paint mask to current center
            y_paint = y_paint_idx + r
            x_paint = x_paint_idx + c
            
            # Keep indices within image bounds
            valid = (y_paint >= 0) & (y_paint < size[0]) & (x_paint >= 0) & (x_paint < size[1])
            new_img[y_paint[valid], x_paint[valid]] = color
            
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

def line_direction(img, pivot, width, height, visited, step=5) -> Union[tuple[int],None]:
    """
    The logic: this is like taking a start point and rotating a vector fixed at this start point. 
    If the vector lies on the line and it does not point in a direction that is already covered,
    that particular direction is returned.

    This assumes that the mean pixel along the vector is minimum when the vector lies along the
    (dark) line. If the point lies in the middle of a line, there are technically two maximums
    but only one is returned to force the vector to move in a certain direction.
    """
    local_color, total_std = mean_disc(pivot, height)
    mean_dict = {}
    for angle in range(0, 360, step):
        tilt = angle * math.pi/180
        mean, std = mean_rect(img, width, height, pivot, angle * math.pi/180)
        new_corner = (pivot[0] + height * math.cos(tilt), pivot[1] - height * math.sin(tilt))
        if (new_corner not in visited) and (mean > local_color + total_std or mean < local_color - total_std):
            mean_dict[new_corner] = mean

    if mean_dict != {}:
        return min(mean_dict, key=mean_dict.get)
    else:
        return None

def num_of_paths(img, pivot, width, height, visited, step=5, local=30) -> int:
    """
    This function determines whether a point is isolated, is at the start of a line
    or in the middle of a line.

    It returns the number of local minimum points for the mean pixel value.
    Local is defined by a region swept by an angle of <local> = 30 degrees.
    This only looks at minimum points that are not already visited.
    """
    local_color, total_std = mean_disc(pivot, height)
    mean_dict = {}
    count = 0

    for angle in range(0, 360, step):
        tilt = angle * math.pi/180
        mean, std = mean_rect(img, width, height, pivot, angle * math.pi/180)
        new_corner = (pivot[0] + height * math.cos(tilt), pivot[1] - height * math.sin(tilt))
        if (new_corner not in visited) and (mean > local_color + total_std or mean < local_color - total_std):
            mean_dict[angle] = (mean, new_corner)

    for angle_range in range(0, 360, local):
        for angle in mean_dict:
            if (angle >= angle_range) and (angle < angle_range + local):
                count += 1

    return count

    if mean_dict != {}:
        return max(mean_dict, key=mean_dict.get)
    else:
        return None



def line_detect(img, start, width, height, visited) -> Union[list[tuple[int]], None]:
    """
    Detects full lines from a starting point.

    Algorithm: if the starting point is not part of a line or in the middle of a line,
    do nothing. 

    If the starting point is the very start (or end) of a line, find the next point along 
    the line and move to that with the current point marked as visited so that the next point
    is treated as the start of a new line.

    Ends when the line ends or intersects another line (which results in more than one way to move)
    """
    ret_lst = []
    num_paths = num_of_paths(img, start, width, height, visited)

    if num_paths != 1:
        return None

    ret_lst.append(start)
    visited.add(start)
    next_move = line_direction(img, start, width, height, visited)

    if next_move not in visited:
        ret_lst = ret_lst + line_detect(img, next_move, width, height, visited)

    return ret_lst
    

def findall_lines(img) -> Union[list[list[tuple[int]]], None]:
    lines = []
    for x in img:
        line = line_detect(img, x, 2, 5, [])
        if line != None:
            lines.append(line)
    return lines

def connecting_pixels(pixel1, pixel2, thickness):
    # returns a mask
    # yet to implement
    pass

def display_lines(img):
    paper_brightness = 255
    new_img = np.full(img.shape, paper_brightness, dtype=np.uint8)
    lines = findall_lines(img)

    for line in lines:
        for i in range(len(lines)):
            new_img[line[i]] = 0
            if i < len(lines) -1 :
                connectors = connecting_pixels(line[i], line[i+1], 3)
                new_img[connectors] = 0
    
    return new_img



img = cv.imread("curvedsurface1.jpeg")
if img is not None:
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # 3. Display the first image
    ax1.imshow(img)
    ax1.set_title('First Image')
    ax1.axis('off') # Hides the pixel coordinate axes
    
    # 4. Display the second image
    new_img = display_lines(img)
    ax2.imshow(new_img)
    ax2.set_title('Second Image')
    ax2.axis('off')
    
    # 5. Render the plot
    plt.tight_layout()
    plt.show()
