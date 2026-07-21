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

    new_img = paper_clean(gray_img)


    plt.imshow(new_img, cmap='gray')
    plt.show()
