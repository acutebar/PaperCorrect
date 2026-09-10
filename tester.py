import detector
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("curve.jpeg")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

x_lim,y_lim = gray_img.shape

blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)
cleaned_img = cv.adaptiveThreshold(
    blurred_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 15
)

plt.imshow(gray_img, cmap='gray')
plt.savefig("gray_img.png", dpi=300, bbox_inches='tight')
plt.imshow(blurred_img, cmap='gray')
#plt.savefig("gaussian_blur.png", dpi=300, bbox_inches='tight')
plt.show()

plt.imshow(cleaned_img, cmap='gray')
#plt.savefig("cv_clean.png",dpi=300, bbox_inches='tight')
plt.show()

my_cleaned_img = detector.paper_clean_fast(blurred_img)
plt.imshow(my_cleaned_img, cmap='gray')
#plt.savefig("simple_clean.png",dpi=300, bbox_inches='tight')
plt.show()


print("\nRunning line detection (this may take a moment)...")
fdetector = detector.LineDetector(width=2, height=10, step=5) 
lines = fdetector.findall_lines(cleaned_img)
lines.sort(key = lambda x: len(x))

line = lines[-5]
#print(line)

line_pts = np.asarray(line)

# (1) Scatter plot of points directly from line
plt.figure()
plt.imshow(cleaned_img, cmap='gray')
plt.scatter(line_pts[:, 1], line_pts[:, 0], c='red', s=2)
plt.savefig("line_points.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# (2) Separate plot with points joined by straight line segments
plt.figure()
plt.imshow(cleaned_img, cmap='gray')
plt.plot(line_pts[:, 1], line_pts[:, 0], color='red', linewidth=1)
plt.savefig("line_connected.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

T = np.linspace(0, 1, 1000)
(x, y), (vx, vy), (ax, ay) = detector.curve_fit(T, line, bin_size=0.13, deg=2)

plt.imshow(cleaned_img, cmap='gray')
plt.plot(y, x, color='red', linewidth=2)
plt.savefig("curve_fit.png",dpi=300, bbox_inches='tight')
plt.show()

