import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from PIL import Image, ExifTags

import detector
from detector.gradient_descent import run_gradient_descent
from detector.energy import surface_fit

def get_focal_length_pixels(image_path):
    """Extracts focal length from EXIF metadata and converts to pixels."""
    img_pil = Image.open(image_path)
    exif = img_pil._getexif()
    
    focal_35mm = None
    if exif:
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break
                
    if focal_35mm is None:
        print("EXIF FocalLengthIn35mmFilm not found. Defaulting to 50mm eq.")
        focal_35mm = 50.0

    img_width = img_pil.size[0]
    return (focal_35mm / 36.0) * img_width

def process_and_plot(image_path):
    # 1. Pipeline Setup & Extraction
    f_pixels = get_focal_length_pixels(image_path)
    print(f"Focal Length computed: {f_pixels:.2f} pixels")
    
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img_h, img_w = img.shape
    
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    cleaned = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 15)
    
    detec = detector.LineDetector(width=2, height=10, step=5)
    raw_lines = detec.findall_lines(cleaned)
    
    # Trace curves and shift origin to center of image
    T = np.linspace(0, 1, 100)
    curves = []
    for line in raw_lines:
        if len(line) > 10:
            centered_line = [(pt[0] - img_w/2, pt[1] - img_h/2) for pt in line]
            # FIX: Instantiate the Curve object instead of calling the raw fitting tuple
            curves.append(detector.Curve(centered_line))
            
    # 2. Optimization
    optimized_cloud = run_gradient_descent(T, curves, f_pixels)
    
    # 3. Surface Construction
    # Create a coordinate grid representing the flat image plane
    x_grid = np.linspace(-img_w/2, img_w/2, 50)
    y_grid = np.linspace(-img_h/2, img_h/2, 50)
    X_img, Y_img = np.meshgrid(x_grid, y_grid)
    
    # Map image coordinates to stereographic (u,v) to sample rho
    R_img = np.sqrt(X_img**2 + Y_img**2 + f_pixels**2)
    U_grid = X_img / (R_img + f_pixels)
    V_grid = Y_img / (R_img + f_pixels)
    
    rho_surface, _, _, _, _, _ = surface_fit(U_grid, V_grid, optimized_cloud, deg=2, bin_size=1)
    
    # Map back to physical 3D space: gamma = rho * lambda * r
    lambda_scale = 1.0 / R_img
    W_grid = rho_surface * lambda_scale
    
    X_3D = W_grid * X_img
    Y_3D = W_grid * Y_img
    Z_3D = W_grid * f_pixels
    
    # 4. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_3D, Y_3D, Z_3D, cmap='viridis', alpha=0.9, edgecolor='none')
    
    curve_artists = []
    for curve in curves:
        x_c, y_c = curve.point_at(T)
        
        # FIX: Flatten nested dimension caused by np.array([t]) in the Curve class
        x_c, y_c = x_c.flatten(), y_c.flatten()
        
        # Calculate 3D position of the curves
        R_c = np.sqrt(x_c**2 + y_c**2 + f_pixels**2)
        u_c = x_c / (R_c + f_pixels)
        v_c = y_c / (R_c + f_pixels)
        
        rho_c, _, _, _, _, _ = surface_fit(u_c, v_c, optimized_cloud, deg=2, bin_size=1)
        
        w_c = rho_c * (1.0 / R_c)
        x_3d_c = w_c * x_c
        y_3d_c = w_c * y_c
        z_3d_c = w_c * f_pixels
        
        line_obj, = ax.plot(x_3d_c, y_3d_c, z_3d_c, color='red', linewidth=2)
        curve_artists.append(line_obj)

    # UI Setup
    ax.set_title("Optimized Paper Shape")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth (Z)")
    
    ax_check = fig.add_axes([0.05, 0.45, 0.15, 0.15])
    check = CheckButtons(ax_check, ['Show Curves'], [True])
    
    def toggle_curves(label):
        visible = not curve_artists[0].get_visible()
        for artist in curve_artists:
            artist.set_visible(visible)
        fig.canvas.draw_idle()
        
    check.on_clicked(toggle_curves)
    
    plt.show()

if __name__ == '__main__':
    #process_and_plot("crump_uncropped.jpeg")
    #process_and_plot("crumpled.jpeg")
    process_and_plot("curve.jpeg")
    #process_and_plot("curvedsurface1.jpeg")
    #process_and_plot("curvedsurface2.jpeg")
    #process_and_plot("curvedsurface3.jpeg")
    #process_and_plot("curvedsurface4.jpeg")
    #process_and_plot("output.jpg")
    #process_and_plot("ruled_paper.jpeg")
    #process_and_plot("straight_line.jpeg")
