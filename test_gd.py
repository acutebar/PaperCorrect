import os
import sys
import glob
import time
import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.interpolate import RegularGridInterpolator
from PIL import Image, ExifTags

import detector
from detector.gradient_descent import run_gradient_descent
from detector.energy import surface_fit

# =============================================================================
# GLOBAL HYPERPARAMETERS - TWEAK THESE FOR TESTING
# =============================================================================
GD_STEPS = 100                   # Number of steps for Gradient Descent
GD_NORM_CUTOFF = 1e-5            # Stopping threshold for gradient norm
GD_LEARNING_RATE = 0.05          # Base learning rate for Adam optimizer
GD_CONTROL_POINTS = 64           # Number of S^2 control points for deformation

TIME_DOMAIN_STEPS = 50           # Number of points to sample along each curve
MESH_DENSITY = 80                # Grid resolution for 3D paper surface evaluation

CURVE_MIN_LENGTH = 10            # Minimum number of pixels to accept a curve
CURVE_MAX_COUNT = 80             # Max number of curves to use in GD
# =============================================================================

def get_focal_length_pixels(image_path):
    try:
        img_pil = Image.open(image_path)
        exif = img_pil._getexif()
        focal_35mm = 50.0
        if exif:
            for tag_id, value in exif.items():
                if ExifTags.TAGS.get(tag_id, tag_id) == 'FocalLengthIn35mmFilm':
                    focal_35mm = float(value)
                    break
        return (focal_35mm / 36.0) * img_pil.size[0]
    except Exception:
        return 2912.0

class PaperCorrectApp:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img_bgr = cv.imread(image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        self.img_rgb = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2RGB)
        self.gray_img = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2GRAY)
        self.f_pixels = get_focal_length_pixels(image_path)
        self.crop_box = None
        
        print(f"Loaded: {image_path} ({self.img_bgr.shape[1]}x{self.img_bgr.shape[0]}), Focal: {self.f_pixels:.1f}px")
        
        self.fig_select = plt.figure(figsize=(10, 8))
        self.ax_img = self.fig_select.add_subplot(111)
        self.ax_img.imshow(self.gray_img, cmap='gray')
        self.ax_img.set_title("Drag rectangle to crop (or leave for full image). Press Enter to run.", fontweight='bold')
        
        self.rs = RectangleSelector(self.ax_img, self.on_select, useblit=True, 
                                    button=[1], interactive=True,
                                    props=dict(facecolor='cyan', edgecolor='blue', alpha=0.2, fill=True))
        
        self.fig_select.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    def on_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.crop_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        
    def on_key(self, event):
        if event.key in ['enter', 'return']:
            plt.close(self.fig_select)
            self.run_pipeline()

    def run_pipeline(self):
        # 1. Setup Crop
        h, w = self.gray_img.shape
        x_min, y_min, x_max, y_max = self.crop_box if self.crop_box else (0, 0, w, h)
        if (x_max - x_min) < 30 or (y_max - y_min) < 30:
            x_min, y_min, x_max, y_max = 0, 0, w, h
            
        cropped_img = self.gray_img[y_min:y_max, x_min:x_max]
        cropped_rgb = self.img_rgb[y_min:y_max, x_min:x_max]
        
        # 2. Detect Lines
        print("Detecting lines...")
        blurred = cv.GaussianBlur(cropped_img, (5, 5), 0)
        cleaned = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 15)
        
        detec = detector.LineDetector(width=2, height=10, step=5)
        raw_lines = detec.findall_lines(cleaned)
        
        valid_lines = [l for l in raw_lines if len(l) > CURVE_MIN_LENGTH]
        valid_lines.sort(key=len, reverse=True)
        selected_lines = valid_lines[:CURVE_MAX_COUNT]
        
        if not selected_lines:
            print("No valid lines found. Try a different region.")
            return
            
        print(f"Selected {len(selected_lines)} curves.")
        
        # Transform lines to image-centered coordinates for GD
        curves_gd = []
        for line in selected_lines:
            centered_line = [(pt[1] + x_min - w / 2.0, pt[0] + y_min - h / 2.0) for pt in line]
            curves_gd.append(detector.Curve(centered_line, deg=2, bin_size=0.13))

        # 3. Gradient Descent Optimization
        print("\nStarting Gradient Descent...")
        T = np.linspace(0, 1, TIME_DOMAIN_STEPS)
        t_start = time.time()
        
        opt_cloud = run_gradient_descent(
            T, curves_gd, f=self.f_pixels, 
            num_points=GD_CONTROL_POINTS, 
            learning_rate=GD_LEARNING_RATE, 
            steps=GD_STEPS, 
            eps=GD_NORM_CUTOFF
        )
        print(f"GD finished in {time.time() - t_start:.2f}s.")
        
        cloud_pts = opt_cloud[:, :2].detach().numpy()
        cloud_vals = opt_cloud[:, 2].detach().numpy()
        
        # 4. Mesh Reconstruction
        X_grid = np.linspace(x_min - w/2.0, x_max - w/2.0, MESH_DENSITY)
        Y_grid = np.linspace(y_min - h/2.0, y_max - h/2.0, MESH_DENSITY)
        X, Y = np.meshgrid(X_grid, Y_grid)
        
        # Convert to polar on S^2 stereographic plane for surface fit
        R = np.sqrt(X**2 + Y**2 + self.f_pixels**2)
        u_mesh, v_mesh = X / (self.f_pixels + R), Y / (self.f_pixels + R)
        
        rho_mesh_tensor = surface_fit(u_mesh, v_mesh, opt_cloud, bin_size=0.5)[0]
        rho_mesh = rho_mesh_tensor.detach().numpy().reshape(X.shape)
        
        # TRUE 3D GEOMETRY: P = rho * (X/R, Y/R, f/R)
        P_X = rho_mesh * (X / R)
        P_Y = rho_mesh * (Y / R)
        P_Z = rho_mesh * (self.f_pixels / R)
        
        # Interpolator for curve placement (interpolate rho, not depth)
        interp = RegularGridInterpolator((Y_grid, X_grid), rho_mesh, bounds_error=False, fill_value=None)
        
        # 5. Visualization Dashboard
        self.show_results(cropped_rgb, selected_lines, curves_gd, T, P_X, P_Y, P_Z, interp, x_min, y_min, w, h, cloud_pts, cloud_vals)

    def show_results(self, cropped_rgb, selected_lines, curves_gd, T, P_X, P_Y, P_Z, interp, x_min, y_min, w, h, cloud_pts, cloud_vals):
        fig = plt.figure(figsize=(14, 7))
        
        # Left Subplot: 2D Detection
        ax_2d = fig.add_subplot(121)
        ax_2d.imshow(cropped_rgb)
        ax_2d.set_title("2D Detected Curves")
        ax_2d.axis('off')
        
        # Right Subplot: 3D Paper Surface
        ax_3d = fig.add_subplot(122, projection='3d')
        
        # Map image texture to surface
        norm_rgb = cropped_rgb.astype(float) / 255.0
        tex = cv.resize(norm_rgb, (MESH_DENSITY, MESH_DENSITY))
        # Flip to match surface coordinate orientation
        tex = tex[::-1, :, :] 
        
        ax_3d.plot_surface(P_X, P_Y, P_Z, facecolors=tex, shade=False, alpha=0.9, edgecolor='none')
        ax_3d.set_title("3D Reconstructed Paper Surface")
        ax_3d.set_xlabel("X (Camera Space)"); ax_3d.set_ylabel("Y (Camera Space)"); ax_3d.set_zlabel("Depth (Z)")
        
        # Overlay Curves
        for i, (line_raw, curve_gd) in enumerate(zip(selected_lines, curves_gd)):
            # 2D Plot (line_raw points are already local to the cropped image)
            pts = np.array(line_raw)
            ax_2d.plot(pts[:, 1], pts[:, 0], 'r.', markersize=2)
            
            # 2D Fitted Curves
            fit_x, fit_y = curve_gd.point_at(T)
            cx, cy = fit_x, fit_y
            
            # Overlay fitted GD curves back onto 2D image
            plot_x = cx + w/2.0 - x_min
            plot_y = cy + h/2.0 - y_min
            ax_2d.plot(plot_x, plot_y, 'b-', linewidth=1, alpha=0.7)
            
            # 3D Plot - Map image points to true 3D rays
            mask = (cx >= x_min - w/2) & (cx <= x_min + w/2) & (cy >= y_min - h/2) & (cy <= y_min + h/2)
            cx, cy = cx[mask], cy[mask]
            
            if len(cx) > 0:
                rho_curve = interp((cy, cx))
                R_curve = np.sqrt(cx**2 + cy**2 + self.f_pixels**2)
                
                curve_3d_x = rho_curve * (cx / R_curve)
                curve_3d_y = rho_curve * (cy / R_curve)
                curve_3d_z = rho_curve * (self.f_pixels / R_curve)
                
                ax_3d.plot(curve_3d_x, curve_3d_y, curve_3d_z, color='cyan', linewidth=2.5, zorder=10)
        
        ax_3d.view_init(elev=25, azim=-65)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    img_files = [f for f in sorted(glob.glob("*.jpeg") + glob.glob("*.jpg") + glob.glob("*.png")) if not f.startswith("test_")]
    
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        target = sys.argv[1]
    else:
        print("Available images:")
        for i, f in enumerate(img_files):
            print(f"[{i}] {f}")
        try:
            choice = input(f"Select image index or type filename (default 'curve.jpeg'): ").strip()
        except EOFError:
            choice = ""
            
        if not choice:
            target = "curve.jpeg"
        elif choice.isdigit() and int(choice) < len(img_files):
            target = img_files[int(choice)]
        else:
            target = choice
            
        if not os.path.exists(target) and img_files: 
            target = img_files[0]
            print(f"File not found, defaulting to {target}")
            
    app = PaperCorrectApp(target)
