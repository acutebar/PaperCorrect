import os
import sys
import glob
import time
import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, CheckButtons
from scipy.interpolate import RegularGridInterpolator
from PIL import Image, ExifTags

import detector
from detector.gradient_descent import run_gradient_descent, generate_flat_rho_cloud, run_multi_start_optimization
from detector.energy import surface_fit, total_energy

# =============================================================================
# GLOBAL HYPERPARAMETERS
# =============================================================================
GD_STEPS = 100                   
GD_NORM_CUTOFF = 1e-5            
GD_LEARNING_RATE = 0.05          
GD_CONTROL_POINTS = 64           

TIME_DOMAIN_STEPS = 50           
MESH_DENSITY = 250                

CURVE_MIN_LENGTH = 5            
CURVE_MAX_COUNT = 200             
ENERGY_CUTOFF = 3.0
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
        self.ax_img.set_title("STAGE 1: Drag rectangle to crop (or leave for full image). Press Enter to run.", fontweight='bold')
        
        self.rs = RectangleSelector(self.ax_img, self.on_crop_select, useblit=True, 
                                    button=[1], interactive=True,
                                    props=dict(facecolor='cyan', edgecolor='blue', alpha=0.2, fill=True))
        
        self.fig_select.canvas.mpl_connect('key_press_event', self.on_crop_key)
        plt.show()

    def on_crop_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.crop_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        
    def on_crop_key(self, event):
        if event.key in ['enter', 'return']:
            plt.close(self.fig_select)
            self.run_detection()

    def run_detection(self):
        h, w = self.gray_img.shape
        x_min, y_min, x_max, y_max = self.crop_box if self.crop_box else (0, 0, w, h)
        if (x_max - x_min) < 30 or (y_max - y_min) < 30:
            x_min, y_min, x_max, y_max = 0, 0, w, h
            
        self.cropped_img = self.gray_img[y_min:y_max, x_min:x_max]
        self.cropped_rgb = self.img_rgb[y_min:y_max, x_min:x_max]
        self.crop_params = (x_min, y_min, x_max, y_max, w, h)
        
        print("Detecting lines...")
        blurred = cv.GaussianBlur(self.cropped_img, (5, 5), 0)
        cleaned = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 15)
        
        detec = detector.LineDetector(width=2, height=10, step=5)
        raw_lines = detec.findall_lines(cleaned)
        
        valid_lines = [l for l in raw_lines if len(l) > CURVE_MIN_LENGTH]
        valid_lines.sort(key=len, reverse=True)
        raw_selected = valid_lines[:CURVE_MAX_COUNT]
        
        if not raw_selected:
            print("No valid lines found. Try a different region.")
            return

        # Initialize baseline variables required for energy evaluation
        flat_cloud = generate_flat_rho_cloud(GD_CONTROL_POINTS, torch.pi/3)
        self.baseline_coords = flat_cloud[:, :2].detach()
        self.baseline_values = flat_cloud[:, 2].detach().clone().requires_grad_(True)
        self.T_array = np.linspace(0, 1, TIME_DOMAIN_STEPS)

        self.selected_lines = []
        self.curves_gd = []
        
        for line in raw_selected:
            centered_line = [(pt[1] + x_min - w / 2.0, pt[0] + y_min - h / 2.0) for pt in line]
            cur_curve = detector.Curve(centered_line, deg=2, bin_size=0.13)
            
            with torch.no_grad():
                e = total_energy(self.T_array, [cur_curve], self.baseline_coords, self.baseline_values, self.f_pixels).item()
                
            if e < ENERGY_CUTOFF:
                self.selected_lines.append(line)
                self.curves_gd.append(cur_curve)

        print(f"Detected {len(self.selected_lines)} curves passing energy cutoff.")
        
        if not self.selected_lines:
            print("No lines passed the energy cutoff.")
            return

        self.interactive_line_selection()

    def interactive_line_selection(self):
        self.fig_lines = plt.figure(figsize=(10, 8))
        self.ax_lines = self.fig_lines.add_subplot(111)
        self.ax_lines.imshow(self.cropped_rgb)
        
        # Default all lines to True (opt-out selection)
        self.line_active = [True] * len(self.curves_gd)
        self.line_picker_map = {}
        
        x_min, y_min, _, _, w, h = self.crop_params
        
        for i, curve in enumerate(self.curves_gd):
            cx, cy = curve.point_at(self.T_array)
            plot_x = cx + w/2.0 - x_min
            plot_y = cy + h/2.0 - y_min
            
            # Start lines as solid cyan
            ln, = self.ax_lines.plot(plot_x, plot_y, '-', color='cyan', alpha=1.0, linewidth=2.5, picker=True, pickradius=5)
            self.line_picker_map[ln] = i
            
        self.update_live_energy()
        self.fig_lines.canvas.mpl_connect('pick_event', self.on_line_pick)
        self.fig_lines.canvas.mpl_connect('key_press_event', self.on_line_key)
        plt.show()

    def update_live_energy(self):
        active_curves = [self.curves_gd[i] for i in range(len(self.curves_gd)) if self.line_active[i]]
        if not active_curves:
            energy_str = "0.0000"
        else:
            with torch.no_grad():
                e = total_energy(self.T_array, active_curves, self.baseline_coords, self.baseline_values, self.f_pixels).item()
            energy_str = f"{e:.4f}"
            
        self.ax_lines.set_title(f"STAGE 2: Click lines to deselect (Red=Inactive). Press Enter to optimize.\nInitial Baseline Energy of Active Lines: {energy_str}", fontweight='bold')
        self.fig_lines.canvas.draw_idle()

    def on_line_pick(self, event):
        ln = event.artist
        if ln in self.line_picker_map:
            idx = self.line_picker_map[ln]
            self.line_active[idx] = not self.line_active[idx]
            
            ln.set_color('cyan' if self.line_active[idx] else 'red')
            ln.set_alpha(1.0 if self.line_active[idx] else 0.4)
            self.update_live_energy()

    def on_line_key(self, event):
        if event.key in ['enter', 'return']:
            plt.close(self.fig_lines)
            self.run_optimization()

    def run_optimization(self):
        active_lines = [self.selected_lines[i] for i in range(len(self.curves_gd)) if self.line_active[i]]
        active_curves_gd = [self.curves_gd[i] for i in range(len(self.curves_gd)) if self.line_active[i]]
        
        if not active_curves_gd:
            print("All lines deselected. Exiting.")
            return

        print(f"\nStarting Gradient Descent on {len(active_curves_gd)} lines...")
        T = np.linspace(0, 1, TIME_DOMAIN_STEPS)
        t_start = time.time()
        
        opt_cloud, final_energy_val = run_multi_start_optimization(
            T, active_curves_gd, f=self.f_pixels, 
            num_points=GD_CONTROL_POINTS, 
            learning_rate=GD_LEARNING_RATE, 
            steps=GD_STEPS, 
            eps=GD_NORM_CUTOFF
        )
        print(f"GD finished in {time.time() - t_start:.2f}s.")
        
        with torch.no_grad():
            final_energy = total_energy(T, active_curves_gd, opt_cloud[:, :2], opt_cloud[:, 2], self.f_pixels).item()
        
        cloud_pts = opt_cloud[:, :2].detach().numpy()
        cloud_vals = opt_cloud[:, 2].detach().numpy()
        
        x_min, y_min, x_max, y_max, w, h = self.crop_params
        X_grid = np.linspace(x_min - w/2.0, x_max - w/2.0, MESH_DENSITY)
        Y_grid = np.linspace(y_min - h/2.0, y_max - h/2.0, MESH_DENSITY)
        X, Y = np.meshgrid(X_grid, Y_grid)
        
        R = np.sqrt(X**2 + Y**2 + self.f_pixels**2)
        u_mesh, v_mesh = X / (self.f_pixels + R), Y / (self.f_pixels + R)
        
        rho_mesh_tensor = surface_fit(u_mesh, v_mesh, opt_cloud, bin_size=0.5)[0]
        rho_mesh = rho_mesh_tensor.detach().numpy().reshape(X.shape)
        
        P_X = rho_mesh * (X / R)
        P_Y = rho_mesh * (Y / R)
        P_Z = rho_mesh * (self.f_pixels / R)
        
        interp = RegularGridInterpolator((Y_grid, X_grid), rho_mesh, bounds_error=False, fill_value=None)
        
        self.show_dashboard(active_lines, active_curves_gd, T, P_X, P_Y, P_Z, interp, final_energy)

    def show_dashboard(self, active_lines, active_curves_gd, T, P_X, P_Y, P_Z, interp, final_energy):
        fig = plt.figure(figsize=(14, 7))
        plt.subplots_adjust(left=0.2) 
        x_min, y_min, x_max, y_max, w, h = self.crop_params
        
        ax_2d = fig.add_subplot(121)
        ax_2d.imshow(self.cropped_rgb)
        ax_2d.set_title("2D Detected Curves")
        ax_2d.axis('off')
        
        ax_3d = fig.add_subplot(122, projection='3d')
        
        norm_rgb = self.cropped_rgb.astype(float) / 255.0
        tex = cv.resize(norm_rgb, (MESH_DENSITY, MESH_DENSITY)) 
        
        self.surf_tex = ax_3d.plot_surface(P_X, P_Y, P_Z, facecolors=tex, shade=False, alpha=0.9, edgecolor='none', rcount=MESH_DENSITY, ccount=MESH_DENSITY)
        self.surf_solid = ax_3d.plot_surface(P_X, P_Y, P_Z, color='gainsboro', shade=True, alpha=0.9, edgecolor='none', rcount=MESH_DENSITY, ccount=MESH_DENSITY)
        self.surf_solid.set_visible(False)
        
        ax_3d.set_title(f"3D Reconstructed Paper Surface\nTotal Final Energy: {final_energy:.4f}")
        ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Depth (Z)")
        
        self.toggleable_lines = []
        
        for i, (line_raw, curve_gd) in enumerate(zip(active_lines, active_curves_gd)):
            pts = np.array(line_raw)
            ln1, = ax_2d.plot(pts[:, 1], pts[:, 0], 'r.', markersize=2)
            
            fit_x, fit_y = curve_gd.point_at(T)
            cx, cy = fit_x, fit_y
            
            plot_x = cx + w/2.0 - x_min
            plot_y = cy + h/2.0 - y_min
            ln2, = ax_2d.plot(plot_x, plot_y, 'b-', linewidth=1, alpha=0.7)
            
            self.toggleable_lines.extend([ln1, ln2])
            
            mask = (cx >= x_min - w/2) & (cx <= x_min + w/2) & (cy >= y_min - h/2) & (cy <= y_min + h/2)
            cx, cy = cx[mask], cy[mask]
            
            if len(cx) > 0:
                rho_curve = interp((cy, cx))
                R_curve = np.sqrt(cx**2 + cy**2 + self.f_pixels**2)
                
                curve_3d_x = rho_curve * (cx / R_curve)
                curve_3d_y = rho_curve * (cy / R_curve)
                curve_3d_z = rho_curve * (self.f_pixels / R_curve)
                
                ln3, = ax_3d.plot(curve_3d_x, curve_3d_y, curve_3d_z, color='cyan', linewidth=2.5, zorder=10)
                self.toggleable_lines.append(ln3)
        
        ax_check = fig.add_axes([0.02, 0.5, 0.12, 0.15])
        self.check_buttons = CheckButtons(ax_check, ['Texture', 'Lines'], [True, True])
        
        def ui_toggle(label):
            if label == 'Texture':
                tex_on = self.check_buttons.get_status()[0]
                self.surf_tex.set_visible(tex_on)
                self.surf_solid.set_visible(not tex_on)
            elif label == 'Lines':
                lines_on = self.check_buttons.get_status()[1]
                for ln in self.toggleable_lines:
                    ln.set_visible(lines_on)
            fig.canvas.draw_idle()
            
        self.check_buttons.on_clicked(ui_toggle)
        
        ax_3d.view_init(elev=25, azim=-65)
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
            choice = input(f"Select image index or type filename: ").strip()
        except EOFError:
            choice = ""
            
        if not choice:
            target = img_files[0] if img_files else "curve.jpeg"
        elif choice.isdigit() and int(choice) < len(img_files):
            target = img_files[int(choice)]
        else:
            target = choice
            
        if not os.path.exists(target) and img_files: 
            target = img_files[0]
            print(f"File not found, defaulting to {target}")
            
    app = PaperCorrectApp(target)
