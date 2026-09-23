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
from detector.gradient_descent import run_gradient_descent, generate_flat_cloud, run_multi_start_optimization
from detector.energy import surface_fit, total_energy

# =============================================================================
# GLOBAL HYPERPARAMETERS
# =============================================================================
GD_STEPS = 100                   
GD_NORM_CUTOFF = 0.5            
GD_LEARNING_RATE = 0.001          
GD_CONTROL_GRID_MULT = 70         # Creates an 8x8 control point grid over the cropped area

TIME_DOMAIN_STEPS = 100           
MESH_DENSITY = 60                

CURVE_MIN_LENGTH = 40            
CURVE_MAX_COUNT = 200             
ENERGY_CUTOFF = 3.0

CURVE_BIN_SIZE = 0.2           # Bin size for B-spline curve fitting in normalized space
SURFACE_BIN_SIZE = 2         # Bin size for 3D surface mesh rendering
SPLINE_DEG = 3
# =============================================================================

def get_focal_length_pixels(image_path):
    IPHONE13_WIDE_FOCAL_35MM = 26.0  # iPhone 13 (non-Pro) main wide camera, 35mm-equivalent
    FILM_DIAGONAL_MM = (36.0 ** 2 + 24.0 ** 2) ** 0.5  # diagonal of a 36x24mm 35mm-film frame
    DEFAULT_SIZE_PX = (4032, 3024)  # iPhone 13 wide camera native 12MP resolution

    try:
        img_pil = Image.open(image_path)
        w_px, h_px = img_pil.size
    except Exception as e:
        print(f"Warning: Could not open {image_path} ({e}). "
              f"Assuming iPhone 13 wide camera at {DEFAULT_SIZE_PX[0]}x{DEFAULT_SIZE_PX[1]}.")
        w_px, h_px = DEFAULT_SIZE_PX
        exif = None
    else:
        exif = img_pil._getexif()

    focal_35mm = None
    if exif:
        for tag_id, value in exif.items():
            if ExifTags.TAGS.get(tag_id, tag_id) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break

    if focal_35mm is None:
        print(f"Warning: EXIF FocalLengthIn35mmFilm not found for {image_path}. "
              f"Defaulting to {IPHONE13_WIDE_FOCAL_35MM}mm (iPhone 13 wide camera).")
        focal_35mm = IPHONE13_WIDE_FOCAL_35MM

    # FocalLengthIn35mmFilm is diagonal-FOV-equivalent, so pixel focal length must be
    # scaled by the sensor's diagonal, not its width, to be correct for non-3:2 aspect ratios.
    diag_px = (w_px ** 2 + h_px ** 2) ** 0.5
    return (focal_35mm / FILM_DIAGONAL_MM) * diag_px

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
        
        # Exact Normalized Bounds for the Cropped Region specifically
        self.x_min_norm = (x_min - w / 2.0) / self.f_pixels
        self.x_max_norm = (x_max - w / 2.0) / self.f_pixels
        self.y_min_norm = (y_min - h / 2.0) / self.f_pixels
        self.y_max_norm = (y_max - h / 2.0) / self.f_pixels
        
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

        # Initialize flat depth cloud tightly bound to the cropped domain
        flat_cloud = generate_flat_cloud(
            self.x_min_norm, self.x_max_norm, 
            self.y_min_norm, self.y_max_norm, 
            mult=GD_CONTROL_GRID_MULT, 
            depth=1.0
        )
        self.baseline_coords = flat_cloud[:, :2].detach()
        self.baseline_values = flat_cloud[:, 2].detach().clone().requires_grad_(True)
        self.T_array = np.linspace(0, 1, TIME_DOMAIN_STEPS)

        self.selected_lines = []
        self.curves_gd = []
        
        for line in raw_selected:
            # Shift center to absolute 0,0 optical center and shrink
            centered_line_norm = [
                ((pt[1] + x_min - w / 2.0) / self.f_pixels, 
                 (pt[0] + y_min - h / 2.0) / self.f_pixels) 
                for pt in line
            ]
            cur_curve = detector.Curve(centered_line_norm, deg=SPLINE_DEG, bin_size=CURVE_BIN_SIZE)
            
            with torch.no_grad():
                e = total_energy(self.T_array, [cur_curve], self.baseline_coords, self.baseline_values, f=1.0).item()
                
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
        
        self.line_active = [True] * len(self.curves_gd)
        self.line_picker_map = {}
        
        x_min, y_min, _, _, w, h = self.crop_params
        
        for i, curve in enumerate(self.curves_gd):
            cx_norm, cy_norm = curve.point_at(self.T_array)
            # Re-scale back to pixel magnitude relative to crop for UI overlay
            plot_x = (cx_norm * self.f_pixels) + w/2.0 - x_min
            plot_y = (cy_norm * self.f_pixels) + h/2.0 - y_min
            
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
                e = total_energy(self.T_array, active_curves, self.baseline_coords, self.baseline_values, f=1.0).item()
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

        print(f"\nStarting Adam Optimization on {len(active_curves_gd)} lines...")
        T = np.linspace(0, 1, TIME_DOMAIN_STEPS)
        t_start = time.time()
        
        opt_cloud, final_energy_val = run_multi_start_optimization(
            T, active_curves_gd, f=1.0, 
            num_points=len(T), 
            learning_rate=GD_LEARNING_RATE, 
            steps=GD_STEPS, 
            eps=GD_NORM_CUTOFF,
            x_start=self.x_min_norm, x_end=self.x_max_norm,
            y_start=self.y_min_norm, y_end=self.y_max_norm,
            mult=GD_CONTROL_GRID_MULT,
            deg=SPLINE_DEG,
            bin_size=SURFACE_BIN_SIZE
        )
        print(f"GD finished in {time.time() - t_start:.2f}s.")
        
        with torch.no_grad():
            final_energy = total_energy(T, active_curves_gd, opt_cloud[:, :2], opt_cloud[:, 2], f=1.0).item()
        
        # Build 3D Mesh tightly across the cropped region
        X_grid_norm = np.linspace(self.x_min_norm, self.x_max_norm, MESH_DENSITY)
        Y_grid_norm = np.linspace(self.y_min_norm, self.y_max_norm, MESH_DENSITY)
        X_norm, Y_norm = np.meshgrid(X_grid_norm, Y_grid_norm)
        
        u_mesh_flat = torch.tensor(X_norm.flatten(), dtype=torch.float64)
        v_mesh_flat = torch.tensor(Y_norm.flatten(), dtype=torch.float64)
        
        # Fit depth parameter w over normalized coordinates (using scaled SURFACE_BIN_SIZE)
        w_mesh_tensor = surface_fit(u_mesh_flat, v_mesh_flat, opt_cloud, deg=SPLINE_DEG, bin_size=SURFACE_BIN_SIZE)[0]
        w_mesh = w_mesh_tensor.detach().numpy().reshape(X_norm.shape)
        
        # Map physical 3D coordinates, scaled back to pixel magnitude
        P_X = w_mesh * (X_norm * self.f_pixels)
        P_Y = -w_mesh * (Y_norm * self.f_pixels)
        P_Z = -w_mesh * self.f_pixels
        
        interp = RegularGridInterpolator((Y_grid_norm, X_grid_norm), w_mesh, bounds_error=False, fill_value=None)
        
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

        pz = P_Z - np.mean(P_Z)
        
        self.surf_tex = ax_3d.plot_surface(P_X, P_Y, pz, facecolors=tex, shade=False, alpha=0.9, edgecolor='none', rcount=MESH_DENSITY, ccount=MESH_DENSITY)
        self.surf_solid = ax_3d.plot_surface(P_X, P_Y, pz, color='gainsboro', shade=True, alpha=0.9, edgecolor='none', rcount=MESH_DENSITY, ccount=MESH_DENSITY)
        self.surf_solid.set_visible(False)

        all_mins = [ax_3d.get_xlim()[0], ax_3d.get_ylim()[0], ax_3d.get_zlim()[0]]
        all_maxs = [ax_3d.get_xlim()[1], ax_3d.get_ylim()[1], ax_3d.get_zlim()[1]]
        common_lim = (min(all_mins), max(all_maxs))
        
        ax_3d.set_xlim(common_lim)
        ax_3d.set_ylim(common_lim)
        ax_3d.set_zlim(common_lim)
        ax_3d.set_box_aspect([1, 1, 1])

        
        ax_3d.set_title(f"3D Reconstructed Paper Surface\nTotal Final Energy: {final_energy:.4f}")
        ax_3d.set_xlabel("X (Pixels)"); ax_3d.set_ylabel("Y (Pixels)"); ax_3d.set_zlabel("Depth (Z)")
        
        self.toggleable_lines = []
        
        for i, (line_raw, curve_gd) in enumerate(zip(active_lines, active_curves_gd)):
            pts = np.array(line_raw)
            ln1, = ax_2d.plot(pts[:, 1], pts[:, 0], 'r.', markersize=2)
            
            fit_x_norm, fit_y_norm = curve_gd.point_at(T)
            cx_norm, cy_norm = fit_x_norm, fit_y_norm
            
            plot_x = (cx_norm * self.f_pixels) + w/2.0 - x_min
            plot_y = (cy_norm * self.f_pixels) + h/2.0 - y_min
            ln2, = ax_2d.plot(plot_x, plot_y, 'b-', linewidth=1, alpha=0.7)
            
            self.toggleable_lines.extend([ln1, ln2])
            
            mask = (cx_norm >= self.x_min_norm) & (cx_norm <= self.x_max_norm) & (cy_norm >= self.y_min_norm) & (cy_norm <= self.y_max_norm)
            cx_valid, cy_valid = cx_norm[mask], cy_norm[mask]
            
            if len(cx_valid) > 0:
                w_curve = interp((cy_valid, cx_valid))
                
                curve_3d_x = w_curve * (cx_valid * self.f_pixels)
                curve_3d_y = -w_curve * (cy_valid * self.f_pixels)
                curve_3d_z = -w_curve * self.f_pixels - np.mean(P_Z)
                
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
        plt.savefig("embedding.png", dpi=300, bbox_inches='tight')
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
