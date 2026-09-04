import os
import sys
import glob
import time
import math
import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, CheckButtons, Button
from PIL import Image, ExifTags

import detector
from detector.gradient_descent import run_gradient_descent
from detector.energy import surface_fit


def get_focal_length_pixels(image_path):
    """Extracts focal length from EXIF metadata and converts to pixels."""
    try:
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
    except Exception as e:
        print(f"Warning: Could not read EXIF metadata ({e}). Defaulting focal length to 2912.0 px.")
        return 2912.0


class AdaptiveStep:
    """Adaptive normalized step for stable gradient descent regardless of image scale."""
    def __init__(self, step_idx, total_steps, base_delta=0.015):
        self.step_idx = step_idx
        self.total_steps = total_steps
        self.max_delta = base_delta / (1.0 + 0.03 * step_idx)

    def __mul__(self, grad):
        norm = torch.norm(grad).item()
        if (self.step_idx + 1) % 5 == 0 or self.step_idx == 0 or self.step_idx == self.total_steps - 1:
            print(f"  [GD Step {self.step_idx+1:3d}/{self.total_steps}] Grad Norm: {norm:10.2f} | Step Delta: {self.max_delta:.4f}")
        if norm > 1e-12:
            return (self.max_delta / norm) * grad
        return 0.0 * grad


class GDFrontEnd:
    """
    Front-end application for PaperCorrect:
    1. Interactive ROI Selection & Curve Scanning (similar to interface.py)
    2. Seamless continuation into PyTorch Gradient Descent
    3. Dual 3D Visualization:
       - Original 3D embedding of paper: P = rho(x) * x
       - S^2 spherical deformation field: rho(x) * x for x in S^2
    """
    def __init__(self, image_path, auto_run=False, output_path=None, steps=30):
        self.image_path = image_path
        self.auto_run = auto_run
        self.output_path = output_path
        self.steps = steps

        # Load image
        self.img_bgr = cv.imread(image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        self.img_rgb = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2RGB)
        self.gray_img = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2GRAY)
        self.img_h, self.img_w = self.gray_img.shape
        self.f_pixels = get_focal_length_pixels(image_path)
        print(f"Loaded: {image_path} ({self.img_w}x{self.img_h}), Focal Length: {self.f_pixels:.1f} px")

        # State variables
        self.crop_box = None
        self.curves = []
        self.curve_artists = []
        self.surface_artist = None
        self.sphere_artist = None
        self.scatter_artist = None
        self.optimized_cloud = None
        self.texture_mode = True
        self.wireframe_mode = False
        self.rs = None
        self.key_cid = None

        # Precomputed 3D data for interactive toggles
        self.X_3D = None
        self.Y_3D = None
        self.Z_3D = None
        self.facecolors_texture = None
        self.facecolors_viridis = None
        self.curve_3d_pts = []

        # Setup 2D Selection Window
        self.fig = plt.figure(figsize=(12, 8))
        self.setup_selection_gui()

        if self.auto_run:
            # Process entire image automatically
            self.crop_box = (0, 0, self.img_w, self.img_h)
            self.start_pipeline()
        else:
            plt.show()

    def setup_selection_gui(self):
        """Sets up the initial 2D view with RectangleSelector and run buttons."""
        self.fig.clf()
        self.ax_img = self.fig.add_axes([0.05, 0.15, 0.70, 0.75])
        self.ax_img.imshow(self.gray_img, cmap='gray')
        self.ax_img.set_title("1. Drag rectangle to crop paper (Optional) -> 2. Press [Enter] or click 'Run GD'",
                               fontsize=11, fontweight='bold')
        self.ax_img.axis('on')

        # Interactive rectangle selector
        self.rs = RectangleSelector(
            self.ax_img, self.on_crop_select,
            useblit=True, button=[1], interactive=True,
            props=dict(facecolor='cyan', edgecolor='blue', alpha=0.2, fill=True)
        )

        # Status text banner
        self.status_text = self.fig.text(
            0.05, 0.05,
            f"Image: {os.path.basename(self.image_path)} | Size: {self.img_w}x{self.img_h} | Ready. Select ROI or press Enter.",
            fontsize=10, style='italic'
        )

        # Control buttons
        self.ax_btn_run = self.fig.add_axes([0.78, 0.65, 0.18, 0.08])
        self.btn_run = Button(self.ax_btn_run, 'Run Gradient Descent\n[Enter]', color='#4CAF50', hovercolor='#45a049')
        self.btn_run.on_clicked(lambda event: self.start_pipeline())

        self.ax_btn_reset = self.fig.add_axes([0.78, 0.54, 0.18, 0.06])
        self.btn_reset = Button(self.ax_btn_reset, 'Reset Crop', color='#f0f0f0', hovercolor='#e0e0e0')
        self.btn_reset.on_clicked(self.on_reset_crop)

        # Key press handler
        self.key_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.draw_idle()

    def on_crop_select(self, eclick, erelease):
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        self.crop_box = (
            max(0, min(x1, x2)),
            max(0, min(y1, y2)),
            min(self.img_w, max(x1, x2)),
            min(self.img_h, max(y1, y2))
        )
        x_min, y_min, x_max, y_max = self.crop_box
        w, h = x_max - x_min, y_max - y_min
        self.status_text.set_text(f"Selected ROI: {w}x{h} px from ({x_min}, {y_min}). Press [Enter] to run GD.")
        self.fig.canvas.draw_idle()

    def on_reset_crop(self, event):
        self.crop_box = None
        self.status_text.set_text("Crop reset. Full image will be used. Press [Enter] to run GD.")
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key in ['enter', 'return']:
            self.start_pipeline()

    def start_pipeline(self):
        """Extracts curves, runs gradient descent, and plots 3D embeddings."""
        # Cleanly disconnect 2D selection tools and event callbacks so they do not interfere with 3D orbit/rotation
        if self.rs is not None:
            self.rs.set_active(False)
            self.rs.disconnect_events()
            self.rs = None

        if self.key_cid is not None:
            self.fig.canvas.mpl_disconnect(self.key_cid)
            self.key_cid = None

        # Reset any toolbar pan/zoom modes
        if self.fig.canvas.toolbar is not None and getattr(self.fig.canvas.toolbar, 'mode', None):
            if self.fig.canvas.toolbar.mode == 'zoom rect':
                self.fig.canvas.toolbar.zoom()
            elif self.fig.canvas.toolbar.mode == 'pan/zoom':
                self.fig.canvas.toolbar.pan()

        # 1. Determine ROI bounds
        if self.crop_box is not None:
            x_min, y_min, x_max, y_max = self.crop_box
            if (x_max - x_min) < 30 or (y_max - y_min) < 30:
                x_min, y_min, x_max, y_max = 0, 0, self.img_w, self.img_h
        else:
            x_min, y_min, x_max, y_max = 0, 0, self.img_w, self.img_h

        cropped_img = self.gray_img[y_min:y_max, x_min:x_max]
        roi_w, roi_h = x_max - x_min, y_max - y_min

        print(f"\n{'='*60}")
        print(f"Step 1: Curve Detection on ROI [{x_min}:{x_max}, {y_min}:{y_max}] ({roi_w}x{roi_h} px)")
        print(f"{'='*60}")
        self.ax_img.set_title("Processing: Scanning curves & fitting...", fontsize=11, color='blue')
        self.status_text.set_text("Scanning curves using LineDetector... Please wait.")
        self.fig.canvas.draw()
        plt.pause(0.01)

        # 2. Line detection
        blurred = cv.GaussianBlur(cropped_img, (5, 5), 0)
        cleaned = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 15)

        detec = detector.LineDetector(width=2, height=10, step=5)
        raw_lines = detec.findall_lines(cleaned)
        print(f"Detected {len(raw_lines)} candidate line traces.")

        # Discard extreme boundary lines if full image was used
        is_full_img = (x_min == 0 and y_min == 0 and x_max == self.img_w and y_max == self.img_h)
        margin = 15

        filtered_lines = []
        for line in raw_lines:
            pts = np.array(line)  # (row, col)
            if is_full_img:
                if (np.any(pts[:, 0] <= margin) or np.any(pts[:, 0] >= roi_h - margin) or
                    np.any(pts[:, 1] <= margin) or np.any(pts[:, 1] >= roi_w - margin)):
                    continue
            filtered_lines.append(line)

        # Sort by line length and take top structural curves
        filtered_lines.sort(key=len, reverse=True)
        min_len = 20 if len(filtered_lines) >= 10 else 10
        valid_lines = [l for l in filtered_lines if len(l) >= min_len]

        # Limit to top 15 curves to keep autograd fast and well-regularized
        selected_lines = valid_lines[:15] if len(valid_lines) > 15 else valid_lines

        if len(selected_lines) == 0:
            print("Warning: No valid lines found in selected region! Falling back to raw lines.")
            selected_lines = raw_lines[:5]

        if len(selected_lines) == 0:
            self.ax_img.set_title("Error: No lines detected. Please select another ROI.", fontsize=11, color='red')
            self.status_text.set_text("Error: No lines detected in this area. Select another box.")
            self.fig.canvas.draw_idle()
            return

        print(f"Selected {len(selected_lines)} prominent curves for optimization.")

        # Trace curves and shift origin to center of true image
        self.curves = []
        for line in selected_lines:
            # pt is (row, col) in crop -> convert to global (x, y) then center
            centered_line = [(pt[1] + x_min - self.img_w / 2.0, pt[0] + y_min - self.img_h / 2.0) for pt in line]
            self.curves.append(detector.Curve(centered_line, deg=2, bin_size=0.13))

        # 3. Gradient Descent
        print(f"\n{'='*60}")
        print(f"Step 2: PyTorch Gradient Descent Optimization ({self.steps} steps)")
        print(f"{'='*60}")
        self.status_text.set_text(f"Running PyTorch Gradient Descent ({self.steps} steps)...")
        self.fig.canvas.draw()
        plt.pause(0.01)

        T = np.linspace(0, 1, 50)
        t_start = time.time()

        # Adaptive step function with smooth decay and live console logging
        def step_fn(i):
            return AdaptiveStep(i, self.steps, base_delta=0.012)

        self.optimized_cloud = run_gradient_descent(
            T, self.curves, f=self.f_pixels, num_points=64,
            step=step_fn, steps=self.steps
        )
        t_elapsed = time.time() - t_start
        print(f"Gradient Descent finished in {t_elapsed:.2f}s.")
        print(f"Optimized rho range: [{self.optimized_cloud[:, 2].min().item():.3f}, {self.optimized_cloud[:, 2].max().item():.3f}]")

        # 4. Construct 3D Representations
        print(f"\n{'='*60}")
        print("Step 3: Constructing 3D Embeddings (rho(x).x)")
        print(f"{'='*60}")

        # --- Representation A: Original 3D Embedding of the Paper ---
        # Discretize the paper ROI in image coordinates
        grid_res = 50
        x_grid = np.linspace(x_min - self.img_w / 2.0, x_max - self.img_w / 2.0, grid_res)
        y_grid = np.linspace(y_min - self.img_h / 2.0, y_max - self.img_h / 2.0, grid_res)
        X_img, Y_img = np.meshgrid(x_grid, y_grid)

        # Ray distance and stereographic projection
        R_img = np.sqrt(X_img**2 + Y_img**2 + self.f_pixels**2)
        U_grid = X_img / (R_img + self.f_pixels)
        V_grid = Y_img / (R_img + self.f_pixels)

        rho_paper, _, _, _, _, _ = surface_fit(U_grid, V_grid, self.optimized_cloud, deg=2, bin_size=0.5)
        rho_paper_np = rho_paper.detach().cpu().numpy()

        # Physical 3D embedding: P = rho(x) * x = (rho / R) * (X_img, Y_img, f)
        W_paper = rho_paper_np / R_img
        self.X_3D = W_paper * X_img
        self.Y_3D = W_paper * Y_img
        self.Z_3D = W_paper * self.f_pixels

        # Build facecolors for texture mapping (full RGB color if available)
        cropped_rgb = self.img_rgb[y_min:y_max, x_min:x_max]
        resized_texture = cv.resize(cropped_rgb, (grid_res - 1, grid_res - 1)).astype(float) / 255.0
        self.facecolors_texture = np.clip(resized_texture, 0.0, 1.0)
        # Normalized depth for viridis colormap
        z_norm = (self.Z_3D - self.Z_3D.min()) / (self.Z_3D.max() - self.Z_3D.min() + 1e-8)
        self.facecolors_viridis = plt.cm.viridis(z_norm[:-1, :-1])

        # 3D curves lying directly on the paper surface
        self.curve_3d_pts = []
        for curve in self.curves:
            x_c, y_c = curve.point_at(T)
            x_c, y_c = np.asarray(x_c).ravel(), np.asarray(y_c).ravel()

            R_c = np.sqrt(x_c**2 + y_c**2 + self.f_pixels**2)
            u_c = x_c / (R_c + self.f_pixels)
            v_c = y_c / (R_c + self.f_pixels)

            rho_c, _, _, _, _, _ = surface_fit(u_c, v_c, self.optimized_cloud, deg=2, bin_size=0.5)
            w_c = rho_c.detach().cpu().numpy() / R_c

            x_3d_c = w_c * x_c
            y_3d_c = w_c * y_c
            z_3d_c = w_c * self.f_pixels
            self.curve_3d_pts.append((x_3d_c, y_3d_c, z_3d_c))

        # --- Representation B: Spherical S^2 Embedding: rho(x).x for x in S^2 ---
        # Discretize S^2 on a spherical cap matching field-of-view cone
        max_r = math.sqrt(max(abs(x_min - self.img_w/2), abs(x_max - self.img_w/2))**2 +
                          max(abs(y_min - self.img_h/2), abs(y_max - self.img_h/2))**2)
        phi_max = min(math.atan2(max_r, self.f_pixels) * 1.15, math.pi / 3.0)

        theta_grid = np.linspace(0, 2 * np.pi, 60)
        phi_grid = np.linspace(0, phi_max, 30)
        Theta_s, Phi_s = np.meshgrid(theta_grid, phi_grid)

        # Unit sphere vector x in S^2
        X_s = np.sin(Phi_s) * np.cos(Theta_s)
        Y_s = np.sin(Phi_s) * np.sin(Theta_s)
        Z_s = np.cos(Phi_s)

        # Stereographic coordinates
        U_s = X_s / (1.0 + Z_s)
        V_s = Y_s / (1.0 + Z_s)

        rho_s, _, _, _, _, _ = surface_fit(U_s, V_s, self.optimized_cloud, deg=2, bin_size=0.5)
        rho_s_np = rho_s.detach().cpu().numpy()

        # Plot rho(x) * x for each x in S^2
        self.Xs_3D = rho_s_np * X_s
        self.Ys_3D = rho_s_np * Y_s
        self.Zs_3D = rho_s_np * Z_s

        # Control cloud mapped back to S^2
        cloud_np = self.optimized_cloud.detach().cpu().numpy()
        u_c, v_c, rho_cloud = cloud_np[:, 0], cloud_np[:, 1], cloud_np[:, 2]
        denom = 1.0 + u_c**2 + v_c**2
        xc_s = 2.0 * u_c / denom
        yc_s = 2.0 * v_c / denom
        zc_s = (1.0 - u_c**2 - v_c**2) / denom
        self.cloud_pts_3d = (rho_cloud * xc_s, rho_cloud * yc_s, rho_cloud * zc_s)

        # 5. Render 3D Dashboard
        self.render_3d_dashboard()

    def render_3d_dashboard(self):
        """Clears the 2D figure and renders the dual 3D visualization dashboard."""
        self.fig.clf()

        # Subplot 1: Original 3D Embedding of the Paper
        self.ax1 = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.surface_artist = self.ax1.plot_surface(
            self.X_3D, self.Y_3D, self.Z_3D,
            facecolors=self.facecolors_texture if self.texture_mode else self.facecolors_viridis,
            shade=False, alpha=0.92,
            edgecolor='black' if self.wireframe_mode else 'none',
            linewidth=0.5 if self.wireframe_mode else 0.0
        )

        self.curve_artists = []
        for x_c, y_c, z_c in self.curve_3d_pts:
            line_artist, = self.ax1.plot(x_c, y_c, z_c, color='red', linewidth=2.5, zorder=5)
            self.curve_artists.append(line_artist)

        self.ax1.set_title("Original 3D Embedding of Paper: $\\mathbf{P} = \\rho(x) \\cdot x$",
                           fontsize=11, fontweight='bold', pad=10)
        self.ax1.set_xlabel("X (Width)")
        self.ax1.set_ylabel("Y (Height)")
        self.ax1.set_zlabel("Depth (Z)")
        self.ax1.view_init(elev=25, azim=-65)

        # Subplot 2: S^2 Spherical Embedding
        self.ax2 = self.fig.add_subplot(1, 2, 2, projection='3d')
        self.sphere_artist = self.ax2.plot_surface(
            self.Xs_3D, self.Ys_3D, self.Zs_3D,
            cmap='plasma', alpha=0.82,
            edgecolor='black' if self.wireframe_mode else 'none',
            linewidth=0.4 if self.wireframe_mode else 0.0
        )

        # Plot control points on S^2
        px, py, pz = self.cloud_pts_3d
        self.scatter_artist = self.ax2.scatter(
            px, py, pz, color='black', s=18, alpha=0.9, label='GD Control Cloud'
        )

        self.ax2.set_title("Deformation Field on $S^2$: $\\rho(x) \\cdot x$ for $x \\in S^2$",
                           fontsize=11, fontweight='bold', pad=10)
        self.ax2.set_xlabel("X")
        self.ax2.set_ylabel("Y")
        self.ax2.set_zlabel("Z")
        self.ax2.view_init(elev=30, azim=-60)
        self.ax2.legend(loc='upper right')

        # Interactive controls (CheckButtons)
        self.ax_check = self.fig.add_axes([0.02, 0.02, 0.16, 0.10])
        self.check = CheckButtons(
            self.ax_check,
            ['Show Curves', 'Paper Texture', 'Wireframe'],
            [True, self.texture_mode, self.wireframe_mode]
        )
        self.check.on_clicked(self.on_checkbox_toggle)

        # Information text
        rho_min = self.optimized_cloud[:, 2].min().item()
        rho_max = self.optimized_cloud[:, 2].max().item()
        info = (f"Image: {os.path.basename(self.image_path)} | Focal: {self.f_pixels:.0f} px | "
                f"Curves: {len(self.curves)} | rho in [{rho_min:.3f}, {rho_max:.3f}] | GD Steps: {self.steps}")
        self.fig.text(0.20, 0.03, info, fontsize=10, style='italic',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='#eef2f7', alpha=0.8))

        self.fig.suptitle(f"PaperCorrect - 3D Paper Shape Reconstruction ({os.path.basename(self.image_path)})",
                          fontsize=13, fontweight='bold', y=0.98)

        if self.output_path:
            plt.savefig(self.output_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D plot to: {self.output_path}")

        self.fig.canvas.draw_idle()

    def on_checkbox_toggle(self, label):
        """Handles toggling of curves, texture, and wireframe."""
        if label == 'Show Curves':
            visible = not self.curve_artists[0].get_visible() if self.curve_artists else False
            for artist in self.curve_artists:
                artist.set_visible(visible)
        elif label == 'Paper Texture':
            self.texture_mode = not self.texture_mode
            self.surface_artist.remove()
            self.surface_artist = self.ax1.plot_surface(
                self.X_3D, self.Y_3D, self.Z_3D,
                facecolors=self.facecolors_texture if self.texture_mode else self.facecolors_viridis,
                shade=False, alpha=0.92,
                edgecolor='black' if self.wireframe_mode else 'none',
                linewidth=0.5 if self.wireframe_mode else 0.0
            )
        elif label == 'Wireframe':
            self.wireframe_mode = not self.wireframe_mode
            edgecolor = 'black' if self.wireframe_mode else 'none'
            lw_surf = 0.5 if self.wireframe_mode else 0.0
            lw_sph = 0.4 if self.wireframe_mode else 0.0
            self.surface_artist.set_edgecolor(edgecolor)
            self.surface_artist.set_linewidth(lw_surf)
            self.sphere_artist.set_edgecolor(edgecolor)
            self.sphere_artist.set_linewidth(lw_sph)

        self.fig.canvas.draw_idle()


def main():
    default_img = "curve.jpeg"
    image_files = sorted(glob.glob("*.jpeg") + glob.glob("*.jpg") + glob.glob("*.png"))
    # Filter out generated test outputs
    image_files = [f for f in image_files if not f.startswith("test_") and not f.startswith("output") and not f.startswith("paper_3d_")]

    selected_path = None
    auto_run = False
    output_save = "paper_3d_embedding.png"

    # Command line argument handling
    for arg in sys.argv[1:]:
        if arg in ['--auto', '--no-crop']:
            auto_run = True
        elif os.path.exists(arg):
            selected_path = arg

    if selected_path is None:
        print("Available images in directory:")
        for i, file in enumerate(image_files):
            marker = " (Default)" if file == default_img else ""
            print(f"  [{i}] {file}{marker}")

        try:
            choice = input(f"\nEnter number or filename to load (Press Enter for '{default_img}'): ").strip()
        except EOFError:
            choice = ""

        if choice:
            if choice.isdigit() and 0 <= int(choice) < len(image_files):
                selected_path = image_files[int(choice)]
            elif os.path.exists(choice):
                selected_path = choice
            else:
                print(f"Warning: '{choice}' not found. Falling back to default: {default_img}")
                selected_path = default_img
        else:
            selected_path = default_img

    if not os.path.exists(selected_path):
        # Fallback to any existing image
        if image_files:
            selected_path = image_files[0]
        else:
            print(f"Error: Could not find any image to load.")
            sys.exit(1)

    print(f"\nLaunching PaperCorrect Front-End with: {selected_path}")
    print("Instructions:")
    print("  1. In the window, drag a rectangle over the paper region (or leave blank for full image).")
    print("  2. Press [Enter] or click 'Run Gradient Descent' to start.")
    print("  3. After GD completes, inspect the 3D embedding and S^2 spherical deformation.\n")

    app = GDFrontEnd(selected_path, auto_run=auto_run, output_path=output_save)


if __name__ == "__main__":
    main()

