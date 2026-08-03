import basics
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Slider

class PaperFlatteningGUI:
    def __init__(self, image_path):
        self.f = 2912.0
        self.T = np.linspace(0, 1, 1000)
        
        # 1. Load Image
        self.img = cv.imread(image_path)
        self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.img_height, self.img_width = self.gray_img.shape
        self.true_center_x = self.img_width / 2.0
        self.true_center_y = self.img_height / 2.0
        
        # Precompute the 3D Grid for the paper surface
        # 50x50 is a good balance between rendering speed and visual fidelity
        self.grid_res = 50
        xx = np.linspace(0, self.img_width - 1, self.grid_res)
        yy = np.linspace(0, self.img_height - 1, self.grid_res)
        self.X_grid, self.Y_grid = np.meshgrid(xx, yy)
        
        # Shift grid to optical center
        self.X_grid_c = self.X_grid - self.true_center_x
        self.Y_grid_c = self.Y_grid - self.true_center_y
        
        # Prepare the image texture for the faces of the 3D surface
        # A grid of NxN points has (N-1)x(N-1) faces
        tex_img = cv.resize(cv.cvtColor(self.img, cv.COLOR_BGR2RGB), (self.grid_res - 1, self.grid_res - 1))
        self.face_colors = tex_img / 255.0
        
        # 2. Setup Figure Layout
        self.fig = plt.figure(figsize=(16, 8))
        self.ax_img = self.fig.add_axes([0.05, 0.4, 0.4, 0.55])
        self.ax_3d = self.fig.add_axes([0.55, 0.4, 0.4, 0.55], projection='3d')
        
        self.ax_img.imshow(self.gray_img, cmap='gray')
        self.ax_img.set_title("1. Drag to Crop -> Press Enter")
        self.ax_3d.set_title("3D Embedding")
        
        # 3. Setup Interactive Selectors
        self.rs = RectangleSelector(self.ax_img, self.on_crop_select,
                                    useblit=True, button=[1], interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('pick_event', self.on_line_pick)
        
        # 4. Setup Rho Deformation Sliders (Gaussian Bump)
        self.ax_amp = self.fig.add_axes([0.15, 0.2, 0.7, 0.03])
        self.ax_cx = self.fig.add_axes([0.15, 0.15, 0.7, 0.03])
        self.ax_cy = self.fig.add_axes([0.15, 0.1, 0.7, 0.03])
        
        self.slider_amp = Slider(self.ax_amp, 'Bump Amplitude', -0.5, 2.0, valinit=0.0)
        self.slider_cx = Slider(self.ax_cx, 'Bump X Center', 0, self.img_width, valinit=self.true_center_x)
        self.slider_cy = Slider(self.ax_cy, 'Bump Y Center', 0, self.img_height, valinit=self.true_center_y)
        
        self.slider_amp.on_changed(self.update_view)
        self.slider_cx.on_changed(self.update_view)
        self.slider_cy.on_changed(self.update_view)
        
        # State variables
        self.crop_box = None
        self.lines = []
        self.line_artists = []
        self.selected_lines = {} # Dictionary to hold multiple selected lines
        
        plt.show()

    def on_crop_select(self, eclick, erelease):
        """Stores the crop coordinates."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.crop_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def on_key_press(self, event):
        """Runs line detection when Enter is pressed."""
        if event.key == 'enter' and self.crop_box is not None:
            self.ax_img.set_title("Processing...")
            self.fig.canvas.draw()
            
            x_min, y_min, x_max, y_max = self.crop_box
            cropped_img = self.gray_img[y_min:y_max, x_min:x_max]
            
            blurred = cv.GaussianBlur(cropped_img, (5, 5), 0)
            cleaned = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 15)
            
            detector = basics.LineDetector(width=2, height=10, step=5)
            local_lines = detector.findall_lines(cleaned)
            
            self.ax_img.clear()
            self.ax_img.imshow(self.gray_img, cmap='gray')
            
            self.lines = []
            self.line_artists = []
            self.selected_lines.clear()
            
            for line in local_lines:
                # FIX: Swap pt[0] and pt[1] to map (row, col) to (X, Y)
                global_line = [(pt[1] + x_min, pt[0] + y_min) for pt in line]
                
                if len(global_line) > 10:
                    self.lines.append(global_line)
                    pts = np.array(global_line)
                    # Plot with picker=5 so they can be clicked
                    line_obj, = self.ax_img.plot(pts[:, 0], pts[:, 1], color='blue', alpha=0.5, picker=5, linewidth=2)
                    self.line_artists.append(line_obj)
            
            self.ax_img.set_title("2. Click lines to toggle selection")
            self.fig.canvas.draw()

    def on_line_pick(self, event):
        """Toggles the selection of a detected line."""
        try:
            line_idx = self.line_artists.index(event.artist)
        except ValueError:
            return
            
        if line_idx in self.selected_lines:
            # Deselect the line
            del self.selected_lines[line_idx]
            self.line_artists[line_idx].set_color('blue')
            self.line_artists[line_idx].set_alpha(0.5)
        else:
            # Select the line and compute its base kinematics
            raw_line = self.lines[line_idx]
            (x, y), (vx, vy), (ax, ay) = basics.curve_fit(self.T, raw_line, bin_size=0.13, deg=2)
            
            x_c = x - self.true_center_x
            y_c = y - self.true_center_y
            
            self.selected_lines[line_idx] = (x_c, y_c, vx, vy, ax, ay, x, y)
            
            self.line_artists[line_idx].set_color('red')
            self.line_artists[line_idx].set_alpha(1.0)
            
        self.update_view(None)

    def update_view(self, val):
        """Dynamically deforms the paper and computes total energy."""
        amp = self.slider_amp.val
        cx = self.slider_cx.val - self.true_center_x
        cy = self.slider_cy.val - self.true_center_y
        sigma = 500.0
        
        self.ax_3d.clear()
        total_energy = 0.0
        
        # 1. Transform and plot the entire paper surface
        r_sq_grid = (self.X_grid_c - cx)**2 + (self.Y_grid_c - cy)**2
        rho_grid = 1.0 + amp * np.exp(-r_sq_grid / (2 * sigma**2))
        
        u_grid = self.X_grid_c**2 + self.Y_grid_c**2 + self.f**2
        w_grid = rho_grid * (u_grid**(-0.5))
        
        X_3d_surf = w_grid * self.X_grid_c
        Y_3d_surf = w_grid * self.Y_grid_c
        Z_3d_surf = w_grid * self.f
        
        self.ax_3d.plot_surface(
            X_3d_surf, Y_3d_surf, Z_3d_surf, 
            facecolors=self.face_colors, rstride=1, cstride=1, 
            alpha=0.8, shade=False
        )
        
        # 2. Process all selected curves and sum their energies
        for (x_c, y_c, vx, vy, ax, ay, raw_x, raw_y) in self.selected_lines.values():
            r_sq = (x_c - cx)**2 + (y_c - cy)**2
            rho = 1.0 + amp * np.exp(-r_sq / (2 * sigma**2))
            
            u = x_c**2 + y_c**2 + self.f**2
            w = rho * (u**(-0.5))
            
            w_dt = np.gradient(w, self.T)
            w_ddt = np.gradient(w_dt, self.T)
            
            energy = basics.compute_projective_bending_energy(
                self.T, x_c, y_c, vx, vy, ax, ay, w, w_dt, w_ddt, self.f
            )
            total_energy += energy
            
            # Draw the transformed curve on top of the surface
            X_3d_curve = w * x_c
            Y_3d_curve = w * y_c
            Z_3d_curve = w * self.f
            self.ax_3d.plot(X_3d_curve, Y_3d_curve, Z_3d_curve, color='red', linewidth=3)
        
        # 3. Update Titles and Axis scaling
        self.ax_img.set_title(f"3. Active Lines: {len(self.selected_lines)} | Deform Rho via sliders")
        self.ax_3d.set_title(f"Total Bending Energy: {total_energy:.4f}")
        
        # Keep the 3D aspect ratio proportional
        if np.ptp(X_3d_surf) > 0:
            self.ax_3d.set_box_aspect([np.ptp(X_3d_surf), np.ptp(Y_3d_surf), np.ptp(Z_3d_surf)])
            
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    app = PaperFlatteningGUI("crump_uncropped.jpeg")
