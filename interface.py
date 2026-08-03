import os
import glob
import basics
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, CheckButtons

class KinematicsGUI:
    def __init__(self, image_path):
        self.f = 2912.0
        self.T = np.linspace(0, 1, 1000)
        
        # 1. Load Image
        self.img = cv.imread(image_path)
        self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.img_height, self.img_width = self.gray_img.shape
        self.true_center_x = self.img_width / 2.0
        self.true_center_y = self.img_height / 2.0
        
        # 2. Setup Figure Layout
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_img = self.fig.add_axes([0.05, 0.1, 0.7, 0.8])
        self.ax_img.imshow(self.gray_img, cmap='gray')
        self.ax_img.set_title("1. Drag to Crop -> Press Enter")
        
        # 3. Setup Interactive Selectors
        self.rs = RectangleSelector(self.ax_img, self.on_crop_select,
                                    useblit=True, button=[1], interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('pick_event', self.on_line_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # 4. Setup Toggle Checkboxes
        self.ax_check = self.fig.add_axes([0.78, 0.45, 0.20, 0.2])
        self.check = CheckButtons(
            self.ax_check, 
            ['Show Velocity', 'Show Acceleration', 'Scale by Magnitude'], 
            [True, True, False]
        )
        self.check.on_clicked(self.on_checkbox_toggle)
        self.show_vel = True
        self.show_acc = True
        self.scale_by_mag = False
        
        # State variables
        self.crop_box = None
        self.lines = []
        self.line_artists = []
        self.current_energy = 0.0
        self.last_mouse_event = None
        
        # Active Kinematics Data
        self.active_x = None
        self.active_y = None
        self.active_vx = None
        self.active_vy = None
        self.active_ax = None
        self.active_ay = None
        
        self.active_max_v = 1.0
        self.active_max_a = 1.0
        
        # FIX: Add a dedicated line artist just for the smooth fit!
        self.fit_artist, = self.ax_img.plot([], [], color='red', linewidth=3, zorder=4)
        
        # Initialize quivers
        self.q_vel = self.ax_img.quiver([0], [0], [0], [0], color='green', angles='xy', scale_units='xy', scale=1, width=0.005, label='Velocity', zorder=5)
        self.q_acc = self.ax_img.quiver([0], [0], [0], [0], color='magenta', angles='xy', scale_units='xy', scale=1, width=0.005, label='Acceleration', zorder=5)
        self.ax_img.legend(loc='upper right')
        
        self.q_vel.set_UVC([0], [0])
        self.q_acc.set_UVC([0], [0])
        
        plt.show()

    def on_crop_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.crop_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def on_key_press(self, event):
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
            
            for line in local_lines:
                global_line = [(pt[1] + x_min, pt[0] + y_min) for pt in line]
                if len(global_line) > 10:
                    self.lines.append(global_line)
                    pts = np.array(global_line)
                    line_obj, = self.ax_img.plot(pts[:, 0], pts[:, 1], color='blue', alpha=0.5, picker=5, linewidth=2)
                    self.line_artists.append(line_obj)
            
            # Re-initialize visual elements
            self.fit_artist, = self.ax_img.plot([], [], color='red', linewidth=3, zorder=4)
            self.q_vel = self.ax_img.quiver([0], [0], [0], [0], color='green', angles='xy', scale_units='xy', scale=1, width=0.005, label='Velocity', zorder=5)
            self.q_acc = self.ax_img.quiver([0], [0], [0], [0], color='magenta', angles='xy', scale_units='xy', scale=1, width=0.005, label='Acceleration', zorder=5)
            self.q_vel.set_UVC([0], [0])
            self.q_acc.set_UVC([0], [0])
            self.ax_img.legend(loc='upper right')
            
            self.ax_img.set_title("2. Click a blue line to select it")
            self.fig.canvas.draw()

    def on_line_pick(self, event):
        try:
            line_idx = self.line_artists.index(event.artist)
        except ValueError:
            return
            
        for artist in self.line_artists:
            artist.set_color('blue')
            artist.set_alpha(0.5)
            
        # Highlight raw line in cyan
        self.line_artists[line_idx].set_color('cyan')
        self.line_artists[line_idx].set_alpha(0.8)
        
        raw_line = self.lines[line_idx]
        (x, y), (vx, vy), (ax, ay) = basics.curve_fit(self.T, raw_line, bin_size=0.13, deg=2)
        
        self.active_x = x
        self.active_y = y
        self.active_vx = vx
        self.active_vy = vy
        self.active_ax = ax
        self.active_ay = ay
        
        self.active_max_v = np.max(np.hypot(vx, vy))
        self.active_max_a = np.max(np.hypot(ax, ay))
        
        # PLOT THE SMOOTH FIT directly on top!
        self.fit_artist.set_data(x, y)
        
        x_c = x - self.true_center_x
        y_c = y - self.true_center_y
        
        u = x_c**2 + y_c**2 + self.f**2
        A = x_c * vx + y_c * vy
        B = vx**2 + vy**2 + x_c * ax + y_c * ay
        
        w = u**(-0.5)
        w_dt = -(u**(-1.5)) * A
        w_ddt = 3 * (u**(-2.5)) * (A**2) - (u**(-1.5)) * B
        
        self.current_energy = basics.compute_projective_bending_energy(self.T, x_c, y_c, vx, vy, ax, ay, w, w_dt, w_ddt, self.f)
        self.ax_img.set_title(f"Energy: {self.current_energy:.2f} | Hover to view kinematics")
        
        if self.last_mouse_event:
            self.on_mouse_move(self.last_mouse_event)
        self.fig.canvas.draw_idle()

    def on_mouse_move(self, event):
        if not event.inaxes == self.ax_img or self.active_x is None:
            return
            
        self.last_mouse_event = event
        mouse_x, mouse_y = event.xdata, event.ydata
        
        distances_sq = (self.active_x - mouse_x)**2 + (self.active_y - mouse_y)**2
        nearest_idx = np.argmin(distances_sq)
        
        px, py = self.active_x[nearest_idx], self.active_y[nearest_idx]
        u_v, v_v = self.active_vx[nearest_idx], self.active_vy[nearest_idx]
        u_a, v_a = self.active_ax[nearest_idx], self.active_ay[nearest_idx]
        
        mag_v = np.hypot(u_v, v_v)
        mag_a = np.hypot(u_a, v_a)
        self.ax_img.set_title(f"Energy: {self.current_energy:.1f} | |Vel|: {mag_v:.0f}, |Acc|: {mag_a:.0f}")
        
        fixed_disp_scale = 100.0
        max_disp_scale = 200.0 
        
        if self.scale_by_mag:
            v_scale = (mag_v / (self.active_max_v + 1e-12)) * max_disp_scale
            a_scale = (mag_a / (self.active_max_a + 1e-12)) * max_disp_scale
        else:
            v_scale = fixed_disp_scale
            a_scale = fixed_disp_scale
        
        u_v_disp = (u_v / (mag_v + 1e-12)) * v_scale
        v_v_disp = (v_v / (mag_v + 1e-12)) * v_scale
        
        u_a_disp = (u_a / (mag_a + 1e-12)) * a_scale
        v_a_disp = (v_a / (mag_a + 1e-12)) * a_scale
        
        if self.show_vel:
            self.q_vel.set_offsets(np.c_[[px], [py]])
            self.q_vel.set_UVC([u_v_disp], [v_v_disp])
        else:
            self.q_vel.set_UVC([0], [0])
            
        if self.show_acc:
            self.q_acc.set_offsets(np.c_[[px], [py]])
            self.q_acc.set_UVC([u_a_disp], [v_a_disp])
        else:
            self.q_acc.set_UVC([0], [0])
            
        self.fig.canvas.draw_idle()

    def on_checkbox_toggle(self, label):
        if label == 'Show Velocity':
            self.show_vel = not self.show_vel
        elif label == 'Show Acceleration':
            self.show_acc = not self.show_acc
        elif label == 'Scale by Magnitude':
            self.scale_by_mag = not self.scale_by_mag
            
        if self.last_mouse_event and self.last_mouse_event.inaxes == self.ax_img:
            self.on_mouse_move(self.last_mouse_event)
        else:
            if not self.show_vel:
                self.q_vel.set_UVC([0], [0])
            if not self.show_acc:
                self.q_acc.set_UVC([0], [0])
            self.fig.canvas.draw_idle()

if __name__ == "__main__":
    default_img = "crump_uncropped.jpeg"
    image_files = glob.glob("*.jpeg") + glob.glob("*.jpg") + glob.glob("*.png")
    image_files.sort()
    
    print("Available images in directory:")
    for i, file in enumerate(image_files):
        marker = " (Default)" if file == default_img else ""
        print(f"  [{i}] {file}{marker}")
        
    choice = input(f"\nEnter the number or filename of the image to load (Press Enter for '{default_img}'): ").strip()
    
    selected_path = default_img
    if choice:
        if choice.isdigit() and 0 <= int(choice) < len(image_files):
            selected_path = image_files[int(choice)]
        elif os.path.exists(choice):
            selected_path = choice
        else:
            print(f"Warning: Could not find '{choice}'. Falling back to default.")
            
    if not os.path.exists(selected_path):
        print(f"Error: The file '{selected_path}' does not exist in this directory.")
    else:
        print(f"Launching Kinematics Explorer with: {selected_path}\n")
        app = KinematicsGUI(selected_path)
