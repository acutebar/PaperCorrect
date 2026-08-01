import cv2 as cv
import numpy as np
import glob
import os

# Import your actual functions and classes
from basics import LineDetector, paper_clean_fast, display_lines

class LineInspectorUI:
    def __init__(self, gray_img, gauss_img, custom_img, all_lines_img, lines):
        self.img_gray = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
        self.img_gauss = cv.cvtColor(gauss_img, cv.COLOR_GRAY2BGR)
        self.img_custom = cv.cvtColor(custom_img, cv.COLOR_GRAY2BGR)
        
        if len(all_lines_img.shape) == 2:
            self.img_all_lines = cv.cvtColor(all_lines_img, cv.COLOR_GRAY2BGR)
        else:
            self.img_all_lines = all_lines_img.copy()
        
        # Swap (row, col) from your detector to (x, y) for OpenCV
        self.lines = []
        for line in lines:
            swapped_line = [(pt[1], pt[0]) for pt in line]
            self.lines.append(swapped_line)
        
        self.current_view = 'interactive' 
        self.mouse_pos = (0, 0)
        self.hovered_idx = -1
        self.locked_idx = -1
        
        self.window_name = "Line Inspector"
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.window_name, 1200, 800)
        cv.setMouseCallback(self.window_name, self._mouse_event)

    def _mouse_event(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv.EVENT_MOUSEMOVE:
            if self.current_view == 'interactive':
                self.hovered_idx = self._get_closest_line_idx(x, y)
        elif event == cv.EVENT_LBUTTONDOWN:
            if self.current_view == 'interactive' and self.hovered_idx != -1:
                if self.locked_idx == self.hovered_idx:
                    self.locked_idx = -1
                else:
                    self.locked_idx = self.hovered_idx

    def _get_closest_line_idx(self, mx, my, distance_threshold=15):
        min_dist = float('inf')
        closest_idx = -1
        mouse_pt = np.array([mx, my])
        
        for i, line in enumerate(self.lines):
            for j in range(len(line) - 1):
                pt1 = np.array(line[j])
                pt2 = np.array(line[j+1])
                
                line_vec = pt2 - pt1
                pt_vec = mouse_pt - pt1
                line_len_sq = np.dot(line_vec, line_vec)
                
                if line_len_sq == 0:
                    dist = np.linalg.norm(pt_vec)
                else:
                    t = max(0, min(1, np.dot(pt_vec, line_vec) / line_len_sq))
                    proj = pt1 + t * line_vec
                    dist = np.linalg.norm(mouse_pt - proj)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    
        return closest_idx if min_dist <= distance_threshold else -1

    def _draw_hud(self, img):
        instructions = [
            "[1] Original View",
            "[2] Gaussian Clean",
            "[3] Custom Clean",
            "[4] All Lines (display_lines)",
            "[5] Interactive Hover Mode",
            "Click a line to Lock/Unlock"
        ]
        
        mode_map = {'gray': 0, 'gauss': 1, 'custom': 2, 'all_lines': 3, 'interactive': 4}
        active_idx = mode_map.get(self.current_view, 4)
        
        for i, text in enumerate(instructions):
            color = (0, 255, 255) if i == active_idx else (200, 200, 200)
            cv.putText(img, text, (20, 40 + (i * 30)), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _draw_line(self, img, line_points, color, thickness=3):
        pts = np.array(line_points, np.int32).reshape((-1, 1, 2))
        cv.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

    def run(self):
        while True:
            if self.current_view == 'gray':
                display_img = self.img_gray.copy()
            elif self.current_view == 'gauss':
                display_img = self.img_gauss.copy()
            elif self.current_view == 'custom':
                display_img = self.img_custom.copy()
            elif self.current_view == 'all_lines':
                display_img = self.img_all_lines.copy() 
            else:
                display_img = cv.addWeighted(self.img_gray, 0.4, np.zeros_like(self.img_gray), 0, 0)
                
                if self.locked_idx != -1:
                    self._draw_line(display_img, self.lines[self.locked_idx], color=(0, 255, 0), thickness=4)
                elif self.hovered_idx != -1:
                    self._draw_line(display_img, self.lines[self.hovered_idx], color=(0, 0, 255), thickness=3)

            self._draw_hud(display_img)
            cv.imshow(self.window_name, display_img)
            
            key = cv.waitKey(15) & 0xFF
            
            if key == 27: 
                break
            elif key == ord('1'):
                self.current_view = 'gray'
            elif key == ord('2'):
                self.current_view = 'gauss'
            elif key == ord('3'):
                self.current_view = 'custom'
            elif key == ord('4'):
                self.current_view = 'all_lines'
            elif key == ord('5'):
                self.current_view = 'interactive'
                
        cv.destroyAllWindows()


def select_image():
    """Scans the current directory for images and prompts the user to select one."""
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(ext))
        # Also check uppercase extensions
        image_files.extend(glob.glob(ext.upper()))
        
    if not image_files:
        print("No image files found in the current directory.")
        return None

    print("\n--- Available Images ---")
    for i, file in enumerate(image_files):
        print(f"[{i}] {file}")
        
    while True:
        try:
            choice = input(f"\nSelect an image (0 - {len(image_files)-1}) or 'q' to quit: ")
            if choice.lower() == 'q':
                return None
            choice = int(choice)
            if 0 <= choice < len(image_files):
                return image_files[choice]
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_cropped_grayscale(image_path):
    """Loads an image, allows the user to crop it via GUI, and returns the grayscale crop."""
    img = cv.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Resize for the crop window if the image is massive, to ensure it fits on screen
    screen_height = 800
    h, w = img.shape[:2]
    scale = 1.0
    if h > screen_height:
        scale = screen_height / h
        display_img = cv.resize(img, (int(w * scale), int(h * scale)))
    else:
        display_img = img.copy()

    print("\n--- Cropping Instructions ---")
    print("1. Click and drag to select the region you want to process.")
    print("2. Press SPACE or ENTER to confirm the crop.")
    print("3. Press 'c' to cancel and use the whole image.")
    
    # Let user select ROI
    roi = cv.selectROI("Select Crop Region (SPACE to confirm)", display_img, showCrosshair=True, fromCenter=False)
    cv.destroyWindow("Select Crop Region (SPACE to confirm)")
    
    x, y, w_box, h_box = roi
    
    # If user cancelled or selected nothing, return full grayscale image
    if w_box == 0 or h_box == 0:
        print("No crop selected. Using the entire image.")
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
    # Scale coordinates back up if we resized the display image
    real_x = int(x / scale)
    real_y = int(y / scale)
    real_w = int(w_box / scale)
    real_h = int(h_box / scale)
    
    cropped_img = img[real_y:real_y+real_h, real_x:real_x+real_w]
    return cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)


if __name__ == "__main__":
    # 1. Let user select the image
    selected_image_path = select_image()
    
    if selected_image_path:
        # 2. Let user crop the image (returns grayscale)
        gray_img = get_cropped_grayscale(selected_image_path)
        
        if gray_img is not None:
            # 3. Pre-process the cropped grayscale image
            blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)
            cleaned_img = cv.adaptiveThreshold(
                blurred_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 15
            )
            
            my_cleaned_img = paper_clean_fast(gray_img) 
            
            # 4. RUN REAL DETECTION 
            print("\nRunning line detection (this may take a moment)...")
            detector = LineDetector(width=2, height=10, step=5) 
            lines = detector.findall_lines(cleaned_img)
            
            all_lines_img = display_lines(cleaned_img, lines, thickness=2)
            
            print(f"Found {len(lines)} lines. Launching UI...")
            
            # 5. Launch Interactive UI
            app = LineInspectorUI(gray_img, cleaned_img, my_cleaned_img, all_lines_img, lines)
            app.run()
