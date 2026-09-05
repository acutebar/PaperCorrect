import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math
from .edge_detector import get_disc_kernel

class LineDetector:
    def __init__(self, width=2, height=5, step=5, sweep=30):
        self.width = width
        self.height = height
        self.step = step
        self.sweep = sweep
        self.angles = np.arange(0, 360, step)
        
        self.templates = {}
        self.tips = {}
        
        self.max_radius = int(np.ceil(height)) + 2
        center = (self.max_radius, self.max_radius)
        
        for angle in self.angles:
            tilt = math.radians(angle)
            tip_r = height * math.cos(tilt)
            tip_c = -height * math.sin(tilt)
            
            mask = np.zeros((2 * self.max_radius + 1, 2 * self.max_radius + 1), dtype=np.uint8)
            cv.line(mask, center, 
                     (int(round(center[1] + tip_c)), int(round(center[0] + tip_r))), 
                     255, thickness=width)
            
            dy, dx = np.where(mask > 0)
            self.templates[angle] = (dy - center[0], dx - center[1])
            self.tips[angle] = (int(round(tip_r)), int(round(tip_c)))

    def get_valid_paths(self, padded_img, pivot, visited_mask, thresh_map):
        pr, pc = pivot
        threshold = thresh_map[pr, pc]
        
        means = []
        for angle in self.angles:
            dy, dx = self.templates[angle]
            mean = np.mean(padded_img[pr + dy, pc + dx])
            
            tr, tc = self.tips[angle]
            tip_r, tip_c = pr + tr, pc + tc
            
            is_visited = visited_mask[tip_r, tip_c]
            means.append((mean, angle, tip_r, tip_c, is_visited))
            
        paths = []
        n = len(means)
        neighbor_count = max(1, self.sweep // self.step)
        
        for i in range(n):
            curr_m = means[i][0]
            is_valley = True
            
            for j in range(1, neighbor_count + 1):
                prev_m = means[(i - j) % n][0]
                next_m = means[(i + j) % n][0]
                
                if curr_m >= prev_m or curr_m > next_m:
                    is_valley = False
                    break
                    
            if is_valley and curr_m < threshold and not means[i][4]:
                paths.append(means[i])
                
        return paths

    def mark_visited(self, visited_mask, p1, p2):
        pt1 = (int(p1[1]), int(p1[0]))
        pt2 = (int(p2[1]), int(p2[0]))
        cv.line(visited_mask, pt1, pt2, 1, thickness=self.width + 2)

    def line_detect(self, padded_img, start, visited_mask, thresh_map):
        paths = self.get_valid_paths(padded_img, start, visited_mask, thresh_map)
        if len(paths) != 1:
            return None
            
        line = [start]
        cv.circle(visited_mask, (int(start[1]), int(start[0])), self.width + 1, 1, -1)
        curr = start
        prev_angle = None
        base_angle = None
        
        while True:
            paths = self.get_valid_paths(padded_img, curr, visited_mask, thresh_map)
            if len(paths) == 0:
                break 
                
            best_path = min(paths, key=lambda x: x[0])
            next_move = (best_path[2], best_path[3])

            cur_angle = best_path[1]

            if prev_angle is not None:
                diff = abs((cur_angle - prev_angle + 180) % 360 - 180)
                total_diff = abs((cur_angle - base_angle + 180) % 360 - 180)
                if diff > 45:
                    break
                elif total_diff > 60:
                    break
            else:
                prev_angle = cur_angle
                base_angle = cur_angle

            prev_angle = cur_angle

            line.append(next_move)
            self.mark_visited(visited_mask, curr, next_move)
            curr = next_move
            
        return line

    def findall_lines(self, img):
        kernel = get_disc_kernel(self.height)
        img_float = img.astype(np.float32)
        
        # Precompute local means and stds globally
        local_mean = cv.filter2D(img_float, -1, kernel, borderType=cv.BORDER_REFLECT)
        local_mean_sq = cv.filter2D(img_float**2, -1, kernel, borderType=cv.BORDER_REFLECT)
        local_var = np.maximum(0, local_mean_sq - (local_mean**2))
        thresh_map = local_mean - (np.sqrt(local_var) * 0.5)

        # Pad variables to prevent bounds checking
        pad = self.max_radius
        padded_img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=255)
        padded_thresh = cv.copyMakeBorder(thresh_map, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)
        
        # Visited mask defaults to 1 (visited) outside the true image bounds
        visited_mask = np.ones(padded_img.shape, dtype=np.uint8)
        visited_mask[pad:-pad, pad:-pad] = 0
        
        lines = []
        dark_pixels = np.argwhere(img == 0)
        
        for r, c in dark_pixels:
            pr, pc = r + pad, c + pad
            if not visited_mask[pr, pc]:
                line_padded = self.line_detect(padded_img, (pr, pc), visited_mask, padded_thresh)
                if line_padded is not None and len(line_padded) > 1:
                    line = [(pr_val - pad, pc_val - pad) for (pr_val, pc_val) in line_padded]
                    lines.append(line)
                    
        return lines

