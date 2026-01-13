import cv2
import numpy as np
import time

class Visualizer:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.ema_naive = None
        self.ema_geometry = None
        self.alpha = 0.15 # Smoothing

    def smooth_gaze(self, gaze_point, mode="geometry"):
        if mode == "naive":
            if self.ema_naive is None: self.ema_naive = np.array(gaze_point)
            else: self.ema_naive = self.alpha * np.array(gaze_point) + (1 - self.alpha) * self.ema_naive
            return self.ema_naive
        else:
            if self.ema_geometry is None: self.ema_geometry = np.array(gaze_point)
            else: self.ema_geometry = self.alpha * np.array(gaze_point) + (1 - self.alpha) * self.ema_geometry
            return self.ema_geometry

    def update_heatmap(self, gaze_point):
        x, y = int(gaze_point[0]), int(gaze_point[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            cv2.circle(self.heatmap, (x, y), 20, 0.1, -1)
            self.heatmap *= 0.99

    def draw_eye_box(self, image, landmarks, color=(0, 255, 0)):
        """ Draws a GREEN rectangle around both eyes """
        h, w, _ = image.shape
        eye_indices = [33, 133, 362, 263, 159, 145, 386, 374]
        coords = []
        for idx in eye_indices:
            lm = landmarks[idx]
            coords.append([lm.x * w, lm.y * h])
        coords = np.array(coords)
        x1, y1 = np.min(coords, axis=0).astype(int)
        x2, y2 = np.max(coords, axis=0).astype(int)
        cv2.rectangle(image, (x1-15, y1-15), (x2+15, y2+15), (0, 255, 0), 2) # Force Green
        return image

    def draw_gaze_overlay(self, image, nasion_pixel, gaze_point, mode_label):
        """ Draws the red vector arrow with length limited to 30% of width """
        nx, ny = int(nasion_pixel[0]), int(nasion_pixel[1])
        gx, gy = int(gaze_point[0]), int(gaze_point[1])
        
        # Calculate vector and limit length to 30% of screen width
        dx, dy = gx - nx, gy - ny
        dist = np.sqrt(dx**2 + dy**2)
        max_dist = self.width * 0.3
        
        if dist > max_dist:
            scale = max_dist / dist
            gx, gy = int(nx + dx * scale), int(ny + dy * scale)
        
        # Red line with arrow
        cv2.arrowedLine(image, (nx, ny), (gx, gy), (0, 0, 255), 3, tipLength=0.2)
        cv2.circle(image, (gx, gy), 5, (0, 0, 255), -1)
        
        cv2.putText(image, f"Mode: {mode_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return image

    def side_by_side(self, image, naive_gaze, geo_gaze, nasion_pixel, landmarks):
        """ Create a split screen view with red arrows and green eye boxes """
        h, w = image.shape[:2]
        canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        # Draw shared eye box first
        image_with_box = self.draw_eye_box(image.copy(), landmarks)
        
        # Left side - NAIVE
        left = self.draw_gaze_overlay(image_with_box.copy(), nasion_pixel, naive_gaze, "NAIVE")
        cv2.putText(left, "NAIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Right side - GEOMETRY-AWARE
        right = self.draw_gaze_overlay(image_with_box.copy(), nasion_pixel, geo_gaze, "GEOMETRY")
        cv2.putText(right, "GEOMETRY-AWARE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        canvas[:, :w] = left
        canvas[:, w:] = right
        return canvas

    def draw_eval_grid(self, image, targets, current_idx):
        """ Draws the target dots for evaluation """
        for i, (tx, ty) in enumerate(targets):
            color = (0, 255, 0) if i == current_idx else (200, 200, 200)
            size = 15 if i == current_idx else 5
            cv2.circle(image, (int(tx * self.width), int(ty * self.height)), size, color, -1)
            cv2.putText(image, str(i+1), (int(tx * self.width)-10, int(ty * self.height)-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def get_heatmap_overlay(self, image):
        heatmap_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)
