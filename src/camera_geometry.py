import numpy as np
import cv2

class CameraGeometry:
    def __init__(self, image_width=640, image_height=480):
        self.width = image_width
        self.height = image_height
        
        # Approximate camera intrinsics
        self.focal_length = image_width 
        self.cx = image_width / 2
        self.cy = image_height / 2

    def normalize_eye(self, image, landmarks, eye_center_idx, target_size=(224, 224)):
        """
        GEOMETRY-AWARE NORMALIZATION:
        1. Calculates head roll to straighten the image.
        2. Normalizes distance/scale based on inter-pupillary distance.
        3. Warps the eye crop into a canonical view.
        """
        # Get reference landmarks (indices 33, 263 are eye corners)
        l_corner = landmarks[33]
        r_corner = landmarks[263]
        
        # Calculate angle (Roll)
        dy = r_corner.y - l_corner.y
        dx = r_corner.x - l_corner.x
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Inter-eye distance for scale
        dist = np.sqrt(dx**2 + dy**2)
        scale = 0.2 / (dist + 1e-6) # Target scale
        
        # Get center for the eye to be normalized
        target_center = landmarks[eye_center_idx]
        cx, cy = target_center.x * self.width, target_center.y * self.height
        
        # Rotation / Scaling matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        # Extract and warp
        eye_img = cv2.warpAffine(image, M, (self.width, self.height))
        
        # Crop around the center
        size = 100 # Fixed size in pixels for the crop before resizing
        x1, y1 = int(cx - size), int(cy - size)
        x2, y2 = int(cx + size), int(cy + size)
        
        # Bounds check
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)
        
        crop = eye_img[y1:y2, x1:x2]
        if crop.size > 0:
            return cv2.resize(crop, target_size)
        return None

    def get_head_pose(self, landmarks):
        """
        Estimates simplified head pose (Pitch, Yaw, Roll).
        Valuable for failure analysis (e.g., detecting 'extreme rotation').
        """
        # Nose tip (1), Chin (152), Left eye corner (33), Right eye corner (263)
        # Simplified ratio-based yaw/pitch detection
        nose = landmarks[1]
        l_eye = landmarks[33]
        r_eye = landmarks[263]
        
        # Yaw: difference in eye-to-nose distances
        yaw = (nose.x - l_eye.x) / (r_eye.x - l_eye.x + 1e-6) - 0.5
        
        # Roll: angle between eyes
        roll = np.arctan2(r_eye.y - l_eye.y, r_eye.x - l_eye.x)
        
        return yaw, roll
