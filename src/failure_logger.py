import os
import time

class FailureLogger:
    def __init__(self, log_path="results/failures.log"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(f"\n--- SESSION START: {time.ctime()} ---\n")

    def log_failure(self, condition, mode):
        timestamp = time.strftime("%H:%M:%S")
        msg = f"[{timestamp}] MODE: {mode} | FAILURE: {condition}\n"
        with open(self.log_path, "a") as f:
            f.write(msg)
        print(f"Logged failure: {condition}")

    def check_conditions(self, landmarks, frame, mode):
        """ Detects potential failures based on geometry and image """
        failures = []
        
        if landmarks is None:
            failures.append("Face Lost")
        else:
            # Check Head Rotation (Yaw)
            nose = landmarks[1]
            l_eye = landmarks[33]
            r_eye = landmarks[263]
            yaw = abs((nose.x - l_eye.x) / (r_eye.x - l_eye.x + 1e-6) - 0.5)
            if yaw > 0.35:
                failures.append("Extreme Head Rotation")

            # Check Lighting (Global brightness)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            if avg_brightness < 40:
                failures.append("Low Lighting")
            elif avg_brightness > 220:
                failures.append("Overexposed / Bright Light")

        for f in failures:
            self.log_failure(f, mode)
        
        return failures
import cv2
import numpy as np
