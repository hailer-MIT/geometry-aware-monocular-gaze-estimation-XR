import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class FaceEyeDetector:
    def __init__(self):
        # Path to the model file
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Key landmark indices for eyes (approximate for the new landmarker)
        # We use standard eye region indices from MediaPipe mesh
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    def get_landmarks(self, image):
        # Convert to MediaPipe Image format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
        
        return detection_result.face_landmarks[0]

    def extract_eye_roi(self, image, landmarks, eye_indices, size=(224, 224)):
        h, w, _ = image.shape
        coords = []
        for idx in eye_indices:
            lm = landmarks[idx]
            coords.append([lm.x * w, lm.y * h])
        
        coords = np.array(coords)
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        
        padding_w = (x_max - x_min) * 0.2
        padding_h = (y_max - y_min) * 0.5
        
        x_min = max(0, int(x_min - padding_w))
        y_min = max(0, int(y_min - padding_h))
        x_max = min(w, int(x_max + padding_w))
        y_max = min(h, int(y_max + padding_h))
        
        eye_roi = image[y_min:y_max, x_min:x_max]
        if eye_roi.size == 0:
            return None
            
        eye_roi = cv2.resize(eye_roi, size)
        return eye_roi

    def get_eyes(self, image):
        landmarks = self.get_landmarks(image)
        if landmarks is None:
            return None, None
            
        left_eye = self.extract_eye_roi(image, landmarks, self.LEFT_EYE_INDICES)
        right_eye = self.extract_eye_roi(image, landmarks, self.RIGHT_EYE_INDICES)
        
        return left_eye, right_eye
