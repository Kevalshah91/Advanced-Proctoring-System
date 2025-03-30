import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import time
import torch
import logging
import os

class ComprehensiveProctoring:
    def __init__(self):
        self._setup_logging()
        self._init_counters()
        self._init_mediapipe()
        self._init_yolo()
        self._init_cascade_classifiers()
        
    def _setup_logging(self):
        logging.basicConfig(
            filename="proctoring_log.txt",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _init_counters(self):
        self.counts = {
            'phone': 0,
            'no_face': 0,
            'multiple_faces': 0,
            'looking_away': 0,
            'no_eyes': 0
        }
        
    def _init_mediapipe(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
    def _init_yolo(self):
        """Initialize YOLOv5 model for phone detection."""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            self.model.conf = 0.5
            self.model.classes = [67]  # Filter for cell phones only
            self.logger.info("YOLOv5 model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading YOLOv5 model: {e}")
            raise
            
    def _init_cascade_classifiers(self):
        """Initialize Haar cascade classifiers for face and eye detection."""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
            self.logger.info("Cascade classifiers loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading cascade classifiers: {e}")
            raise

    def detect_phone(self, frame):
        """Detect phones using YOLOv5."""
        results = self.model(frame)
        detections = []
        
        for *box, conf, cls in results.xyxy[0]:
            if conf > self.model.conf:
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    'confidence': float(conf),
                    'box': (x1, y1, x2, y2)
                })
                
                # Draw detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'Phone: {conf:.2f}', 
                          (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 0, 255), 2)
                
        return detections

    def detect_eyes(self, roi_gray, roi_color):
        """Detect and verify presence of eyes."""
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            maxSize=(90, 90)
        )
        
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            return True
        return False

    def detect_looking_away(self, face_landmarks, frame_shape):
        """Detect if person is looking away using face landmarks."""
        try:
            h, w = frame_shape[:2]
            
            nose_tip = face_landmarks.landmark[4]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            nose_x = int(nose_tip.x * w)
            eye_center_x = (left_eye.x + right_eye.x) * w / 2
            
            return abs(nose_x - eye_center_x) > w * 0.15
            
        except Exception as e:
            self.logger.error(f"Error in detect_looking_away: {e}")
            return False

    def log_suspicious_activity(self, activity):
        """Log suspicious activities with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.warning(f"{activity}")
        
    def process_frame(self, frame):
        """Process a single frame with all detection methods."""
        if frame is None:
            return None, []
            
        warnings = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phone Detection using YOLO
        phone_detections = self.detect_phone(frame)
        if phone_detections:
            self.counts['phone'] += 1
            if self.counts['phone'] > 5:
                warnings.append("WARNING: Phone detected!")
                self.log_suspicious_activity("Phone detected")
        else:
            self.counts['phone'] = max(0, self.counts['phone'] - 1)
        
        # Face Detection using Haar Cascade
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            self.counts['no_face'] += 1
            if self.counts['no_face'] > 30:
                warnings.append("WARNING: No face detected!")
                self.log_suspicious_activity("No face detected")
        elif len(faces) > 1:
            self.counts['multiple_faces'] += 1
            warnings.append("WARNING: Multiple faces detected!")
            self.log_suspicious_activity("Multiple faces detected")
        else:
            self.counts['no_face'] = 0
            self.counts['multiple_faces'] = 0
            
            # Process single face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                
                # Eye Detection
                if not self.detect_eyes(roi_gray, roi_color):
                    self.counts['no_eyes'] += 1
                    if self.counts['no_eyes'] > 10:
                        warnings.append("WARNING: Eyes not visible!")
                        self.log_suspicious_activity("Eyes not visible")
                else:
                    self.counts['no_eyes'] = 0
        
        # Face Mesh Processing for Looking Away Detection
        face_mesh_results = self.face_mesh.process(frame_rgb)
        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            if self.detect_looking_away(face_landmarks, frame.shape):
                self.counts['looking_away'] += 1
                if self.counts['looking_away'] > 45:
                    warnings.append("WARNING: Looking away from screen!")
                    self.log_suspicious_activity("Looking away from screen")
            else:
                self.counts['looking_away'] = 0
        
        # Draw warnings
        for i, text in enumerate(warnings):
            cv2.putText(frame, text, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, warnings

    def run(self):
        """Run the comprehensive proctoring system."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("Could not open video capture")
            return
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Could not read frame")
                    break
                    
                processed_frame, warnings = self.process_frame(frame)
                if processed_frame is None:
                    continue
                    
                cv2.imshow("Comprehensive Proctoring System", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()

if __name__ == "__main__":
    proctoring_system = ComprehensiveProctoring()
    proctoring_system.run()