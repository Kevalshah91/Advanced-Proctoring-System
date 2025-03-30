# Yo ! before using this make sure you checkout the proc_model folder 
# Regards, Keval Shah 

import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import time

class ProctoringSystems:
    def __init__(self):
        self.phone_detected_count = 0
        self.multiple_faces_count = 0
        self.no_face_count = 0
        self.looking_away_count = 0
        self.last_alert_time = time.time()
        self.alert_cooldown = 5 
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def initialize_models(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
            
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.net = cv2.dnn.readNetFromCaffe('proc_models/deploy.prototxt', 
                                               'proc_models/mobilenet_iter_73000.caffemodel')
            return True
        except Exception as e:
            print(f"Error initializing models: {e}")
            return False

    def detect_looking_away(self, face_landmarks, frame):
        # Get face orientation from landmarks
        nose_tip = face_landmarks.landmark[4]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        # Convert to pixel coordinates
        h, w = frame.shape[:2]
        nose_x = int(nose_tip.x * w)
        nose_y = int(nose_tip.y * h)
        
        eye_center_x = (left_eye.x + right_eye.x) * w / 2
        if abs(nose_x - eye_center_x) > w * 0.15:  # Increased threshold for looking away
            return True
        return False

    def log_suspicious_activity(self, activity):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("proctoring_log.txt", "a") as f:
            f.write(f"{timestamp}: {activity}\n")

    def detect_eyes(self, roi_gray, roi_color):
        # Attempt to detect eyes with glasses-specific cascade
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,  # Reduced from 5 to be more lenient
            minSize=(20, 20),  # Reduced minimum size
            maxSize=(90, 90)
        )
        
        # Check if eyes were detected and draw them
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            return True
        return False

    def process_frame(self, frame):
        if frame is None:
            return None
            
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with Face Mesh
        face_mesh_results = self.face_mesh.process(frame_rgb)
        
        # Initialize status text
        status_text = []
        
        # Phone Detection
        blob = cv2.dnn.blobFromImage(frame, 0.0843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        phone_detected = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                class_id = int(detections[0, 0, i, 1])
                if class_id == 67:  
                    phone_detected = True
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], 
                                                              frame.shape[1], frame.shape[0]])
                    (x, y, x2, y2) = box.astype("int")
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                    status_text.append("WARNING: Mobile phone detected!")
                    self.phone_detected_count += 1
                    self.log_suspicious_activity("Mobile phone detected")
        
        # Face Detection and Analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            self.no_face_count += 1
            if self.no_face_count > 30:  # About 1 second at 30 FPS
                status_text.append("WARNING: No face detected!")
                self.log_suspicious_activity("No face detected")
        elif len(faces) > 1:
            self.multiple_faces_count += 1
            status_text.append("WARNING: Multiple faces detected!")
            self.log_suspicious_activity("Multiple faces detected")
        else:
            self.no_face_count = 0
            self.multiple_faces_count = 0
            
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            
            consecutive_no_eyes = 0
            if not self.detect_eyes(roi_gray, roi_color):
                consecutive_no_eyes += 1
                if consecutive_no_eyes > 10:  
                    status_text.append("Warning: Please ensure your eyes are visible")
                    self.log_suspicious_activity("Eyes not consistently visible")
            else:
                consecutive_no_eyes = 0
        
        # Check face orientation using Face Mesh with reduced sensitivity
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                if self.detect_looking_away(face_landmarks, frame):
                    self.looking_away_count += 1
                    if self.looking_away_count > 45:  
                        status_text.append("WARNING: Looking away from screen!")
                        self.log_suspicious_activity("Looking away from screen")
                else:
                    self.looking_away_count = 0
        
        # Draw status text
        for i, text in enumerate(status_text):
            cv2.putText(frame, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

    def run(self):
        if not self.initialize_models():
            return
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video capture")
            return
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                    
                processed_frame = self.process_frame(frame)
                if processed_frame is None:
                    continue
                    
                cv2.imshow("Proctoring System", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error during processing: {e}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()

if __name__ == "__main__":
    proctoring_system = ProctoringSystems()
    proctoring_system.run()