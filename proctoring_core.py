import cv2
import numpy as np
import time
import threading
import os
import base64
from datetime import datetime
import mss
import pyttsx3
from mtcnn import MTCNN
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
import json

# -------------------------
# CONFIGURATION
# -------------------------
class ProctorConfig:
    # Thresholds
    MOUTH_OPEN_THRESHOLD = 0.05  # For talking detection
    GAZE_THRESHOLD = 0.3         # For looking away
    BLINK_THRESHOLD = 0.21       # For eyes closed
    AUDIO_THRESHOLD = 0.02       # For voice detection
    
    # Paths
    RECORDINGS_DIR = "static/recordings"
    REPORTS_DIR = "static/reports"
    MODELS_DIR = "models"

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

# -------------------------
# ADVANCED PROCTORING ENGINE
# -------------------------
class ProctoringEngine:
    def __init__(self, student_id, reference_embedding=None):
        self.student_id = student_id
        self.start_time = datetime.now()
        self.violations = []
        self.frames_recorded = 0
        self.is_running = False
        
        # 1. MTCNN (Face Detection)
        print("Loading MTCNN...")
        self.mtcnn_detector = MTCNN()
        
        # 2. MediaPipe (Face Mesh - Landmarks)
        print("Loading MediaPipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        # 3. FaceNet-PyTorch (Identity Verification)
        print("Loading FaceNet...")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.reference_embedding = reference_embedding
        
        # 4. Screen Recorder (MSS)
        self.screen_recorder = mss.mss()
        
        # 5. TTS Engine (Alert Speaker)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150) # Speed of speech
        
        # Video Writer for Evidence
        self.video_writer = None
        self.report_data = {
            "student_id": student_id,
            "start_time": str(self.start_time),
            "violations": [],
            "metrics": {}
        }

    def start_session(self):
        """Initialize recording threads"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(ProctorConfig.RECORDINGS_DIR, f"{self.student_id}_{timestamp}.mp4")
        
        # Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 5.0, (640, 480))
        self.is_running = True
        self.report_data['video_path'] = video_path
        
        # Start Screen Recording Thread
        self.screen_thread = threading.Thread(target=self._record_screen_loop, daemon=True)
        self.screen_thread.start()
        
        print(f"✅ Proctoring Session Started for {self.student_id}")

    def process_frame(self, frame):
        """Main processing loop using all 4 libraries"""
        if not self.is_running: return frame, {}

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {
            "faces": 0,
            "identity_verified": True,
            "mouth_open": False,
            "gaze": "Center",
            "eyes_closed": False,
            "objects": [] # To be filled by object detector
        }

        # --- 1. FACE DETECTION (MTCNN) ---
        # MTCNN detects faces and returns bounding boxes
        mtcnn_faces = self.mtcnn_detector.detect_faces(rgb_frame)
        results['faces'] = len(mtcnn_faces)

        if results['faces'] == 0:
            self._log_violation("No Face Detected", "High")
            self._speak_alert("Please face the camera")
        
        elif results['faces'] > 1:
            self._log_violation("Multiple Faces Detected", "High")
            self._speak_alert("Multiple people detected")

        # --- 2. IDENTITY VERIFICATION (FaceNet) ---
        # If reference embedding is provided, verify the largest face
        if self.reference_embedding is not None and len(mtcnn_faces) > 0:
            # Assuming first face is the main one for simplicity
            x, y, width, height = mtcnn_faces[0]['box']
            # Crop face
            face_crop = rgb_frame[y:y+height, x:x+width]
            if face_crop.size > 0:
                try:
                    # Preprocess for FaceNet
                    face_img = cv2.resize(face_crop, (160, 160))
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR) # FaceNet expects BGR usually or handled via tensor
                    face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                    face_tensor = fixed_image_standardization(face_tensor)
                    
                    # Get embedding
                    with torch.no_grad():
                        embedding = self.facenet(face_tensor)
                    
                    # Calculate distance (Euclidean)
                    dist = (self.reference_embedding - embedding).norm().item()
                    if dist > 1.1: # Threshold for VGGFace2
                        results['identity_verified'] = False
                        self._log_violation("Identity Mismatch", "Critical")
                except Exception as e:
                    print(f"FaceNet error: {e}")

        # --- 3. LANDMARKS (MediaPipe) - Eye & Mouth ---
        mp_results = self.face_mesh.process(rgb_frame)
        
        if mp_results.multi_face_landmarks:
            for face_landmarks in mp_results.multi_face_landmarks:
                # Mouth Detection (Upper Lip: 13, Lower Lip: 14)
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                mouth_dist = np.sqrt((top_lip.x - bottom_lip.x)**2 + (top_lip.y - bottom_lip.y)**2)
                
                if mouth_dist > ProctorConfig.MOUTH_OPEN_THRESHOLD:
                    results['mouth_open'] = True
                    self._log_violation("Mouth Movement (Talking)", "Medium")

                # Eye Gaze & Blink Detection
                # Using Iris landmarks (468, 473) if available (refine_landmarks=True)
                if len(face_landmarks.landmark) > 468:
                    left_iris = face_landmarks.landmark[468]
                    right_iris = face_landmarks.landmark[473]
                    
                    # Blink detection (Eye Aspect Ratio logic simplified via iris visibility or EAR)
                    # Here we use simple Y position check for looking down/up
                    if left_iris.y < 0.35: results['gaze'] = "Looking Up"
                    elif left_iris.y > 0.65: results['gaze'] = "Looking Down"
                    elif left_iris.x < 0.35: results['gaze'] = "Looking Left"
                    elif left_iris.x > 0.65: results['gaze'] = "Looking Right"
                    else: results['gaze'] = "Center"

                    if results['gaze'] != "Center":
                        self._log_violation(f"Gaze Detected: {results['gaze']}", "Low")

                # Draw Mesh (Optional for Debugging/Reporting)
                # mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)

        # --- SAVE FRAME TO EVIDENCE VIDEO ---
        if self.video_writer:
            resized_frame = cv2.resize(frame, (640, 480))
            self.video_writer.write(resized_frame)

        return frame, results

    def _record_screen_loop(self):
        """Background thread to record screen using MSS"""
        while self.is_running:
            try:
                monitor = self.screen_recorder.monitors[0]
                screenshot = np.array(self.screen_recorder.grab(monitor))
                # Drop alpha channel and convert BGR
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                # You could save screenshots here or stream them
                time.sleep(1.0) # Capture screen every 1 second to save CPU
            except Exception as e:
                print(f"Screen recording error: {e}")

    def _log_violation(self, message, severity):
        """Log violation with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        violation = {"time": timestamp, "message": message, "severity": severity}
        self.violations.append(violation)
        self.report_data['violations'].append(violation)
        print(f"[{timestamp}] ALERT: {message}")

    def _speak_alert(self, message):
        """Text-to-Speech Alert"""
        def speak():
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        threading.Thread(target=speak, daemon=True).start()

    def stop_session(self):
        """Stop recording and generate report"""
        self.is_running = False
        if self.video_writer:
            self.video_writer.release()
        
        self.report_data['end_time'] = str(datetime.now())
        self._generate_report()
        print("✅ Session Stopped and Report Generated.")

    def _generate_report(self):
        """Generate PDF and HTML Report"""
        # 1. JSON Log
        json_path = os.path.join(ProctorConfig.REPORTS_DIR, f"{self.student_id}_log.json")
        with open(json_path, 'w') as f:
            json.dump(self.report_data, f, indent=4)

        # 2. PDF Report
        pdf_path = os.path.join(ProctorConfig.REPORTS_DIR, f"{self.student_id}_report.pdf")
        c = pdf_canvas.Canvas(pdf_path, pagesize=letter)
        
        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, f"Proctoring Report: {self.student_id}")
        
        # Metrics
        c.setFont("Helvetica", 12)
        y = 700
        c.drawString(100, y, f"Start Time: {self.report_data['start_time']}")
        y -= 20
        c.drawString(100, y, f"End Time: {self.report_data['end_time']}")
        y -= 20
        c.drawString(100, y, f"Total Violations: {len(self.report_data['violations'])}")
        y -= 40

        # Violations List
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y, "Violations Log:")
        y -= 20
        
        c.setFont("Helvetica", 10)
        for v in self.report_data['violations']:
            c.drawString(100, y, f"[{v['time']}] {v['message']} ({v['severity']})")
            y -= 15
            if y < 50: 
                c.showPage()
                y = 750
        
        c.save()
        print(f"✅ Report saved at {pdf_path}")

    @staticmethod
    def get_reference_embedding(image_path):
        """Static method to generate embedding for registration"""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        img_tensor = fixed_image_standardization(img_tensor)
        
        with torch.no_grad():
            embedding = model(img_tensor)
        return embedding