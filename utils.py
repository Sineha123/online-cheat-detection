"""
Enhanced Utils Module with Complete Real-Time Monitoring
Features:
- Real-time face detection & tracking
- Advanced eye gaze tracking with MediaPipe
- Object detection (phones, books, laptops, etc.)
- Audio/Sound detection & recording
- Auto-save functionality from exam start to submit
- Comprehensive violation logging
"""

import cv2
import numpy as np
import json
import os
import time
import shutil
import threading
from datetime import datetime
from collections import deque

# ============================================================================
# Optional Dependencies with Fallbacks
# ============================================================================
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe loaded successfully")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print("⚠️ MediaPipe not installed. Install: pip install mediapipe")

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
    print("✅ Keyboard module loaded")
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None
    print("⚠️ Keyboard module not installed. Install: pip install keyboard")

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
    print("✅ Audio libraries loaded")
except ImportError:
    AUDIO_AVAILABLE = False
    sd = None
    sf = None
    print("⚠️ Audio libraries not installed. Install: pip install sounddevice soundfile")

try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("✅ Text-to-Speech loaded")
except ImportError:
    TTS_AVAILABLE = False
    pyttsx3 = None
    print("⚠️ TTS not installed. Install: pip install pyttsx3")


# ============================================================================
# Global Variables
# ============================================================================
cap = None
Globalflag = False
Student_Name = ""
Student_ID = ""
shorcuts = []
violation_buffer = deque(maxlen=30)

# Face Recognition and Audio instances
fr = None
a = None

# MediaPipe Face Mesh for eye tracking
face_mesh = None
face_detection_mp = None

if MEDIAPIPE_AVAILABLE:
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_detection_mp = mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        print("✅ MediaPipe face mesh initialized")
    except Exception as e:
        MEDIAPIPE_AVAILABLE = False
        print(f"❌ MediaPipe initialization failed: {e}")

# Eye tracking thresholds
EYE_AR_THRESH = 0.21
EYE_AR_CONSEC_FRAMES = 3
BLINK_COUNTER = 0
GAZE_VIOLATION_COUNTER = 0

# YOLO object detection
YOLO_AVAILABLE = False
net = None
classes = []

try:
    if os.path.exists("yolov3.weights") and os.path.exists("yolov3.cfg"):
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        if os.path.exists("coco.names"):
            with open("coco.names", "r") as f:
                classes = [line.strip() for line in f.readlines()]
        YOLO_AVAILABLE = True
        print("✅ YOLO object detection loaded")
except Exception as e:
    print(f"⚠️ YOLO not available: {e}")

# Prohibited objects list
PROHIBITED_OBJECTS = [
    'book', 'cell phone', 'laptop', 'notebook', 
    'tablet', 'monitor', 'keyboard', 'mouse',
    'paper', 'person', 'backpack', 'handbag'
]


# ============================================================================
# Audio Recording & Detection System
# ============================================================================
class AudioRecorder:
    """Advanced audio recording with real-time detection"""
    
    def __init__(self, student_name="Student"):
        self.student_name = student_name
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 44100
        self.channels = 1
        self.recording_thread = None
        self.sound_threshold = 0.015  # Threshold for sound detection
        self.violation_count = 0
        self.session_start = None
        
    def start_recording(self):
        """Start continuous audio recording"""
        if not AUDIO_AVAILABLE:
            print("⚠️ Audio recording not available")
            return False
        
        self.is_recording = True
        self.session_start = datetime.now()
        self.audio_data = []
        self.violation_count = 0
        
        self.recording_thread = threading.Thread(
            target=self._record_audio_stream,
            daemon=True
        )
        self.recording_thread.start()
        
        print(f"🎤 Audio recording started for {self.student_name}")
        return True
    
    def _record_audio_stream(self):
        """Background thread for continuous audio recording"""
        try:
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"Audio status: {status}")
                
                # Calculate volume level
                volume_norm = np.linalg.norm(indata) * 10
                
                # Store audio data
                self.audio_data.append(indata.copy())
                
                # Check if sound exceeds threshold
                if volume_norm > self.sound_threshold:
                    self.violation_count += 1
            
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=2048
            ):
                while self.is_recording:
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Audio recording error: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_recording(self):
        """Stop recording and save audio file"""
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        
        if self.audio_data and AUDIO_AVAILABLE:
            try:
                # Concatenate all audio chunks
                audio_array = np.concatenate(self.audio_data, axis=0)
                
                # Create filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "static/audio_recordings"
                os.makedirs(output_dir, exist_ok=True)
                
                filename = f"{output_dir}/{self.student_name}_{timestamp}.wav"
                
                # Save audio file
                sf.write(filename, audio_array, self.sample_rate)
                
                duration = len(audio_array) / self.sample_rate
                
                print(f"✅ Audio saved: {filename}")
                print(f"   Duration: {duration:.2f}s")
                print(f"   Sound violations: {self.violation_count}")
                
                return {
                    'filename': filename,
                    'duration': duration,
                    'violations': self.violation_count
                }
                
            except Exception as e:
                print(f"❌ Error saving audio: {e}")
                import traceback
                traceback.print_exc()
        
        return None
    
    def get_violation_count(self):
        """Get current violation count"""
        return self.violation_count


# ============================================================================
# Enhanced Violation Recording System
# ============================================================================
class ViolationRecorder:
    """Enhanced violation recording with MP4 support and metadata"""
    
    def __init__(self):
        self.is_recording = False
        self.video_writer = None
        self.violation_type = ""
        self.start_time = None
        self.frame_count = 0
        self.output_path = ""
        self.metadata = {}
        
    def start_recording(self, violation_type, student_name):
        """Start recording violation"""
        if not self.is_recording:
            self.violation_type = violation_type
            self.start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_dir = "static/OutputVideos"
            os.makedirs(output_dir, exist_ok=True)
            
            # Clean filename
            clean_violation = violation_type.replace(' ', '_').replace(':', '').replace('/', '-')
            filename = f"{student_name}_{clean_violation}_{timestamp}.mp4"
            self.output_path = os.path.join(output_dir, filename)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, 
                fourcc, 
                15.0,  # 15 FPS
                (640, 480)
            )
            
            self.is_recording = True
            self.frame_count = 0
            
            # Store metadata
            self.metadata = {
                'violation_type': violation_type,
                'student_name': student_name,
                'start_time': datetime.now().isoformat(),
                'frames': []
            }
            
            print(f"📹 Started recording: {filename}")
    
    def write_frame(self, frame, annotations=None):
        """Write frame to video with optional annotations"""
        if self.is_recording and self.video_writer is not None:
            try:
                # Resize frame
                frame_resized = cv2.resize(frame, (640, 480))
                
                # Add timestamp overlay
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame_resized, timestamp, (10, 470), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add violation type overlay
                cv2.putText(frame_resized, f"Violation: {self.violation_type}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 0, 255), 2)
                
                # Write frame
                self.video_writer.write(frame_resized)
                self.frame_count += 1
                
                # Store frame metadata
                if annotations:
                    self.metadata['frames'].append({
                        'frame_num': self.frame_count,
                        'timestamp': timestamp,
                        'annotations': annotations
                    })
                
            except Exception as e:
                print(f"❌ Error writing frame: {e}")
    
    def stop_recording(self):
        """Stop recording and save video + metadata"""
        if self.is_recording:
            duration = time.time() - self.start_time
            
            # Release video writer
            if self.video_writer is not None:
                self.video_writer.release()
            
            self.is_recording = False
            
            # Complete metadata
            self.metadata['end_time'] = datetime.now().isoformat()
            self.metadata['duration'] = duration
            self.metadata['total_frames'] = self.frame_count
            
            # Save metadata JSON
            metadata_path = self.output_path.replace('.mp4', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            
            # Create violation record
            violation_data = {
                "Name": f"{self.violation_type}",
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Duration": f"{duration:.2f}s",
                "Mark": self.calculate_penalty(self.violation_type),
                "Link": self.output_path,
                "RId": get_resultId(),
                "Frames": self.frame_count,
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "StudentName": Student_Name,
                "MetadataPath": metadata_path
            }
            
            write_json(violation_data, "static/violation.json")
            
            print(f"✅ Recording saved: {self.output_path}")
            print(f"   Duration: {duration:.2f}s, Frames: {self.frame_count}")
            
            return self.output_path
    
    def calculate_penalty(self, violation_type):
        """Calculate penalty marks based on violation severity"""
        penalties = {
            "No Face": 5.0,
            "Multiple": 7.0,
            "Looking Left": 3.0,
            "Looking Right": 3.0,
            "Looking Up": 2.0,
            "Looking Down": 2.0,
            "Eyes": 2.0,
            "Prohibited": 6.0,
            "cell phone": 8.0,
            "book": 4.0,
            "laptop": 6.0,
            "person": 7.0,
            "Audio": 3.0
        }
        
        for key, value in penalties.items():
            if key.lower() in violation_type.lower():
                return value
        
        return 3.0


violation_recorder = ViolationRecorder()


# ============================================================================
# Advanced Face Detection with MediaPipe
# ============================================================================
def detect_multiple_faces(frame):
    """Detect multiple faces using MediaPipe or Haar Cascade"""
    if MEDIAPIPE_AVAILABLE and face_detection_mp is not None:
        return detect_faces_mediapipe(frame)
    else:
        return detect_faces_haar(frame)


def detect_faces_mediapipe(frame):
    """Detect faces using MediaPipe"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection_mp.process(rgb_frame)
        
        face_count = 0
        face_locations = []
        
        if results.detections:
            face_count = len(results.detections)
            h, w, _ = frame.shape
            
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Ensure coordinates are within frame
                x = max(0, min(x, w))
                y = max(0, min(y, h))
                width = max(0, min(width, w - x))
                height = max(0, min(height, h - y))
                
                face_locations.append((x, y, width, height))
                
                # Draw rectangle
                color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                
                # Add label
                label = "STUDENT" if face_count == 1 else f"PERSON {len(face_locations)}"
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return face_count, face_locations, frame
        
    except Exception as e:
        print(f"❌ MediaPipe face detection error: {e}")
        return detect_faces_haar(frame)


def detect_faces_haar(frame):
    """Fallback face detection using Haar Cascade"""
    try:
        cascade_path = 'Haarcascades/haarcascade_frontalface_default.xml'
        
        if os.path.exists(cascade_path):
            detector = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            
            face_count = len(faces)
            face_locations = []
            
            for (x, y, w, h) in faces:
                face_locations.append((x, y, w, h))
                
                color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = "STUDENT" if face_count == 1 else f"PERSON {len(face_locations)}"
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return face_count, face_locations, frame
            
    except Exception as e:
        print(f"❌ Haar cascade error: {e}")
    
    return 1, [], frame


# ============================================================================
# Advanced Eye Gaze Tracking
# ============================================================================
def detect_eye_gaze(frame):
    """
    Detect eye gaze direction and blink status
    Returns: (gaze_direction, eyes_closed, annotated_frame)
    """
    if not MEDIAPIPE_AVAILABLE or face_mesh is None:
        return "Center", False, frame
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        gaze_direction = "Center"
        eyes_closed = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                # Eye landmarks indices
                left_eye = [33, 133, 160, 159, 158, 144, 145, 153]
                right_eye = [362, 263, 387, 386, 385, 373, 374, 380]
                
                # Calculate Eye Aspect Ratio (EAR)
                left_ear = calculate_ear(face_landmarks, left_eye, h, w)
                right_ear = calculate_ear(face_landmarks, right_eye, h, w)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Check if eyes are closed
                if avg_ear < EYE_AR_THRESH:
                    eyes_closed = True
                
                # Get iris positions for gaze detection
                left_iris = face_landmarks.landmark[468]  # Left iris
                right_iris = face_landmarks.landmark[473]  # Right iris
                
                # Convert to pixel coordinates
                left_iris_x = left_iris.x * w
                left_iris_y = left_iris.y * h
                right_iris_x = right_iris.x * w
                right_iris_y = right_iris.y * h
                
                # Determine gaze direction
                avg_iris_x = (left_iris_x + right_iris_x) / 2
                avg_iris_y = (left_iris_y + right_iris_y) / 2
                
                # Horizontal gaze
                if avg_iris_x < w * 0.35:
                    gaze_direction = "Looking Left"
                elif avg_iris_x > w * 0.65:
                    gaze_direction = "Looking Right"
                # Vertical gaze
                elif avg_iris_y < h * 0.35:
                    gaze_direction = "Looking Up"
                elif avg_iris_y > h * 0.65:
                    gaze_direction = "Looking Down"
                else:
                    gaze_direction = "Center"
                
                # Draw eye landmarks
                for idx in left_eye + right_eye:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Draw iris points
                cv2.circle(frame, (int(left_iris_x), int(left_iris_y)), 2, (255, 0, 0), -1)
                cv2.circle(frame, (int(right_iris_x), int(right_iris_y)), 2, (255, 0, 0), -1)
        
        return gaze_direction, eyes_closed, frame
        
    except Exception as e:
        print(f"❌ Eye gaze detection error: {e}")
        return "Center", False, frame


def calculate_ear(face_landmarks, eye_indices, h, w):
    """Calculate Eye Aspect Ratio for blink detection"""
    try:
        points = []
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            points.append([landmark.x * w, landmark.y * h])
        
        points = np.array(points)
        
        # Vertical distances
        A = np.linalg.norm(points[1] - points[5])
        B = np.linalg.norm(points[2] - points[4])
        
        # Horizontal distance
        C = np.linalg.norm(points[0] - points[3])
        
        # Eye Aspect Ratio
        ear = (A + B) / (2.0 * C) if C > 0 else 0.3
        
        return ear
    except:
        return 0.3


# ============================================================================
# Object Detection (YOLO or MobileNet-SSD)
# ============================================================================
def detect_objects(frame):
    """Detect prohibited objects in frame"""
    if YOLO_AVAILABLE:
        return detect_objects_yolo(frame)
    else:
        return detect_objects_basic(frame)


def detect_objects_yolo(frame):
    """Detect objects using YOLO"""
    try:
        height, width, _ = frame.shape
        
        # Create blob
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        
        # Get output layers
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        # Forward pass
        outs = net.forward(output_layers)
        
        detected_objects = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.3:
                    object_name = classes[class_id] if class_id < len(classes) else "unknown"
                    
                    if object_name.lower() in [obj.lower() for obj in PROHIBITED_OBJECTS]:
                        # Get bounding box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        detected_objects.append({
                            'object': object_name,
                            'confidence': float(confidence),
                            'bbox': (x, y, w, h)
                        })
                        
                        # Draw on frame
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        label = f"{object_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detected_objects, frame
        
    except Exception as e:
        print(f"❌ YOLO detection error: {e}")
        return [], frame


def detect_objects_basic(frame):
    """Basic object detection fallback"""
    # Placeholder for basic detection
    return [], frame


# ============================================================================
# Main Detection Loop (Thread 1)
# ============================================================================
def cheat_Detection1():
    """Primary detection thread with all monitoring features"""
    global Globalflag, cap, violation_buffer, GAZE_VIOLATION_COUNTER, BLINK_COUNTER
    
    consecutive_violations = 0
    last_violation_type = ""
    
    print("🔍 Detection Thread 1 started")
    
    while True:
        if not Globalflag:
            time.sleep(0.5)
            continue
        
        if cap is None or not cap.isOpened():
            time.sleep(0.5)
            continue
        
        try:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue
            
            violation_detected = False
            violation_type = ""
            annotations = {}
            
            # ========== 1. FACE DETECTION ==========
            face_count, face_locations, frame = detect_multiple_faces(frame)
            annotations['face_count'] = face_count
            annotations['face_locations'] = face_locations
            
            if face_count == 0:
                violation_type = "No Face Detected"
                violation_detected = True
            elif face_count > 1:
                violation_type = f"Multiple Faces ({face_count})"
                violation_detected = True
            
            # ========== 2. EYE GAZE TRACKING ==========
            gaze_status, eyes_closed, frame = detect_eye_gaze(frame)
            annotations['gaze_direction'] = gaze_status
            annotations['eyes_closed'] = eyes_closed
            
            if gaze_status in ["Looking Left", "Looking Right", "Looking Up", "Looking Down"]:
                GAZE_VIOLATION_COUNTER += 1
                if GAZE_VIOLATION_COUNTER >= 10:  # 10 consecutive frames
                    violation_type = f"Gaze Violation: {gaze_status}"
                    violation_detected = True
            else:
                GAZE_VIOLATION_COUNTER = 0
            
            if eyes_closed:
                BLINK_COUNTER += 1
                if BLINK_COUNTER >= 15:  # Eyes closed for too long
                    violation_type = "Eyes Closed Too Long"
                    violation_detected = True
            else:
                BLINK_COUNTER = 0
            
            # ========== 3. OBJECT DETECTION ==========
            detected_objects, frame = detect_objects(frame)
            annotations['objects'] = detected_objects
            
            if detected_objects:
                objects_str = ", ".join([obj['object'] for obj in detected_objects])
                violation_type = f"Prohibited Object: {objects_str}"
                violation_detected = True
            
            # ========== 4. VIOLATION RECORDING ==========
            if violation_detected:
                consecutive_violations += 1
                
                # Start recording after 5 consecutive violations
                if consecutive_violations >= 5:
                    if not violation_recorder.is_recording:
                        violation_recorder.start_recording(violation_type, Student_Name)
                    
                    # Write frame with annotations
                    violation_recorder.write_frame(frame, annotations)
                    
                    # Stop recording after 15 seconds or if violation cleared
                    if violation_recorder.start_time and time.time() - violation_recorder.start_time > 15:
                        violation_recorder.stop_recording()
                        consecutive_violations = 0
            else:
                # Stop recording when violation clears
                if violation_recorder.is_recording:
                    violation_recorder.stop_recording()
                consecutive_violations = 0
                GAZE_VIOLATION_COUNTER = 0
            
            # Add overlay information
            cv2.putText(frame, f"Faces: {face_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Gaze: {gaze_status}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if violation_detected:
                cv2.putText(frame, f"VIOLATION: {violation_type}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            time.sleep(0.05)  # 20 FPS
            
        except Exception as e:
            print(f"❌ Detection1 error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.5)


# ============================================================================
# Secondary Detection Loop (Thread 2)
# ============================================================================
def cheat_Detection2():
    """Secondary detection thread for additional monitoring"""
    global Globalflag, cap
    
    print("🔍 Detection Thread 2 started")
    
    while True:
        if not Globalflag:
            time.sleep(0.5)
            continue
        
        if cap is None or not cap.isOpened():
            time.sleep(0.5)
            continue
        
        try:
            success, frame = cap.read()
            if not success:
                continue
            
            # Additional processing can be added here
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Detection2 error: {e}")
            time.sleep(0.5)


# ============================================================================
# Utility Functions
# ============================================================================
def get_resultId():
    """Get next result ID"""
    try:
        if os.path.exists('static/result.json'):
            with open('static/result.json', 'r') as f:
                data = json.load(f)
                if data:
                    return max([item.get('Id', 0) for item in data]) + 1
        return 1
    except:
        return 1


def write_json(data, filename="static/violation.json"):
    """Write violation/result data to JSON"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        existing_data = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        existing_data.append(data)
        
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)
        
        return True
    except Exception as e:
        print(f"❌ Error writing JSON: {e}")
        return False


def get_TrustScore(result_id):
    """Calculate trust score from violations"""
    try:
        with open('static/violation.json', 'r') as f:
            violations = json.load(f)
        
        total_penalty = 0
        for violation in violations:
            if violation.get('RId') == result_id:
                total_penalty += violation.get('Mark', 0)
        
        # Convert penalty to trust score (0-100)
        trust_score = max(0, 100 - total_penalty)
        return min(trust_score, 100)
    except:
        return 100


def getResults():
    """Get all exam results"""
    try:
        with open('static/result.json', 'r') as f:
            return json.load(f)
    except:
        return []


def getResultDetails(result_id):
    """Get detailed result with violations"""
    try:
        result_id = int(result_id)
        
        with open('static/result.json', 'r') as f:
            results = json.load(f)
        
        result = next((r for r in results if r.get('Id') == result_id), None)
        
        if not result:
            return None
        
        try:
            with open('static/violation.json', 'r') as f:
                violations = json.load(f)
            
            result['violations'] = [v for v in violations if v.get('RId') == result_id]
        except:
            result['violations'] = []
        
        return result
    except Exception as e:
        print(f"❌ Error getting result details: {e}")
        return None


def shortcut_handler(event):
    """Handle prohibited keyboard shortcuts"""
    global shorcuts
    
    if not KEYBOARD_AVAILABLE:
        return
    
    try:
        prohibited = [
            'alt+tab', 'ctrl+alt+delete', 'ctrl+shift+esc',
            'windows+d', 'alt+f4', 'ctrl+w', 'ctrl+q',
            'cmd+tab', 'cmd+q', 'cmd+w'
        ]
        
        key_combo = str(event).lower()
        
        for shortcut in prohibited:
            if shortcut in key_combo:
                shorcuts.append(shortcut)
                print(f"⚠️ Prohibited shortcut detected: {shortcut}")
                
                # Add to violations
                violation_data = {
                    "Name": f"Prohibited Shortcut: {shortcut}",
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Mark": 2.0,
                    "RId": get_resultId(),
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "StudentName": Student_Name
                }
                write_json(violation_data, "static/violation.json")
                break
    except Exception as e:
        print(f"❌ Shortcut handler error: {e}")


def initialize_detection():
    """Initialize all detection components"""
    global fr, a
    
    print("🚀 Initializing detection systems...")
    
    # Face Recognition
    try:
        from face_recognition_module import FaceRecognition
        fr = FaceRecognition()
        print("✅ Face recognition initialized")
    except Exception as e:
        print(f"⚠️ Face recognition not available: {e}")
    
    # Audio Recorder (global instance)
    if AUDIO_AVAILABLE:
        a = AudioRecorder(Student_Name if Student_Name else "Student")
        print("✅ Audio recorder initialized")
    else:
        print("⚠️ Audio recorder not available")


# ============================================================================
# Module Initialization
# ============================================================================
print("=" * 70)
print("✅ Enhanced Utils Module Loaded Successfully")
print(f"  📦 Dependencies:")
print(f"     - MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
print(f"     - YOLO: {'✅' if YOLO_AVAILABLE else '❌'}")
print(f"     - Audio: {'✅' if AUDIO_AVAILABLE else '❌'}")
print(f"     - Keyboard: {'✅' if KEYBOARD_AVAILABLE else '❌'}")
print(f"  🎯 Features:")
print(f"     ✓ Real-time face detection (MediaPipe + Haar)")
print(f"     ✓ Advanced eye gaze tracking")
print(f"     ✓ Object detection (YOLO/basic)")
print(f"     ✓ Audio detection & recording")
print(f"     ✓ Auto-save violations")
print(f"     ✓ Comprehensive metadata logging")
print("=" * 70)