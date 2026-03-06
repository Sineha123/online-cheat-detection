import mediapipe as mp
import math
import threading

_mp_face_mesh = mp.solutions.face_mesh
_global_face_mesh = _mp_face_mesh.FaceMesh(
    static_image_mode=True, # Critical for threads processing disjointed frames
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.3,  # Lowered to reduce false "no face" detections
)
_face_lock = threading.Lock()

class FaceAnalyzer:
    def __init__(self):
        pass

    def process_frame(self, frame_rgb):
        """
        Processes a zero-copy frame buffer. Stateless and Thread-Safe.
        Returns face_detected, yaw_angle, ear, iris_offset_ratio, and landmarks.
        """
        with _face_lock:
            results = _global_face_mesh.process(frame_rgb)
        
        face_detected = False
        yaw_angle = 0.0
        ear = 0.0
        iris_offset_ratio = 0.0
        out_landmarks = []
        
        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            out_landmarks = face_landmarks.landmark
            
            yaw_angle = self._estimate_yaw(out_landmarks)
            ear = self._calculate_ear(out_landmarks)
            iris_offset_ratio = self._calculate_iris_offset(out_landmarks)
            
        return face_detected, yaw_angle, ear, iris_offset_ratio, out_landmarks
        
    def _estimate_yaw(self, landmarks):
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]
        
        eye_center_x = (left_eye.x + right_eye.x) / 2.0
        face_width = abs(right_eye.x - left_eye.x)
        
        if face_width == 0:
            face_width = 0.0001
            
        offset = nose_tip.x - eye_center_x
        normalized_offset = offset / face_width
        return (normalized_offset / 0.20) * 25.0

    def _distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _calculate_ear(self, landmarks):
        left_h = self._distance(landmarks[33], landmarks[133])
        left_v1 = self._distance(landmarks[160], landmarks[144])
        left_v2 = self._distance(landmarks[158], landmarks[153])
        ear_left = (left_v1 + left_v2) / (2.0 * left_h + 1e-6)

        right_h = self._distance(landmarks[362], landmarks[263])
        right_v1 = self._distance(landmarks[385], landmarks[380])
        right_v2 = self._distance(landmarks[387], landmarks[373])
        ear_right = (right_v1 + right_v2) / (2.0 * right_h + 1e-6)
        
        return (ear_left + ear_right) / 2.0

    def _calculate_iris_offset(self, landmarks):
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        
        left_inner = landmarks[133]
        left_outer = landmarks[33]
        
        right_inner = landmarks[362]
        right_outer = landmarks[263]

        left_eye_width = abs(left_inner.x - left_outer.x) + 1e-6
        left_iris_pos = (left_iris.x - left_outer.x) / left_eye_width - 0.5
        
        right_eye_width = abs(right_outer.x - right_inner.x) + 1e-6
        right_iris_pos = (right_iris.x - right_inner.x) / right_eye_width - 0.5
        
        return (left_iris_pos + right_iris_pos) / 2.0

    def close(self):
        pass
