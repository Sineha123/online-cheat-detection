import mediapipe as mp
import math
import threading

_mp_face_mesh = mp.solutions.face_mesh
_global_face_mesh = _mp_face_mesh.FaceMesh(
    static_image_mode=False,  # use video mode for speed + temporal smoothing
    max_num_faces=3,          # detect multiple faces in frame
    refine_landmarks=True, 
    min_detection_confidence=0.50,
    min_tracking_confidence=0.50,
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
        pitch_angle = 0.0
        ear = 0.0
        iris_offset_ratio = (0.0, 0.0)  # (x, y)
        out_landmarks = []
        
        if results.multi_face_landmarks:
            # Pick the largest face box to avoid side-face artifacts
            areas = []
            for lm in results.multi_face_landmarks:
                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                areas.append((area, lm))
            face_detected = True
            face_landmarks = max(areas, key=lambda t: t[0])[1] if areas else results.multi_face_landmarks[0]
            out_landmarks = face_landmarks.landmark
            
            yaw_angle, pitch_angle = self._estimate_yaw_pitch(out_landmarks)
            ear = self._calculate_ear(out_landmarks)
            iris_offset_ratio = self._calculate_iris_offsets(out_landmarks)
            
        return face_detected, yaw_angle, pitch_angle, ear, iris_offset_ratio, out_landmarks
        
    def _estimate_yaw_pitch(self, landmarks):
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]
        chin = landmarks[152]
        
        eye_center_x = (left_eye.x + right_eye.x) / 2.0
        eye_center_y = (left_eye.y + right_eye.y) / 2.0
        face_width = abs(right_eye.x - left_eye.x)
        face_height = abs(chin.y - eye_center_y)
        
        if face_width == 0:
            face_width = 0.0001
        if face_height == 0:
            face_height = 0.0001
            
        # Yaw
        offset_x = nose_tip.x - eye_center_x
        normalized_offset_x = offset_x / face_width
        yaw = (normalized_offset_x / 0.20) * 25.0

        # Pitch
        # Standardize center offset to assume camera is roughly straight-on
        offset_y = nose_tip.y - eye_center_y 
        normalized_offset_y = offset_y / face_height
        # Approximate scaling to degrees using standard face ratio ~0.50 offset at rest
        pitch = ((normalized_offset_y - 0.45) / 0.20) * 20.0
        
        return yaw, pitch

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

    def _calculate_iris_offsets(self, landmarks):
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        
        left_inner = landmarks[133]
        left_outer = landmarks[33]
        
        right_inner = landmarks[362]
        right_outer = landmarks[263]

        left_eye_width = abs(left_inner.x - left_outer.x) + 1e-6
        left_eye_height = abs(landmarks[159].y - landmarks[145].y) + 1e-6
        left_iris_pos = (left_iris.x - left_outer.x) / left_eye_width - 0.5
        left_iris_pos_y = (left_iris.y - landmarks[145].y) / left_eye_height - 0.5
        
        right_eye_width = abs(right_outer.x - right_inner.x) + 1e-6
        right_eye_height = abs(landmarks[386].y - landmarks[374].y) + 1e-6
        right_iris_pos = (right_iris.x - right_inner.x) / right_eye_width - 0.5
        right_iris_pos_y = (right_iris.y - landmarks[374].y) / right_eye_height - 0.5
        
        iris_x = (left_iris_pos + right_iris_pos) / 2.0
        iris_y = (left_iris_pos_y + right_iris_pos_y) / 2.0
        return (iris_x, iris_y)

    def close(self):
        pass
