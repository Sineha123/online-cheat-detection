import time
import config_vision as config
import threading
import time  # used for debounced warning timing

class DecisionEngine:
    def __init__(self):
        self.student_states = {}
        self._lock = threading.Lock()

    def get_state(self, sid):
        with self._lock:
            if sid not in self.student_states:
                self.student_states[sid] = {
                    "warning_count": 0,
                    "last_warning_time": 0.0,
                    "condition_start": {},
                    "frame_streaks": {}  # condition_name -> consecutive frames active
                }
            return self.student_states[sid]

    def reset_student(self, sid):
        with self._lock:
            self.student_states[str(sid)] = {
                "warning_count": 0,
                "last_warning_time": 0.0,
                "condition_start": {},
                "frame_streaks": {}
            }

    def add_penalty(self, sid, amount):
        state = self.get_state(sid)
        with self._lock:
            state["warning_count"] += amount
            state["last_warning_time"] = time.time()
        return state["warning_count"]

    def _check_condition(self, state, condition_name, is_active, required_time, current_time):
        if is_active:
            if condition_name not in state["condition_start"]:
                state["condition_start"][condition_name] = current_time
            elif current_time - state["condition_start"][condition_name] >= required_time:
                return True
        else:
            if condition_name in state["condition_start"]:
                del state["condition_start"][condition_name]
        return False

    def _check_condition_frames(self, state, condition_name, is_active, required_frames):
        """
        Frame-based debouncing. Returns True once the condition has been active
        for the requested consecutive frame count.
        """
        streaks = state.setdefault("frame_streaks", {})
        current = streaks.get(condition_name, 0)
        if is_active:
            current += 1
            streaks[condition_name] = current
            if current >= required_frames:
                return True
        else:
            streaks[condition_name] = 0
        return False

    def evaluate(self, sid, face_detected, person_count, yaw_angle, pitch_angle, ear, iris_offset_ratio, banned_objects):
        """
        Evaluates signals and returns active UI alerts and the total penalty score.
        """
        active_alerts = []
        state = self.get_state(sid)
        current_time = time.time()
        iris_x = 0.0
        iris_y = 0.0
        if isinstance(iris_offset_ratio, (list, tuple)) and len(iris_offset_ratio) >= 2:
            iris_x, iris_y = float(iris_offset_ratio[0]), float(iris_offset_ratio[1])
        else:
            iris_x = float(iris_offset_ratio or 0.0)
            iris_y = 0.0
            
        penalty_increment = 0
        
        # Rule 1 - No Face / Hands covering face
        if self._check_condition(state, "no_face", not face_detected, config.TIME_NO_FACE, current_time):
            active_alerts.append("Face not detected")
            penalty_increment += 20

        # Rule 1b - Eyes / gaze off-screen (horizontal AND vertical)
        gaze_off_horizontal = (abs(iris_x) > config.IRIS_OFFSET_THRESHOLD)
        gaze_off_vertical = (abs(iris_y) > getattr(config, 'IRIS_OFFSET_THRESHOLD_Y', 0.20))
        gaze_off = gaze_off_horizontal or gaze_off_vertical
        if self._check_condition(state, "gaze_off", gaze_off, config.TIME_GAZING, current_time):
            if gaze_off_horizontal and abs(iris_x) >= abs(iris_y):
                direction = "left" if iris_x < 0 else "right"
            else:
                direction = "up" if iris_y < 0 else "down"
            active_alerts.append(f"Gaze off screen ({direction})")
            penalty_increment += 10

        # Rule 1c - Head turned away (Yaw and Pitch)
        yaw_flag = abs(yaw_angle) > config.YAW_THRESHOLD_DEG
        pitch_flag = abs(pitch_angle) > getattr(config, 'PITCH_THRESHOLD_DEG', 20.0)
        head_turned = yaw_flag or pitch_flag
        
        if self._check_condition(state, "head_turned", head_turned, config.TIME_HEAD_TURNED, current_time):
            if yaw_flag and abs(yaw_angle) >= abs(pitch_angle):
                direction = "left" if yaw_angle < 0 else "right"
            else:
                direction = "up" if pitch_angle < 0 else "down"
            active_alerts.append(f"Head turned away ({direction})")
            penalty_increment += 10

        # Rule 2 - Prohibited object detection
        has_banned_object = bool(banned_objects)
        # Require consecutive frames with a prohibited object to reduce flicker/shadows
        if self._check_condition_frames(state, "banned_object", has_banned_object, getattr(config, "BANNED_FRAMES_REQUIRED", 12)):
            labels = sorted({str(obj.get("label", "Object")).lower() for obj in banned_objects})
            label_str = ", ".join(labels) or "Object"
            active_alerts.append(f"Prohibited object detected: {label_str}")
            
            # Weighted penalty for explicitly dangerous items
            if any(p in label_str for p in ['phone', 'cell', 'smart']):
                penalty_increment += 40
            else:
                penalty_increment += 30

        # Rule 3 - Multiple persons/faces in frame
        has_multiple_persons = int(person_count or 0) > 1
        if self._check_condition(state, "multiple_persons", has_multiple_persons, config.TIME_MULTIPLE_PERSONS, current_time):
            active_alerts.append(f"Multiple persons detected ({int(person_count)})")
            penalty_increment += 50
            
        # --- Rule Aggregation and Scoring Logic ---
        with self._lock:
            # Always apply the raw penalty accumulation
            if penalty_increment > 0:
                # Add a universal cooldown so scores don't skyrocket continuously every exact frame
                if current_time - state["last_warning_time"] > getattr(config, 'WARNING_COOLDOWN_SEC', 3.0):
                    state["warning_count"] += penalty_increment
                    state["last_warning_time"] = current_time
                    
            return active_alerts, state["warning_count"]
