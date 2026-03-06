import time
import config_vision as config
import threading

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
                    "condition_start": {}
                }
            return self.student_states[sid]

    def reset_student(self, sid):
        with self._lock:
            self.student_states[str(sid)] = {
                "warning_count": 0,
                "last_warning_time": 0.0,
                "condition_start": {}
            }

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

    def evaluate(self, sid, face_detected, person_count, yaw_angle, ear, iris_offset_ratio, banned_objects):
        """
        Evaluates signals and returns active UI alerts and the total penalty score.
        """
        active_alerts = []
        state = self.get_state(sid)
        current_time = time.time()
        
        # Rule 1 - No Face (ONLY Rule Active)
        if self._check_condition(state, "no_face", not face_detected, config.TIME_NO_FACE, current_time):
            active_alerts.append("Face not detected")

        # Rule 1b - Eyes / gaze off-screen
        gaze_off = abs(iris_offset_ratio) > config.IRIS_OFFSET_THRESHOLD
        if self._check_condition(state, "gaze_off", gaze_off, config.TIME_GAZING, current_time):
            direction = "left" if iris_offset_ratio < 0 else "right"
            active_alerts.append(f"Gaze off screen ({direction})")

        # Rule 1c - Head turned away
        if self._check_condition(state, "head_turned", abs(yaw_angle) > config.YAW_THRESHOLD_DEG, config.TIME_HEAD_TURNED, current_time):
            direction = "left" if yaw_angle < 0 else "right"
            active_alerts.append(f"Head turned away ({direction})")

        # Rule 2 - Prohibited object detection
        has_banned_object = bool(banned_objects)
        if self._check_condition(state, "banned_object", has_banned_object, config.TIME_BANNED_OBJECT, current_time):
            labels = ", ".join(sorted({str(obj.get("label", "Object")) for obj in banned_objects})) or "Object"
            active_alerts.append(f"Prohibited object detected: {labels}")

        # Rule 3 - Multiple persons/faces in frame
        has_multiple_persons = int(person_count or 0) > 1
        if self._check_condition(state, "multiple_persons", has_multiple_persons, config.TIME_MULTIPLE_PERSONS, current_time):
            active_alerts.append(f"Multiple persons detected ({int(person_count)})")
            
        # --- Rule Aggregation and Scoring Logic ---
        num_alerts = len(active_alerts)
        
        with self._lock:
            if num_alerts > 0:
                # Check if we should instantly flag them (>= 3 alerts)
                if num_alerts >= config.INSTANT_PENALTY_THRESHOLD:
                    state["warning_count"] += 1
                    state["last_warning_time"] = current_time
                # Otherwise, only flag if the cooldown period has elapsed
                elif current_time - state["last_warning_time"] > config.WARNING_COOLDOWN_SEC:
                    state["warning_count"] += 1
                    state["last_warning_time"] = current_time
                    
            return active_alerts, state["warning_count"]
