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
        Simplified: ONLY checks for 'Face not detected'
        """
        active_alerts = []
        state = self.get_state(sid)
        current_time = time.time()
        
        # Rule 1 - No Face (ONLY Rule Active)
        if self._check_condition(state, "no_face", not face_detected, config.TIME_NO_FACE, current_time):
            active_alerts.append("Face not detected")
            
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

