"""
Warning System Module
Handles student warnings and exam termination
"""

import threading
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

class WarningSystem:
    """Track warnings per student and emit events when thresholds reached."""
    
    def __init__(self, socketio_instance, admin_monitor=None, max_warnings=3):
        self.socketio = socketio_instance
        self.admin_monitor = admin_monitor
        self.max_warnings = max_warnings
        self.auto_terminate = False  # Default OFF — admin must explicitly enable
        self.lock = threading.Lock()
        self.warnings = {}  # student_id -> count
        self.violations = {}  # student_id -> list of violations
        self.student_names = {}  # student_id -> name
        self.last_warning_at = {}  # student_id -> epoch seconds
        self.last_warning_type_at = {}  # student_id -> {type: epoch seconds}

        # Apply a minimum gap for ALL warning increments — 3.0 seconds only.
        self.global_gap_seconds = 3.0
        self.type_gap_seconds = {
            'NO_FACE': 5.0,
            'MULTIPLE_FACES': 5.0,
            'VOICE_DETECTED': 5.0,
            'PROHIBITED_OBJECT': 5.0,
            'TAB_SWITCH': 4.0,
            'PROHIBITED_SHORTCUT': 4.0,
            'HEAD_MOVEMENT': 5.0,
            'DISTRACTION': 5.0,
            'STUDENT_LEFT_SEAT': 5.0,
            'EYES_CLOSED': 5.0,
            'GAZE_LEFT': 5.0,
            'GAZE_RIGHT': 5.0,
            'GAZE_UP': 5.0,
            'GAZE_DOWN': 5.0,
            'IDENTITY_MISMATCH': 5.0,
            'TERMINATED_BY_ADMIN': 0.0
        }

    def set_auto_terminate(self, enabled: bool):
        with self.lock:
            self.auto_terminate = enabled
        print(f"⚙️ WarningSystem Auto-Terminate configured: {self.auto_terminate}")

    def initialize_student(self, student_id, student_name):
        """Initialize tracking for a new student"""
        sid = str(student_id)
        with self.lock:
            self.warnings[sid] = 0
            self.violations[sid] = []
            self.student_names[sid] = student_name
            self.last_warning_at[sid] = 0.0
            self.last_warning_type_at[sid] = {}
        
        print(f"✓ Warning system initialized for student {student_id} - {student_name}")
        
        # Emit initial state to admin
        if self.socketio:
            self.socketio.emit('students_list', {'students': [
                {
                    'student_id': student_id, 
                    'student_name': student_name, 
                    'warnings': 0, 
                    'violations': []
                }
            ]}, namespace='/admin')

    def add_warning(self, student_id, vtype, details=None, emit_to_student=True):
        """Add warning and check if exam should be terminated"""
        sid = str(student_id)
        with self.lock:
            self.warnings.setdefault(sid, 0)
            self.violations.setdefault(sid, [])
            self.last_warning_at.setdefault(sid, 0.0)
            self.last_warning_type_at.setdefault(sid, {})

            now = time.time()
            vtype_norm = str(vtype or 'UNKNOWN').upper()
            g_gap = float(self.global_gap_seconds)
            t_gap = float(self.type_gap_seconds.get(vtype_norm, self.global_gap_seconds))
            last_global = float(self.last_warning_at.get(sid, 0.0))
            last_type = float(self.last_warning_type_at[sid].get(vtype_norm, 0.0))

            # Hard gap for all warnings: don't increment if warning is too soon.
            if (now - last_global) < g_gap or (now - last_type) < t_gap:
                return False
            
            # Increment warning count. If auto-terminate is true, cap at max_warnings so it doesn't inflate.
            if self.auto_terminate:
                if self.warnings[sid] < self.max_warnings:
                    self.warnings[sid] += 1
            else:
                self.warnings[sid] += 1
                
            self.last_warning_at[sid] = now
            self.last_warning_type_at[sid][vtype_norm] = now
            
            # Create violation record
            violation = {
                'type': vtype, 
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                'details': details
            }
            self.violations[sid].append(violation)
            
            count = self.warnings[sid]
            student_name = self.student_names.get(sid, 'Unknown')

        print(f"⚠ Warning #{count} for student {student_id}: {vtype} - {details}")

        # Emit to admin UI
        if self.socketio:
            self.socketio.emit('student_violation', {
                'student_id': student_id,
                'student_name': student_name,
                'total_warnings': min(count, self.max_warnings),
                'violation': violation,
                'type': vtype,
                'details': details,
                'source': 'server'
            }, namespace='/admin')
            
            # Emit to student UI (avoid client re-emitting)
            if emit_to_student:
                self.socketio.emit('student_violation', {
                    'student_id': student_id,
                    'student_name': student_name,
                    'total_warnings': min(count, self.max_warnings),
                    'violation': violation,
                    'type': vtype,
                    'details': details,
                    'source': 'server'
                }, namespace='/student')

        # If threshold reached, emit termination
        if count >= self.max_warnings:
            if not self.auto_terminate:
                # Let admin decide; notify them that student reached threshold
                if self.socketio and count == self.max_warnings:
                    self.socketio.emit('student_needs_review', {
                        'student_id': student_id,
                        'student_name': student_name,
                        'warnings': count
                    }, namespace='/admin')
                return False
                
            reason = f"Reached {self.max_warnings} warnings for violations: {vtype}"
            print(f"🚨 TERMINATING EXAM for student {student_id}: {reason}")
            
            if self.socketio:
                # Notify admin
                self.socketio.emit('student_exam_terminated', {
                    'student_id': student_id,
                    'student_name': student_name,
                    'reason': reason
                }, namespace='/admin')
                
                # Notify student
                self.socketio.emit('exam_terminated', {
                    'student_id': student_id,
                    'reason': reason,
                    'auto_terminated': True
                }, namespace='/student')
            
            return True  # terminated
        
        return False

    def get_warnings(self, student_id):
        """Get current warning count for student"""
        sid = str(student_id)
        with self.lock:
            return min(self.warnings.get(sid, 0), self.max_warnings)

    def get_violations(self, student_id):
        """Get all violations for student"""
        sid = str(student_id)
        with self.lock:
            return list(self.violations.get(sid, []))

    def reset_student(self, student_id):
        """Reset warnings for student"""
        sid = str(student_id)
        with self.lock:
            self.warnings[sid] = 0
            self.violations[sid] = []
            self.last_warning_at[sid] = 0.0
            self.last_warning_type_at[sid] = {}
        print(f"🧹 Cleared warnings for student {student_id}")

    def manually_terminate_student(self, student_id, reason="Manual termination by Admin"):
        """Instantly terminate a student exam regardless of warning count."""
        sid = str(student_id)
        with self.lock:
            student_name = self.student_names.get(sid, 'Unknown')
        
        print(f"🚨 MANUAL TERMINATE for student {student_id}: {reason}")
        if self.socketio:
            self.socketio.emit('student_exam_terminated', {
                'student_id': sid,
                'student_name': student_name,
                'reason': reason
            }, namespace='/admin')
            
            self.socketio.emit('exam_terminated', {
                'student_id': sid,
                'reason': reason,
                'auto_terminated': False
            }, namespace='/student')
        return True


class TabSwitchDetector:
    """Detects tab switching and adds warnings"""
    
    def __init__(self, warning_system, threshold=3):
        self.warning_system = warning_system
        self.threshold = threshold
        self.lock = threading.Lock()
        self.switch_counts = {}  # student_id -> count

    def initialize_student(self, student_id):
        """Initialize tab switch tracking for student"""
        with self.lock:
            self.switch_counts[student_id] = 0
        print(f"✓ Tab switch detector initialized for student {student_id}")

    def detect_tab_switch(self, student_id):
        """Detect tab switch and add warning if threshold reached"""
        sid = str(student_id)
        with self.lock:
            self.switch_counts.setdefault(sid, 0)
            self.switch_counts[sid] += 1
            count = self.switch_counts[sid]

        print(f"🔄 Tab switch detected for student {student_id}. Count: {count}/{self.threshold}")

        if count >= self.threshold:
            terminated = self.warning_system.add_warning(
                student_id, 
                'tab_switch', 
                f'{count} tab switches detected'
            )
            return {'terminated': terminated, 'count': count}
        
        return {'terminated': False, 'count': count}


print("=" * 60)
print("✓ Warning system module loaded successfully")
print("  - WarningSystem: ✓ Available")
print("  - TabSwitchDetector: ✓ Available") 
print("=" * 60)
