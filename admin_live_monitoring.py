"""
Final Merged Admin Live Monitoring Module - ALL FEATURES INCLUDED
Features:
1. Real-time Face Detection (Dynamic)
2. Eye Gaze Tracking (MediaPipe)
3. Object Detection (Cell Phone & Book with Color Coding)
4. Audio Monitoring & Recording
5. Auto-Save Violations (Video + JSON)
"""

import base64
import threading
import time
import os
import json
import sys
import re
from datetime import datetime
from collections import deque
ALLOW_SERVER_CAMERA_FALLBACK = (os.getenv('ALLOW_SERVER_CAMERA_FALLBACK', '0') == '1')
ALLOW_SERVER_AUDIO_MONITOR = (os.getenv('ALLOW_SERVER_AUDIO_MONITOR', '1') == '1')

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Try importing cv2 with fallback
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None
    print("⚠️ OpenCV (cv2) not available.")

# MediaPipe for advanced face mesh and eye tracking
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    mp_python = None
    vision = None
    print("MediaPipe not available. Install: pip install mediapipe")

# Audio detection/recording (Windows-friendly backends)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except Exception:
    pyaudio = None
    PYAUDIO_AVAILABLE = False

try:
    import soundfile as sf
except Exception:
    sf = None

AUDIO_AVAILABLE = SOUNDDEVICE_AVAILABLE or PYAUDIO_AVAILABLE
if not AUDIO_AVAILABLE:
    print("Audio libraries not available (sounddevice/PyAudio).")

# Sensitivity configuration (env-tunable)
VOICE_SAMPLE_RATE = int(os.getenv('VOICE_SAMPLE_RATE', '16000'))
_VOICE_CHUNK_DEFAULT = max(512, VOICE_SAMPLE_RATE // 4)  # ~0.25s windows for faster detection
VOICE_CHUNK_SIZE = int(os.getenv('VOICE_CHUNK_SIZE', str(_VOICE_CHUNK_DEFAULT)))
VOICE_RMS_THRESHOLD = float(os.getenv('VOICE_RMS_THRESHOLD', '0.010'))  # lowered: catch soft voices
VOICE_NOISE_MULTIPLIER = float(os.getenv('VOICE_NOISE_MULTIPLIER', '1.4'))  # less suppression
VOICE_NOISE_FLOOR_MIN = float(os.getenv('VOICE_NOISE_FLOOR_MIN', '0.005'))
VOICE_CONTINUOUS_SECONDS = float(os.getenv('VOICE_CONTINUOUS_SECONDS', '8.0'))  # Trigger in 8.0s per user request
VOICE_SILENCE_RESET_SECONDS = float(os.getenv('VOICE_SILENCE_RESET_SECONDS', '1.0'))
VOICE_BACKEND = (os.getenv('VOICE_BACKEND', 'auto') or 'auto').strip().lower()  # auto|sounddevice|pyaudio
VOICE_DEVICE_INDEX = os.getenv('VOICE_DEVICE_INDEX', '').strip()  # optional int
VOICE_SAVE_RAW_AUDIO = (os.getenv('VOICE_SAVE_RAW_AUDIO', '1') == '1')

# Face landmarks fallback (used when MediaPipe is unavailable)
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except Exception:
    face_recognition = None
    FACE_REC_AVAILABLE = False

# OCR (optional) for detecting written material / notes/books pages
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False

# ============================================================================
# Camera Simulator
# ============================================================================
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except Exception:
    face_recognition = None
    FACE_REC_AVAILABLE = False

def detect_gaze_fallback(frame):
    """Fallback gaze and eye-closure detection when MediaPipe is unavailable."""
    if not FACE_REC_AVAILABLE or frame is None or not CV2_AVAILABLE:
        return "Center", False
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_list = face_recognition.face_landmarks(rgb)
        if not landmarks_list:
            return "Center", False
        lm = landmarks_list[0]
        left_eye = lm.get('left_eye', [])
        right_eye = lm.get('right_eye', [])
        nose_bridge = lm.get('nose_bridge', [])
        if not left_eye or not right_eye or not nose_bridge:
            return "Center", False

        left_eye_center = np.mean(np.array(left_eye), axis=0)
        right_eye_center = np.mean(np.array(right_eye), axis=0)
        eye_mid_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
        eye_span = max(1.0, abs(right_eye_center[0] - left_eye_center[0]))
        nose_x = float(nose_bridge[-1][0])
        ratio = (nose_x - eye_mid_x) / eye_span

        if ratio < -0.08:
            gaze = "Looking Left"
        elif ratio > 0.08:
            gaze = "Looking Right"
        else:
            gaze = "Center"

        def eye_open_ratio(points):
            pts = np.array(points, dtype=np.float32)
            min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
            min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
            w = max(1.0, max_x - min_x)
            h = max(0.0, max_y - min_y)
            return h / w

        eyes_closed = ((eye_open_ratio(left_eye) + eye_open_ratio(right_eye)) / 2.0) < 0.18
        return gaze, eyes_closed
    except Exception:
        return "Center", False

class CameraSimulator:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_count = 0
        
    def isOpened(self):
        return True
        
    def read(self):
        if not CV2_AVAILABLE:
            return True, None
            
        # Create a synthetic frame
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 100
        
        cv2.putText(frame, "CAMERA SIMULATOR", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Simulated Face
        center_x, center_y = 320, 240
        cv2.ellipse(frame, (center_x, center_y), (80, 100), 0, 0, 360, (255, 200, 150), -1)
        cv2.circle(frame, (center_x - 30, center_y - 20), 10, (50, 50, 50), -1)
        cv2.circle(frame, (center_x + 30, center_y - 20), 10, (50, 50, 50), -1)
        
        # Moving indicator
        x = (self.frame_count * 2) % (self.width - 100)
        cv2.circle(frame, (x + 50, 400), 10, (0, 255, 0), -1)
        
        self.frame_count += 1
        return True, frame


# ============================================================================
# Audio Monitor
# ============================================================================
class AudioMonitor:
    """Background RMS voice detector with sounddevice/PyAudio fallback."""

    def __init__(self, student_id, student_name):
        self.student_id = student_id
        self.student_name = student_name
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = VOICE_SAMPLE_RATE
        self.chunk_size = VOICE_CHUNK_SIZE
        self.min_threshold = VOICE_RMS_THRESHOLD
        self.noise_multiplier = VOICE_NOISE_MULTIPLIER
        self.noise_floor_min = VOICE_NOISE_FLOOR_MIN
        self.continuous_seconds = VOICE_CONTINUOUS_SECONDS
        self.silence_reset_seconds = VOICE_SILENCE_RESET_SECONDS
        self.noise_floor = None
        self.voice_start_ts = None
        self.last_voice_ts = 0.0
        self.recording_thread = None
        self._event_lock = threading.Lock()
        self._violation_event = False
        self.backend = "none"
        self._last_rms_log_ts = 0.0

    def start_monitoring(self):
        """Start audio monitoring in background thread."""
        if not AUDIO_AVAILABLE:
            print(f"Audio monitoring not available for {self.student_name}")
            return

        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.recording_thread.start()
        print(f"Audio monitoring started for {self.student_name}")

    def consume_violation_event(self):
        """Return True once per detected continuous-voice event."""
        with self._event_lock:
            if self._violation_event:
                self._violation_event = False
                return True
            return False

    def _mark_voice_event(self):
        with self._event_lock:
            self._violation_event = True

    def _process_audio_block(self, samples):
        if samples is None or len(samples) == 0:
            return

        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
        now = time.time()
        zcr = 0.0
        dom_freq = 0.0
        band_ratio = 0.0
        voice_like = False

        if self.noise_floor is None:
            self.noise_floor = max(self.noise_floor_min, rms)
        elif rms < (self.min_threshold * 1.2):
            # Adapt baseline on lower-energy windows only to ignore speech peaks.
            self.noise_floor = (0.97 * self.noise_floor) + (0.03 * rms)

        dynamic_threshold = max(
            self.min_threshold * 0.8,
            self.noise_floor_min * 0.8,
            float(self.noise_floor or self.noise_floor_min) * (self.noise_multiplier * 0.85)
        )

        # Strict voice gate for human speech-like audio.
        if len(samples) >= 512:
            try:
                x = np.asarray(samples, dtype=np.float32)
                x = np.clip(x, -1.0, 1.0)
                zcr = float(np.mean(np.abs(np.diff(np.signbit(x)).astype(np.float32))))

                window = np.hanning(len(x)).astype(np.float32)
                spec = np.abs(np.fft.rfft(x * window))
                freqs = np.fft.rfftfreq(len(x), d=1.0 / float(self.sample_rate))
                total_energy = float(np.sum(spec) + 1e-9)

                voice_band = (freqs >= 85.0) & (freqs <= 3400.0)
                band_energy = float(np.sum(spec[voice_band])) if np.any(voice_band) else 0.0
                band_ratio = band_energy / total_energy

                if np.any(voice_band):
                    band_spec = spec[voice_band]
                    band_freqs = freqs[voice_band]
                    dom_freq = float(band_freqs[int(np.argmax(band_spec))])

                voice_like = (
                    (band_ratio >= 0.22) and
                    (70.0 <= dom_freq <= 3400.0) and
                    (0.002 <= zcr <= 0.50)
                )
            except Exception:
                voice_like = False

        if (now - self._last_rms_log_ts) >= 1.0:
            print(
                f"[audio][{self.student_id}] rms={rms:.4f} thr={dynamic_threshold:.4f} "
                f"noise_floor={float(self.noise_floor or 0.0):.4f} zcr={zcr:.3f} "
                f"dom={dom_freq:.1f}Hz band={band_ratio:.2f} voice_like={voice_like}"
            )
            self._last_rms_log_ts = now

        if (rms >= dynamic_threshold) and voice_like:
            self.last_voice_ts = now
            if self.voice_start_ts is None:
                self.voice_start_ts = now
            elif (now - self.voice_start_ts) >= self.continuous_seconds:
                self._mark_voice_event()
                # Prevent repeated flags for one long utterance.
                self.voice_start_ts = now
        else:
            if self.voice_start_ts is not None and (now - self.last_voice_ts) >= self.silence_reset_seconds:
                self.voice_start_ts = None

    def _record_sounddevice(self):
        device_index = None
        if VOICE_DEVICE_INDEX:
            try:
                device_index = int(VOICE_DEVICE_INDEX)
            except Exception:
                device_index = None

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            mono = np.asarray(indata[:, 0], dtype=np.float32)
            if VOICE_SAVE_RAW_AUDIO:
                self.audio_data.append(mono.copy())
            self._process_audio_block(mono)

        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype='float32',
            device=device_index
        ):
            while self.is_recording:
                time.sleep(0.08)

    def _record_pyaudio(self):
        input_device_index = None
        if VOICE_DEVICE_INDEX:
            try:
                input_device_index = int(VOICE_DEVICE_INDEX)
            except Exception:
                input_device_index = None
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=input_device_index
        )
        try:
            while self.is_recording:
                raw = stream.read(self.chunk_size, exception_on_overflow=False)
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if VOICE_SAVE_RAW_AUDIO:
                    self.audio_data.append(pcm.copy())
                self._process_audio_block(pcm)
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            pa.terminate()

    def _record_audio(self):
        """Background thread for microphone processing."""
        tried = []
        if VOICE_BACKEND == 'sounddevice':
            backend_order = ['sounddevice']
        elif VOICE_BACKEND == 'pyaudio':
            backend_order = ['pyaudio']
        else:
            backend_order = ['sounddevice', 'pyaudio']

        for backend in backend_order:
            try:
                if backend == 'sounddevice' and SOUNDDEVICE_AVAILABLE and sd is not None:
                    self.backend = "sounddevice"
                    self._record_sounddevice()
                    return
                if backend == 'pyaudio' and PYAUDIO_AVAILABLE and pyaudio is not None:
                    self.backend = "pyaudio"
                    self._record_pyaudio()
                    return
            except Exception as e:
                tried.append(f"{backend}: {e}")

        self.backend = "none"
        if tried:
            print(f"Audio recording failed for {self.student_name}. Tried -> {' | '.join(tried)}")
        else:
            print(f"No audio backend available for {self.student_name}")

    def stop_monitoring(self):
        """Stop audio monitoring and optionally save raw audio."""
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)

        if VOICE_SAVE_RAW_AUDIO and self.audio_data and AUDIO_AVAILABLE and sf is not None:
            try:
                audio_array = np.concatenate([np.asarray(x, dtype=np.float32) for x in self.audio_data], axis=0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = ''.join(ch if ch.isalnum() else '_' for ch in str(self.student_name)).strip('_') or f"student_{self.student_id}"
                filename = f"static/audio_recordings/{self.student_id}_{safe_name}_{timestamp}.wav"

                os.makedirs("static/audio_recordings", exist_ok=True)
                sf.write(filename, audio_array, self.sample_rate)
                print(f"Audio saved: {filename}")
                return filename
            except Exception as e:
                print(f"Audio save error: {e}")
        return None

# ============================================================================
class ViolationAutoSaver:
    """Automatically saves all violations during exam."""

    def __init__(self, student_id, student_name):
        self.student_id = student_id
        self.student_name = student_name
        self.violations = []
        self.video_writer = None
        self.is_recording = False
        self.session_start = datetime.now()
        self.output_path = None
        self.session_key = None
        self.codec_used = None
        self.frames_written = 0

    def start_session(self):
        """Start exam session recording."""
        try:
            timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
            safe_name = ''.join(ch if ch.isalnum() else '_' for ch in str(self.student_name)).strip('_') or f"student_{self.student_id}"
            self.session_key = f"{self.student_id}_{safe_name}_{timestamp}"
            output_dir = "static/exam_sessions"
            os.makedirs(output_dir, exist_ok=True)

            self.output_path = os.path.join(output_dir, f"{self.session_key}.mp4")

            if CV2_AVAILABLE:
                # Try multiple codecs for Windows/OpenCV compatibility.
                codec_candidates = ['mp4v', 'avc1', 'H264', 'XVID', 'MJPG']
                self.video_writer = None
                self.codec_used = None
                for codec in codec_candidates:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        writer = cv2.VideoWriter(self.output_path, fourcc, 12.0, (640, 480))
                        if writer is not None and writer.isOpened():
                            self.video_writer = writer
                            self.codec_used = codec
                            break
                        if writer is not None:
                            writer.release()
                    except Exception:
                        continue

                if self.video_writer is not None:
                    self.is_recording = True
                    print(f"Session recording started: {self.output_path} (codec={self.codec_used})")
                else:
                    self.is_recording = False
                    print(f"Video writer init failed for session: {self.output_path}")
        except Exception as e:
            print(f"Error starting session recording: {e}")

    def add_violation(self, violation_type, severity="medium", frame=None):
        """Add a violation to the log."""
        violation = {
            "type": violation_type,
            "severity": severity,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_id": self.student_id,
            "student_name": self.student_name,
        }
        self.violations.append(violation)

        if frame is not None:
            self.add_frame(frame)

    def add_frame(self, frame):
        """Continuously write session frames so every attempt has evidence."""
        if frame is None or self.video_writer is None or not self.is_recording:
            return
        try:
            if not isinstance(frame, np.ndarray):
                return
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            self.video_writer.write(frame_resized)
            self.frames_written += 1
        except Exception as e:
            print(f"Frame write error: {e}")

    def end_session(self):
        """End exam session and save all data."""
        try:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False

            session_data = {
                "student_id": self.student_id,
                "student_name": self.student_name,
                "session_start": self.session_start.strftime("%Y-%m-%d %H:%M:%S"),
                "session_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_violations": len(self.violations),
                "violations": self.violations,
                "video_path": self.output_path,
                "codec": self.codec_used,
                "frames_written": int(self.frames_written),
            }

            safe_key = self.session_key or f"{self.student_id}_{int(time.time())}"
            json_path = f"static/exam_sessions/{safe_key}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=4)

            print(f"Session data saved: {json_path}")
            return session_data
        except Exception as e:
            print(f"Error ending session: {e}")
            return None


# ============================================================================
class AdminMonitor:
    """Enhanced admin monitoring with automatic streaming on exam start"""
    
    def __init__(self, socketio_instance, fps=5, warning_system=None):
        self.socketio = socketio_instance
        self.fps = fps
        self.warning_system = warning_system   # ← NEW: reference to WarningSystem
        self.running = {}
        self.lock = threading.Lock()
        self.student_info = {}
        self.eye_trackers = {}
        self.audio_monitors = {}
        self.auto_savers = {}
        # Track how many times each violation was seen (prevents spamming)
        self.violation_cooldown = {}   # student_id -> {vtype: last_warned_count}
        self.face_motion_state = {}    # student_id -> {'cx': float, 'cy': float, 'rapid_frames': int}
        self.ocr_state = {}            # student_id -> {'last_run_frame': int, 'last_text': str}
        self.face_detector = None
        if hasattr(mp, 'tasks') and hasattr(mp.tasks, 'vision'):
            try:
                base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task') # Using face landmarker instead since we have it downloaded
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=5,
                    min_face_detection_confidence=0.25,
                    min_face_presence_confidence=0.25,
                    min_tracking_confidence=0.25
                )
                self.face_detector = vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                print(f"Failed to load alternate face detector for counting: {e}")
                self.face_detector = None

    def _count_faces_fusion(self, frame, haar_faces):
        """Use Haar + MediaPipe + face_recognition(HOG) and keep highest count."""
        count = len(haar_faces) if haar_faces else 0
        if frame is None or not CV2_AVAILABLE:
            return count
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if hasattr(self, 'face_detector') and getattr(self, 'face_detector', None) is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = self.face_detector.detect(mp_image)
                mp_count = len(results.detections) if results and hasattr(results, 'detections') and results.detections else 0
                count = max(count, mp_count)
            elif FACE_REC_AVAILABLE:
                # HOG fallback catches many multi-face cases missed by Haar.
                h, w = frame.shape[:2]
                scale = 1.0
                if max(h, w) > 960:
                    scale = 960.0 / float(max(h, w))
                if scale < 1.0:
                    small = cv2.resize(rgb, (int(w * scale), int(h * scale)))
                else:
                    small = rgb
                fr_locs = face_recognition.face_locations(small, model='hog')
                fr_count = len(fr_locs)
                count = max(count, fr_count)
            return int(count)
        except Exception:
            return count

    def _detect_written_material_ocr(self, frame, student_id, frame_count):
        """
        OCR-based detector for notes/books/pages visible in frame.
        Returns (detected: bool, details: str).
        """
        if frame is None or not CV2_AVAILABLE:
            return False, ''
        # Limit OCR frequency for performance.
        st = self.ocr_state.get(student_id, {'last_run_frame': -9999, 'last_text': ''})
        if (frame_count - int(st.get('last_run_frame', -9999))) < 10:
            return False, ''
        st['last_run_frame'] = int(frame_count)
        self.ocr_state[student_id] = st
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 9)
            if OCR_AVAILABLE:
                text = pytesseract.image_to_string(
                    bw,
                    config='--oem 3 --psm 6 -c preserve_interword_spaces=1'
                )
                cleaned = re.sub(r'[^A-Za-z0-9\s]', ' ', text or '')
                words = [w for w in cleaned.split() if len(w) >= 3]
                if len(words) >= 12:
                    excerpt = ' '.join(words[:10])
                    st['last_text'] = excerpt
                    self.ocr_state[student_id] = st
                    return True, f"Written material detected: {excerpt}"
                return False, ''
            # Fallback (no OCR lib): text-like edge density heuristic
            edges = cv2.Canny(bw, 60, 140)
            edge_density = float(np.count_nonzero(edges)) / float(edges.size or 1)
            if edge_density > 0.12:
                return True, "Text-like written material pattern detected"
            return False, ''
        except Exception:
            return False, ''
        
    def start_monitoring(self, student_id, student_name):
        """Start comprehensive monitoring for a student"""
        with self.lock:
            if student_id in self.running and self.running[student_id]:
                print(f"⚠️ Monitoring already active for {student_name}")
                return
                
            self.running[student_id] = True
            self.student_info[student_id] = {
                'name': student_name,
                'violations': [],
                'start_time': datetime.now(),
                'frame_count': 0
            }
        
        # Initialize subsystems
        self.eye_trackers[student_id] = EyeGazeTracker()
        
        if AUDIO_AVAILABLE and ALLOW_SERVER_AUDIO_MONITOR:
            audio_monitor = AudioMonitor(student_id, student_name)
            audio_monitor.start_monitoring()
            self.audio_monitors[student_id] = audio_monitor
        
        auto_saver = ViolationAutoSaver(student_id, student_name)
        auto_saver.start_session()
        self.auto_savers[student_id] = auto_saver
        
        print(f"✅ Enhanced monitoring started for {student_name} (ID: {student_id})")
        
        # Notify Admin Dashboard
        if self.socketio:
            self.socketio.emit('students_list', {'students': [{
                'student_id': student_id, 
                'student_name': student_name, 
                'warnings': 0, 
                'violations': []
            }]}, namespace='/admin')
        
        # Start streaming thread
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        executor.submit(self._enhanced_stream_loop, student_id)
        
        print(f"🎥 Camera streaming thread started for {student_name}")
    
    def stop_monitoring(self, student_id):
        """Stop monitoring and save all data"""
        with self.lock:
            self.running[student_id] = False
        
        if student_id in self.audio_monitors:
            self.audio_monitors[student_id].stop_monitoring()
            del self.audio_monitors[student_id]
        
        if student_id in self.auto_savers:
            self.auto_savers[student_id].end_session()
            del self.auto_savers[student_id]
        
        if student_id in self.eye_trackers:
            del self.eye_trackers[student_id]
        
        with self.lock:
            if student_id in self.student_info:
                del self.student_info[student_id]
        self.face_motion_state.pop(student_id, None)
        self.ocr_state.pop(student_id, None)
        
        print(f"✅ Monitoring stopped for student {student_id}")
    
    def _enhanced_stream_loop(self, student_id):
        """Main streaming loop with all detections"""
        try:
            # Import here to avoid circular dependency
            from app import (
                camera_streamer, detect_faces,
                detect_people_opencv, get_latest_student_frame, active_exam_students, active_exam_students_lock,
                _persist_behavior_violation
            )
            
            student_name = self.student_info.get(student_id, {}).get('name', 'Unknown')
            frame_count = 0
            violation_buffer = deque(maxlen=5)
            
            eye_tracker = self.eye_trackers.get(student_id)
            auto_saver = self.auto_savers.get(student_id)
            audio_monitor = self.audio_monitors.get(student_id)
            
            print(f"🎥 Starting enhanced video stream for {student_name}")
            
            # Ensure camera is running only when explicit server fallback is enabled.
            if ALLOW_SERVER_CAMERA_FALLBACK and (not camera_streamer.running):
                try:
                    camera_streamer.start()
                except Exception as e:
                    print(f"❌ Failed to start camera: {e}")
            
            while True:
                # Check running status
                with self.lock:
                    if not self.running.get(student_id, False):
                        print(f"🛑 Stopping stream for {student_name}")
                        break
                
                try:
                    # 1. GET PRE-PROCESSED FRAME FROM APP.PY
                    # Use get_latest_student_frame from app.py
                    item = {}
                    with latest_student_frames_lock:
                        # Access the shared dictionary directly to get the rich item object
                        item = latest_student_frames.get(str(student_id), {})
                    
                    frame = item.get('processed_frame')
                    b64_frame = item.get('processed_frame_b64')
                    snapshot = item.get('status_snapshot', {})
                    last_labels = item.get('last_prohibited_object_labels', [])
                    no_live_frame = False

                    if frame is None and ALLOW_SERVER_CAMERA_FALLBACK:
                        try:
                            frame = camera_streamer.read()
                        except Exception as cam_err:
                            print(f"❌ Camera read error: {cam_err}")
                            time.sleep(0.5)
                            continue
                            
                    if frame is None:
                        no_live_frame = True
                        if CV2_AVAILABLE and np is not None:
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame, "NO STUDENT CAMERA FRAME", (120, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            b64_frame = base64.b64encode(jpeg.tobytes()).decode('ascii')
                        else:
                            time.sleep(0.4)
                            continue

                    if auto_saver and frame is not None:
                        auto_saver.add_frame(frame)
                    
                    # 2. AUDIO MODALITY (Only process audio here)
                    audio_violation = False
                    if audio_monitor and audio_monitor.consume_violation_event():
                        audio_violation = True
                        violation_buffer.append("audio_detected")
                    
                    if audio_violation and self.warning_system:
                        try:
                            print(f"🎙️ Audio violation → WARNING: student={student_id}")
                            terminated = self.warning_system.add_warning(student_id, "VOICE_DETECTED", "Voice/Noise Detected")
                            if terminated:
                                print(f"🛑 EXAM TERMINATED for student {student_id} - 3 warnings reached")
                                with self.lock:
                                    self.running[student_id] = False
                        except Exception as e:
                            print(f"⚠️ Warning error: {e}")

                    # 3. EMIT FRAME TO ADMIN DASHBOARD
                    # Send the pre-processed B64 frame directly to ensure all bounding boxes (lines) are shown.
                    if b64_frame is None and frame is not None:
                        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        b64_frame = base64.b64encode(jpeg.tobytes()).decode('ascii')

                    if self.socketio and b64_frame:
                        try:
                            # Reconstruct simplified object list for UI badges based on the snapshot
                            prohibited_objects = [{'label': lbl} for lbl in last_labels]
                            all_objects = [{'label': lbl} for lbl in item.get('last_visible_object_labels', [])]
                            
                            self.socketio.emit('student_frame', {
                                'student_id':      student_id,
                                'student_name':    student_name,
                                'frame':           b64_frame,
                                'face_count':      snapshot.get('face_count', 0),
                                'face_status':     "Normal" if snapshot.get('face_count', 0) == 1 else "No Face" if snapshot.get('face_count', 0) == 0 else "Multiple Faces",
                                'gaze_direction':  snapshot.get('gaze_direction', 'CENTER'),
                                'eyes_closed':     snapshot.get('eyes_closed', False),
                                'objects':         prohibited_objects,
                                'all_objects':     all_objects,
                                'audio_violation': audio_violation,
                                'violations':      list(violation_buffer),
                                'warnings_count':  self.warning_system.get_warnings(student_id) if self.warning_system else 0,
                                'timestamp':       time.time()
                            }, namespace='/admin')
                        except Exception as e:
                            print(f"❌ SocketIO emit error: {e}")
                    
                    frame_count += 1
                    time.sleep(1.0 / max(self.fps, 1))
                
                except Exception as e:
                    print(f"❌ Error in stream loop: {e}")
                    time.sleep(1.0)
                    
        except Exception as e:
            print(f"❌ Fatal error in enhanced stream: {e}")
            import traceback
            traceback.print_exc()


def setup_admin_socketio(socketio, admin_monitor):
    """Setup SocketIO events for admin dashboard"""
    try:
        from flask_socketio import emit
        from flask import session
    except ImportError:
        print("⚠️ Flask-SocketIO not available")
        return

    @socketio.on('connect', namespace='/admin')
    def admin_connect():
        user = session.get('user')
        if not user or user.get('Role') != 'ADMIN':
            return False
        print('✅ Admin connected to enhanced monitoring dashboard')
        emit('connection_response', {'status': 'connected', 'features': [
            'Real-time face detection', 'Eye gaze tracking', 'Object detection (Phone/Book)', 
            'Audio monitoring', 'Auto-save violations'
        ]})
        try:
            with admin_monitor.lock:
                students = []
                for sid, info in admin_monitor.student_info.items():
                    students.append({
                        'student_id': sid,
                        'student_name': info.get('name', f'Student {sid}'),
                        'warnings': admin_monitor.warning_system.get_warnings(sid) if admin_monitor.warning_system else 0,
                        'violations': info.get('violations', []),
                    })
            emit('students_list', {'students': students})
        except Exception:
            emit('students_list', {'students': []})

    @socketio.on('disconnect', namespace='/admin')
    def admin_disconnect():
        print('⚠️ Admin disconnected')

    @socketio.on('request_student_feed', namespace='/admin')
    def handle_feed_request(data):
        student_id = data.get('student_id')
        emit('feed_started', {'student_id': student_id})

    @socketio.on('request_all_feeds', namespace='/admin')
    def handle_all_feeds_request():
        try:
            with admin_monitor.lock:
                students = []
                for sid, info in admin_monitor.student_info.items():
                    students.append({
                        'student_id': sid,
                        'student_name': info.get('name', f'Student {sid}'),
                        'warnings': admin_monitor.warning_system.get_warnings(sid) if admin_monitor.warning_system else 0,
                        'violations': info.get('violations', []),
                    })
            emit('students_list', {'students': students})
        except Exception:
            emit('students_list', {'students': []})

    @socketio.on('terminate_exam', namespace='/admin')
    def handle_terminate_exam(data):
        student_id = data.get('student_id')
        reason = data.get('reason', 'Manual termination by admin')
        
        print(f"❌ Admin terminating exam for student {student_id}: {reason}")
        admin_monitor.stop_monitoring(student_id)
        
        socketio.emit('exam_terminated', {'reason': reason, 'by': 'admin'}, namespace='/student')
        emit('termination_success', {'student_id': student_id})


# Module Initialization
print("=" * 70)
print("✅ Admin Monitoring Module Loaded — YOLOv11 Object Detection")
print(f"  - OpenCV:    {'✅ Available' if CV2_AVAILABLE else '❌ Not available'}")
print(f"  - MediaPipe: {'✅ Available' if MEDIAPIPE_AVAILABLE else '❌ Not available'}")
print(f"  - Audio:     {'✅ Available' if AUDIO_AVAILABLE else '❌ Not available'}")
print("  - Active Features:")
print("    ✓ Real-time Face Detection (Haar Cascade)")
print("    ✓ Eye Gaze Tracking (MediaPipe)")
print("    ✓ Object Detection — YOLOv11 (cell phone, book, laptop)")
print("    ✓ Instant warning on prohibited object detection")
print("    ✓ Auto-terminate after 3 warnings")
print("    ✓ Audio Monitoring")
print("    ✓ Auto-Save Violations (Video + JSON)")
print("=" * 70)
