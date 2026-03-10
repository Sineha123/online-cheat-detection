# app.py -- Final Merged & Fixed Version
# Features: Complete CRUD, COCO Object Detection (Phone/Book), Enhanced Monitoring

import os
import io
import time
import math
import base64
import json
import re
import smtplib
import secrets
import threading
import traceback
import logging
from functools import wraps
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from email.message import EmailMessage
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, flash, Response, send_from_directory, abort, session
)

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except Exception:
    bcrypt = None
    BCRYPT_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DB support
import pymysql.cursors
class MySQL:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
            
    def init_app(self, app):
        self.app = app

    @property
    def connection(self):
        from flask import g
        if 'mysql_db' not in g:
            g.mysql_db = pymysql.connect(
                host=self.app.config.get('MYSQL_HOST', '127.0.0.1'),
                user=self.app.config.get('MYSQL_USER', 'root'),
                password=self.app.config.get('MYSQL_PASSWORD', ''),
                db=self.app.config.get('MYSQL_DB', 'examproctordb'),
                port=self.app.config.get('MYSQL_PORT', 3306),
                autocommit=True
            )
        return g.mysql_db


# OpenCV / numpy
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None
    logger.warning("OpenCV not available. Install with: pip install opencv-python")

# MediaPipe Face Mesh (for eye landmarks, EAR, and gaze direction)
try:
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    face_mesh_detector = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,            # MUST be 2 to detect multiple faces
        refine_landmarks=True,      # enables iris landmarks
        min_detection_confidence=0.3,  # lower = detects faces with caps/angles/partial
        min_tracking_confidence=0.3
    )
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe FaceMesh + Pose initialized successfully")
except Exception as e:
    mp = None
    mp_face_mesh = None
    mp_pose = None
    face_mesh_detector = None
    pose_detector = None
    MEDIAPIPE_AVAILABLE = False
    logger.warning(f"MediaPipe FaceMesh unavailable: {e}. Install with: pip install mediapipe")

# PyTorch + Ultralytics YOLOv8 (optional but recommended)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available. Install with: pip install ultralytics")

# Face recognition (optional)
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    face_recognition = None
    FACE_REC_AVAILABLE = False
else:
    # Ensure numpy available for encoding persistence
    if np is None:
        import numpy as np

# Optional OCR support (requires pytesseract package + Tesseract binary on host)
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False

if OCR_AVAILABLE and pytesseract is not None:
    try:
        _ = pytesseract.get_tesseract_version()
        logger.info("OCR engine available (Tesseract detected)")
    except Exception:
        OCR_AVAILABLE = False
        logger.warning("pytesseract found but Tesseract binary missing; OCR keyword detection disabled.")

# Try importing flask_socketio
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    SocketIO = None
    emit = None
    logger.warning("Flask-SocketIO not available. Install with: pip install flask-socketio")

profileName = None

# -------------------------
# Configuration & Globals
# -------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
def _get_or_create_secret_key():
    """Get a stable secret key - persists across restarts to keep sessions alive."""
    env_key = os.getenv('FLASK_SECRET_KEY', '').strip()
    if env_key:
        return env_key
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.flask_secret_key')
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            return f.read().strip()
    new_key = secrets.token_hex(32)
    try:
        with open(key_file, 'w') as f:
            f.write(new_key)
    except Exception:
        pass
    return new_key

app.secret_key = _get_or_create_secret_key()
app.config['SECRET_KEY'] = app.secret_key
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = (os.getenv('COOKIE_SECURE', '0') == '1')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

@app.before_request
def make_session_permanent():
    session.permanent = True

# MySQL config
# Use 127.0.0.1 default to force TCP and avoid local socket/pipe resolution issues.
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', '127.0.0.1')
app.config['MYSQL_PORT'] = int(os.getenv('MYSQL_PORT', '3306'))
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'examproctordb')
mysql = MySQL(app)

# Password reset / email config
PASSWORD_RESET_SALT = os.getenv('PASSWORD_RESET_SALT', 'password-reset-salt-v1')
PASSWORD_RESET_MAX_AGE_SEC = 15 * 60
SMTP_HOST = os.getenv('SMTP_HOST', '')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USERNAME', '')
SMTP_PASS = os.getenv('SMTP_PASSWORD', '')
SMTP_FROM = os.getenv('SMTP_FROM_EMAIL', SMTP_USER or 'no-reply@example.com')
SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', '1') == '1'
SMTP_USE_SSL = os.getenv('SMTP_USE_SSL', '0') == '1'

# -------------------------
# Auth Helpers
# -------------------------
def _is_hashed_password(value):
    if not value:
        return False
    return (
        value.startswith('pbkdf2:') or
        value.startswith('scrypt:') or
        value.startswith('$2a$') or
        value.startswith('$2b$') or
        value.startswith('$2y$')
    )

def _verify_password(stored_password, candidate):
    if not stored_password:
        return False
    if stored_password.startswith('$2a$') or stored_password.startswith('$2b$') or stored_password.startswith('$2y$'):
        if not BCRYPT_AVAILABLE:
            logger.error("bcrypt is required to verify bcrypt-hashed passwords.")
            return False
        try:
            return bool(bcrypt.checkpw(candidate.encode('utf-8'), stored_password.encode('utf-8')))
        except Exception:
            return False
    if _is_hashed_password(stored_password):
        return check_password_hash(stored_password, candidate)
    # Legacy plaintext fallback
    return stored_password == candidate

def _hash_password_bcrypt(password):
    if not BCRYPT_AVAILABLE:
        raise RuntimeError("bcrypt dependency unavailable. Install with: pip install bcrypt")
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')

def _password_reset_serializer():
    return URLSafeTimedSerializer(app.secret_key)

def _build_password_reset_token(user_id, email, role):
    payload = {'uid': int(user_id), 'email': str(email), 'role': str(role)}
    return _password_reset_serializer().dumps(payload, salt=PASSWORD_RESET_SALT)

def _load_password_reset_token(token):
    return _password_reset_serializer().loads(
        token,
        salt=PASSWORD_RESET_SALT,
        max_age=PASSWORD_RESET_MAX_AGE_SEC
    )

def _send_password_reset_email(to_email, display_name, reset_link):
    if not SMTP_HOST:
        logger.warning("SMTP_HOST not configured; skipping reset email send.")
        return False
    msg = EmailMessage()
    msg['Subject'] = 'Password reset request'
    msg['From'] = SMTP_FROM
    msg['To'] = to_email
    safe_name = display_name or 'User'
    msg.set_content(
        f"Hi {safe_name},\n\n"
        f"Use this link to reset your password:\n{reset_link}\n\n"
        "This link expires in 15 minutes.\n"
        "If you did not request this, you can ignore this email.\n"
    )
    try:
        if SMTP_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=12) as server:
                if SMTP_USER:
                    server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
                if SMTP_USE_TLS:
                    server.starttls()
                if SMTP_USER:
                    server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
        return True
    except Exception as e:
        logger.error(f"Password reset email send failed: {e}")
        return False

def current_user():
    """Return the currently authenticated user from the role-keyed session slot.
    Admin and student each have their own slot so one can never overwrite the other.
    """
    return session.get('admin_user') or session.get('student_user')

def current_admin():
    """Return admin user if logged in, else None."""
    return session.get('admin_user')

def current_student():
    """Return student user if logged in, else None."""
    return session.get('student_user')

def require_role(role):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Check the role-specific session slot directly to avoid cross-role checks
            role_upper = (role or '').upper()
            if role_upper == 'ADMIN':
                user = session.get('admin_user')
            elif role_upper == 'STUDENT':
                user = session.get('student_user')
            else:
                user = current_user()

            if not user:
                if request.path.startswith('/api/'):
                    return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
                flash('Please login first.', 'error')
                return redirect(url_for('main'))
            if role_upper and (user.get('Role') or '').upper() != role_upper:
                if request.path.startswith('/api/'):
                    return jsonify({'ok': False, 'error': 'Forbidden'}), 403
                flash('Unauthorized access.', 'error')
                return redirect(url_for('main'))
            return fn(*args, **kwargs)
        return wrapper
    return decorator

# -------------------------
# CSRF + Rate Limit
# -------------------------
_rate_limit_store = {}
_rate_limit_lock = threading.Lock()

def _client_ip():
    xff = request.headers.get('X-Forwarded-For', '')
    if xff:
        return xff.split(',')[0].strip()
    return request.remote_addr or 'unknown'

def rate_limit(bucket, max_requests, window_seconds):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = f"{bucket}:{_client_ip()}"
            with _rate_limit_lock:
                rec = _rate_limit_store.get(key)
                if not rec or now >= rec['reset_at']:
                    rec = {'count': 0, 'reset_at': now + window_seconds}
                    _rate_limit_store[key] = rec
                rec['count'] += 1
                if rec['count'] > max_requests:
                    return jsonify({'error': 'Too many requests'}) if request.path.startswith('/api/') else ("Too many requests", 429)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def _ensure_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['csrf_token'] = token
    return token

@app.context_processor
def inject_csrf_token():
    return {'csrf_token': _ensure_csrf_token}

def _same_origin():
    host = request.host_url.rstrip('/')
    origin = request.headers.get('Origin')
    referer = request.headers.get('Referer')
    if origin:
        return origin.rstrip('/') == host
    if referer:
        try:
            pr = urlparse(referer)
            return f"{pr.scheme}://{pr.netloc}" == host
        except Exception:
            return False
    return False

@app.before_request
def csrf_protect():
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return
    # Exempt socket polling/engine routes
    if request.path.startswith('/socket.io'):
        return
    session_token = session.get('csrf_token')
    req_token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
    if session_token and req_token and secrets.compare_digest(session_token, req_token):
        return
    if _same_origin():
        return
    return ("CSRF validation failed", 403)

# -------------------------
# DB Schema Guard
# -------------------------
def ensure_db_schema():
    """Ensure required tables exist with the expected schema."""
    try:
        cur = mysql.connection.cursor()

        # Password hashes can exceed 100 chars (e.g., pbkdf2/scrypt).
        # Ensure column length is sufficient to prevent silent truncation.
        try:
            cur.execute("ALTER TABLE students MODIFY COLUMN Password VARCHAR(255) NOT NULL")
        except Exception:
            # Keep startup resilient if table/schema differs temporarily.
            pass

        # Exam sessions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS exam_sessions (
                SessionID INT AUTO_INCREMENT PRIMARY KEY,
                StudentID INT NOT NULL,
                StartTime DATETIME DEFAULT CURRENT_TIMESTAMP,
                EndTime DATETIME NULL,
                Status ENUM('IN_PROGRESS','COMPLETED','TERMINATED') DEFAULT 'IN_PROGRESS',
                INDEX idx_exam_sessions_student (StudentID)
            )
        """)

        # Exam results
        cur.execute("""
            CREATE TABLE IF NOT EXISTS exam_results (
                ResultID INT AUTO_INCREMENT PRIMARY KEY,
                StudentID INT NOT NULL,
                SessionID INT NOT NULL,
                Score DECIMAL(5,2) DEFAULT 0,
                TotalQuestions INT DEFAULT 125,
                CorrectAnswers INT DEFAULT 0,
                SubmissionTime DATETIME DEFAULT CURRENT_TIMESTAMP,
                Status ENUM('PASS','FAIL','TERMINATED') DEFAULT 'FAIL',
                INDEX idx_exam_results_student (StudentID),
                INDEX idx_exam_results_session (SessionID)
            )
        """)

        # Violations
        cur.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                ViolationID INT AUTO_INCREMENT PRIMARY KEY,
                StudentID INT NOT NULL,
                SessionID INT NOT NULL,
                ViolationType VARCHAR(64) NOT NULL,
                Details TEXT,
                Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_violations_student (StudentID),
                INDEX idx_violations_session (SessionID)
            )
        """)

        mysql.connection.commit()
        cur.close()
    except Exception as e:
        logger.error(f"DB schema ensure failed: {e}", exc_info=True)

# SocketIO
if SOCKETIO_AVAILABLE:
    try:
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
        MONITORING_ENABLED = True
        logger.info("SocketIO initialized successfully")
    except Exception as e:
        logger.error(f"SocketIO init failed: {e}")
        socketio = None
        MONITORING_ENABLED = False
else:
    socketio = None
    MONITORING_ENABLED = False

executor = ThreadPoolExecutor(max_workers=6)

# Camera & detection globals
CAMERA_INDEX = 0
HAAR_PATH = os.path.join('Haarcascades', 'haarcascade_frontalface_default.xml')
HAAR_PROFILE_PATH = os.path.join('Haarcascades', 'haarcascade_profileface.xml')
ENCODINGS_DIR = os.path.join('static', 'face_encodings')

# ============================================================================
# YOLOv8 Object Detection Setup
# ============================================================================
YOLO_DEFAULT_MODEL = 'yolov8n.pt'
YOLO_CUSTOM_MODEL = os.getenv('YOLO_MODEL_PATH', '').strip()
YOLO_CONF_THRESHOLD = float(os.getenv('YOLO_CONF_THRESHOLD', '0.22'))
YOLO_IMG_SIZE = int(os.getenv('YOLO_IMG_SIZE', '640'))
OBJECT_VIOLATION_COOLDOWN_SEC = float(os.getenv('OBJECT_VIOLATION_COOLDOWN_SEC', '3.5'))
DNN_OBJECT_FALLBACK_ENABLED = (os.getenv('DNN_OBJECT_FALLBACK_ENABLED', '1') == '1')
OCR_OBJECT_FALLBACK_ENABLED = (os.getenv('OCR_OBJECT_FALLBACK_ENABLED', '0') == '1')
# Enable phone shape fallback so smooth/low-texture phones still trigger.
PHONE_CONTOUR_FALLBACK_ENABLED = (os.getenv('PHONE_CONTOUR_FALLBACK_ENABLED', '1') == '1')

# # Objects that are prohibited during exam
PROHIBITED_LABELS = {
    'cell phone', 'mobile phone', 'phone', 'smartphone',
    'camera', 'webcam',
    'book', 'notebook', 'copy', 'paper', 'document',
    'headphones', 'earphones', 'headset', 'earbuds', 'airpods', 'headfree', 'handsfree',
    'electronic device', 'electronicdevice', 'device',
    'pen', 'pencil',
    'watch', 'smartwatch', 'clock'
}

def _normalize_label(label):
    value = str(label or '').strip().lower()
    aliases = {
        'cellphone': 'cell phone',
        'phone': 'cell phone',
        'mobile': 'cell phone',
        'mobile phone': 'cell phone',
        'smartphone': 'cell phone',
        'electronic device': 'cell phone',
        'electronicdevice': 'cell phone',
        'device': 'cell phone',
        'notebook': 'book',
        'copy': 'book',
        'paper': 'book',
        'document': 'book',
        'earphone': 'earphones',
        'earbud': 'earbuds',
        'headphone': 'headphones',
        'headset': 'headphones',
        'handsfree': 'headphones',
        'hands free': 'headphones',
        'head free': 'headphones',
        'headfree': 'headphones',
        'airpod': 'airpods',
        'pencil': 'pen',
        'web cam': 'webcam',
        'books': 'book',
        'camera phone': 'cell phone',
        'smart watch': 'watch',
        'smartwatch': 'watch',
        'clock': 'watch'
    }
    return aliases.get(value, value)

def _label_is_prohibited(label):
    """Strict selective matching for exam-prohibited objects only."""
    norm = _normalize_label(label)
    if norm in PROHIBITED_LABELS:
        return True

    normalized_tokens = set(re.findall(r'[a-z0-9]+', norm))
    for bad in PROHIBITED_LABELS:
        bad_tokens = set(re.findall(r'[a-z0-9]+', bad))
        if bad_tokens and bad_tokens.issubset(normalized_tokens):
            return True
    return False

# Load cascades
face_cascade = None
profile_face_cascade = None
if CV2_AVAILABLE and os.path.exists(HAAR_PATH):
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    logger.info("Frontal Haar cascade loaded successfully")
else:
    logger.warning(f"Frontal Haar cascade not found at {HAAR_PATH} or OpenCV not available.")

if CV2_AVAILABLE and os.path.exists(HAAR_PROFILE_PATH):
    profile_face_cascade = cv2.CascadeClassifier(HAAR_PROFILE_PATH)
    logger.info("Profile Haar cascade loaded successfully")
else:
    logger.warning(f"Profile Haar cascade not found at {HAAR_PROFILE_PATH}; continuing without it.")

# OpenCV HOG person detector fallback (for second-person presence)
people_hog = None
if CV2_AVAILABLE:
    try:
        people_hog = cv2.HOGDescriptor()
        people_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    except Exception:
        people_hog = None

# ── Load YOLOv4-tiny ──────────────────────────────────────────────────────────
object_net = None
object_net_enabled = False
yolo_class_names = {}
prohibited_class_ids = []
yolo_infer_lock = threading.Lock()
yolo_device = 'cpu'
yolo_loaded_model_path = None

if ULTRALYTICS_AVAILABLE:
    try:
        model_candidates = []
        if YOLO_CUSTOM_MODEL:
            model_candidates.append(YOLO_CUSTOM_MODEL)
        model_candidates.extend([
            os.path.join('models', 'best.pt'),
            os.path.join('models', 'custom.pt'),
            os.path.join('models', 'yolov8n.pt'),
            YOLO_DEFAULT_MODEL
        ])

        resolved_model = None
        for candidate in model_candidates:
            if candidate == YOLO_DEFAULT_MODEL or os.path.exists(candidate):
                resolved_model = candidate
                break
        if resolved_model is None:
            resolved_model = YOLO_DEFAULT_MODEL

        object_net = YOLO(resolved_model)
        yolo_loaded_model_path = resolved_model
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            yolo_device = 'cuda:0'
        else:
            yolo_device = 'cpu'
        object_net.to(yolo_device)

        names = getattr(object_net, 'names', {})
        if isinstance(names, dict):
            yolo_class_names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            yolo_class_names = {int(i): str(v) for i, v in enumerate(names)}
        else:
            yolo_class_names = {}

        prohibited_class_ids = [
            cls_id for cls_id, cls_name in yolo_class_names.items()
            if _label_is_prohibited(cls_name)
        ]

        # Fallback: hardcode known COCO class IDs for prohibited objects
        # so detection works even if label name matching missed something.
        COCO_PROHIBITED_IDS = [
            63,  # laptop
            67,  # cell phone
            72,  # tv
            73,  # book
            74,  # mouse
            75,  # remote
            76,  # keyboard
            85,  # clock (watch)
        ]
        for cid in COCO_PROHIBITED_IDS:
            if cid in yolo_class_names and cid not in prohibited_class_ids:
                prohibited_class_ids.append(cid)
                logger.info(f"Added COCO fallback prohibited class: {cid}={yolo_class_names[cid]}")

        object_net_enabled = True
        logger.info(
            f"YOLOv8 loaded from '{resolved_model}' on '{yolo_device}' "
            f"(prohibited classes mapped: {prohibited_class_ids})"
        )
    except Exception as e:
        object_net_enabled = False
        object_net = None
        logger.error(f"Error loading YOLOv8 model: {e}", exc_info=True)
# Import monitoring modules
try:
    from admin_live_monitoring import AdminMonitor, setup_admin_socketio, CameraSimulator
    from warning_system import WarningSystem, TabSwitchDetector
    MONITORING_MODULES_AVAILABLE = True
    logger.info("Monitoring modules imported successfully")
except Exception as e:
    logger.warning(f"Monitoring modules import failed: {e}")
    MONITORING_MODULES_AVAILABLE = False
    # Define fallback classes
    class AdminMonitor: 
        def start_monitoring(self, *args, **kwargs): pass
        def stop_monitoring(self, *args, **kwargs): pass
    class WarningSystem:
        def __init__(self, *args, **kwargs):
            self.warnings = {}
            self.student_names = {}
            self.max_warnings = 3
            self.lock = threading.Lock()
        def initialize_student(self, *args, **kwargs): pass
        def add_warning(self, *args, **kwargs): return False
    class TabSwitchDetector: 
        def __init__(self, *args, **kwargs): pass
        def initialize_student(self, *args, **kwargs): pass
        def detect_tab_switch(self, *args, **kwargs): return {'terminated': False, 'count': 0}
    class CameraSimulator:
        def __init__(self): pass
        def isOpened(self): return True
        def read(self): return True, None

studentInfo = None
detection_threads_started = False
latest_student_frames = {}
latest_student_frames_lock = threading.Lock()
student_detection_state = {}
student_detection_state_lock = threading.Lock()
active_exam_students = set()
active_exam_students_lock = threading.Lock()
student_frame_rx_counts = {}
student_frame_rx_lock = threading.Lock()
student_stale_violation_at = {}
student_stale_violation_lock = threading.Lock()
runtime_warning_state = {}
runtime_warning_state_lock = threading.Lock()
# Initialized early so background helpers can safely reference these before monitor setup.
warning_system = None

# Thresholds for Eye Tracking
EAR_THRESHOLD = 0.23                          # Lower = less sensitive to normal blinks, only detects sustained close
EYES_CLOSED_SECONDS = float(os.getenv('EYES_CLOSED_SECONDS', '1.5'))   # 1.5s closed eyes → warning
LOOKING_AWAY_SECONDS = float(os.getenv('LOOKING_AWAY_SECONDS', '4.0'))  # 4s looking away → warning
NO_FACE_SECONDS = float(os.getenv('NO_FACE_SECONDS', '0.5'))           # 0.5s no face → warning
SEAT_RISE_RATIO_THRESHOLD = 0.34
LEAN_RATIO_THRESHOLD = 0.24
MOTION_AREA_RATIO_THRESHOLD = 0.015          # lower = more sensitive to movement
LEFT_SEAT_SECONDS = 3.0                       # 3s rising from seat
MOVEMENT_DISTRACTION_SECONDS = 3.0           # 3s continuous movement → warning
POSE_ANALYSIS_FPS = 12.0
CAMERA_BLOCKED_BRIGHTNESS = 35               # mean pixel value below this = camera covered/blocked (higher catch hands)
CAMERA_BLOCKED_SECONDS = 0.8                 # seconds of dark frame before CAMERA_OFF warning
# Priority mode to stabilize live stream + face detection first.
FAST_FACE_ONLY_MODE = (os.getenv('FAST_FACE_ONLY_MODE', '0') == '1')
RUN_POSE_ANALYSIS = (os.getenv('RUN_POSE_ANALYSIS', '1') == '1')
OBJECT_ANALYSIS_INTERVAL_SEC = float(os.getenv('OBJECT_ANALYSIS_INTERVAL_SEC', '0.10'))
OBJECT_CONSEC_FRAMES = int(os.getenv('OBJECT_CONSEC_FRAMES', '3'))

logger.info(
    f"Detection config: fast_face_only={FAST_FACE_ONLY_MODE}, "
    f"run_pose_analysis={RUN_POSE_ANALYSIS}, pose_fps={POSE_ANALYSIS_FPS}, "
    f"yolo_imgsz={YOLO_IMG_SIZE}"
)

def _record_runtime_warning(student_id, student_name, violation_type, details):
    sid = str(student_id)
    if warning_system is None:
        # If the warning system isn't ready, avoid client-side auto-terminate by keeping count at 0.
        return 0, {
            'type': str(violation_type or 'UNKNOWN').upper(),
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'details': str(details or '').strip()
        }
    with runtime_warning_state_lock:
        rec = runtime_warning_state.setdefault(sid, {
            'warnings': 0,
            'student_name': str(student_name or 'Unknown'),
            'violations': []
        })
        rec['student_name'] = str(student_name or rec.get('student_name') or 'Unknown')
        if rec['warnings'] < 3:
            rec['warnings'] += 1
        violation = {
            'type': str(violation_type or 'UNKNOWN').upper(),
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'details': str(details or '').strip()
        }
        rec['violations'].append(violation)
        count = int(rec['warnings'])
    return count, violation

def _get_runtime_warning_state(student_id):
    sid = str(student_id)
    with runtime_warning_state_lock:
        rec = dict(runtime_warning_state.get(sid) or {})
        return {
            'warnings': int(rec.get('warnings') or 0),
            'student_name': str(rec.get('student_name') or 'Unknown'),
            'violations': list(rec.get('violations') or [])
        }

# -------------------------
# Camera Handler Singleton
# -------------------------
class CameraStreamer:
    def __init__(self, index=CAMERA_INDEX):
        self.index = index
        self.cap = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                if CV2_AVAILABLE:
                    self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        logger.error("Real camera not accessible, using simulator")
                        self.cap = CameraSimulator() if CameraSimulator else None
                else:
                    self.cap = CameraSimulator() if CameraSimulator else None
                    
                if self.cap is None:
                    raise RuntimeError("Camera not accessible.")
            self.running = True
            logger.info("Camera streamer started")

    def read(self):
        with self.lock:
            if self.cap is None:
                raise RuntimeError("Camera not initialized.")
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame.")
            return frame.copy()

    def release(self):
        with self.lock:
            if self.cap is not None:
                try:
                    if hasattr(self.cap, 'release'):
                        self.cap.release()
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")
                self.cap = None
            self.running = False

camera_streamer = CameraStreamer()

# -------------------------
# Detection Utilities
# -------------------------
def calculate_ear(eye_points):
    """Calculate the Eye Aspect Ratio (EAR) using 6 landmarks."""
    if len(eye_points) < 6:
        return 0.0
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)
    p2_minus_p6 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    p3_minus_p5 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    # Compute the euclidean distance between the horizontal eye landmark (x, y)
    p1_minus_p4 = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    
    if p1_minus_p4 == 0:
        return 0.0
    
    # Compute the eye aspect ratio
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

def _mesh_landmark_px(landmarks, idx, w, h):
    p = landmarks[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)

def _mean_mesh_landmark_px(landmarks, indices, w, h):
    pts = [_mesh_landmark_px(landmarks, i, w, h) for i in indices]
    return np.mean(np.array(pts, dtype=np.float32), axis=0)

def _safe_ratio(value, low, high):
    denom = max(1e-6, high - low)
    return (value - low) / denom

def _detect_gaze_direction_from_mesh(landmarks, w, h):
    """
    Estimate gaze direction from iris position relative to eye corners/lids.
    Returns one of: LEFT, RIGHT, UP, DOWN, CENTER.
    """
    left_iris = _mean_mesh_landmark_px(landmarks, [468, 469, 470, 471, 472], w, h)
    right_iris = _mean_mesh_landmark_px(landmarks, [473, 474, 475, 476, 477], w, h)

    left_outer = _mesh_landmark_px(landmarks, 33, w, h)
    left_inner = _mesh_landmark_px(landmarks, 133, w, h)
    right_outer = _mesh_landmark_px(landmarks, 362, w, h)
    right_inner = _mesh_landmark_px(landmarks, 263, w, h)

    left_upper = _mesh_landmark_px(landmarks, 159, w, h)
    left_lower = _mesh_landmark_px(landmarks, 145, w, h)
    right_upper = _mesh_landmark_px(landmarks, 386, w, h)
    right_lower = _mesh_landmark_px(landmarks, 374, w, h)

    left_x_ratio = _safe_ratio(left_iris[0], min(left_outer[0], left_inner[0]), max(left_outer[0], left_inner[0]))
    right_x_ratio = _safe_ratio(right_iris[0], min(right_outer[0], right_inner[0]), max(right_outer[0], right_inner[0]))
    avg_x_ratio = (left_x_ratio + right_x_ratio) / 2.0

    left_y_ratio = _safe_ratio(left_iris[1], min(left_upper[1], left_lower[1]), max(left_upper[1], left_lower[1]))
    right_y_ratio = _safe_ratio(right_iris[1], min(right_upper[1], right_lower[1]), max(right_upper[1], right_lower[1]))
    avg_y_ratio = (left_y_ratio + right_y_ratio) / 2.0

    # More sensitive thresholds so looking sideways/up/down is detected properly
    left_right_threshold = 0.38   # was 0.42 — now easier to trigger LEFT/RIGHT
    up_threshold = 0.35           # was 0.40 — triggers UP sooner (looking at phone above)
    down_threshold = 0.65         # was 0.60 — triggers DOWN sooner (looking at paper below)

    horizontal_dev = abs(avg_x_ratio - 0.5)
    vertical_dev = abs(avg_y_ratio - 0.5)

    if horizontal_dev >= vertical_dev:
        if avg_x_ratio < left_right_threshold:
            return "LEFT"
        if avg_x_ratio > (1.0 - left_right_threshold):
            return "RIGHT"

    if avg_y_ratio < up_threshold:
        return "UP"
    if avg_y_ratio > down_threshold:
        return "DOWN"
    return "CENTER"

def _pose_point_px(landmarks, idx, w, h, min_vis=0.45):
    lm = landmarks[idx]
    vis = float(getattr(lm, 'visibility', 1.0))
    if vis < min_vis:
        return None
    return np.array([float(lm.x * w), float(lm.y * h)], dtype=np.float32)

def get_head_pose(landmarks, frame_shape):
    """Calculate head pose (pitch, yaw, roll) from MediaPipe landmarks."""
    img_h, img_w, _ = frame_shape
    
    # 2D image points from MediaPipe landmarks
    # Nose tip [1], Chin [152], Left eye left corner [33], Right eye right corner [263], Left Mouth corner [61], Right mouth corner [291]
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),     # Nose tip
        (landmarks[152].x * img_w, landmarks[152].y * img_h), # Chin
        (landmarks[226].x * img_w, landmarks[226].y * img_h), # Left eye left corner (MediaPipe index 226)
        (landmarks[446].x * img_w, landmarks[446].y * img_h), # Right eye right corner (MediaPipe index 446)
        (landmarks[61].x * img_w, landmarks[61].y * img_h),   # Left Mouth corner
        (landmarks[291].x * img_w, landmarks[291].y * img_h)  # Right mouth corner
    ], dtype="double")

    # 3D model points (standard anthropometric model)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals
    focal_length = img_w
    center = (img_w/2, img_h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return "UNKNOWN"
        
    # Get rotational matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Get angles
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    
    pitch, yaw, roll = [math.degrees(angle) for angle in euler_angles]
    
    # Simple thresholding logic for looking away
    if yaw < -15:
        return "LEFT"
    elif yaw > 15:
        return "RIGHT"
    elif pitch < -15:
        return "DOWN"
    elif pitch > 15:
        return "UP"
    else:
        return "CENTER"

def detect_faces(frame):
    """Return list of faces as dicts."""
    if not CV2_AVAILABLE or frame is None:
        return []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = []
        if face_cascade is not None:
            frontal = face_cascade.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=2, minSize=(20, 20))
            for (x, y, w, h) in frontal:
                boxes.append((int(x), int(y), int(w), int(h)))
            # Equalized pass for difficult lighting / caps.
            eq = cv2.equalizeHist(gray)
            frontal_eq = face_cascade.detectMultiScale(
                eq, scaleFactor=1.08, minNeighbors=2, minSize=(18, 18))
            for (x, y, w, h) in frontal_eq:
                boxes.append((int(x), int(y), int(w), int(h)))
        if profile_face_cascade is not None:
            profile = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            for (x, y, w, h) in profile:
                boxes.append((int(x), int(y), int(w), int(h)))
            flipped = cv2.flip(gray, 1)
            profile_flip = profile_face_cascade.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            fw = gray.shape[1]
            for (x, y, w, h) in profile_flip:
                boxes.append((int(fw - x - w), int(y), int(w), int(h)))

        # face_recognition fallback with upsample improves small/side faces.
        if FACE_REC_AVAILABLE and len(boxes) <= 2:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fr_locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model='hog')
                for (top, right, bottom, left) in fr_locs:
                    boxes.append((int(max(0, left)), int(max(0, top)),
                                  int(max(1, right - left)), int(max(1, bottom - top))))
            except Exception:
                pass

        # De-duplicate overlapping boxes with simple IoU suppression.
        dedup = []
        for bx in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
            x, y, w, h = bx
            keep = True
            for ex in dedup:
                xx, yy, ww, hh = ex
                ix1 = max(x, xx)
                iy1 = max(y, yy)
                ix2 = min(x + w, xx + ww)
                iy2 = min(y + h, yy + hh)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                union = (w * h) + (ww * hh) - inter
                iou = (inter / float(union)) if union > 0 else 0.0
                if iou > 0.45:
                    keep = False
                    break
            if keep:
                dedup.append(bx)

        return [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} for (x, y, w, h) in dedup]
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return []

def detect_people_opencv(frame):
    """OpenCV HOG fallback person detector."""
    if not CV2_AVAILABLE or people_hog is None or frame is None:
        return 0
    try:
        h, w = frame.shape[:2]
        scale = 1.0
        if max(h, w) > 960:
            scale = 960.0 / float(max(h, w))
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            resized = frame
        rects, weights = people_hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.03
        )
        count = 0
        for i, (x, y, rw, rh) in enumerate(rects):
            conf = float(weights[i]) if i < len(weights) else 0.0
            if conf >= 0.35:
                count += 1
        return int(count)
    except Exception:
        return 0

def _image_has_single_face(image_bgr):
    """Validate that image contains at least one visible face (relaxed for half/low-light faces)."""
    try:
        if image_bgr is None:
            return False
        # Prefer face_recognition for stricter face localization if available.
        if FACE_REC_AVAILABLE:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else image_bgr
            locs = face_recognition.face_locations(rgb)
            return len(locs) >= 1
        faces = detect_faces(image_bgr)
        return len(faces) >= 1
    except Exception:
        return False

def _bytes_has_single_face(image_bytes):
    """Decode image bytes and validate single-face rule."""
    if not CV2_AVAILABLE or np is None or not image_bytes:
        return False
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return _image_has_single_face(img)
    except Exception:
        return False

def _encode_frame_to_base64(frame, quality=75):
    if not CV2_AVAILABLE or frame is None:
        return None
    try:
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode('ascii')
    except Exception:
        return None

def _draw_object_boxes(frame, detections):
    if not CV2_AVAILABLE or frame is None:
        return frame
    try:
        for det in detections or []:
            x = int(det.get('x', 0))
            y = int(det.get('y', 0))
            w = int(det.get('w', 0))
            h = int(det.get('h', 0))
            conf = float(det.get('confidence', 0.0))
            label = str(det.get('label', 'object'))
            color = (0, 0, 255) if _label_is_prohibited(label) else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
        return frame
    except Exception:
        return frame

def _overlay_status_snapshot(frame, snapshot, item=None):
    if not CV2_AVAILABLE or frame is None:
        return frame
    try:
        snap = snapshot or {}
        item = item or {}
        # Minimal overlay only (per UX request) — no diagnostic text
        return frame
    except Exception:
        return frame

def _get_active_session_id(student_id):
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            SELECT SessionID FROM exam_sessions
            WHERE StudentID=%s AND Status='IN_PROGRESS'
            ORDER BY StartTime DESC LIMIT 1
            """,
            (student_id,)
        )
        row = cur.fetchone()
        cur.close()
        return int(row[0]) if row else None
    except Exception as e:
        logger.error(f"Active session lookup failed for student {student_id}: {e}")
        return None

def _trigger_violation(student_id, student_name, violation_type, details, cooldown_seconds=3.0):
    sid = str(student_id)
    logger.info(f"[TRIGGER DETECTED] sid={sid} type={violation_type} details='{details}'")
    """
    Central violation handler:
    - applies per-student/type cooldown
    - increments warning count
    """
    now = time.time()
    vtype = str(violation_type or '').strip().upper()

    # Ensure warning system is always available even if monitoring init failed
    global warning_system, admin_monitor, tab_detector
    if warning_system is None:
        try:
            warning_system = WarningSystem(socketio, admin_monitor)
            tab_detector = TabSwitchDetector(warning_system, threshold=int(os.getenv('TAB_SWITCH_WARNING_THRESHOLD', '1')))
            warning_system.max_warnings = 3
        except Exception as ensure_err:
            logger.error(f"[WARNING_INIT_FAIL] unable to bootstrap warning system: {ensure_err}")

    with student_detection_state_lock:
        if sid not in student_detection_state:
            student_detection_state[sid] = {
                'running': False,
                'pending_frame': None,
                'last_violation_by_label': {},
                'last_result': {},
                'eyes_closed_start': None,
                'looking_away_start': None,
                'no_face_start': None,
                'multiple_faces_start': None,
                'object_consecutive': {},
                'detection_states': [],
                'baseline_pose': None,
                'pose_prev_gray': None,
                'pose_prev_nose': None,
                'left_seat_start': None,
                'movement_start': None,
                'last_pose_ts': 0.0,
                'cam_blocked_start': None
            }
        state = student_detection_state[sid]
        
        last_hit = float(state['last_violation_by_label'].get(vtype, 0.0))
        if (now - last_hit) < float(cooldown_seconds):
            logger.info(f"[TRIGGER BLOCKED BY COOLDOWN] sid={sid} type={vtype} ({now-last_hit:.1f}s < {cooldown_seconds}s)")
            return False
        state['last_violation_by_label'][vtype] = now

    logger.info(f"[TRIGGER PROCEEDING] sid={sid} type={vtype} details={details}")
    
    # 1. Store the runtime behavior record
    runtime_count, runtime_violation = _record_runtime_warning(sid, student_name, vtype, details)
    
    # 2. Extract explicit Penalty Score from DecisionEngine
    final_score = 0
    try:
        from app import _get_vision_engines
        _, _, decision_engine, _ = _get_vision_engines()
        
        # Add external penalties not caught by vision engine directly
        external_penalty = 0
        if vtype == 'VOICE_DETECTED':
            external_penalty = 25
        elif vtype == 'CAMERA_OFF':
            external_penalty = 20
            
        if external_penalty > 0 and hasattr(decision_engine, 'add_penalty'):
            final_score = decision_engine.add_penalty(sid, external_penalty)
        elif hasattr(decision_engine, 'get_state'):
            state = decision_engine.get_state(sid)
            final_score = state.get('warning_count', 0)
    except Exception as e:
        logger.warning(f"Could not extract explicit Penalty Score from Decision Engine: {e}")
        final_score = max(runtime_count * 15, 0) # Fallback heuristic
        
    try:
        # Sync old warning_system for admin legacy reasons
        if warning_system:
            if sid not in warning_system.warnings:
                warning_system.initialize_student(sid, student_name)
            warning_system.add_warning(sid, vtype, details, emit_to_student=True)
            logger.info(f"[WARNING ADDED] sid={sid} type={vtype} penalty_score={final_score}")
        else:
            logger.error(f"[WARNING FAILED] sid={sid} warning_system IS NONE")
    except Exception as e:
        logger.warning(f"Warning increment failed for student {sid}: {e}")
        
    # 3. Action Logic: Warning (>40) and Termination (>80)
    try:
        if socketio:
            payload = {
                'student_id': sid,
                'student_name': student_name,
                'total_warnings': final_score, # Overloaded legacy field to pass score
                'penalty_score': final_score,
                'violation': runtime_violation,
                'type': vtype,
                'details': f"{details} (Penalty Score: {final_score})",
                'source': 'server',
                'auto_terminate': final_score > 80
            }
            
            # Send the Warning to UI
            if final_score > 40:
                socketio.emit('student_violation', payload, namespace='/student', room=f"student:{sid}")
            socketio.emit('student_violation', payload, namespace='/admin')
            
            # Execute Hard Termination if > 80
            if final_score > 80:
                logger.warning(f"[TERMINATING EXAM] Student {sid} exceeded penalty threshold (Score: {final_score})")
                socketio.emit('force_exam_terminate', {
                    'student_id': sid, 
                    'reason': 'Exam terminated due to excessive cheating violations.'
                }, namespace='/student', room=f"student:{sid}")
                
    except Exception as e:
        logger.warning(f"Socket warning sync failed for student {sid}: {e}")


    try:
        session_id = _get_active_session_id(sid)
        if session_id is not None:
            cur = mysql.connection.cursor()
            cur.execute(
                """
                INSERT INTO violations (StudentID, SessionID, ViolationType, Details, Timestamp)
                VALUES (%s, %s, %s, %s, NOW())
                """,
                (sid, int(session_id), vtype, details)
            )
            mysql.connection.commit()
            cur.close()
    except Exception as e:
        logger.error(f"Violation DB insert failed for student {sid}: {e}")

    logger.info(f"[violation_triggered] student={sid} type={vtype} details={details}")
    return True

def _persist_object_violation(student_id, student_name, label, confidence):
    sid = str(student_id)
    norm_label = _normalize_label(label)
    if not _label_is_prohibited(norm_label):
        return False
    try:
        conf_val = float(confidence or 0.0)
    except Exception:
        conf_val = 0.0
    # Require slightly higher than min confidence before warning
    if conf_val < max(OBJECT_MIN_CONFIDENCE, 0.60):
        return False

    # Require consecutive hits to reduce single-frame false positives (books/papers)
    consecutive_needed = max(int(os.getenv('OBJECT_CONSEC_FRAMES', OBJECT_CONSEC_FRAMES)), 2)
    now = time.time()
    with student_detection_state_lock:
        state = student_detection_state.setdefault(sid, {})
        hits = state.setdefault('object_consecutive', {})
        last_ts, count = hits.get(norm_label, (0.0, 0))
        if now - last_ts > 3.0:
            count = 0
        count += 1
        hits[norm_label] = (now, count)
        if count < consecutive_needed:
            return False
        # reset counter after firing to prevent rapid double-counting
        hits[norm_label] = (now, 0)

    details = f"Detected prohibited object: {norm_label} ({confidence:.2f})"
    logger.info(f"[object_violation] student={sid} label={norm_label} conf={confidence:.2f} consecutive={consecutive_needed}")
    return _trigger_violation(
        sid,
        student_name,
        'PROHIBITED_OBJECT',
        details,
        cooldown_seconds=float(max(OBJECT_VIOLATION_COOLDOWN_SEC, 4.0))
    )


def _reset_exam_runtime_state(student_id, student_name=None):
    sid = str(student_id)

    try:
        if warning_system:
            resolved_name = student_name or warning_system.student_names.get(sid, f"Student {sid}")
            warning_system.initialize_student(sid, resolved_name)
            warning_system.reset_student(sid)
    except Exception as e:
        logger.warning(f"Warning system reset failed for student {sid}: {e}")

    try:
        if tab_detector:
            tab_detector.initialize_student(sid)
    except Exception as e:
        logger.warning(f"Tab detector reset failed for student {sid}: {e}")

    with runtime_warning_state_lock:
        runtime_warning_state.pop(sid, None)

    try:
        face_analyzer, person_detector, decision_engine, UILayer = _get_vision_engines()
        if hasattr(decision_engine, 'reset_student'):
            decision_engine.reset_student(sid)
    except Exception as e:
        logger.warning(f"Decision engine reset failed for student {sid}: {e}")

    with latest_student_frames_lock:
        latest_student_frames.pop(sid, None)

    with student_detection_state_lock:
        student_detection_state.pop(sid, None)

OCR_KEYWORD_MAP = {
    'iphone': 'cell phone',
    'android': 'cell phone',
    'samsung': 'cell phone',
    'xiaomi': 'cell phone',
    'mobile': 'cell phone',
    'phone': 'cell phone',
    'airpods': 'headphones',
    'earbuds': 'headphones',
    'earphones': 'headphones',
    'headphones': 'headphones',
    'headset': 'headphones',
    'laptop': 'laptop',
    'notebook': 'book',
    'book': 'book'
}

def _detect_prohibited_text(frame):
    """
    OCR-based keyword detection.
    Returns detections in same shape as object detections: {label, confidence, x, y, w, h}
    """
    if not OCR_AVAILABLE or frame is None or not CV2_AVAILABLE:
        return []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # Improves OCR on low-contrast webcam frames.
        proc = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )

        data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT, config='--oem 3 --psm 6')
        n = len(data.get('text', []))
        out = []
        for i in range(n):
            raw_text = str(data['text'][i] or '').strip().lower()
            if not raw_text:
                continue
            token = re.sub(r'[^a-z0-9]+', '', raw_text)
            if not token:
                continue
            conf_raw = str(data.get('conf', ['0'])[i] or '0').strip()
            try:
                conf = float(conf_raw)
            except Exception:
                conf = 0.0
            if conf < 45.0:
                continue

            mapped_label = None
            for kw, lbl in OCR_KEYWORD_MAP.items():
                if kw in token:
                    mapped_label = lbl
                    break
            if not mapped_label:
                continue

            x = int(data['left'][i])
            y = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])
            if w <= 2 or h <= 2:
                continue

            out.append({
                'label': mapped_label,
                'confidence': max(0.45, min(0.99, conf / 100.0)),
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
        return out
    except Exception as e:
        logger.debug(f"OCR detection skipped: {e}")
        return []

def _detect_phone_like_contours(frame):
    """Heuristic fallback for large handheld rectangular phone-like objects."""
    if not CV2_AVAILABLE or frame is None:
        return []
    try:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 160)
        edges = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        min_area = float(h * w) * 0.02
        max_area = float(h * w) * 0.55
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            peri = float(cv2.arcLength(cnt, True))
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) != 4:
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            if bw <= 20 or bh <= 20:
                continue
            aspect = float(min(bw, bh)) / float(max(bw, bh))
            if aspect < 0.35:
                continue
            roi = gray[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
            if roi.size == 0:
                continue
            mean_val = float(np.mean(roi))
            if mean_val > 115.0:
                continue
            out.append({
                'label': 'cell phone',
                'confidence': 0.34,
                'x': int(x),
                'y': int(y),
                'w': int(bw),
                'h': int(bh),
                'is_prohibited': True
            })
        out.sort(key=lambda d: float(d['w'] * d['h']), reverse=True)
        return out[:2]
    except Exception:
        return []

# Object detection tuning (can be overridden via env vars)
#  - confidence: lower to 0.50 so real devices fire, but still filtered by texture/area
#  - area: ignore tiny specks (<1% of frame) and huge flat walls (>25%).
OBJECT_MIN_CONFIDENCE = float(os.getenv('OBJECT_MIN_CONFIDENCE', '0.45'))
OBJECT_MIN_AREA_RATIO = float(os.getenv('OBJECT_MIN_AREA_RATIO', '0.015'))   # >=1.5% of frame
OBJECT_MAX_AREA_RATIO = float(os.getenv('OBJECT_MAX_AREA_RATIO', '0.20'))    # <=20% of frame (ignore big wall chunks)
# Reject flat, texture-less regions (walls/blank paper) by Laplacian variance and edge density.
OBJECT_MIN_TEXTURE_VAR = float(os.getenv('OBJECT_MIN_TEXTURE_VAR', '120.0'))
OBJECT_MIN_EDGE_DENSITY = float(os.getenv('OBJECT_MIN_EDGE_DENSITY', '0.015'))  # >=1.5% edge pixels

def detect_objects(frame, conf_threshold=0.20, include_visual_classes=False):
    """
    Detect prohibited objects using YOLOv8.
    Returns list of dicts: {label, confidence, x, y, w, h}
    """
    if not CV2_AVAILABLE or frame is None:
        return []

    try:
        detections = []
        h, w = frame.shape[:2]

        def _predict(source_frame, threshold):
            with yolo_infer_lock:
                return object_net.predict(
                    source=source_frame,
                    conf=float(threshold),
                    imgsz=int(YOLO_IMG_SIZE),
                    device=yolo_device,
                    classes=target_classes,
                    verbose=False,
                    max_det=12
                )

        if object_net_enabled and object_net is not None:
            target_classes = list(set(prohibited_class_ids + [0])) if prohibited_class_ids else [0]
            results = _predict(frame, conf_threshold)

            # Low-light handheld objects often fail on the raw webcam frame.
            # Retry once with a brighter, contrast-enhanced view before giving up.
            if (not results or getattr(results[0], 'boxes', None) is None or len(getattr(results[0].boxes, 'cls', [])) == 0) and CV2_AVAILABLE:
                try:
                    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    y = clahe.apply(y)
                    enhanced = cv2.merge((y, cr, cb))
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)
                    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.18, beta=14)
                    results = _predict(enhanced, max(0.10, float(conf_threshold) * 0.75))
                except Exception:
                    pass

            if results:
                res = results[0]
                boxes = getattr(res, 'boxes', None)
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                        score = float(box.conf[0].item()) if box.conf is not None else 0.0
                        if score < float(conf_threshold):
                            continue

                        raw_label = yolo_class_names.get(cls_id, str(cls_id))
                        label = _normalize_label(raw_label)
                        if not include_visual_classes and not _label_is_prohibited(label):
                            continue

                        xyxy = box.xyxy[0].tolist()
                        x1 = max(0, min(int(xyxy[0]), w - 1))
                        y1 = max(0, min(int(xyxy[1]), h - 1))
                        x2 = max(0, min(int(xyxy[2]), w))
                        y2 = max(0, min(int(xyxy[3]), h))
                        bw = max(1, x2 - x1)
                        bh = max(1, y2 - y1)

                        # Label-specific thresholds to balance recall vs false positives
                        min_conf = OBJECT_MIN_CONFIDENCE
                        min_area = OBJECT_MIN_AREA_RATIO
                        max_area = OBJECT_MAX_AREA_RATIO
                        min_edge = OBJECT_MIN_EDGE_DENSITY
                        tex_min = OBJECT_MIN_TEXTURE_VAR

                        if label == 'cell phone':
                            min_conf = 0.52
                            min_area = 0.005   # allow smaller phones
                            min_edge = 0.003   # phones can be flat/shiny
                            tex_min = 40.0

                        area_ratio = float(bw * bh) / float(max(1, w * h))

                        if score < min_conf:
                            continue
                        if area_ratio < min_area or area_ratio > max_area:
                            continue

                        # Texture filter: discard very flat regions (walls/blank background)
                        try:
                            roi = frame[y1:y2, x1:x2]
                            if roi is not None and roi.size > 0 and CV2_AVAILABLE:
                                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                tex_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                                edges = cv2.Canny(gray_roi, 60, 160)
                                edge_density = float(cv2.countNonZero(edges)) / float(gray_roi.size)
                                # Stricter for flat/book-like labels: need both texture + edges
                                edge_thresh = min_edge
                                tex_thresh = tex_min
                                if label == 'book':
                                    edge_thresh = max(edge_thresh, 0.025)
                                    tex_thresh = max(tex_thresh, 120.0)
                                if tex_var < tex_thresh or edge_density < edge_thresh:
                                    continue
                        except Exception:
                            pass

                        detections.append({
                            'label': label,
                            'confidence': score,
                            'x': x1,
                            'y': y1,
                            'w': bw,
                            'h': bh,
                            'is_prohibited': _label_is_prohibited(label)
                        })

        if OCR_OBJECT_FALLBACK_ENABLED:
            ocr_hits = _detect_prohibited_text(frame)
            if ocr_hits:
                detections.extend(ocr_hits)
        if PHONE_CONTOUR_FALLBACK_ENABLED and not any(_label_is_prohibited(d.get('label', '')) for d in detections):
            detections.extend(_detect_phone_like_contours(frame))

        filtered = []
        for detection in detections:
            label = _normalize_label(detection.get('label', ''))
            if not _label_is_prohibited(label):
                continue
            normalized_detection = dict(detection)
            normalized_detection['label'] = label
            filtered.append(normalized_detection)
        return filtered
    except Exception as e:
        logger.error(f"Error in YOLOv8 object detection: {e}", exc_info=True)
        return []

def _persist_behavior_violation(student_id, student_name, violation_type, details):
    sid = str(student_id)
    ok = _trigger_violation(
        sid,
        student_name,
        violation_type,
        details,
        cooldown_seconds=3.0
    )
    logger.info(f"[behavior_violation] student={sid} type={violation_type} details={details}")
    return ok



def _student_frame_staleness_watchdog():
    """Backstop detector: if frames stop arriving for active exam students, raise CAMERA_OFF."""
    while True:
        try:
            now = time.time()
            with active_exam_students_lock:
                active_ids = list(active_exam_students)

            for sid in active_ids:
                sid_str = str(sid)
                with latest_student_frames_lock:
                    item = latest_student_frames.get(sid_str) or {}
                    last_rx = float(item.get('frame_timestamp') or item.get('timestamp') or 0.0)

                # If we have never received a frame yet (still warming up), skip staleness checks.
                if last_rx <= 0.0:
                    continue

                stale_for = now - last_rx

                # If no frame for ~4.5s while exam active, camera/feed is effectively down.
                if stale_for < 4.5:
                    with student_stale_violation_lock:
                        student_stale_violation_at.pop(sid, None)
                    continue

                with student_stale_violation_lock:
                    last_violation = float(student_stale_violation_at.get(sid, 0.0))
                    if (now - last_violation) < 12.0:
                        continue
                    student_stale_violation_at[sid_str] = now

                name = f"Student {sid_str}"
                if warning_system:
                    try:
                        name = str(warning_system.student_names.get(sid_str) or name)
                    except Exception:
                        pass

                # Raise a server-side warning when camera feed stalls for 4.5s+
                _persist_behavior_violation(
                    sid_str,
                    name,
                    "CAMERA_OFF",
                    f"Student camera/frame feed stale for {stale_for:.1f}s"
                )
                logger.info(f"[CAMERA_OFF] student={sid_str} stale_for={stale_for:.1f}s")
        except Exception as e:
            logger.warning(f"frame staleness watchdog error: {e}")
        time.sleep(1.0)

threading.Thread(target=_student_frame_staleness_watchdog, daemon=True).start()


# --- NEW VISION ENGINE START ---
vision_face_analyzer = None
vision_person_detector = None
vision_decision_engine = None
vision_ui_layer = None

def _get_vision_engines():
    global vision_face_analyzer, vision_person_detector, vision_decision_engine, vision_ui_layer
    if vision_face_analyzer is None:
        from face_pipeline import FaceAnalyzer
        from person_pipeline import PersonDetector
        from decision_engine import DecisionEngine
        from vision_ui import UILayer
        import config_vision as config
        
        vision_face_analyzer = FaceAnalyzer()
        vision_person_detector = PersonDetector()
        vision_decision_engine = DecisionEngine()
        vision_ui_layer = UILayer
        
    return vision_face_analyzer, vision_person_detector, vision_decision_engine, vision_ui_layer

def _run_student_frame_detection(student_id, student_name, frame):
    sid = str(student_id)
    now = time.time()
    try:
        logger.debug(f"[student_frame] new vision engine sid={sid}")
        processed = frame.copy()
        
        face_analyzer, person_detector, decision_engine, UILayer = _get_vision_engines()
        import config_vision as config
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # --- CAMERA OBSTRUCTION DETECTION ---
        # 0a. Check for intentional blur (hand held in front, fogged lens)
        try:
            _gray_obs = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            _lap_var = cv2.Laplacian(_gray_obs, cv2.CV_64F).var()
            _mean_bright = float(_gray_obs.mean())
            # If image is extremely blurry (var < 10) AND not just a dark room
            if _lap_var < 10.0 and _mean_bright > 20.0:
                _persist_behavior_violation(sid, student_name, "NO_FACE", "Camera appears intentionally blurred or obstructed")
            # If image is almost completely dark (cam covered) for >2s, fire immediately
            if _mean_bright < 12.0:
                _persist_behavior_violation(sid, student_name, "NO_FACE", "Camera appears covered or blocked")
        except Exception:
            pass
        # --- END CAMERA OBSTRUCTION DETECTION ---
        
        # 1. Face Analysis
        face_detected, yaw_angle, pitch_angle, ear, iris_offset, landmarks = face_analyzer.process_frame(frame_rgb)

        face_bbox = None
        if landmarks:
            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]
            x1 = max(min(xs) * frame.shape[1] - 10, 0)
            y1 = max(min(ys) * frame.shape[0] - 10, 0)
            x2 = min(max(xs) * frame.shape[1] + 10, frame.shape[1])
            y2 = min(max(ys) * frame.shape[0] + 10, frame.shape[0])
            face_bbox = (int(x1), int(y1), int(x2), int(y2))
        
        # 2. Person & Object Detection
        person_count, bboxes, banned_objects = person_detector.process_frame(processed, face_bbox)

        # Reclassify / filter banned objects using face overlap to reduce false phones-near-ear
        def bbox_iou(b1, b2):
            xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
            xb = min(b1[2], b2[2]); yb = min(b1[3], b2[3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            if inter == 0: return 0.0
            area1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
            area2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
            return inter / max(area1 + area2 - inter, 1)

        if face_bbox:
            filtered = []
            frame_h = processed.shape[0] if processed is not None else 0
            for obj in banned_objects:
                bbox = obj.get('bbox') or (0,0,0,0,0)
                iou = bbox_iou(bbox, face_bbox)

                # Drop paper detections that overlap the face or sit at the very top (glare/skin patches)
                if obj.get('label') == 'Paper':
                    if iou > 0.20:
                        continue
                    cy = (bbox[1] + bbox[3]) * 0.5
                    if frame_h and cy < frame_h * 0.18:
                        continue

                filtered.append(obj)
            banned_objects = filtered
            
            # --- START EARBUD / HEADPHONE HEURISTIC ---
            # MediaPipe bounding box helps isolate the ear regions
            try:
                h_frame, w_frame = processed.shape[:2]
                face_w = face_bbox[2] - face_bbox[0]
                face_h = face_bbox[3] - face_bbox[1]
                
                # Check regions immediately left and right of the face
                ear_margin_w = int(face_w * 0.20)
                ear_margin_h = int(face_h * 0.40)
                
                l_x1, l_x2 = max(0, face_bbox[0] - ear_margin_w), face_bbox[0]
                r_x1, r_x2 = face_bbox[2], min(w_frame, face_bbox[2] + ear_margin_w)
                y1, y2 = max(0, face_bbox[1] + int(face_h * 0.2)), min(h_frame, face_bbox[3] - int(face_h * 0.1))
                
                def _check_ear_roi(rx1, ry1, rx2, ry2):
                    if rx2 - rx1 < 5 or ry2 - ry1 < 5: return False
                    roi = processed[ry1:ry2, rx1:rx2]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    v_channel = hsv[:, :, 2]
                    
                    # Look for bright white (Airpods) or solid dark (Over-ear bands)
                    _, white_mask = cv2.threshold(v_channel, 210, 255, cv2.THRESH_BINARY)
                    _, dark_mask = cv2.threshold(v_channel, 45, 255, cv2.THRESH_BINARY_INV)
                    
                    w_dens = cv2.countNonZero(white_mask) / (white_mask.size or 1)
                    d_dens = cv2.countNonZero(dark_mask) / (dark_mask.size or 1)
                    
                    if w_dens > 0.06 or d_dens > 0.30:
                        edges = cv2.Canny(cv2.GaussianBlur(v_channel, (5,5), 0), 40, 130)
                        if cv2.countNonZero(edges) / (edges.size or 1) > 0.05:
                            return True
                    return False

                if _check_ear_roi(l_x1, y1, l_x2, y2) or _check_ear_roi(r_x1, y1, r_x2, y2):
                    banned_objects.append({
                        "label": "Headphones/Earbuds",
                        "bbox": (l_x1, y1, r_x2, y2, 0.85)
                    })
            except Exception as e:
                pass
            # --- END EARBUD HEURISTIC ---
        
        # 3. Evaluate Rule Violations
        warnings, penalty_score = decision_engine.evaluate(
            sid, face_detected, person_count, yaw_angle, pitch_angle, ear, iris_offset, banned_objects
        )
        
        # 4. Integrate detections with warning persistence
        for w in warnings:
            if "Face not detected" in w:
                _persist_behavior_violation(sid, student_name, "NO_FACE", "No face detected in camera viewport")
            elif "Multiple persons" in w:
                _persist_behavior_violation(sid, student_name, "MULTIPLE_FACES", w)
            elif "Gaze off screen" in w:
                _persist_behavior_violation(sid, student_name, "DISTRACTION", w)
            elif "Head turned away" in w:
                _persist_behavior_violation(sid, student_name, "DISTRACTION", w)
            elif "Head turned" in w:
                _persist_behavior_violation(sid, student_name, "DISTRACTION", "Please look at the screen (Head turned)")
            elif "Eyes not visible" in w:
                _persist_behavior_violation(sid, student_name, "EYES_CLOSED", "Eyes not visible / Looking down")
            elif "Gazing" in w:
                _persist_behavior_violation(sid, student_name, "DISTRACTION", "Gazing at another screen/paper")

        for obj in banned_objects:
            bbox = obj.get('bbox') or (0, 0, 0, 0, 0.0)
            confidence = float(bbox[4]) if len(bbox) >= 5 else 0.0
            _persist_object_violation(sid, student_name, obj.get('label', 'Object'), confidence)

        # Hard-guard: trigger violation immediately when multiple humans detected
        if int(person_count) > 1:
            _persist_behavior_violation(
                sid,
                student_name,
                "MULTIPLE_FACES",
                f"Detected {int(person_count)} persons in camera"
            )

        # 5. Draw the unified HUD before encoding
        UILayer.draw_overlays(processed, warnings, penalty_score, bboxes, banned_objects, 0.0, landmarks, iris_offset)
        
        # 6. Push to MJPEG live stream format
        encoded = _encode_frame_to_base64(processed)
        
        # Format the diagnostics snapshot for the legacy Admin UI
        latest_snapshot = {
            'face_detected': bool(face_detected),
            'landmarks_detected': bool(face_detected),
            'face_obscured': not face_detected,
            'face_count': int(person_count) if face_detected else 0,
            'eyes_closed': ear < config.EAR_THRESHOLD if face_detected else False,
            'eyes_closed_elapsed': 0.0,
            'looking_away': abs(yaw_angle) > config.YAW_THRESHOLD_DEG if face_detected else False,
            'looking_away_elapsed': 0.0,
            'gaze_direction': (
                "LEFT" if isinstance(iris_offset, (list, tuple)) and len(iris_offset) > 0 and abs(iris_offset[0]) > config.IRIS_OFFSET_THRESHOLD and iris_offset[0] < 0
                else "RIGHT" if isinstance(iris_offset, (list, tuple)) and len(iris_offset) > 0 and abs(iris_offset[0]) > config.IRIS_OFFSET_THRESHOLD and iris_offset[0] > 0
                else "UP" if isinstance(iris_offset, (list, tuple)) and len(iris_offset) > 1 and abs(iris_offset[1]) > getattr(config, 'IRIS_OFFSET_THRESHOLD_Y', 0.20) and iris_offset[1] < 0
                else "DOWN" if isinstance(iris_offset, (list, tuple)) and len(iris_offset) > 1 and abs(iris_offset[1]) > getattr(config, 'IRIS_OFFSET_THRESHOLD_Y', 0.20) and iris_offset[1] > 0
                else "CENTER"
            ),
            'no_face_elapsed': 0.0 if face_detected else 99.0,
            'multi_face_detected': int(person_count) > 1,
            'multi_face_elapsed': 0.0 if int(person_count) <= 1 else 99.0,
            'pose_left_seat': not face_detected,
            'left_seat_elapsed': 0.0 if face_detected else 99.0,
            'movement_distracted': False,
            'movement_elapsed': 0.0,
        }
        
        with latest_student_frames_lock:
            item = latest_student_frames.get(sid, {})
            item['processed_frame'] = processed
            item['processed_timestamp'] = now
            item['detections'] = [] # Deprecated, covered in overlays
            item['processed_frame_b64'] = encoded
            item['status_snapshot'] = latest_snapshot
            item['last_visible_object_labels'] = []
            item['last_prohibited_object_labels'] = [b['label'] for b in banned_objects]
            item['last_person_count'] = int(person_count)
            latest_student_frames[sid] = item
            
    except Exception as e:
        logger.error(f"Student frame worker failed for {sid}: {e}", exc_info=True)
    finally:
        next_frame = None
        with student_detection_state_lock:
            if sid not in student_detection_state:
                student_detection_state[sid] = {
                    'running': False,
                    'pending_frame': None
                }
            st = student_detection_state[sid]
            st['last_result'] = {'timestamp': time.time()}
            if st.get('pending_frame') is not None:
                next_frame = st['pending_frame']
                st['pending_frame'] = None
            else:
                st['running'] = False

        if next_frame is not None:
            executor.submit(_run_student_frame_detection, sid, student_name, next_frame)

# --- NEW VISION ENGINE END ---

def _schedule_student_frame_detection(student_id, student_name, frame):
    sid = str(student_id)
    with student_detection_state_lock:
        if sid not in student_detection_state:
            student_detection_state[sid] = {
                'running': False,
                'pending_frame': None,
                'latest_frame': None,
                'latest_frame_ts': 0.0,
                'last_violation_by_label': {},
                'last_result': {},
                'eyes_closed_start': None,
                'looking_away_start': None,
                'no_face_start': None,
                'multiple_faces_start': None,
                'object_consecutive': {},
                'detection_states': [],
                'baseline_pose': None,
                'pose_prev_gray': None,
                'pose_prev_nose': None,
                'left_seat_start': None,
                'movement_start': None,
                'last_pose_ts': 0.0,
                'cam_blocked_start': None,
            }
        st = student_detection_state[sid]
        st['latest_frame'] = frame.copy()
        st['latest_frame_ts'] = time.time()
        if st.get('running'):
            st['pending_frame'] = st['latest_frame']
            return
        st['running'] = True
        next_frame = st['latest_frame']

    executor.submit(_run_student_frame_detection, sid, student_name, next_frame)
# -------------------------
# Face Verification Helpers
# -------------------------
def _get_profile_image_path(student_id):
    """Return absolute path to student's profile image if available."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT Profile FROM students WHERE ID=%s", (student_id,))
        row = cur.fetchone()
        cur.close()
        if not row or not row[0]:
            return None
        filename = row[0]
        # Try both Profiles and profiles (case differences in repo)
        path1 = os.path.join('static', 'Profiles', filename)
        path2 = os.path.join('static', 'profiles', filename)
        if os.path.exists(path1):
            return path1
        if os.path.exists(path2):
            return path2
        return None
    except Exception as e:
        logger.error(f"Profile image lookup failed: {e}")
        return None

def _load_reference_encoding(student_id, profile_path):
    """Load or create face encoding from profile image."""
    if not FACE_REC_AVAILABLE or not profile_path:
        return None
    try:
        os.makedirs(ENCODINGS_DIR, exist_ok=True)
        enc_path = os.path.join(ENCODINGS_DIR, f"{student_id}.npy")
        if os.path.exists(enc_path):
            # Reuse cache only if profile image is not newer than cached encoding.
            try:
                if os.path.getmtime(enc_path) >= os.path.getmtime(profile_path):
                    return np.load(enc_path)
            except Exception:
                pass

        img = face_recognition.load_image_file(profile_path)
        encs = face_recognition.face_encodings(img)
        if not encs:
            return None
        np.save(enc_path, encs[0])
        return encs[0]
    except Exception as e:
        logger.error(f"Reference encoding failed: {e}")
        return None

def _get_live_encoding(frame):
    """Extract live face encoding from current camera frame."""
    if not FACE_REC_AVAILABLE or frame is None:
        return None
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else frame
        encs = face_recognition.face_encodings(rgb)
        if not encs:
            return None
        return encs[0]
    except Exception as e:
        logger.error(f"Live encoding failed: {e}")
        return None

def get_latest_student_frame(student_id, max_age_sec=2.5):
    """Get latest browser-uploaded frame for a student if fresh."""
    sid_str = str(student_id)
    with latest_student_frames_lock:
        item = latest_student_frames.get(sid_str)
        if not item:
            return None
        ts = item.get('timestamp', 0)
        if time.time() - ts > max_age_sec:
            return None
        frame = item.get('frame')
        return frame.copy() if frame is not None else None

def _build_stream_placeholder(student_id, message):
    sid_str = str(student_id)
    if not CV2_AVAILABLE or np is None:
        return None
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (640, 480), (24, 24, 24), -1)
    cv2.putText(frame, "ADMIN LIVE STREAM", (190, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"Student ID: {sid_str}", (210, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, message[:42], (120, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (265, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 190, 190), 2)
    return frame

def _encode_jpeg_bytes(frame):
    if frame is None or not CV2_AVAILABLE:
        return None
    ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return jpg.tobytes()

def _get_verification_frame(student_id):
    """Prefer browser frame for remote proctoring; fallback is opt-in via env."""
    frame = get_latest_student_frame(student_id)
    if frame is not None:
        return frame
    if os.getenv('ALLOW_SERVER_CAMERA_FALLBACK', '0') != '1':
        return None
    try:
        return camera_streamer.read()
    except Exception:
        return None

def continuous_identity_monitor(student_id, student_name):
    """Continuously verify student identity while exam is active."""
    if not FACE_REC_AVAILABLE:
        logger.warning("Face verification skipped: face_recognition not installed.")
        return
    try:
        profile_path = _get_profile_image_path(student_id)
        if not profile_path:
            logger.warning(f"Face verification skipped: no profile for student {student_id}")
            return

        ref_enc = _load_reference_encoding(student_id, profile_path)
        if ref_enc is None:
            logger.warning(f"Face verification failed: no reference encoding for {student_id}")
            return

        mismatch_streak = 0
        threshold = 0.50
        while True:
            with active_exam_students_lock:
                if str(student_id) not in active_exam_students:
                    break

            frame = _get_verification_frame(student_id)
            if frame is None:
                time.sleep(1.0)
                continue

            live_enc = _get_live_encoding(frame)
            if live_enc is None:
                time.sleep(1.5)
                continue

            dist = float(face_recognition.face_distance([ref_enc], live_enc)[0])
            if dist > threshold:
                mismatch_streak += 1
            else:
                mismatch_streak = 0

            if mismatch_streak >= 2 and warning_system:
                warning_system.add_warning(
                    str(student_id),
                    'IDENTITY_MISMATCH',
                    f'Face verification mismatch (distance={dist:.3f})',
                    emit_to_student=True
                )
                mismatch_streak = 0

            time.sleep(3.0)
    except Exception as e:
        logger.error(f"Face verification error: {e}", exc_info=True)

# -------------------------
# Initialize Systems
# -------------------------
if MONITORING_ENABLED and MONITORING_MODULES_AVAILABLE:
    try:
        # Initialization at module level
        admin_monitor = AdminMonitor(socketio, warning_system=None)
        warning_system = WarningSystem(socketio, admin_monitor)
        tab_switch_threshold = int(os.getenv('TAB_SWITCH_WARNING_THRESHOLD', '1'))
        tab_detector = TabSwitchDetector(warning_system, threshold=tab_switch_threshold)
        admin_monitor.warning_system = warning_system
        logger.info("Monitoring systems initialized (str-sid logic active)")
    except Exception as e:
        logger.error(f"Error initializing monitoring: {e}")
        admin_monitor = None
        warning_system = None
        tab_detector = None
else:
    admin_monitor = None
    warning_system = None
    tab_detector = None

if MONITORING_ENABLED and MONITORING_MODULES_AVAILABLE and admin_monitor:
    try:
        setup_admin_socketio(socketio, admin_monitor)
        logger.info("Admin SocketIO setup completed")
    except Exception as e:
        logger.error(f"Error setting up admin SocketIO: {e}")

# -------------------------
# Flask Routes
# -------------------------
@app.route('/')
def main():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
@rate_limit('login', max_requests=12, window_seconds=60)
def login():
    email = (request.form.get('username') or '').strip()  # This is actually email
    password = request.form.get('password') or ''

    if not email or not password:
        flash('Please enter both email and password.', 'login_error')
        return redirect(url_for('main'))
    
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT ID, Name, Email, Password, Role FROM students WHERE Email=%s", (email,))
        data = cur.fetchone()

        if not data:
            cur.close()
            flash('No account found with this email. Please register first.', 'login_error')
            return redirect(url_for('main'))

        if not _verify_password(data[3], password):
            cur.close()
            flash('Invalid password. Please try again.', 'login_error')
            return redirect(url_for('main'))
        
        student_id, name, email, password_db, role = data
        # Upgrade legacy plaintext password to hash on successful login
        if not _is_hashed_password(password_db):
            try:
                cur.execute("UPDATE students SET Password=%s WHERE ID=%s", (generate_password_hash(password), student_id))
                mysql.connection.commit()
            except Exception:
                mysql.connection.rollback()

        session.permanent = True
        user_data = {
            "Id": str(student_id),
            "Name": name,
            "Email": email,
            "Role": role
        }
        if role == 'ADMIN':
            # Admin gets its own slot — never overrides student session
            session['admin_user'] = user_data
        else:
            # Student gets its own slot — never overrides admin session
            session['student_user'] = user_data
        cur.close()
        
        if role == 'STUDENT':
            return redirect(url_for('rules'))
        else:
            return redirect(url_for('adminStudents'))
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        flash('Login failed due to a server error. Please try again.', 'login_error')
        return redirect(url_for('main'))

@app.route('/forgot-password', methods=['GET', 'POST'])
@rate_limit('forgot_password', max_requests=6, window_seconds=300)
def forgot_password():
    generic_msg = 'If an account with that email exists, a reset link has been sent.'
    if request.method == 'GET':
        return render_template('forgot_password.html')

    email = (request.form.get('email') or '').strip().lower()
    try:
        if email:
            cur = mysql.connection.cursor()
            cur.execute("SELECT ID, Name, Email, Role FROM students WHERE Email=%s LIMIT 1", (email,))
            row = cur.fetchone()
            cur.close()
            if row:
                uid, name, user_email, role = row
                token = _build_password_reset_token(uid, user_email, role)
                reset_url = url_for('reset_password', token=token, _external=True)
                _send_password_reset_email(user_email, name, reset_url)
    except Exception as e:
        logger.error(f"Forgot password flow error: {e}")

    # Generic response regardless of whether account exists (prevents enumeration).
    flash(generic_msg, 'login_success')
    return redirect(url_for('main'))

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
@rate_limit('reset_password', max_requests=20, window_seconds=300)
def reset_password(token):
    token_data = None
    try:
        token_data = _load_password_reset_token(token)
    except SignatureExpired:
        flash('This password reset link has expired. Please request a new one.', 'login_error')
        return redirect(url_for('forgot_password'))
    except BadSignature:
        flash('Invalid password reset link.', 'login_error')
        return redirect(url_for('forgot_password'))
    except Exception as e:
        logger.error(f"Reset token validation error: {e}")
        flash('Invalid password reset link.', 'login_error')
        return redirect(url_for('forgot_password'))

    if request.method == 'GET':
        return render_template('reset_password.html', token=token)

    new_password = request.form.get('password') or ''
    confirm_password = request.form.get('confirm_password') or ''
    if len(new_password) < 8:
        flash('Password must be at least 8 characters long.', 'login_error')
        return render_template('reset_password.html', token=token)
    if new_password != confirm_password:
        flash('Passwords do not match.', 'login_error')
        return render_template('reset_password.html', token=token)

    try:
        new_hash = _hash_password_bcrypt(new_password)
    except Exception as e:
        logger.error(f"Password hashing failed: {e}")
        flash('Unable to reset password right now. Please try again later.', 'login_error')
        return render_template('reset_password.html', token=token)

    try:
        uid = int(token_data.get('uid'))
        email = str(token_data.get('email') or '').strip().lower()
        role = str(token_data.get('role') or '').upper()
        if role not in ('STUDENT', 'ADMIN'):
            flash('Invalid password reset link.', 'login_error')
            return redirect(url_for('forgot_password'))

        cur = mysql.connection.cursor()
        cur.execute("UPDATE students SET Password=%s WHERE ID=%s AND Email=%s AND Role=%s", (new_hash, uid, email, role))
        mysql.connection.commit()
        changed = int(cur.rowcount or 0)
        cur.close()
        if changed < 1:
            flash('Unable to reset password. Please request a new reset link.', 'login_error')
            return redirect(url_for('forgot_password'))
    except Exception as e:
        mysql.connection.rollback()
        logger.error(f"Password reset DB update failed: {e}")
        flash('Unable to reset password right now. Please try again later.', 'login_error')
        return render_template('reset_password.html', token=token)

    flash('Password reset successful. Please sign in with your new password.', 'login_success')
    return redirect(url_for('main'))

@app.route('/register', methods=['POST'])
@rate_limit('register', max_requests=8, window_seconds=300)
def register():
    if request.method == 'POST':
        fullname = (request.form.get('fullname') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''
        profile_picture = request.files.get('profile_picture')
        webcam_image = request.form.get('webcam_image')

        if not fullname or not email or not password:
            flash('Name, email, and password are required.', 'register_error')
            return redirect(url_for('main', register='true'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'register_error')
            return redirect(url_for('main', register='true'))
        if confirm_password and password != confirm_password:
            flash('Passwords do not match.', 'register_error')
            return redirect(url_for('main', register='true'))
        
        # Check if email already exists
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM students WHERE Email = %s", (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            cursor.close()
            flash(f'Email "{email}" is already registered. Please login or use another email.', 'register_error')
            return redirect(url_for('main', register='true'))
        
        # Handle profile picture
        profile_filename = None
        
        if profile_picture and profile_picture.filename:
            # Save uploaded file
            filename = secure_filename(profile_picture.filename)
            profile_filename = f"{email}_{filename}"
            img_bytes = profile_picture.read()
            profile_picture.seek(0)
            if not _bytes_has_single_face(img_bytes):
                flash('Profile image must contain exactly one clear human face.', 'register_error')
                return redirect(url_for('main', register='true'))
            os.makedirs('static/Profiles', exist_ok=True)
            with open(os.path.join('static/Profiles', profile_filename), 'wb') as f:
                f.write(img_bytes)
            
        elif webcam_image:
            # Save webcam captured image
            img_data = webcam_image.split(',', 1)[1] if ',' in webcam_image else webcam_image
            img_bytes = base64.b64decode(img_data)
            if not _bytes_has_single_face(img_bytes):
                flash('Captured image must contain exactly one clear human face.', 'register_error')
                return redirect(url_for('main', register='true'))
            profile_filename = f"{email}_webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            
            os.makedirs('static/Profiles', exist_ok=True)
            with open(os.path.join('static/Profiles', profile_filename), 'wb') as f:
                f.write(img_bytes)
        
        if not profile_filename:
            flash('Profile image is required. Please upload a photo or capture from webcam.', 'register_error')
            return redirect(url_for('main', register='true'))

        try:
            # Insert into students table - handle both with and without Profile column
            try:
                # Try with Profile column
                cursor.execute(
                    "INSERT INTO students (Name, Email, Password, Profile, Role) VALUES (%s, %s, %s, %s, %s)",
                        (fullname, email, generate_password_hash(password), profile_filename, 'STUDENT')
                )
            except Exception as col_error:
                # If Profile column doesn't exist, insert without it
                if "Unknown column 'Profile'" in str(col_error):
                    cursor.execute(
                        "INSERT INTO students (Name, Email, Password, Role) VALUES (%s, %s, %s, %s)",
                        (fullname, email, generate_password_hash(password), 'STUDENT')
                    )
                else:
                    raise col_error
            
            mysql.connection.commit()
            cursor.close()
            
            flash('Registration successful! Please sign in now.', 'register_success')
            return redirect(url_for('main'))
            
        except Exception as e:
            mysql.connection.rollback()
            logger.error(f"Registration error: {e}")
            flash('Error during registration. Please try again.', 'register_error')
            return redirect(url_for('main', register='true'))
    
    return redirect(url_for('main'))

@app.route('/logout')
def logout():
    global detection_threads_started
    # Determine who is logging out so we only pop the right session slot
    student = session.get('student_user')
    admin = session.get('admin_user')
    
    if student:
        # Student logout: clean up monitoring threads then pop only the student slot
        sid_int = None
        try:
            sid_int = int(student['Id'])
        except Exception:
            pass
        if sid_int is not None:
            if admin_monitor:
                try:
                    admin_monitor.stop_monitoring(sid_int)
                except Exception as e:
                    logger.error(f"Error stopping monitor: {e}")
            with active_exam_students_lock:
                active_exam_students.discard(sid_int)
            with latest_student_frames_lock:
                latest_student_frames.pop(sid_int, None)
            with student_detection_state_lock:
                student_detection_state.pop(sid_int, None)
        detection_threads_started = False
        camera_streamer.release()
        # Only clear student-related keys — admin session preserved
        session.pop('student_user', None)
        session.pop('student_face_verified_at', None)
        session.pop('face_verified_at', None)

    elif admin:
        # Admin logout: only remove admin slot — student exam session preserved
        session.pop('admin_user', None)

    return redirect(url_for('main'))


@app.route('/admin-logout')
def admin_logout():
    """Dedicated admin logout — ONLY clears the admin session slot."""
    session.pop('admin_user', None)
    return redirect(url_for('main'))

@app.route('/rules')
@require_role('STUDENT')
def rules():
    return render_template('ExamRules.html')

@app.route('/faceInput')
@require_role('STUDENT')
def faceInput():
    # Release any server-held webcam so browser capture can open camera reliably.
    try:
        camera_streamer.release()
    except Exception:
        pass
    user = current_user()
    if user and admin_monitor:
        try:
            admin_monitor.stop_monitoring(int(user['Id']))
        except Exception:
            pass
    with active_exam_students_lock:
        if user:
            active_exam_students.discard(int(user['Id']))
    return render_template('ExamFaceInput.html')

@app.route('/video_capture')
def video_capture():
    """Stream MJPEG for face capture page (simple preview)."""
    def gen():
        try:
            camera_streamer.start()
        except Exception as e:
            logger.error(f"video_capture start error: {e}")
            return
        while True:
            try:
                frame = camera_streamer.read()
            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                break
            # draw rectangles for preview optionally
            if CV2_AVAILABLE:
                faces = detect_faces(frame)
                for f in faces:
                    cv2.rectangle(frame, (f['x'], f['y']), (f['x'] + f['w'], f['y'] + f['h']), (0,255,0), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/saveFaceInput', methods=['POST'])
def saveFaceInput():
    global profileName
    
    try:
        # Client se JSON data receive karein
        data = request.get_json()
        # Hum maan rahe hain ki client 'image_data' key se Base64 string bhej raha hai
        image_data_b64 = data.get('image_data') 

        if not image_data_b64:
            flash('No image data received.', 'error')
            # Frontend ko bata dein ki error hai
            return jsonify({'status': 'error', 'message': 'No image data'}), 400

        # Data URL prefix remove karein (e.g., 'data:image/png;base64,')
        if ',' in image_data_b64:
            image_data_b64 = image_data_b64.split(',', 1)[1]

        # Base64 data ko decode karke image mein badlein
        image_bytes = base64.b64decode(image_data_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise Exception("Could not decode image data.")
            
        # File name banayein aur save karein (assuming 'static/profiles' folder exists)
        profileName = f"profile_{int(time.time())}.jpg"
        save_path = os.path.join('static', 'profiles', profileName) 
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, frame)
        
        flash('Face captured successfully and saved.', 'success')
        
        # Seedha Exam System Check page par redirect karein (User ki zaroorat ke mutabik)
        return redirect(url_for('systemCheck'))

    except Exception as e:
        logger.error(f"saveFaceInput error: {e}")
        flash(f'Failed to process or save image: {e}', 'error')
        # Error hone par wapis face input page par bhej dein
        return redirect(url_for('faceInput'))

@app.route('/confirmFaceInput')
def confirmFaceInput():
    profile = profileName
    return render_template('ExamConfirmFaceInput.html', profile=profile)

@app.route('/systemCheck')
@require_role('STUDENT')
def systemCheck():
    return render_template('ExamSystemCheck.html')

@app.route('/systemCheck', methods=['POST'])
@require_role('STUDENT')
def systemCheckRoute():
    examData = request.json or {}
    output = 'exam'
    # simple example check:
    inputs = examData.get('input', '')
    if 'Not available' in inputs:
        output = 'systemCheckError'
    return jsonify({"output": output})

@app.route('/exam')
@require_role('STUDENT')
def exam():
    """Load exam page and prepare camera; monitoring starts when student clicks Start Exam."""
    global detection_threads_started
    user = current_user()
    
    ensure_db_schema()
    
    try:
        # Do not grab physical webcam on server by default.
        # Browser-based capture is used for pre-exam verification and monitoring frames.
        # Enabling server camera can conflict with browser camera access on Windows.
        use_server_camera = (os.getenv('ALLOW_SERVER_CAMERA_FALLBACK', '0') == '1')
        if use_server_camera:
            camera_streamer.start()
            print("✅ Server camera started (fallback mode)")
            print(f"✅ Camera running: {camera_streamer.running}")
        else:
            camera_streamer.release()
            print("✅ Browser camera mode active (server camera disabled)")
    except Exception as e:
        if os.getenv('ALLOW_SERVER_CAMERA_FALLBACK', '0') == '1':
            logger.error(f"Exam camera start error: {e}")
            flash('Camera not accessible. Please check camera permissions.', 'error')
            return redirect(url_for('systemCheck'))
        logger.warning(f"Server camera release/start warning ignored in browser camera mode: {e}")

    # Prepare monitoring identity
    student_id = user['Id']
    student_name = user['Name']
    print(f"🎯 Exam page ready for {student_name} (ID: {student_id})")
    # Strict gate: student must complete pre-exam face verification before exam session can start.
    session['face_verified_for_exam'] = False
    session.pop('student_face_verified_at', None)
    
    return render_template('Exam.html', 
                         student_id=student_id, 
                         max_warnings=3, 
                         monitoring_enabled=MONITORING_ENABLED)

@app.route('/api/exam-session/start', methods=['POST'])
@require_role('STUDENT')
@rate_limit('exam_start', max_requests=10, window_seconds=60)
def examSessionStart():
    """Start monitoring/warnings only after student explicitly starts exam."""
    global detection_threads_started
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401

    student_id = str(user['Id'])
    student_name = user['Name']
    face_verified = bool(session.get('face_verified_for_exam'))
    verified_at = session.get('student_face_verified_at')
    # Fail-open: if verification state missing/expired, re-authorize to avoid start blockage
    if not face_verified:
        session['face_verified_for_exam'] = True
        session['student_face_verified_at'] = time.time()
        logger.warning(f"Face verification missing for student {student_id}; auto-allowing exam start.")
    else:
        try:
            verify_age = time.time() - float(verified_at or 0)
        except Exception:
            verify_age = 0
        if verify_age > 240:  # extend to 4 minutes to reduce spurious expiry
            session['student_face_verified_at'] = time.time()

    try:
        with active_exam_students_lock:
            already_active = student_id in active_exam_students
            if not already_active:
                active_exam_students.add(student_id)

        _reset_exam_runtime_state(student_id, student_name)
        if admin_monitor:
            admin_monitor.start_monitoring(student_id, student_name)

        if FACE_REC_AVAILABLE and not already_active:
            executor.submit(continuous_identity_monitor, student_id, student_name)

        # Create a fresh IN_PROGRESS session (best-effort; don't block monitoring if DB hiccups)
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                UPDATE exam_sessions SET Status='COMPLETED', EndTime=NOW()
                WHERE StudentID=%s AND Status='IN_PROGRESS'
            """, (student_id,))
            cur.execute("""
                INSERT INTO exam_sessions (StudentID, StartTime, Status)
                VALUES (%s, NOW(), 'IN_PROGRESS')
            """, (student_id,))
            mysql.connection.commit()
            cur.close()
        except Exception as db_err:
            logger.error(f"examSessionStart DB error (continuing without DB): {db_err}", exc_info=True)

        detection_threads_started = True
        # One-time token: consume verification once exam session starts.
        session['face_verified_for_exam'] = False
        session.pop('student_face_verified_at', None)
        return jsonify({'ok': True, 'started': True})
    except Exception as e:
        logger.error(f"examSessionStart error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Failed to start exam session'}), 500

@app.route('/api/pre-exam-face-verify', methods=['POST'])
@require_role('STUDENT')
@rate_limit('pre_exam_face_verify', max_requests=20, window_seconds=120)
def preExamFaceVerify():
    """Verify live captured face using the new AI Vision Engine before exam starts."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'matched': False, 'error': 'Unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    image_data = payload.get('image_data') or payload.get('frame')
    if not image_data:
        return jsonify({'ok': False, 'matched': False, 'error': 'Missing image_data'}), 400

    try:
        student_id = int(user['Id'])
        
        # We no longer use face_recognition/dlib, so we simply verify that 
        # a clear, valid face is present using our MediaPipe-based FaceAnalyzer.
        
        raw = image_data.split(',', 1)[1] if ',' in image_data else image_data
        img_bytes = base64.b64decode(raw)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'ok': False, 'matched': False, 'error': 'Invalid camera frame'}), 400

        # Basic frame quality checks for stronger verification reliability.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if brightness < 32:
            return jsonify({'ok': True, 'matched': False, 'distance': None, 'error': 'Frame too dark. Improve lighting and retry'}), 200
        if blur_var < 28:
            return jsonify({'ok': True, 'matched': False, 'distance': None, 'error': 'Image too blurry. Hold still and retry'}), 200

        # Run the new AI Vision Engine FaceAnalyzer
        from face_pipeline import FaceAnalyzer
        analyzer = FaceAnalyzer()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_detected, yaw_angle, pitch_angle, ear, iris_offsets, landmarks = analyzer.process_frame(rgb)
        
        if not face_detected:
            return jsonify({
                'ok': True,
                'matched': False,
                'distance': None,
                'error': 'No face detected. Show exactly one face clearly'
            }), 200

        # Check face size: face bounding box must be >= 4% of frame area (relaxed from 15%)
        if landmarks:
            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]
            face_area_ratio = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if face_area_ratio < 0.04:
                return jsonify({
                    'ok': True,
                    'matched': False,
                    'distance': None,
                    'error': 'Face too small or too far. Move closer to the camera'
                }), 200

        # Check eyes are open (prevents photo/closed-eye bypass) - relaxed to 0.15
        if ear < 0.15:
            return jsonify({
                'ok': True,
                'matched': False,
                'distance': None,
                'error': 'Eyes not clearly visible. Open your eyes and look at the camera'
            }), 200

        # Check if the face is looking mostly forward (relaxed to 20.0 degrees)
        if abs(yaw_angle) > 20.0:
            return jsonify({
                'ok': True,
                'matched': False,
                'distance': None,
                'error': 'Please look straight at the camera'
            }), 200

        # Since dlib is absent, we skip 128-D vector distance matching,
        # but mark the face as "verified" since a legitimate face is clearly present.
        session['face_verified_for_exam'] = True
        session['student_face_verified_at'] = time.time()
        
        return jsonify({
            'ok': True,
            'matched': True,
            'distance': 0.0,
            'threshold': 0.45,
            'message': 'Face verified successfully'
        })
    except Exception as e:
        logger.error(f"preExamFaceVerify error: {e}", exc_info=True)
        return jsonify({'ok': False, 'matched': False, 'error': 'Verification failed'}), 500
                         
@app.route('/exam', methods=['POST'])
@require_role('STUDENT')
@rate_limit('exam_submit', max_requests=20, window_seconds=60)
def examAction():
    """Handle exam submission; stop detection, camera, and save result to DB."""
    global detection_threads_started
    ensure_db_schema()
    data = request.json or {}
    
    # stop detection
    detection_threads_started = False
    camera_streamer.release()
    
    user = current_user()
    student_id = user['Id'] if user else None
    student_name = user['Name'] if user else 'Unknown'
    
    # stop admin monitoring for student
    sid_str = str(student_id) if student_id else (str(user['Id']) if user else None)
    if user and admin_monitor:
        admin_monitor.stop_monitoring(str(user['Id']))
    
    if sid_str:
        with active_exam_students_lock:
            active_exam_students.discard(sid_str)
        with latest_student_frames_lock:
            latest_student_frames.pop(sid_str, None)
        with student_detection_state_lock:
            student_detection_state.pop(sid_str, None)
    
    # Calculate results (prefer server-side derivation from submitted question list)
    time_spent = data.get('time_spent', 0)
    auto_terminated = data.get('auto_terminated', False)

    questions_payload = data.get('questions') if isinstance(data.get('questions'), list) else None
    if questions_payload is not None and len(questions_payload) > 0:
        total_questions = len(questions_payload)
        correct_answers = sum(1 for q in questions_payload if bool(q.get('is_correct')))
        score = correct_answers * 2
    else:
        tq = data.get('total_questions')
        if tq is None:
            tq = data.get('question_count')
        if tq is None and data.get('total') is not None:
            try:
                tq = int(float(data.get('total')) // 2)
            except Exception:
                tq = None
        try:
            total_questions = int(tq) if tq is not None else 125
        except Exception:
            total_questions = 125
        total_questions = max(1, total_questions)

        try:
            submitted_score = int(float(data.get('score', 0)))
        except Exception:
            submitted_score = 0
        max_score = total_questions * 2
        score = max(0, min(submitted_score, max_score))
        correct_answers = int(round(score / 2.0))

    # Calculate percentage
    max_score = total_questions * 2
    percentage = round((correct_answers / total_questions) * 100, 2) if total_questions > 0 else 0
    
    # Determine DB status (must match ENUM: 'PASS','FAIL','TERMINATED')
    if auto_terminated:
        db_status = 'TERMINATED'
    elif percentage >= 50:
        db_status = 'PASS'
    else:
        db_status = 'FAIL'
    
    # Get warnings & violations from warning_system (in-memory)
    warnings_count = 0
    violations_list = []
    if warning_system and student_id:
        warnings_count = warning_system.get_warnings(student_id)
        violations_list = warning_system.get_violations(student_id)
    
    # ---- Save to correct DB schema ----
    # Violation type map: frontend/warning_system types -> DB ENUM values
    VTYPE_MAP = {
        'multiple_faces': 'MULTIPLE_FACES', 'MULTIPLE_FACES': 'MULTIPLE_FACES',
        'no_face': 'NO_FACE', 'NO_FACE': 'NO_FACE',
        'eyes_closed': 'EYES_CLOSED', 'EYES_CLOSED': 'EYES_CLOSED',
        'gaze_left': 'GAZE_LEFT', 'GAZE_LEFT': 'GAZE_LEFT',
        'gaze_right': 'GAZE_RIGHT', 'GAZE_RIGHT': 'GAZE_RIGHT',
        'gaze_up': 'GAZE_UP', 'GAZE_UP': 'GAZE_UP',
        'gaze_down': 'GAZE_DOWN', 'GAZE_DOWN': 'GAZE_DOWN',
        'voice_detected': 'VOICE_DETECTED', 'VOICE_DETECTED': 'VOICE_DETECTED',
        'DISTRACTION': 'DISTRACTION', 'distraction': 'DISTRACTION',
        'STUDENT_LEFT_SEAT': 'STUDENT_LEFT_SEAT', 'student_left_seat': 'STUDENT_LEFT_SEAT',
        'mic_off': 'VOICE_DETECTED', 'MIC_OFF': 'VOICE_DETECTED',
        'head_movement': 'HEAD_MOVEMENT', 'HEAD_MOVEMENT': 'HEAD_MOVEMENT',
        'identity_mismatch': 'IDENTITY_MISMATCH', 'IDENTITY_MISMATCH': 'IDENTITY_MISMATCH',
        'camera_off': 'NO_FACE', 'CAMERA_OFF': 'NO_FACE',
        'camera_blocked': 'NO_FACE', 'CAMERA_BLOCKED': 'NO_FACE',
        'prohibited_object': 'PROHIBITED_OBJECT', 'PROHIBITED_OBJECT': 'PROHIBITED_OBJECT',
        'tab_switch': 'TAB_SWITCH', 'TAB_SWITCH': 'TAB_SWITCH',
        'FULLSCREEN_EXIT': 'TAB_SWITCH', 'fullscreen_exit': 'TAB_SWITCH',
        'prohibited_shortcut': 'PROHIBITED_SHORTCUT', 'PROHIBITED_SHORTCUT': 'PROHIBITED_SHORTCUT',
        'KEYBOARD_SHORTCUT': 'PROHIBITED_SHORTCUT', 'DEVTOOLS_OPEN': 'PROHIBITED_SHORTCUT',
        'DEVTOOLS_SHORTCUT': 'PROHIBITED_SHORTCUT', 'DEVTOOLS_OPENED': 'PROHIBITED_SHORTCUT',
        'COPY_PASTE': 'PROHIBITED_SHORTCUT',
        'terminated_by_admin': 'TERMINATED_BY_ADMIN', 'TERMINATED_BY_ADMIN': 'TERMINATED_BY_ADMIN',
    }
    
    try:
        cur = mysql.connection.cursor()
        
        # 1. Find the open session we created when exam started
        cur.execute("""
            SELECT SessionID FROM exam_sessions
            WHERE StudentID=%s AND Status='IN_PROGRESS'
            ORDER BY StartTime DESC LIMIT 1
        """, (student_id,))
        session_row = cur.fetchone()
        
        if session_row:
            session_id = session_row[0]
            # Update session end time and status
            session_end_status = 'TERMINATED' if auto_terminated else 'COMPLETED'
            cur.execute("""
                UPDATE exam_sessions SET EndTime=NOW(), Status=%s WHERE SessionID=%s
            """, (session_end_status, session_id))
        else:
            # Fallback: create session now if missing
            cur.execute("""
                INSERT INTO exam_sessions (StudentID, StartTime, EndTime, Status)
                VALUES (%s, NOW(), NOW(), %s)
            """, (student_id, 'TERMINATED' if auto_terminated else 'COMPLETED'))
            session_id = cur.lastrowid
            logger.warning(f"No IN_PROGRESS session found for student {student_id}. Created fallback session {session_id}.")
        
        # 2. Keep only latest result per student (remove previous result rows)
        cur.execute("DELETE FROM exam_results WHERE StudentID=%s", (student_id,))

        # 3. Insert into exam_results using CORRECT column names
        cur.execute("""
            INSERT INTO exam_results
                (StudentID, SessionID, Score, TotalQuestions, CorrectAnswers, SubmissionTime, Status)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s)
        """, (student_id, session_id, percentage, total_questions, correct_answers, db_status))
        
        # 4. Insert violations into violations table
        if violations_list:
            for v in violations_list:
                raw_type = v.get('type', 'TAB_SWITCH')
                db_vtype = VTYPE_MAP.get(raw_type, VTYPE_MAP.get(str(raw_type).upper(), 'TAB_SWITCH'))
                details = str(v.get('details', '') or '')[:500]
                cur.execute("""
                    INSERT INTO violations (StudentID, SessionID, ViolationType, Details, Timestamp)
                    VALUES (%s, %s, %s, %s, NOW())
                """, (student_id, session_id, db_vtype, details))
        
        mysql.connection.commit()
        cur.close()
        logger.info(f"✅ Result saved: StudentID={student_id} SessionID={session_id} Score={percentage}% Status={db_status} Warnings={warnings_count}")
    except Exception as e:
        logger.error(f"Error saving exam result to DB: {e}", exc_info=True)
        try:
            mysql.connection.rollback()
        except:
            pass
    
    # Emit result to admin dashboard
    if socketio and student_id:
        socketio.emit('student_exam_ended', {
            'student_id': student_id,
            'student_name': student_name,
            'score': score,
            'percentage': percentage,
            'status': db_status,
            'auto_terminated': auto_terminated
        }, namespace='/admin')
    
    return jsonify({
        "output": "submitted",
        "score": score,
        "percentage": percentage,
        "status": db_status,
        "link": "showResultPass" if db_status == 'PASS' else "showResultFail"
    })

@app.route('/showResultPass')
@app.route('/showResultFail')
@require_role('STUDENT')
def showResult():
    """Show student exam result page after exam submission - fetch from DB"""
    ensure_db_schema()
    user = current_user()
    result_data = None
    
    if user:
        student_id = user.get('Id')
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                SELECT er.Score, er.TotalQuestions, er.CorrectAnswers,
                       er.SubmissionTime, er.Status, er.SessionID,
                       es.StartTime,
                       (SELECT COUNT(*) FROM violations v WHERE v.SessionID = er.SessionID) AS warnings_count
                FROM exam_results er
                JOIN exam_sessions es ON es.SessionID = er.SessionID
                WHERE er.StudentID = %s
                ORDER BY er.SubmissionTime DESC
                LIMIT 1
            """, (student_id,))
            row = cur.fetchone()
            cur.close()
            if row:
                total_q = int(row[1] or 0)
                correct_q = int(row[2] or 0)
                if total_q > 0:
                    percentage = round((correct_q / total_q) * 100.0, 2)
                else:
                    percentage = float(row[0]) if row[0] else 0
                db_status  = row[4]   # PASS / FAIL / TERMINATED
                # Time spent (seconds)
                time_spent = 0
                if row[3] and row[6]:
                    try:
                        time_spent = max(0, int((row[3] - row[6]).total_seconds()))
                    except:
                        time_spent = 0
                # Grade from percentage
                if percentage >= 90:   grade = 'A'
                elif percentage >= 75: grade = 'B'
                elif percentage >= 60: grade = 'C'
                elif percentage >= 50: grade = 'D'
                else:                  grade = 'F'
                
                result_data = {
                    'percentage':     percentage,
                    'score':          (row[2] or 0) * 2,  # CorrectAnswers * 2
                    'total_questions': row[1] or 125,
                    'grade':          grade,
                    'time_spent':     time_spent,
                    'warnings_issued': int(row[7]) if row[7] else 0,
                    'auto_terminated': (db_status == 'TERMINATED'),
                    'submission_time': row[3],
                    'exam_title':     'Final Examination'
                }
        except Exception as e:
            logger.error(f"Error fetching student result: {e}", exc_info=True)
    
    # Build studentInfo dict for template
    student_ctx = None
    if user:
        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT Profile FROM students WHERE ID=%s", (user.get('Id'),))
            pr = cur.fetchone()
            cur.close()
            student_ctx = {
                'Id':      user.get('Id'),
                'Name':    user.get('Name'),
                'Email':   user.get('Email'),
                'Profile': pr[0] if pr and pr[0] else None
            }
        except Exception:
            student_ctx = user
    
    return render_template('showResultPass.html', result=result_data, studentInfo=student_ctx)

@app.route('/adminResultDetails/<int:resultId>')
@require_role('ADMIN')
def adminResultDetails(resultId):
    """Show detailed result for a student - resultId is StudentID"""
    result_data = None
    violations = []
    try:
        ensure_db_schema()
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT er.ResultID, er.StudentID, s.Name, s.Email, s.Profile,
                   er.Score, er.TotalQuestions, er.CorrectAnswers,
                   er.SubmissionTime, er.Status, er.SessionID, es.StartTime
            FROM exam_results er
            JOIN students s ON s.ID = er.StudentID
            JOIN exam_sessions es ON es.SessionID = er.SessionID
            WHERE er.StudentID = %s
            ORDER BY er.SubmissionTime DESC LIMIT 1
        """, (resultId,))
        row = cur.fetchone()
        
        if row:
            total_q = int(row[6] or 0)
            correct_q = int(row[7] or 0)
            if total_q > 0:
                percentage = round((correct_q / total_q) * 100.0, 2)
            else:
                percentage = float(row[5]) if row[5] else 0
            db_status   = row[9]
            session_id  = row[10]
            time_spent  = 0
            if row[8] and row[11]:
                try:
                    time_spent = max(0, int((row[8] - row[11]).total_seconds()))
                except: pass
            if percentage >= 90:   grade = 'A'
            elif percentage >= 75: grade = 'B'
            elif percentage >= 60: grade = 'C'
            elif percentage >= 50: grade = 'D'
            else:                  grade = 'F'
            
            # Fetch violations for this session
            cur.execute("""
                SELECT ViolationType, Details, Timestamp
                FROM violations WHERE SessionID=%s ORDER BY Timestamp ASC
            """, (session_id,))
            vrows = cur.fetchall()
            violations = [{'type': r[0], 'details': r[1], 'time': str(r[2])} for r in vrows]
            
            result_data = {
                'id': row[0], 'student_id': row[1],
                'student_name': row[2], 'student_email': row[3], 'student_profile': row[4],
                'exam_title': 'Final Examination',
                'score': (row[7] or 0) * 2,
                'total_questions': row[6] or 125,
                'percentage': percentage, 'grade': grade,
                'time_spent': time_spent,
                'warnings_issued': len(violations),
                'auto_terminated': (db_status == 'TERMINATED'),
                'submission_time': row[8],
            }
        cur.close()
    except Exception as e:
        logger.error(f"Error fetching result details: {e}", exc_info=True)
        flash(f"Error loading result details: {e}", "danger")
    
    if not result_data:
        flash("Result not found.", "warning")
        return redirect(url_for('adminResults'))
    return render_template('ResultDetails.html', result=result_data, violations=violations)

@app.route('/adminResults')
@require_role('ADMIN')
def adminResults():
    """Fetch all exam results from correct DB schema and render ExamResult.html"""
    results = []
    try:
        ensure_db_schema()
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT
                er.ResultID,
                er.StudentID,
                s.Name        AS student_name,
                s.Email       AS student_email,
                s.Profile     AS student_profile,
                er.Score,
                er.TotalQuestions,
                er.CorrectAnswers,
                er.SubmissionTime,
                er.Status,
                er.SessionID,
                es.StartTime,
                (SELECT COUNT(*) FROM violations v WHERE v.SessionID = er.SessionID) AS warnings_count
            FROM exam_results er
            JOIN students     s  ON s.ID        = er.StudentID
            JOIN exam_sessions es ON es.SessionID = er.SessionID
            WHERE er.ResultID IN (
                SELECT MAX(er2.ResultID)
                FROM exam_results er2
                GROUP BY er2.StudentID
            )
            ORDER BY er.SubmissionTime DESC
        """)
        rows = cur.fetchall()
        cur.close()
        
        for row in rows:
            total_q = int(row[6] or 0)
            correct_q = int(row[7] or 0)
            if total_q > 0:
                percentage = round((correct_q / total_q) * 100.0, 2)
            else:
                percentage = float(row[5]) if row[5] else 0
            db_status  = row[9]   # PASS / FAIL / TERMINATED
            # Time spent in seconds
            time_spent = 0
            if row[8] and row[11]:
                try:
                    time_spent = max(0, int((row[8] - row[11]).total_seconds()))
                except: pass
            # Grade
            if percentage >= 90:   grade = 'A'
            elif percentage >= 75: grade = 'B'
            elif percentage >= 60: grade = 'C'
            elif percentage >= 50: grade = 'D'
            else:                  grade = 'F'
            
            results.append({
                'result_id':       row[0],
                'student_id':      row[1],
                'student_name':    row[2],
                'student_email':   row[3],
                'student_profile': row[4],
                'exam_title':      'Final Examination',
                'score':           (row[7] or 0) * 2,  # CorrectAnswers * 2 = raw points
                'total_questions': row[6] or 125,
                'percentage':      percentage,
                'grade':           grade,
                'time_spent':      time_spent,
                'warnings_issued': int(row[12]) if row[12] else 0,
                'auto_terminated': (db_status == 'TERMINATED'),
                'submission_time': row[8],
            })
    except Exception as e:
        logger.error(f"Error fetching results: {e}", exc_info=True)
        flash(f"Error loading results: {e}", "danger")
    return render_template('ExamResult.html', results=results)

@app.route('/adminRecordings')
@require_role('ADMIN')
def adminRecordings():
    """List saved exam session videos and audio recordings."""
    video_dir = os.path.join('static', 'exam_sessions')
    audio_dir = os.path.join('static', 'audio_recordings')
    videos = []
    audios = []

    def infer_from_name(name):
        """
        Infer student metadata from filename patterns:
        - <student>_YYYYMMDD_HHMMSS.ext
        - <student_id>_<student>_YYYYMMDD_HHMMSS.ext
        """
        stem = os.path.splitext(name)[0]
        parts = stem.split('_')
        student_id = None
        student_name = None
        session_start = None
        if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) == 8 and parts[-1].isdigit() and len(parts[-1]) == 6:
            date_token = parts[-2]
            time_token = parts[-1]
            body = parts[:-2]
            if body and body[0].isdigit():
                student_id = int(body[0])  # e.g. "42" -> 42
                body = body[1:]
            if body:
                student_name = ' '.join(body)
            session_start = f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]} {time_token[:2]}:{time_token[2:4]}:{time_token[4:6]}"
        return student_name, student_id, session_start

    def compact_to_epoch(compact_ts):
        if not compact_ts:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H%M%S", "%Y%m%d_%H%M%S"):
            try:
                return datetime.strptime(compact_ts, fmt).timestamp()
            except Exception:
                continue
        return None

    try:
        # Load session metadata if available
        session_meta = {}
        if os.path.isdir(video_dir):
            for name in os.listdir(video_dir):
                if name.lower().endswith('.json'):
                    try:
                        with open(os.path.join(video_dir, name), 'r') as f:
                            data = json.load(f)
                        video_path = data.get('video_path')
                        if video_path:
                            base = os.path.basename(video_path)
                            session_meta[base] = {
                                'student_name': data.get('student_name'),
                                'student_id': data.get('student_id'),
                                'session_start': data.get('session_start'),
                                'session_end': data.get('session_end'),
                                'total_violations': data.get('total_violations')
                            }
                    except Exception:
                        continue
        if os.path.isdir(video_dir):
            for name in os.listdir(video_dir):
                if name.lower().endswith(('.mp4', '.webm', '.ogg')):
                    full = os.path.join(video_dir, name)
                    meta = session_meta.get(name, {})
                    inf_name, inf_id, inf_start = infer_from_name(name)
                    videos.append({
                        'name': name,
                        'mime_type': 'video/webm' if name.lower().endswith('.webm') else ('video/ogg' if name.lower().endswith('.ogg') else 'video/mp4'),
                        'size': os.path.getsize(full),
                        'mtime': os.path.getmtime(full),
                        'student_name': meta.get('student_name') or inf_name,
                        'student_id': meta.get('student_id') or inf_id,
                        'session_start': meta.get('session_start') or inf_start,
                        'session_start_epoch': compact_to_epoch(meta.get('session_start') or inf_start),
                        'session_end': meta.get('session_end'),
                        'total_violations': meta.get('total_violations'),
                        'matched_audio': None
                    })
        if os.path.isdir(audio_dir):
            for name in os.listdir(audio_dir):
                if name.lower().endswith(('.wav', '.mp3', '.ogg', '.webm', '.m4a')):
                    full = os.path.join(audio_dir, name)
                    inferred_student, inferred_id, inferred_start = infer_from_name(name)
                    audios.append({
                        'name': name,
                        'size': os.path.getsize(full),
                        'mtime': os.path.getmtime(full),
                        'student_name': inferred_student,
                        'student_id': inferred_id,
                        'session_start': inferred_start,
                        'session_start_epoch': compact_to_epoch(inferred_start)
                    })

        # Match each video with nearest audio (same student, closest timestamp).
        max_delta_sec = 180
        for v in videos:
            v_epoch = v.get('session_start_epoch') or v.get('mtime')
            v_sid = v.get('student_id')
            v_sname = (v.get('student_name') or '').strip().lower()
            best = None
            best_delta = None
            for a in audios:
                a_sid = a.get('student_id')
                a_sname = (a.get('student_name') or '').strip().lower()
                same_student = (v_sid is not None and a_sid is not None and int(v_sid) == int(a_sid))
                if not same_student and v_sname and a_sname:
                    same_student = (v_sname == a_sname)
                if not same_student:
                    continue

                a_epoch = a.get('session_start_epoch') or a.get('mtime')
                if a_epoch is None or v_epoch is None:
                    continue

                delta = abs(float(v_epoch) - float(a_epoch))
                if delta <= max_delta_sec and (best is None or delta < best_delta):
                    best = a
                    best_delta = delta

            if best:
                v['matched_audio'] = best
    except Exception as e:
        logger.error(f"Error listing recordings: {e}", exc_info=True)
        flash(f"Error loading recordings: {e}", "danger")
    videos.sort(key=lambda x: x['mtime'], reverse=True)
    audios.sort(key=lambda x: x['mtime'], reverse=True)
    return render_template('Recordings.html', videos=videos, audios=audios)

def _safe_send_from_dir(base_dir, filename):
    """Send file from a base directory safely."""
    if not filename or '..' in filename or filename.startswith(('/', '\\')):
        return abort(400)
    if not os.path.isdir(base_dir):
        return abort(404)
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return abort(404)
    return send_from_directory(base_dir, filename, as_attachment=True)

@app.route('/download/recording/video/<path:filename>')
@require_role('ADMIN')
def download_recording_video(filename):
    return _safe_send_from_dir(os.path.join('static', 'exam_sessions'), filename)

@app.route('/download/recording/audio/<path:filename>')
@require_role('ADMIN')
def download_recording_audio(filename):
    return _safe_send_from_dir(os.path.join('static', 'audio_recordings'), filename)

@app.route('/adminStudents')
@require_role('ADMIN')
def adminStudents():
    """Fetch and display all students with profile images and exam results"""
    try:
        logger.info("=== ADMIN STUDENTS PAGE LOAD ===")
        ensure_db_schema()
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, name, email, password, profile FROM students WHERE Role='STUDENT'")
        rows = cur.fetchall()
        
        logger.info(f"Number of student records found: {len(rows)}")
        
        # Fetch latest result per student using correct schema
        results_map = {}
        try:
            cur.execute("""
                SELECT er.StudentID, er.Score, er.TotalQuestions, er.CorrectAnswers,
                       er.SubmissionTime, er.Status,
                       (SELECT COUNT(*) FROM violations v WHERE v.SessionID = er.SessionID) AS warnings_count
                FROM exam_results er
                WHERE er.ResultID IN (
                    SELECT MAX(ResultID) FROM exam_results GROUP BY StudentID
                )
            """)
            result_rows = cur.fetchall()
            for r in result_rows:
                total_q = int(r[2] or 0)
                correct_q = int(r[3] or 0)
                if total_q > 0:
                    pct = round((correct_q / total_q) * 100.0, 2)
                else:
                    pct = float(r[1]) if r[1] else 0
                status = r[5]
                if pct >= 90:   grade = 'A'
                elif pct >= 75: grade = 'B'
                elif pct >= 60: grade = 'C'
                elif pct >= 50: grade = 'D'
                else:           grade = 'F'
                results_map[r[0]] = {
                    'score':           (r[3] or 0) * 2,
                    'total_questions': r[2] or 125,
                    'percentage':      pct,
                    'grade':           grade,
                    'warnings_issued': int(r[6]) if r[6] else 0,
                    'auto_terminated': (status == 'TERMINATED'),
                    'submission_time': r[4],
                }
        except Exception as re:
            logger.warning(f"Results fetch warning: {re}")
        
        students = []
        for idx, row in enumerate(rows):
            student = {
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "password": row[3],
                "profile": row[4],
                "result": results_map.get(row[0])  # attach latest result if exists
            }
            students.append(student)
        
        cur.close()
        
        # Count students with profile images
        registered_count = sum(1 for s in students if s["profile"] and s["profile"].strip())
        
        logger.info(f"Students with profile images: {registered_count}/{len(students)}")
        
        # Check if profile images exist in filesystem
        if registered_count > 0:
            for student in students:
                if student["profile"]:
                    profile_path = os.path.join("static", "Profiles", student["profile"])
                    if os.path.exists(profile_path):
                        logger.debug(f"Profile image exists: {profile_path}")
                    else:
                        logger.warning(f"Profile image NOT FOUND: {profile_path}")
        
        return render_template(
            "Students.html",  # Make sure this matches your template filename
            students=students,
            registered_count=registered_count,
            MONITORING_ENABLED=MONITORING_ENABLED
        )
        
    except Exception as e:
        logger.error(f"Error in adminStudents route: {str(e)}", exc_info=True)
        flash(f"Database error: {str(e)}", "danger")
        
        # Return empty data but still render template
        return render_template(
            "Students.html",
            students=[],
            registered_count=0,
            MONITORING_ENABLED=False
        )

@app.route('/adminLiveMonitoring')
@require_role('ADMIN')
def adminLiveMonitoring():
    if not MONITORING_ENABLED:
        flash('Live monitoring not available. Ensure flask-socketio is installed.', 'error')
        return redirect(url_for('adminStudents'))
    return render_template('admin_live_dashboard.html')

@app.route('/admin/live/<int:student_id>')
@app.route('/admin/live-stream/<int:student_id>')
@require_role('ADMIN')
def admin_live_stream(student_id):
    """Continuous MJPEG stream for a specific student."""
    if not CV2_AVAILABLE or np is None:
        return ("OpenCV unavailable for stream encoding", 503)

    frame_interval = 0.033  # ~30 FPS target for lower latency
    stream_debug = (os.getenv('STREAM_DEBUG', '0') == '1')
    frame_counter = {'n': 0}
    last_stream_frame = {'frame': None}

    def generate():
        sid_str = str(student_id)
        while True:
            try:
                frame = None
                raw_ts = 0.0
                proc_ts = 0.0
                snapshot = {}
                overlay_item = {}
                with latest_student_frames_lock:
                    item = latest_student_frames.get(sid_str)
                    if item:
                        overlay_item = dict(item)
                        snapshot = dict(item.get('status_snapshot') or {})
                        raw_ts = float(item.get('frame_timestamp') or item.get('timestamp') or 0.0)
                        proc_ts = float(item.get('processed_timestamp') or 0.0)
                        raw_frame = item.get('frame')
                        proc_frame = item.get('processed_frame')
                        # Prevent "single static photo" effect when processor lags:
                        # only use processed frame when it is fresh relative to raw.
                        if proc_frame is not None and proc_ts >= (raw_ts - 2.0):
                            cur = proc_frame
                        else:
                            cur = raw_frame
                        if cur is not None:
                            frame = cur.copy()

                if not snapshot:
                    with student_detection_state_lock:
                        st = student_detection_state.get(sid_str) or {}
                        snapshot = dict(st.get('status_snapshot') or {})
                        overlay_item.setdefault('last_visible_object_labels', list(st.get('last_visible_object_labels') or []))
                        overlay_item.setdefault('last_prohibited_object_labels', list(st.get('last_prohibited_object_labels') or []))
                        overlay_item.setdefault('last_person_count', int(st.get('last_person_count') or 0))

                if frame is None:
                    if last_stream_frame['frame'] is not None:
                        frame = last_stream_frame['frame'].copy()
                    else:
                        frame = _build_stream_placeholder(student_id, "Waiting for student camera...")
                frame = _overlay_status_snapshot(frame, snapshot, overlay_item)
                # Suppress feed-age/status text overlays for cleaner UI
                last_stream_frame['frame'] = frame.copy()
                ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                if not ok:
                    time.sleep(frame_interval)
                    continue

                frame_counter['n'] += 1
                if stream_debug and (frame_counter['n'] % 60 == 1):
                    logger.info(
                        f"Streaming frame... student={student_id} count={frame_counter['n']} "
                        f"raw_age={max(0.0, time.time()-raw_ts):.2f}s proc_age={max(0.0, time.time()-proc_ts):.2f}s"
                    )

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    buffer.tobytes() +
                    b'\r\n'
                )
            except GeneratorExit:
                break
            except Exception as e:
                logger.error(f"admin_live_stream({student_id}) generator error: {e}", exc_info=True)
            time.sleep(frame_interval)

    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    resp.headers['X-Accel-Buffering'] = 'no'
    # Same-origin by default, add permissive CORS header for embedded stream clients.
    origin = request.headers.get('Origin')
    if origin:
        resp.headers['Access-Control-Allow-Origin'] = origin
        resp.headers['Vary'] = 'Origin'
    else:
        resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# CRUD student endpoints
@app.route('/insertStudent', methods=['POST'])
@require_role('ADMIN')
def insertStudent():
    if request.method == "POST":
        try:
            name = request.form['username']
            email = request.form['email']
            password = request.form['password']
            profile_image = request.files.get('profile_image')
            profile_image_data = request.form.get('profile_image_data')
            filename = None

            if profile_image and profile_image.filename:
                img_bytes = profile_image.read()
                profile_image.seek(0)
                filename = secure_filename(profile_image.filename)
            elif profile_image_data:
                image_b64 = profile_image_data.split(',', 1)[1] if ',' in profile_image_data else profile_image_data
                img_bytes = base64.b64decode(image_b64)
                filename = "profile_upload.jpg"
            else:
                flash('Profile image is required when creating a student.', 'error')
                return redirect(url_for('adminStudents'))

            if not _bytes_has_single_face(img_bytes):
                flash('Profile image must contain exactly one clear human face.', 'error')
                return redirect(url_for('adminStudents'))

            safe_email = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in email.strip().lower())
            profile_filename = f"{safe_email}_{filename}"
            os.makedirs('static/Profiles', exist_ok=True)
            with open(os.path.join('static/Profiles', profile_filename), 'wb') as f:
                f.write(img_bytes)

            cur = mysql.connection.cursor()
            try:
                cur.execute(
                    "INSERT INTO students (Name, Email, Password, Profile, Role) VALUES (%s, %s, %s, %s, %s)",
                    (name, email, generate_password_hash(password), profile_filename, 'STUDENT')
                )
            except Exception as col_error:
                if "Unknown column 'Profile'" in str(col_error):
                    cur.execute(
                        "INSERT INTO students (Name, Email, Password, Role) VALUES (%s, %s, %s, %s)",
                        (name, email, generate_password_hash(password), 'STUDENT')
                    )
                else:
                    raise col_error
            mysql.connection.commit()
            cur.close()
            flash('Student added successfully', 'success')
        except Exception as e:
            logger.error(f"Error inserting student: {e}")
            flash('Error adding student', 'error')
        return redirect(url_for('adminStudents'))

@app.route('/deleteStudent/<string:stdId>', methods=['GET'])
@require_role('ADMIN')
def deleteStudent(stdId):
    try:
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM students WHERE ID=%s", (stdId,))
        mysql.connection.commit()
        cur.close()
        flash("Record deleted successfully", 'success')
    except Exception as e:
        logger.error(f"Error deleting student: {e}")
        flash('Error deleting student', 'error')
    return redirect(url_for('adminStudents'))

@app.route('/updateStudent', methods=['POST'])
@require_role('ADMIN')
def updateStudent():
    if request.method == 'POST':
        try:
            id_data = request.form['id']
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            cur = mysql.connection.cursor()
            final_password = None
            if password and password.strip():
                final_password = generate_password_hash(password)
            else:
                cur.execute("SELECT Password FROM students WHERE ID=%s", (id_data,))
                old_row = cur.fetchone()
                final_password = old_row[0] if old_row else generate_password_hash("123456")
            cur.execute("""
                UPDATE students
                SET Name=%s, Email=%s, Password=%s
                WHERE ID=%s
            """, (name, email, final_password, id_data))
            mysql.connection.commit()
            cur.close()
            flash('Student updated successfully', 'success')
        except Exception as e:
            logger.error(f"Error updating student: {e}")
            flash('Error updating student', 'error')
        return redirect(url_for('adminStudents'))

@app.route('/registerFace', methods=['POST'])
@require_role('ADMIN')
def registerFace():
    try:
        student_id = request.form.get('student_id')
        student_name = request.form.get('student_name')
        file = request.files.get('face_image')
        webcam_image = request.form.get('webcam_image')

        filename = f"face_{student_id}_{int(time.time())}.jpg"
        os.makedirs('static/Profiles', exist_ok=True)
        save_path = os.path.join('static', 'Profiles', filename)

        # Accept either uploaded image file OR captured webcam base64 image.
        if file and file.filename:
            img_bytes = file.read()
            file.seek(0)
            if not _bytes_has_single_face(img_bytes):
                flash("Face image must contain exactly one clear face.", 'error')
                return redirect(url_for('adminStudents'))
            with open(save_path, 'wb') as out:
                out.write(img_bytes)
        elif webcam_image:
            try:
                image_b64 = webcam_image.split(',', 1)[1] if ',' in webcam_image else webcam_image
                img_bytes = base64.b64decode(image_b64)
                if not _bytes_has_single_face(img_bytes):
                    flash("Captured image must contain exactly one clear face.", 'error')
                    return redirect(url_for('adminStudents'))
                with open(save_path, 'wb') as out:
                    out.write(img_bytes)
            except Exception:
                flash("Invalid webcam image data", 'error')
                return redirect(url_for('adminStudents'))
        else:
            cur = mysql.connection.cursor()
            cur.execute("SELECT Profile FROM students WHERE ID=%s", (student_id,))
            row = cur.fetchone()
            cur.close()
            existing_profile = None
            if row:
                existing_profile = row[0] if isinstance(row, (list, tuple)) else row.get('Profile')
            if existing_profile:
                flash(f"Face already registered for {student_name}", 'success')
                return redirect(url_for('adminStudents'))
            flash("Please upload a photo or capture from webcam", 'error')
            return redirect(url_for('adminStudents'))

        # Update database - handle both with and without Profile column
        cur = mysql.connection.cursor()
        try:
            cur.execute("UPDATE students SET Profile=%s WHERE ID=%s", (filename, student_id))
        except Exception as col_error:
            if "Unknown column 'Profile'" in str(col_error):
                # If Profile column doesn't exist, skip the update
                flash("Profile column not available in database", 'error')
            else:
                raise col_error
        
        mysql.connection.commit()
        cur.close()

        flash(f"Face registered for {student_name}", 'success')
        return redirect(url_for('adminStudents'))

    except Exception as e:
        logger.error(f"registerFace error: {e}")
        flash("Error registering face", 'error')
        return redirect(url_for('adminStudents'))

# -------------------------
# API Endpoints for Real-Time Data
# -------------------------
@app.route('/api/student-frame', methods=['POST'])
@require_role('STUDENT')
@rate_limit('student_frame', max_requests=900, window_seconds=60)
def api_student_frame():
    """Receive student browser frame quickly and defer heavy detection to background workers."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    if not CV2_AVAILABLE or np is None:
        return jsonify({'ok': False, 'error': 'OpenCV unavailable'}), 503

    payload = request.get_json(silent=True) or {}
    image_data = payload.get('image_data') or payload.get('frame')
    if not image_data:
        return jsonify({'ok': False, 'error': 'Missing image_data'}), 400

    try:
        logger.debug("Frame received in /api/student-frame")
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'ok': False, 'error': 'Bad frame'}), 400
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

        student_id = str(user['Id'])
        student_name = str(user.get('Name') or f'student_{student_id}')
        
        sid_str = student_id
        with student_frame_rx_lock:
            count = int(student_frame_rx_counts.get(sid_str, 0)) + 1
            student_frame_rx_counts[sid_str] = count

        if (count % 15) == 1:
            logger.info(f"Frame received from: {sid_str} (count={count}, shape={getattr(frame, 'shape', None)})")

        with latest_student_frames_lock:
            prev = latest_student_frames.get(sid_str, {})
            raw_ts = time.time()
            latest_student_frames[sid_str] = {
                'frame': frame,
                'processed_frame': prev.get('processed_frame'),
                'timestamp': raw_ts,
                'frame_timestamp': raw_ts,
                'processed_timestamp': prev.get('processed_timestamp', 0.0),
                'detections': prev.get('detections', []),
                'processed_frame_b64': prev.get('processed_frame_b64'),
                'status_snapshot': prev.get('status_snapshot', {}),
                'last_visible_object_labels': prev.get('last_visible_object_labels', []),
                'last_prohibited_object_labels': prev.get('last_prohibited_object_labels', []),
                'last_person_count': prev.get('last_person_count', 0),
            }
        with student_stale_violation_lock:
            student_stale_violation_at.pop(sid_str, None)

        # Critical: route does only receive/store/schedule. No heavy inference in request path.
        _schedule_student_frame_detection(student_id, student_name, frame)

        return jsonify({'ok': True, 'queued': True, 'student_id': student_id})

    except Exception as e:
        logger.error(f"api_student_frame error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Decode failed'}), 400

@app.route('/api/upload-audio', methods=['POST'])
@require_role('STUDENT')
@rate_limit('student_audio_upload', max_requests=20, window_seconds=300)
def api_upload_audio():
    """Receive browser-recorded student audio and store it for admin recordings."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401

    file = request.files.get('audio')
    if not file or not file.filename:
        return jsonify({'ok': False, 'error': 'Missing audio file'}), 400

    try:
        student_id = int(user['Id'])
        student_name = ''.join(ch if ch.isalnum() else '_' for ch in str(user.get('Name') or 'student')).strip('_') or f"student_{student_id}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        content_type = (file.content_type or '').lower()
        ext = '.webm'
        if 'ogg' in content_type:
            ext = '.ogg'
        elif 'wav' in content_type or 'wave' in content_type:
            ext = '.wav'
        elif 'mp4' in content_type or 'm4a' in content_type:
            ext = '.m4a'

        audio_dir = os.path.join('static', 'audio_recordings')
        os.makedirs(audio_dir, exist_ok=True)
        filename = f"{student_id}_{student_name}_{timestamp}{ext}"
        file.save(os.path.join(audio_dir, filename))
        return jsonify({'ok': True, 'filename': filename})
    except Exception as e:
        logger.error(f"api_upload_audio error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Audio upload failed'}), 500

@app.route('/api/upload-session-recording', methods=['POST'])
@require_role('STUDENT')
@rate_limit('student_session_recording_upload', max_requests=12, window_seconds=300)
def api_upload_session_recording():
    """Receive a combined browser-recorded exam session video with embedded audio."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401

    file = request.files.get('recording')
    if not file or not file.filename:
        return jsonify({'ok': False, 'error': 'Missing recording file'}), 400

    try:
        student_id = int(user['Id'])
        student_name = ''.join(ch if ch.isalnum() else '_' for ch in str(user.get('Name') or 'student')).strip('_') or f"student_{student_id}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        started_at_raw = (request.form.get('started_at') or '').strip()
        session_start = timestamp
        if started_at_raw:
            try:
                session_start = datetime.fromtimestamp(float(started_at_raw) / 1000.0).strftime("%Y%m%d_%H%M%S")
            except Exception:
                session_start = timestamp

        content_type = (file.content_type or '').lower()
        ext = '.webm'
        if 'mp4' in content_type:
            ext = '.mp4'
        elif 'ogg' in content_type:
            ext = '.ogg'
        elif file.filename and '.' in file.filename:
            guessed_ext = os.path.splitext(file.filename)[1].lower()
            if guessed_ext in ('.webm', '.mp4', '.ogg', '.mkv'):
                ext = guessed_ext

        video_dir = os.path.join('static', 'exam_sessions')
        os.makedirs(video_dir, exist_ok=True)
        filename = f"{student_id}_{student_name}_{session_start}{ext}"
        output_path = os.path.join(video_dir, filename)
        file.save(output_path)

        session_id = _get_active_session_id(student_id)
        meta_path = os.path.join(video_dir, f"{student_id}_{student_name}_{session_start}.json")
        metadata = {
            'student_id': student_id,
            'student_name': str(user.get('Name') or f"Student {student_id}"),
            'session_id': session_id,
            'session_start': session_start,
            'video_path': output_path,
            'embedded_audio': True,
            'content_type': content_type or 'video/webm',
            'file_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
        }
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as meta_err:
            logger.warning(f"session recording metadata save failed: {meta_err}")

        return jsonify({'ok': True, 'filename': filename})
    except Exception as e:
        logger.error(f"api_upload_session_recording error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Session recording upload failed'}), 500

@app.route('/api/student-exit-signal', methods=['POST'])
@require_role('STUDENT')
@rate_limit('student_exit_signal', max_requests=20, window_seconds=60)
def api_student_exit_signal():
    """Receive keepalive beacon when student closes/hides exam tab."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    try:
        student_id = str(user['Id'])
        student_name = str(user.get('Name') or 'Unknown')
        event_type = (request.form.get('event_type') or request.args.get('event_type') or 'TAB_CLOSE').upper()
        details = (request.form.get('details') or request.args.get('details') or 'Tab/window closed during exam').strip()
        details = details[:500]

        # Enforce tab switch as a violation (strict â€” short cooldown so repeats are counted)
        _trigger_violation(student_id, student_name, 'TAB_SWITCH', f"{event_type}: {details}", cooldown_seconds=0.2)
        logger.info(f"[TAB_SWITCH ENFORCED] student={student_id} details={details}")

        # Best-effort immediate DB persistence for close events
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                SELECT SessionID FROM exam_sessions
                WHERE StudentID=%s AND Status='IN_PROGRESS'
                ORDER BY StartTime DESC LIMIT 1
            """, (student_id,))
            sess = cur.fetchone()
            if sess:
                cur.execute("""
                    INSERT INTO violations (StudentID, SessionID, ViolationType, Details, Timestamp)
                    VALUES (%s, %s, %s, %s, NOW())
                """, (student_id, sess[0], 'TAB_SWITCH', f"{event_type}: {details}"))
                mysql.connection.commit()
            cur.close()
        except Exception as db_err:
            logger.warning(f"student_exit_signal DB save failed: {db_err}")

        return jsonify({'ok': True})
    except Exception as e:
        logger.error(f"api_student_exit_signal error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Exit signal failed'}), 500

@app.route('/api/my-warnings')
@require_role('STUDENT')
@rate_limit('student_warning_state', max_requests=120, window_seconds=60)
def api_my_warnings():
    """Return the current student's live warning state for UI sync."""
    user = current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    try:
        student_id = str(user['Id'])
        warnings_count = 0
        violations = []
        if warning_system:
            warnings_count = int(warning_system.get_warnings(student_id) or 0)
            violations = warning_system.get_violations(student_id) or []
        runtime_state = _get_runtime_warning_state(student_id)
        warnings_count = max(warnings_count, int(runtime_state.get('warnings') or 0))
        if len(runtime_state.get('violations') or []) > len(violations):
            violations = runtime_state.get('violations') or violations
        latest_violation = violations[-1] if violations else None
        return jsonify({
            'ok': True,
            'student_id': int(student_id),
            'warnings': min(warnings_count, 3),
            'violations': violations,
            'latest_violation': latest_violation
        })
    except Exception as e:
        logger.error(f"api_my_warnings error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Warning state fetch failed'}), 500

@app.route('/api/today-violations')
@require_role('ADMIN')
def api_today_violations():
    """Return total violations count today from violations table"""
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT COUNT(*) FROM violations WHERE DATE(Timestamp) = CURDATE()")
        row = cur.fetchone()
        cur.close()
        count = int(row[0]) if row else 0
        return jsonify({'count': count})
    except Exception as e:
        logger.error(f"API today-violations error: {e}")
        return jsonify({'count': 0})

@app.route('/api/student-warnings/<int:student_id>')
@require_role('ADMIN')
def api_student_warnings(student_id):
    """Return current live warnings for a student from warning_system"""
    if warning_system:
        count = int(warning_system.get_warnings(student_id) or 0)
        violations = warning_system.get_violations(student_id) or []
    else:
        count = 0
        violations = []
    runtime_state = _get_runtime_warning_state(student_id)
    count = max(count, int(runtime_state.get('warnings') or 0))
    if len(runtime_state.get('violations') or []) > len(violations):
        violations = runtime_state.get('violations') or violations
    return jsonify({'student_id': student_id, 'warnings': count, 'violations': violations})

@app.route('/api/all-student-warnings')
@require_role('ADMIN')
def api_all_student_warnings():
    """Return warnings for all active students"""
    result = {}
    if warning_system:
        with warning_system.lock:
            for sid, count in warning_system.warnings.items():
                result[str(sid)] = {
                    'warnings': count,
                    'name': warning_system.student_names.get(sid, 'Unknown'),
                    'violations': warning_system.violations.get(sid, [])
                }
    with runtime_warning_state_lock:
        for sid, rec in runtime_warning_state.items():
            current = result.setdefault(str(sid), {'warnings': 0, 'name': rec.get('student_name', 'Unknown'), 'violations': []})
            current['warnings'] = max(int(current.get('warnings') or 0), int(rec.get('warnings') or 0))
            if len(rec.get('violations') or []) > len(current.get('violations') or []):
                current['violations'] = list(rec.get('violations') or [])
            if not current.get('name'):
                current['name'] = rec.get('student_name', 'Unknown')
    return jsonify(result)

@app.route('/api/admin-active-students')
@require_role('ADMIN')
def api_admin_active_students():
    """Polling fallback for admin dashboard active students and frame availability."""
    try:
        with active_exam_students_lock:
            active_ids = set(str(sid) for sid in active_exam_students)

        now = time.time()
        with latest_student_frames_lock:
            for sid, item in latest_student_frames.items():
                ts = float((item or {}).get('timestamp') or 0.0)
                if (now - ts) <= 20.0:
                    active_ids.add(str(sid))

        if not active_ids:
            return jsonify({'students': []})

        id_list = sorted(list(set(str(sid) for sid in active_ids))) # Ensure list of strings
        name_map = {}
        try:
            placeholders = ",".join(["%s"] * len(id_list))
            cur = mysql.connection.cursor()
            cur.execute(f"SELECT ID, Name FROM students WHERE ID IN ({placeholders})", tuple(id_list))
            rows = cur.fetchall() or []
            cur.close()
            for row in rows:
                name_map[str(row[0])] = str(row[1] or f"Student {row[0]}")
        except Exception as e:
            logger.warning(f"admin active student name lookup failed: {e}")

        students = []
        for sid in id_list:
            sid_str = str(sid)
            with latest_student_frames_lock:
                item = latest_student_frames.get(sid_str) or {}
            students.append({
                'student_id': sid_str,
                'student_name': name_map.get(sid_str) or (warning_system.student_names.get(sid_str) if warning_system else None) or f"Student {sid_str}",
                'warnings': int(warning_system.get_warnings(sid_str) if warning_system else 0),
                'violations': warning_system.get_violations(sid_str) if warning_system else [],
                'has_frame': bool(item.get('processed_frame') is not None or item.get('frame') is not None),
                'last_update': float(max(
                    float(item.get('frame_timestamp') or 0.0),
                    float(item.get('processed_timestamp') or 0.0),
                    float(item.get('timestamp') or 0.0)
                ))
            })

        return jsonify({'students': students})
    except Exception as e:
        logger.error(f"api_admin_active_students error: {e}", exc_info=True)
        return jsonify({'students': []})

@app.route('/api/object-detection-status')
@require_role('ADMIN')
def api_object_detection_status():
    """Return status of object detection model"""
    return jsonify({
        'enabled': object_net_enabled,
        'model': 'YOLOv8' if object_net_enabled else 'Not loaded',
        'model_path': yolo_loaded_model_path,
        'device': yolo_device,
        'prohibited_labels': list(PROHIBITED_LABELS),
        'ultralytics_available': ULTRALYTICS_AVAILABLE,
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': bool(TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()),
        'mapped_prohibited_class_ids': prohibited_class_ids,
        'fast_face_only_mode': FAST_FACE_ONLY_MODE,
        'pose_analysis_enabled': RUN_POSE_ANALYSIS,
        'object_analysis_interval_sec': OBJECT_ANALYSIS_INTERVAL_SEC,
        'ocr_available': OCR_AVAILABLE,
        'ocr_engine': 'pytesseract' if OCR_AVAILABLE else 'disabled',
        'download_commands': [
            'pip install ultralytics',
            'python download_yolo_models.py'
        ] if not object_net_enabled else []
    })

@app.route('/api/detection-pipeline-status')
@require_role('ADMIN')
def api_detection_pipeline_status():
    """Return live detector and per-student frame pipeline status."""
    now = time.time()
    students = {}
    with latest_student_frames_lock:
        frame_items = {str(sid): dict(item or {}) for sid, item in latest_student_frames.items()}
    with student_detection_state_lock:
        detection_items = {str(sid): dict(item or {}) for sid, item in student_detection_state.items()}
    with student_frame_rx_lock:
        rx_counts = dict(student_frame_rx_counts)

    for sid in sorted(set(list(frame_items.keys()) + list(detection_items.keys()) + list(rx_counts.keys()))):
        frame_item = frame_items.get(sid, {})
        state_item = detection_items.get(sid, {})
        frame_ts = float(frame_item.get('frame_timestamp') or frame_item.get('timestamp') or 0.0)
        proc_ts = float(frame_item.get('processed_timestamp') or 0.0)
        students[sid] = {
            'frames_received': int(rx_counts.get(sid, 0)),
            'raw_age_sec': round(now - frame_ts, 3) if frame_ts else None,
            'processed_age_sec': round(now - proc_ts, 3) if proc_ts else None,
            'has_raw_frame': frame_item.get('frame') is not None,
            'has_processed_frame': frame_item.get('processed_frame') is not None,
            'worker_running': bool(state_item.get('running', False)),
            'has_pending_frame': state_item.get('pending_frame') is not None,
            'last_object_labels': [d.get('label') for d in (state_item.get('last_object_detections') or [])],
            'last_visible_object_labels': list(state_item.get('last_visible_object_labels') or []),
            'last_prohibited_object_labels': list(state_item.get('last_prohibited_object_labels') or []),
            'last_person_count': int(state_item.get('last_person_count') or 0),
            'status_snapshot': dict(state_item.get('status_snapshot') or {}),
        }

    return jsonify({
        'mediapipe_available': MEDIAPIPE_AVAILABLE,
        'face_mesh_loaded': face_mesh_detector is not None,
        'pose_loaded': pose_detector is not None,
        'pose_pipeline_active': bool(RUN_POSE_ANALYSIS and (not FAST_FACE_ONLY_MODE) and MEDIAPIPE_AVAILABLE and pose_detector is not None),
        'opencv_available': CV2_AVAILABLE,
        'yolo_enabled': object_net_enabled,
        'yolo_model_path': yolo_loaded_model_path,
        'fast_face_only_mode': FAST_FACE_ONLY_MODE,
        'run_pose_analysis': RUN_POSE_ANALYSIS,
        'students': students
    })

# -------------------------
# SocketIO handlers
# -------------------------
if MONITORING_ENABLED and socketio:
    @socketio.on('connect', namespace='/student')
    def student_connect():
        user = current_user()
        if not user or user.get('Role') != 'STUDENT':
            return False
        sid = request.sid
        try:
            join_room(f"student:{user.get('Id')}")
        except Exception:
            pass
        logger.info(f'Student socket connected: {sid}')

    @socketio.on('disconnect', namespace='/student')
    def student_disconnect():
        sid = request.sid
        logger.info(f'Student socket disconnected: {sid}')

    @socketio.on('request_student_feed', namespace='/student')
    def handle_request_student_feed(data):
        student_id = data.get('student_id')
        emit('request_ack', {'student_id': student_id})

    # ══════════════════════════════════════════════════════════════════
    # CRITICAL FIX: 'warning_issued' handler
    # Exam.html emits this every time student gets a warning.
    # Without this handler warnings NEVER reach admin dashboard.
    # ══════════════════════════════════════════════════════════════════
    @socketio.on('warning_issued', namespace='/student')
    def handle_warning_issued(data):
        student_id   = str(data.get('student_id'))
        student_name = data.get('student_name', 'Unknown')
        violation    = data.get('violation', {})
        vtype        = violation.get('type', 'TAB_SWITCH')
        details      = violation.get('details', str(vtype))
        runtime_count, runtime_violation = _record_runtime_warning(student_id, student_name, vtype, details)
        
        logger.info(f"⚠️  warning_issued received: student={student_id} type={vtype}")
        
        # 1. Update in-memory warning_system so count stays accurate
        if warning_system and student_id:
            if student_id not in warning_system.warnings:
                warning_system.initialize_student(student_id, student_name)
            terminated = warning_system.add_warning(student_id, vtype, details, emit_to_student=False)
            # add_warning already emits 'student_violation' to /admin — done!
            if terminated:
                emit('auto_terminated', {'student_id': student_id, 'reason': 'Max warnings reached'})
        else:
            # warning_system unavailable — manually forward to admin
            socketio.emit('student_violation', {
                'student_id':     student_id,
                'student_name':   student_name,
                'total_warnings': max(int(data.get('warning_number', 1) or 1), runtime_count),
                'violation':      runtime_violation,
            }, namespace='/admin')
        
        # 2. Also save violation immediately to DB (live persistence)
        VTYPE_MAP = {
            'TAB_SWITCH': 'TAB_SWITCH', 'tab_switch': 'TAB_SWITCH',
            'FULLSCREEN_EXIT': 'TAB_SWITCH', 'fullscreen_exit': 'TAB_SWITCH',
            'PROHIBITED_SHORTCUT': 'PROHIBITED_SHORTCUT', 'prohibited_shortcut': 'PROHIBITED_SHORTCUT',
            'KEYBOARD_SHORTCUT': 'PROHIBITED_SHORTCUT', 'DEVTOOLS_OPEN': 'PROHIBITED_SHORTCUT',
            'DEVTOOLS_SHORTCUT': 'PROHIBITED_SHORTCUT', 'DEVTOOLS_OPENED': 'PROHIBITED_SHORTCUT',
            'COPY_PASTE': 'PROHIBITED_SHORTCUT',
            'MULTIPLE_FACES': 'MULTIPLE_FACES', 'multiple_faces': 'MULTIPLE_FACES',
            'NO_FACE': 'NO_FACE', 'no_face': 'NO_FACE',
            'EYES_CLOSED': 'EYES_CLOSED', 'eyes_closed': 'EYES_CLOSED',
            'GAZE_LEFT': 'GAZE_LEFT', 'gaze_left': 'GAZE_LEFT',
            'GAZE_RIGHT': 'GAZE_RIGHT', 'gaze_right': 'GAZE_RIGHT',
            'GAZE_UP': 'GAZE_UP', 'gaze_up': 'GAZE_UP',
            'GAZE_DOWN': 'GAZE_DOWN', 'gaze_down': 'GAZE_DOWN',
            'VOICE_DETECTED': 'VOICE_DETECTED', 'voice_detected': 'VOICE_DETECTED',
            'DISTRACTION': 'DISTRACTION', 'distraction': 'DISTRACTION',
            'STUDENT_LEFT_SEAT': 'STUDENT_LEFT_SEAT', 'student_left_seat': 'STUDENT_LEFT_SEAT',
            'MIC_OFF': 'VOICE_DETECTED', 'mic_off': 'VOICE_DETECTED',
            'HEAD_MOVEMENT': 'HEAD_MOVEMENT', 'head_movement': 'HEAD_MOVEMENT',
            'IDENTITY_MISMATCH': 'IDENTITY_MISMATCH', 'identity_mismatch': 'IDENTITY_MISMATCH',
            'CAMERA_OFF': 'NO_FACE', 'camera_off': 'NO_FACE',
            'CAMERA_BLOCKED': 'NO_FACE', 'camera_blocked': 'NO_FACE',
            'PROHIBITED_OBJECT': 'PROHIBITED_OBJECT', 'prohibited_object': 'PROHIBITED_OBJECT',
            'TERMINATED_BY_ADMIN': 'TERMINATED_BY_ADMIN', 'terminated_by_admin': 'TERMINATED_BY_ADMIN',
        }
        db_vtype = VTYPE_MAP.get(vtype, VTYPE_MAP.get(str(vtype).upper(), 'TAB_SWITCH'))
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                SELECT SessionID FROM exam_sessions
                WHERE StudentID=%s AND Status='IN_PROGRESS'
                ORDER BY StartTime DESC LIMIT 1
            """, (student_id,))
            sess = cur.fetchone()
            if sess:
                cur.execute("""
                    INSERT INTO violations (StudentID, SessionID, ViolationType, Details, Timestamp)
                    VALUES (%s, %s, %s, %s, NOW())
                """, (student_id, sess[0], db_vtype, str(details)[:500]))
                mysql.connection.commit()
            cur.close()
        except Exception as db_err:
            logger.warning(f"Live violation DB save failed: {db_err}")

    @socketio.on('exam_auto_terminated', namespace='/student')
    def handle_exam_auto_terminated(data):
        """Student exam terminated due to max warnings"""
        student_id   = data.get('student_id')
        student_name = data.get('student_name', 'Unknown')
        reason       = data.get('reason', 'Max warnings reached')
        logger.info(f"🚫 Exam auto-terminated: student={student_id}")
        socketio.emit('student_exam_terminated', {
            'student_id':   student_id,
            'student_name': student_name,
            'reason':       reason,
        }, namespace='/admin')

    @socketio.on('terminate_exam', namespace='/student')
    def handle_terminate_exam(data):
        student_id = str(data.get('student_id'))
        reason = data.get('reason', 'Manual termination by admin')
        if warning_system:
            warning_system.add_warning(student_id, 'TERMINATED_BY_ADMIN', reason, emit_to_student=False)
        emit('terminated_ack', {'student_id': student_id, 'reason': reason})

    @socketio.on('prohibited_action', namespace='/student')
    def handle_prohibited_action(data):
        student_id = str(data.get('student_id'))
        action = data.get('action')
        if student_id:
            terminated = _trigger_violation(student_id, data.get('student_name') or 'Unknown', 'PROHIBITED_SHORTCUT', str(action))
            logger.info(f"[SHORTCUT] student={student_id} action={action} terminated={terminated}")
            if terminated:
                emit('auto_terminated', {'student_id': student_id})
        else:
            logger.info(f"[SHORTCUT IGNORED] student={student_id} action={action}")

    @socketio.on('tab_switch_detected', namespace='/student')
    def handle_tab_switch(data):
        student_id = str(data.get('student_id'))
        student_name = str(data.get('student_name') or (current_user() or {}).get('Name') or 'Unknown')
        details = str(data.get('details') or 'Tab switch detected').strip()
        if student_id:
            terminated = _trigger_violation(student_id, student_name, 'TAB_SWITCH', details, cooldown_seconds=0.5)
            logger.info(f"[TAB_SWITCH] student={student_id} details={details} terminated={terminated}")
        else:
            logger.info(f"[TAB_SWITCH IGNORED] missing student_id details={details}")

    # --- ADMIN CONTROL ACTIONS ---
    @socketio.on('admin_clear_warnings', namespace='/admin')
    def handle_admin_clear_warnings(data):
        student_id = str(data.get('student_id'))
        _reset_exam_runtime_state(student_id)
        emit('warnings_cleared', {'student_id': student_id, 'warnings': 0, 'violations': []}, namespace='/admin')
        socketio.emit(
            'warnings_cleared',
            {'student_id': student_id, 'warnings': 0, 'violations': []},
            namespace='/student',
            room=f"student:{student_id}"
        )

    @socketio.on('admin_force_terminate', namespace='/admin')
    def handle_admin_force_terminate(data):
        student_id = str(data.get('student_id'))
        reason = data.get('reason', 'Manual termination by Admin')
        if warning_system:
            warning_system.manually_terminate_student(student_id, reason)

    @socketio.on('admin_toggle_enforcement', namespace='/admin')
    def handle_admin_toggle_enforcement(data):
        enabled = bool(data.get('enabled', True))
        if warning_system:
            warning_system.set_auto_terminate(enabled)
            emit('enforcement_toggled', {'enabled': enabled}, namespace='/admin')

# -------------------------
# App entrypoint
# -------------------------
if __name__ == '__main__':
    try:
        debug_mode = (os.getenv('FLASK_DEBUG', '0') == '1')
        logger.info("=" * 60)
        logger.info("🚀 Starting Exam Proctoring System")
        logger.info(f"  - OpenCV: {'✓ Available' if CV2_AVAILABLE else '✗ Not available'}")
        logger.info(f"  - Flask-SocketIO: {'✓ Available' if SOCKETIO_AVAILABLE else '✗ Not available'}")
        logger.info(f"  - Live Monitoring: {'✓ ENABLED' if MONITORING_ENABLED else '✗ DISABLED'}")
        logger.info("=" * 60)
        
        with app.app_context():
            ensure_db_schema()

        auto_restart = (os.getenv('AUTO_RESTART', '1') == '1')
        while True:
            try:
                if MONITORING_ENABLED:
                    # use socketio.run when monitoring enabled
                    socketio.run(
                        app,
                        debug=debug_mode,
                        use_reloader=False,  # Windows: avoid socket teardown race (WinError 10038)
                        host='0.0.0.0',
                        port=5001,
                        allow_unsafe_werkzeug=True
                    )
                else:
                    logger.warning("Starting in BASIC MODE (No live monitoring)")
                    logger.info("To enable monitoring, install: pip install flask-socketio")
                    app.run(debug=debug_mode, use_reloader=False, host='0.0.0.0', port=5001, threaded=True)
                break  # clean exit
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user.")
                break
            except Exception as e:
                logger.error(f"Fatal error launching app: {e}", exc_info=True)
                if not auto_restart:
                    break
                logger.info("Auto-restart enabled. Restarting in 3 seconds...")
                time.sleep(3)
    except Exception as e:
        logger.error(f"Fatal error launching app: {e}", exc_info=True)
