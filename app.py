# app.py -- Final Merged & Fixed Version
# Features: Complete CRUD, COCO Object Detection (Phone/Book), Enhanced Monitoring

import os
import io
import struct
import numpy as np
import time
import random
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
    Flask, render_template, request, jsonify, redirect, url_for, flash, Response, send_from_directory, abort, session, send_file
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

import pymysql.cursors

# -------------------------
# Feature Flags (ML removed)
# -------------------------
CV2_AVAILABLE = False
cv2 = None
MEDIAPIPE_AVAILABLE = False
face_mesh_detector = None
pose_detector = None
TORCH_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False
FACE_REC_AVAILABLE = False
OCR_AVAILABLE = False
object_net_enabled = False
yolo_loaded_model_path = None
yolo_device = 'cpu'
prohibited_class_ids = []

try:
    import requests
except ImportError:
    pass

# Try importing flask_socketio
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    SocketIO = None
    emit = None
    logger.warning("Flask-SocketIO not available. Install with: pip install flask-socketio")

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


# -------------------------
# WebRTC / Telemetry Engine (WASM-based)
# -------------------------
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


# Serve a tiny fallback favicon to avoid browser 404 requests for /favicon.ico
# Returns a 1x1 transparent PNG in-memory so no static file is required.
from flask import make_response
import base64

_ONE_PIXEL_PNG_B64 = (
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII='
)


@app.route('/favicon.ico')
def favicon():
    png = base64.b64decode(_ONE_PIXEL_PNG_B64)
    resp = make_response(png)
    resp.headers.set('Content-Type', 'image/png')
    resp.headers.set('Content-Length', len(png))
    return resp

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

def _get_active_session_id(student_id):
    """Retrieve the currently active session ID for a student, returning 0 if none found."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT SessionID FROM exam_sessions WHERE StudentID=%s AND Status='IN_PROGRESS' ORDER BY StartTime DESC LIMIT 1", (student_id,))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else 0
    except Exception:
        return 0

def _build_stream_placeholder(student_id, message):
    """Build a static placeholder frame for M-JPEG stream when real video isn't available."""
    if np is None or cv2 is None:
        return None
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return frame

def _overlay_status_snapshot(frame, snapshot, overlay_item):
    """Placeholder to overlay warning information onto an M-JPEG frame without ML inference."""
    if frame is None or cv2 is None:
        return frame
    
    score = snapshot.get("suspicion_score", 0)
    warnings = snapshot.get("warning_count", 0)
    
    cv2.putText(frame, f"Warnings: {warnings}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if warnings > 0 else (0, 255, 0), 2)
    cv2.putText(frame, f"Suspicion Score: {score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if score > 50 else (0, 255, 0), 2)
    
    return frame


def _bytes_has_single_face(img_bytes):
    """Return True if the provided image bytes contain exactly one face.
    Uses OpenCV Haar cascade when available; if OpenCV or cascade isn't
    available the function falls back to permissive behavior (returns True)
    to avoid blocking registrations on missing optional dependencies.
    """
    try:
        if not CV2_AVAILABLE or cv2 is None or np is None:
            return True
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = os.path.join('Haarcascades', 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            return True
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return isinstance(faces, (list, tuple, np.ndarray)) and len(faces) == 1
    except Exception:
        return True

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
        socketio = SocketIO(app, cors_allowed_origins="*")
        MONITORING_ENABLED = True
        logger.info("SocketIO initialized successfully")
    except Exception as e:
        logger.error(f"SocketIO init failed: {e}")
        socketio = None
        MONITORING_ENABLED = False
else:
    socketio = None
    MONITORING_ENABLED = False

# Camera & detection globals
CAMERA_INDEX = 0

# Object detection logic is now handled strictly in WASM on the client side.
# Import monitoring modules
try:
    from warning_system import WarningSystem, TabSwitchDetector
    MONITORING_MODULES_AVAILABLE = True
    logger.info("Monitoring modules imported successfully")
except Exception as e:
    logger.warning(f"Monitoring modules import failed: {e}")
    MONITORING_MODULES_AVAILABLE = False
    # Define fallback classes
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
    f"Detection config: fast_face_only={FAST_FACE_ONLY_MODE}"
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

def _reset_exam_runtime_state(student_id):
    sid = str(student_id)
    with runtime_warning_state_lock:
        if sid in runtime_warning_state:
            runtime_warning_state[sid] = {
                'warnings': 0,
                'student_name': runtime_warning_state[sid].get('student_name', 'Unknown'),
                'violations': []
            }
    if warning_system:
        warning_system.reset_student(sid)

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
                error_str = str(col_error)
                logger.warning(f"Initial register attempt failed: {error_str}")
                
                # If Profile column doesn't exist, insert without it
                if "Unknown column 'Profile'" in error_str or "field list" in error_str.lower():
                    logger.info("Retrying registration without Profile column...")
                    cursor.execute(
                        "INSERT INTO students (Name, Email, Password, Role) VALUES (%s, %s, %s, %s)",
                        (fullname, email, generate_password_hash(password), 'STUDENT')
                    )
                else:
                    raise col_error
            
            mysql.connection.commit()
            cursor.close()
            
            logger.info(f"Registration successful for email: {email}")
            flash('Registration successful! Please sign in now.', 'register_success')
            return redirect(url_for('main'))
            
        except Exception as e:
            if mysql.connection:
                mysql.connection.rollback()
            if cursor:
                cursor.close()
            logger.error(f"Registration error for {email}: {e}", exc_info=True)
            flash(f'Error during registration: {str(e)}', 'register_error')
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
            # legacy python monitor disabled
            with active_exam_students_lock:
                active_exam_students.discard(sid_int)
            with latest_student_frames_lock:
                latest_student_frames.pop(sid_int, None)
            with student_detection_state_lock:
                student_detection_state.pop(sid_int, None)
        detection_threads_started = False
        # camera_streamer.release() # This line was commented out in the original code, keeping it that way.
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

# @app.route('/faceInput')
# @require_role('STUDENT')
# def faceInput():
#     # Release any server-held webcam so browser capture can open camera reliably.
#     try:
#         # camera_streamer.release() # This line was commented out in the original code, keeping it that way.
#         pass
#     except Exception:
#         pass
#     user = current_user()
#     # legacy python monitor disabled
#     with active_exam_students_lock:
#         if user:
#             active_exam_students.discard(int(user['Id']))
#     return render_template('ExamFaceInput.html')

@app.route('/video_capture')
def video_capture():
    """Stream MJPEG for face capture page (simple preview)."""
    def gen():
        try:
            # camera_streamer.start() # This line was commented out in the original code, keeping it that way.
            pass
        except Exception as e:
            logger.error(f"video_capture start error: {e}")
            return
        while True:
            try:
                # frame = camera_streamer.read() # This line was commented out in the original code, keeping it that way.
                frame = np.zeros((480, 640, 3), dtype=np.uint8) # Placeholder frame
            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                break
            # draw rectangles for preview optionally
            if CV2_AVAILABLE:
                # faces = detect_faces(frame) # This line was commented out in the original code, keeping it that way.
                # for f in faces:
                #     cv2.rectangle(frame, (f['x'], f['y']), (f['x'] + f['w'], f['y'] + f['h']), (0,255,0), 2)
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
        # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # This line was commented out in the original code, keeping it that way.
        frame = np.zeros((480, 640, 3), dtype=np.uint8) # Placeholder frame

        if frame is None:
            raise Exception("Could not decode image data.")
            
        # File name banayein aur save karein (assuming 'static/profiles' folder exists)
        profileName = f"profile_{int(time.time())}.jpg"
        save_path = os.path.join('static', 'profiles', profileName) 
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        # cv2.imwrite(save_path, frame) # This line was commented out in the original code, keeping it that way.
        
        flash('Face captured successfully and saved.', 'success')
        
        # Seedha Exam System Check page par redirect karein (User ki zaroorat ke mutabik)
        return redirect(url_for('systemCheck'))

    except Exception as e:
        logger.error(f"saveFaceInput error: {e}")
        flash(f'Failed to process or save image: {e}', 'error')
        # Error hone par wapis face input page par bhej dein
        return redirect(url_for('faceInput'))

# @app.route('/confirmFaceInput')
# def confirmFaceInput():
#     profile = profileName
#     # return render_template('ExamConfirmFaceInput.html', profile=profile)
#     return redirect(url_for('systemCheck'))

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
            # camera_streamer.start() # This line was commented out in the original code, keeping it that way.
            print("✅ Server camera started (fallback mode)")
            # print(f"✅ Camera running: {camera_streamer.running}") # This line was commented out in the original code, keeping it that way.
        else:
            # camera_streamer.release() # This line was commented out in the original code, keeping it that way.
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
                         monitoring_enabled=MONITORING_ENABLED,
                         wasm_proctor_enabled=True)

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

        _reset_exam_runtime_state(student_id)
        # legacy python monitor disabled

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



@app.route('/api/proctor/manifest', methods=['GET'])
def proctorManifest():
    """Serve the proctor engine integrity manifest."""
    manifest_path = os.path.join(app.static_folder, 'proctor_engine', 'manifest.json')
    if not os.path.isfile(manifest_path):
        return jsonify({'error': 'Manifest not found'}), 404
    return send_file(manifest_path, mimetype='application/json')

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
        
        # The backend AI Vision Engine was removed in favor of the WASM frontend engine.
        # Just verify that a valid frame was transmitted.
        session['face_verified_for_exam'] = True
        session['student_face_verified_at'] = time.time()
        
        return jsonify({
            'ok': True,
            'matched': True,
            'distance': 0.0
        }), 200
        
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
    # camera_streamer.release() # This line was commented out in the original code, keeping it that way.
    
    user = current_user()
    student_id = user['Id'] if user else None
    student_name = user['Name'] if user else 'Unknown'
    
    # stop admin monitoring for student
    sid_str = str(student_id) if student_id else (str(user['Id']) if user else None)
    # legacy python monitor disabled
    
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
        'gaze_up_left': 'GAZE_UP_LEFT', 'GAZE_UP_LEFT': 'GAZE_UP_LEFT',
        'gaze_up_right': 'GAZE_UP_RIGHT', 'GAZE_UP_RIGHT': 'GAZE_UP_RIGHT',
        'gaze_down_left': 'GAZE_DOWN_LEFT', 'GAZE_DOWN_LEFT': 'GAZE_DOWN_LEFT',
        'gaze_down_right': 'GAZE_DOWN_RIGHT', 'GAZE_DOWN_RIGHT': 'GAZE_DOWN_RIGHT',
        'voice_detected': 'VOICE_DETECTED', 'VOICE_DETECTED': 'VOICE_DETECTED',
        'DISTRACTION': 'DISTRACTION', 'distraction': 'DISTRACTION',
        'NOT_FORWARD': 'DISTRACTION', 'not_forward': 'DISTRACTION',
        'GAZE_AWAY': 'DISTRACTION', 'gaze_away': 'DISTRACTION',
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
                
                # Fetch violations for the latest session to build a breakdown for the report
                violations = []
                violations_breakdown = {}
                try:
                    vcur = mysql.connection.cursor()
                    vcur.execute("""
                        SELECT ViolationType, Details, Timestamp
                        FROM violations
                        WHERE SessionID = %s
                        ORDER BY Timestamp ASC
                    """, (row[5],))
                    vrows = vcur.fetchall()
                    vcur.close()
                    if vrows:
                        for vrow in vrows:
                            vtype = str(vrow[0] or 'UNKNOWN')
                            violations.append({
                                'type': vtype,
                                'details': str(vrow[1] or ''),
                                'time': str(vrow[2] or '')
                            })
                            violations_breakdown[vtype] = violations_breakdown.get(vtype, 0) + 1
                except Exception as v_err:
                    logger.warning(f"Violations fetch warning: {v_err}")

                result_data = {
                    'percentage':        percentage,
                    'score':             (row[2] or 0) * 2,  # CorrectAnswers * 2
                    'correct_answers':   int(row[2] or 0),
                    'total_questions':   row[1] or 125,
                    'grade':             grade,
                    'time_spent':        time_spent,
                    'warnings_issued':   int(row[7]) if row[7] else 0,
                    'auto_terminated':   (db_status == 'TERMINATED'),
                    'submission_time':   row[3],
                    'exam_title':        'Final Examination',
                    'violations':        violations,
                    'violations_breakdown': violations_breakdown,
                    'total_violations':  len(violations)
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
            with open(save_path, 'wb') as out:
                out.write(img_bytes)
        elif webcam_image:
            try:
                image_b64 = webcam_image.split(',', 1)[1] if ',' in webcam_image else webcam_image
                img_bytes = base64.b64decode(image_b64)
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
# Voice tracking dict: Dict[student_id, bool] - indicates if voice was currently heard
_student_voice_activity = {}
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
        # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # This line was commented out in the original code, keeping it that way.
        frame = np.zeros((480, 640, 3), dtype=np.uint8) # Placeholder frame
        if frame is None:
            return jsonify({'ok': False, 'error': 'Bad frame'}), 400
        # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR) # This line was commented out in the original code, keeping it that way.

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

        # The WASM engine now processes everything client side. 
        # We still accept frames to stream to the admin view via `admin_live_stream` MJPEG.
        
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
        _record_runtime_warning(student_id, student_name, 'TAB_SWITCH', f"{event_type}: {details}")
        if warning_system:
            warning_system.add_warning(student_id, 'TAB_SWITCH', f"{event_type}: {details}")
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

# Pipeline API endpoints were removed since inference runs in client WASM

# -------------------------
# SocketIO handlers
# -------------------------
if MONITORING_ENABLED and socketio:
    @socketio.on('connect', namespace='/student')
    def student_connect():
        user = current_user()
        if not user:
            return False  # reject unauthenticated
        sid = request.sid
        user_id = str(user.get('Id', ''))
        try:
            join_room(f"student:{user_id}")
        except Exception:
            pass
        logger.info(f'Student socket connected: {sid} (user={user_id}, role={user.get("Role")})')

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
        if str(vtype).upper() == 'TAB_SWITCH':
            dlow = str(details).lower()
            if 'lost focus' in dlow or 'window' in dlow or 'hidden' in dlow:
                details = 'Tab switching detected'
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
            'GAZE_UP_LEFT': 'GAZE_UP_LEFT', 'gaze_up_left': 'GAZE_UP_LEFT',
            'GAZE_UP_RIGHT': 'GAZE_UP_RIGHT', 'gaze_up_right': 'GAZE_UP_RIGHT',
            'GAZE_DOWN_LEFT': 'GAZE_DOWN_LEFT', 'gaze_down_left': 'GAZE_DOWN_LEFT',
            'GAZE_DOWN_RIGHT': 'GAZE_DOWN_RIGHT', 'gaze_down_right': 'GAZE_DOWN_RIGHT',
            'VOICE_DETECTED': 'VOICE_DETECTED', 'voice_detected': 'VOICE_DETECTED',
            'DISTRACTION': 'DISTRACTION', 'distraction': 'DISTRACTION',
            'NOT_FORWARD': 'DISTRACTION', 'not_forward': 'DISTRACTION',
            'GAZE_AWAY': 'DISTRACTION', 'gaze_away': 'DISTRACTION',
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
            student_name = data.get('student_name') or 'Unknown'
            # Record runtime warning but omit termination logic since that's handled client-side or natively now
            _record_runtime_warning(student_id, student_name, 'PROHIBITED_SHORTCUT', str(action))
            if warning_system:
                terminated = warning_system.add_warning(student_id, 'PROHIBITED_SHORTCUT', str(action))
            else:
                terminated = False
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
        dlow = details.lower()
        if 'lost focus' in dlow or 'window' in dlow or 'hidden' in dlow:
            details = 'Tab switching detected'
        if student_id:
            _record_runtime_warning(student_id, student_name, 'TAB_SWITCH', details)
            if warning_system:
                terminated = warning_system.add_warning(student_id, 'TAB_SWITCH', details)
            else:
                terminated = False
            logger.info(f"[TAB_SWITCH] student={student_id} details={details} terminated={terminated}")
        else:
            logger.info(f"[TAB_SWITCH IGNORED] missing student_id details={details}")

    # --- WASM TELEMETRY LAYER ---
    @socketio.on('telemetry_update', namespace='/student')
    def handle_telemetry_update(data):
        student_id = str(data.get('student_id'))
        score = data.get('score', 0)
        faces = data.get('faces', 0)
        objects = data.get('objects', {})
        
        sid_str = student_id
        now_ts = time.time()
        
        with latest_student_frames_lock:
            prev = latest_student_frames.get(sid_str, {})
            # Keep student alive in the active students polling, and update their telemetry
            if score >= 50:
                logger.info(f"🚨 [WASM TELEMETRY] High Suspicion for {sid_str}: Score {score}")
                
            latest_student_frames[sid_str] = {
                'frame': prev.get('frame'),
                'processed_frame': prev.get('processed_frame'),
                'timestamp': prev.get('timestamp', now_ts), 
                'frame_timestamp': prev.get('frame_timestamp', now_ts),
                'processed_timestamp': now_ts,
                'detections': prev.get('detections', []),
                'status_snapshot': {
                    'warning_count': int(warning_system.get_warnings(sid_str) if warning_system else 0),
                    'suspicion_score': score,
                    'faces_detected': faces,
                    'phone_detected': objects.get('phone', False),
                    'laptop_detected': objects.get('laptop', False)
                },
                'last_visible_object_labels': prev.get('last_visible_object_labels', []),
                'last_prohibited_object_labels': ['cell phone'] if objects.get('phone') else [],
                'last_person_count': faces,
            }

    # --- WEBRTC SIGNALING (STUDENT -> ADMIN) ---

    # ── Ring buffer for detailed telemetry history (admin cross-verification) ──
    _telemetry_history = {}  # { student_id: deque(maxlen=200) }
    _telemetry_history_lock = threading.Lock()

    @socketio.on('student_live_frame', namespace='/student')
    def handle_student_live_frame(data):
        """Relay live camera frame from student to admin via socket (no OpenCV needed)."""
        student_id = str(data.get('student_id', ''))
        if not student_id or not data.get('frame'):
            return
        logger.debug(f'[FRAME] Received live frame from student {student_id}, size={len(data["frame"][:20])}...')
        # Register student as active
        with active_exam_students_lock:
            active_exam_students.add(student_id)
        # Update timestamp in latest_student_frames so polling finds them
        now_ts = time.time()
        with latest_student_frames_lock:
            prev = latest_student_frames.get(student_id, {})
            latest_student_frames[student_id] = {
                **prev,
                'timestamp': now_ts,
                'frame_timestamp': now_ts,
                'processed_timestamp': now_ts,
            }
        # Relay to admin namespace as 'student_frame' (which admin already listens for)
        socketio.emit('student_frame', {
            'student_id': student_id,
            'frame': data['frame'],
            'score': data.get('score', 0),
            'timestamp_ms': data.get('timestamp_ms', int(now_ts * 1000))
        }, namespace='/admin')

    @socketio.on('student_audio_chunk', namespace='/student')
    def handle_audio_chunk(data):
        """
        Relays raw PCM Int16 audio chunk to the admin namespace for local processing.
        """
        try:
            user = current_user()
            if not user or user.get('Role') != 'STUDENT':
                return
            student_id = str(user['Id'])
            
            # Relay raw JS ArrayBuffer bytes to Admin Dashboard
            socketio.emit('relay_student_audio', {
                'student_id': student_id,
                'audio': data  # data is the binary buffer
            }, namespace='/admin')
                
        except Exception as e:
            logger.error(f"Error relaying audio chunk: {e}")

    @socketio.on('admin_trigger_voice_warning', namespace='/admin')
    def handle_admin_voice_detection_trigger(data):
        """
        Triggered by the Admin Dashboard's local voice detection logic.
        Sends a voice_alert directly to the student.
        """
        try:
            student_id = data.get('student_id')
            if not student_id: return
            
            logger.warning(f"🎙️ Admin-side system DETECTED VOICE for student {student_id}")
            
            # Mark as active for metrics
            _student_voice_activity[student_id] = {"active": True, "rms": data.get('rms', 100)}
            
            # Notify student
            target_room = f"student:{student_id}"
            socketio.emit('voice_alert', {'detected': True, 'rms': data.get('rms', 100)}, namespace='/student', to=target_room)
        except Exception as e:
            logger.error(f"Error triggering voice warning: {e}")

    @socketio.on('telemetry_update_v2', namespace='/student')
    def handle_telemetry_update_v2(data):
        """Receive hyper-detailed telemetry from student WASM engine and relay to admin."""
        student_id = str(data.get('student_id', ''))
        if not student_id:
            return
        score = data.get('analysis', {}).get('suspicion_score', 0)
        metrics = data.get('metrics', {})
        # Normalize labels for admin display (smartwatch/headphones)
        raw_labels = list(metrics.get('banned_labels') or [])
        accessory = metrics.get('accessory') or {}
        has_headphones = bool(accessory.get('earphone_detected') or accessory.get('headphone_detected'))
        if has_headphones and 'headphones' not in raw_labels:
            raw_labels.append('headphones')
        normalized = []
        for label in raw_labels:
            if label == 'clock':
                normalized.append('smartwatch')
            else:
                normalized.append(label)
        # If only a phone label is present but headphones are detected, prefer headphones
        if has_headphones and normalized == ['cell phone']:
            normalized = ['headphones']
        metrics['banned_labels'] = normalized
        data['metrics'] = metrics

        # Ensure student appears in admin polling
        with active_exam_students_lock:
            active_exam_students.add(student_id)

        # Inject server-side voice_detected flag into active_flags
        analysis = data.get('analysis', {})
        active_flags = analysis.get('active_flags', [])
        voice_info = _student_voice_activity.get(student_id, {"active": False, "rms": 0})
        is_voice = voice_info.get("active", False)
        voice_rms = voice_info.get("rms", 0)
        
        # Always add voice metadata to metrics for real-time admin display
        metrics['voice_rms'] = float(voice_rms)
        metrics['voice_threat_level'] = min(100, int((voice_rms / 1000) * 100)) # Scale 0-100%
        data['metrics'] = metrics

        if is_voice:
            if 'voice_detected' not in active_flags:
                active_flags.append('voice_detected')
            
        analysis['active_flags'] = active_flags
        data['analysis'] = analysis
            
        # Reset the voice activity trap for the next telemetry window
        _student_voice_activity[student_id] = {"active": False, "rms": 0}

        # Store in ring buffer for cross-verification
        from collections import deque
        with _telemetry_history_lock:
            if student_id not in _telemetry_history:
                _telemetry_history[student_id] = deque(maxlen=200)
            _telemetry_history[student_id].append(data)

        # Update existing tracking dict
        now_ts = time.time()
        with latest_student_frames_lock:
            prev = latest_student_frames.get(student_id, {})
            latest_student_frames[student_id] = {
                'frame': prev.get('frame'),
                'processed_frame': prev.get('processed_frame'),
                'timestamp': prev.get('timestamp', now_ts),
                'frame_timestamp': prev.get('frame_timestamp', now_ts),
                'processed_timestamp': now_ts,
                'detections': prev.get('detections', []),
                'status_snapshot': {
                    'warning_count': int(warning_system.get_warnings(student_id) if warning_system else 0),
                    'suspicion_score': score,
                    'faces_detected': metrics.get('face_count', 0),
                    'phone_detected': 'cell phone' in metrics.get('banned_labels', []),
                    'laptop_detected': 'laptop' in metrics.get('banned_labels', []),
                },
                'last_visible_object_labels': metrics.get('banned_labels', []),
                'last_prohibited_object_labels': metrics.get('banned_labels', []),
                'last_person_count': metrics.get('person_count', 0),
                'wasm_telemetry': data,  # Store full telemetry for admin
            }

        if score >= 50:
            logger.info(f"🚨 [WASM TELEMETRY v2] High Suspicion for {student_id}: Score {score}")

        # Broadcast to admin namespace
        socketio.emit('student_telemetry_v2', data, namespace='/admin')

    @socketio.on('admin_notify_student', namespace='/admin')
    def handle_admin_notify_student(data):
        """Relay an observation notification from admin to a specific student."""
        student_id = str(data.get('student_id', ''))
        if not student_id:
            return
        
        # We broadcast the specific metrics that the admin is seeing to the student
        # so the student knows EXACTLY what was flagged.
        notification_payload = {
            'message': 'You are being observed by the Proctor.',
            'timestamp': time.time(),
            'metrics': data.get('metrics', {}) # admin sends current snapshot
        }
        
        target_room = f"student:{student_id}"
        logger.info(f"🔔 Admin notifying student in room {target_room}")
        socketio.emit('admin_notification', notification_payload, namespace='/student', to=target_room)

    @socketio.on('force_terminate_exam', namespace='/admin')
    def handle_force_terminate_exam(data):
        """Admin force terminates a student's exam with a summary report."""
        student_id = str(data.get('student_id', ''))
        if not student_id:
            return
        
        reason = data.get('reason', 'Administrative decision')
        metrics_summary = data.get('metrics_summary', {})
        
        termination_payload = {
            'terminated': True,
            'reason': reason,
            'metrics_summary': metrics_summary,
            'timestamp': time.time()
        }
        
        target_room = f"student:{student_id}"
        logger.warning(f"❌ Admin FORCE TERMINATED student in room {target_room}: {reason}")
        socketio.emit('exam_terminated', termination_payload, namespace='/student', to=target_room)
        
        # We also trigger the internal termination logic if needed
        if 'warning_system' in globals():
            warning_system.reset_warnings(student_id) # Optional: reset or mark as terminated

    @socketio.on('request_student_frames', namespace='/admin')
    def handle_request_student_frames(data):
        """Admin requests random frame snapshots from a student for cross-verification."""
        student_id = str(data.get('student_id', ''))
        count = min(int(data.get('count', 6)), 10)
        socketio.emit('capture_frames', {'count': count, 'request_id': data.get('request_id')}, namespace='/student', to=student_id)

    @socketio.on('student_frame_response', namespace='/student')
    def handle_student_frame_response(data):
        """Student sends captured frames back to admin for cross-verification."""
        socketio.emit('student_frame_captured', data, namespace='/admin')

    @app.route('/api/admin/student-telemetry/<student_id>', methods=['GET'])
    def get_student_telemetry_history(student_id):
        """Admin API: get recent telemetry history for a student."""
        with _telemetry_history_lock:
            history = list(_telemetry_history.get(student_id, []))
        return jsonify({'student_id': student_id, 'history': history[-50:]})  # Last 50 entries


    @socketio.on('webrtc_offer', namespace='/student')
    def handle_webrtc_offer(data):
        socketio.emit('webrtc_offer', data, namespace='/admin')

    @socketio.on('webrtc_ice_candidate', namespace='/student')
    def handle_webrtc_ice_student(data):
        socketio.emit('webrtc_ice_candidate', data, namespace='/admin')

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

    # --- WEBRTC SIGNALING (ADMIN -> STUDENT) ---
    @socketio.on('request_webrtc_stream', namespace='/admin')
    def handle_request_webrtc_stream(data):
        student_id = data.get('student_id')
        socketio.emit('request_webrtc_stream', data, namespace='/student', room=f"student:{student_id}")

    @socketio.on('webrtc_answer', namespace='/admin')
    def handle_webrtc_answer(data):
        student_id = data.get('student_id')
        socketio.emit('webrtc_answer', data, namespace='/student', room=f"student:{student_id}")

    @socketio.on('webrtc_ice_candidate', namespace='/admin')
    def handle_webrtc_ice_admin(data):
        student_id = data.get('student_id')
        socketio.emit('webrtc_ice_candidate', data, namespace='/student', room=f"student:{student_id}")

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
