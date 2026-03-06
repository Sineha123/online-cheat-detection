# AI Vision — Online Exam Proctor

The Online Exam Proctor System is a Flask-based web application for monitoring remote exam sessions using browser webcam capture, computer vision, behavior analysis, and warning/violation tracking.

This repository contains both the exam platform and the live proctoring pipeline:
- student-side exam workflow
- admin-side live monitoring dashboard
- camera frame ingestion and background detection workers
- warning and violation management
- result storage and review

The project preserves online exam integrity by detecting suspicious behavior such as missing face, multiple faces, object usage, gaze distraction, head turning, tab switching, and voice/noise activity.

## Project Overview

The Online Exam Proctor System is a computer vision-based project designed to ensure the integrity and fairness of online exams. As remote learning and online education grow, the need for a robust proctoring system becomes crucial to prevent cheating and maintain the credibility of the examination process. This project uses computer vision and AI technologies to monitor and analyze students' behavior during the examination and detect suspicious activities. It also includes voice/noise monitoring to strengthen anti-cheating enforcement.

## System Architecture

The project has 2 primary user roles:
- `Student`
- `Admin`

High-level flow:

1. Student opens the Flask web app and logs in.
2. Student completes system checks (camera, mic, browser).
3. Student registers and verifies their face.
4. Student starts the exam.
5. Student browser captures webcam frames using `getUserMedia`.
6. Browser sends frames to Flask `/api/student-frame` as base64 JPEG.
7. A background worker processes the latest frame using:
   - MediaPipe FaceMesh for face detection and landmarks
   - YOLOv8 for person and object detection
   - DecisionEngine for rule evaluation with temporal debouncing
8. Violations are persisted to MySQL and emitted via Socket.IO.
9. Admin opens the dashboard and monitors active students in real time.
10. Admin can clear warnings, terminate exams, or toggle auto-termination.

## End-to-End Monitoring Pipeline

### 1. Student Browser Capture

The student exam page:
- opens webcam with `navigator.mediaDevices.getUserMedia`
- renders video frames to a hidden canvas
- exports frames as JPEG base64
- uploads them to `/api/student-frame` using `requestAnimationFrame`
- monitors microphone via Web Audio API
- tracks tab visibility via `visibilitychange`

### 2. Flask Frame Receive Layer

Route:
- `/api/student-frame`

Responsibilities:
- receive base64 image payload from browser
- decode into OpenCV BGR frame
- normalize frame size
- update `latest_student_frames` dictionary
- schedule background detection worker

Design rule:
- heavy AI inference never runs in the request handler

### 3. Background Detection Layer

Main worker:
- `_run_student_frame_detection(...)` in `app.py`

Processing steps:
- `FaceAnalyzer.process_frame()` — MediaPipe FaceMesh with 478 landmarks
  - returns: face_detected, yaw_angle, ear, iris_offset_ratio
  - min_detection_confidence: 0.3
  - thread-safe via global lock
- `PersonDetector.process_frame()` — YOLOv8 nano
  - returns: person_count, bounding_boxes, banned_objects
  - detects: persons, phones, books, laptops
- `DecisionEngine.evaluate()` — rule engine with temporal debouncing
  - conditions must persist for configurable durations before warnings fire
  - returns: active_warnings list, penalty_score
- `UILayer.draw_overlays()` — renders status HUD on processed frame
  - bounding boxes, alert panels, status text

### 4. Live Stream Layer

Admin live stream route:
- `/admin/live/<student_id>`

Responsibilities:
- serve MJPEG stream of processed frames
- prefer processed frame when fresh
- fall back to raw frame when needed
- preserve overlays and detector status on fallback frames

### 5. Warning / Violation Layer

The system tracks:
- live warnings per student
- warning cooldowns (prevent spam)
- persisted violations in MySQL
- exam termination thresholds

Violation types:
- `NO_FACE` — face not visible for 3+ seconds (currently active)
- `MULTIPLE_FACES` — more than one person (currently disabled)
- `DISTRACTION` — head turned or gazing away (currently disabled)
- `EYES_CLOSED` — eyes closed for too long (currently disabled)
- `TAB_SWITCH` — left the exam tab (currently disabled)
- `VOICE_DETECTED` — microphone detected voice/noise
- `PROHIBITED_OBJECT` — phone, book, or laptop detected (currently disabled)
- `CAMERA_OFF` — camera feed stopped (currently disabled)

Note: Only `NO_FACE` detection is currently active. All other rules are commented out in `app.py` and `decision_engine.py` and can be re-enabled individually.

### 6. Admin Review Layer

Admin can:
- monitor all active students in a grid layout
- view per-student live video feeds with AI overlays
- clear warnings for a student
- force terminate a student's exam
- toggle auto-termination on/off (defaults to OFF)
- receive real-time toast notifications for violations
- inspect results and violation history
- review saved recordings

## Tech Stack

### Backend
- Python 3.9+
- Flask 2.3.3
- Flask-SocketIO 5.3.6
- PyMySQL 1.1.0
- Eventlet 0.33.3
- bcrypt 4.1.3

### Computer Vision / AI
- OpenCV 4.8.1
- MediaPipe 0.10.7 (FaceMesh with 478 landmarks, iris tracking)
- Ultralytics YOLOv8 Nano (real-time object detection)
- YOLOv4-tiny (legacy fallback)
- Haar Cascades (fast frontal face detection fallback)
- pytesseract 0.3.10 (optional OCR)

### Frontend
- HTML5 / CSS3 / JavaScript
- Bootstrap 5
- Socket.IO client
- Web Audio API
- Canvas API / getUserMedia

## Main Features

### 1. Web Application

Student-side pages:
- Login / Signup / Password Reset
- Rules and Regulations
- System Compatibility Check
- Face Registration (webcam capture or file upload)
- Face Confirmation (identity verification before exam)
- Timed MCQ Exam with real-time proctoring
- Exam Result (pass / fail / terminated)

Admin-side pages:
- Student Management (listing, face registration)
- Active Students Dashboard (live monitoring grid)
- Live Video Feed (MJPEG with AI overlays)
- Exam Results
- Result Details (per-student violation breakdown)
- Recordings / Violations review

### 2. Face Detection and Analysis

- face presence detection (MediaPipe FaceMesh)
- yaw angle estimation (head turning detection)
- Eye Aspect Ratio calculation (eyes closed detection)
- iris offset calculation (gaze tracking)
- configurable detection confidence (currently 0.3)
- temporal debouncing (face must be missing 3 seconds before warning)

### 3. Object Detection

Targeted prohibited items:
- cell phone (COCO class 67)
- book (COCO class 73)
- laptop (COCO class 63)

Additional fallback classes:
- mouse, scissors, teddy bear, refrigerator

### 4. Exam Integrity Controls

- tab switching detection (visibilitychange API)
- fullscreen enforcement
- admin warning escalation
- auto-termination after 3 warnings (when enabled by admin)
- auto-termination defaults to OFF — admin must explicitly enable it

### 5. Audio Monitoring

- voice / noise detection via Web Audio API
- audio-based warning generation
- session audio recording

### 6. Admin Controls

- Auto-Terminate toggle (ON/OFF, defaults OFF)
- Per-student Clear Warnings button
- Per-student Force Terminate button
- Real-time toast notifications via Socket.IO
- Live MJPEG video feed per student

## Current Architecture in the Codebase

### Core Files

- `app.py`
  - main Flask application (~3900 lines)
  - all routes, Socket.IO handlers, frame pipeline
  - detection workers, admin APIs, auth logic
  - violation persistence and warning integration

- `face_pipeline.py`
  - MediaPipe FaceMesh wrapper
  - thread-safe with global lock and static_image_mode
  - returns face_detected, yaw_angle, ear, iris_offset

- `person_pipeline.py`
  - YOLOv8 wrapper for person and object detection
  - detects persons, phones, books, laptops

- `decision_engine.py`
  - rule evaluation with temporal debouncing
  - per-student state tracking
  - configurable time thresholds per condition

- `warning_system.py`
  - per-student warning counters
  - violation throttling with cooldowns
  - auto-termination threshold (3 warnings)
  - auto-terminate toggle (defaults OFF)

- `config_vision.py`
  - all tunable parameters
  - detection thresholds (yaw, EAR, iris offset)
  - temporal debounce durations
  - warning cooldown settings

- `vision_ui.py`
  - OpenCV HUD overlay renderer
  - draws bounding boxes, status panels, alert text

- `admin_live_monitoring.py`
  - legacy admin monitoring utilities
  - audio monitor integration

### Template Files (22 total)

Student flow:
- `login.html`, `signup.html`, `forgot_password.html`, `reset_password.html`
- `ExamRules.html`, `ExamSystemCheck.html`, `ExamSystemCheckError.html`
- `ExamFaceInput.html`, `ExamConfirmFaceInput.html`
- `Exam.html` (main exam page, ~150KB with full proctoring client)
- `ExamResult.html`, `ExamResultPass.html`, `ExamResultFail.html`, `showResultPass.html`

Admin flow:
- `Students.html` (main admin dashboard, ~75KB)
- `Results.html`, `ResultDetails.html`, `ResultDetailsVideo.html`
- `Recordings.html`
- `admin_live_dashboard.html`

## Data Flow in Detail

### Student to Backend

`Browser camera → Canvas → JPEG base64 → /api/student-frame`

### Backend Runtime State

In-memory structures:
- `latest_student_frames`
  - latest raw frame
  - latest processed frame
  - timestamps
  - object labels
  - status snapshot

- `active_exam_students`
  - set of currently active student IDs
  - protected by lock

### Backend to Admin

- real-time events via Socket.IO (`/admin` namespace)
- MJPEG live stream via `/admin/live/<student_id>`
- polling fallback via `/api/admin-active-students`

### Backend to Database

Persistent storage:
- students (accounts, profiles, roles)
- exam_sessions (start/end times, status)
- exam_results (scores, grades)
- violations (type, details, timestamp)
- profiles (face images, detection status)

## Database Schema

Database name: `examproctordb`

### Tables

- `students` — user accounts with role-based access (ADMIN/STUDENT)
- `exam_sessions` — tracks exam lifecycle (IN_PROGRESS, COMPLETED, TERMINATED)
- `exam_results` — stores scores, grades, and submission status
- `violations` — records every detected violation with type and details
- `profiles` — stores student face registration images

### Key Relationships

- `exam_sessions.StudentID` → `students.ID`
- `exam_results.StudentID` → `students.ID`
- `exam_results.SessionID` → `exam_sessions.SessionID`
- `violations.StudentID` → `students.ID`
- `violations.SessionID` → `exam_sessions.SessionID`
- `profiles.student_id` → `students.ID`

All foreign keys have `ON DELETE CASCADE`.

## Configuration Reference

All in `config_vision.py`:

Detection thresholds:
- `YAW_THRESHOLD_DEG = 25` — degrees of head turn before flagging
- `EAR_THRESHOLD = 0.18` — eye aspect ratio below this = eyes closed
- `IRIS_OFFSET_THRESHOLD = 0.28` — iris offset beyond this = looking away

Temporal debouncing (seconds a condition must persist):
- `TIME_NO_FACE = 3.0`
- `TIME_HEAD_TURNED = 2.0`
- `TIME_EYES_CLOSED = 1.0`
- `TIME_GAZING = 2.0`
- `TIME_MULTIPLE_PERSONS = 1.5`
- `TIME_BANNED_OBJECT = 1.5`

Warning settings:
- `WARNING_COOLDOWN_SEC = 5`
- `INSTANT_PENALTY_THRESHOLD = 3`

## Setup and Running

### Requirements

- Python 3.9+
- MySQL 5.7+ or MariaDB 10.4+
- Webcam
- Modern browser (Chrome/Edge recommended)

### Basic Run

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Download YOLO weights: `python download_yolo_models.py`
5. Import database: `mysql -u root -p < examproctordb.sql`
6. Run: `python app.py`
7. Open: `http://127.0.0.1:5001`

### Environment Variables (optional)

- `MYSQL_HOST` — database host (default: 127.0.0.1)
- `MYSQL_PORT` — database port (default: 3306)
- `MYSQL_USER` — database user (default: root)
- `MYSQL_PASSWORD` — database password (default: empty)
- `MYSQL_DB` — database name (default: examproctordb)
- `FLASK_SECRET_KEY` — session secret key

### Important Notes

- webcam access requires `localhost` or `127.0.0.1` origin (or HTTPS)
- MySQL must be running before starting the app
- YOLO model file (`yolov8n.pt`) must be downloaded
- better lighting significantly improves detection quality

## Repository Structure

```text
.
├── app.py                       # Main Flask application
├── face_pipeline.py             # MediaPipe FaceMesh wrapper
├── person_pipeline.py           # YOLOv8 person/object detector
├── decision_engine.py           # Rule engine with temporal debouncing
├── warning_system.py            # Warning counters and enforcement
├── config_vision.py             # Tunable parameters
├── vision_ui.py                 # OpenCV HUD overlay renderer
├── admin_live_monitoring.py     # Admin monitoring utilities
├── proctoring_core.py           # Core proctoring helpers
├── utils.py                     # Legacy CV utilities
├── create_users.py              # Test user creation script
├── download_yolo_models.py      # YOLO model downloader
├── requirements.txt             # Python dependencies
├── examproctordb.sql            # Database schema
├── models/                      # Pre-trained model files
├── Haarcascades/                # Haar cascade classifiers
├── templates/                   # 22 Jinja2 HTML templates
├── static/
│   ├── css/                     # Stylesheets
│   ├── js/                      # Client-side scripts
│   ├── img/                     # UI assets
│   ├── Profiles/                # Student face photos (gitignored)
│   └── audio_recordings/        # Session recordings (gitignored)
└── utils/                       # Additional utility files
```

## Security

- password hashing with bcrypt / pbkdf2-sha256
- session management with HTTP-only cookies
- CSRF token validation on all POST requests
- role-based access control (`@require_role('ADMIN')`)
- camera permissions require secure origin
- input sanitization and length limits
- SameSite=Lax cookie policy

## Debugging

### Check model status

- `/api/object-detection-status` — YOLO loaded status, model path, device, class mapping

### Check pipeline status

- `/api/detection-pipeline-status` — frames received, frame ages, worker state, labels

### Check live stream

- `/admin/live/<student_id>` — continuous MJPEG with face/object/status overlays

### Console logs

Important log families:
- frame received / decoded
- object inference results
- face detection status
- warning generation / cooldown
- violation persistence

## Future Improvement Areas

- split `app.py` into separate modules
- separate detection engine from Flask routes
- dedicated face detector independent from FaceMesh
- stronger occlusion and partial-face detection
- better low-light performance
- test suite for frame pipeline and warning logic
- containerized deployment with Docker

## Reference

Project details PDF:
- `OEP Project.pdf`
