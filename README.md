# Online Exam Proctor – Architecture & Setup

AI-driven proctoring web app (Flask + Socket.IO) that ingests webcam streams, analyzes gaze/head/object/tab activity, issues warnings, and auto-terminates exams after violations. Scoring is computed on the server; results and violations persist to MySQL.

## What the system does
- Live video ingestion from the student browser (WebRTC → base64 JPEG → Flask).
- Vision stack: MediaPipe Face Mesh (face/iris/pose), YOLOv8 (primary objects/person), optional GroundingDINO (text-guided rare objects), heuristic fallbacks for phones.
- Behavior engine: gaze (left/right/down/away) with smoothing + 3s timer, head pose, multi-face, camera-off detection, tab switching, prohibited objects.
- Warning engine: 3 warnings max, 3s cooldown between warnings, 3rd warning → 3s grace → termination. Terminated exams show `FAILED - CHEATING DETECTED`.
- Scoring: per-question marks, normalized answers (trim+lower), server-side percentage + pass/fail vs env `PASS_PERCENTAGE` (default 50).

## Tech stack
- Backend: Python 3.10+, Flask 2.3, Flask-SocketIO, Eventlet, PyMySQL.
- CV/AI: OpenCV, MediaPipe 0.10, Ultralytics YOLOv8 (default `yolov8n.pt`, supports `yolov8m.pt`), optional GroundingDINO (text prompts), pytesseract OCR fallback (disabled by default).
- Frontend: HTML/JS/CSS (Bootstrap), Socket.IO client, Canvas/WebRTC, Web Audio API.
- DB: MySQL/MariaDB (`examproctordb.sql`).

## System architecture (data flow)
1) Browser captures webcam → base64 JPEG POST `/api/student-frame` (~10–12 FPS).  
2) Flask request enqueues latest frame; background worker does:  
   - Face Mesh (478 landmarks) → iris offsets, EAR, yaw/pitch.  
   - Gaze classification (smoothing over 5 frames; upward ignored).  
   - YOLOv8 objects + optional GroundingDINO merge; 8+ frame persistence (books need ≥10).  
   - Tab switch events via Socket.IO/visibilitychange.  
   - Decision/warning: apply cooldowns, increment warning count, emit to UI + admin.  
3) Warnings: 3s gap; on 3rd warning schedule termination in 3s.  
4) Results: on submit/termination, server computes score, sets status (PASS/FAIL/FAILED - CHEATING DETECTED), stores sessions/results/violations in MySQL, emits to admin dashboard.  

## Key rules & thresholds (defaults)
- Warnings: MAX_WARNINGS=3, WARNING_COOLDOWN=3s, TERMINATION_DELAY=3s.  
- Gaze: LEFT/RIGHT/DOWN/AWAY must persist ≥3s (`GAZE_TIME_THRESHOLD`); smoothing frames=5; yaw limit=30°, pitch up ignored, pitch down < -20° → DOWN; upward gaze ignored globally.  
- Objects: confidence filters + texture/edge/entropy; books: conf ≥0.65, area 2.2–15%, edge ≥0.06, texture ≥220, ≥10 consecutive frames. Top-of-frame hits ignored (`OBJECT_TOP_IGNORE_RATIO`).  
- Camera-off: frame staleness >4.5s triggers `CAMERA_OFF`.  
- Scoring: normalized answers; percentage = score / total_marks * 100; pass if ≥ `PASS_PERCENTAGE` (50). Auto-terminated exams labeled `FAILED - CHEATING DETECTED`.  

## Environment variables (common)
```
MYSQL_HOST=127.0.0.1
MYSQL_USER=root
MYSQL_PASSWORD=...
MYSQL_DB=examproctordb
FLASK_SECRET_KEY=<random>

PASS_PERCENTAGE=50
MAX_WARNINGS=3
WARNING_COOLDOWN=3
TERMINATION_DELAY=3

GAZE_TIME_THRESHOLD=3
GAZE_SMOOTH_FRAMES=5
HEAD_YAW_LIMIT=30
HEAD_PITCH_UP_LIMIT=20
HEAD_PITCH_DOWN_LIMIT=-20
GAZE_PROCESS_EVERY_N=2

OBJECT_CONSEC_FRAMES=8
OBJECT_MIN_PIXELS=4000
OBJECT_TOP_IGNORE_RATIO=0.12

GROUNDING_DINO_ENABLED=0
GROUNDING_DINO_CONFIG=path/to/config.py
GROUNDING_DINO_CHECKPOINT=path/to/weights.pth

# Debug flags (all default 0)
GAZE_DEBUG=0
OBJECT_DEBUG=0
SCORE_DEBUG=0
PERF_DEBUG=0
FRAME_DUMP=0
FRAME_DUMP_EVERY=10
DEBUG_PATH=debug
```

## Installation & run
1) `python -m venv .venv && .venv\Scripts\activate` (Windows).  
2) `pip install -r requirements.txt` (optionally install `transformers`, `groundingdino-py`, `torch`, `supervision` if using GroundingDINO).  
3) `python download_yolo_models.py` (downloads `models/yolov8n.pt`; place `yolov8m.pt` if desired and set `YOLO_MODEL_PATH`).  
4) Import DB schema: `mysql -u root -p < examproctordb.sql`.  
5) Set env vars (PowerShell example):  
   ```powershell
   $env:MYSQL_USER="root"; $env:MYSQL_PASSWORD="yourpass"; $env:MYSQL_DB="examproctordb"
   $env:FLASK_SECRET_KEY="$(Get-Content .flask_secret_key)"
   ```  
6) Run: `python app.py` → open `http://127.0.0.1:5001`.  

## Components (files)
- `app.py` – routes, Socket.IO, frame pipeline, scoring, warning/termination logic, admin APIs.
- `face_pipeline.py` – MediaPipe Face Mesh wrapper (refine_landmarks=True, iris offsets, EAR, yaw/pitch).
- `person_pipeline.py` – YOLOv8 wrapper for person/object detection; integrates with `detect_objects` filters.
- `decision_engine.py` – rule debouncing; gaze handled upstream.
- `warning_system.py` – counts warnings, enforces cooldowns, schedules termination after 3rd warning.
- `config_vision.py` – thresholds/debounce settings.
- `vision_ui.py` – OpenCV overlays for admin live view.
- `admin_live_monitoring.py` – admin utilities + audio monitor hooks.
- `download_yolo_models.py` – fetch YOLO weights.
- `examproctordb.sql` – MySQL schema.

## Debugging (env-gated)
- `GAZE_DEBUG=1` – log per-frame gaze: dir, yaw/pitch, iris offsets, elapsed.
- `OBJECT_DEBUG=1` – log filtered objects (label, conf, bbox).
- `SCORE_DEBUG=1` – log each question compare and marks.
- `PERF_DEBUG=1` – log face/object/total frame timings; YOLO infer time.
- `FRAME_DUMP=1` – save every Nth processed frame to `DEBUG_PATH/<sid>/` (default N=10).

## Current safeguards against false positives
- Upward gaze ignored; blinks suppressed via EAR.
- Book/paper detection tightened (edge/texture/entropy, aspect window, top-of-frame ignore, 10-frame persistence).
- Phone detection allows small boxes but still requires texture/edge or fallback contour heuristic.
- Camera-off only after >4.5s stale frames.

## Result storage
- Tables: `exam_sessions` (IN_PROGRESS/COMPLETED/TERMINATED), `exam_results` (Score %, TotalQuestions, CorrectAnswers, Status), `violations` (type, details, timestamp), `students`, `profiles`.
- Terminated exams labeled `FAILED - CHEATING DETECTED` for UI while DB session status remains `TERMINATED`.

## Optional hybrid detector (GroundingDINO)
- Enable with `GROUNDING_DINO_ENABLED=1` + config/ckpt paths; merges text-prompt detections (phones, book/paper, earbuds, USB, wires) with YOLO, deduped by IoU. Use when small/rare objects are missed by YOLO alone.  

## High-level architecture diagram (text)
```
Browser cam → /api/student-frame → frame queue
   → Face Mesh (iris/EAR/yaw/pitch)
   → Gaze classifier (smoothing + 3s timer; up ignored)
   → YOLOv8 + optional GroundingDINO (+ filters/persistence)
   → Tab switch monitor
   → Decision + warning cooldowns
   → Socket.IO events to student/admin
   → Warning counts (3 max) → 3s termination timer
   → Result persistence (MySQL) + status emit
```

## Usage tips
- Start with defaults; enable one debug flag at a time when tuning.
- For stricter objects: raise `OBJECT_MIN_TEXTURE_VAR`/`OBJECT_MIN_EDGE_DENSITY`, increase `OBJECT_CONSEC_FRAMES`.
- For fewer gaze warnings: increase `GAZE_TIME_THRESHOLD` or `GAZE_SMOOTH_FRAMES`; widen `HEAD_YAW_LIMIT`.

This README reflects the current codebase as configured in March 2026. Update envs, weights, and thresholds to suit your deployment environment. 
