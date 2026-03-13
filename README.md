# AI Vision — Online Exam Proctor

The Online Exam Proctor System is a Flask-based web application for monitoring remote exam sessions. It ensures the integrity and fairness of online exams using a **Hybrid WASM + Server AI architecture**, real-time browser webcam capture, computer vision, behavior analysis, and automated warning/violation tracking.

This repository contains:
- Student-side exam workflow (with client-side AI inference)
- Admin-side Live Monitoring Dashboard (with WebRTC video and real-time telemetry)
- Secure backend API for tampering prevention and violation management
- Result storage and post-exam review tools

## Project Overview

As remote learning and online education grow, the need for a robust proctoring system becomes crucial to prevent cheating and maintain the credibility of the examination process. This project preserves online exam integrity by detecting suspicious behavior such as:
- **Face Issues**: Missing face, multiple faces detected
- **Gaze & Head Pos**: Looking away, head turning (Yaw/Pitch/Roll)
- **Prohibited Objects**: Cell phones, books, laptops, etc.
- **Environment Rules**: Tab switching, hiding the browser window
- **Audio Monitoring**: Voice or background noise detection

## System Architecture: Hybrid WASM & WebRTC

To maximize privacy, reduce server bandwidth by 99%, and provide low-latency monitoring to Admins, the system has been upgraded to a **Hybrid Detection Architecture**.

### High-level flow
1. **Student Login**: Student opens the Flask web app, logs in, and completes identity verification.
2. **Client-Side AI (WASM/TFJS)**: During the exam, the browser downloads lightweight AI models (COCO-SSD and BlazeFace). 
3. **Local Inference**: The student's browser processes the webcam feed locally via WebGL/WebAssembly at ~15-30 FPS.
4. **Suspicion Score**: A real-time Suspicion Score (0-100) is calculated purely in the browser and displayed to the student in an Evaluator Bar.
5. **Telemetry Stream**: The browser sends only a high-frequency JSON telemetry packet (the Score and metadata like face count, active flags) to the server via secure Socket.IO websockets.
6. **WebRTC Video Stream**: For live CCTV, the browser opens a Peer-to-Peer WebRTC connection directly to the Admin Dashboard for low-latency video streaming.
7. **Server-Side Trust Engine (The Trap)**: To prevent students from tampering with the client-side JS score, the browser sends exactly 1 raw frame per second (configurable) to the server. The Python server can run heavy YOLO/MediaPipe models on this fallback frame for validation.
8. **Admin Local Verification**: The Admin dashboard runs its own local WASM instance (`AdminTamperVerifier`). When the admin clicks "Verify Tamper", the admin's browser runs inference on the student's live feed and compares the result against the student's self-reported telemetry. A significant mismatch immediately flags the student for **TAMPERING_DETECTED**.

## End-to-End Monitoring Pipeline

### 1. Client-Side Inference Engine (Student)
- Loads `@tensorflow/tfjs`, `coco-ssd`, and `blazeface` via CDN.
- Captures webcam frames to a hidden canvas and evaluates them locally.
- Detects multiple faces, missing faces, or prohibited objects.
- Calculates the Suspicion Score based on calibrated threshold configs (`detection_config.js`).
- Streams WebRTC video directly to listening Admins.

### 2. Flask Socket.IO Telemetry Layer
- Route: `/student` (WebSockets)
- Receives the 0-100 suspicion score and detailed metadata via the `telemetry_update_v2` event.
- Relays real-time data to the Admin namespace (`/adminStudents`) instantly without waiting for HTTP POSTs.

### 3. Admin Live Dashboard
- Route: `/admin/live_dashboard`
- Displays a grid of actively testing students showing their live status (Safe, Watch, Alert, Critical).
- Warns Admins of prohibited objects directly on the student cards in real-time.
- Shows a high-quality, low-latency live `<video>` feed directly from the student's webcam (WebRTC).
- Features a **Verify Tamper** button for cross-verification of student metrics using the admin's own local inference engine.

## Tech Stack

### Backend
- Python 3.9+
- Flask 2.3.3
- Flask-SocketIO 5.3.6 (with Eventlet for async handling)
- PyMySQL (Database interactions)
- Werkzeug security (Password hashing)

### Computer Vision / AI (Frontend & Backend)
- **Frontend**: TensorFlow.js (WASM/WebGL backend), BlazeFace (Face detection), COCO-SSD (Object detection)
- **Backend**: OpenCV 4.8.1, Ultralytics YOLOv11 Nano (Fallback real-time object detection), MediaPipe (FaceMesh with 478 landmarks)

### Frontend Structure
- Vanilla JS, HTML5, CSS3
- WebRTC (Peer-to-Peer live streaming)
- Socket.IO client (Real-time events)
- Web Audio API (Ambient noise / speech detection)
- Canvas API / MediaDevices API

## Database Schema

Database engine: MySQL / MariaDB  
Database name: `examproctordb`

### Core Tables
1. **`students`**: User accounts with role-based access (`ADMIN` or `STUDENT`). 
2. **`profiles`**: Stores student face registration image paths for identity verification.
3. **`exam_sessions`**: Tracks exam lifecycle (`IN_PROGRESS`, `COMPLETED`, `TERMINATED`), StartTime, and EndTime.
4. **`exam_results`**: Stores scores, correct answers, total questions, and submission status.
5. **`violations`**: Records every detected violation with `ViolationType` (e.g., TAB_SWITCH, MULTIPLE_FACES, TAMPERING), detailed reasons, and timestamps.

### Key Relationships
- Foreign keys link all session, result, and violation data securely to the `students` table via `StudentID`.
- `ON DELETE CASCADE` is implemented to maintain referential integrity.

## Setup and Running

### Requirements
- Python 3.9+
- MySQL 5.7+ or MariaDB 10.4+
- Webcam
- Modern browser (Chrome/Edge recommended)

### Installation & Execution
1. Clone the repository.
2. Create and activate a Python virtual environment: `python3 -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Setup Database: `mysql -u root -p < examproctordb.sql`
5. *(Optional Backend CV)*: Download YOLO weights: `python download_yolo_models.py`
6. Sync WASM manifests: `./scripts/sync_proctor_assets.sh` (Updates integrity hashes)
7. Run the application: `python app.py`
8. Open your browser to `http://127.0.0.1:5001` (Note: Webcam access requires `localhost`, `127.0.0.1`, or an HTTPS origin).

## Security & Architecture Details

- **Integrity Checks**: The WASM scripts (`proctor_core.js`, `student_engine.js`) are checksum-verified on load. Modification of core client-scripts restricts the student from entering the exam.
- **Pass hashing**: pbkdf2-sha256 (via Werkzeug)
- **WebRTC Signaling**: Built directly into the Flask Socket.IO layer — no external STUN/TURN needed for local environments.
- **Admin Verification**: The `AdminTamperVerifier` enables completely trustless remote monitoring. If a student modifies their local JS objects to force a `0` score, the Admin clicking "Verify Tamper" will run inference on their own machine, see a score mismatch, and log a `TAMPERING` violation.

## Future Improvement Areas
- Dockerize the application for easier deployment.
- Implement an external STUN/TURN server config for WebRTC across strict corporate firewalls.
- Split `app.py` into smaller blueprints (e.g., Auth, Exam API, Admin GUI).
- Integrate an LLM for automated post-exam transcript and behavior review.on and partial-face detection
- better low-light performance
- test suite for frame pipeline and warning logic
- containerized deployment with Docker

## Reference

Project details PDF:
- `OEP Project.pdf`




