<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=40&pause=1000&color=3B82F6&center=true&vCenter=true&width=800&lines=AI+Vision+Online+Exam+Proctor;Hybrid+WASM+%2B+WebRTC+Architecture;Zero-Latency+Proctoring+Engine" alt="Typing SVG" />
</h1>

<div align="center">

  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
  ![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
  ![WebRTC](https://img.shields.io/badge/WebRTC-333333?style=for-the-badge&logo=webrtc&logoColor=white)
  ![Socket.io](https://img.shields.io/badge/Socket.io-010101?style=for-the-badge&logo=socket.io&logoColor=white)
  ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)

  **A highly-scalable, privacy-preserving AI proctoring system using Edge AI inference.**
  
  <br />
</div>

---

## 📖 The Vision

As remote learning scales, traditional server-side proctoring faces massive bottlenecks: high bandwidth costs, massive cloud compute bills, and latency delays. 

**This project flips the paradigm.** By pushing heavy computer vision models (YOLO/BlazeFace/FaceMesh) to the **Edge** via WebAssembly (WASM) and WebGL, we achieve:
- 🚀 **Zero-latency** inference (~30 FPS directly in the browser)
- 📉 **99% reduction** in server bandwidth and GPU costs
- 🔒 **Privacy-by-Design** (No mass video streaming to third-party APIs)

---

## 🏗 System Architecture (Hybrid Edge/Server)

The project leverages a **Hybrid Detection Architecture** combining client-side speed with server-side trust verification.

```mermaid
graph TD
    subgraph "🎓 Student Browser (Edge)"
        Cam[Webcam] --> TFJS[TFJS / WASM Engine]
        TFJS --> |BlazeFace + COCO-SSD| LocalScoring[Local Suspicion Score Tracker]
        TFJS --> |FaceMesh| LocalScoring
        Cam --> |Peer-to-Peer Video| RTC[WebRTC Stream]
    end

    subgraph "⚡ Proctordb Server (Flask)"
        LocalScoring --> |Socket.io Telemetry (15Hz)| Socket[Telemetry Relay]
        Cam --> |Fall-back Frame (1Hz)| YOLO[Python YOLOv11 Server]
        YOLO --> Trap[Score Mismatch Trap]
    end

    subgraph "👮‍♂️ Admin Dashboard"
        RTC --> AdminVideo[Zero-latency CCTV Feed]
        Socket --> |Live Telemetry| AdminCards[Live Alert Cards]
        AdminVideo --> |WASM Local Check| AdminVerify[Admin Local Tamper Verifier]
    end

    Trap -.-> |If hacked JS detected| Strike[TAMPER_DETECTED Strike]
    AdminVerify -.-> |Flags mismatched score| Strike
```

### How it works ⚙️
1. **Client-Side Heavy Lifting:** The browser downloads lightweight TFJS models. The student's webcam feed is processed entirely locally.
2. **Telemetry Stream:** The browser emits high-speed compressed JSON (Suspicion Score, Face counts, Active Flags) via WebSockets.
3. **WebRTC Video Stream:** A true Peer-to-Peer connection is forged between the Student and Admin for Live CCTV.
4. **The Trap (Anti-Tampering):** To prevent students from modifying the JS to return a `0` score, the admin can click **"🛡️ Verify Tamper"**. This runs the admin's local WASM instance against the student's live feed. If the student's self-reported score drastically differs from the admin's calculation, they are flagged for massive cheating.

---

## 🔥 Key Detections

| Detection Type | AI Model | Description |
|---|---|---|
| **Multiple Faces / No Face** | *BlazeFace / Haar Cascade* | Flags if another person sits in, or if the student leaves. |
| **Gaze Tracking** | *MediaPipe Iris* | Detects if the student is reading off a hidden screen left/right. |
| **Head Pose (Yaw/Pitch/Roll)** | *MediaPipe FaceMesh* | Calculates geometry of 478 landmarks to detect physical head turning. |
| **Prohibited Objects** | *COCO-SSD / YOLOv11* | Detects cell phones (`#67`), books (`#73`), and laptops (`#63`). |
| **Eyes Closed (Sleep)** | *Eye Aspect Ratio (EAR)* | Measures distance between eyelid landmarks to detect long blinks or sleeping. |
| **Environment Violations** | *JavaScript APIs* | Catches tab-switching, exiting fullscreen, and background voice/noise. |

---

## 🛠 For Developers: Deep Dive

### Directory Structure & File Roles
```text
.
├── app.py                       # 🚀 Main Flask Core (3900+ LOC) - Sockets, Auth, Route Handlers
├── face_pipeline.py             # 👤 MediaPipe Python Wrapper (Server-side validation)
├── person_pipeline.py           # 📦 YOLOv11 Python Wrapper (Server-side validation)
├── decision_engine.py           # 🧠 Rule engine (temporal debouncing & score logic)
├── config_vision.py             # ⚙️ Master configuration (Edit thresholds here!)
├── static/
│   ├── proctor_engine/          
│   │   ├── runtime/             # ⚡ Core WASM evaluation logic (proctor_core.js)
│   │   ├── config/              # ⚡ Client-side configs (detection_config.js)
│   │   └── models/              # ONNX/TFJS model weights
│   └── css/ js/ img/
├── templates/                   # 🎨 Jinja2 HTML Templates (Exam, Dashboards, Auth)
└── requirements.txt             # 📦 Python deps
```

### The Telemetry Payload
The student client streams metadata at high frequency via `telemetry_update_v2`:
```json
{
  "safety_level": 85,
  "risk_score": 15,
  "verdict": "GOOD_TO_GO",
  "face_count": 1,
  "person_count": 1,
  "banned_labels": ["cell phone"],
  "active_flags": {
    "no_face": false,
    "banned_object": true,
    "looking_away": false
  }
}
```

### How to Tweak Sensitivity
To configure how strict the AI is, developers just need to modify `static/proctor_engine/config/detection_config.js` for the frontend, and `config_vision.py` for the backend fallback.

```javascript
// Example: Make head-turning strict
yaw_threshold: 18, 
pitch_risk: 15,
```

---

## 💻 Installation & Setup

### 1. Requirements
- Python 3.9+
- MySQL Server 5.7+ / MariaDB
- A modern browser with WebCam permissions enabled

### 2. Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/exam-proctor.git
cd exam-proctor

# Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# Install Dependencies
pip install -r requirements.txt

# Hydrate the Database
mysql -u root -p < examproctordb.sql

# Start the Flask Server
python app.py
```

### 3. Accessing the System
- **Student Portal:** `http://127.0.0.1:5001/`
- **Admin Dashboard:** `http://127.0.0.1:5001/admin/dashboard`
  *(Note: WebRTC and WebCam APIs strictly require `localhost`, `127.0.0.1`, or a valid `https://` proxy to work due to browser security policies!)*

---

## 🛡️ Security Measures
- **Password Crypto:** PBKDF2-SHA256 (Werkzeug)
- **Session Auth:** Secure, HTTP-Only cookies with `SameSite=Lax`.
- **RBAC:** Hardened route decorators (`@require_role('ADMIN')`).
- **WASM Integrity:** Core scripts are checksum hashed (`sync_proctor_assets.sh`). If modified, exam initialization fails.

<br>
<p align="center">
  <i>Built to make online education fairer, faster, and truly scalable.</i>
</p>
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=3B82F6&height=100&section=footer" width="100%"/>
</p>on and partial-face detection
- better low-light performance
- test suite for frame pipeline and warning logic
- containerized deployment with Docker

## Reference

Project details PDF:
- `OEP Project.pdf`




