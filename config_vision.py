# Configuration Constants
import os

# Camera Config
CAMERA_ID = 0
TARGET_FPS = 30

# Feature Toggles
# Set these to False to temporarily disable heavy subsystems without code removal.
ENABLE_OBJECT_DETECTION = True   # keep object/person boxes active
ENABLE_GAZE = True               # iris + head-pose gaze tracking enabled
ENABLE_WARNINGS = True   # keep warning pipeline active
ENABLE_CAMERA_OBSTRUCTION = False  # keep blur/covered warnings off
DISABLE_FACE_WARNINGS = False
DISABLE_VOICE_WARNINGS = False

# YOLO Settings
# YOLOv11m — single source of truth for all object/person detection.
YOLO_MODEL_PATH = os.path.join("models", "yolov11m.pt")
# Backward compatibility for legacy imports
YOLO_MODEL_NAME = YOLO_MODEL_PATH
# Run object detector every N frames to balance CPU and latency. (1 = every frame)
OBJECT_PROCESS_EVERY_N = 1
YOLO_BANNED_CLASSES = [
    43, # knife (often pen)
    63, # laptop
    64, # mouse
    65, # remote
    66, # keyboard
    67, # cell phone
    73, # book
    74, # clock / smartwatch
    76, # scissors (often pen)
    79  # toothbrush (often pen)
]
YOLO_PERSON_CLASS = 0
YOLO_PERSON_CONFIDENCE = 0.65        # tighten to reduce phantom humans (curtains/shadows)
YOLO_BANNED_CONFIDENCE = 0.45        # tuned for phones/books

# Prohibited object set and thresholds
PROHIBITED_OBJECTS = [
    "phone", "book", "paper", "usb",
    "headphones", "pen", "camera",
    "smartwatch"
]
DEFAULT_OBJECT_THRESHOLD = 0.50
OBJECT_THRESHOLDS = {
    "book": 0.40,
    "phone": 0.40,
    "paper": 0.45,
    "usb": 0.50,
    "headphones": 0.50,
    "pen": 0.55,
    "camera": 0.60,
    "smartwatch": 0.50,
}

# Area validation (ratio of bbox area to frame area)
AREA_LIMITS = {
    "book": (0.02, 0.40),      # widened for closer/bigger covers
    "paper": (0.02, 0.35),
    "phone": (0.005, 0.30),
    "usb": (0.005, 0.05),
    "pen": (0.002, 0.03),
    "smartwatch": (0.005, 0.08),
}

# Texture / edge filters to suppress flat walls and glare patches
TEXTURE_VARIANCE_THRESHOLD = 25.0   # looser to keep books/paper in low light
EDGE_DENSITY_THRESHOLD = 0.015      # looser to keep textured covers

# Book-specific heuristic thresholds (OpenCV fallback)
BOOK_MIN_AREA_RATIO = 0.02
BOOK_MAX_AREA_RATIO = 0.55
BOOK_MIN_ASPECT = 0.30     # allow portrait/landscape
BOOK_MAX_ASPECT = 1.90
BOOK_EDGE_DENSITY = 0.005  # covers often have patterns even if low contrast

# Spatial filters
OBJECT_TOP_IGNORE_RATIO = 0.05  # smaller ignore band so near-ear phones aren’t skipped
OBJECT_MIN_PIXELS = 1500        # allow smaller ear-level objects

# Temporal persistence
TEMPORAL_FRAMES_REQUIRED = 1    # fastest confirmation (pre-change)

# Optional advanced detector
GROUNDING_DINO_ENABLED = str(os.getenv("GROUNDING_DINO_ENABLED", "0")).lower() not in ("0", "false")

# Person box sanity bounds (ratios relative to frame) to keep small / partial people
PERSON_MIN_AREA_RATIO = 0.02         # ignore tiny ghost shapes in background
PERSON_MAX_AREA_RATIO = 0.95
PERSON_MIN_ASPECT = 0.20   # width / height
PERSON_MAX_ASPECT = 1.35

# Paper detection (heuristic)
PAPER_MIN_AREA_RATIO = 0.010   # require at least 1% of frame to avoid small glare specks
PAPER_MAX_AREA_RATIO = 0.70    # reject huge wall/furniture patches
PAPER_MIN_ASPECT = 0.30        # avoid super skinny false strips
PAPER_MAX_ASPECT = 2.4         # reject extremely wide banners
PAPER_BRIGHT_THRESH = 190      # brighter requirement to drop gray fabric/skin
PAPER_EDGE_DENSITY = 0.050     # higher edge density to avoid smooth shadows

# Head Pose / Yaw Thresholds
YAW_THRESHOLD_DEG = 25.0             # User requested exact 25 degrees
PITCH_THRESHOLD_DEG = 20.0           # User requested exact 20 degrees for looking up/down
NORMALIZED_OFFSET_THRESHOLD = 0.35   # Relaxed (was 0.20) for more freedom reading

# Strict Gaze Thresholds
EAR_THRESHOLD = 0.15 # Lower means eyes closed
# Trigger only when eyes are clearly off-screen so centered gaze is ignored.
IRIS_OFFSET_THRESHOLD = 0.18 # Relaxed to allow looking around screen comfortably
IRIS_OFFSET_THRESHOLD_Y = 0.45  # Relaxed to allow looking down at keyboard/bottom screen easily
GAZE_LINE_LENGTH = 160       # Pixels for eye-tracking overlay line

# Warning Aggregation Settings
WARNING_COOLDOWN_SEC = 5.0  # 5 seconds between penalty accumulations to avoid rapid-fire spikes
INSTANT_PENALTY_THRESHOLD = 3 # If >= 3 rules broken simultaneously, bypass cooldown

# Temporal Debouncing (Seconds a condition must be held before triggering a warning)
TIME_NO_FACE = 1.0  # faster no-face trigger (face warnings will be muted separately)
TIME_HEAD_TURNED = 2.5  # 2.5s sustained head-turn before warning
TIME_EYES_CLOSED = 1.0
TIME_GAZING = 2.5       # 2.5s looking away triggers warning
TIME_MULTIPLE_PERSONS = 0.4  # slight debounce to avoid one-frame spikes
TIME_BANNED_OBJECT = 0.5             # legacy time-based path (kept for compatibility)
# Frame-based debounce for banned objects (handled in detector; keep lightweight here)
BANNED_FRAMES_REQUIRED = 1

# Colors (BGR)
COLOR_WARNING = (0, 0, 255)    # Red
COLOR_NORMAL = (0, 255, 0)     # Green
COLOR_INFO = (255, 255, 0)     # Cyan/Yellow
COLOR_TEXT = (255, 255, 255)   # White
