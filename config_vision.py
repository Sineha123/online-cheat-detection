# Configuration Constants

# Camera Config
CAMERA_ID = 0
TARGET_FPS = 30

# YOLO Settings
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_SKIP_FRAMES = 1   # run YOLO every frame for snappier detection
YOLO_BANNED_CLASSES = [
    67, # cell phone
    73, # book
    77  # laptop
]
YOLO_PERSON_CLASS = 0
YOLO_PERSON_CONFIDENCE = 0.55
YOLO_BANNED_CONFIDENCE = 0.18
# Person box sanity bounds (ratios relative to frame) to keep small / partial people
PERSON_MIN_AREA_RATIO = 0.01
PERSON_MAX_AREA_RATIO = 0.95
PERSON_MIN_ASPECT = 0.20   # width / height
PERSON_MAX_ASPECT = 1.35

# Paper detection (heuristic)
PAPER_MIN_AREA_RATIO = 0.003   # >=0.3% of frame (captures small/angled sheets)
PAPER_MAX_AREA_RATIO = 0.85
PAPER_MIN_ASPECT = 0.20        # allow tall/narrow edge-on sheets
PAPER_MAX_ASPECT = 3.0
PAPER_BRIGHT_THRESH = 170      # grayscale/L-channel threshold (more tolerant)
PAPER_EDGE_DENSITY = 0.030     # fallback edge-density threshold (more sensitive)

# Head Pose / Yaw Thresholds
YAW_THRESHOLD_DEG = 25.0
NORMALIZED_OFFSET_THRESHOLD = 0.20  # ~25 degrees

# Strict Gaze Thresholds
EAR_THRESHOLD = 0.15 # Lower means eyes closed
# Trigger only when eyes are clearly off-screen so centered gaze is ignored.
IRIS_OFFSET_THRESHOLD = 0.07 # How far the iris can deviate from center of eye
IRIS_OFFSET_THRESHOLD_Y = 0.20  # Higher tolerance vertically to avoid false "up"
GAZE_LINE_LENGTH = 160       # Pixels for eye-tracking overlay line

# Warning Aggregation Settings
WARNING_COOLDOWN_SEC = 3.0
INSTANT_PENALTY_THRESHOLD = 3 # If >= 3 rules broken simultaneously, bypass cooldown

# Temporal Debouncing (Seconds a condition must be held before triggering a warning)
TIME_NO_FACE = 1.5  # Faster response to missing face
TIME_HEAD_TURNED = 2.0
TIME_EYES_CLOSED = 1.0
TIME_GAZING = 0.8
TIME_MULTIPLE_PERSONS = 0.4
TIME_BANNED_OBJECT = 0.5

# Colors (BGR)
COLOR_WARNING = (0, 0, 255)    # Red
COLOR_NORMAL = (0, 255, 0)     # Green
COLOR_INFO = (255, 255, 0)     # Cyan/Yellow
COLOR_TEXT = (255, 255, 255)   # White
