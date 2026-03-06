# Configuration Constants

# Camera Config
CAMERA_ID = 0
TARGET_FPS = 30

# YOLO Settings
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_SKIP_FRAMES = 10  # run YOLO every N frames
YOLO_BANNED_CLASSES = [
    67, # cell phone
    73, # book
    77  # laptop
]
YOLO_PERSON_CLASS = 0

# Head Pose / Yaw Thresholds
YAW_THRESHOLD_DEG = 25.0
NORMALIZED_OFFSET_THRESHOLD = 0.20  # ~25 degrees

# Strict Gaze Thresholds
EAR_THRESHOLD = 0.15 # Lower means eyes closed
IRIS_OFFSET_THRESHOLD = 0.04 # How far the iris can deviate from center of eye

# Warning Aggregation Settings
WARNING_COOLDOWN_SEC = 3.0
INSTANT_PENALTY_THRESHOLD = 3 # If >= 3 rules broken simultaneously, bypass cooldown

# Temporal Debouncing (Seconds a condition must be held before triggering a warning)
TIME_NO_FACE = 3.0  # Must be missing face for 3 full seconds before warning
TIME_HEAD_TURNED = 2.0
TIME_EYES_CLOSED = 1.0
TIME_GAZING = 2.0
TIME_MULTIPLE_PERSONS = 0.5
TIME_BANNED_OBJECT = 1.0

# Colors (BGR)
COLOR_WARNING = (0, 0, 255)    # Red
COLOR_NORMAL = (0, 255, 0)     # Green
COLOR_INFO = (255, 255, 0)     # Cyan/Yellow
COLOR_TEXT = (255, 255, 255)   # White
