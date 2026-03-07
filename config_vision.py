# Configuration Constants

# Camera Config
CAMERA_ID = 0
TARGET_FPS = 30

# YOLO Settings
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_SKIP_FRAMES = 1   # run YOLO every frame for snappier detection
YOLO_BANNED_CLASSES = [
    67, # cell phone
    73  # book
]
YOLO_PERSON_CLASS = 0
YOLO_PERSON_CONFIDENCE = 0.65        # tighten to reduce phantom humans (curtains/shadows)
YOLO_BANNED_CONFIDENCE = 0.55        # tuned for phones/books
BANNED_MIN_AREA_RATIO = 0.0040       # drop tiny specs that mimic devices
BANNED_MIN_ASPECT = 0.20             # avoid super skinny noise
BANNED_MAX_ASPECT = 3.5
BANNED_FRAMES_REQUIRED = 10          # phone/book must persist for 10 frames
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
TIME_BANNED_OBJECT = 0.5             # legacy time-based path (kept for compatibility)
# Frame-based debounce for banned objects
BANNED_FRAMES_REQUIRED = 10          # ~10 consecutive frames before warning

# Colors (BGR)
COLOR_WARNING = (0, 0, 255)    # Red
COLOR_NORMAL = (0, 255, 0)     # Green
COLOR_INFO = (255, 255, 0)     # Cyan/Yellow
COLOR_TEXT = (255, 255, 255)   # White
