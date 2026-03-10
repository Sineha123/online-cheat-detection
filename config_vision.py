# Configuration Constants

# Camera Config
CAMERA_ID = 0
TARGET_FPS = 30

# YOLO Settings
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_SKIP_FRAMES = 1   # run YOLO every frame for snappier detection
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
TIME_NO_FACE = 3.5  # Face block tolerance (~3-4s)
TIME_HEAD_TURNED = 4.5  # Set to looser than gaze
TIME_EYES_CLOSED = 1.0
TIME_GAZING = 3.5       # User requested 3-4s for staring away
TIME_MULTIPLE_PERSONS = 0.4
TIME_BANNED_OBJECT = 0.5             # legacy time-based path (kept for compatibility)
# Frame-based debounce for banned objects
BANNED_FRAMES_REQUIRED = 10          # ~10 consecutive frames before warning

# Colors (BGR)
COLOR_WARNING = (0, 0, 255)    # Red
COLOR_NORMAL = (0, 255, 0)     # Green
COLOR_INFO = (255, 255, 0)     # Cyan/Yellow
COLOR_TEXT = (255, 255, 255)   # White
