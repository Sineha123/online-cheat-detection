/**
 * ═══════════════════════════════════════════════════════════
 *  PROCTOR DETECTION CONFIG  —  v1.0 LOCKED
 * ═══════════════════════════════════════════════════════════
 *
 *  Central configuration for the entire detection pipeline.
 *  Every threshold, weight, and label list lives here.
 *  Both student_engine.js and admin_verifier.js import from
 *  this single source of truth.
 *
 *  To recalibrate: edit ONLY this file.
 * ═══════════════════════════════════════════════════════════
 */

// ── YOLO Model ──────────────────────────────────────────
export const YOLO_INPUT_SIZE = 640;

export const COCO_CLASSES = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
  "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
  "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
  "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
  "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
  "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
  "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
  "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
];

// ── Banned & Accessory Labels ───────────────────────────
export const BANNED_LABELS = new Set([
  'cell phone',
  'laptop',
  'book',       // Paper/notes
  'clock',      // Smartwatch
  'mouse',      // Wired objects
  'keyboard',   // Wired objects
  'remote',     // Clickers/wired devices
  'tablet'
]);

export const ACCESSORY_LABELS = new Set([
  "headphone","headphones","headset","earphone","earphones","earbud","earbuds","wire","cable"
]);

export const MONITORED_OBJECT_LABELS = Array.from(new Set([...BANNED_LABELS, ...ACCESSORY_LABELS]));

// ── Per-class Confidence Thresholds ─────────────────────
export const CLASS_CONF_THRESHOLDS = {
  person: 0.50,
  'cell phone': 0.15,
  'laptop': 0.20,
  'book': 0.15,
  'clock': 0.20,
  'remote': 0.15,
  'mouse': 0.15,
  'keyboard': 0.15,
  'tablet': 0.28, // Kept original tablet value as it wasn't in the instruction's list of changes
  headphone: 2.0, // Disabled
  headphones: 2.0, // Disabled
  headset: 2.0, // Disabled
  earphone: 2.0, earphones: 2.0, earbud: 2.0, earbuds: 2.0,
  wire: 2.0, // Disabled
  cable: 2.0 // Disabled
};

// ── Minimum Area Ratio (bbox area / frame area) ─────────
export const MIN_AREA_RATIO_BY_LABEL = {
  person: 0.01,
  "cell phone": 4e-4, book: 15e-4, laptop: 4e-3, tablet: 20e-4,
  remote: 5e-4, mouse: 5e-4, keyboard: 20e-4,
  headphone: 22e-4, headphones: 22e-4, headset: 22e-4,
  earphone: 6e-4, earphones: 6e-4, earbud: 5e-4, earbuds: 5e-4,
  wire: 20e-5, cable: 20e-5
};

// ── Minimum Short-side Pixels ───────────────────────────
export const MIN_SHORT_SIDE_PX_BY_LABEL = {
  person: 40,
  "cell phone": 8, book: 15, laptop: 20, tablet: 18,
  remote: 8, mouse: 8, keyboard: 15,
  headphone: 20, headphones: 20, headset: 20,
  earphone: 7, earphones: 7, earbud: 6, earbuds: 6,
  wire: 4, cable: 4
};

// ── Face Landmark Indices ───────────────────────────────
export const LEFT_EYE  = [33, 160, 158, 133, 153, 144];
export const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

// ── Temporal Stabilization ──────────────────────────────
export const OBJECT_STABLE_FRAMES    = 2;
export const OBJECT_EMA_ALPHA        = 0.34;
export const OBJECT_EMA_DECAY        = 0.78;
export const OBJECT_HIGH_CONF_MARGIN = 0.20;
export const ACCESSORY_STABLE_FRAMES = 2;
export const ACCESSORY_EMA_ALPHA     = 0.35;
export const LIGHTING_EMA_ALPHA      = 0.28;
export const LIGHTING_MIN_SCORE      = 0.52;

// ── Accessory Heuristic Thresholds ──────────────────────
export const ACCESSORY_SCORE_THRESHOLDS = {
  wire:      2.0,
  earphone:  2.0,
  headphone: 2.0
};

// ── Evaluation: Risk Weights (evaluateRealtime) ─────────
export const EVAL = {
  yaw_threshold:        24,    // degrees — head yaw off-axis
  yaw_risk:             10,
  pitch_threshold:      18,    // degrees — head pitch off-axis
  pitch_risk:            9,
  roll_threshold:       15,    // degrees — head roll tilt
  roll_risk:             6,
  gaze_yaw_threshold:   22,    // degrees — eye gaze horizontal
  gaze_yaw_risk:        11,
  gaze_pitch_threshold: 16,    // degrees — eye gaze vertical
  gaze_pitch_risk:       9,
  bad_lighting_risk:    12,
  wire_risk:            0,
  earphone_risk:        0,
  headphone_risk:       0,
  no_face_risk:         35,    // CRITICAL — hiding face
  multi_face_risk:      25,    // multiple faces
  safety_good_threshold: 72    // safetyLevel >= this → GOOD_TO_GO
};
