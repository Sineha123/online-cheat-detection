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
  "remote","keyboard","cell phone","microwave","oven","toaster","sink",
  "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
];

// ── Banned & Accessory Labels ───────────────────────────
export const BANNED_LABELS = new Set([
  "cell phone","book","laptop","tablet","remote","mouse","keyboard"
]);

export const ACCESSORY_LABELS = new Set([
  "headphone","headphones","headset","earphone","earphones","earbud","earbuds","wire","cable"
]);

export const MONITORED_OBJECT_LABELS = Array.from(new Set([...BANNED_LABELS, ...ACCESSORY_LABELS]));

// ── Per-class Confidence Thresholds ─────────────────────
export const CLASS_CONF_THRESHOLDS = {
  person: 0.50,
  "cell phone": 0.32, book: 0.34, laptop: 0.35, tablet: 0.34,
  remote: 0.34, mouse: 0.34, keyboard: 0.34,
  headphone: 0.35, headphones: 0.35, headset: 0.35,
  earphone: 0.33, earphones: 0.33, earbud: 0.33, earbuds: 0.33,
  wire: 0.32, cable: 0.32
};

// ── Minimum Area Ratio (bbox area / frame area) ─────────
export const MIN_AREA_RATIO_BY_LABEL = {
  person: 0.01,
  "cell phone": 6e-4, book: 24e-4, laptop: 6e-3, tablet: 32e-4,
  remote: 8e-4, mouse: 8e-4, keyboard: 31e-4,
  headphone: 18e-4, headphones: 18e-4, headset: 18e-4,
  earphone: 8e-4, earphones: 8e-4, earbud: 7e-4, earbuds: 7e-4,
  wire: 35e-5, cable: 35e-5
};

// ── Minimum Short-side Pixels ───────────────────────────
export const MIN_SHORT_SIDE_PX_BY_LABEL = {
  person: 40,
  "cell phone": 10, book: 22, laptop: 28, tablet: 24,
  remote: 11, mouse: 10, keyboard: 22,
  headphone: 16, headphones: 16, headset: 16,
  earphone: 9, earphones: 9, earbud: 8, earbuds: 8,
  wire: 6, cable: 6
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
  wire:      0.58,
  earphone:  0.62,
  headphone: 0.67
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
  wire_risk:            14,
  earphone_risk:        18,
  headphone_risk:       16,
  no_face_risk:         35,    // CRITICAL — hiding face
  multi_face_risk:      25,    // multiple faces
  safety_good_threshold: 72    // safetyLevel >= this → GOOD_TO_GO
};
