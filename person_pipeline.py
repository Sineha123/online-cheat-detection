import re
import threading
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config_vision as config

# Initialize the heavier YOLOv8m model once, globally.
try:
    _global_yolo_model = YOLO(getattr(config, "YOLO_MODEL_PATH", getattr(config, "YOLO_MODEL_NAME", "models/yolov8m.pt")))
except Exception as e:
    raise RuntimeError(
        f"Failed to load YOLO model at {getattr(config, 'YOLO_MODEL_PATH', getattr(config, 'YOLO_MODEL_NAME', 'models/yolov8m.pt'))}. "
        "Download yolov8m.pt into the models/ folder."
    ) from e

# Warm up once to avoid first-inference stall during exam start.
try:
    _ = _global_yolo_model(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
except Exception:
    pass

_yolo_lock = threading.Lock()


class PersonDetector:
    """
    Runs YOLOv8m for people + prohibited object detection and applies a cascade
    of post-filters (confidence, area, texture, edges, temporal persistence).
    """

    def __init__(self):
        self.class_names = getattr(_global_yolo_model, "names", {}) or {}
        self.prohibited_set = set(getattr(config, "PROHIBITED_OBJECTS", []))
        self.default_threshold = getattr(config, "DEFAULT_OBJECT_THRESHOLD", 0.70)
        self.history_frames_required = getattr(config, "TEMPORAL_FRAMES_REQUIRED", 3)
        self.edge_threshold = getattr(config, "EDGE_DENSITY_THRESHOLD", 0.02)
        self.texture_threshold = getattr(config, "TEXTURE_VARIANCE_THRESHOLD", 50.0)
        self.object_states: Dict[str, Dict] = {}
        self._grounding_warned = False
        self.enable_ear_heuristic = True  # ear-level heuristic for headphones

        # Map raw YOLO/COCO labels to our canonical prohibited labels.
        self.coco_to_prohibited = {
            "cell phone": "phone",
            "mobile phone": "phone",
            "smartphone": "phone",
            "phone": "phone",
            "book": "book",
            "notebook": "book",
            "paper": "paper",
            "copy": "paper",
            "document": "paper",
            "person": "person",
            "laptop": "laptop",
            "keyboard": "keyboard",
            "mouse": "mouse",
            "clock": "smartwatch",
            "watch": "smartwatch",
            "smart watch": "smartwatch",
            "headphone": "headphones",
            "headphones": "headphones",
            "headset": "headphones",
            "earphone": "headphones",
            "earphones": "headphones",
            # Misclassifications that often map to pens
            "toothbrush": "pen",
            "knife": "pen",
            "scissors": "pen",
        }

    # --- Internal helpers -------------------------------------------------
    def _state(self, sid: str):
        if sid not in self.object_states:
            self.object_states[sid] = {
                "history": {},
                "frame_idx": 0,
                "last_result": (0, [], []),
            }
        return self.object_states[sid]

    def _canonical_label(self, raw_label) -> Optional[str]:
        value = str(raw_label or "").strip().lower()
        if value in self.coco_to_prohibited:
            return self.coco_to_prohibited[value]

        # token-based contains matching
        tokens = set(re.findall(r"[a-z0-9]+", value))
        for raw, mapped in self.coco_to_prohibited.items():
            raw_tokens = set(re.findall(r"[a-z0-9]+", raw))
            if raw_tokens and raw_tokens.issubset(tokens):
                return mapped
        return None

    def _history_key(self, label: str, bbox: Tuple[float, float, float, float], frame_shape):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        gx = int((cx / max(w, 1)) * 3)  # 3x3 grid bucket
        gy = int((cy / max(h, 1)) * 3)
        return f"{label}:{gx}:{gy}"

    def _update_history(self, state, key):
        history = state["history"]
        current_frame = state["frame_idx"]
        entry = history.get(key, {"count": 0, "last_frame": current_frame - 1})
        if entry["last_frame"] == current_frame - 1:
            entry["count"] += 1
        else:
            entry["count"] = 1
        entry["last_frame"] = current_frame
        history[key] = entry

        # Clean up stale history to avoid unbounded growth
        stale_keys = [k for k, v in history.items() if current_frame - v["last_frame"] > self.history_frames_required + 2]
        for k in stale_keys:
            history.pop(k, None)

        return entry["count"] >= self.history_frames_required

    def _passes_area_limits(self, label: str, area_ratio: float):
        limits = getattr(config, "AREA_LIMITS", {}).get(label)
        if not limits:
            return True
        min_area, max_area = limits
        return min_area <= area_ratio <= max_area

    def _passes_texture(self, roi):
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.var(gray)) >= self.texture_threshold

    def _passes_edges(self, roi):
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        density = float(np.count_nonzero(edges)) / float(edges.size or 1)
        return density >= self.edge_threshold

    def _display_label(self, canonical: str) -> str:
        if canonical.lower() == "usb":
            return "USB"
        return canonical.replace("_", " ").title()

    def _maybe_accept_candidate(self, label, bbox, conf, frame, state):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        area = max((x2 - x1), 0) * max((y2 - y1), 0)
        frame_area = float(max(h * w, 1))
        area_ratio = area / frame_area

        if y1 < h * getattr(config, "OBJECT_TOP_IGNORE_RATIO", 0.0):
            return None
        if area < getattr(config, "OBJECT_MIN_PIXELS", 0):
            return None
        if not self._passes_area_limits(label, area_ratio):
            return None

        threshold = getattr(config, "OBJECT_THRESHOLDS", {}).get(label, self.default_threshold)
        if conf < threshold:
            return None

        roi = frame[max(int(y1), 0):max(int(y2), 0), max(int(x1), 0):max(int(x2), 0)]
        if not self._passes_texture(roi):
            return None
        if not self._passes_edges(roi):
            return None

        key = self._history_key(label, bbox, frame.shape)
        if not self._update_history(state, key):
            return None

        return {
            "label": self._display_label(label),
            "bbox": (int(x1), int(y1), int(x2), int(y2), float(conf)),
        }

    def _maybe_run_grounding_dino(self, frame, state):
        """
        Optional hook for GroundingDINO. Kept lightweight and opt-in.
        """
        if not getattr(config, "GROUNDING_DINO_ENABLED", False):
            return []
        if not self._grounding_warned:
            print("[GroundingDINO] GROUNDING_DINO_ENABLED=1 but runtime detector not wired; skipping.")
            self._grounding_warned = True
        return []

    # --- Public API -------------------------------------------------------
    def process_frame(self, frame, face_bbox=None, student_id=None):
        """
        Runs YOLOv8m and post-filters. Includes temporal persistence (>=3 consecutive frames).
        """
        sid = str(student_id) if student_id is not None else "__default__"
        state = self._state(sid)
        state["frame_idx"] += 1
        frame_idx = state["frame_idx"]

        # Throttle detection frequency if configured
        if frame_idx % max(getattr(config, "OBJECT_PROCESS_EVERY_N", 1), 1) != 0:
            return state["last_result"]

        person_count = 0
        banned_objects = []
        person_boxes = []

        h, w, _ = frame.shape
        frame_area = float(max(h * w, 1))
        min_area = getattr(config, "PERSON_MIN_AREA_RATIO", 0.01)
        max_area = getattr(config, "PERSON_MAX_AREA_RATIO", 0.95)
        min_aspect = getattr(config, "PERSON_MIN_ASPECT", 0.2)
        max_aspect = getattr(config, "PERSON_MAX_ASPECT", 1.35)
        paper_candidates = []

        with _yolo_lock:
            results = _global_yolo_model(frame, verbose=False)

        if len(results) > 0:
            r = results[0]
            boxes = r.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (float(x1), float(y1), float(x2), float(y2))
                area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
                aspect = (x2 - x1) / max((y2 - y1), 1)

                if cls_id == config.YOLO_PERSON_CLASS and conf >= config.YOLO_PERSON_CONFIDENCE:
                    if min_area <= area_ratio <= max_area and min_aspect <= aspect <= max_aspect:
                        person_count += 1
                        person_boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
                    continue

                raw_label = self.class_names.get(cls_id, cls_id) if isinstance(self.class_names, dict) else cls_id
                canonical = self._canonical_label(raw_label)
                if not canonical or canonical not in self.prohibited_set:
                    continue

                candidate = self._maybe_accept_candidate(canonical, bbox, conf, frame, state)
                if candidate:
                    banned_objects.append(candidate)

        # Heuristic paper detection (large bright rectangle + edge density)
        try:
            def _bbox_iou(a, b):
                ax1, ay1, ax2, ay2, _ = a
                bx1, by1, bx2, by2, _ = b
                inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
                inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    return 0.0
                inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area_a = (ax2 - ax1) * (ay2 - ay1)
                area_b = (bx2 - bx1) * (by2 - by1)
                denom = float(area_a + area_b - inter) or 1.0
                return inter / denom

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            bright_thresh = getattr(config, "PAPER_BRIGHT_THRESH", 190)
            _, bright_mask = cv2.threshold(l_channel, bright_thresh, 255, cv2.THRESH_BINARY)
            blur = cv2.GaussianBlur(bright_mask, (5, 5), 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(blur, kernel, iterations=2)
            edges = cv2.Canny(closed, 50, 140)
            combined = cv2.bitwise_or(closed, edges)

            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area <= 1:
                    continue
                area_ratio = area / frame_area
                if area_ratio < getattr(config, "PAPER_MIN_AREA_RATIO", 0.010) or area_ratio > getattr(config, "PAPER_MAX_AREA_RATIO", 0.70):
                    continue

                rect = cv2.minAreaRect(cnt)
                (cw, ch) = rect[1]
                if cw == 0 or ch == 0:
                    continue
                aspect_rot = max(cw, ch) / max(min(cw, ch), 1)

                x, y, ww, hh = cv2.boundingRect(cnt)
                aspect = ww / max(hh, 1)
                min_aspect = getattr(config, "PAPER_MIN_ASPECT", 0.30)
                max_aspect = getattr(config, "PAPER_MAX_ASPECT", 2.4)
                if aspect < min_aspect and aspect_rot < min_aspect:
                    continue
                if aspect > max_aspect and aspect_rot > max_aspect:
                    continue

                region_mean = 0.0
                if hh > 0 and ww > 0:
                    region_mean = cv2.mean(l_channel[y:y + hh, x:x + ww])[0]
                if region_mean < (bright_thresh - 10):
                    continue

                bbox_paper = (float(x), float(y), float(x + ww), float(y + hh), 0.65)
                if any(_bbox_iou(bbox_paper, existing) > 0.6 for existing in paper_candidates):
                    continue
                paper_candidates.append(bbox_paper)

                candidate = self._maybe_accept_candidate("paper", bbox_paper[:4], bbox_paper[4], frame, state)
                if candidate:
                    banned_objects.append(candidate)

            # Edge density fallback for low-contrast paper filling most of the view
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges_lo = cv2.Canny(gray, 60, 150)
            density = float(cv2.countNonZero(edges_lo)) / float(edges_lo.size or 1)
            if density > getattr(config, "PAPER_EDGE_DENSITY", 0.050):
                h2, w2 = gray.shape[:2]
                bbox_paper = (float(w2 * 0.04), float(h2 * 0.04), float(w2 * 0.96), float(h2 * 0.96), 0.60)
                if not any(_bbox_iou(bbox_paper, existing) > 0.6 for existing in paper_candidates):
                    candidate = self._maybe_accept_candidate("paper", bbox_paper[:4], bbox_paper[4], frame, state)
                    if candidate:
                        banned_objects.append(candidate)

            # --- Book heuristic (covers, not necessarily bright) ---
            try:
                edges_book = cv2.Canny(cv2.GaussianBlur(frame, (5,5), 0), 60, 150)
                contours_book, _ = cv2.findContours(edges_book, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours_book:
                    area = cv2.contourArea(cnt)
                    if area <= 10:
                        continue
                    area_ratio = area / frame_area
                    if area_ratio < getattr(config, "BOOK_MIN_AREA_RATIO", 0.04) or area_ratio > getattr(config, "BOOK_MAX_AREA_RATIO", 0.55):
                        continue
                    x, y, ww, hh = cv2.boundingRect(cnt)
                    aspect = ww / max(hh, 1)
                    if aspect < getattr(config, "BOOK_MIN_ASPECT", 0.35) or aspect > getattr(config, "BOOK_MAX_ASPECT", 1.80):
                        continue
                    roi = frame[y:y+hh, x:x+ww]
                    edge_roi = edges_book[y:y+hh, x:x+ww]
                    edge_density = float(cv2.countNonZero(edge_roi)) / float(edge_roi.size or 1)
                    if edge_density < getattr(config, "BOOK_EDGE_DENSITY", 0.010):
                        continue
                    bbox_book = (float(x), float(y), float(x + ww), float(y + hh), 0.60)
                    candidate = self._maybe_accept_candidate("book", bbox_book[:4], bbox_book[4], frame, state)
                    if candidate:
                        banned_objects.append(candidate)
            except Exception:
                pass
        except Exception:
            pass

        # Optional secondary detector (stubbed) for earbuds/usb/wires/cheat sheets
        try:
            banned_objects.extend(self._maybe_run_grounding_dino(frame, state))
        except Exception:
            pass

        state["last_result"] = (person_count, person_boxes, banned_objects)
        return state["last_result"]
