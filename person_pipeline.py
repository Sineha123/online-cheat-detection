from ultralytics import YOLO
import config_vision as config
import threading
import re

# Initialize heavily loaded model globally, once
_global_yolo_model = YOLO(config.YOLO_MODEL_NAME)
# Warm up once to avoid first-inference stall during exam start
try:
    import numpy as _np
    _ = _global_yolo_model(_np.zeros((320, 320, 3), dtype=_np.uint8), verbose=False)
except Exception:
    pass
_yolo_lock = threading.Lock()

class PersonDetector:
    def __init__(self):
        self.class_names = getattr(_global_yolo_model, 'names', {}) or {}
        self.selective_label_map = {
            # Phones â†’ treat as generic electronic device
            'cellphone': 'Mobile Phone',
            'cell phone': 'Mobile Phone',
            'mobile phone': 'Mobile Phone',
            'smartphone': 'Mobile Phone',
            'phone': 'Mobile Phone',
            # Books / copies / notebooks
            'book': 'Book',
            'notebook': 'Book',
            'copy': 'Document',
            'paper': 'Document',
            'document': 'Document',
        }

    def _normalize_label(self, label):
        value = str(label or '').strip().lower()
        direct = self.selective_label_map.get(value)
        if direct:
            return direct
        tokens = set(re.findall(r'[a-z0-9]+', value))
        for raw, mapped in self.selective_label_map.items():
            raw_tokens = set(re.findall(r'[a-z0-9]+', raw))
            if raw_tokens and raw_tokens.issubset(tokens):
                return mapped
        return None
        
    def process_frame(self, frame, face_bbox=None):
        """
        Runs YOLOv8 person detection and banned object detection on the frame.
        Stateless: Safe for concurrent threads.
        """
        with _yolo_lock:
            results = _global_yolo_model(frame, verbose=False)
        
        person_count = 0
        banned_objects = []
        bboxes = []
        h, w, _ = frame.shape
        frame_area = float(max(h * w, 1))
        min_area = getattr(config, "PERSON_MIN_AREA_RATIO", 0.01)
        max_area = getattr(config, "PERSON_MAX_AREA_RATIO", 0.95)
        min_aspect = getattr(config, "PERSON_MIN_ASPECT", 0.2)
        max_aspect = getattr(config, "PERSON_MAX_ASPECT", 1.35)
        paper_candidates = []
        
        if len(results) > 0:
            r = results[0]
            boxes = r.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2), conf)
                area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
                aspect = (x2 - x1) / max((y2 - y1), 1)

                if cls_id == config.YOLO_PERSON_CLASS and conf >= config.YOLO_PERSON_CONFIDENCE:
                    # Heuristic filters to avoid phantom second-person from objects/books
                    if min_area <= area_ratio <= max_area and min_aspect <= aspect <= max_aspect:
                        person_count += 1
                        bboxes.append(bbox)
                else:
                    raw_label = self.class_names.get(cls_id, cls_id) if isinstance(self.class_names, dict) else cls_id
                    normalized_label = self._normalize_label(raw_label)
                    if conf < config.YOLO_BANNED_CONFIDENCE:
                        continue
                    if area_ratio < getattr(config, "BANNED_MIN_AREA_RATIO", 0.0025):
                        continue
                    if aspect < getattr(config, "BANNED_MIN_ASPECT", 0.15) or aspect > getattr(config, "BANNED_MAX_ASPECT", 4.0):
                        continue
                    label = normalized_label or ("Mobile Phone" if cls_id in getattr(config, "YOLO_BANNED_CLASSES", []) else None)
                    if not label:
                        continue
                    banned_objects.append({
                        "label": label,
                        "bbox": bbox
                    })

        # Heuristic paper detection (large bright rectangle + edge density)
        try:
            import cv2

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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]

            bright_thresh = getattr(config, "PAPER_BRIGHT_THRESH", 190)
            # Combine brightness mask + edges to catch bright or low-texture paper
            _, bright_mask = cv2.threshold(l_channel, bright_thresh, 255, cv2.THRESH_BINARY)
            blur = cv2.GaussianBlur(bright_mask, (5, 5), 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=2)
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

                # Require region to actually be bright to avoid walls/backgrounds
                region_mean = 0.0
                if hh > 0 and ww > 0:
                    region_mean = cv2.mean(l_channel[y:y+hh, x:x+ww])[0]
                if region_mean < (bright_thresh - 10):
                    continue

                bbox = (int(x), int(y), int(x + ww), int(y + hh), 0.65)
                # Skip duplicates that overlap with an existing paper box
                if any(_bbox_iou(bbox, existing) > 0.6 for existing in paper_candidates):
                    continue
                paper_candidates.append(bbox)
                banned_objects.append({
                    "label": "Paper",
                    "bbox": bbox
                })

            # Edge density fallback for low-contrast paper filling most of the view
            edges_lo = cv2.Canny(gray, 60, 150)
            density = float(cv2.countNonZero(edges_lo)) / float(edges_lo.size or 1)
            if density > getattr(config, "PAPER_EDGE_DENSITY", 0.050):
                h2, w2 = gray.shape[:2]
                bbox = (int(w2 * 0.04), int(h2 * 0.04), int(w2 * 0.96), int(h2 * 0.96), 0.60)
                if not any(_bbox_iou(bbox, existing) > 0.6 for existing in paper_candidates):
                    banned_objects.append({"label": "Paper", "bbox": bbox})

        except Exception:
            pass
                
        return person_count, bboxes, banned_objects
