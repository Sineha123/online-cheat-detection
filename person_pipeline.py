from ultralytics import YOLO
import config_vision as config
import threading
import re

# Initialize heavily loaded model globally, once
_global_yolo_model = YOLO(config.YOLO_MODEL_NAME)
_yolo_lock = threading.Lock()

class PersonDetector:
    def __init__(self):
        self.class_names = getattr(_global_yolo_model, 'names', {}) or {}
        self.selective_label_map = {
            # Phones (already implemented, keep as-is)
            'cellphone': 'Phone',
            'cell phone': 'Phone',
            'mobile phone': 'Phone',
            'smartphone': 'Phone',
            'phone': 'Phone',
            # Cameras
            'camera': 'Camera',
            'webcam': 'Camera',
            # Books / copies / notebooks
            'book': 'Book',
            'notebook': 'Book',
            'copy': 'Book',
            'paper': 'Book',
            'document': 'Book',
            # Headphones / earphones
            'headphone': 'Earphones',
            'headphones': 'Earphones',
            'headset': 'Earphones',
            'handsfree': 'Earphones',
            'hands free': 'Earphones',
            'head free': 'Earphones',
            'earphone': 'Earphones',
            'earphones': 'Earphones',
            'earbud': 'Earphones',
            'earbuds': 'Earphones',
            'airpods': 'Earphones'
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
        
    def process_frame(self, frame):
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
                    label = normalized_label or ("Electronic Device" if cls_id in getattr(config, "YOLO_BANNED_CLASSES", []) else None)
                    if not label:
                        continue
                    banned_objects.append({
                        "label": label,
                        "bbox": bbox
                    })

        # Heuristic paper detection (large bright rectangle)
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, getattr(config, "PAPER_BRIGHT_THRESH", 180), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area <= 1:
                    continue
                area_ratio = area / frame_area
                if area_ratio < getattr(config, "PAPER_MIN_AREA_RATIO", 0.01) or area_ratio > getattr(config, "PAPER_MAX_AREA_RATIO", 0.70):
                    continue
                x, y, ww, hh = cv2.boundingRect(cnt)
                aspect = ww / max(hh, 1)
                if aspect < getattr(config, "PAPER_MIN_ASPECT", 0.35) or aspect > getattr(config, "PAPER_MAX_ASPECT", 2.2):
                    continue
                bbox = (int(x), int(y), int(x + ww), int(y + hh), 0.35)
                paper_candidates.append(bbox)
                banned_objects.append({
                    "label": "Paper",
                    "bbox": bbox
                })
            # Edge density fallback for low-contrast paper
            edges = cv2.Canny(gray, 60, 140)
            density = float(cv2.countNonZero(edges)) / float(edges.size or 1)
            if density > getattr(config, "PAPER_EDGE_DENSITY", 0.045):
                h2, w2 = gray.shape[:2]
                bbox = (int(w2*0.05), int(h2*0.05), int(w2*0.95), int(h2*0.95), 0.3)
                banned_objects.append({"label": "Paper", "bbox": bbox})
        except Exception:
            pass
                
        return person_count, bboxes, banned_objects
