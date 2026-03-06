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
                    if 0.03 <= area_ratio <= 0.9 and 0.25 <= aspect <= 0.9:
                        person_count += 1
                        bboxes.append(bbox)
                else:
                    raw_label = self.class_names.get(cls_id, cls_id) if isinstance(self.class_names, dict) else cls_id
                    normalized_label = self._normalize_label(raw_label)
                    if not normalized_label or conf < config.YOLO_BANNED_CONFIDENCE:
                        continue
                    banned_objects.append({
                        "label": normalized_label,
                        "bbox": bbox
                    })
                
        return person_count, bboxes, banned_objects
