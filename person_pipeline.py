from ultralytics import YOLO
import config_vision as config
import threading

# Initialize heavily loaded model globally, once
_global_yolo_model = YOLO(config.YOLO_MODEL_NAME)
_yolo_lock = threading.Lock()

class PersonDetector:
    def __init__(self):
        # Mapping COCO class IDs to readable names
        self.class_names = {
            67: "Phone",
            73: "Book",
            77: "Laptop"
        }
        
    def process_frame(self, frame):
        """
        Runs YOLOv8 person detection and banned object detection on the frame.
        Stateless: Safe for concurrent threads.
        """
        classes_to_detect = [config.YOLO_PERSON_CLASS] + config.YOLO_BANNED_CLASSES
        
        with _yolo_lock:
            results = _global_yolo_model(frame, verbose=False, classes=classes_to_detect)
        
        person_count = 0
        banned_objects = []
        bboxes = []
        
        if len(results) > 0:
            r = results[0]
            boxes = r.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2), conf)
                
                if cls_id == config.YOLO_PERSON_CLASS:
                    person_count += 1
                    bboxes.append(bbox)
                elif cls_id in self.class_names:
                    banned_objects.append({
                        "label": self.class_names[cls_id],
                        "bbox": bbox
                    })
                
        return person_count, bboxes, banned_objects
