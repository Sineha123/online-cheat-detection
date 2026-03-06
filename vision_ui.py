import cv2
import config_vision as config

class UILayer:
    @staticmethod
    def draw_overlays(frame, warnings, penalty_score, bboxes, banned_objects, fps):
        """
        Draws text, HUD, and warnings onto the shared frame buffer.
        """
        h, w, _ = frame.shape
        
        # Draw Warnings
        y_offset = 60
        if warnings:
            for i, warning in enumerate(warnings):
                text = f"WARNING: {warning}"
                cv2.putText(frame, text, (20, y_offset + (i * 45)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, config.COLOR_WARNING, 4)
        else:
            cv2.putText(frame, "STATUS: ALL GOOD", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, config.COLOR_NORMAL, 4)
                        
        # Draw Penalty Score in Top Right
        score_text = f"PENALTY SCORE: {penalty_score}"
        score_color = config.COLOR_WARNING if penalty_score > 0 else config.COLOR_NORMAL
        cv2.putText(frame, score_text, (w - 400, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, score_color, 3)
                        
        # Draw Person BBoxes
        for bbox in bboxes:
            x1, y1, x2, y2, conf = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_INFO, 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, max(y1-10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_INFO, 2)
                        
        # Draw Banned Objects (Phones/Books) BBoxes
        for obj in banned_objects:
            label = obj["label"]
            x1, y1, x2, y2, conf = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_WARNING, 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1-10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_WARNING, 2)
                        
        # Draw FPS below Penalty Score
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 400, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, config.COLOR_INFO, 2)
