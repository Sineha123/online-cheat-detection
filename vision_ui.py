import cv2
import config_vision as config

class UILayer:
    @staticmethod
    def draw_overlays(frame, warnings, penalty_score, bboxes, banned_objects, fps, landmarks=None, iris_offset=0.0):
        """
        Draws text, HUD, and warnings onto the shared frame buffer.
        """
        h, w, _ = frame.shape
        
        # Draw minimal Warnings (red only)
        y_offset = 50
        if warnings:
            for i, warning in enumerate(warnings):
                text = f"WARNING: {warning}"
                cv2.putText(frame, text, (20, y_offset + (i * 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, config.COLOR_WARNING, 3)
        else:
            cv2.putText(frame, "STATUS: OK", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, config.COLOR_NORMAL, 3)

        # Draw Penalty Score in Top Right (red only)
        score_text = f"PENALTY SCORE: {penalty_score}"
        cv2.putText(frame, score_text, (w - 320, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, config.COLOR_WARNING, 3)
                        
        # Draw Person BBoxes
        for bbox in bboxes:
            x1, y1, x2, y2, conf = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_WARNING, 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, max(y1-10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WARNING, 2)
                        
        # Draw Banned Objects (Phones/Books) BBoxes
        for obj in banned_objects:
            label = obj["label"]
            x1, y1, x2, y2, conf = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_WARNING, 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1-10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_WARNING, 2)

        # Eye-tracking overlay
        if landmarks:
            try:
                left_eye = landmarks[33]
                right_eye = landmarks[263]
                eye_center = (
                    int((left_eye.x + right_eye.x) / 2 * w),
                    int((left_eye.y + right_eye.y) / 2 * h)
                )
                # Eye boxes
                left_idxs = [33, 133, 160, 159, 158, 144, 153, 154, 155, 173]
                right_idxs = [263, 362, 385, 380, 381, 382, 386, 374, 373, 390]
                def eye_box(idxs):
                    xs = [landmarks[i].x * w for i in idxs]
                    ys = [landmarks[i].y * h for i in idxs]
                    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                l_box = eye_box(left_idxs)
                r_box = eye_box(right_idxs)
                for (x1, y1, x2, y2) in (l_box, r_box):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 1)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.line(frame, (x1, cy), (x2, cy), (0, 255, 0), 1)
                    cv2.line(frame, (cx, y1), (cx, y2), (0, 255, 0), 1)

                direction = 1 if iris_offset >= 0 else -1
                magnitude = abs(iris_offset)
                length = int(max(magnitude * config.GAZE_LINE_LENGTH, 20))
                end_point = (eye_center[0] + direction * length, eye_center[1])

                color = config.COLOR_NORMAL
                thickness = 2
                if magnitude > config.IRIS_OFFSET_THRESHOLD:
                    color = config.COLOR_WARNING
                    thickness = 3
                    cv2.arrowedLine(frame, eye_center, end_point, color, thickness, tipLength=0.3)
                else:
                    # Draw a small calming cross when gaze is on-screen
                    cv2.line(frame, (eye_center[0] - 10, eye_center[1]), (eye_center[0] + 10, eye_center[1]), color, 2)
                    cv2.line(frame, (eye_center[0], eye_center[1] - 6), (eye_center[0], eye_center[1] + 6), color, 2)
            except Exception:
                pass
                        
        # Draw FPS below Penalty Score
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 320, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COLOR_WARNING, 2)
