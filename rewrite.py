import sys

with open('app.py', 'r') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if line.startswith("def _run_student_frame_detection(student_id, student_name, frame):"):
        start_idx = i
    if line.startswith("def _schedule_student_frame_detection(student_id, student_name, frame):"):
        end_idx = i
        break

if start_idx == -1 or end_idx == -1:
    print(f"Error: start={start_idx}, end={end_idx}")
    sys.exit(1)

new_code = """
# --- NEW VISION ENGINE START ---
vision_face_analyzer = None
vision_person_detector = None
vision_decision_engine = None
vision_ui_layer = None

def _get_vision_engines():
    global vision_face_analyzer, vision_person_detector, vision_decision_engine, vision_ui_layer
    if vision_face_analyzer is None:
        from face_pipeline import FaceAnalyzer
        from person_pipeline import PersonDetector
        from decision_engine import DecisionEngine
        from vision_ui import UILayer
        import config_vision as config
        
        vision_face_analyzer = FaceAnalyzer()
        vision_person_detector = PersonDetector()
        vision_decision_engine = DecisionEngine()
        vision_ui_layer = UILayer
        
    return vision_face_analyzer, vision_person_detector, vision_decision_engine, vision_ui_layer

def _run_student_frame_detection(student_id, student_name, frame):
    sid = str(student_id)
    now = time.time()
    try:
        logger.debug(f"[student_frame] new vision engine sid={sid}")
        processed = frame.copy()
        
        face_analyzer, person_detector, decision_engine, UILayer = _get_vision_engines()
        import config_vision as config
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # 1. Face Analysis
        face_detected, yaw_angle, ear, iris_offset, landmarks = face_analyzer.process_frame(frame_rgb)
        
        # 2. Person & Object Detection
        person_count, bboxes, banned_objects = person_detector.process_frame(processed)
        
        # 3. Evaluate Rule Violations
        warnings, penalty_score = decision_engine.evaluate(
            sid, face_detected, person_count, yaw_angle, ear, iris_offset, banned_objects
        )
        
        # 4. Integrate with Legacy DB persistence
        for w in warnings:
            if "Face not detected" in w:
                _persist_behavior_violation(sid, student_name, "NO_FACE", "No face detected in camera viewport")
            elif "Multiple persons" in w:
                _persist_behavior_violation(sid, student_name, "MULTIPLE_FACES", "Multiple persons detected")
            elif "Head turned" in w:
                _persist_behavior_violation(sid, student_name, "DISTRACTION", "Please look at the screen (Head turned)")
            elif "Eyes not visible" in w:
                _persist_behavior_violation(sid, student_name, "EYES_CLOSED", "Eyes not visible / Looking down")
            elif "Gazing" in w:
                _persist_behavior_violation(sid, student_name, "DISTRACTION", "Gazing at another screen/paper")
                
        for obj in banned_objects:
            _persist_object_violation(sid, student_name, obj['label'], float(obj['bbox'][4]))

        # 5. Draw the unified HUD before encoding
        UILayer.draw_overlays(processed, warnings, penalty_score, bboxes, banned_objects, 0.0)
        
        # 6. Push to MJPEG live stream format
        encoded = _encode_frame_to_base64(processed)
        
        # Format the diagnostics snapshot for the legacy Admin UI
        latest_snapshot = {
            'face_detected': bool(face_detected),
            'landmarks_detected': bool(face_detected),
            'face_obscured': not face_detected,
            'face_count': int(person_count) if face_detected else 0,
            'eyes_closed': ear < config.EAR_THRESHOLD if face_detected else False,
            'eyes_closed_elapsed': 0.0,
            'looking_away': abs(yaw_angle) > config.YAW_THRESHOLD_DEG if face_detected else False,
            'looking_away_elapsed': 0.0,
            'gaze_direction': "CENTER" if abs(iris_offset) < config.IRIS_OFFSET_THRESHOLD else "AWAY",
            'no_face_elapsed': 0.0 if face_detected else 99.0,
            'multi_face_detected': person_count > 1,
            'multi_face_elapsed': 0.0,
            'pose_left_seat': False,
            'left_seat_elapsed': 0.0,
            'movement_distracted': False,
            'movement_elapsed': 0.0,
        }
        
        with latest_student_frames_lock:
            item = latest_student_frames.get(sid, {})
            item['processed_frame'] = processed
            item['processed_timestamp'] = now
            item['detections'] = [] # Deprecated, covered in overlays
            item['processed_frame_b64'] = encoded
            item['status_snapshot'] = latest_snapshot
            item['last_visible_object_labels'] = []
            item['last_prohibited_object_labels'] = [b['label'] for b in banned_objects]
            item['last_person_count'] = int(person_count)
            latest_student_frames[sid] = item
            
    except Exception as e:
        logger.error(f"Student frame worker failed for {sid}: {e}", exc_info=True)
    finally:
        next_frame = None
        with student_detection_state_lock:
            if sid not in student_detection_state:
                student_detection_state[sid] = {
                    'running': False,
                    'pending_frame': None
                }
            st = student_detection_state[sid]
            st['last_result'] = {'timestamp': time.time()}
            if st.get('pending_frame') is not None:
                next_frame = st['pending_frame']
                st['pending_frame'] = None
            else:
                st['running'] = False

        if next_frame is not None:
            executor.submit(_run_student_frame_detection, sid, student_name, next_frame)

# --- NEW VISION ENGINE END ---

"""

lines = lines[:start_idx] + [new_code] + lines[end_idx:]

with open('app.py', 'w') as f:
    f.writelines(lines)

print("Successfully replaced _run_student_frame_detection in app.py!")
