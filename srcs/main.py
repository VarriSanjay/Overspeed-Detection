from ultralytics import YOLO
import easyocr
import cv2
import re
import numpy as np
import csv
import os
from datetime import datetime
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort

# ─── Load Models ───────────────────────────────────────────────
vehicle_model = YOLO("yolov8n.pt")
plate_model   = YOLO(r"C:\ML\runs\detect\number_plate_model3\weights\best.pt")
reader        = easyocr.Reader(['en'])
tracker       = DeepSort(max_age=30)

# ─── Settings ──────────────────────────────────────────────────
SPEED_LIMIT          = 60
VEHICLE_CLASSES      = [2, 3, 5, 7]
METERS_PER_PIXEL     = 0.05
FPS_FALLBACK         = 30
SPEED_HISTORY_LEN    = 10
PLATE_DETECT_EVERY   = 2       
OCR_CONF_THRESHOLD   = 0.2    
PRUNE_AFTER          = 90

CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ─── CSV Setup ─────────────────────────────────────────────────
CSV_FILE    = "overspeeding_plates.csv"
logged_ids  = set()

file_exists = os.path.isfile(CSV_FILE)
csv_file    = open(CSV_FILE, 'a', newline='', encoding='utf-8')
csv_writer  = csv.writer(csv_file)
if not file_exists:
    csv_writer.writerow(["Timestamp", "Track_ID", "Vehicle_Type",
                          "License_Plate", "Speed_kmh"])

def log_violation(track_id, vehicle_type, plate, speed):
    if track_id in logged_ids:
        return
    csv_writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        track_id, vehicle_type, plate, round(speed, 1)
    ])
    csv_file.flush()
    logged_ids.add(track_id)
    print(f"[CSV] ID:{track_id} | {vehicle_type} | Plate:{plate} | {round(speed,1)} km/h")

# ─── Helper Functions ───────────────────────────────────────────
def clean_plate(text):
    return re.sub(r'[^A-Z0-9\-]', '', text.upper())

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def estimate_speed(prev_center, curr_center, fps, mpp):
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    return np.sqrt(dx**2 + dy**2) * mpp * fps * 3.6

# ─── State ─────────────────────────────────────────────────────
prev_positions  = {}
speed_histories = defaultdict(lambda: deque(maxlen=SPEED_HISTORY_LEN))
plate_texts     = {}
vehicle_types   = {}
last_seen       = {}

# ─── Video I/O ─────────────────────────────────────────────────
cap = cv2.VideoCapture(r"C:\ML\car.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
w, h = int(cap.get(3)), int(cap.get(4))

out  = cv2.VideoWriter("overspeeding_output.mp4",
                        cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_count = 0

# ─── Main Loop ─────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Step 1: Detect vehicles
    vehicle_results = vehicle_model.predict(
        frame, classes=VEHICLE_CLASSES, conf=0.4, verbose=False
    )

    detections = []
    for result in vehicle_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2-x1, y2-y1],
                                float(box.conf[0]), int(box.cls[0])))

    # Step 2: Track vehicles
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center = get_center((x1, y1, x2, y2))
        last_seen[track_id] = frame_count

        # Vehicle type
        det_class = getattr(track, 'det_class', None)
        if det_class is not None:
            vehicle_types[track_id] = CLASS_NAMES.get(int(det_class), "Vehicle")
        vehicle_type = vehicle_types.get(track_id, "Vehicle")

        # Step 3: Speed estimation
        if track_id in prev_positions:
            raw_speed = estimate_speed(prev_positions[track_id], center, fps, METERS_PER_PIXEL)
            speed_histories[track_id].append(raw_speed)

        prev_positions[track_id] = center
        history = speed_histories[track_id]
        speed   = round(float(np.mean(history)) if history else 0.0, 1)
        is_overspeeding = speed > SPEED_LIMIT

        #  Step 4: Plate detection (ALWAYS RUNS)
        if frame_count % PLATE_DETECT_EVERY == 0:
            crop = frame[max(0,y1):y2, max(0,x1):x2]

            if crop.shape[0] > 10 and crop.shape[1] > 10:
                for pr in plate_model.predict(crop, conf=0.3, verbose=False):
                    for pbox in pr.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                        plate_crop = crop[py1:py2, px1:px2]

                        if plate_crop.shape[0] < 10 or plate_crop.shape[1] < 10:
                            continue

                        # OCR
                        text = ""
                        for (_, t, c) in reader.readtext(plate_crop):
                            if c > OCR_CONF_THRESHOLD:
                                text += t + " "

                        cleaned = clean_plate(text)

                        if cleaned:
                            plate_texts[track_id] = cleaned
                            print(f"[OCR] ID:{track_id} Plate:{cleaned}")

        # Step 5: Log violation
        plate = plate_texts.get(track_id, "")
        if is_overspeeding and plate:
            log_violation(track_id, vehicle_type, plate, speed)

        # Step 6: Draw annotations
        plate_label = plate_texts.get(track_id, "---")
        color = (0, 0, 255) if is_overspeeding else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {vehicle_type} {speed}km/h",
                    (x1, max(0, y1-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Plate: {plate_label}",
                    (x1, max(0, y1-5)),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if is_overspeeding:
            cv2.putText(frame, "OVERSPEEDING!",
                        (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Cleanup old tracks
    for tid in [t for t, f in last_seen.items() if frame_count - f > PRUNE_AFTER]:
        for d in [prev_positions, speed_histories, plate_texts,
                  vehicle_types, last_seen]:
            d.pop(tid, None)

    # HUD
    cv2.putText(frame, f"Speed Limit: {SPEED_LIMIT} km/h",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Overspeeding Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── Cleanup ───────────────────────────────────────────────────
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print("\nDone! Video → overspeeding_output.mp4")
print(f"CSV → {CSV_FILE} ({len(logged_ids)} violations logged)")