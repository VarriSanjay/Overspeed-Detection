from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Load models
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# Video input
cap = cv2.VideoCapture("ocrtest.mp4")

class_names = model.names

frame_count = 0
saved_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # YOLO detection
    results = model.predict(frame, classes=[2,3,5,7], conf=0.4, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append(([x1, y1, w, h], conf, cls))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        label = f"ID {track_id}"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)

        # BIG TEXT
        cv2.putText(frame, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,255,255), 3)

    # 🔥 Save ONLY 3 frames spaced apart
    if frame_count % 30 == 0 and saved_frames < 3:
        cv2.imwrite(f"tracking_frame_{saved_frames+1}.jpg", frame)
        print(f"Saved tracking_frame_{saved_frames+1}.jpg")
        saved_frames += 1

    if saved_frames == 3:
        break

cap.release()
cv2.destroyAllWindows()

print("Done! 3 frames saved for PPT.")