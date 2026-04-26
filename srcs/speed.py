from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort

# load models
model = YOLO('yolov8n.pt')
tracker = Sort()

# load video
cap = cv2.VideoCapture('car.mp4')

# dictionary to store previous positions
prev_positions = {}
speeds = {}

fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []

    # run YOLO
    results = model(frame)[0]

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        class_id = int(class_id)

        # filter vehicles + confidence
        if score > 0.5 and class_id in [2, 3, 5, 7]:
            detections.append([x1, y1, x2, y2, score])

    # convert to numpy array
    detections = np.asarray(detections)

    # tracking
    tracks = tracker.update(detections)

    for track in tracks:
        x1, y1, x2, y2, track_id = track
        track_id = int(track_id)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # calculate speed
        if track_id in prev_positions:
            prev_cx, prev_cy = prev_positions[track_id]

            distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            speed = distance * fps   # pixels per second

            speeds[track_id] = speed

        prev_positions[track_id] = (cx, cy)

        # draw box
        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0), 2)

        # display speed
        speed_text = f"ID {track_id} | Speed: {speeds.get(track_id, 0):.1f}"

        cv2.putText(frame,
                    speed_text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    cv2.imshow("Tracking + Speed", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()