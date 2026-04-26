from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Load image
image_path = "test_image.jpg"
frame = cv2.imread(image_path)
h_img, w_img, _ = frame.shape

# Dynamic scaling
font_scale = max(1.5, min(w_img, h_img) / 500)
thickness = int(font_scale * 3)

results = model.predict(frame, classes=[2, 3, 5, 7], conf=0.4)
class_names = model.names
output = frame.copy()

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_name = class_names[int(box.cls[0])]

        # ── Two-line label ──────────────────────────────────────────────
        line1 = f"{cls_name}  {conf:.2f}"
        line2 = f"({x1},{y1}) - ({x2},{y2})"

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        # Measure both lines
        (w1, h1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, thickness - 1)

        banner_w = max(w1, w2) + 16
        banner_h = h1 + h2 + 24

        # ── Clamp banner position inside image ─────────────────────────
        x_text = x1
        if x_text + banner_w > w_img:
            x_text = w_img - banner_w - 6

        y_banner = y1 - banner_h - 6
        if y_banner < 0:
            y_banner = y2 + 6          # flip below box if no room above

        # ── Black background banner ────────────────────────────────────
        cv2.rectangle(output,
                      (x_text, y_banner),
                      (x_text + banner_w, y_banner + banner_h),
                      (0, 0, 0), -1)

        # ── Line 1: Class name + confidence (yellow) ───────────────────
        cv2.putText(output, line1,
                    (x_text + 8, y_banner + h1 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 255),       # yellow
                    thickness,
                    cv2.LINE_AA)

        # ── Line 2: Coordinates (white, slightly smaller) ──────────────
        cv2.putText(output, line2,
                    (x_text + 8, y_banner + h1 + h2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.75,
                    (255, 255, 255),     # white
                    thickness - 1,
                    cv2.LINE_AA)

        # ── Center dot ─────────────────────────────────────────────────
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(output, (cx, cy), int(8 * font_scale), (0, 0, 255), -1)

# Show
cv2.imshow("YOLOv8n – Vehicle Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save
cv2.imwrite("yolo_perfect_output.jpg", output)
print("Saved: yolo_perfect_output.jpg")