from ultralytics import YOLO
import easyocr
import cv2

# Load models
plate_model = YOLO(r"C:\ML\runs\detect\number_plate_model3\weights\best.pt")
reader = easyocr.Reader(['en'])

# Load image
image_path = "test_img.jpg"   # <-- replace with your image
frame = cv2.imread(image_path)

output = frame.copy()

# Detect number plate
results = plate_model.predict(frame, conf=0.4)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Crop plate
        plate_crop = frame[y1:y2, x1:x2]

        # OCR
        text = ""
        for (_, t, conf) in reader.readtext(plate_crop):
            if conf > 0.3:
                text += t + " "

        text = text.strip().upper()

        if text:
            label = f"Plate: {text}"

            # BIG TEXT SETTINGS
            font_scale = 1.5
            thickness = 3

            (w, h), _ = cv2.getTextSize(label,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale,
                                        thickness)

            # Adjust position
            y_text = y1 - 10 if y1 - 10 > 20 else y1 + h + 20

            # Background box
            cv2.rectangle(output,
                          (x1, y_text - h - 10),
                          (x1 + w, y_text),
                          (0, 0, 0), -1)

            # Put text
            cv2.putText(output,
                        label,
                        (x1, y_text - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 255),
                        thickness,
                        cv2.LINE_AA)

# Show result
cv2.imshow("OCR Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save for PPT
cv2.imwrite("ocr_output.jpg", output)

print("Saved: ocr_output.jpg")