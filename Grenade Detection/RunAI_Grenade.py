from ultralytics import YOLO
import cv2
import os

# === Load YOLOv8 model ===
model_path = r"C:\Users\ADMIN\Documents\Grenade.v1i.yolov8\AI\rust_detection_yolov8\weights\best.pt"
model = YOLO(model_path)

# === Open webcam ===
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# === Set detection confidence threshold ===
CONFIDENCE_THRESHOLD = 0.85

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run detection on the frame
    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if conf >= CONFIDENCE_THRESHOLD:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Label with confidence
                text = f'{label}: {conf:.2f}'
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Grenade Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
