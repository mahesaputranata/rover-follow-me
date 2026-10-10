# quick test — save as test_door.py and run it
from ultralytics import YOLO
import cv2

model = YOLO("data/models/door_yolov8n.pt")  # from tools/download_door_model.py
print("Model classes:", model.names)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

results = model(frame, conf=0.25, verbose=True)[0]
print("Detections:", len(results.boxes))
for box in results.boxes:
    print(f"  class: {model.names[int(box.cls[0])]}, conf: {float(box.conf[0]):.2f}")