import cv2
from ultralytics import YOLO

# Pre-trained YOLOv8 Model Load करणे
model = YOLO('yolov8n.pt')

def detect_vehicles_from_frame(frame):
    results = model(frame)
    vehicle_count = 0
    
    # Vehicle Classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    vehicle_classes = [2, 3, 5, 7]
    
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) in vehicle_classes:
                vehicle_count += 1
                
    return vehicle_count