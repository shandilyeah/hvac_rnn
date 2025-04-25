from ultralytics import YOLO
import cv2
import math
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase
cred = credentials.Certificate("autonomous-hvac-firebase-adminsdk-fbsvc-ff4e9e2340.json")  
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://autonomous-hvac-default-rtdb.firebaseio.com/'
})
people_ref = db.reference('/people_counts')  

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Initialize variables for frame processing
last_processed_time = time.time()
people_count = 0
last_boxes = []
timestamp_str = ""
detection_details = []  # To store individual detection probabilities

while True:
    success, img = cap.read()
    current_time = time.time()
    
    # Process 1 frame per second
    if current_time - last_processed_time >= 1:
        # Perform object detection
        results = model(img, stream=True)
        
        # Reset counters and boxes
        people_count = 0
        last_boxes = []
        detection_details = []
        
        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])  # Get detection confidence
                
                if classNames[cls] == "person":
                    # Store bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    last_boxes.append((x1, y1, x2, y2))
                    people_count += 1
                    
                    # Store detection details
                    detection_details.append({
                        'confidence': confidence,
                        'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    })
        
        # Update timestamp and processing time
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_processed_time = current_time
        
        # Send data to Firebase with probabilities
        try:
            people_ref.push({
                'timestamp': timestamp_str,
                'count': people_count,
                'detections': detection_details,
                'average_confidence': sum(d['confidence'] for d in detection_details)/len(detection_details) if detection_details else 0
            })
            print(f"Data sent to Firebase: {timestamp_str} - {people_count} people")
            print(f"Detection details: {detection_details}")
        except Exception as e:
            print(f"Firebase error: {e}")

    # Draw bounding boxes and text on current frame
    for (x1, y1, x2, y2) in last_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
    # Display confidence info on frame
    confidence_text = f"Avg Conf: {sum(d['confidence'] for d in detection_details)/len(detection_details):.2f}" if detection_details else "No detections"
    cv2.putText(img, f"People: {people_count} - {timestamp_str}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, confidence_text,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Webcam', img)
    
    # Exit on 'q' press
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()