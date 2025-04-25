# Author: Rutvik Joshi
# Date: 04-24-2025
# Description: This script uses YOLOv5 to detect people in a video stream from an Intel RealSense camera. It also uses depth information to determine if the detected person is real or a flat image. The results are logged to Firebase.
#
# Import necessary libraries
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import warnings
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
cred = credentials.Certificate("autonomous-hvac-firebase-adminsdk-fbsvc-402a01a865.json")  
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://autonomous-hvac-default-rtdb.firebaseio.com/'
})
people_ref = db.reference('/yolo-depth')
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5s from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Setup RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
FIREBASE_URL = "https://autonomous-hvac-default-rtdb.firebaseio.com/"
private_key ="aGW7AtRO9pa1WCi4bx2MYuWyuliJ6G6l86SZuO6M9xU"
def get_timestamp():
    return datetime.now().isoformat()

def log_to_firebase(count):
    timestamp = get_timestamp()
    data = {
        "device" : "webcam",
        "count": count,
        "timestamp": timestamp
    }
    print("Logging to Firebase:", data)

    try:
        response = requests.post(FIREBASE_URL, json=data)
        print("Log response:", response.text)
        response.close()
    except Exception as e:
        print("Logging failed:", e)

# Timer initialization
last_logged = time.time()
try:
    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_scale = depth_frame.get_units()  # Convert raw depth to meters

        # Run YOLO on the color frame
        results = model(color_image)
        detections = results.xyxy[0].cpu().numpy()  # Format: [x1, y1, x2, y2, conf, cls]

        real_person_count = 0  # Initialize count for this frame
        fake_person_count = 0  # Initialize count for this frame
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) != 0 or conf < 0.5:  # Class 0 = person
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(color_image.shape[1] - 1, x2), min(color_image.shape[0] - 1, y2)

            # Extract depth in the bounding box
            bbox_depth = depth_image[y1:y2, x1:x2].astype(np.float32) * depth_scale
            valid_depth = bbox_depth[bbox_depth > 0]

            if valid_depth.size < 500:
                label = "Depth Noise"
            else:
                depth_mean = np.mean(valid_depth)
                depth_std = np.std(valid_depth)
                depth_range = np.max(valid_depth) - np.min(valid_depth)

                # Decision rule based on depth variance and range
                if depth_range > 0.4 and depth_std > 0.1:
                    label = "Real Person"
                    real_person_count += 1

                else:
                    label = "Flat Image"
                    fake_person_count += 1

            # Debug print
            # print(f"Box ({x1},{y1},{x2},{y2}) | Mean: {depth_mean:.2f} | Std: {depth_std:.3f} | Range: {depth_range:.3f} → {label}")

            # Annotate
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        current_time = time.time()
        if current_time - last_logged >= 1:
            try:
                iso_timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
                people_ref.push({
                    'timestamp': iso_timestamp,
                    'People count': real_person_count,
                    'Fake count': fake_person_count
                })
                print(f"Data sent to Firebase: {iso_timestamp} - Real {real_person_count} people | Fake: {fake_person_count}  people")
            except Exception as e:
                print(f"Firebase error: {e}")
            # log_to_firebase(real_person_count)
            last_logged = current_time

        # Show result
        cv2.imshow("RealSense + YOLO", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
