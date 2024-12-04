from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

from pathlib import Path  
script_dir = Path(__file__).resolve().parent
video_path = script_dir / 'video3.mp4'
# Initialize YOLO model
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Starting Btrolley detection. Press 'q' to quit.")

# Initialization variables
subject_id = None
tracked_objects = {}
next_object_id = 1
init_start_time = None
frame_skip = 4  # Process every 4th frame
frame_count = 0

def calculate_cost_matrix(tracked_objects, detections):
    num_tracks = len(tracked_objects)
    num_detections = len(detections)

    cost_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)

    for i, track_data in enumerate(tracked_objects.values()):
        for j, detection in enumerate(detections):
            bbox = detection[:4]
            iou = calculate_iou(track_data['bbox'], bbox)
            cost_matrix[i, j] = 1 - iou  # Cost is inverse of IoU (lower is better)

    return cost_matrix

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames based on frame_skip value
    if frame_count % frame_skip != 0:
        continue

    h, w, _ = frame.shape
    frame_center = (w // 2, h // 2)

    # Run YOLO inference
    results = model(frame)

    # Collect detections
    detections = []
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box
        class_id = int(detection.cls[0])  # Class ID
        confidence = detection.conf[0]  # Confidence score

        if confidence > 0.5 and class_id == 0:
            detections.append((x1, y1, x2, y2, confidence))

    if subject_id is None:
        if init_start_time is None:
            init_start_time = time.time()
        
        # Wait for 5 seconds before choosing the subject
        if time.time() - init_start_time >= 5:
            closest_person = None
            min_distance = float('inf')

            for detection in detections:
                x1, y1, x2, y2, _ = detection
                person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                distance = np.linalg.norm(np.array(person_center) - np.array(frame_center))
                if distance < min_distance:
                    min_distance = distance
                    closest_person = detection

            if closest_person:
                subject_id = next_object_id
                tracked_objects[subject_id] = {'bbox': closest_person[:4], 'age': 0}
                next_object_id += 1
                print("[INFO] Subject selected.")
    else:
        if len(detections) > 0:
            cost_matrix = calculate_cost_matrix(tracked_objects, detections)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = list(zip(row_ind, col_ind))

            for track_idx, detection_idx in matches:
                track_id = list(tracked_objects.keys())[track_idx]
                bbox = detections[detection_idx][:4]
                tracked_objects[track_id]['bbox'] = bbox
                tracked_objects[track_id]['age'] = 0

            unmatched_tracks = set(tracked_objects.keys()) - {list(tracked_objects.keys())[i] for i, _ in matches}
            unmatched_detections = set(range(len(detections))) - {d for _, d in matches}

            for detection_idx in unmatched_detections:
                bbox = detections[detection_idx][:4]
                tracked_objects[next_object_id] = {'bbox': bbox, 'age': 0}
                next_object_id += 1

            for track_id in unmatched_tracks:
                tracked_objects[track_id]['age'] += 1

            old_tracks = [track_id for track_id, track_data in tracked_objects.items() if track_data['age'] > 30]
            for track_id in old_tracks:
                del tracked_objects[track_id]
        else:
            tracked_objects[subject_id]['age'] += 1

        if subject_id not in tracked_objects or tracked_objects[subject_id]['age'] > 30:
            print("[INFO] Subject lost.")
            subject_id = None
            init_start_time = time.time()

    if subject_id in tracked_objects:
        bbox = tracked_objects[subject_id]['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, "Subject", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for track_id, track_data in tracked_objects.items():
        if track_id != subject_id:
            bbox = track_data['bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(frame, f"ID {track_id}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Btrolley Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
